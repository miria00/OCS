"""Walk a CVXPY LinOp tree and emit a JAX forward map (no matrix is formed).

CVXPY canonicalizes every affine expression to a symbolic LinOp tree
(``cvxpy/lin_ops/lin_op.py``) BEFORE flattening it into a sparse coefficient
matrix (``canonInterface.get_problem_matrix``).  This module intercepts at the
tree: it walks the LinOp DAG and composes a pure-JAX ``forward(x)`` that
evaluates the affine map ``A x + b`` directly, so

    matvec(x)  = forward(x) - forward(0)          (the linear part, A x)
    rmatvec(y) = vjp of forward                    (A^T y, exact for affine)

and the coefficient matrix ``A`` is never formed, stored, or indexed.  This is
the operator-native canonicalization seam: structure that CVXPY's backends
destroy (a ``conv`` node becomes a Toeplitz matrix, a ``kron`` node an explicit
Kronecker product) is instead evaluated by its fast action.

Conventions (matching CVXPY): variables are flattened and stacked in
column-major (Fortran) order; ``id_to_col`` maps variable id to its offset in
the stacked vector, exactly as in ``InverseData.var_offsets``.

Parameters are NOT supported here: run the ``EvalParams`` reduction first so
every parameter is a concrete constant by the time the tree is walked.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


def _flat_f(z):
    """Flatten in Fortran (column-major) order, CVXPY's convention."""
    return jnp.reshape(z, (-1,), order="F")


class LinOpWalker:
    """Evaluates LinOp trees against a stacked variable vector ``x``.

    ``penv`` maps parameter id -> value; a ``param`` node looks its value up
    there, so the same tree evaluates for changing parameter data (the
    parametric, same-operator regime).  With ``penv=None`` a param node raises.
    """

    def __init__(self, id_to_col: dict, x_length: int, penv: dict | None = None):
        self.id_to_col = dict(id_to_col)
        self.x_length = int(x_length)
        self.penv = penv

    # ---- constants -------------------------------------------------------
    def const_value(self, lin_op, keep_sparse=False):
        """Evaluate a constant-only LinOp subtree to a concrete array.

        With ``keep_sparse`` a sparse CVXPY constant retains its coordinate
        arrays, so the matmul paths apply it by gather and scatter-add
        and its entries are never expanded.  Every other consumer (an
        additive constant, an elementwise factor, a convolution kernel, a
        Kronecker factor) needs the entries themselves and receives a dense
        array, as before.
        """
        t = lin_op.type
        if t == "scalar_const":
            return jnp.asarray(lin_op.data)
        if t == "dense_const":
            return jnp.asarray(lin_op.data)
        if t == "sparse_const":
            if keep_sparse:
                return _sparse_constant(lin_op.data)
            return jnp.asarray(lin_op.data.toarray())
        if t == "param":
            if self.penv is None:
                raise NotImplementedError(
                    "parameter reached the operator walker without a parameter "
                    "environment; run EvalParams or pass penv")
            return jnp.reshape(jnp.asarray(self.penv[lin_op.data]),
                               lin_op.shape, order="F") if lin_op.shape \
                else jnp.asarray(self.penv[lin_op.data])
        # a constant subtree built from other nodes (e.g. -c, c.T): evaluate it
        # through the generic path with a zero variable vector
        zeros = jnp.zeros(self.x_length)
        return self.eval(lin_op, zeros)

    # ---- the recursive evaluator ----------------------------------------
    def eval(self, op, x):
        t = op.type

        if t == "variable":
            col = self.id_to_col[op.data]
            size = int(np.prod(op.shape)) if op.shape else 1
            seg = x[col:col + size]
            return jnp.reshape(seg, op.shape, order="F") if op.shape else seg[0]

        if t in ("scalar_const", "dense_const", "sparse_const", "param"):
            return self.const_value(op)

        if t == "no_op":
            return jnp.zeros(op.shape)

        if t == "neg":
            return -self.eval(op.args[0], x)

        if t == "sum":
            out = self.eval(op.args[0], x)
            for a in op.args[1:]:
                out = out + self.eval(a, x)
            return out

        if t == "promote":
            return jnp.broadcast_to(self.eval(op.args[0], x), op.shape)

        if t == "broadcast_to":
            return jnp.broadcast_to(self.eval(op.args[0], x), op.shape)

        if t == "mul":                      # constant @ arg
            C = self.const_value(op.data, keep_sparse=True)
            v = self.eval(op.args[0], x)
            return self._matmul(C, v, op.shape)

        if t == "rmul":                     # arg @ constant
            C = self.const_value(op.data, keep_sparse=True)
            v = self.eval(op.args[0], x)
            return self._matmul(v, C, op.shape)

        if t == "mul_elem":
            C = self.const_value(op.data)
            return jnp.multiply(C, self.eval(op.args[0], x))

        if t == "div":
            C = self.const_value(op.data)
            return jnp.divide(self.eval(op.args[0], x), C)

        if t == "index":
            keys = tuple(op.data)
            out = self.eval(op.args[0], x)[keys]
            # CVXPY drops the axes an integer index removes (its slices keep
            # them as singletons); the LinOp carries the resulting shape.
            if out.shape != tuple(op.shape):
                out = jnp.reshape(out, op.shape, order="F")
            return out

        if t == "transpose":
            axes = op.data[0] if op.data else None
            return jnp.transpose(self.eval(op.args[0], x), axes=axes)

        if t == "reshape":
            return jnp.reshape(self.eval(op.args[0], x), op.shape, order="F")

        if t == "sum_entries":
            axis, keepdims = op.data
            return jnp.sum(self.eval(op.args[0], x), axis=axis,
                           keepdims=bool(keepdims))

        if t == "trace":
            return jnp.trace(self.eval(op.args[0], x))

        if t == "diag_vec":                 # vector -> diagonal matrix (offset k)
            k = op.data if op.data is not None else 0
            return jnp.diag(self.eval(op.args[0], x), k=int(k))

        if t == "diag_mat":                 # matrix -> its k-th diagonal
            k = op.data if op.data is not None else 0
            return jnp.diagonal(self.eval(op.args[0], x), offset=int(k))

        if t == "upper_tri":                # strict upper triangle, as a vector
            v = self.eval(op.args[0], x)
            n, m = v.shape
            idx = np.triu_indices(n, k=1, m=m)
            return v[idx]

        if t == "hstack":
            # CVXPY's backends define hstack as the concatenation of the
            # arguments' column-major vectors, read back in the LinOp's shape
            # (so a (1, 1) block and a (1,) block stack, as they do there).
            out = jnp.concatenate([_flat_f(self.eval(a, x)) for a in op.args])
            return jnp.reshape(out, op.shape, order="F")

        if t == "vstack":
            # numpy semantics, as in CVXPY's backends: 1-D arguments are rows.
            return jnp.vstack([self.eval(a, x) for a in op.args])

        if t == "concatenate":
            axis = op.data[0] if isinstance(op.data, (list, tuple)) else op.data
            parts = [self.eval(a, x) for a in op.args]
            if axis is None:                # numpy: flatten each in C order
                return jnp.concatenate([jnp.ravel(v) for v in parts])
            return jnp.concatenate(parts, axis=axis)

        if t == "conv":                     # 1-D convolution, kernel in .data
            kern = jnp.ravel(self.const_value(op.data))
            v = jnp.ravel(self.eval(op.args[0], x))
            k_len, v_len = kern.shape[0], v.shape[0]
            L = k_len + v_len - 1
            if k_len * v_len <= (1 << 18):  # small: direct convolution
                out = jnp.convolve(kern, v, mode="full")
            else:                            # long kernel: FFT convolution,
                Kf = jnp.fft.rfft(kern, n=L)  # O(L log L) instead of O(nk);
                Vf = jnp.fft.rfft(v, n=L)     # this is where the Toeplitz
                out = jnp.fft.irfft(Kf * Vf, n=L)  # matrix would be enormous
            return jnp.reshape(out, op.shape, order="F")

        if t == "kron_l":                   # variable is the LEFT kron factor
            C = self.const_value(op.data)
            v = self.eval(op.args[0], x)
            return jnp.kron(v, C)

        if t == "kron_r":                   # variable is the RIGHT kron factor
            C = self.const_value(op.data)
            v = self.eval(op.args[0], x)
            return jnp.kron(C, v)

        raise NotImplementedError(f"LinOp type '{t}' not yet supported "
                                  "by the operator walker")

    @staticmethod
    def _matmul(L, R, out_shape):
        """CVXPY mul semantics: matrix product with scalar fallback.  A sparse
        constant on either side is applied by its sparse action; the
        product with the dense operand is dense."""
        sparse_L, sparse_R = _is_sparse(L), _is_sparse(R)
        if not sparse_L:
            L = jnp.asarray(L)
        if not sparse_R:
            R = jnp.asarray(R)
        if L.ndim == 0 or R.ndim == 0:
            out = (L.todense() if sparse_L else L) * (R.todense() if sparse_R else R)
        elif sparse_L:
            # Constant @ constant is normally folded by CVXPY. Its dense
            # result is retained here for the same fallback as before.
            out = L.matmul(R.todense() if sparse_R else R)
        elif sparse_R:
            out = R.rmatmul(L)
        else:
            out = L @ R
        return jnp.reshape(out, out_shape, order="F") if out.shape != tuple(out_shape) \
            else out


def _is_sparse(value):
    return isinstance(value, _SparseConstant)


@dataclass(frozen=True, eq=False)
class _SparseConstant:
    """Internal two-dimensional COO constant using ordinary JAX primitives.

    Matrix products retain only nonzeros, gathered dense values, and the dense
    output. Repeated/unsorted coordinates are summed by scatter-add. JAX derives
    the linear transpose from these same primitives; no sparse extension or
    assembled problem matrix is required.
    """

    data: object
    rows: object
    cols: object
    shape: tuple[int, int]
    ndim = 2

    def _apply(self, operand, source, target, axis, size):
        gathered = jnp.moveaxis(jnp.take(operand, source, axis=axis), axis, 0)
        weights = self.data.reshape((-1,) + (1,) * (gathered.ndim - 1))
        values = weights * gathered
        out = jnp.zeros((size,) + gathered.shape[1:], dtype=values.dtype)
        return jnp.moveaxis(out.at[target].add(values), 0, axis)

    def matmul(self, operand):
        operand = jnp.asarray(operand)
        axis = 0 if operand.ndim == 1 else -2
        if operand.ndim == 0 or operand.shape[axis] != self.shape[1]:
            raise ValueError("sparse constant left product has incompatible shape")
        return self._apply(operand, self.cols, self.rows, axis, self.shape[0])

    def rmatmul(self, operand):
        operand = jnp.asarray(operand)
        if operand.ndim == 0 or operand.shape[-1] != self.shape[0]:
            raise ValueError("sparse constant right product has incompatible shape")
        return self._apply(operand, self.rows, self.cols, -1, self.shape[1])

    def todense(self):
        # Scalar multiplication asks for the entries themselves; matmul with
        # a dense variable never calls this path.
        return jnp.zeros(self.shape, dtype=self.data.dtype).at[
            self.rows, self.cols].add(self.data)


def _sparse_constant(matrix):
    """Copy only the nonzeros of a CVXPY constant, in solver precision."""
    coo = matrix.tocoo()
    if len(coo.shape) != 2 or any(d > np.iinfo(np.int32).max for d in coo.shape):
        raise ValueError("sparse constant requires a two-dimensional int32 shape")
    return _SparseConstant(jnp.asarray(coo.data, jnp.float32),
                           jnp.asarray(coo.row, jnp.int32),
                           jnp.asarray(coo.col, jnp.int32),
                           tuple(int(d) for d in coo.shape))


def make_affine_forward(lin_ops, id_to_col, x_length):
    """Compose a list of LinOp trees into one stacked affine forward map.

    Returns ``forward(x) -> A x + b`` where the output stacks each tree's
    result flattened in Fortran order (CVXPY's constraint-stacking order).
    """
    walker = LinOpWalker(id_to_col, x_length)

    def forward(x):
        outs = [_flat_f(walker.eval(op, x)) for op in lin_ops]
        return jnp.concatenate(outs) if len(outs) > 1 else outs[0]

    return forward


def make_affine_operator(lin_ops, id_to_col, x_length):
    """Matrix-free (matvec, rmatvec, offset) for stacked LinOp trees.

    ``matvec(x) = A x``, ``rmatvec(y) = A^T y``, ``offset = b``; the matrix A
    is never formed.  The adjoint comes from one reverse-mode VJP of the
    forward map, which is exact because the map is affine.
    """
    forward = make_affine_forward(lin_ops, id_to_col, x_length)
    x0 = jnp.zeros(x_length)
    offset, matvec = linearize_affine(forward, x0)
    _, vjp = jax.vjp(forward, x0)

    def rmatvec(y):
        return vjp(y)[0]

    return matvec, rmatvec, offset


def linearize_affine(forward, x0):
    """``(forward(x0), x -> A x)`` for an affine ``forward``.

    The linear part is the forward-mode linearization at ``x0`` (exact for an
    affine map), not ``forward(x) - forward(x0)``: the tangent of a constant
    is a symbolic zero, so the offset is never added and then subtracted,
    which in float32 would drop the low bits of ``A x`` whenever the offset
    is large next to it (a constraint ``x + y == target + y`` with
    ``y ~ 1e5`` lost four digits of ``x`` that way).
    """
    offset, lin = jax.linearize(forward, x0)
    return offset, lin


# ---- parametric (same-operator) support ---------------------------------

def make_param_forward(lin_ops, id_to_col, x_length):
    """``forward(x, penv) -> A x + b(penv)`` for trees that may hold parameters.

    ``penv`` maps parameter id -> value.  When no parameter multiplies the
    variable (checked by ``check_param_operator``), ``A`` does not depend on
    ``penv`` and only the offset ``b`` (and the objective's cost) varies:
    CVXPY Parameters become exactly OCS's per-instance offsets, the
    compile-once / solve-many regime.
    """
    def forward(x, penv):
        walker = LinOpWalker(id_to_col, x_length, penv)
        outs = [_flat_f(walker.eval(op, x)) for op in lin_ops]
        return jnp.concatenate(outs) if len(outs) > 1 else outs[0]

    return forward


_INTERACTING = {"mul", "rmul", "mul_elem", "div", "conv", "kron_l", "kron_r"}


def _is_linop(obj):
    return hasattr(obj, "type") and hasattr(obj, "args") and hasattr(obj, "shape")


def check_param_operator(op):
    """Scan a LinOp tree: returns (has_var, has_param, bad).

    ``bad`` is True when a parameter and the variable meet at a multiplicative
    node -- then the linear operator itself depends on the parameter
    (a parameterized-K family), which the same-operator bridge must reject.
    """
    hv = op.type == "variable"
    hp = op.type == "param"
    bad = False
    dv = dp = False
    if _is_linop(op.data):
        dv, dp, dbad = check_param_operator(op.data)
        bad |= dbad
    av = ap = False
    for a in op.args:
        v_, p_, b_ = check_param_operator(a)
        av |= v_; ap |= p_; bad |= b_
    if op.type in _INTERACTING:
        bad |= (dp and av) or (dv and ap)
    return hv or dv or av, hp or dp or ap, bad
