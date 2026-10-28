"""Cone metrics drawn from the automorphism group: the closed-form weighted projection.

For a closed convex cone ``C`` and ``Sigma = W^T W`` with ``W`` a linear
automorphism of ``C`` (``W C = C``), the ``Sigma``-weighted projection has the
closed form

    prox_{I_C}^{Sigma}(v) = W^{-1} Pi_C(W v),

because ``z in C`` iff ``W z in C``: the weighted problem is a Euclidean
projection in the coordinates ``u = W z``. For the symmetric (self-scaled)
cones the positive definite automorphisms are exactly the quadratic
representations ``P(w)`` of the cone's Euclidean Jordan algebra, ``w`` an
interior point, and ``P(w)^{1/2} = P(w^{1/2})``, so ``W = P(w^{1/2})`` and
``W^{-1} = P(w^{-1/2})`` are closed form as well.

The class chain for a symmetric-cone block, stated once and used everywhere:

    scalar (1 dof)  subset  automorphism {P(w)}  subset  arbitrary SPD (no closed form)

with the diagonal class OUTSIDE this chain except for its scalar members: a
positive diagonal ``D`` is an automorphism of the second-order cone only when
``D J D = mu J``, which forces every entry of ``D`` to be equal. The
automorphism class is "full" in the sense that ``P(w)`` mixes coordinates; it
is NOT arbitrary SPD (``k`` degrees of freedom per second-order cone against
``k(k+1)/2``, ``k(k+1)/2`` per ``k x k`` PSD block against ``k^2(k^2+1)/2``),
and no message or record in this module calls it ``full``.

What is implemented:

* second-order cone ``L^k_+ = {(x0, xbar): ||xbar||_2 <= x0}`` (``SecondOrderCone``
  and the CVXPY-layout ``SOCBatch``): ``J = diag(1, -I)``, ``det(w) = w0^2 -
  ||wbar||^2``, ``P(w) z = 2 w (w^T z) - det(w) J z``, applied in ``O(k)``;
  Jordan spectral functions from ``lam_{1,2} = w0 +- ||wbar||``;
* positive semidefinite cone ``S^k_+`` (``PSDCone``, row-major ``k*k`` block):
  ``P(S) Z = S Z S``, applied as two ``k x k`` congruences.

The exponential and power cones are not symmetric cones, have no self-scaled
barrier and no such family of positive definite scalings; a declaration on
them is refused with that reason (``ocs.solver.metric_construct``).

Numerical domain
----------------
The weighted projection returns a float32 point. Its float64 KKT defect grows
with the spread of ``P(w)``'s eigenvalues (``cond P(w) = (lam1/lam2)^2`` for a
second-order cone, ``cond(S)^2`` for PSD) exactly as the rotated norm prox in
``block_metric`` does, so an element is admitted only inside
``MAX_AUTOMORPHISM_COND``. That constant is PROVISIONAL: it reuses
``block_metric.MAX_FULL_COND`` (measured for the rotated norm prox on KEVIN)
until the automorphism ladder has been run and recorded. An element outside it
is REJECTED at declaration, never clipped; ``restrict_cone_condition`` is the
explicit way out, and a caller who uses it must recompute the step
normalization.
"""
from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from . import block_metric as _bm
from .cones import sym_eigh as _sym_eigh

KIND = "automorphism"
ALGEBRAS = ("soc", "psd")

# PROVISIONAL: inherited from the rotated norm prox measurement, see the module
# docstring. Read at call time everywhere (never bound as a default argument)
# so that a test or a later measurement can lower it.
MAX_AUTOMORPHISM_COND = _bm.MAX_FULL_COND
AUTOMORPHISM_DOMAIN_STATUS = ("provisional: reuses block_metric.MAX_FULL_COND measured for the "
                              "rotated norm prox; the automorphism prox ladder is recorded in the "
                              "candidate receipt, not yet in this constant")

_ELEMENTS_IN_META = 8
_F32 = np.finfo(np.float32)


def _mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


# =============================================================== structure
def symmetric_cone_structure(block, size):
    """``("soc", k, cones)``, ``("psd", k, 1)``, or None.

    Exact types only (a subclass or an instance-level ``prox`` override may
    change the function), mirroring ``block_metric.euclidean_norm_structure``.
    ``k`` is the Jordan-algebra dimension of ONE cone (``1 + d`` for a
    second-order cone, the matrix order for PSD) and ``cones`` how many cones
    the term stacks.
    """
    from . import cones
    if "prox" in vars(block):
        return None
    t = type(block)
    size = int(size)
    if t is cones.SecondOrderCone:
        return ("soc", size, 1) if size >= 2 else None
    if t is cones.SOCBatch:
        k = 1 + int(block.d)
        return ("soc", k, int(block.q)) if size == k * int(block.q) and k >= 2 else None
    if t is cones.PSDCone:
        k = int(block.k)
        return ("psd", k, 1) if size == k * k and k >= 1 else None
    return None


def cone_layout(block):
    """How the term's rows are laid out: one tuple the gather/scatter reads."""
    from . import cones
    t = type(block)
    if t is cones.SOCBatch:
        return ("socbatch", int(block.q), int(block.d))
    if t is cones.PSDCone:
        return ("psd", int(block.k))
    if t is cones.SecondOrderCone:
        return ("soc",)
    raise ValueError(f"{t.__name__} is not a symmetric-cone block")


def degrees_of_freedom(algebra, k):
    return int(k) if algebra == "soc" else int(k * (k + 1) // 2)


def jordan_rank(algebra, k):
    return 2 if algebra == "soc" else int(k)


# ============================================================ batched layout
# ``rows`` maps the term's flat vector to one row per cone, (cones, k) for the
# second-order cone and (1, k, k) for PSD; ``unrows`` is its exact inverse.
# The SOCBatch form matches ``ocs.solver.domain_checks._rows`` (CVXPY layout
# u = [t (q); vec_F(Z) (d*q)], cone j = (t_j, Z[:, j])).
def rows(layout, v, xp=jnp):
    kind = layout[0]
    if kind == "socbatch":
        q, d = layout[1], layout[2]
        z = xp.reshape(v[q:], (d, q), order="F").T
        return xp.concatenate([v[:q, None], z], axis=1)
    if kind == "psd":
        k = layout[1]
        return xp.reshape(v, (1, k, k))
    return v[None, :]


def unrows(layout, R, xp=jnp):
    kind = layout[0]
    if kind == "socbatch":
        return xp.concatenate([R[:, 0], xp.reshape(R[:, 1:].T, (-1,), order="F")])
    if kind == "psd":
        return xp.reshape(R, (-1,))
    return R[0]


# ====================================================== Jordan algebra, host
def _soc_eig64(w):
    wbar = w[:, 1:]
    nb = np.sqrt(np.sum(wbar * wbar, axis=1))
    return w[:, 0] + nb, w[:, 0] - nb, nb


def soc_spectral64(w, fn):
    """f(w) = f(lam1) c1 + f(lam2) c2 on rows (count, k), float64."""
    w = np.asarray(w, np.float64)
    lam1, lam2, nb = _soc_eig64(w)
    direction = w[:, 1:] / np.maximum(nb, np.finfo(np.float64).tiny)[:, None]
    f1, f2 = fn(lam1), fn(lam2)
    return np.concatenate([(0.5 * (f1 + f2))[:, None], (0.5 * (f1 - f2))[:, None] * direction], axis=1)


def soc_det64(w):
    lam1, lam2, _ = _soc_eig64(np.asarray(w, np.float64))
    return lam1 * lam2


def apply_soc64(a, a_det, R):
    """P(a) applied to rows R: 2 a (a . R) - det(a) J R, float64."""
    a = np.asarray(a, np.float64)
    R = np.asarray(R, np.float64)
    dot = np.sum(a * R, axis=-1, keepdims=True)
    JR = np.concatenate([R[..., :1], -R[..., 1:]], axis=-1)
    return 2.0 * a * dot - np.asarray(a_det, np.float64)[..., None] * JR


def psd_spectral64(S, fn):
    values, vectors = np.linalg.eigh(0.5 * (S + np.swapaxes(S, -1, -2)))
    out = np.matmul(vectors * fn(values)[..., None, :], np.swapaxes(vectors, -1, -2))
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def apply_psd64(A, R):
    return np.matmul(np.matmul(A, R), A)


def matrix_of(algebra, element):
    """Host float64 P(w) (k x k) or P(S) = S kron S (k^2 x k^2), for references."""
    element = np.asarray(element, np.float64)
    if algebra == "soc":
        k = element.size
        J = -np.eye(k)
        J[0, 0] = 1.0
        return 2.0 * np.outer(element, element) - float(soc_det64(element[None])[0]) * J
    return np.kron(element, element)


# ==================================================== Jordan algebra, device
def soc_eig(w):
    """(lam1, lam2, dir) of rows ``w`` (..., k): the Jordan spectral frame."""
    wbar = w[..., 1:]
    nb = jnp.sqrt(jnp.sum(wbar * wbar, axis=-1))
    tiny = jnp.asarray(jnp.finfo(w.dtype).tiny, w.dtype)
    direction = wbar / jnp.maximum(nb, tiny)[..., None]
    return w[..., 0] + nb, w[..., 0] - nb, direction


def soc_spectral(w, fn):
    """Spectral function f(w) = f(lam1) c1 + f(lam2) c2 on rows (..., k).

    When ``lam1 == lam2`` the second component is multiplied by zero, so the
    ``max(nb, tiny)`` guard in the frame is exact, not a tolerance.
    """
    lam1, lam2, direction = soc_eig(w)
    f1, f2 = fn(lam1), fn(lam2)
    return jnp.concatenate([(0.5 * (f1 + f2))[..., None],
                            (0.5 * (f1 - f2))[..., None] * direction], axis=-1)


def soc_det(w):
    lam1, lam2, _ = soc_eig(w)
    return lam1 * lam2


def apply_soc(a, a_det, R):
    """P(a) R^T per row: ``2 a (a . R) - det(a) J R``. ``a`` broadcasts over rows."""
    dot = jnp.sum(a * R, axis=-1, keepdims=True)
    JR = jnp.concatenate([R[..., :1], -R[..., 1:]], axis=-1)
    return 2.0 * a * dot - a_det[..., None] * JR


def apply_psd(A, R):
    """Congruence A R A per (..., k, k) row, at HIGHEST precision."""
    return _mm(_mm(A, R), A)


def psd_spectral(S, fn):
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2))
    values, vectors = _sym_eigh(S)
    out = _mm(vectors * fn(values)[..., None, :], jnp.swapaxes(vectors, -1, -2))
    return 0.5 * (out + jnp.swapaxes(out, -1, -2))


# ================================================== the Euclidean projections
# Copies of the shipped projection bodies (``cones.py``) written over rows, so
# that the cone classes stay byte-identical in this change. A test pins each
# copy against the class method it mirrors.
def soc_project(R):
    """Euclidean projection onto L^k_+ of every row of R (count, k)."""
    s, z = R[:, 0], R[:, 1:]
    nz = jnp.linalg.norm(z, axis=1)
    inside = nz <= s
    below = nz <= -s
    scale = (nz + s) / (2.0 * nz + 1e-30)
    col = jnp.where(inside, 1.0, jnp.where(below, 0.0, scale))
    proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
    proj_z = z * col[:, None]
    return jnp.concatenate([proj_s[:, None], proj_z], axis=1)


def psd_project(M):
    """Euclidean projection onto S^k_+ of every (k, k) slice of M (count, k, k)."""
    M = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    _, U = _sym_eigh(M)
    U = U / jnp.sqrt(jnp.sum(U * U, axis=-2, keepdims=True))
    MU = _mm(M, U)
    w = jnp.sum(U * MU, axis=-2)
    return _mm(U * jnp.maximum(w, 0.0)[..., None, :], jnp.swapaxes(U, -1, -2))


# ======================================================== the domain message
def _domain_message(cond, limit, algebra, what="cone automorphism metric"):
    return (f"{what}: an automorphism element must have condition number at most {limit:g} "
            f"(cond P(w) = (lam1/lam2)^2 for a second-order cone, cond(S)^2 for a PSD cone); "
            f"this one has {cond:.6g}. Beyond this bound the float32 weighted projection has not "
            f"been validated against the declared {_bm.PROX_KKT_BAR:g} proximal optimality bar "
            "(the bound is provisional and inherited from the rotated norm prox). Either restrict "
            "the element explicitly with ocs.prox.cone_metric.restrict_cone_condition(algebra, "
            "elements, max_cond) and RECOMPUTE the step normalization against the restricted "
            "element, or declare 'scalar'. An element you supply is never clipped for you.")


# =============================================================== the carrier
class ConeAutomorphism:
    """One Jordan element per cone (or one shared) for a symmetric-cone term.

    ``algebra`` is ``"soc"`` or ``"psd"``. For ``"soc"`` the elements are
    ``w`` of shape ``(count, k)`` (or ``(k,)``), each interior (``w0 >
    ||wbar||``); for ``"psd"`` they are ``S`` of shape ``(count, k, k)`` (or
    ``(k, k)``), each exactly symmetric and positive definite. ``count == 1``
    on a term with several cones means ONE element shared by every cone
    (sharing ``"term"``). The effective dual metric of cone ``j`` is ``c
    P(w_j)`` with ``c`` the term's ONE dual step, and the dual update is the
    weighted Moreau identity in closed form, ``y <- v - W Pi_C(W^{-1} v)``,
    ``W = P(w_j^{1/2})``, in which ``c`` cancels.

    Validated on the host in float64 once; stored on the device in float32
    with ``w^{1/2}``, ``w^{-1/2}`` and their determinants (``S^{1/2}``,
    ``S^{-1/2}`` for PSD) cached, so the compiled loop never forms ``w0^2 -
    ||wbar||^2`` in float32. An element outside ``MAX_AUTOMORPHISM_COND`` is
    rejected here, never clipped.
    """

    __slots__ = ("algebra", "k", "count", "sharing", "layout", "origin", "cond", "cond_per_block",
                 "w_host", "S_host", "w", "w_det", "half", "half_det", "invhalf", "invhalf_det",
                 "S", "S_half", "S_invhalf")

    def __init__(self, algebra, elements, *, layout=None, sharing=None, origin=None):
        if algebra not in ALGEBRAS:
            raise ValueError(f"ConeAutomorphism: algebra must be one of {ALGEBRAS}, got {algebra!r}")
        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.asarray(elements, dtype=np.float64)
        if algebra == "soc":
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or not arr.size or arr.shape[1] < 2:
                raise ValueError("ConeAutomorphism.soc: w must be a (k,) or (count, k) array with k >= 2")
            if not np.all(np.isfinite(arr)):
                raise ValueError("ConeAutomorphism.soc: every element must be finite")
            lam1, lam2, _nb = _soc_eig64(arr)
            if not np.all(lam2 > 0):
                j = int(np.argmin(lam2))
                raise ValueError(f"ConeAutomorphism.soc: every w must be interior (w0 > ||wbar||_2); "
                                 f"element {j} has w0 - ||wbar||_2 = {lam2[j]!r}")
            if float(np.max(lam1)) ** 2 > float(_F32.max) or float(np.min(lam2)) ** 2 < float(_F32.tiny):
                raise ValueError("ConeAutomorphism.soc: P(w) must be representable in float32 "
                                 "(lam1^2 below float32 max, lam2^2 above float32 tiny)")
            arr = np.ascontiguousarray(arr)
            det = lam1 * lam2
            self.w_host, self.S_host = arr, None
            self.cond_per_block = (lam1 / lam2) ** 2
            self.w = jnp.asarray(arr, jnp.float32)
            self.w_det = jnp.asarray(det, jnp.float32)
            self.half = jnp.asarray(soc_spectral64(arr, np.sqrt), jnp.float32)
            self.half_det = jnp.asarray(np.sqrt(det), jnp.float32)
            self.invhalf = jnp.asarray(soc_spectral64(arr, lambda t: 1.0 / np.sqrt(t)), jnp.float32)
            self.invhalf_det = jnp.asarray(1.0 / np.sqrt(det), jnp.float32)
            self.S = self.S_half = self.S_invhalf = None
            self.count, self.k = arr.shape
        else:
            if arr.ndim == 2:
                arr = arr[None]
            if arr.ndim != 3 or not arr.size or arr.shape[1] != arr.shape[2]:
                raise ValueError("ConeAutomorphism.psd: S must be a (k, k) or (count, k, k) array")
            if not np.all(np.isfinite(arr)):
                raise ValueError("ConeAutomorphism.psd: every element must be finite")
            if not np.array_equal(arr, np.swapaxes(arr, -1, -2)):
                raise ValueError("ConeAutomorphism.psd: every S must be exactly symmetric")
            values, vectors = np.linalg.eigh(arr)
            if not np.all(values > 0):
                j = int(np.argmin(values.min(axis=-1)))
                raise ValueError(f"ConeAutomorphism.psd: every S must be positive definite; element "
                                 f"{j} has smallest eigenvalue {values[j].min()!r}")
            if float(values.max()) ** 2 > float(_F32.max) or float(values.min()) ** 2 < float(_F32.tiny):
                raise ValueError("ConeAutomorphism.psd: P(S) must be representable in float32")
            arr = np.ascontiguousarray(arr)
            self.S_host, self.w_host = arr, None
            self.cond_per_block = (values.max(axis=-1) / values.min(axis=-1)) ** 2

            def power(t):
                out = np.matmul(vectors * (values ** t)[..., None, :], np.swapaxes(vectors, -1, -2))
                return 0.5 * (out + np.swapaxes(out, -1, -2))
            self.S = jnp.asarray(arr, jnp.float32)
            self.S_half = jnp.asarray(power(0.5), jnp.float32)
            self.S_invhalf = jnp.asarray(power(-0.5), jnp.float32)
            self.w = self.w_det = self.half = self.half_det = self.invhalf = self.invhalf_det = None
            self.count, self.k = arr.shape[0], arr.shape[1]
        self.algebra = algebra
        self.cond = float(np.max(self.cond_per_block))
        limit = MAX_AUTOMORPHISM_COND
        if not (self.cond <= limit):
            raise ValueError(_domain_message(self.cond, limit, algebra))
        if sharing not in (None, "per_block", "term"):
            raise ValueError("ConeAutomorphism: sharing must be None, 'per_block' or 'term'")
        self.sharing = sharing
        self.layout = None if layout is None else tuple(layout)
        self.origin = None if origin is None else dict(origin)

    # ------------------------------------------------------------ constructors
    @classmethod
    def soc(cls, w, **kw):
        """Second-order cone elements ``w`` of shape ``(k,)`` or ``(count, k)``."""
        return cls("soc", w, **kw)

    @classmethod
    def psd(cls, S, **kw):
        """PSD elements ``S`` of shape ``(k, k)`` or ``(count, k, k)``."""
        return cls("psd", S, **kw)

    @classmethod
    def from_nt(cls, block, s, z):
        """The Nesterov-Todd scaling point of the pair, see ``nt_scaling_point``."""
        return nt_scaling_point(block, s, z)

    @classmethod
    def from_state(cls, algebra, value, layout, sharing, origin=None):
        """From the constructor's decoded host state, layout already resolved."""
        return cls(algebra, value, layout=layout, sharing=sharing, origin=origin)

    def bound(self, layout, sharing):
        """A copy bound to a term's layout and sharing; the caller's object is untouched."""
        other = object.__new__(type(self))
        for name in self.__slots__:
            setattr(other, name, getattr(self, name))
        other.layout = tuple(layout)
        other.sharing = sharing
        return other

    # ---------------------------------------------------------------- queries
    @property
    def kind(self):
        return KIND

    @property
    def elements(self):
        return self.w_host if self.algebra == "soc" else self.S_host

    @property
    def is_identity(self):
        """Exactly the identity metric: every element is e (soc) or I (psd), bitwise."""
        if self.algebra == "soc":
            e = np.zeros(self.k)
            e[0] = 1.0
            return bool(np.all(self.w_host == e))
        return bool(np.all(self.S_host == np.eye(self.k)))

    @property
    def cones(self):
        """How many cones the BOUND term stacks, None on an unbound carrier.

        Read from the layout, which ``bound`` fixes at normalization: an
        ``SOCBatch`` of ``q`` cones stacks ``q``, a ``SecondOrderCone`` and a
        ``PSDCone`` stack one. ``count`` is a different number: how many
        Jordan elements are stored, 1 when one element is shared by the whole
        term and ``cones`` when each cone carries its own.
        """
        if self.layout is None:
            return None
        return int(self.layout[1]) if self.layout[0] == "socbatch" else 1

    @property
    def degrees_of_freedom(self):
        return degrees_of_freedom(self.algebra, self.k)

    @property
    def jordan_rank(self):
        return jordan_rank(self.algebra, self.k)

    def matrix(self, index=0):
        """Host float64 ambient matrix P(w_index) (k x k) or P(S) (k^2 x k^2)."""
        return matrix_of(self.algebra, self.elements[index])

    def describe(self):
        elems = self.elements
        out = {"kind": KIND, "algebra": self.algebra, "k": int(self.k), "count": int(self.count),
               "sharing": self.sharing, "layout": self.layout,
               "cond": self.cond, "cond_min": float(np.min(self.cond_per_block)),
               "cond_limit": MAX_AUTOMORPHISM_COND,
               "cond_limit_status": AUTOMORPHISM_DOMAIN_STATUS,
               "degrees_of_freedom_per_element": self.degrees_of_freedom,
               "jordan_rank": self.jordan_rank,
               "class_chain": "scalar (1 dof) subset automorphism {P(w)} subset SPD (no closed form); "
                              "diagonal is outside except its scalar members",
               "stored_device_arrays": (["w (count,k)", "det(w)", "w^1/2", "det(w^1/2)", "w^-1/2", "det(w^-1/2)"]
                                        if self.algebra == "soc" else ["S (count,k,k)", "S^1/2", "S^-1/2"])}
        if self.count <= _ELEMENTS_IN_META:
            out["elements"] = elems.tolist()
        else:
            out["elements_sha256"] = hashlib.sha256(
                np.ascontiguousarray(elems.astype(np.float32)).tobytes()).hexdigest()
        if self.origin is not None:
            out["origin"] = dict(self.origin)
        return out

    def __repr__(self):
        return f"ConeAutomorphism(algebra={self.algebra}, k={self.k}, count={self.count}, sharing={self.sharing})"


# ================================================ declaration through block_metrics
def normalize_cone_metric(declared, term_index, block, size):
    """Validate an explicit ``ConeAutomorphism`` for one term; return ``(k, ranges)``.

    ``k`` is the term's row count and ``ranges`` is the single entry
    ``(0, 1, carrier)``, because the whole term is ONE plumbing block. How
    many cones the term stacks is ``carrier.cones``, never a range bound. The
    entry is in the tuple shape the norm-block path uses before
    ``check_static_steps`` appends its static field, so the batched dispatch,
    the step check and the forward-product application need no second code
    path for the key, and the range bounds of every carrier family alike live
    in ``[0, size // k)``.
    """
    struct = symmetric_cone_structure(block, size)
    if struct is None:
        raise ValueError(
            f"block_metrics[{term_index}]: a ConeAutomorphism is a metric for a symmetric-cone "
            f"block of exact type SecondOrderCone, SOCBatch or PSDCone; {type(block).__name__} "
            f"over {size} rows is not one.")
    algebra, k, cones = struct
    if declared.algebra != algebra or declared.k != k or declared.count not in (1, cones):
        raise ValueError(
            f"block_metrics[{term_index}]: ConeAutomorphism(algebra={declared.algebra}, "
            f"k={declared.k}, count={declared.count}) does not match {type(block).__name__} over "
            f"{size} rows (expected algebra={algebra}, k={k}, count={cones} or 1)")
    sharing = "term" if (declared.count == 1 and cones > 1) else "per_block"
    carrier = declared.bound(cone_layout(block), sharing)
    return int(size), ((0, 1, carrier),)


def check_uniform_step(sigma_rows, carrier, term_index, start, end):
    """The term's fixed dual step must be ONE value; refused otherwise, never averaged."""
    flat = np.asarray(sigma_rows, np.float64).reshape(-1)
    if not np.all(flat == flat[0]):
        raise ValueError(
            f"block_metrics[{term_index}]: an automorphism cone metric requires ONE dual step for "
            f"the whole term (the per-cone scaling is carried by the Jordan element w_i, since "
            f"c P(w) = P(sqrt(c) w)); rows {start}:{end} carry {np.unique(flat).size} distinct "
            "values. It is never averaged or collapsed silently.")


# ========================================================= the weighted prox
def prox_automorphism(carrier, v):
    """prox_{I_C}^{P(w)}(v) = W^{-1} Pi_C(W v), W = P(w^{1/2}); the term's flat vector."""
    R = rows(carrier.layout, v)
    if carrier.algebra == "soc":
        U = apply_soc(carrier.half, carrier.half_det, R)
        Z = apply_soc(carrier.invhalf, carrier.invhalf_det, soc_project(U))
    else:
        U = apply_psd(carrier.S_half, R)
        Z = apply_psd(carrier.S_invhalf, psd_project(U))
    return unrows(carrier.layout, Z)


def automorphism_moreau(block, v, s, ranges):
    """Dual update of one cone term: ``v - W Pi_C(W^{-1} v)`` per cone.

    This is ``prox_{f*}^{Sigma^{-1}}(v) = v - Sigma prox_f^{Sigma}(Sigma^{-1} v)``
    with ``Sigma = c P(w)``: ``Pi_C`` is positively homogeneous, so the scalar
    step ``c`` cancels and the update reads no step at all, exactly as the
    shipped cone prox uses its step only in a finiteness gate. A step that is
    not finite, positive and uniform over the term returns NaN (the backstop
    of ``check_static_steps``, which refuses it eagerly on the host).
    """
    del block  # a cone indicator carries no parameters
    carrier = ranges[0][2]
    s = jnp.broadcast_to(jnp.asarray(s, v.dtype), v.shape)
    valid = jnp.all(jnp.isfinite(s) & (s > 0)) & jnp.all(s == s[0])
    R = rows(carrier.layout, v)
    if carrier.algebra == "soc":
        U = apply_soc(carrier.invhalf, carrier.invhalf_det, R)
        Y = R - apply_soc(carrier.half, carrier.half_det, soc_project(U))
    else:
        U = apply_psd(carrier.S_invhalf, R)
        Y = R - apply_psd(carrier.S_half, psd_project(U))
    return jnp.where(valid, unrows(carrier.layout, Y), jnp.nan)


def apply_automorphism(carrier, u, power=1):
    """``P(w)^power u`` on the term's flat vector, power 1 or 0.5 (cached elements)."""
    R = rows(carrier.layout, u)
    if carrier.algebra == "soc":
        a, det = (carrier.w, carrier.w_det) if power == 1 else (carrier.half, carrier.half_det)
        return unrows(carrier.layout, apply_soc(a, det, R))
    A = carrier.S if power == 1 else carrier.S_half
    return unrows(carrier.layout, apply_psd(A, R))


# ============================================================ the domain, explicit
def cond_of(algebra, elements):
    """Per-element condition number of P(.) on the host, float64."""
    elements = np.asarray(elements, np.float64)
    if algebra == "soc":
        lam1, lam2, _ = _soc_eig64(elements.reshape(-1, elements.shape[-1]))
        return (lam1 / lam2) ** 2
    values = np.linalg.eigvalsh(0.5 * (elements + np.swapaxes(elements, -1, -2)))
    return (values.max(axis=-1) / values.min(axis=-1)) ** 2


def restrict_cone_condition(algebra, elements, max_cond=None):
    """Explicitly restrict elements to ``cond P <= max_cond``; returns ``(restricted, info)``.

    Second-order cone: the small Jordan eigenvalue ``lam2`` of every element
    with ``(lam1/lam2)^2 > max_cond`` is raised to ``lam1 / sqrt(max_cond)``
    (a hair inside, ``* (1 + 1e-9)``), keeping ``lam1`` and the frame; PSD: the
    eigenvalues of ``S`` below ``lam_max / sqrt(max_cond)`` are raised to that
    value. Both DOMINATE the original in the Loewner order (``P(w') >= P(w)``,
    ``S' kron S' >= S kron S``), so ``||Sigma^1/2 K T^1/2||`` cannot fall and a
    normalization measured on the original no longer certifies the step
    condition: it must be recomputed. NOTHING CALLS THIS FOR YOU; a supplied
    element is rejected, not clipped.
    """
    max_cond = MAX_AUTOMORPHISM_COND if max_cond is None else float(max_cond)
    if not (np.isfinite(max_cond) and max_cond >= 1.0):
        raise ValueError("max_cond must be finite and at least 1")
    elements = np.asarray(elements, np.float64)
    before = cond_of(algebra, elements)
    ratio = np.sqrt(max_cond)
    if algebra == "soc":
        w = elements.reshape(-1, elements.shape[-1])
        lam1, lam2, nb = _soc_eig64(w)
        floor = (lam1 / ratio) * (1 + 1e-9)
        lam2_new = np.maximum(lam2, floor)
        changed = lam2_new != lam2
        direction = w[:, 1:] / np.maximum(nb, np.finfo(np.float64).tiny)[:, None]
        out = np.concatenate([(0.5 * (lam1 + lam2_new))[:, None],
                              (0.5 * (lam1 - lam2_new))[:, None] * direction], axis=1)
        out = np.where(changed[:, None], out, w).reshape(elements.shape)
        detail = {"lam2_raised_from": lam2[changed].tolist(), "lam2_raised_to": lam2_new[changed].tolist()}
    else:
        k = elements.shape[-1]
        S = 0.5 * (elements + np.swapaxes(elements, -1, -2)).reshape(-1, k, k)
        values, vectors = np.linalg.eigh(S)
        top = values.max(axis=-1, keepdims=True)
        floor = (top / ratio) * (1 + 1e-9)
        values_new = np.maximum(values, floor)
        changed = np.any(values_new != values, axis=-1)
        out = np.matmul(vectors * values_new[..., None, :], np.swapaxes(vectors, -1, -2))
        out = 0.5 * (out + np.swapaxes(out, -1, -2))
        out = np.where(changed[:, None, None], out, S).reshape(elements.shape)
        detail = {"eigenvalues_before": values[changed].tolist(), "eigenvalues_after": values_new[changed].tolist()}
    after = cond_of(algebra, out)
    info = {"restricted": bool(np.any(changed)), "max_cond": max_cond,
            "cond_before": float(np.max(before)), "cond_after": float(np.max(after)),
            "cones_restricted": np.flatnonzero(changed).tolist(),
            "cond_before_per_cone": before.tolist(), "cond_after_per_cone": after.tolist(),
            "step_normalization": "must be recomputed against the restricted element", **detail}
    return out, info


# ======================================================== NT scaling point
def nt_scaling_point(block, s, z):
    """The Nesterov-Todd scaling point of an interior pair, as a ``ConeAutomorphism``.

    Reads only the block-local vectors ``s`` (the block's primal slack ``K x +
    l`` in the block's own layout) and ``z`` (its dual variable) and never
    touches the operator. Solves, per cone,

        P(w) s = z,    w = P(z^{1/2}) (P(z^{1/2}) s)^{-1/2}   (Jordan algebra),

    so that ``Sigma = P(w)`` carries a primal-space residual to a dual increment
    (units ``[z]/[s]``, the direction the PDHG dual update ``y + Sigma (K xbar
    + l)`` needs): for the isotropic pair ``s = alpha e``, ``z = beta e`` this
    gives ``Sigma = (beta/alpha) I``, the dual-over-primal balance that the
    diagonal equilibration also produces. The reversed convention ``P(w) z = s``
    yields the inverse element. Whether either is the better PDHG metric is
    NOT established by this release; this helper is an opt-in initializer.

    The element is det-normalized, ``w <- w / det_J(w)^{1/r}`` with the Jordan
    rank ``r`` (2 for a second-order cone, ``k`` for PSD), so only the shape is
    carried and the overall scale is left to the step normalization. Both
    vectors must be strictly interior; otherwise nothing is returned and
    nothing is projected inward. The compiler runs the constructor before any
    data is bound, so there is no current iterate at compile time: the caller
    supplies a representative interior pair, and re-running the constructor at
    a warm start is a later entry point, not this one.
    """
    struct = symmetric_cone_structure(block, np.asarray(s).size)
    if struct is None:
        raise ValueError(f"nt_scaling_point: {type(block).__name__} is not a symmetric-cone block "
                         "of exact type SecondOrderCone, SOCBatch or PSDCone")
    algebra, k, cones = struct
    layout = cone_layout(block)
    s64 = np.asarray(s, np.float64).reshape(-1)
    z64 = np.asarray(z, np.float64).reshape(-1)
    if s64.shape != z64.shape or s64.size != (k * cones if algebra == "soc" else k * k):
        raise ValueError("nt_scaling_point: s and z must both have the block's size")
    if not (np.all(np.isfinite(s64)) and np.all(np.isfinite(z64))):
        raise ValueError("nt_scaling_point: s and z must be finite")
    S = rows(layout, s64, np)
    Z = rows(layout, z64, np)
    if algebra == "soc":
        _, s_min, _ = _soc_eig64(S)
        _, z_min, _ = _soc_eig64(Z)
    else:
        S = 0.5 * (S + np.swapaxes(S, -1, -2))
        Z = 0.5 * (Z + np.swapaxes(Z, -1, -2))
        s_min = np.linalg.eigvalsh(S).min(axis=-1)
        z_min = np.linalg.eigvalsh(Z).min(axis=-1)
    bad = ~((s_min > 0) & (z_min > 0))
    if np.any(bad):
        j = int(np.argmax(bad))
        raise ValueError(
            f"nt_scaling_point: the pair is not interior on cone {j} (min Jordan eigenvalue s: "
            f"{s_min[j]!r}, z: {z_min[j]!r}); the Nesterov-Todd scaling point is defined for "
            "interior pairs only. No point is returned and nothing is projected inward for you.")
    if algebra == "soc":
        zh = soc_spectral64(Z, np.sqrt)
        u = apply_soc64(zh, soc_det64(zh), S)
        w = apply_soc64(zh, soc_det64(zh), soc_spectral64(u, lambda t: t ** -0.5))
        scale = np.sqrt(soc_det64(w))                       # det_J(w)^{1/2}, r = 2
        w = w / scale[:, None]
    else:
        zh = psd_spectral64(Z, np.sqrt)
        u = apply_psd64(zh, S)
        w = apply_psd64(zh, psd_spectral64(u, lambda t: t ** -0.5))
        w = 0.5 * (w + np.swapaxes(w, -1, -2))
        scale = np.linalg.det(w) ** (1.0 / k)               # det(S)^{1/k}, r = k
        w = w / scale[:, None, None]
        w = 0.5 * (w + np.swapaxes(w, -1, -2))
    origin = {"source": "nt_scaling_point", "direction": "P(w) s = z",
              "equation": "P(det_scale_removed[j] * w_j) s_j = z_j for every cone j; the stored "
                          "element is det-normalized, so P(w_j) s_j = z_j / det_scale_removed[j]^2",
              "det_normalized": True, "det_scale_removed": np.asarray(scale).tolist(),
              "merit_as_pdhg_metric": "not established"}
    return ConeAutomorphism(algebra, w, layout=layout, sharing="per_block", origin=origin)
