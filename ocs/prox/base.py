"""The proximal-block interface: OCS's prox library.

A block is accessed only through its scaled proximal operator

    prox_f^D(v) = argmin_u  f(u) + (1/2)(u - v)^T D (u - v).

Every block declares:
  * ``prox(v, t)`` — the scalar-metric prox, D = (1/t) I (always available);
  * ``scaling_class`` — the richest implemented metric capability:
        "scalar"   : (1/t) I
        "diagonal" : diag(d), d > 0
        "matrix"   : any symmetric positive-definite D
    Coordinate separability suffices for diagonal metrics; supported dense
    quadratics also accept diagonal metrics. These labels describe this
    implementation, not universal mathematical impossibility boundaries.
  * ``smooth_grad`` — grad f if f is smooth (used for exact termination), else None.

The calculus layer (``scaled``, ``precomposed``, ``with_linear_quadratic``)
folds a positive scaling, an admissible affine pre-composition, and a convex
linear+quadratic term into a base block through metric-preserving transforms.
Coupled scalar-tier blocks require uniform |a| and uniform quadratic curvature.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SCALAR, DIAGONAL, MATRIX = "scalar", "diagonal", "matrix"


def _parameter(value, name):
    """Calculus parameters are fixed, finite scalars or coordinate vectors."""
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or nonempty coordinate vector")
    return jnp.asarray(array)


def _validate_step(v, t, scaling_class):
    """Reject invalid concrete metrics; traced invalid metrics propagate NaN.

    A vector step is accepted for a scalar-tier block only when every entry is
    equal, as produced by the compiler's tied equilibration spans.
    """
    dtype = jnp.result_type(jnp.asarray(v).dtype, jnp.float32)
    traced = any(isinstance(leaf, jax.core.Tracer)
                 for leaf in jax.tree_util.tree_leaves(t))
    if not traced:
        # A fixed metric is concrete only when its containers do not hide a
        # traced leaf. Array-protocol conversion can itself reveal a tracer.
        with jax.ensure_compile_time_eval():
            concrete = jnp.asarray(t, dtype=dtype)
        if not isinstance(concrete, jax.core.Tracer):
            # Preserve the backend's own conversion/comparison semantics.
            with jax.ensure_compile_time_eval():
                if jnp.broadcast_shapes(concrete.shape, jnp.shape(v)) != jnp.shape(v):
                    raise ValueError("prox step must be scalar or broadcast to the point's shape")
                valid = jnp.all(jnp.isfinite(concrete) & (concrete > 0))
                if scaling_class == SCALAR:
                    valid &= jnp.all(concrete == concrete.reshape(-1)[0])
                accepted = bool(valid)
            if accepted:
                return concrete
        # Invalid concrete metrics retain the original evaluator below. It
        # raises eagerly but propagates NaN under any enclosing trace, even
        # when both this point and this metric are closed-over constants.
        t = concrete
    t = jnp.asarray(t, dtype=dtype)
    if jnp.broadcast_shapes(t.shape, jnp.shape(v)) != jnp.shape(v):
        raise ValueError("prox step must be scalar or broadcast to the point's shape")
    valid = jnp.all(jnp.isfinite(t) & (t > 0))
    if scaling_class == SCALAR:
        valid &= jnp.all(t == t.reshape(-1)[0])
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("prox step must be finite and positive, and uniform for a scalar-tier block")
    return jnp.where(valid, t, jnp.nan)


def _finite_step(t):
    """Retain NaN propagation without a second large static reduction."""
    if not isinstance(t, jax.core.Tracer):
        with jax.ensure_compile_time_eval():
            return bool(jnp.all(jnp.isfinite(t)))
    return jnp.all(jnp.isfinite(t))


def _validate_matrix_metric(v, D):
    D = jnp.asarray(D, dtype=jnp.result_type(jnp.asarray(v).dtype, jnp.float32))
    if v.ndim != 1 or D.shape != (v.size, v.size):
        raise ValueError("matrix metric must be square and match the point's dimension")
    valid = jnp.all(jnp.isfinite(D)) & jnp.all(D == D.T)
    valid &= jnp.all(jnp.linalg.eigvalsh(D) > 0)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("matrix metric must be finite, symmetric, and positive definite")
    return jnp.where(valid, D, jnp.nan)


def _inherit_contract(wrapper, base):
    wrapper.scaling_class = base.scaling_class
    if base.scaling_class == MATRIX and type(base).prox_matrix is ProxBlock.prox_matrix:
        # A custom declaration alone does not supply a full-matrix oracle.
        wrapper.scaling_class = DIAGONAL
    wrapper.coordinate_separable = getattr(base, "coordinate_separable", False)
    wrapper.eval_kind = base.eval_kind
    if hasattr(base, "dim"):
        wrapper.dim = base.dim


def _has_oracle(block, name):
    """Whether this instance supplies a value/conjugate oracle, including wrappers."""
    if name not in ("value", "conj_value"):
        return callable(getattr(block, name, None))
    declared = getattr(block, name + "_available", None)
    if declared is not None:
        return bool(declared)
    implementation = getattr(type(block), name, None)
    return callable(implementation) and implementation is not getattr(ProxBlock, name)


def _check_parameter_dimension(block, *parameters):
    if hasattr(block, "dim"):
        for parameter in parameters:
            if np.broadcast_shapes(parameter.shape, (block.dim,)) != (block.dim,):
                raise ValueError("calculus parameter shape must match the base block's dimension")


def _fixed_domain_ok(block, point):
    """Check exact indicator domains when a coordinate is fixed by zero a.

    Diagnostic value oracles can allow roundoff-sized violations. A constant
    outside the actual domain nevertheless makes an affine composition improper.
    """
    from .library import NonNeg, Equality, Box
    if type(block) is NonNeg:
        return jnp.all(point >= 0)
    if type(block) is Equality:
        return jnp.all(point == 0)
    if type(block) is Box:
        return jnp.all((point >= block.lo) & (point <= block.hi))
    if type(block) in (_Scaled, _LinQuad):
        return _fixed_domain_ok(block.base, point)
    if type(block) is _AffinePre:
        return _fixed_domain_ok(block.base, block.a * point - block.b)
    return jnp.isfinite(block.value(point))


def _known_quadratic(block):
    # Imports are delayed because the built-ins inherit ProxBlock from this module.
    from .library import SumSquares, Quadratic, Zero
    if type(block) in (SumSquares, Quadratic, Zero):
        return True
    return type(block) in (_Scaled, _AffinePre, _LinQuad) and _known_quadratic(block.base)


def _quadratic_coefficients(block, size, dtype):
    """Exact algebraic H,q for a recognized quadratic, including singular H."""
    from .library import SumSquares, Quadratic, Zero
    if type(block) is SumSquares:
        return jnp.eye(size, dtype=dtype), jnp.zeros(size, dtype=dtype)
    if type(block) is Zero:
        return jnp.zeros((size, size), dtype=dtype), jnp.zeros(size, dtype=dtype)
    if type(block) is Quadratic:
        return block.P, block.q
    H, q = _quadratic_coefficients(block.base, size, dtype)
    if type(block) is _Scaled:
        return block.c * H, block.c * q
    if type(block) is _AffinePre:
        a = jnp.broadcast_to(block.a, (size,))
        b = jnp.broadcast_to(block.b, (size,))
        transformed = a[:, None] * H * a[None, :]
        hb = jnp.matmul(H, b, precision=jax.lax.Precision.HIGHEST)
        return 0.5 * (transformed + transformed.T), a * (q - hb)
    return H + jnp.diag(jnp.broadcast_to(block.e, (size,))), q + block.d


def _diagonal_quadratic_coefficients(block):
    """Scalar/vector quadratic coefficients; never allocate a dense Hessian."""
    from .library import SumSquares, Quadratic, Zero
    if type(block) is SumSquares:
        return 1.0, 0.0
    if type(block) is Zero:
        return 0.0, 0.0
    if type(block) is Quadratic:
        return jnp.diag(block.P), block.q
    h, q = _diagonal_quadratic_coefficients(block.base)
    if type(block) is _Scaled:
        return block.c * h, block.c * q
    if type(block) is _AffinePre:
        return h * block.a * block.a, block.a * (q - h * block.b)
    return h + block.e, q + block.d


def _absolute_value_slopes(block):
    """Left/right slopes at zero for the built-in two-piece losses.

    Read parameters at evaluation time so grouped wrappers use their own
    traced coefficients rather than cached values from a representative.
    """
    from .library import L1, Pinball
    if type(block) is L1:
        return -block.lam, block.lam
    if type(block) is Pinball:
        return block.tau - 1.0, block.tau
    if type(block) is _Scaled:
        slopes = _absolute_value_slopes(block.base)
        if slopes is not None:
            return block.c * slopes[0], block.c * slopes[1]
    return None


def _quadratic_conjugate(block, y):
    y = jnp.asarray(y, dtype=jnp.result_type(jnp.asarray(y).dtype, jnp.float32))
    if block.coordinate_separable:
        h, q = _diagonal_quadratic_coefficients(block)
        h, shifted = jnp.broadcast_arrays(jnp.asarray(h, dtype=y.dtype), y - q)
        positive = h > 0
        value = 0.5 * jnp.sum(jnp.where(positive, shifted * shifted /
                                       jnp.where(positive, h, 1.0), 0.0))
        in_range = jnp.all(positive | (shifted == 0))
        valid = jnp.all(jnp.isfinite(h) & (h >= 0) & jnp.isfinite(shifted))
        return jnp.where(valid, jnp.where(in_range, value - block.value(jnp.zeros_like(y)),
                                        jnp.inf), jnp.nan)
    from .library import _quadratic_conj_value
    H, q = _quadratic_coefficients(block, y.size, y.dtype)
    return _quadratic_conj_value(H, q, y) - block.value(jnp.zeros_like(y))


class ProxBlock:
    scaling_class = SCALAR
    eval_kind = "closed_form"                # closed_form | spectral | newton | bisection | cg
    smooth_grad = None                       # override with a callable if smooth

    def prox(self, v, t):
        """prox_{t f}(v) = argmin_u f(u) + (1/2t)||u - v||_2^2.
        ``t`` is a scalar (scalar class) or a per-coordinate vector (diagonal)."""
        raise NotImplementedError

    def prox_matrix(self, v, D):
        """Full-matrix-metric prox: argmin_u f(u) + (1/2)(u-v)^T D (u-v), D a dense
        PD matrix. The built-in quadratic implementations solve
        (Q + D) u = D v - q. Blocks without an implementation raise."""
        raise NotImplementedError(
            f"{type(self).__name__} is scaling_class '{self.scaling_class}': a full-"
            "matrix proximal operator is not implemented for this block")

    def value(self, z):
        """f(z) -- the block's value, for the primal objective / duality gap."""
        raise NotImplementedError

    def conj_value(self, y):
        """f^*(y) -- the conjugate value, for the dual objective / duality gap."""
        raise NotImplementedError

    # ---- OPTIONAL objective-lower-bound capability ------------------------
    # ``conj_value`` alone cannot support a valid objective lower bound: several
    # library conjugates admit a feasibility tolerance in their indicator, and
    # the gap reporting path replaces +inf by 0 (the "finite-part surrogate").
    # Both make the reported dual value a surrogate, not a bound: it can sit
    # ABOVE the primal value at the same point.  A block opts in to the bound
    # only by declaring ALL THREE of the following exactly.  A block that does
    # not opt in makes the objective certificate UNAVAILABLE for the whole
    # problem; it is never silently replaced by a surrogate.
    #
    # The prox-only interface is unchanged: nothing here is required of a
    # block, and no solver path consults it.
    bound_capable = False

    #: True when ``f`` is the indicator of a set, so that ``value`` is 0 on the
    #: set and +inf off it and ``prox`` is the PROJECTION onto it for every
    #: step.  The objective-gap acceptance uses this to measure relative
    #: primal feasibility separately from the objective gap, because a gap
    #: computed at an infeasible point is not a bound on anything.
    is_indicator = False

    def conj_in_domain(self, y):
        """Exact, tolerance-free boolean: is y in dom f^*?

        No slack of any kind is admissible here.  A block that cannot decide
        this exactly must leave ``bound_capable`` False.

        This test is LOCAL to the block.  It is necessary for dual feasibility
        and nowhere near sufficient: the dual must also satisfy the coupling
        condition ``-K^T y - c in dom g^*`` (for ``g = 0``, exact equality
        ``K^T y + c = 0``), which no per-block test can see.  The compiler
        checks that separately and refuses a bound without it.

        Epistemic limit: "tolerance-free" here means tolerance-free on the
        REPRESENTED float32 data.  That is not a rigorous real-arithmetic
        enclosure, and agreement with an independent float64 evaluation is
        evidence rather than proof."""
        raise NotImplementedError

    #: True when the objective-bound evaluator must decide this block's
    #: conjugate-domain membership with ``conj_in_domain_resolved`` (a
    #: resolved sign) instead of ``conj_in_domain``: the PSD cone.
    resolved_sign_domain = False

    def conj_in_domain_resolved(self, y):
        """``conj_in_domain`` with a RESOLVED sign, for the objective-bound
        evaluator.  Never looser than ``conj_in_domain``.  Blocks whose
        membership decision has no unresolved band (orthants, boxes, norm
        balls with their own conservative envelopes) inherit it unchanged;
        ``PSDCone`` overrides it. """
        return self.conj_in_domain(y)

    def conj_domain_radius(self, y):
        """Largest s >= 0 with s*y in dom f^*  (+inf when dom f^* is all of R^m).

        This is what makes a returned dual usable as a bound: the lam/||K^T y||_inf
        scaling of an l1 audit is exactly this radius for L1, and expressing it
        per block is what generalises that scaling off one family.  It is a hint
        for the search; every point it produces is re-checked with
        ``conj_in_domain``.  Operator-free: it sees only the block's own argument."""
        raise NotImplementedError

    def conj_domain_project(self, y):
        """Euclidean projection of y onto (the closure of) dom f^*.

        Used to CORRECT a dual's conjugate-domain violation instead of shrinking
        the whole direction for it.  Like the radius, it is only a proposal: the
        corrected point is re-checked with ``conj_in_domain`` before any bound
        built from it is reported."""
        raise NotImplementedError

    def conj_domain_project_exact(self, y):
        """The projection proposal WITHOUT its interior safety adjustment.

        Some projections pull the proposal a few ulps inside the domain so the
        exact re-check is not decided by rounding (the second-order cones).
        Under ``witness_contraction="once"`` the dual correction rounds use this
        exact projection and apply the safety adjustment one time, at the end,
        on the materialized candidate; the candidate is then re-checked by the
        unchanged ``conj_in_domain`` and coupling gates.  Blocks whose
        ``conj_domain_project`` adds no such adjustment inherit it unchanged."""
        return self.conj_domain_project(y)

    def group_key(self):
        """Compatibility/diagnostic label, not a batching equivalence contract.

        Dispatch packs parameters for recognized built-ins explicitly. Custom
        blocks only share a batch when the same object instance is repeated.
        """
        return type(self).__name__

    # ---- calculus (POGS-style); each returns a new ProxBlock -------------
    def scaled(self, c):
        return _Scaled(self, c)

    def precomposed(self, a=1.0, b=0.0):
        return _AffinePre(self, a, b)

    def with_linear_quadratic(self, d=0.0, e=0.0):
        return _LinQuad(self, d, e)


class _Scaled(ProxBlock):
    """h(u) = c f(u).  prox_{t h} = prox_{(t c) f}."""
    def __init__(self, base, c):
        coefficient = np.asarray(_parameter(c, "c"))
        if coefficient.ndim != 0 or not np.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("scaling coefficient c must be a finite positive scalar; zero scaling of an extended-valued block is undefined")
        self.base, self.c = base, float(coefficient)
        _inherit_contract(self, base)
        # The exact conjugate-domain contract scales with c.  For h = c f,
        # h^*(y) = c f^*(y / c), so dom h^* = c dom f^*: membership is tested on
        # y / c, the admissible radius of y is the radius of y / c, and the
        # projection onto c dom f^* is c times the projection of y / c.  It is
        # forwarded only when the base declares the contract exactly; nothing is
        # invented for a base that does not, and c > 0 is enforced above.
        self.bound_capable = bool(getattr(base, "bound_capable", False))
        self.is_indicator = bool(getattr(base, "is_indicator", False))
        self.value_available = _has_oracle(base, "value")
        self.conj_value_available = _has_oracle(base, "conj_value")
        if base.smooth_grad is not None:
            self.smooth_grad = lambda s, g=base.smooth_grad, c=self.c: c * g(s)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        result = self.base.prox(v, t * self.c)
        return jnp.where(_finite_step(t), result, jnp.nan)

    def prox_matrix(self, v, D):
        if self.scaling_class != MATRIX:
            return super().prox_matrix(v, D)
        return self.base.prox_matrix(v, _validate_matrix_metric(v, D) / self.c)

    def value(self, z):
        return self.c * self.base.value(z)

    def conj_value(self, y):
        return self.c * self.base.conj_value(y / self.c)

    def conj_in_domain(self, y):
        return self.base.conj_in_domain(y / self.c)

    @property
    def resolved_sign_domain(self):
        # Read from the base, not stored: wrappers keep their instance fields.
        return bool(getattr(self.base, "resolved_sign_domain", False))

    def conj_in_domain_resolved(self, y):
        return self.base.conj_in_domain_resolved(y / self.c)

    def conj_domain_radius(self, y):
        return self.base.conj_domain_radius(y / self.c)

    def conj_domain_project(self, y):
        return self.c * self.base.conj_domain_project(y / self.c)

    def conj_domain_project_exact(self, y):
        return self.c * self.base.conj_domain_project_exact(y / self.c)


class _AffinePre(ProxBlock):
    """h(u) = f(a u - b), elementwise a, b.
    prox_{t h}(v) = (prox_{t a^2 f}(a v - b) + b) / a.

    Recognized quadratics and scaled L1/Pinball losses use original-coordinate
    formulas. The generic transform can lose accuracy when the inner proximal
    displacement is smaller than floating-point resolution at a*v-b; it is
    an exact-arithmetic identity, not a universal floating-point guarantee.
    """
    def __init__(self, base, a, b):
        self.base = base
        self.a, self.b = _parameter(a, "a"), _parameter(b, "b")
        np.broadcast_shapes(self.a.shape, self.b.shape)
        _inherit_contract(self, base)
        _check_parameter_dimension(self, self.a, self.b)
        self._has_zero = bool(np.any(np.asarray(self.a) == 0))
        self._quadratic = _known_quadratic(base)
        self.value_available = _has_oracle(base, "value")
        self.conj_value_available = ((self._quadratic and self.value_available) if self._has_zero
                                     else _has_oracle(base, "conj_value"))
        if base.scaling_class == SCALAR:
            magnitudes = np.abs(np.asarray(self.a)).reshape(-1)
            if self._has_zero or not np.all(magnitudes == magnitudes[0]):
                raise ValueError("a scalar-tier coupled block requires a common nonzero absolute affine coefficient")
        if self._has_zero and not self.coordinate_separable and not self._quadratic:
            raise ValueError("zero affine coefficients require a separable block or a recognized quadratic")
        if self._has_zero and self.scaling_class == MATRIX and not self._quadratic:
            self.scaling_class = DIAGONAL
        if self._has_zero and self.coordinate_separable and not self._quadratic:
            # A fixed coordinate outside dom(f_i) makes the entire composition
            # improper. Use a proximal point to supply feasible other coordinates.
            aa, bb = jnp.broadcast_arrays(jnp.atleast_1d(self.a), jnp.atleast_1d(self.b))
            try:
                feasible = base.prox(-bb, jnp.ones_like(bb))
                fixed_ok = _fixed_domain_ok(base, jnp.where(aa == 0, -bb, feasible))
            except NotImplementedError as error:
                raise ValueError("zero affine coefficients require a value oracle to check fixed-coordinate feasibility") from error
            if not bool(fixed_ok):
                raise ValueError("zero affine coefficient fixes a coordinate outside the base block's domain")
        if self._has_zero and not self.coordinate_separable:
            self.eval_kind = "linear_solve"
        if base.smooth_grad is not None:
            self.smooth_grad = lambda s: self.a * base.smooth_grad(self.a * s - self.b)
        self.bound_capable = bool(getattr(base, "bound_capable", False) and not self._has_zero)
        self.is_indicator = bool(getattr(base, "is_indicator", False))

    def conj_in_domain(self, y):
        if not self.bound_capable:
            return super().conj_in_domain(y)
        return jnp.all(jnp.isfinite(y)) & self.base.conj_in_domain(y / self.a)

    @property
    def resolved_sign_domain(self):
        return bool(getattr(self.base, "resolved_sign_domain", False))

    def conj_in_domain_resolved(self, y):
        if not self.bound_capable:
            return super().conj_in_domain_resolved(y)
        return jnp.all(jnp.isfinite(y)) & self.base.conj_in_domain_resolved(y / self.a)

    def conj_domain_radius(self, y):
        if not self.bound_capable:
            return super().conj_domain_radius(y)
        return self.base.conj_domain_radius(y / self.a)

    def conj_domain_project(self, y):
        if not self.bound_capable:
            return super().conj_domain_project(y)
        # A domain-feasible proposal. For nonuniform a and a coupled block
        # it need not be the Euclidean projection in the original coordinates.
        return self.a * self.base.conj_domain_project(y / self.a)

    def conj_domain_project_exact(self, y):
        if not self.bound_capable:
            return super().conj_domain_project_exact(y)
        return self.a * self.base.conj_domain_project_exact(y / self.a)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        if self._quadratic and self.coordinate_separable:
            h, q = _diagonal_quadratic_coefficients(self)
            return (v - t * q) / (1.0 + t * h)
        if self._quadratic:
            H, q = _quadratic_coefficients(self, v.size, v.dtype)
            metric = jnp.broadcast_to(1.0 / t, v.shape)
            return jnp.linalg.solve(H + jnp.diag(metric), metric * v - q)
        a, b = self.a, self.b
        safe_a = jnp.where(a != 0, a, 1.0)
        slopes = _absolute_value_slopes(self.base)
        if slopes is not None:
            # At the kink u=b/a, the subgradient spans the two original-unit
            # slopes a*s_left and a*s_right (their order flips when a<0).
            # Clip the displacement directly, without subtracting two nearly
            # equal values in the transformed coordinates.
            first, second = -(t * a) * slopes[0], -(t * a) * slopes[1]
            shift = jnp.clip((b - a * v) / safe_a,
                             jnp.minimum(first, second), jnp.maximum(first, second))
            result = jnp.where(a != 0, v + shift, v)
            return jnp.where(_finite_step(t), result, jnp.nan)
        transformed = a * v - b
        result = v + (self.base.prox(transformed, t * safe_a * safe_a) - transformed) / safe_a
        result = jnp.where(a != 0, result, v)
        return jnp.where(_finite_step(t), result, jnp.nan)

    def prox_matrix(self, v, D):
        if self.scaling_class != MATRIX:
            return super().prox_matrix(v, D)
        D = _validate_matrix_metric(v, D)
        if self._has_zero:
            if not self._quadratic:
                raise NotImplementedError("singular affine full-matrix prox requires a recognized quadratic")
            H, q = _quadratic_coefficients(self, v.size, v.dtype)
            rhs = jnp.matmul(D, v, precision=jax.lax.Precision.HIGHEST) - q
            return jnp.linalg.solve(H + D, rhs)
        a = jnp.broadcast_to(self.a, v.shape)
        transformed = (D / a[:, None]) / a[None, :]
        transformed = 0.5 * (transformed + transformed.T)
        return (self.base.prox_matrix(a * v - self.b, transformed) + self.b) / a

    def value(self, z):
        return self.base.value(self.a * z - self.b)

    def conj_value(self, y):
        if self._has_zero:
            if self._quadratic:
                return _quadratic_conjugate(self, y)
            raise NotImplementedError("a noninvertible affine composition requires a coordinate conjugate oracle")
        dual = y / self.a
        return self.base.conj_value(dual) + jnp.sum(dual * self.b)


class _LinQuad(ProxBlock):
    """h(u) = f(u) + d^T u + (1/2) u^T diag(e) u.
    prox_{t h}(v) = prox_{t' f}((v - t d)/(1 + t e)),  t' = t/(1 + t e)."""
    def __init__(self, base, d, e):
        self.base = base
        self.d, self.e = _parameter(d, "d"), _parameter(e, "e")
        np.broadcast_shapes(self.d.shape, self.e.shape)
        if np.any(np.asarray(self.e) < 0):
            raise ValueError("quadratic coefficients e must be nonnegative")
        _inherit_contract(self, base)
        _check_parameter_dimension(self, self.d, self.e)
        curvature = np.asarray(self.e).reshape(-1)
        if base.scaling_class == SCALAR and not np.all(curvature == curvature[0]):
            raise ValueError("a scalar-tier coupled block requires uniform added quadratic curvature")
        self._e_mode = 0 if np.all(curvature == 0) else (1 if np.all(curvature > 0) else 2)
        self._quadratic = _known_quadratic(base)
        self.value_available = _has_oracle(base, "value")
        self.conj_value_available = (_has_oracle(base, "conj_value") if self._e_mode == 0
                                     else self.value_available and (self._e_mode == 1 or self._quadratic))
        if base.smooth_grad is not None:
            self.smooth_grad = lambda s: base.smooth_grad(s) + self.d + self.e * s

    @property
    def bound_capable(self):
        # A linear shift translates the domain. Strictly positive curvature
        # makes the conjugate finite everywhere, with its actual compound
        # value below; a flag alone is never used in place of that value.
        declared = (type(self.base).__module__.startswith("ocs.prox.")
                    or "bound_capable" in type(self.base).__dict__)
        return bool(declared and getattr(self.base, "bound_capable", False)
                    and self.conj_value_available and self._e_mode in (0, 1))

    def conj_in_domain(self, y):
        if not self.bound_capable:
            return super().conj_in_domain(y)
        finite = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(self.d))
        if self._e_mode == 0:
            return finite & self.base.conj_in_domain(y - self.d)
        return finite & jnp.all(jnp.isfinite(self.e) & (self.e > 0))

    @property
    def resolved_sign_domain(self):
        return bool(getattr(self.base, "resolved_sign_domain", False))

    def conj_in_domain_resolved(self, y):
        if not self.bound_capable:
            return super().conj_in_domain_resolved(y)
        finite = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(self.d))
        if self._e_mode == 0:
            return finite & self.base.conj_in_domain_resolved(y - self.d)
        return finite & jnp.all(jnp.isfinite(self.e) & (self.e > 0))

    def conj_domain_radius(self, y):
        if not self.bound_capable:
            return super().conj_domain_radius(y)
        # For a translated domain this is a search upper bound, not a claim
        # that all points on the ray are feasible. Every trial is checked by
        # conj_in_domain before its value can be used. A zero linear term
        # retains the base's exact radial bound.
        if self._e_mode == 0:
            return jnp.where(jnp.all(self.d == 0), self.base.conj_domain_radius(y),
                             jnp.where(jnp.all(jnp.isfinite(y)), jnp.inf, 0.))
        return jnp.where(jnp.all(jnp.isfinite(y)), jnp.inf, 0.)

    def conj_domain_project(self, y):
        if not self.bound_capable:
            return super().conj_domain_project(y)
        if self._e_mode == 0:
            return self.d + self.base.conj_domain_project(y - self.d)
        return y

    def conj_domain_project_exact(self, y):
        if not self.bound_capable:
            return super().conj_domain_project_exact(y)
        if self._e_mode == 0:
            return self.d + self.base.conj_domain_project_exact(y - self.d)
        return y

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        scale = 1.0 + t * self.e
        result = self.base.prox((v - t * self.d) / scale, t / scale)
        return jnp.where(_finite_step(t), result, jnp.nan)

    def prox_matrix(self, v, D):
        if self.scaling_class != MATRIX:
            return super().prox_matrix(v, D)
        D = _validate_matrix_metric(v, D)
        metric = D + jnp.diag(jnp.broadcast_to(self.e, v.shape))
        rhs = jnp.matmul(D, v, precision=jax.lax.Precision.HIGHEST) - self.d
        center = jnp.linalg.solve(metric, rhs)
        return self.base.prox_matrix(center, metric)

    def value(self, z):
        return self.base.value(z) + jnp.sum(self.d * z) + 0.5 * jnp.sum(self.e * z * z)

    def conj_value(self, y):
        shifted = y - self.d
        from .library import NonNeg
        if type(self.base) is NonNeg and self._e_mode == 1:
            positive = jnp.maximum(shifted, 0.)
            value = .5 * jnp.sum(positive * positive / self.e)
            return jnp.where(self.conj_in_domain(y), value, jnp.inf)
        if self._e_mode == 0:
            return self.base.conj_value(shifted)
        if self._e_mode == 1:
            # Completing the square gives the unique Fenchel maximizer.
            point = self.base.prox(shifted / self.e, 1.0 / self.e)
            value = jnp.sum(shifted * point) - self.base.value(point) - 0.5 * jnp.sum(self.e * point * point)
            if self.base.smooth_grad is not None:
                # A finite inner Newton solve need not attain the supremum.
                # Strong convexity bounds its missing conjugate value from
                # above, so an approximate prox cannot overstate a dual lower.
                residual = self.base.smooth_grad(point) + self.e * point - shifted
                value = value + 0.5 * jnp.sum(residual * residual / self.e)
            return value
        if self._quadratic:
            return _quadratic_conjugate(self, y)
        raise NotImplementedError("partly zero added curvature requires a coordinate conjugate oracle")
