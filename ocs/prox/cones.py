"""Cone projections (prox of cone indicators): second-order, PSD, exponential,
power.  Each is coupled across its block, so its scaling class is "scalar".
All are vmap-able (one cone per row), which is what the type-grouped dispatch
relies on. Exponential and power program support remains experimental.

The conjugate of a cone indicator is the indicator of its polar C^o = -C^*.
Even for a self-dual cone, the conjugate domain is the NEGATIVE cone.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .base import ProxBlock, SCALAR, _validate_step
from ._exponential_cone import project_unit as _proj_exp_unit
from ._power_cone import proj_pow

_TOL = 1e-7

#: Resolved-sign envelope of the PSD cone's evaluator membership test
#: (``PSDCone.conj_in_domain_resolved``): lambda_max(sym y) must not exceed
#: -PSD_RESOLVED_SIGN_C * k * eps32 * ||sym y||_2, the PSD analogue of the
#: second-order cone's ``(2 n + 8) eps`` in ``_conservative_norm_leq``.  It is
#: below the conservative projection's interior margin 8 k eps32 max|lambda|,
#: so a once-adjusted proposal clears it.
PSD_RESOLVED_SIGN_C = 2.0


def _zero_slices(S):
    return jnp.all(S == 0, axis=(-2, -1))


def sym_eigvalsh(S):
    """``jnp.linalg.eigvalsh`` of symmetric (..., k, k) slices, exact on zero slices.

    The GPU eigensolver can return NaN for an exactly zero matrix, in some
    calls and not in others (RTX 5090, JAX 0.11.1, 2026-10-27), and the dual
    of a strictly feasible PSD block converges to exactly zero.  The spectrum
    of a zero slice is known, so the eigensolver never sees one: an identity
    stands in and its eigenvalues are replaced by zeros.  Every other slice
    takes the unchanged call, so its eigenvalues are bit-identical.
    """
    zero = _zero_slices(S)
    eye = jnp.eye(S.shape[-1], dtype=S.dtype)
    w = jnp.linalg.eigvalsh(jnp.where(zero[..., None, None], eye, S))
    return jnp.where(zero[..., None], jnp.zeros_like(w), w)


def sym_eigh(S):
    """``jnp.linalg.eigh`` of symmetric (..., k, k) slices, exact on zero slices.

    As ``sym_eigvalsh``. A zero slice returns zero eigenvalues and the
    identity basis, an exact eigendecomposition of the zero matrix.
    """
    zero = _zero_slices(S)
    eye = jnp.eye(S.shape[-1], dtype=S.dtype)
    w, U = jnp.linalg.eigh(jnp.where(zero[..., None, None], eye, S))
    return jnp.where(zero[..., None], jnp.zeros_like(w), w), U


# =====================================================================
# Second-order cone  { (s, z) : ||z||_2 <= s }   (s is the first entry)
# =====================================================================
def _conservative_norm_leq(z, radius, axis=0):
    """Sufficient represented-data norm test, refusing unresolved roundoff.

    Scale before squaring to avoid overflow/underflow. The margin covers the
    scaled squares and reduction; it is subtracted from admissibility, never
    added as feasibility slack. Axis-aligned cases and small integer square
    sums are exactly decidable, including their boundary. No f64 vector or
    global JAX precision change is introduced.
    """
    a = jnp.abs(z)
    scale = jnp.max(a, axis=axis)
    safe = jnp.where(scale > 0, scale, 1.)
    normalized = z / jnp.expand_dims(safe, axis)
    ss = jnp.sum(normalized * normalized, axis=axis)
    rr = radius / safe
    gamma = (2 * z.shape[axis] + 8) * jnp.finfo(z.dtype).eps
    interior = ss * (1. + gamma) <= rr * rr * (1. - gamma)
    axis_exact = (jnp.sum(z != 0, axis=axis) <= 1) & (scale <= radius)
    integer = (jnp.all(z == jnp.round(z), axis=axis)
               & (radius == jnp.round(radius)) & (scale <= 4096.)
               & (radius <= 4096.))
    small_z = jnp.clip(z, -4096., 4096.)
    exact_ss = jnp.sum(small_z * small_z, axis=axis)
    integer_exact = (integer & (z.shape[axis] * scale * scale <= 2**22)
                     & (exact_ss <= radius * radius))
    normal = jnp.asarray(True)
    if z.dtype == jnp.float32:
        # CUDA may flush subnormal arithmetic. Refuse that unresolved case
        # using the stored bits, not an arithmetic comparison with zero.
        bits = jax.lax.bitcast_convert_type(z, jnp.uint32) & jnp.uint32(0x7fffffff)
        rb = jax.lax.bitcast_convert_type(radius, jnp.uint32) & jnp.uint32(0x7fffffff)
        normal = (jnp.all((bits == 0) | (bits >= 0x00800000), axis=axis)
                  & ((rb == 0) | (rb >= 0x00800000)))
    return (jnp.all(jnp.isfinite(z), axis=axis) & jnp.isfinite(radius)
            & normal & (radius >= 0) & (interior | axis_exact | integer_exact))


#: Float32 ulps by which ``SecondOrderCone.conj_domain_project`` pulls the
#: spatial part of a float32 proposal inside the polar (since 0.3.8).  The
#: Moreau residual is formed in scoped float64 and rounded once, so it lies
#: within about one ulp of the boundary; 16 ulps clear the float64 membership
#: test below at every dimension.
SOC_INSIDE_PULL_ULPS = 16


def _float64_norm_leq(z, radius, axis=0):
    """``_conservative_norm_leq`` decided on the represented float32 values in
    scoped float64 (since 0.3.8).

    A float32 value converts to float64 exactly, so the test is the same
    sufficient test on the same numbers, with the envelope (2 n + 8) eps64 in
    place of (2 n + 8) eps32.  The float32 envelope grows with the dimension:
    at n = 512 it is 1.2e-4, larger than the tolerances the objective bound is
    asked to certify.  The float32 subnormal refusal is kept, decided on the
    stored bits before the conversion.  The barrier keeps XLA from re-deriving
    z at higher precision, as in ``PSDCone.conj_in_domain_resolved``.
    Float64 inputs are decided by ``_conservative_norm_leq`` unchanged.
    """
    if z.dtype != jnp.float32:
        return _conservative_norm_leq(z, radius, axis)
    bits = jax.lax.bitcast_convert_type(z, jnp.uint32) & jnp.uint32(0x7fffffff)
    rb = jax.lax.bitcast_convert_type(radius, jnp.uint32) & jnp.uint32(0x7fffffff)
    normal = (jnp.all((bits == 0) | (bits >= 0x00800000), axis=axis)
              & ((rb == 0) | (rb >= 0x00800000)))
    with jax.enable_x64():
        z64 = jax.lax.optimization_barrier(z).astype(jnp.float64)
        r64 = jax.lax.optimization_barrier(radius).astype(jnp.float64)
        ok = _conservative_norm_leq(z64, r64, axis)
    return normal & ok


class SecondOrderCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "closed_form"

    # f = indicator of the second-order cone K, which is self dual, so
    # f^* = indicator of the polar -K.  The contract below is EXACT on the
    # represented data: -y is in K when ||y[1:]|| <= -y[0], with no slack.
    # ``value`` keeps its documented 1e-5 slack and is not consulted here.
    is_indicator = True
    bound_capable = True

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        s, z = v[0], v[1:]
        nz = jnp.linalg.norm(z)
        inside = nz <= s
        below = nz <= -s
        scale = (nz + s) / (2.0 * nz + 1e-30)
        proj_z = jnp.where(inside, z, jnp.where(below, 0.0 * z, scale * z))
        proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
        return jnp.where(jnp.all(jnp.isfinite(t)),
                         jnp.concatenate([jnp.reshape(proj_s, (1,)), proj_z]), jnp.nan)

    def value(self, z):
        return jnp.where(jnp.linalg.norm(z[1:]) <= z[0] + 1e-5, 0.0, jnp.inf)

    def conj_value(self, y):                       # polar = negative self-dual cone
        return jnp.where(self.conj_in_domain(y), 0., jnp.inf)

    def _project(self, v):
        s, z = v[0], v[1:]
        nz = jnp.linalg.norm(z)
        inside = nz <= s
        below = nz <= -s
        scale = (nz + s) / (2.0 * nz + 1e-30)
        proj_z = jnp.where(inside, z, jnp.where(below, 0.0 * z, scale * z))
        proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
        return jnp.concatenate([jnp.reshape(proj_s, (1,)), proj_z])

    def conj_in_domain(self, y):
        # Either sufficient test admits: the float32 envelope, or the same
        # test on the represented values in scoped float64 (since 0.3.8).
        return (_conservative_norm_leq(y[1:], -y[0])
                | _float64_norm_leq(y[1:], -y[0]))

    def conj_domain_radius(self, y):
        # dom f^* is a cone: every positive multiple of a member is a member,
        # and no positive multiple of a non-member is one.
        return jnp.where(self.conj_in_domain(y), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        # Moreau: y = P_K(y) + P_{-K}(y), so the projection onto the polar is
        # the residual of the Euclidean projection onto K.  That residual lies
        # ON the boundary ||p[1:]|| = -p[0], where the exact membership test is
        # decided by rounding and differs between backends.  The proposal is
        # therefore pulled a few ulps INSIDE the polar by shrinking the
        # spatial part; it is a proposal, re-checked exactly by the caller.
        # For float32 data the residual is formed in scoped float64 and the
        # pull is SOC_INSIDE_PULL_ULPS ulps at every dimension (since 0.3.8).
        # Through 0.3.7 it was formed in float32 and pulled (4 n + 16) eps32,
        # which lowers the dual bound by about (4 n + 16) eps32 s / |p*|
        # relative, with s the cone multiplier: 1.5e-4 at n = 512.
        if y.dtype == jnp.float32:
            with jax.enable_x64():
                y64 = jax.lax.optimization_barrier(y).astype(jnp.float64)
                p = (y64 - self._project(y64)).astype(jnp.float32)
            shrink = jnp.float32(1.0 - SOC_INSIDE_PULL_ULPS * float(jnp.finfo(jnp.float32).eps))
        else:
            p = y - self._project(y)
            shrink = jnp.maximum(0., 1.0 - (4 * (y.size-1) + 16) * jnp.finfo(y.dtype).eps)
        return jnp.concatenate([p[:1], p[1:] * shrink])

    def conj_domain_project_exact(self, y):
        # The Moreau residual itself, with no inside-pull.  A point already in
        # the polar is returned unchanged (its projection onto K is zero), so
        # repeating this projection does not move a valid candidate.  The
        # inside-pull above is applied once by the caller, on the final
        # candidate, under ``witness_contraction="once"``.
        return y - self._project(y)


# =====================================================================
# PSD cone  { X = X^T : X >= 0 }, X packed as a k*k row-major vector
# =====================================================================
class PSDCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "spectral"
    is_indicator = True
    bound_capable = True

    def __init__(self, k):
        self.k = int(k)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        k = self.k
        M = v.reshape(k, k)
        M = 0.5 * (M + M.T)
        _, U = sym_eigh(M)
        # Repair eigenpair roundoff before clipping. Small primal steps can
        # amplify GPU eigenvector norm and eigenvalue errors during PDHG.
        U = U / jnp.sqrt(jnp.sum(U * U, axis=0))
        MU = jnp.matmul(M, U, precision=jax.lax.Precision.HIGHEST)
        w = jnp.sum(U * MU, axis=0)
        # Reduced-precision GPU products make this projection non-idempotent
        # near the requested solver tolerance. Preserve float32 accuracy here.
        Mp = jnp.matmul(U * jnp.maximum(w, 0.0), U.T,
                        precision=jax.lax.Precision.HIGHEST)
        return jnp.where(jnp.all(jnp.isfinite(t)), Mp.reshape(-1), jnp.nan)

    def value(self, z):
        M = z.reshape(self.k, self.k)
        w = sym_eigvalsh(0.5 * (M + M.T))
        return jnp.where(jnp.min(w) >= -1e-4, 0.0, jnp.inf)

    def conj_value(self, y):                       # polar = negative self-dual cone
        return jnp.where(self.conj_in_domain(y), 0.0, jnp.inf)

    def conj_in_domain(self, y):
        # On the full matrix carrier, skew components annihilate every
        # symmetric primal matrix. Only sym(y) must be negative semidefinite.
        # This spectral test has no feasibility slack; it is still a numerical
        # evaluation, not a rigorous real-arithmetic eigenvalue enclosure.
        M = y.reshape(self.k, self.k)
        return jnp.all(jnp.isfinite(y)) & (jnp.max(sym_eigvalsh(0.5 * (M + M.T))) <= 0)

    #: The objective-bound evaluator decides this block's membership with
    #: ``conj_in_domain_resolved`` (see ``ocs.model.compile``).
    resolved_sign_domain = True

    def conj_in_domain_resolved(self, y):
        # Membership with a RESOLVED sign, for the objective-bound evaluator:
        # stricter than ``conj_in_domain``, never looser.  y is admitted only if
        # ``conj_in_domain`` admits it AND, on the represented float32 values,
        #     lambda_max(sym y) <= -PSD_RESOLVED_SIGN_C * k * eps32 * ||sym y||_2
        # with sym y and its eigenvalues computed in scoped float64 (sym of
        # float32 data is exact in float64).  A sign inside that envelope is
        # unresolved at the float32 scale on which the data and the coupling
        # were formed, so it is NOT admitted: there is no small-negative-
        # eigenvalue allowance in either direction.  The zero matrix (||sym y||
        # = 0) is admitted.  The barrier keeps XLA from folding the round trip
        # float32 -> float64 into a re-derivation of y at higher precision
        # (the 0.3.5 piecewise-linear pattern), so the tested numbers are
        # exactly the represented ones.  Still a numerical evaluation on
        # represented data, not a real-arithmetic enclosure of an arbitrary
        # float32 adjoint.
        k = self.k
        gamma = PSD_RESOLVED_SIGN_C * k * float(jnp.finfo(jnp.float32).eps)
        admitted = self.conj_in_domain(y)
        with jax.enable_x64():
            y32 = y.astype(jnp.float32)
            y64 = jax.lax.optimization_barrier(y32).astype(jnp.float64)
            M = y64.reshape(k, k)
            w = sym_eigvalsh(0.5 * (M + M.T))
            resolved = jnp.all(jnp.isfinite(w)) & (w[-1] <= -gamma * jnp.max(jnp.abs(w)))
        return admitted & resolved

    def conj_domain_radius(self, y):
        return jnp.where(self.conj_in_domain(y), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        M = y.reshape(self.k, self.k)
        sym = 0.5 * (M + M.T)
        w, U = sym_eigh(sym)
        negative = jnp.matmul(U * jnp.minimum(w, 0.0), U.T,
                              precision=jax.lax.Precision.HIGHEST)
        negative = 0.5 * (negative + negative.T)
        # An interior proposal avoids accepting eigenvector reconstruction
        # noise as a domain guarantee; the consumer rechecks membership.
        margin = 8 * self.k * jnp.finfo(y.dtype).eps * jnp.max(jnp.abs(w))
        negative -= margin * jnp.eye(self.k, dtype=y.dtype)
        return (negative + 0.5 * (M - M.T)).reshape(-1)

    def domain_distance(self, v):
        # ||P(v) - v||_F, the distance the projection moves v, for the
        # objective-bound evaluator's relative primal feasibility (since
        # 0.3.8).  With M = v as a k x k matrix, P(M) - M is the skew part
        # minus U min(lambda, 0) U^T of the symmetric part, two orthogonal
        # pieces, so the distance is
        #     sqrt(||skew M||_F^2 + ||min(lambda(sym M), 0)||_2^2).
        # It is read from eigenvalues in scoped float64 on the represented
        # float32 values, never from a reconstructed projection: in float32
        # the reconstruction alone moves a strictly interior k = 1024 matrix
        # by about 1e-5 relative.  Each eigenvalue is lowered by the envelope
        # 8 k eps64 max|lambda| before clipping, so the distance is not
        # understated, and the float32 result is rounded upward.
        k = self.k
        with jax.enable_x64():
            v64 = jax.lax.optimization_barrier(v.astype(jnp.float32)).astype(jnp.float64)
            M = v64.reshape(k, k)
            skew = 0.5 * (M - M.T)
            w = sym_eigvalsh(0.5 * (M + M.T))
            envelope = 8 * k * float(jnp.finfo(jnp.float64).eps) * jnp.max(jnp.abs(w))
            negative = jnp.minimum(w - envelope, 0.0)
            d64 = jnp.sqrt(jnp.sum(skew * skew) + jnp.sum(negative * negative))
            d32 = d64.astype(jnp.float32)
            d32 = jnp.where(d32.astype(jnp.float64) < d64,
                            jnp.nextafter(d32, jnp.float32(jnp.inf)), d32)
            d32 = jnp.where(jnp.all(jnp.isfinite(v64)), d32, jnp.float32(jnp.inf))
        return d32

    def group_key(self):
        return ("PSDCone", self.k)


# =====================================================================
# Exponential cone  cl{ (r,s,t) : s>0, t >= s exp(r/s) }
# =====================================================================
def _in_exp(r, s, t, tol=_TOL):
    s_safe = jnp.maximum(s, 1e-30)
    c1 = (s > 0) & (t - s * jnp.exp(r / s_safe) >= -tol * (1.0 + jnp.abs(t)))
    c2 = (r <= tol) & (jnp.abs(s) <= tol) & (t >= -tol)
    return c1 | c2


def proj_exp(v):
    """Project in normalized cone units, then restore the input magnitude.

    Cone projection is positively homogeneous. Normalizing makes the inner
    root find's numerical safeguards relative and prevents a tiny input from
    being declared feasible solely by its absolute magnitude.
    """
    v = jnp.asarray(v, jnp.float32)
    scale = jnp.max(jnp.abs(v))
    unit = v / jnp.where(scale > 0, scale, 1.0)
    projected = scale * _proj_exp_unit(unit)
    return jnp.where(scale == 0, jnp.zeros_like(v), projected)


def _in_exp_dual(u, s, w, tol=_TOL):
    # dual of {(r,s,t): s>0, t >= s e^{r/s}} is
    #   {(u,v,w): u<0, -u e^{v/u} <= e w} U {(0,v,w): v>=0, w>=0}
    u_safe = jnp.minimum(u, -1e-30)
    c1 = (u < 0) & (-u * jnp.exp(s / u_safe) <= jnp.e * w + tol * (1.0 + jnp.abs(w)))
    c2 = (jnp.abs(u) <= tol) & (s >= -tol) & (w >= -tol)
    return c1 | c2


class ExpCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "newton"

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(jnp.all(jnp.isfinite(t)), proj_exp(v), jnp.nan)

    def value(self, z):                            # primal cone indicator
        return jnp.where(_in_exp(z[0], z[1], z[2]), 0.0, jnp.inf)

    def conj_value(self, y):                       # indicator conjugate is the polar cone
        return jnp.where(_in_exp_dual(-y[0], -y[1], -y[2]), 0.0, jnp.inf)


# =====================================================================
# Power cone  K_pow^alpha = { x,y>=0 : x^a y^{1-a} >= |z| }
# =====================================================================
def _in_pow(x, y, z, a, tol=_TOL):
    xs = jnp.maximum(x, 0.0); ys = jnp.maximum(y, 0.0)
    lhs = (xs ** a) * (ys ** (1.0 - a))
    return (x >= -tol) & (y >= -tol) & (lhs - jnp.abs(z) >= -tol * (1.0 + jnp.abs(z)))


def _in_pow_dual(u, v, w, a, tol=_TOL):
    # dual of {(x,y,z): x,y>=0, x^a y^{1-a} >= |z|} is
    #   {(u,v,w): u,v>=0, (u/a)^a (v/(1-a))^{1-a} >= |w|}
    us = jnp.maximum(u, 0.0); vs = jnp.maximum(v, 0.0)
    lhs = ((us / a) ** a) * ((vs / (1.0 - a)) ** (1.0 - a))
    return (u >= -tol) & (v >= -tol) & (lhs - jnp.abs(w) >= -tol * (1.0 + jnp.abs(w)))


class PowerCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "bisection"

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(jnp.all(jnp.isfinite(t)), proj_pow(v, self.alpha), jnp.nan)

    def value(self, z):                            # primal cone indicator
        return jnp.where(_in_pow(z[0], z[1], z[2], self.alpha), 0.0, jnp.inf)

    def conj_value(self, y):                       # indicator conjugate is the polar cone
        return jnp.where(_in_pow_dual(-y[0], -y[1], -y[2], self.alpha), 0.0, jnp.inf)

    def group_key(self):
        return ("PowerCone", self.alpha)


# =====================================================================
# Batched cones in CVXPY's stacked layout (for the cvxpy_mf bridge)
# =====================================================================
class PowerConeBatch(ProxBlock):
    """q power cones stacked CVXPY-style: u = [x (q); y (q); z (q)],
    triple j = (x_j, y_j, z_j) with x_j^a y_j^(1-a) >= |z_j|.
    Projection vmapped over triples (one bracketed scalar solve per triple)."""
    scaling_class = SCALAR
    eval_kind = "bisection"

    def __init__(self, q, alpha=0.5):
        self.q = int(q); self.alpha = float(alpha)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        q = self.q
        triples = jnp.stack([v[:q], v[q:2 * q], v[2 * q:]], axis=1)   # (q, 3)
        proj = jax.vmap(lambda w: proj_pow(w, self.alpha))(triples)
        return jnp.where(jnp.all(jnp.isfinite(t)),
                         jnp.concatenate([proj[:, 0], proj[:, 1], proj[:, 2]]), jnp.nan)

    def value(self, z):
        q = self.q
        ok = jnp.all(_in_pow(z[:q], z[q:2 * q], z[2 * q:], self.alpha))
        return jnp.where(ok, 0.0, jnp.inf)

    def conj_value(self, y):
        q = self.q
        ok = jnp.all(_in_pow_dual(-y[:q], -y[q:2 * q], -y[2 * q:], self.alpha))
        return jnp.where(ok, 0.0, jnp.inf)

    def group_key(self):
        return ("PowerConeBatch", self.q, self.alpha)


class SOCBatch(ProxBlock):
    """q second-order cones stacked CVXPY-style: u = [t (q); vec_F(Z) (d*q)],
    cone j = (t_j, Z[:, j]).  One vectorized projection over all q cones."""
    scaling_class = SCALAR
    eval_kind = "closed_form"
    # Same exact polar contract as SecondOrderCone, per cone: -y_j in K_j when
    # ||Z_j|| <= -t_j for every j, no slack.  ``value`` keeps its 1e-5 slack.
    is_indicator = True
    bound_capable = True

    def __init__(self, q, d):
        self.q, self.d = int(q), int(d)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        q, d = self.q, self.d
        s = v[:q]
        Z = jnp.reshape(v[q:], (d, q), order="F")
        nz = jnp.linalg.norm(Z, axis=0)
        inside = nz <= s
        below = nz <= -s
        scale = (nz + s) / (2.0 * nz + 1e-30)
        col = jnp.where(inside, 1.0, jnp.where(below, 0.0, scale))
        proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
        proj_Z = Z * col[None, :]
        return jnp.where(jnp.all(jnp.isfinite(t)),
                         jnp.concatenate([proj_s, jnp.reshape(proj_Z, (-1,), order="F")]), jnp.nan)

    def value(self, z):
        q, d = self.q, self.d
        nz = jnp.linalg.norm(jnp.reshape(z[q:], (d, q), order="F"), axis=0)
        ok = jnp.max(nz - z[:q]) <= 1e-5
        return jnp.where(ok, 0.0, jnp.inf)

    def conj_value(self, y):                       # polar = negative self-dual cone
        return jnp.where(self.conj_in_domain(y), 0., jnp.inf)

    def _project(self, v):
        q, d = self.q, self.d
        s = v[:q]
        Z = jnp.reshape(v[q:], (d, q), order="F")
        nz = jnp.linalg.norm(Z, axis=0)
        inside = nz <= s
        below = nz <= -s
        scale = (nz + s) / (2.0 * nz + 1e-30)
        col = jnp.where(inside, 1.0, jnp.where(below, 0.0, scale))
        proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
        return jnp.concatenate([proj_s, jnp.reshape(Z * col[None, :], (-1,), order="F")])

    def conj_in_domain(self, y):
        q, d = self.q, self.d
        z = jnp.reshape(y[q:], (d, q), order="F")
        return jnp.all(_conservative_norm_leq(z, -y[:q], axis=0))

    def conj_domain_radius(self, y):
        return jnp.where(self.conj_in_domain(y), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        # The same boundary argument as SecondOrderCone: pull each cone a few
        # ulps inside the polar so the exact re-check is not a rounding coin flip.
        p = y - self._project(y)
        shrink = jnp.maximum(0., 1.0 - (4 * self.d + 16) * jnp.finfo(y.dtype).eps)
        return jnp.concatenate([p[:self.q], p[self.q:] * shrink])

    def conj_domain_project_exact(self, y):
        # As for SecondOrderCone: the Moreau residual with no inside-pull.
        return y - self._project(y)

    def group_key(self):
        return ("SOCBatch", self.q, self.d)


class ExpConeBatch(ProxBlock):
    """q exponential cones stacked CVXPY-style: u = [x (q); y (q); z (q)],
    triple j = (x_j, y_j, z_j).  Projection vmapped over triples."""
    scaling_class = SCALAR
    eval_kind = "newton"

    def __init__(self, q):
        self.q = int(q)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        q = self.q
        triples = jnp.stack([v[:q], v[q:2 * q], v[2 * q:]], axis=1)   # (q, 3)
        proj = jax.vmap(proj_exp)(triples)
        return jnp.where(jnp.all(jnp.isfinite(t)),
                         jnp.concatenate([proj[:, 0], proj[:, 1], proj[:, 2]]), jnp.nan)

    def value(self, z):
        q = self.q
        ok = jnp.all(_in_exp(z[:q], z[q:2 * q], z[2 * q:]))
        return jnp.where(ok, 0.0, jnp.inf)

    def conj_value(self, y):
        q = self.q
        ok = jnp.all(_in_exp_dual(-y[:q], -y[q:2 * q], -y[2 * q:]))
        return jnp.where(ok, 0.0, jnp.inf)

    def group_key(self):
        return ("ExpConeBatch", self.q)
