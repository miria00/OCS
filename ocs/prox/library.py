"""Starter prox blocks.  Each is GPU-native (pure JAX), declares its scaling
class and eval kind, and exposes value / conj_value for the duality gap.  The
full cone vocabulary (SOC, PSD, exp, pow) lives in cones.py.

  SumSquares  1/2||u||^2         prox = v/(1+t)          diagonal, closed_form
  L1          lam||u||_1         prox = soft(v, t lam)   diagonal, closed_form
  L2          lam||u||_2         block shrinkage         scalar,   closed_form
  NonNeg      indicator{u>=0}    prox = max(v,0)         diagonal, closed_form
  Box         indicator{lo<=u<=hi} prox = clip(v,lo,hi)  diagonal, closed_form
  Zero        0 (free)           prox = v                matrix,   closed_form
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .base import (ProxBlock, SCALAR, DIAGONAL, MATRIX,
                   _validate_step, _validate_matrix_metric, _finite_step)

_INF = jnp.inf


#: Relative shrink applied to the conjugate-domain SEARCH HINTS (never to the
#: exact domain TEST).  Scaling by lam/||y||_2 in float32 can leave a vector
#: whose recomputed norm rounds a few ulps above lam, which the exact test then
#: rejects.  A hint that lands strictly inside costs at most this much of the
#: bound and can never make an invalid bound valid.
_DOMAIN_HINT_SHRINK = 1.0 - 1e-6

#: Narrowest slope pair the piecewise-linear conjugate interpolates across.  A
#: narrower pair is skipped; the wider pairs that remain are still exact upper
#: bounds, so skipping can only raise the value, and no float32 flush of a
#: nearly equal slope pair can enter a division.
_PWL_MIN_SLOPE_GAP = float(2.0 ** 30 * np.finfo(np.float32).tiny)


def _has_subnormal(v):
    """Whether a float32 array holds a nonzero subnormal, decided on its bits."""
    v = jnp.asarray(v)
    if v.dtype != jnp.float32:
        return jnp.asarray(False)
    bits = jax.lax.bitcast_convert_type(v, jnp.uint32) & 0x7fffffff
    return jnp.any((bits != 0) & (bits < 0x00800000))


def _order_key(v):
    """An integer image of a float array that orders exactly as its values do,
    decided on the bits.  A floating-point comparison can read a subnormal
    operand as zero: XLA:CPU flushes them, and not uniformly between eager and
    jitted code (measured 2026-10-23: with s_min = 0, y = -1.4e-45 passed
    ``y >= s_min`` eagerly and failed it under jit).  Integer comparisons of
    these keys are exact on every backend.  -0 and +0 share the key 0.  NaN has
    no place in the order; callers exclude it."""
    v = jnp.asarray(v)
    itype = {2: jnp.int16, 4: jnp.int32, 8: jnp.int64}[v.dtype.itemsize]
    bits = jax.lax.bitcast_convert_type(v, itype)
    magnitude = bits & jnp.asarray(np.iinfo(itype).max, itype)
    return jnp.where(bits < 0, -magnitude, magnitude)


def _nonnegative_parameter(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _quadratic_conj_value(P, q, y):
    """Conjugate of a numerical PSD quadratic, including its range condition.

    Positive eigenvalues are retained, including small positive curvature.
    Nonpositive eigenvalues within the eigensolver's relative roundoff scale
    are treated as null directions. Range checks have no absolute unit floor.
    This is a floating-point value oracle, not an exact rank certificate.
    """
    w = y-q
    eigenvalues, basis = jnp.linalg.eigh(P)
    coordinates = jnp.matmul(basis.T, w, precision=jax.lax.Precision.HIGHEST)
    eps = jnp.finfo(P.dtype).eps
    spectral_scale = jnp.max(jnp.abs(eigenvalues))
    curvature_roundoff = 16 * P.shape[0] * eps * spectral_scale
    positive = eigenvalues > 0
    residual = jnp.max(jnp.abs(jnp.where(positive, 0, coordinates)))
    data_scale = jnp.maximum(jnp.max(jnp.abs(w)),
                            jnp.maximum(jnp.max(jnp.abs(y)), jnp.max(jnp.abs(q))))
    in_range = residual <= 16 * P.shape[0] * eps * data_scale
    value = .5 * jnp.sum(jnp.where(positive, coordinates**2 /
                                  jnp.where(positive, eigenvalues, 1), 0))
    valid = (jnp.all(jnp.isfinite(P)) & jnp.all(jnp.isfinite(w))
             & (jnp.min(eigenvalues) >= -curvature_roundoff))
    return jnp.where(valid, jnp.where(in_range, value, jnp.inf), jnp.nan)


class SumSquares(ProxBlock):
    """f(u) = 1/2 ||u||_2^2 (self-conjugate).  Quadratic -> admits a full matrix
    metric: prox_f^D(v) = (I + D)^{-1} D v."""
    scaling_class = MATRIX
    coordinate_separable = True
    smooth_grad = staticmethod(lambda s: s)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return v / (1.0 + t)                       # scalar / diagonal metric

    def prox_matrix(self, v, D):
        D = _validate_matrix_metric(v, D)
        k = v.shape[0]
        rhs = jnp.matmul(D, v, precision=jax.lax.Precision.HIGHEST)
        return jnp.linalg.solve(jnp.eye(k, dtype=v.dtype) + D, rhs)

    def value(self, z):
        return 0.5 * jnp.sum(z * z)

    def conj_value(self, y):
        return 0.5 * jnp.sum(y * y)

    # Objective lower bound: dom f^* = R^m and the conjugate value is exact.
    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y))

    def conj_domain_radius(self, y):
        return jnp.where(jnp.all(jnp.isfinite(y)), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        return y


class Quadratic(ProxBlock):
    """f(u) = 1/2 u^T P u + q^T u,  P symmetric PSD (dense).  The archetype of the
    MATRIX scaling tier: for any PD metric D,
        prox_f^D(v) = (P + D)^{-1}(D v - q),
    a single dense linear solve.  The scalar/diagonal prox is the same with
    D = (1/t) I / diag(1/t_i).  (Used as a preconditioned quadratic block; see
    examples/matrix_metric_demo.py.)"""
    scaling_class = MATRIX
    eval_kind = "linear_solve"

    def __init__(self, P, q=None):
        matrix = np.asarray(P, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
            raise ValueError("P must be a nonempty square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("P must be finite")
        scale = float(np.max(np.abs(matrix)))
        tolerance = 16 * matrix.shape[0] * np.finfo(np.float32).eps * scale
        if np.max(np.abs(matrix-matrix.T)) > tolerance:
            raise ValueError("P must be symmetric")
        matrix = .5*matrix + .5*matrix.T
        # Validate the actual float32 matrix in higher precision. A float32
        # global tolerance would accept real negative curvature (e.g. -1e-6
        # beside a unit eigenvalue), making large-step proximal problems
        # nonconvex. Double precision is used only for constructor validation.
        psd_tolerance = 16 * matrix.shape[0] * np.finfo(np.float64).eps * scale
        smallest = float(np.linalg.eigvalsh(matrix.astype(np.float64)).min())
        if np.any(np.diag(matrix) < 0) or smallest < -psd_tolerance:
            raise ValueError("P must be positive semidefinite")
        # Objective-lower-bound capability.  dom f^* is all of R^n exactly when
        # P is POSITIVE DEFINITE.  When P is singular the conjugate is finite
        # only on the affine set q + range(P), and deciding membership of a
        # range exactly is not something this block can do in float32, so it
        # declares no contract at all rather than a tolerant one.
        self.bound_capable = bool(smallest > psd_tolerance)
        self.P = jnp.asarray(matrix)
        self.dim = int(self.P.shape[0])
        self.q = (jnp.zeros(self.dim, jnp.float32) if q is None
                  else jnp.asarray(q, jnp.float32))
        if self.q.shape != (self.dim,) or not np.all(np.isfinite(np.asarray(self.q))):
            raise ValueError("q must be a finite vector matching P")
        self.coordinate_separable = bool(np.all(matrix == np.diag(np.diag(matrix))))
        self.smooth_grad = lambda s: jnp.matmul(
            self.P, s, precision=jax.lax.Precision.HIGHEST) + self.q

    def prox(self, v, t):
        if v.shape != (self.dim,):
            raise ValueError("point must match the quadratic dimension")
        t = _validate_step(v, t, self.scaling_class)
        d = jnp.broadcast_to(1.0 / t, (self.dim,))
        return jnp.linalg.solve(self.P + jnp.diag(d), d * v - self.q)

    def prox_matrix(self, v, D):
        if v.shape != (self.dim,):
            raise ValueError("point must match the quadratic dimension")
        D = _validate_matrix_metric(v, D)
        rhs = jnp.matmul(D, v, precision=jax.lax.Precision.HIGHEST) - self.q
        return jnp.linalg.solve(self.P + D, rhs)

    def value(self, z):
        pz = jnp.matmul(self.P, z, precision=jax.lax.Precision.HIGHEST)
        return (0.5 * jnp.dot(z, pz, precision=jax.lax.Precision.HIGHEST)
                + jnp.dot(self.q, z, precision=jax.lax.Precision.HIGHEST))

    def conj_value(self, y):
        return _quadratic_conj_value(self.P, self.q, y)

    def conj_in_domain(self, y):
        # Reached only when P is positive definite, where dom f^* = R^n and
        # finiteness is the whole test.  Guarded anyway, so that a singular P
        # can never be certified through this path.
        if not self.bound_capable:
            return jnp.asarray(False)
        return jnp.all(jnp.isfinite(y))

    def conj_domain_radius(self, y):
        if not self.bound_capable:
            return jnp.asarray(0.0, jnp.float32)
        return jnp.where(jnp.all(jnp.isfinite(y)), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        return y


class L1(ProxBlock):
    """f(u) = lam ||u||_1;  f^* = indicator{ ||y||_inf <= lam }."""
    scaling_class = DIAGONAL
    coordinate_separable = True

    def __init__(self, lam=1.0):
        self.lam = _nonnegative_parameter(lam, "lam")

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t * self.lam, 0.0)

    def value(self, z):
        return self.lam * jnp.sum(jnp.abs(z))

    def conj_value(self, y):
        # EXACT indicator.  The former 1e-6 ABSOLUTE slack accepted a dual
        # outside dom f^* at value 0 (0.1% of slack at lam = 1e-3), which is
        # what let the reported dual value exceed the primal value at the same
        # point.  A conjugate indicator carries no tolerance.
        return jnp.where(jnp.max(jnp.abs(y)) <= self.lam, 0.0, _INF)

    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y)) & (jnp.max(jnp.abs(y)) <= self.lam)

    def conj_domain_radius(self, y):
        peak = jnp.max(jnp.abs(y))
        finite = jnp.all(jnp.isfinite(y))
        radius = jnp.where(peak > 0, self.lam / jnp.where(peak > 0, peak, 1.0), jnp.inf)
        return jnp.where(finite, radius, 0.0)

    def conj_domain_project(self, y):
        return jnp.clip(y, -self.lam, self.lam)

    def group_key(self):
        return ("L1", self.lam)


class L2(ProxBlock):
    """f(u) = lam ||u||_2 (not squared);  f^* = indicator{ ||y||_2 <= lam }."""
    scaling_class = SCALAR

    def __init__(self, lam=1.0):
        self.lam = _nonnegative_parameter(lam, "lam")

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        nrm = jnp.linalg.norm(v)
        return jnp.maximum(0.0, 1.0 - t * self.lam / (nrm + 1e-30)) * v

    def value(self, z):
        return self.lam * jnp.linalg.norm(z)

    def conj_value(self, y):
        # EXACT indicator.  The former 1e-6 ABSOLUTE slack accepted a dual
        # outside dom f^* at value 0, which is what lets a reported dual value
        # sit ABOVE the primal value at the same point.  A conjugate indicator
        # carries no tolerance.
        return jnp.where(jnp.linalg.norm(y) <= self.lam, 0.0, _INF)

    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y)) & (jnp.linalg.norm(y) <= self.lam)

    def conj_domain_radius(self, y):
        nrm = jnp.linalg.norm(y)
        finite = jnp.all(jnp.isfinite(y))
        radius = jnp.where(nrm > 0,
                           _DOMAIN_HINT_SHRINK * self.lam
                           / jnp.where(nrm > 0, nrm, 1.0), jnp.inf)
        return jnp.where(finite, radius, 0.0)

    def conj_domain_project(self, y):
        nrm = jnp.linalg.norm(y)
        scale = jnp.where(nrm > self.lam,
                          _DOMAIN_HINT_SHRINK * self.lam
                          / jnp.where(nrm > 0, nrm, 1.0), 1.0)
        return scale * y

    def group_key(self):
        return ("L2", self.lam)


class GroupL2(ProxBlock):
    """f(u) = lam sum_g ||u_g||_2 over contiguous groups of size `group_size`
    (isotropic TV when group_size=2 on a 2-D gradient; row-group lasso for
    feature selection).  prox = per-group block shrinkage; one block evaluates
    all groups via a single reshape+vmap."""
    scaling_class = SCALAR

    def __init__(self, lam=1.0, group_size=2):
        self.lam = _nonnegative_parameter(lam, "lam")
        self.g = int(group_size)
        if self.g <= 0 or self.g != group_size:
            raise ValueError("group_size must be a positive integer")

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        ts = jnp.mean(t)                                 # coupled block -> uniform step
        R = v.reshape(-1, self.g)
        nrm = jnp.linalg.norm(R, axis=1, keepdims=True)
        R = jnp.maximum(0.0, 1.0 - ts * self.lam / (nrm + 1e-30)) * R
        return R.reshape(-1)

    def value(self, z):
        return self.lam * jnp.sum(jnp.linalg.norm(z.reshape(-1, self.g), axis=1))

    def conj_value(self, y):
        # EXACT indicator; see the note on L1 and L2.  No tolerance.
        return jnp.where(self._peak(y) <= self.lam, 0.0, _INF)

    bound_capable = True

    def _peak(self, y):
        """max_g ||y_g||_2, the quantity dom f^* is defined by."""
        return jnp.max(jnp.linalg.norm(y.reshape(-1, self.g), axis=1))

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y)) & (self._peak(y) <= self.lam)

    def conj_domain_radius(self, y):
        mx = self._peak(y)
        finite = jnp.all(jnp.isfinite(y))
        radius = jnp.where(mx > 0,
                           _DOMAIN_HINT_SHRINK * self.lam
                           / jnp.where(mx > 0, mx, 1.0), jnp.inf)
        return jnp.where(finite, radius, 0.0)

    def conj_domain_project(self, y):
        R = y.reshape(-1, self.g)
        nrm = jnp.linalg.norm(R, axis=1, keepdims=True)
        scale = jnp.where(nrm > self.lam,
                          _DOMAIN_HINT_SHRINK * self.lam
                          / jnp.where(nrm > 0, nrm, 1.0), 1.0)
        return (scale * R).reshape(-1)

    def group_key(self):
        return ("GroupL2", self.lam, self.g)


class PiecewiseLinearConvex(ProxBlock):
    """Separable convex piecewise-linear  f(u) = sum_i max_j (s_j u_i + r_j)  (a max
    of P affine pieces per coordinate).  Models PWL convex costs / the epigraph of
    concave PWL utilities (network utility maximization, quantile-type losses).
    The prox has no closed form for general pieces, but the optimality condition
    u = v - t*g'(u) is monotone (g' = active slope, nondecreasing), so a short
    bisection solves it robustly and vectorizes over all coordinates.

    Its conjugate is  f^*(y) = sum_i g^*(y_i)  with  dom g^* = [s_min, s_max]:
    on that interval g^* is the lower convex envelope of the points (s_j, -r_j)
    evaluated at y, and it is +inf off it (``conj_value`` below)."""
    scaling_class = DIAGONAL
    coordinate_separable = True
    eval_kind = "bisection"

    def __init__(self, slopes, intercepts):
        self.s = jnp.asarray(slopes, jnp.float32)              # (P,)
        self.r = jnp.asarray(intercepts, jnp.float32)          # (P,)
        if (self.s.ndim != 1 or not self.s.size or self.r.shape != self.s.shape
                or not np.all(np.isfinite(np.asarray(self.s)))
                or not np.all(np.isfinite(np.asarray(self.r)))):
            raise ValueError("slopes and intercepts must be nonempty finite vectors of equal shape")

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        s, r = self.s, self.r
        smin, smax = jnp.min(s), jnp.max(s)
        lo = v - t * smax; hi = v - t * smin                   # u = v - t*slope(u) in [.,.]
        for _ in range(40):
            u = 0.5 * (lo + hi)
            slope = s[jnp.argmax(s[None, :] * u[:, None] + r[None, :], axis=1)]
            resid = u + t * slope - v                          # nondecreasing in u
            hi = jnp.where(resid > 0.0, u, hi)
            lo = jnp.where(resid > 0.0, lo, u)
        return 0.5 * (lo + hi)

    def value(self, z):
        return jnp.sum(jnp.max(self.s[None, :] * z[:, None] + self.r[None, :], axis=1))

    # ---- objective lower bound: the exact conjugate-domain contract -------
    # Per coordinate, g(u) = max_j (s_j u + r_j) has the conjugate
    #     g^*(y) = sup_u (y u - g(u))
    #            = min { -sum_j lam_j r_j : lam >= 0, sum_j lam_j = 1, sum_j lam_j s_j = y },
    # the lower convex envelope of the points (s_j, -r_j), on the closed
    # interval dom g^* = [s_min, s_max], and +inf off it.  In one dimension two
    # points suffice, so on the interval g^* is the smallest linear
    # interpolation between (s_j, -r_j) and (s_k, -r_k) over the pairs
    # s_j <= y <= s_k, s_j < s_k, together with -r_j wherever y = s_j.  This
    # covers P = 1 (the single point {s}), repeated slopes (the largest
    # intercept wins), unsorted pieces, pieces that are never active (their
    # points lie above the envelope), and the endpoints y = s_min and y = s_max,
    # where the value is -max{r_j : s_j = s_min} (resp. s_max), the limit of
    # y u - g(u) as u -> -inf (resp. +inf).
    #
    # Rounding direction.  Every candidate is an exact upper bound on g^*(y)
    # (weak duality for a convex combination), so the minimum over candidates
    # can only be too high, never too low, in exact arithmetic.  The
    # evaluation is the scoped float64 assessment of the represented float32
    # data; each candidate carries its float64 rounding envelope, the sum
    # carries an order-independent summation envelope, and the total is
    # rounded up to the argument's dtype.  A value that is exact stays exact
    # (one coordinate at a slope, or an envelope that is zero).  The reported conjugate value is
    # therefore never below the conjugate of the represented data, and the dual
    # value built from it is never overstated by this block.  (Evaluating
    # y u - g(u) at the float32-rounded kinks, the other exact formula, rounds
    # the other way: a concave function at a rounded maximizer lies below its
    # maximum.)  No global X64 setting changes.
    bound_capable = True

    def _slope_keys(self, dtype):
        # The slopes and their order keys in the argument's dtype.  Widening a
        # normal float32 is exact; only a subnormal slope can be flushed on
        # the way, a case conj_in_domain refuses conservatively.
        s = self.s.astype(dtype)
        return s, _order_key(s)

    def conj_in_domain(self, y):
        # EXACT closed-interval test on the represented data; no slack.  The
        # order is decided on the bits (``_order_key``), so a backend that
        # flushes subnormal operands cannot admit a point outside
        # [s_min, s_max], such as y = -1.4e-45 with s_min = 0, nor refuse one
        # inside it.  The endpoints are the extreme keys, not a float min/max.
        y = jnp.asarray(y)
        dtype = jnp.result_type(y.dtype, jnp.float32)
        y = y.astype(dtype)
        _, keys = self._slope_keys(dtype)
        ky = _order_key(y)
        inside = jnp.isfinite(y) & (ky >= jnp.min(keys)) & (ky <= jnp.max(keys))
        if dtype != jnp.float32:
            # A float32 subnormal slope may reach a wider dtype flushed to zero,
            # and then the order of a y below float32's smallest normal against
            # it is not decided: such a y is refused (conservative).
            inside &= ~(_has_subnormal(self.s)
                        & (jnp.abs(y) < float(np.finfo(np.float32).tiny)))
        return jnp.all(inside)

    def conj_value(self, y):
        y = jnp.asarray(y)
        dtype = jnp.result_type(y.dtype, jnp.float32)
        y = y.astype(dtype).reshape(-1)
        inside = self.conj_in_domain(y)
        pieces = int(self.s.shape[0])
        count = int(y.shape[0])
        unit = float(np.finfo(np.float64).eps) / 2.0           # float64 unit roundoff
        tiny = float(np.finfo(np.float32).tiny)                # float32 flush scale
        # A represented float32 subnormal may be flushed to zero on its way in;
        # only then is an absolute allowance needed.  Decided on the bits.
        flushed = _has_subnormal(y) | _has_subnormal(self.s) | _has_subnormal(self.r)
        # Which candidate applies at y is an ORDER question, decided on the
        # bits like the domain test, so a flushed operand cannot select an
        # interval that does not contain y or the point candidate of a slope
        # y only rounds to.  The float64 arithmetic below is unchanged.
        _, KS = self._slope_keys(dtype)
        KY = _order_key(y)
        with jax.enable_x64():
            f64 = jnp.float64
            S, R, Y = self.s.astype(f64), self.r.astype(f64), y.astype(f64)

            def candidate(index, best):
                j, k = index // pieces, index % pieces
                sj, sk, rj, rk = S[j], S[k], R[j], R[k]
                width = sk - sj
                usable = width >= _PWL_MIN_SLOPE_GAP
                width = jnp.where(usable, width, 1.0)
                rise = rk - rj
                part = ((Y - sj) / width) * rise
                # The seven float64 roundings of -(r_j + part) + envelope cost
                # at most 7 unit (|r_j| + |part|); with flushed inputs, add their
                # absolute error times the sensitivities 1, 1, |slope| (three
                # times).  No float64 underflow occurs for normal float32 data.
                envelope = 8.0 * unit * (jnp.abs(rj) + jnp.abs(part))
                envelope = envelope + jnp.where(
                    flushed, 8.0 * tiny * (2.0 + 3.0 * (jnp.abs(rise) / width)), 0.0)
                interpolated = jnp.where(usable & (KY >= KS[j]) & (KY <= KS[k]),
                                         -(rj + part) + envelope, jnp.inf)
                # The point candidate is exact unless r_j itself was flushed
                # (then off by less than tiny): with a flushed input it carries
                # that absolute allowance, and without one it stays exact.
                single = jnp.where((j == k) & (KY == KS[j]),
                                   -rj + jnp.where(flushed, 8.0 * tiny, 0.0), jnp.inf)
                return jnp.minimum(best, jnp.minimum(interpolated, single))

            values = jax.lax.fori_loop(0, pieces * pieces, candidate,
                                       jnp.full(Y.shape, jnp.inf, f64))
            total = jnp.sum(values)
            if count > 1:
                # Any summation order errs by at most (count - 1) unit sum|v|;
                # this allowance also covers its own computation and addition.
                total = total + 4.0 * (count + 1) * unit * jnp.sum(jnp.abs(values))
            rounded = total.astype(dtype)
            # The barrier keeps XLA's excess-precision simplification from
            # folding the round trip float64 -> dtype -> float64 into the
            # identity under jit, which would skip the upward step.
            back = jax.lax.optimization_barrier(rounded).astype(f64)
            rounded = jnp.where(back < total,
                                jnp.nextafter(rounded, jnp.asarray(jnp.inf, dtype)), rounded)
        return jnp.where(inside, rounded, jnp.asarray(jnp.inf, dtype))

    def conj_domain_radius(self, y):
        # sup { a >= 0 : a y in [s_min, s_max]^n }, the base-class semantics.
        # The interval need not contain 0: when s_min > 0 or s_max < 0 the
        # admissible scalings form an interval [a_lo, a_hi] that excludes 0, so
        # scaling y toward the origin LEAVES the domain.  The value is then a
        # search upper bound, not a claim that [0, a_hi] is admissible (the
        # translated-domain convention of ``with_linear_quadratic``); when no
        # a >= 0 is admissible (a coordinate of the wrong sign, a zero
        # coordinate with 0 outside the interval, or a_lo > a_hi) it is 0.
        # Every trial point is re-checked by ``conj_in_domain``.
        smin, smax = jnp.min(self.s), jnp.max(self.s)
        finite = jnp.all(jnp.isfinite(y))
        positive, negative = y > 0, y < 0
        safe = jnp.where(positive | negative, y, 1.0)
        upper = jnp.where(positive, smax / safe, jnp.where(negative, smin / safe, jnp.inf))
        lower = jnp.where(positive, smin / safe, jnp.where(negative, smax / safe, -jnp.inf))
        a_hi = jnp.min(upper, initial=jnp.inf)
        a_lo = jnp.maximum(jnp.max(lower, initial=-jnp.inf), 0.0)
        zero_ok = jnp.all(positive | negative | ((smin <= 0) & (smax >= 0)))
        admissible = finite & zero_ok & (a_lo <= a_hi)
        return jnp.where(admissible, a_hi, 0.0)

    def conj_domain_project(self, y):
        # The domain is closed and the test is non-strict, so the clip lands in
        # it exactly; no interior adjustment is needed.  Where the backend
        # flushes a subnormal operand the clip can land one subnormal outside
        # (measured on XLA:CPU under jit: -0 for s_min = 1.4e-45), so the
        # result is re-decided on the bits and any coordinate still outside
        # takes the endpoint itself.  On a backend that compares exactly that
        # correction never fires and the result is the clip, bit for bit.
        p = jnp.clip(y, jnp.min(self.s), jnp.max(self.s))
        s, keys = self._slope_keys(p.dtype)
        lo, hi = jnp.argmin(keys), jnp.argmax(keys)
        kp = _order_key(p)
        return jnp.where(jnp.isnan(p), p,
                         jnp.where(kp < keys[lo], s[lo], jnp.where(kp > keys[hi], s[hi], p)))

    def group_key(self):
        return ("PiecewiseLinearConvex", self.s.shape[0], float(self.s.sum()), float(self.r.sum()))


class NonNeg(ProxBlock):
    """f = indicator{u >= 0};  f^* = indicator{y <= 0} (polar)."""
    scaling_class = DIAGONAL
    coordinate_separable = True

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(_finite_step(t), jnp.maximum(v, 0.0), jnp.nan)

    def value(self, z):
        return jnp.where(jnp.min(z) >= -1e-6, 0.0, _INF)

    def conj_value(self, y):
        return jnp.where(jnp.max(y) <= 1e-6, 0.0, _INF)

    # The tolerance above is a diagnostic oracle.  The bound capability below
    # is exact and carries no slack; inside dom f^* the conjugate value is 0.
    # An indicator: value is 0 on the set and +inf off it, and prox is
    # the projection.  Feasibility is measured separately from the gap.
    is_indicator = True
    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y)) & (jnp.max(y) <= 0.0)

    def conj_domain_radius(self, y):
        # dom f^* is a cone: a positive scaling never leaves or enters it.
        return jnp.where(self.conj_in_domain(y), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        return jnp.minimum(y, 0.0)


class Box(ProxBlock):
    """f = indicator{lo <= u <= hi};  f^*(y) = sum max(lo y, hi y) (support fn)."""
    scaling_class = DIAGONAL
    coordinate_separable = True

    def __init__(self, lo=0.0, hi=1.0):
        with np.errstate(over="ignore", invalid="ignore"):
            lo = np.asarray(lo, dtype=np.float32)
            hi = np.asarray(hi, dtype=np.float32)
        low, high = np.broadcast_arrays(lo, hi)
        if (np.any(np.isnan(low)) or np.any(np.isnan(high))
                or np.any(low > high) or np.any(low == np.inf) or np.any(high == -np.inf)):
            raise ValueError("box bounds must describe a nonempty interval in each coordinate")
        self.lo, self.hi = jnp.asarray(lo), jnp.asarray(hi)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(_finite_step(t), jnp.clip(v, self.lo, self.hi), jnp.nan)

    def value(self, z):
        ok = (jnp.min(z - self.lo) >= -1e-6) & (jnp.max(z - self.hi) <= 1e-6)
        return jnp.where(ok, 0.0, _INF)

    def conj_value(self, y):
        # Choose the endpoint by the sign, avoiding 0*infinity on unbounded
        # coordinates. At a zero multiplier the support contribution is zero.
        endpoint = jnp.where(y > 0, self.hi, jnp.where(y < 0, self.lo, 0.0))
        return jnp.sum(endpoint * y)

    # The support function is exact everywhere it is finite; dom f^* is the set
    # of y whose support value is finite (all of R^m for finite bounds).
    # An indicator: value is 0 on the set and +inf off it, and prox is
    # the projection.  Feasibility is measured separately from the gap.
    is_indicator = True
    bound_capable = True

    def conj_in_domain(self, y):
        return (jnp.all(jnp.isfinite(y))
                & jnp.all(jnp.isfinite(self.hi) | (y <= 0.0))
                & jnp.all(jnp.isfinite(self.lo) | (y >= 0.0)))

    def conj_domain_radius(self, y):
        return jnp.where(self.conj_in_domain(y), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        projected = jnp.where(jnp.isfinite(self.hi), y, jnp.minimum(y, 0.0))
        return jnp.where(jnp.isfinite(self.lo), projected, jnp.maximum(projected, 0.0))

    def group_key(self):
        return ("Box", self.lo, self.hi)


class Zero(ProxBlock):
    """f = 0 (free);  f^* = indicator{y = 0}."""
    scaling_class = MATRIX
    coordinate_separable = True
    smooth_grad = staticmethod(jnp.zeros_like)

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(_finite_step(t), v, jnp.nan)

    def prox_matrix(self, v, D):
        D = _validate_matrix_metric(v, D)
        return jnp.where(jnp.all(jnp.isfinite(D)), v, jnp.nan)

    def value(self, z):
        return jnp.asarray(0.0, jnp.float32)

    def conj_value(self, y):
        # EXACT indicator: dom f^* is the single point 0.  The former 1e-6
        # absolute slack reported the value 0 for duals that are NOT in
        # dom f^*, which makes the reported dual value a surrogate.
        return jnp.where(jnp.all(y == 0.0), 0.0, _INF)

    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y)) & jnp.all(y == 0.0)

    def conj_domain_radius(self, y):
        # dom f^* = {0}: if y is 0 every scaling stays in it, and if y is not 0
        # no positive scaling ever enters it.
        return jnp.where(jnp.all(y == 0.0), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        return jnp.zeros_like(y)


class Equality(ProxBlock):
    """f = indicator{u = 0} -- enforces the affine constraint (Kx + l) = 0.
    prox projects to 0;  f^*(y) = 0 for all y (the {0}-cone is self-polar-free)."""
    scaling_class = DIAGONAL
    coordinate_separable = True

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        return jnp.where(_finite_step(t), jnp.zeros_like(v), jnp.nan)

    def value(self, z):
        return jnp.where(jnp.max(jnp.abs(z)) <= 1e-5, 0.0, _INF)

    def conj_value(self, y):
        return jnp.asarray(0.0, jnp.float32)

    # f^* = 0 on all of R^m, exactly.
    # An indicator: value is 0 on the set and +inf off it, and prox is
    # the projection.  Feasibility is measured separately from the gap.
    is_indicator = True
    bound_capable = True

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y))

    def conj_domain_radius(self, y):
        return jnp.where(jnp.all(jnp.isfinite(y)), jnp.inf, 0.0)

    def conj_domain_project(self, y):
        return y


class Pinball(ProxBlock):
    """Quantile / pinball loss at level tau in (0,1):
        f(u) = sum_i  tau*max(u_i,0) + (1-tau)*max(-u_i,0).
    Separable, piecewise linear; the prox is an ASYMMETRIC soft-threshold
    (shrink by t*tau on the right, t*(1-tau) on the left).  Its conjugate is the
    box indicator{ tau-1 <= y <= tau }, so it contributes an exact duality gap.
    Quantile regression is  min_theta  sum rho_tau(y - X theta)  =  f(-X theta + y)."""
    scaling_class = DIAGONAL
    coordinate_separable = True
    bound_capable = True

    def __init__(self, tau=0.5):
        self.tau = float(tau)
        if not np.isfinite(self.tau) or not 0 <= self.tau <= 1:
            raise ValueError("tau must be finite and between zero and one")

    def prox(self, v, t):
        t = _validate_step(v, t, self.scaling_class)
        pos = v - t * self.tau
        neg = v + t * (1.0 - self.tau)
        return jnp.where(pos > 0.0, pos, jnp.where(neg < 0.0, neg, 0.0))

    def value(self, z):
        return jnp.sum(self.tau * jnp.maximum(z, 0.0) + (1.0 - self.tau) * jnp.maximum(-z, 0.0))

    def conj_value(self, y):
        return jnp.where(self.conj_in_domain(y), 0.0, _INF)

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y) & (y >= self.tau - 1.0) & (y <= self.tau))

    def conj_domain_radius(self, y):
        safe = jnp.where(y != 0, y, 1.0)
        radii = jnp.where(y > 0, self.tau / safe,
                          jnp.where(y < 0, (self.tau - 1.0) / safe, jnp.inf))
        return jnp.where(jnp.all(jnp.isfinite(y)), jnp.min(radii), 0.0)

    def conj_domain_project(self, y):
        return jnp.clip(y, self.tau - 1.0, self.tau)

    def group_key(self):
        return ("Pinball", self.tau)


class Logistic(ProxBlock):
    """Logistic / softplus loss  f(u) = sum_i log(1 + exp(u_i)) -- smooth, convex,
    separable.  No closed-form prox; solved per-coordinate by a few Newton steps on
        u + t*sigmoid(u) - v = 0     (strongly convex, quadratic convergence).
    grad f = sigmoid(u) (exposed for exact termination); f^* is the binary entropy
    y log y + (1-y) log(1-y) on (0,1).  Logistic regression is f(diag(-b) X theta)."""
    scaling_class = DIAGONAL
    coordinate_separable = True
    eval_kind = "newton"
    smooth_grad = staticmethod(lambda s: jax.nn.sigmoid(s))
    bound_capable = True

    def prox(self, v, t):
        # Preserve the finite-loss t=0 limit used by existing callers. Zero
        # entries give the identity; negative or nonfinite steps still fail.
        raw_step = jnp.asarray(t, dtype=jnp.result_type(v, jnp.float32))
        checked = _validate_step(v, jnp.where(raw_step == 0, 1., raw_step), self.scaling_class)
        t = jnp.where(raw_step == 0, 0., checked)
        # solve  u + t*sigmoid(u) = v  (monotone in u; root bracketed in [v-t, v]
        # since sigmoid in (0,1)).  SAFEGUARDED Newton: take the Newton step when it
        # stays in the bracket, else bisect -- plain Newton oscillates for large t
        # because sigmoid is flat-then-steep.
        lo = v - t
        hi = v + 0.0 * t
        u = 0.5 * (lo + hi)
        for _ in range(45):
            sg = jax.nn.sigmoid(u)
            h = u + t * sg - v
            lo = jnp.where(h < 0.0, u, lo)                     # keep h(lo) < 0 < h(hi)
            hi = jnp.where(h > 0.0, u, hi)
            un = u - h / (1.0 + t * sg * (1.0 - sg))           # Newton step
            # Once h is at its arithmetic roundoff floor, keep the root. A
            # rounded Newton update can equal a bracket endpoint; rejecting
            # it would bisect far away again. Merely accepting every endpoint
            # update would instead allow Newton to oscillate for large t.
            roundoff = jnp.finfo(h.dtype).eps * (jnp.abs(u) + jnp.abs(t * sg) + jnp.abs(v))
            # Near-endpoint Newton jumps can also alternate across an almost
            # unchanged bracket. Require an interior margin so rejected steps
            # bisect and make progress even in the sigmoid's flat tails.
            margin = 0.1 * (hi - lo)
            candidate = jnp.where((un > lo + margin) & (un < hi - margin),
                                  un, 0.5 * (lo + hi))
            u = jnp.where(jnp.abs(h) <= roundoff, u, candidate)
        return u

    def value(self, z):
        return jnp.sum(jnp.logaddexp(0.0, z))

    def conj_value(self, y):
        interior = (y > 0) & (y < 1)
        safe = jnp.where(interior, y, 0.5)
        ent = jnp.sum(jnp.where(interior,
            safe * jnp.log(safe) + (1.0 - safe) * jnp.log1p(-safe), 0.0))
        return jnp.where(self.conj_in_domain(y), ent, _INF)

    def conj_in_domain(self, y):
        return jnp.all(jnp.isfinite(y) & (y >= 0) & (y <= 1))

    def conj_domain_radius(self, y):
        safe = jnp.where(y > 0, y, 1.0)
        return jnp.where(jnp.all(jnp.isfinite(y) & (y >= 0)),
                         jnp.min(jnp.where(y > 0, 1.0 / safe, jnp.inf)), 0.0)

    def conj_domain_project(self, y):
        return jnp.clip(y, 0.0, 1.0)
