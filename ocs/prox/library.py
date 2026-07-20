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

from .base import ProxBlock, SCALAR, DIAGONAL, MATRIX

_INF = jnp.inf


class SumSquares(ProxBlock):
    """f(u) = 1/2 ||u||_2^2 (self-conjugate).  Quadratic -> admits a full matrix
    metric: prox_f^D(v) = (I + D)^{-1} D v."""
    scaling_class = MATRIX
    smooth_grad = staticmethod(lambda s: s)

    def prox(self, v, t):
        return v / (1.0 + t)                       # scalar / diagonal metric

    def prox_matrix(self, v, D):
        k = v.shape[0]
        return jnp.linalg.solve(jnp.eye(k, dtype=v.dtype) + D, D @ v)

    def value(self, z):
        return 0.5 * jnp.sum(z * z)

    def conj_value(self, y):
        return 0.5 * jnp.sum(y * y)


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
        self.P = jnp.asarray(P, jnp.float32)
        self.dim = int(self.P.shape[0])
        self.q = (jnp.zeros(self.dim, jnp.float32) if q is None
                  else jnp.asarray(q, jnp.float32))
        self.smooth_grad = lambda s: self.P @ s + self.q

    def prox(self, v, t):
        d = jnp.broadcast_to(1.0 / t, (self.dim,))
        return jnp.linalg.solve(self.P + jnp.diag(d), d * v - self.q)

    def prox_matrix(self, v, D):
        return jnp.linalg.solve(self.P + D, D @ v - self.q)

    def value(self, z):
        return 0.5 * jnp.dot(z, self.P @ z) + jnp.dot(self.q, z)

    def conj_value(self, y):
        w = y - self.q
        return 0.5 * jnp.dot(w, jnp.linalg.solve(self.P, w))


class L1(ProxBlock):
    """f(u) = lam ||u||_1;  f^* = indicator{ ||y||_inf <= lam }."""
    scaling_class = DIAGONAL

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def prox(self, v, t):
        return jnp.sign(v) * jnp.maximum(jnp.abs(v) - t * self.lam, 0.0)

    def value(self, z):
        return self.lam * jnp.sum(jnp.abs(z))

    def conj_value(self, y):
        return jnp.where(jnp.max(jnp.abs(y)) <= self.lam + 1e-6, 0.0, _INF)

    def group_key(self):
        return ("L1", self.lam)


class L2(ProxBlock):
    """f(u) = lam ||u||_2 (not squared);  f^* = indicator{ ||y||_2 <= lam }."""
    scaling_class = SCALAR

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def prox(self, v, t):
        nrm = jnp.linalg.norm(v)
        return jnp.maximum(0.0, 1.0 - t * self.lam / (nrm + 1e-30)) * v

    def value(self, z):
        return self.lam * jnp.linalg.norm(z)

    def conj_value(self, y):
        return jnp.where(jnp.linalg.norm(y) <= self.lam + 1e-6, 0.0, _INF)

    def group_key(self):
        return ("L2", self.lam)


class GroupL2(ProxBlock):
    """f(u) = lam sum_g ||u_g||_2 over contiguous groups of size `group_size`
    (isotropic TV when group_size=2 on a 2-D gradient; row-group lasso for
    feature selection).  prox = per-group block shrinkage; one block evaluates
    all groups via a single reshape+vmap."""
    scaling_class = SCALAR

    def __init__(self, lam=1.0, group_size=2):
        self.lam = float(lam); self.g = int(group_size)

    def prox(self, v, t):
        ts = jnp.mean(t)                                 # coupled block -> uniform step
        R = v.reshape(-1, self.g)
        nrm = jnp.linalg.norm(R, axis=1, keepdims=True)
        R = jnp.maximum(0.0, 1.0 - ts * self.lam / (nrm + 1e-30)) * R
        return R.reshape(-1)

    def value(self, z):
        return self.lam * jnp.sum(jnp.linalg.norm(z.reshape(-1, self.g), axis=1))

    def conj_value(self, y):
        mx = jnp.max(jnp.linalg.norm(y.reshape(-1, self.g), axis=1))
        return jnp.where(mx <= self.lam + 1e-6, 0.0, _INF)

    def group_key(self):
        return ("GroupL2", self.lam, self.g)


class PiecewiseLinearConvex(ProxBlock):
    """Separable convex piecewise-linear  f(u) = sum_i max_j (s_j u_i + r_j)  (a max
    of P affine pieces per coordinate).  Models PWL convex costs / the epigraph of
    concave PWL utilities (network utility maximization, quantile-type losses).
    The prox has no closed form for general pieces, but the optimality condition
    u = v - t*g'(u) is monotone (g' = active slope, nondecreasing), so a short
    bisection solves it robustly and vectorizes over all coordinates."""
    scaling_class = DIAGONAL
    eval_kind = "bisection"

    def __init__(self, slopes, intercepts):
        self.s = jnp.asarray(slopes, jnp.float32)              # (P,)
        self.r = jnp.asarray(intercepts, jnp.float32)          # (P,)

    def prox(self, v, t):
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

    def group_key(self):
        return ("PiecewiseLinearConvex", self.s.shape[0], float(self.s.sum()), float(self.r.sum()))


class NonNeg(ProxBlock):
    """f = indicator{u >= 0};  f^* = indicator{y <= 0} (polar)."""
    scaling_class = DIAGONAL

    def prox(self, v, t):
        return jnp.maximum(v, 0.0)

    def value(self, z):
        return jnp.where(jnp.min(z) >= -1e-6, 0.0, _INF)

    def conj_value(self, y):
        return jnp.where(jnp.max(y) <= 1e-6, 0.0, _INF)


class Box(ProxBlock):
    """f = indicator{lo <= u <= hi};  f^*(y) = sum max(lo y, hi y) (support fn)."""
    scaling_class = DIAGONAL

    def __init__(self, lo=0.0, hi=1.0):
        self.lo, self.hi = lo, hi

    def prox(self, v, t):
        return jnp.clip(v, self.lo, self.hi)

    def value(self, z):
        ok = (jnp.min(z - self.lo) >= -1e-6) & (jnp.max(z - self.hi) <= 1e-6)
        return jnp.where(ok, 0.0, _INF)

    def conj_value(self, y):
        return jnp.sum(jnp.maximum(self.lo * y, self.hi * y))

    def group_key(self):
        return ("Box", self.lo, self.hi)


class Zero(ProxBlock):
    """f = 0 (free);  f^* = indicator{y = 0}."""
    scaling_class = MATRIX

    def prox(self, v, t):
        return v

    def value(self, z):
        return jnp.asarray(0.0, jnp.float32)

    def conj_value(self, y):
        return jnp.where(jnp.max(jnp.abs(y)) <= 1e-6, 0.0, _INF)


class Equality(ProxBlock):
    """f = indicator{u = 0} -- enforces the affine constraint (Kx + l) = 0.
    prox projects to 0;  f^*(y) = 0 for all y (the {0}-cone is self-polar-free)."""
    scaling_class = DIAGONAL

    def prox(self, v, t):
        return jnp.zeros_like(v)

    def value(self, z):
        return jnp.where(jnp.max(jnp.abs(z)) <= 1e-5, 0.0, _INF)

    def conj_value(self, y):
        return jnp.asarray(0.0, jnp.float32)


class Pinball(ProxBlock):
    """Quantile / pinball loss at level tau in (0,1):
        f(u) = sum_i  tau*max(u_i,0) + (1-tau)*max(-u_i,0).
    Separable, piecewise linear; the prox is an ASYMMETRIC soft-threshold
    (shrink by t*tau on the right, t*(1-tau) on the left).  Its conjugate is the
    box indicator{ tau-1 <= y <= tau }, so it contributes an exact duality gap.
    Quantile regression is  min_theta  sum rho_tau(y - X theta)  =  f(-X theta + y)."""
    scaling_class = DIAGONAL

    def __init__(self, tau=0.5):
        self.tau = float(tau)

    def prox(self, v, t):
        pos = v - t * self.tau
        neg = v + t * (1.0 - self.tau)
        return jnp.where(pos > 0.0, pos, jnp.where(neg < 0.0, neg, 0.0))

    def value(self, z):
        return jnp.sum(self.tau * jnp.maximum(z, 0.0) + (1.0 - self.tau) * jnp.maximum(-z, 0.0))

    def conj_value(self, y):
        ok = (jnp.min(y) >= self.tau - 1.0 - 1e-6) & (jnp.max(y) <= self.tau + 1e-6)
        return jnp.where(ok, 0.0, _INF)

    def group_key(self):
        return ("Pinball", self.tau)


class Logistic(ProxBlock):
    """Logistic / softplus loss  f(u) = sum_i log(1 + exp(u_i)) -- smooth, convex,
    separable.  No closed-form prox; solved per-coordinate by a few Newton steps on
        u + t*sigmoid(u) - v = 0     (strongly convex, quadratic convergence).
    grad f = sigmoid(u) (exposed for exact termination); f^* is the binary entropy
    y log y + (1-y) log(1-y) on (0,1).  Logistic regression is f(diag(-b) X theta)."""
    scaling_class = DIAGONAL
    eval_kind = "newton"
    smooth_grad = staticmethod(lambda s: jax.nn.sigmoid(s))

    def prox(self, v, t):
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
            u = jnp.where((un > lo) & (un < hi), un, 0.5 * (lo + hi))
        return u

    def value(self, z):
        return jnp.sum(jnp.logaddexp(0.0, z))

    def conj_value(self, y):
        yc = jnp.clip(y, 1e-7, 1.0 - 1e-7)
        ent = jnp.sum(yc * jnp.log(yc) + (1.0 - yc) * jnp.log(1.0 - yc))
        ok = (jnp.min(y) >= -1e-6) & (jnp.max(y) <= 1.0 + 1e-6)
        return jnp.where(ok, ent, _INF)
