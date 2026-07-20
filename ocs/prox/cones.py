"""Cone projections (prox of cone indicators): second-order, PSD, exponential,
power.  Each is coupled across its block, so its scaling class is "scalar".
All are vmap-able (one cone per row), which is what the type-grouped dispatch
relies on.  proj_exp / proj_pow are ported from the validated CORGI cones.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .base import ProxBlock, SCALAR

_TOL = 1e-7


# =====================================================================
# Second-order cone  { (s, z) : ||z||_2 <= s }   (s is the first entry)
# =====================================================================
class SecondOrderCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "closed_form"

    def prox(self, v, t):
        s, z = v[0], v[1:]
        nz = jnp.linalg.norm(z)
        inside = nz <= s
        below = nz <= -s
        scale = (nz + s) / (2.0 * nz + 1e-30)
        proj_z = jnp.where(inside, z, jnp.where(below, 0.0 * z, scale * z))
        proj_s = jnp.where(inside, s, jnp.where(below, 0.0, scale * nz))
        return jnp.concatenate([jnp.reshape(proj_s, (1,)), proj_z])

    def value(self, z):
        return jnp.where(jnp.linalg.norm(z[1:]) <= z[0] + 1e-5, 0.0, jnp.inf)

    def conj_value(self, y):                       # self-dual
        return self.value(y)


# =====================================================================
# PSD cone  { X = X^T : X >= 0 }, X packed as a k*k row-major vector
# =====================================================================
class PSDCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "spectral"

    def __init__(self, k):
        self.k = int(k)

    def prox(self, v, t):
        k = self.k
        M = v.reshape(k, k)
        M = 0.5 * (M + M.T)
        w, U = jnp.linalg.eigh(M)
        Mp = (U * jnp.maximum(w, 0.0)) @ U.T
        return Mp.reshape(-1)

    def value(self, z):
        M = z.reshape(self.k, self.k)
        w = jnp.linalg.eigvalsh(0.5 * (M + M.T))
        return jnp.where(jnp.min(w) >= -1e-4, 0.0, jnp.inf)

    def conj_value(self, y):                       # self-dual
        return self.value(y)

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
    v = jnp.asarray(v, jnp.float32)
    r, s, t = v[0], v[1], v[2]

    def boundary_point(u):
        eu = jnp.exp(u)
        a = r * u + s + t * eu
        b = u * u + 1.0 + jnp.exp(2.0 * u)
        y = jnp.maximum(a / b, 0.0)
        return jnp.stack([y * u, y, y * eu])

    def H(u):
        eu = jnp.exp(u); e2 = jnp.exp(2.0 * u)
        a = r * u + s + t * eu; b = u * u + 1.0 + e2
        ap = r + t * eu; bp = 2.0 * u + 2.0 * e2
        return 2.0 * ap * b - a * bp

    U, ng = 20.0, 192
    us = jnp.linspace(-U, U, ng)
    P = jax.vmap(boundary_point)(us)
    u0 = us[jnp.argmin(jnp.sum((P - v) ** 2, axis=1))]

    dH = jax.grad(H)
    def body(st):
        u, it = st
        return jnp.clip(u - H(u) / (dH(u) + 1e-30), -U, U), it + 1
    u_ref, _ = lax.while_loop(lambda st: st[1] < 25, body, (u0, 0))

    A = boundary_point(u_ref)
    B = jnp.array([jnp.minimum(r, 0.0), 0.0, jnp.maximum(t, 0.0)])
    O = jnp.zeros(3)
    cands = jnp.stack([A, B, O])
    feas = jax.vmap(lambda p: _in_exp(p[0], p[1], p[2]))(cands)
    dd = jnp.where(feas, jnp.sum((cands - v) ** 2, axis=1), jnp.inf)
    return jnp.where(_in_exp(r, s, t), v, cands[jnp.argmin(dd)])


class ExpCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "newton"

    def prox(self, v, t):
        return proj_exp(v)

    def value(self, z):                            # indicator; dual cone omitted
        return jnp.where(_in_exp(z[0], z[1], z[2]), 0.0, jnp.inf)


# =====================================================================
# Power cone  K_pow^alpha = { x,y>=0 : x^a y^{1-a} >= |z| }
# =====================================================================
def _in_pow(x, y, z, a, tol=_TOL):
    xs = jnp.maximum(x, 0.0); ys = jnp.maximum(y, 0.0)
    lhs = (xs ** a) * (ys ** (1.0 - a))
    return (x >= -tol) & (y >= -tol) & (lhs - jnp.abs(z) >= -tol * (1.0 + jnp.abs(z)))


def proj_pow(v, alpha=0.5):
    v = jnp.asarray(v, jnp.float32)
    x0, y0, z0 = v[0], v[1], v[2]
    a = jnp.asarray(alpha, jnp.float32)
    az = jnp.abs(z0); sz = jnp.sign(z0)

    def p_of(x, y):
        return (jnp.maximum(x, 1e-30) ** a) * (jnp.maximum(y, 1e-30) ** (1.0 - a))

    def grads_hess(x, y):
        p = p_of(x, y)
        px = a * p / x; py = (1.0 - a) * p / y; res = p - az
        gx = (x - x0) + res * px; gy = (y - y0) + res * py
        pxx = a * (a - 1.0) * p / (x * x)
        pyy = (1.0 - a) * (-a) * p / (y * y)
        pxy = a * (1.0 - a) * p / (x * y)
        return gx, gy, 1.0 + px * px + res * pxx, 1.0 + py * py + res * pyy, px * py + res * pxy

    def body(st):
        x, y, it = st
        gx, gy, hxx, hyy, hxy = grads_hess(x, y)
        det = hxx * hyy - hxy * hxy
        det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
        x_new = jnp.maximum(x - (hyy * gx - hxy * gy) / det, 1e-12)
        y_new = jnp.maximum(y - (hxx * gy - hxy * gx) / det, 1e-12)
        return x_new, y_new, it + 1

    x_s, y_s, _ = lax.while_loop(lambda st: st[2] < 40,
                                 body, (jnp.maximum(x0, 1e-2), jnp.maximum(y0, 1e-2), 0))
    A = jnp.stack([x_s, y_s, sz * p_of(x_s, y_s)])
    O = jnp.zeros(3)
    cands = jnp.stack([A, O])
    feas = jax.vmap(lambda p: _in_pow(p[0], p[1], p[2], a))(cands)
    dd = jnp.where(feas, jnp.sum((cands - v) ** 2, axis=1), jnp.inf)
    return jnp.where(_in_pow(x0, y0, z0, a), v, cands[jnp.argmin(dd)])


class PowerCone(ProxBlock):
    scaling_class = SCALAR
    eval_kind = "newton"

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)

    def prox(self, v, t):
        return proj_pow(v, self.alpha)

    def value(self, z):                            # indicator; dual cone omitted
        return jnp.where(_in_pow(z[0], z[1], z[2], self.alpha), 0.0, jnp.inf)

    def group_key(self):
        return ("PowerCone", self.alpha)
