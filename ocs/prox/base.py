"""The proximal-block interface — the heart of OCS's prox library.

A block is accessed only through its scaled proximal operator

    prox_f^D(v) = argmin_u  f(u) + (1/2)(u - v)^T D (u - v).

Every block declares:
  * ``prox(v, t)`` — the scalar-metric prox, D = (1/t) I (always available);
  * ``scaling_class`` — the richest admissible D:
        "scalar"   : only (1/t) I
        "diagonal" : diag(d), d > 0   (iff f is coordinatewise separable)
        "matrix"   : any D >= 0        (quadratics / affine)
  * ``smooth_grad`` — grad f if f is smooth (used for exact termination), else None.

The calculus layer (``scale``, ``precompose_affine``, ``add_linear_quadratic``)
folds a scaling, an affine pre-composition, and a linear+quadratic term into ANY
base block automatically, via closed-form parameter transforms — so we never hand
-code every combination.  This mirrors the POGS prox library.
"""
from __future__ import annotations

import jax.numpy as jnp

SCALAR, DIAGONAL, MATRIX = "scalar", "diagonal", "matrix"


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
        PD matrix.  Only the MATRIX tier (quadratic / affine f) admits a closed form,
        prox_f^D(v) = (Q + D)^{-1}(D v - q); other tiers raise."""
        raise NotImplementedError(
            f"{type(self).__name__} is scaling_class '{self.scaling_class}': a full-"
            "matrix metric is admissible only for quadratic/affine blocks (the matrix tier)")

    def value(self, z):
        """f(z) -- the block's value, for the primal objective / duality gap."""
        raise NotImplementedError

    def conj_value(self, y):
        """f^*(y) -- the conjugate value, for the dual objective / duality gap."""
        raise NotImplementedError

    def group_key(self):
        """Signature for type-grouped GPU dispatch: blocks with equal
        (group_key, block size) are evaluated together in one vmap kernel.
        Override to include parameters that change the prox."""
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
        self.base, self.c = base, float(c)
        self.scaling_class = base.scaling_class
        if base.smooth_grad is not None:
            self.smooth_grad = lambda s, g=base.smooth_grad, c=self.c: c * g(s)

    def prox(self, v, t):
        return self.base.prox(v, t * self.c)


class _AffinePre(ProxBlock):
    """h(u) = f(a u - b), elementwise a, b.
    prox_{t h}(v) = (prox_{t a^2 f}(a v - b) + b) / a."""
    def __init__(self, base, a, b):
        self.base = base
        self.a = jnp.asarray(a, jnp.float32)
        self.b = jnp.asarray(b, jnp.float32)
        self.scaling_class = base.scaling_class

    def prox(self, v, t):
        a, b = self.a, self.b
        return (self.base.prox(a * v - b, t * a * a) + b) / a


class _LinQuad(ProxBlock):
    """h(u) = f(u) + d^T u + (1/2) u^T diag(e) u.
    prox_{t h}(v) = prox_{t' f}((v - t d)/(1 + t e)),  t' = t/(1 + t e)."""
    def __init__(self, base, d, e):
        self.base = base
        self.d = jnp.asarray(d, jnp.float32)
        self.e = jnp.asarray(e, jnp.float32)
        self.scaling_class = base.scaling_class

    def prox(self, v, t):
        scale = 1.0 + t * self.e
        return self.base.prox((v - t * self.d) / scale, t / scale)
