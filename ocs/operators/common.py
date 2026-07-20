"""Common concrete operators: identity, diagonal, dense, sparse."""
from __future__ import annotations

import jax.numpy as jnp

from .base import LinearOperator


class Identity(LinearOperator):
    def __init__(self, n):
        super().__init__((n, n))

    def matvec(self, x):
        return x

    def rmatvec(self, y):
        return y


class Diagonal(LinearOperator):
    """K = diag(d)."""
    def __init__(self, d):
        d = jnp.asarray(d, jnp.float32)
        super().__init__((d.shape[0], d.shape[0]))
        self.d = d

    def matvec(self, x):
        return self.d * x

    def rmatvec(self, y):
        return self.d * y


class Weighted(LinearOperator):
    """Row-scaled operator  K = diag(w) B.  Scales the rows of a base operator by
    w (e.g. per-measurement weights / confidences); a wide-range w makes K
    row-ill-conditioned even when B is well conditioned."""
    def __init__(self, base, w):
        w = jnp.asarray(w, jnp.float32)
        super().__init__(base.shape)
        self.base = base
        self.w = w

    def matvec(self, x):
        return self.w * self.base.matvec(x)

    def rmatvec(self, y):
        return self.base.rmatvec(self.w * y)


class Dense(LinearOperator):
    """K = A, an explicit dense matrix.  For small/validation use only —
    the whole point of OCS is to avoid this at scale."""
    def __init__(self, A):
        A = jnp.asarray(A, jnp.float32)
        super().__init__(A.shape)
        self.A = A

    def matvec(self, x):
        return self.A @ x

    def rmatvec(self, y):
        return self.A.T @ y
