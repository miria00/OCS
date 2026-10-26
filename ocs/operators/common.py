"""Common concrete operators: identity, diagonal, dense, sparse."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import LinearOperator


class Identity(LinearOperator):
    def __init__(self, n):
        super().__init__((n, n))

    def matvec(self, x):
        return x

    def rmatvec(self, y):
        return y

    # |I|^{o2} = I: the squared-operator oracle is exact for free
    def sq_matvec(self, x):
        return x

    def sq_rmatvec(self, y):
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

    def sq_matvec(self, x):
        return (self.d * self.d) * x

    def sq_rmatvec(self, y):
        return (self.d * self.d) * y


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
    the whole point of OCS is to avoid this at scale.

    Products request HIGHEST matmul precision, including vmapped batches,
    without changing array dtypes or the global JAX configuration. Reduced
    GPU multiplication precision can otherwise prevent tight convergence.
    This setting is local to Dense; custom operator callbacks choose their
    own product precision.
    """
    def __init__(self, A):
        A = jnp.asarray(A, jnp.float32)
        super().__init__(A.shape)
        self.A = A

    def matvec(self, x):
        return jnp.matmul(self.A, x, precision=jax.lax.Precision.HIGHEST)

    def rmatvec(self, y):
        return jnp.matmul(self.A.T, y, precision=jax.lax.Precision.HIGHEST)

    def sq_matvec(self, x):
        return jnp.matmul(self.A * self.A, x, precision=jax.lax.Precision.HIGHEST)

    def sq_rmatvec(self, y):
        return jnp.matmul((self.A * self.A).T, y,
                          precision=jax.lax.Precision.HIGHEST)


# ---- operator algebra: composition, sum, scale ---------------------------
# These keep the operator-pure contract: the composed operator is again touched
# only through matvec / rmatvec, and no matrix is ever formed.  They are what let
# the surface DSL preserve structure --- e.g. a wavelet of a gradient
# ``W @ G @ x`` stays two matrix-free operators, never a stacked matrix.

class Compose(LinearOperator):
    """K = A B  (apply B, then A).  matvec = A(B x);  adjoint = Bᵀ(Aᵀ y).
    Requires A.shape[1] == B.shape[0]."""
    def __init__(self, A, B):
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"cannot compose operators of shapes {A.shape} and {B.shape}: "
                f"inner dims {A.shape[1]} != {B.shape[0]}")
        super().__init__((A.shape[0], B.shape[1]))
        self.A, self.B = A, B

    def matvec(self, x):
        return self.A.matvec(self.B.matvec(x))

    def rmatvec(self, y):
        return self.B.rmatvec(self.A.rmatvec(y))


class OpSum(LinearOperator):
    """K = A + B, same shape.  matvec = A x + B x;  adjoint = Aᵀ y + Bᵀ y."""
    def __init__(self, A, B):
        if A.shape != B.shape:
            raise ValueError(
                f"cannot add operators of shapes {A.shape} and {B.shape}")
        super().__init__(A.shape)
        self.A, self.B = A, B

    def matvec(self, x):
        return self.A.matvec(x) + self.B.matvec(x)

    def rmatvec(self, y):
        return self.A.rmatvec(y) + self.B.rmatvec(y)


class Scale(LinearOperator):
    """K = c A, a scalar multiple.  matvec = c (A x);  adjoint = c (Aᵀ y)."""
    def __init__(self, c, A):
        super().__init__(A.shape)
        self.c = float(c)
        self.A = A

    def matvec(self, x):
        return self.c * self.A.matvec(x)

    def rmatvec(self, y):
        return self.c * self.A.rmatvec(y)
