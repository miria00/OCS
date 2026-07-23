"""The linear-operator interface: forward (matvec) and adjoint (rmatvec) only.

An operator is never a stored matrix.  It exposes two functions and a shape.
When the forward map is JAX-linear, the adjoint can be obtained for free with
``jax.linear_transpose``; see ``from_forward``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


class LinearOperator:
    """Base class.  Subclasses implement ``matvec`` and ``rmatvec``.

    shape : (m, n)   maps R^n -> R^m
    well_conditioned : CERTIFIES that this operator's rows and columns
        already have ~uniform (unit-scale) energy -- e.g. an orthonormal
        transform.  It is a scale certificate, NOT a mere "individually well
        conditioned" hint: stacking well_conditioned blocks is only safe when
        each is at a COMMON unit scale, so a rescaled transform (e.g.
        Weighted(F, 100)) must leave this False even though it is well
        conditioned on its own.  When every stacked block certifies this, the
        stacked operator's row/column norms are already balanced and
        compile-time equilibration probing is skipped.
    """
    well_conditioned = False

    def __init__(self, shape):
        self.shape = tuple(shape)

    # forward:  x in R^n  |->  K x in R^m
    def matvec(self, x):
        raise NotImplementedError

    # adjoint:  y in R^m  |->  K^T y in R^n
    def rmatvec(self, y):
        raise NotImplementedError

    # K @ x  builds an affine expression (K x + 0); see model.expression
    def __matmul__(self, var):
        from ..model.expression import AffineExpr
        return AffineExpr(op=self, var=var)

    @classmethod
    def from_forward(cls, forward, n, m):
        """Build an operator from a JAX-linear forward map; adjoint is free."""
        op = cls((m, n))
        example = jnp.zeros(n, jnp.float32)
        rt = jax.linear_transpose(forward, example)
        op.matvec = forward
        op.rmatvec = lambda y: rt(y)[0]
        return op

    def adjoint_test(self, key, tol=1e-4):
        """Check <Kx, y> == <x, K^T y> on random vectors (sanity)."""
        m, n = self.shape
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (n,), jnp.float32)
        y = jax.random.normal(k2, (m,), jnp.float32)
        lhs = jnp.vdot(self.matvec(x), y)
        rhs = jnp.vdot(x, self.rmatvec(y))
        return float(jnp.abs(lhs - rhs) / (1 + jnp.abs(lhs)))
