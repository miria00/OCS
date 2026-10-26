"""Data scales for homogeneous cones beside a positive-definite quadratic.

No iteration metric or absolute unit enters these references. The operator
is accessed only through forward and adjoint products. Other problem classes
retain their existing relative scales.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from ..prox.base import (_Scaled, _AffinePre, _LinQuad, _known_quadratic,
                         _quadratic_coefficients)
from ..prox.library import SumSquares, Quadratic, Zero, Equality, Box
from ..prox.cones import ExpConeBatch, PowerConeBatch, SOCBatch
from .domain_checks import _indicator, _rows, _norm


def _diagonal_quadratic(block):
    """Known h,b for g(x)=.5*sum(h*x*x)+b.T*x+constant, or None."""
    if type(block) is SumSquares:
        return 1.0, 0.0
    if type(block) is Zero:
        return 0.0, 0.0
    if type(block) is Quadratic and block.coordinate_separable:
        return jnp.diag(block.P), block.q
    if type(block) not in (_Scaled, _AffinePre, _LinQuad):
        return None
    base = _diagonal_quadratic(block.base)
    if base is None:
        return None
    h, b = base
    if type(block) is _Scaled:
        return block.c * h, block.c * b
    if type(block) is _AffinePre:
        return h * block.a * block.a, block.a * (b - h * block.b)
    return h + block.e, b + block.d


def make_cone_reference_scales(g_block, ops, blocks, slices):
    """Return per-block graph/domain scales, or None when unavailable.

    For a recognized positive-definite quadratic, x_ref=-H^{-1}(b+c) is
    intrinsic primal data. Diagonal H uses elementwise arithmetic; a dense H
    is factored once per family, then uses triangular solves. A cone receives
    K_i*x_ref only when its offset is zero. Its dual
    reference is w*(||w||/||H^(-1/2)*K_i.T*w||)^2, w=K_i*x_ref. Under
    K_i'=alpha*K_i and (g',c')=gamma*(g,c), these references scale by alpha
    and gamma/alpha respectively. Batched cones retain their existing checks:
    a single Rayleigh quotient for their stacked operator could allow another
    cone's data to inflate a local dual scale. They need separate reference
    operators before this construction can be used without losing isolation.
    A singular or numerically non-positive-definite dense Hessian receives no
    reference scale. No pseudoinverse, regularization, or absolute floor is
    used to extend the positive-definite formula to that case.
    """
    indicators = tuple(_indicator(block) for block in blocks)
    enabled = tuple(b is not None and type(b) not in (
                        Equality, Box, ExpConeBatch, PowerConeBatch, SOCBatch)
                    for b in indicators)
    if not any(enabled):
        return None
    quadratic = _diagonal_quadratic(g_block)
    if quadratic is not None:
        h, b = quadratic
        factor = None
    elif _known_quadratic(g_block):
        dimension = ops[0].shape[1]
        H, b = _quadratic_coefficients(g_block, dimension, jnp.float32)
        # H belongs to the proximal block, not to the problem operator. Its
        # dense representation already exists for this quadratic. Factor only
        # once here; never form K, K.T K, or a dense Hessian for a diagonal block.
        matrix = np.asarray(H, dtype=np.float32)
        if not np.all(np.isfinite(matrix)):
            return None
        try:
            cholesky = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(cholesky)) or np.any(np.diag(cholesky) <= 0):
            return None
        factor = jnp.asarray(cholesky)
    else:
        return None

    def references(data):
        l, c = data
        direction = c + jnp.asarray(b, c.dtype)
        if factor is None:
            diagonal = jnp.broadcast_to(jnp.asarray(h, c.dtype), c.shape)
            eligible = jnp.all(jnp.isfinite(diagonal) & (diagonal > 0))
            point = -direction / jnp.where(diagonal > 0, diagonal, 1.0)
        else:
            eligible = jnp.asarray(True)
            intermediate = solve_triangular(factor, direction, lower=True)
            point = -solve_triangular(factor.T, intermediate, lower=False)
        zero = jnp.asarray(0.0, c.dtype)
        scales, any_active = [], jnp.asarray(False)
        for op, indicator, use, (start, end) in zip(ops, indicators, enabled, slices):
            if not use:
                scales.append((zero, zero, zero, zero))
                continue
            active = eligible & jnp.all(l[start:end] == 0)
            primal = op.matvec(point)
            adjoint = op.rmatvec(primal)
            inverse_metric = (adjoint / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1.0))
                              if factor is None else solve_triangular(factor, adjoint, lower=True))
            wn, an = _norm(primal), _norm(inverse_metric)
            ratio = wn / jnp.where(an > 0, an, 1.0)
            dual = (primal * ratio) * ratio
            dual = jnp.where((wn > 0) & (an == 0), jnp.inf, dual)
            scales.append(tuple(jnp.where(active, v, zero) for v in (
                _norm(primal), _norm(dual),
                _norm(_rows(indicator, primal), axis=1),
                _norm(_rows(indicator, dual), axis=1))))
            any_active |= active
        objective = jnp.where(any_active, jnp.abs(jnp.vdot(direction, point)), zero)
        return tuple(scales), objective

    return references
