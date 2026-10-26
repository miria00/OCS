"""Independent domain, polar-domain, and complementarity diagnostics.

These checks do not subtract a dual iterate from its proximal update.  In
particular, a large multiplier cannot erase primal infeasibility through
float32 cancellation.  Exponential/power membership uses homogeneous analytic
inequalities, not the iterative projection routine.  Those violations are
feasibility measures, not exact distances or objective-error bounds.

Complementarity is normalized by finite primal/dual objective expressions.
When their scale is zero, a zero numerator reports zero and a nonzero one
reports infinity: no arbitrary unit floor is silently introduced.  A small
result is a numerical KKT diagnostic; it is not an exact duality certificate.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..prox.base import ProxBlock, _Scaled, _has_oracle
from ..prox.cones import (
    ExpCone, ExpConeBatch, PowerCone, PowerConeBatch, PSDCone,
    SecondOrderCone, SOCBatch,
)
from ..prox.library import Box, Equality, NonNeg, Zero


_INDICATORS = (Equality, NonNeg, Box, SecondOrderCone, SOCBatch, PSDCone,
               ExpCone, ExpConeBatch, PowerCone, PowerConeBatch)
_EXPERIMENTAL = (ExpCone, ExpConeBatch, PowerCone, PowerConeBatch)


def _indicator(block):
    if type(block) is _Scaled and block.c > 0:
        return _indicator(block.base)
    return block if type(block) in _INDICATORS else None


def _norm(v, axis=None):
    scale = jnp.max(jnp.abs(v), axis=axis, keepdims=True, initial=0.0)
    value = scale * jnp.sqrt(jnp.sum((v / jnp.where(scale > 0, scale, 1.0)) ** 2,
                                    axis=axis, keepdims=True))
    return jnp.squeeze(value, axis=axis) if axis is not None else jnp.squeeze(value)


def _ratio(value, scale):
    ratio = jnp.where(scale > 0, value / jnp.where(scale > 0, scale, 1.0),
                       jnp.where(value == 0, 0.0, jnp.inf))
    return jnp.where(jnp.isfinite(value) & jnp.isfinite(scale), ratio, jnp.inf)


def _rows(block, v):
    if type(block) in (ExpConeBatch, PowerConeBatch):
        q = block.q
        return jnp.stack([v[:q], v[q:2*q], v[2*q:]], axis=1)
    if type(block) is SOCBatch:
        q = block.q
        z = jnp.reshape(v[q:], (block.d, q), order="F").T
        return jnp.concatenate([v[:q, None], z], axis=1)
    return v[None, :]


def _exp_violation(v):
    """Homogeneous logarithmic form of t >= s exp(r/s), plus its closure."""
    r, s, t = v[:, 0], v[:, 1], v[:, 2]
    tiny = jnp.finfo(v.dtype).tiny
    bound = s * (jnp.log(jnp.maximum(t, tiny)) - jnp.log(jnp.maximum(s, tiny)))
    interior = jnp.maximum(jnp.maximum(r - bound, -t), 0.0)
    # A positive s requires positive t; avoid an underflowed logarithm accepting
    # t=0 merely because r is very negative.
    interior = jnp.where(t > 0, interior, jnp.maximum(interior, s))
    closure = jnp.maximum(jnp.maximum(r, jnp.abs(s)), jnp.maximum(-t, 0.0))
    return jnp.where(s > 0, interior, closure)


def _power_violation(v, alpha, weights=None):
    """Distance to the nearest of three explicit feasible witnesses.

    The forward inequality |z|-x**a*y**(1-a) is ill-conditioned near a
    zero x or y. Instead either lower |z|, raise x, or raise y to obtain a
    point in the cone. Each witness distance bounds the true cone distance
    from above. No iterative projection or absolute tolerance is involved.
    ``weights`` maps displacement back from polar coordinates when needed.
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    xp, yp, az = jnp.maximum(x, 0.0), jnp.maximum(y, 0.0), jnp.abs(z)
    beta = 1.0-alpha
    weights = jnp.ones(3, v.dtype) if weights is None else jnp.asarray(weights, v.dtype)

    def distance(dx, dy, dz):
        displacement = jnp.stack([dx, dy, dz], axis=1)*weights
        value = _norm(displacement, axis=1)
        return jnp.where(jnp.all(jnp.isfinite(displacement), axis=1), value, jnp.inf)

    bound = xp**alpha * yp**beta
    lower_z = distance(xp-x, yp-y, jnp.maximum(az-bound, 0.0))
    log_z = jnp.log(jnp.where(az > 0, az, 1.0))
    log_x = jnp.log(jnp.where(xp > 0, xp, 1.0))
    log_y = jnp.log(jnp.where(yp > 0, yp, 1.0))
    required_x = jnp.where(az == 0, 0.0, jnp.exp((log_z-beta*log_y)/alpha))
    required_y = jnp.where(az == 0, 0.0, jnp.exp((log_z-alpha*log_x)/beta))
    # A positive third coordinate cannot use an underflowed zero lift as a
    # feasible witness. Other finite witnesses remain available in that case.
    valid_x = (az == 0) | ((yp > 0) & (required_x > 0))
    valid_y = (az == 0) | ((xp > 0) & (required_y > 0))
    lift_x = jnp.where(valid_x,
        distance(jnp.maximum(xp, required_x)-x, yp-y, jnp.zeros_like(z)), jnp.inf)
    lift_y = jnp.where(valid_y,
        distance(xp-x, jnp.maximum(yp, required_y)-y, jnp.zeros_like(z)), jnp.inf)
    return jnp.minimum(lower_z, jnp.minimum(lift_x, lift_y))


def _cone_violation(block, rows, polar=False):
    """Rows are independently normalized before this function is called."""
    kind = type(block)
    if kind is Equality:
        return jnp.zeros(rows.shape[0]) if polar else _norm(rows, axis=1)
    if kind is NonNeg:
        return _norm(jnp.maximum(rows if polar else -rows, 0.0), axis=1)
    if kind in (ExpCone, ExpConeBatch):
        if polar:
            # -y belongs to K_exp^* iff (y_1,y_0,-e*y_2) belongs to K_exp.
            rows = jnp.stack([rows[:, 1], rows[:, 0], -jnp.e * rows[:, 2]], axis=1)
        return _exp_violation(rows)
    if kind in (PowerCone, PowerConeBatch):
        if polar:
            rows = jnp.stack([-rows[:, 0] / block.alpha,
                              -rows[:, 1] / (1.0 - block.alpha), -rows[:, 2]], axis=1)
            # A primal witness w maps to (-a*w0, -(1-a)*w1, -w2)
            # in the polar. Measure its displacement in the original units.
            return _power_violation(rows, block.alpha,
                                    (block.alpha, 1.0-block.alpha, 1.0))
        return _power_violation(rows, block.alpha)
    if kind in (SecondOrderCone, SOCBatch):
        t, z = rows[:, 0], rows[:, 1:]
        nz = _norm(z, axis=1)
        factor = (nz + t) / jnp.where(nz > 0, 2.0 * nz, 1.0)
        pz = jnp.where((nz <= t)[:, None], z,
                       jnp.where((nz <= -t)[:, None], 0.0, factor[:, None] * z))
        pt = jnp.where(nz <= t, t, jnp.where(nz <= -t, 0.0, (nz + t) / 2.0))
        projection = jnp.concatenate([pt[:, None], pz], axis=1)
        return _norm(projection if polar else rows - projection, axis=1)
    # One PSD cone per block.  Projection also checks the required symmetry;
    # the polar allows a skew-symmetric part, as it should in the ambient space.
    projection = block.prox(rows[0], 1.0)[None, :]
    return _norm(projection if polar else rows - projection, axis=1)


def _support(block, y):
    if type(block) is not Box:
        return jnp.asarray(0.0, y.dtype)
    lo, hi = jnp.asarray(block.lo), jnp.asarray(block.hi)
    terms = jnp.where(y > 0, hi * y, jnp.where(y < 0, lo * y, 0.0))
    return jnp.sum(terms)


def _block_checks(block, s, y, kx, l, dual_scale=None, reference_scales=None):
    # None marks a structurally absent offset (including the primal g block).
    # Explicit offset callbacks always retain their actual arrays and norms.
    if type(block) is Box:
        lo, hi = jnp.asarray(block.lo), jnp.asarray(block.hi)
        bounds = jnp.maximum(jnp.abs(jnp.where(jnp.isfinite(lo), lo, 0.0)),
                             jnp.abs(jnp.where(jnp.isfinite(hi), hi, 0.0)))
        offset_norm = jnp.asarray(0.0, s.dtype) if l is None else _norm(l)
        scale = jnp.maximum(jnp.maximum(_norm(kx), offset_norm),
                             _norm(jnp.broadcast_to(bounds, s.shape)))
        domain = _ratio(_norm(s - jnp.clip(s, lo, hi)), scale)
        prohibited = jnp.where(~jnp.isfinite(hi), jnp.maximum(y, 0.0), 0.0)
        prohibited += jnp.where(~jnp.isfinite(lo), jnp.minimum(y, 0.0), 0.0)
        polar = _ratio(_norm(prohibited), _norm(y) if dual_scale is None else dual_scale)
        support = _support(block, y)
        complementarity = jnp.abs(support - jnp.vdot(s, y))
        return domain, polar, complementarity, support

    sr, yr = _rows(block, s), _rows(block, y)
    offset_norm = (jnp.asarray(0.0, s.dtype) if l is None
                   else _norm(_rows(block, l), axis=1))
    scale = jnp.maximum(_norm(_rows(block, kx), axis=1), offset_norm)
    if reference_scales is not None:
        scale = jnp.maximum(scale, reference_scales[2])
    normalized = sr / jnp.where(scale > 0, scale, 1.0)[:, None]
    violation = _cone_violation(block, normalized)
    violation = jnp.where(scale > 0, violation,
                           jnp.where(_norm(sr, axis=1) == 0, 0.0, jnp.inf))
    violation = jnp.where(jnp.isfinite(scale), violation, jnp.inf)
    ys = _norm(yr, axis=1)
    if reference_scales is not None:
        ys = jnp.maximum(ys, reference_scales[3])
    if dual_scale is not None:
        ys = jnp.maximum(ys, dual_scale)
    yn = yr / jnp.where(ys > 0, ys, 1.0)[:, None]
    polar = _cone_violation(block, yn, polar=True)
    polar = jnp.where(jnp.isfinite(ys), polar, jnp.inf)
    # Absolute values before summing prevent cancellation between cones.
    complementarity = jnp.sum(jnp.abs(jnp.sum(sr * yr, axis=1)))
    return (jnp.max(violation, initial=0.0), jnp.max(polar, initial=0.0),
            complementarity, jnp.asarray(0.0, s.dtype))


def _has_value(block, name):
    return _has_oracle(block, name)


def make_domain_checks(matvec, rmatvec, g_block, blocks, slices, *,
                       zero_offsets=None, zero_cost=False):
    """Return a jit/vmap-compatible check(x,y,(l,c)) function.

    Static attributes ``has_indicators`` and ``has_experimental_cones`` let the
    caller avoid this extra work on unconstrained families.  Unsupported custom
    blocks are left to the generic prox-graph criterion; this helper makes no
    claim about their domains.  Non-indicator finite values/conjugates are used
    for objective normalization when available.  Missing values set the returned
    ``objective_scale_available`` flag false rather than inventing a bound.
    """
    gi = _indicator(g_block)
    indicators = [_indicator(block) for block in blocks]
    has_indicators = gi is not None or any(block is not None for block in indicators)
    has_experimental = any(type(block) in _EXPERIMENTAL
                           for block in [gi] + indicators if block is not None)
    spans = tuple((int(a), int(b)) for a, b in slices)
    if len(blocks) != len(spans):
        raise ValueError("one slice is required per dual block")
    # These facts come from absent data callbacks, never array-value inference.
    has_structural_facts = zero_offsets is not None or zero_cost
    zero_offsets = ((False,) * len(spans) if zero_offsets is None
                    else tuple(zero_offsets))
    if len(zero_offsets) != len(spans):
        raise ValueError("one structural-zero flag is required per dual block")

    def check(x, y, data, primal_dual_scale=None, reference_scales=None,
              objective_reference=None):
        l, c = data
        zero = jnp.asarray(0.0, x.dtype)
        if not has_indicators:
            return {"r_domain": zero, "r_polar": zero, "r_complementarity": zero,
                    "objective_scale_available": jnp.asarray(False)}
        kx = matvec(x)
        kty = rmatvec(y)
        v = -(kty + c)
        cost_norm = zero if zero_cost else _norm(c)
        if primal_dual_scale is None:
            primal_dual_scale = jnp.maximum(_norm(kty), cost_norm)
        domain, polar, complementarity = zero, zero, zero
        primal, dual = jnp.vdot(c, x), jnp.vdot(l, y)
        available = jnp.asarray(True)
        objective_anchor = cost_norm * _norm(x)
        if objective_reference is not None:
            objective_anchor = jnp.maximum(objective_anchor, objective_reference)
        if gi is not None:
            dr, pr, cr, support = _block_checks(gi, x, v, x, None, primal_dual_scale)
            domain, polar, complementarity = dr, pr, cr
            dual -= support
        elif g_block is not None and type(g_block) is not Zero:
            if _has_value(g_block, "value") and _has_value(g_block, "conj_value"):
                pv, dv = g_block.value(x), g_block.conj_value(v)
                anchor = g_block.value(jnp.zeros_like(x))
                objective_anchor += jnp.where(jnp.isfinite(anchor), jnp.abs(anchor), 0.0)
                available &= jnp.isfinite(pv) & jnp.isfinite(dv)
                primal += jnp.where(jnp.isfinite(pv), pv, 0.0)
                dual -= jnp.where(jnp.isfinite(dv), dv, 0.0)
            else:
                available = jnp.asarray(False)
        for i, (block, indicator, (a, b)) in enumerate(zip(blocks, indicators, spans)):
            s, yi = kx[a:b] + l[a:b], y[a:b]
            if indicator is not None:
                refs = None if reference_scales is None else reference_scales[i]
                dr, pr, cr, support = _block_checks(
                    indicator, s, yi, kx[a:b],
                    None if zero_offsets[i] else l[a:b], reference_scales=refs)
                domain, polar = jnp.maximum(domain, dr), jnp.maximum(polar, pr)
                complementarity += cr
                dual -= support
            elif type(block) is not Zero:
                if _has_value(block, "value") and _has_value(block, "conj_value"):
                    pv, dv = block.value(s), block.conj_value(yi)
                    anchor = block.value(l[a:b])
                    objective_anchor += jnp.where(jnp.isfinite(anchor), jnp.abs(anchor), 0.0)
                    available &= jnp.isfinite(pv) & jnp.isfinite(dv)
                    primal += jnp.where(jnp.isfinite(pv), pv, 0.0)
                    dual -= jnp.where(jnp.isfinite(dv), dv, 0.0)
                else:
                    available = jnp.asarray(False)
        scale = jnp.maximum(objective_anchor, jnp.maximum(jnp.abs(primal), jnp.abs(dual)))
        cr = _ratio(complementarity, scale)
        if has_structural_facts:
            offsets_finite = jnp.asarray(True)
            for absent, (a, b) in zip(zero_offsets, spans):
                if not absent:
                    offsets_finite &= jnp.all(jnp.isfinite(l[a:b]))
        else:
            offsets_finite = jnp.all(jnp.isfinite(l))
        cost_finite = jnp.asarray(True) if zero_cost else jnp.all(jnp.isfinite(c))
        finite = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
                  & offsets_finite & cost_finite
                  & jnp.isfinite(objective_anchor) & jnp.isfinite(primal)
                  & jnp.isfinite(dual) & jnp.isfinite(complementarity))
        return {
            "r_domain": jnp.where(finite & jnp.isfinite(domain), domain, jnp.inf),
            "r_polar": jnp.where(finite & jnp.isfinite(polar), polar, jnp.inf),
            "r_complementarity": jnp.where(finite & ~jnp.isnan(cr), cr, jnp.inf),
            "objective_scale_available": available & finite,
        }

    check.has_indicators = has_indicators
    check.has_experimental_cones = has_experimental
    return check
