"""Numerical domain-aware Farkas and recession checks using operator products.

For x in G and Kx+l in C, a separating direction d satisfies
    support_G(-K.T d) + support_C(d) < l.T d.
For cones, the supports are zero on the polar and infinite elsewhere.  A
large iterate norm or a cosine with the data is not a certificate.

Recession checks here cover linear objectives with explicit cone/box domains.
They additionally require a numerically feasible base point.  These are
floating-point certificates at the requested tolerance, not exact arithmetic
proofs; unsupported domains/objective recession functions return false.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..prox.base import _AffinePre, _LinQuad, _Scaled
from ..prox.cones import (PSDCone, SecondOrderCone, SOCBatch,
                          ExpCone, ExpConeBatch, PowerCone, PowerConeBatch)
from .domain_checks import _rows, _cone_violation
from ..prox.library import (
    Box, Equality, GroupL2, L1, L2, Logistic, NonNeg, PiecewiseLinearConvex,
    Pinball, Quadratic, SumSquares, Zero,
)
from .equilibrate import _power_norm
from .._operands import operand_jit


_NONLINEAR_CONES = (ExpCone, ExpConeBatch, PowerCone, PowerConeBatch)
_CONES = (Equality, NonNeg, SecondOrderCone, SOCBatch, PSDCone, *_NONLINEAR_CONES)
_FULL_DOMAIN = (Zero, SumSquares, Quadratic, L1, L2, GroupL2, Pinball,
                Logistic, PiecewiseLinearConvex)


def _domain(block):
    """Return a domain descriptor and whether the objective is its indicator."""
    if block is None or type(block) is Zero:
        return ("full", None), True
    if type(block) in (PowerCone, PowerConeBatch):
        if not math.isfinite(block.alpha) or not 0 < block.alpha < 1:
            return ("unsupported", None), False
    if type(block) in _CONES:
        return ("cone", block), True
    if type(block) is Box:
        lo, hi = np.asarray(block.lo), np.asarray(block.hi)
        if (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
                and np.all(lo <= hi)):
            return ("box", block), True
        return ("unsupported", None), False
    if type(block) in _FULL_DOMAIN:
        return ("full", None), False
    if type(block) is _Scaled and math.isfinite(block.c) and block.c > 0:
        return _domain(block.base)
    if type(block) is _LinQuad:
        descriptor, _ = _domain(block.base)
        # Adding finite linear/quadratic terms preserves the domain, but the
        # objective recession function is deliberately outside this checker.
        return descriptor, False
    if type(block) is _AffinePre:
        descriptor, _ = _domain(block.base)
        if descriptor[0] == "full":
            return descriptor, False
    return ("unsupported", None), False


def _norm(v):
    # Scaling first avoids squaring very large divergence iterates in float32.
    scale = jnp.max(jnp.abs(v), initial=0.0)
    return jnp.where(scale > 0, scale * jnp.linalg.norm(v / jnp.where(
        scale > 0, scale, 1.0)), 0.0)


def _ratio(value, scale):
    return jnp.where(scale > 0, value / jnp.where(scale > 0, scale, 1.0),
                     jnp.where(value == 0, 0.0, jnp.inf))


def _project_domain(descriptor, v):
    kind, block = descriptor
    return block.prox(v, jnp.asarray(1.0, v.dtype)) if kind != "full" else v


def _support_direction(descriptor, v):
    """Project onto directions with finite domain support."""
    kind, block = descriptor
    if kind == "full":
        return jnp.zeros_like(v)
    if kind == "cone":
        return v - block.prox(v, jnp.asarray(1.0, v.dtype))
    return v                              # every direction supports a finite box


def _nonlinear_cone_distance(block, v, *, polar=False):
    """Homogeneous feasible-witness bounds independent of the cone projector.

    For the exponential cone, either lower its first coordinate at positive
    second/third coordinates, or use the closed face (r<=0,s=0,t>=0).
    Polar witnesses use (v1,v0,-e*v2) in the primal exponential cone and are
    measured back in the original coordinates. Power witnesses share the
    independent lower-third/raise-first/raise-second bounds used by stopping.
    These are conservative numerical distances, not exact metric projections.
    """
    rows = _rows(block, v)
    scale = jnp.max(jnp.abs(rows), axis=1, initial=0.0)
    unit = rows / jnp.where(scale > 0, scale, 1.0)[:, None]
    if type(block) in (ExpCone, ExpConeBatch):
        if polar:
            r, s, t = unit[:, 1], unit[:, 0], -jnp.e * unit[:, 2]
            closed = jnp.stack([s, jnp.maximum(r, 0.0),
                                jnp.maximum(-t, 0.0) / jnp.e], axis=1)
        else:
            r, s, t = unit[:, 0], unit[:, 1], unit[:, 2]
            closed = jnp.stack([jnp.maximum(r, 0.0), s,
                                jnp.maximum(-t, 0.0)], axis=1)
        closed_distance = jnp.linalg.norm(closed, axis=1)
        tiny = jnp.finfo(v.dtype).tiny
        interior = (s >= tiny) & (t >= tiny)
        bound = s * (jnp.log(jnp.maximum(t, tiny)) -
                     jnp.log(jnp.maximum(s, tiny)))
        interior_distance = jnp.where(interior,
            jnp.maximum(r - bound, 0.0), jnp.inf)
        distance = jnp.minimum(closed_distance, interior_distance)
    else:
        distance = _cone_violation(block, unit, polar=polar)
    distance = jnp.where(jnp.isfinite(scale), scale * distance, jnp.inf)
    return _norm(distance)


def _domain_distance(descriptor, v):
    if descriptor[0] == "cone" and type(descriptor[1]) in _NONLINEAR_CONES:
        return _nonlinear_cone_distance(descriptor[1], v)
    return _norm(v - _project_domain(descriptor, v))


def _support(descriptor, v):
    """Support value plus distance to its domain; never clips infinities."""
    kind, block = descriptor
    if kind == "full":
        return jnp.asarray(0.0, v.dtype), _norm(v)
    if kind == "cone":
        distance = (_nonlinear_cone_distance(block, v, polar=True)
                    if type(block) in _NONLINEAR_CONES else _norm(block.prox(v, 1.0)))
        return jnp.asarray(0.0, v.dtype), distance
    value = jnp.sum(jnp.maximum(jnp.asarray(block.lo) * v,
                                jnp.asarray(block.hi) * v))
    return value, jnp.asarray(0.0, v.dtype)


def _domain_scale(descriptor, v):
    if descriptor[0] != "box":
        return jnp.asarray(0.0, v.dtype)
    block = descriptor[1]
    bounds = jnp.maximum(jnp.abs(jnp.asarray(block.lo)),
                         jnp.abs(jnp.asarray(block.hi)))
    return _norm(jnp.broadcast_to(bounds, v.shape))


def _recession_distance(descriptor, v):
    if descriptor[0] == "box":
        return _norm(v)                    # finite box has recession cone {0}
    return _domain_distance(descriptor, v)


def make_status_checks(matvec, rmatvec, g_block, blocks, slices, n, m,
                       tol=1e-6, *, operator_norm=None):
    """Build check(x, y, data=(l,c)), usable with jit or vmap.

    Supported domains: Equality, NonNeg, finite Box, SOC/SOCBatch, PSD, and
    exponential/power cones (single or batched), and known full-domain library
    objectives (including positive scalings and finite linear/quadratic
    additions). Unknown custom/affine-precomposed domains remain unsupported.
    Exponential/power support, recession and base checks use independent
    feasible-witness bounds after projector-based direction repair.
    Unboundedness is checked only
    when all objectives apart from c.T x are indicators or Zero.

    An optional existing raw operator-norm estimate avoids another setup
    pass.  Otherwise eight matrix-free power steps estimate the normalization
    scale.  No entries, rows, columns, or system matrices are inspected.
    """
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("certificate tolerance must be positive and finite")
    if len(blocks) != len(slices):
        raise ValueError("one slice is required per dual block")
    gd, g_linear = _domain(g_block)
    described = [_domain(block) for block in blocks]
    domains = [item[0] for item in described]
    zero_support_direction = all(domain[0] == "full" for domain in domains)
    supported = gd[0] != "unsupported" and all(
        domain[0] != "unsupported" for domain in domains)
    recession_supported = supported and g_linear and all(
        item[1] for item in described)
    spans = tuple((int(a), int(b)) for a, b in slices)
    nK = (jnp.asarray(operator_norm, jnp.float32) if operator_norm is not None
          else _power_norm(matvec, rmatvec, n, iters=8)) if supported else None

    @operand_jit
    def check(x, y, data):
        l, c = data
        zero = jnp.asarray(0.0, x.dtype)
        infinity = jnp.asarray(jnp.inf, x.dtype)
        result = {
            "primal_infeasible": jnp.asarray(False),
            "unbounded": jnp.asarray(False),
            "primal_infeasibility_supported": jnp.asarray(supported),
            "recession_supported": jnp.asarray(recession_supported),
            "dual_direction": jnp.zeros((m,), y.dtype),
            "primal_direction": jnp.zeros((n,), x.dtype),
            "feasible_base": jnp.zeros((n,), x.dtype),
            "primal_infeasibility_residual": infinity,
            "primal_infeasibility_margin": zero,
            "recession_residual": infinity,
            "recession_margin": zero,
            "base_feasibility_residual": infinity,
        }
        if not supported:
            return result

        finite = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
                  & jnp.all(jnp.isfinite(l)) & jnp.all(jnp.isfinite(c))
                  & jnp.isfinite(nK) & (nK >= 0))
        # Repair the proposed ray only within each support domain; feasibility
        # of the *coupled* primal support direction is then checked separately.
        d = jnp.zeros_like(y)
        for descriptor, (a, b) in zip(domains, spans):
            d = d.at[a:b].set(_support_direction(descriptor, y[a:b]))
        dn = _norm(d)
        d = d / jnp.where(dn > 0, dn, 1.0)
        # Full domains admit only the zero support direction. Use linearity
        # instead of tracing a potentially large constant-zero adjoint (FFT).
        ktd = jnp.zeros((n,), x.dtype) if zero_support_direction else rmatvec(d)
        support_g, violation_g = _support(gd, -ktd)
        support_c, support_abs = zero, jnp.abs(support_g)
        ray_residual = _ratio(violation_g, nK)
        for descriptor, (a, b) in zip(domains, spans):
            value, violation = _support(descriptor, d[a:b])
            support_c += value
            support_abs += jnp.abs(value)
            ray_residual = jnp.maximum(ray_residual, violation)
        separation = jnp.dot(l, d) - support_g - support_c
        separation_scale = jnp.maximum(_norm(l), support_abs)
        margin = _ratio(separation, separation_scale)
        result.update({
            "dual_direction": d,
            "primal_infeasibility_residual": ray_residual,
            "primal_infeasibility_margin": margin,
            "primal_infeasible": (finite & (dn > 0) & (ray_residual <= tol)
                                  & jnp.isfinite(margin) & (margin > tol)),
        })
        if not recession_supported:
            return result

        # The current primal iterate proposes a direction, but is not assumed
        # feasible.  A separate base point is mandatory; zero is often a known
        # feasible point on homogeneous LPs and avoids iterate drift there.
        xn = _norm(x)
        e = x / jnp.where(xn > 0, xn, 1.0)
        ke = matvec(e)
        recession_residual = _recession_distance(gd, e)
        for descriptor, (a, b) in zip(domains, spans):
            distance = _recession_distance(descriptor, ke[a:b])
            recession_residual = jnp.maximum(recession_residual,
                                              _ratio(distance, nK))

        def base_residual(base, kb):
            scale = jnp.maximum(_norm(base), _domain_scale(gd, base))
            residual = _ratio(_domain_distance(gd, base), scale)
            for descriptor, (a, b) in zip(domains, spans):
                s = kb[a:b] + l[a:b]
                scale = jnp.maximum(jnp.maximum(_norm(kb[a:b]), _norm(l[a:b])),
                                     _domain_scale(descriptor, s))
                distance = _domain_distance(descriptor, s)
                residual = jnp.maximum(residual, _ratio(distance, scale))
            return residual

        zero_base_residual = base_residual(jnp.zeros_like(x), jnp.zeros_like(l))
        # Re-evaluate the actual returned point rather than reconstructing Kx
        # from the normalized ray; those paths can round differently in f32.
        current_base_residual = base_residual(x, matvec(x))
        base_error = jnp.minimum(zero_base_residual, current_base_residual)
        base = jnp.where(zero_base_residual <= current_base_residual,
                         jnp.zeros_like(x), x)
        descent_margin = _ratio(-jnp.dot(c, e), _norm(c))
        result.update({
            "primal_direction": e,
            "feasible_base": base,
            "recession_residual": recession_residual,
            "recession_margin": descent_margin,
            "base_feasibility_residual": base_error,
            "unbounded": (finite & (xn > 0) & (recession_residual <= tol)
                          & (base_error <= tol) & jnp.isfinite(descent_margin)
                          & (descent_margin > tol)
                          & ~result["primal_infeasible"]),
        })
        return result

    # A full dual domain admits only the zero support direction.  When
    # recession checking is unavailable as well, both exceptional status
    # flags are statically false.  Status-only callers can avoid tracing
    # large constant-zero diagnostics; the full report remains unchanged.
    check.status_may_certify = supported and (not zero_support_direction or recession_supported)
    return check
