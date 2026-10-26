"""Operator-only stopping evaluated at a primal-dual point.

``iterate`` and ``data`` retain the historical fixed-point normalizations.
``relative`` measures the proximal graph mismatch in the original primal
units and stationarity in gradient units. Its feasibility denominator excludes
the dual multiplier, so increasing a multiplier cannot hide a violated
constraint. All modes share their evaluator with certificate reporting.

These are numerical optimality residuals, not bounds on objective error for an
arbitrary convex function. Relative mode is supplemented by domain and gap
checks by the compiler when the block contracts support them.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..prox.library import Zero
from .domain_checks import _indicator
from .._operands import set_window


NORMS = ("iterate", "data", "relative")


def stable_norm(value):
    """Euclidean norm without squaring the original float32 magnitudes."""
    scale = jnp.max(jnp.abs(value), initial=0.0)
    divisor = jnp.where(scale > 0, scale, 1.0)
    return scale * jnp.linalg.norm(value / divisor)


def relative_ratio(numerator, denominator):
    """Pure relative ratio, with 0/0 = 0 and nonzero/0 = infinity.

    There is no additive unit floor. Homogeneous zero-data problems may need
    exact zeros; callers must not silently weaken this with an absolute unit.
    Nonfinite arithmetic never passes a stopping test.
    """
    valid = jnp.isfinite(numerator) & jnp.isfinite(denominator)
    ratio = numerator / jnp.where(denominator > 0, denominator, 1.0)
    ratio = jnp.where(denominator > 0, ratio,
                      jnp.where(numerator == 0, 0.0, jnp.inf))
    return jnp.where(valid, ratio, jnp.inf)


def make_dual_repair(matvec, smooth_terms=(), offset_index=0):
    """Recover smooth dual blocks from x; the solver returns this same dual."""
    smooth = tuple((int(a), int(b), g) for a, b, g in smooth_terms)

    def repair(x, y, data):
        if not smooth:
            return y
        supplied_finite = jnp.all(jnp.isfinite(y))
        original = y
        s = matvec(x) + data[offset_index]
        for a, b, grad in smooth:
            y = set_window(y, a, b, grad(s[a:b]))
        return jnp.where(supplied_finite, y, original)

    return repair


def _balanced_graph(block, point, dual, scale, dual_scale=None):
    """Direct prox graph test, with a certificate metric independent of PDHG.

    For positive primal/dual scales, t=P/Y transforms as a^2/gamma under
    f'(u)=gamma*f(u/a), point'=a*point, dual'=gamma*dual/a.
    With a zero scale, a fallback metric is used only for exact membership:
    zero displacement passes; every nonzero displacement reports infinity.
    """
    yn = stable_norm(dual) if dual_scale is None else dual_scale
    active = (scale > 0) & (yn > 0)
    step = jnp.where(active, scale / jnp.where(yn > 0, yn, 1.0), 1.0)
    projected = block.prox(point + step * dual, step)
    distance = stable_norm(point - projected)
    error = jnp.where(active, relative_ratio(distance, scale),
                      jnp.where(distance == 0, 0.0, jnp.inf))
    valid = (jnp.isfinite(scale) & jnp.isfinite(yn)
             & jnp.isfinite(step) & (step > 0)
             & jnp.isfinite(distance))
    error = jnp.where(valid, error, jnp.inf)
    return error, step, active


def make_relative_components(matvec, rmatvec, g_block, blocks, slices,
                             adjoint_with_scale=None, cone_reference_scales=None,
                             zero_offsets=None, zero_cost=False):
    """Numerical KKT checks defined by the problem and supplied point alone.

    No iteration metric, equilibration, or progress history enters this
    evaluator. Nonsmooth graph checks evaluate primal proxes directly to avoid
    cancellation in y-prox_fstar(y). Smooth blocks use their gradient equation.
    Data reference subgradients transform in the same units as the supplied
    dual; nonsmooth references use the same balanced certificate step.
    """
    spans = tuple(slices)
    # Compiler-supplied structural facts only. A supplied callback returning
    # numerical zeros must retain the ordinary norm and finite checks.
    has_structural_facts = zero_offsets is not None or zero_cost
    zero_offsets = ((False,) * len(spans) if zero_offsets is None
                    else tuple(zero_offsets))
    if len(zero_offsets) != len(spans):
        raise ValueError("one structural-zero flag is required per dual block")
    if type(g_block) is Zero:
        g_block = None

    def pullback(y):
        if adjoint_with_scale is not None:
            return adjoint_with_scale(y)
        value = rmatvec(y)
        return value, stable_norm(value)

    def components(x, y, data):
        l, c = data
        cone_scales, objective_reference = (cone_reference_scales(data)
            if cone_reference_scales is not None else (None, None))
        kx = matvec(x)
        kty, pullback_scale = pullback(y)
        reference = jnp.zeros_like(y)
        rd = jnp.asarray(0.0, x.dtype)
        for i, (blk, (a, b)) in enumerate(zip(blocks, spans)):
            s, yi, li = kx[a:b] + l[a:b], y[a:b], l[a:b]
            li_norm = jnp.asarray(0.0, li.dtype) if zero_offsets[i] else stable_norm(li)
            grad = getattr(blk, "smooth_grad", None)
            if grad is not None:
                current, anchor = grad(s), grad(li)
                scale = jnp.maximum(stable_norm(yi), jnp.maximum(
                    stable_norm(current), stable_norm(anchor)))
                error = relative_ratio(stable_norm(yi - current), scale)
            else:
                scale = jnp.maximum(stable_norm(kx[a:b]), li_norm)
                dual_scale = stable_norm(yi)
                if cone_scales is not None:
                    scale = jnp.maximum(scale, cone_scales[i][0])
                    dual_scale = jnp.maximum(dual_scale, cone_scales[i][1])
                error, step, active = _balanced_graph(blk, s, yi, scale, dual_scale)
                if _indicator(blk) is not None:
                    # Zero is a normal at every point of a convex set. With
                    # y=0, projection gives witness (z,0), with no dual error.
                    projected = blk.prox(s, jnp.asarray(1.0, s.dtype))
                    feasibility = relative_ratio(stable_norm(s - projected), scale)
                    error = jnp.where(stable_norm(yi) == 0, feasibility, error)
                anchor = jnp.where(active, (li - blk.prox(li, step)) / step, 0.0)
            reference = set_window(reference, a, b, anchor)
            rd = jnp.maximum(rd, error)

        q = kty + c
        _ref, reference_scale = pullback(reference)
        cost_norm = jnp.asarray(0.0, c.dtype) if zero_cost else stable_norm(c)
        stationarity_scale = jnp.maximum(jnp.maximum(pullback_scale, reference_scale),
                                          cost_norm)
        grad_g = getattr(g_block, "smooth_grad", None)
        if g_block is not None and grad_g is None:
            # Implicit identity-block dual is -q; its graph is the remaining
            # KKT equation when a primal term is routed out of the dual stack.
            rp, _step, _active = _balanced_graph(
                g_block, x, -q, stable_norm(x), stationarity_scale)
            if _indicator(g_block) is not None:
                projected = g_block.prox(x, jnp.asarray(1.0, x.dtype))
                feasibility = relative_ratio(stable_norm(x - projected), stable_norm(x))
                rp = jnp.where(stable_norm(q) == 0, feasibility, rp)
        else:
            gx = jnp.zeros_like(x) if grad_g is None else grad_g(x)
            g0 = jnp.zeros_like(x) if grad_g is None else grad_g(jnp.zeros_like(x))
            scale = jnp.maximum(stationarity_scale, jnp.maximum(
                stable_norm(gx), stable_norm(g0)))
            rp = relative_ratio(stable_norm(q + gx), scale)
        if has_structural_facts:
            offsets_finite = jnp.asarray(True)
            for absent, (a, b) in zip(zero_offsets, spans):
                if not absent:
                    offsets_finite &= jnp.all(jnp.isfinite(l[a:b]))
        else:
            offsets_finite = jnp.all(jnp.isfinite(l))
        cost_finite = jnp.asarray(True) if zero_cost else jnp.all(jnp.isfinite(c))
        finite = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
                  & offsets_finite & cost_finite & jnp.isfinite(stationarity_scale))
        result = {
            "r_primal": jnp.where(finite & ~jnp.isnan(rp), rp, jnp.inf),
            "r_dual": jnp.where(finite & ~jnp.isnan(rd), rd, jnp.inf),
            "_stationarity_scale": stationarity_scale,
        }
        if cone_scales is not None:
            result["_cone_reference_scales"] = cone_scales
            result["_objective_reference"] = objective_reference
        return result

    return components


def make_residual_components(matvec, rmatvec, prox_primal, prox_dual,
                             tau, sigma, *, norm="iterate", slices=(),
                             smooth_terms=(), offset_index=0, primal_grad=None):
    """Return the shared residual dictionary for the supplied (x, y).

    This function never replaces y. A solve may recover smooth dual blocks
    before calling it, but certificate(x, y) always checks the actual pair.
    This factory retains historical iterate/data modes only. The relative
    contract is built by make_relative_components and receives no solver steps.
    """
    if norm not in ("iterate", "data"):
        raise ValueError("use make_relative_components for step-independent relative stopping")

    def components(x, y, data):
        l, c = data[offset_index], data[1 - offset_index]
        Kx = matvec(x)
        KTy = rmatvec(y)
        xp = prox_primal(x - tau * KTy, tau, data)
        yp = prox_dual(y + sigma * Kx, sigma, data)
        dx, dy = x - xp, y - yp
        if norm == "iterate":
            rp = stable_norm(dx) / (1.0 + stable_norm(x))
            rd = stable_norm(dy) / (1.0 + stable_norm(y))
        elif norm == "data":
            dp = (stable_norm(x) + stable_norm(tau * KTy)
                  + stable_norm(tau * c))
            dd = (stable_norm(y) + stable_norm(sigma * Kx)
                  + stable_norm(sigma * l))
            rp = relative_ratio(stable_norm(dx), dp)
            rd = relative_ratio(stable_norm(dy), dd)
        finite = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
                  & jnp.all(jnp.isfinite(l)) & jnp.all(jnp.isfinite(c)))
        rp = jnp.where(finite & jnp.isfinite(rp), rp, jnp.inf)
        rd = jnp.where(finite & jnp.isfinite(rd), rd, jnp.inf)
        return {"r_primal": rp, "r_dual": rd}

    return components


def make_residual(matvec, rmatvec, prox_primal, prox_dual, tau, sigma,
                  smooth_terms=(), offset_index=0, norm="iterate", slices=()):
    components = make_residual_components(
        matvec, rmatvec, prox_primal, prox_dual, tau, sigma, norm=norm,
        slices=slices, smooth_terms=smooth_terms, offset_index=offset_index)
    repair = make_dual_repair(matvec, smooth_terms, offset_index)

    def residual(x, y, data):
        values = components(x, repair(x, y, data), data)
        return jnp.maximum(values["r_primal"], values["r_dual"])

    return residual


def make_residual_param(matvec, rmatvec, prox_primal, prox_dual,
                        smooth_terms=(), offset_index=0, norm="iterate", slices=()):
    """The same evaluator with runtime metrics for experimental online runs."""
    if norm not in NORMS:
        raise ValueError(f"residual_norm must be one of {NORMS}, got {norm!r}")

    def residual(x, y, data, tau, sigma):
        evaluate = make_residual(matvec, rmatvec, prox_primal, prox_dual,
                                  tau, sigma, smooth_terms=smooth_terms,
                                  offset_index=offset_index, norm=norm, slices=slices)
        return evaluate(x, y, data)

    return residual
