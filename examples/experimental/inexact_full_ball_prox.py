"""Experimental adaptive type-2 projection onto a product of Euclidean balls.

This helper does NOT modify OCS's public solver or the frozen fixed-20-step
metric pilots.  ``estimated_budget_met`` is a bounded floating-point diagnostic,
not an interval certificate or a finite-precision convergence theorem.

Contract: one fixed problem, fixed SPD metrics, exact other prox operations,
dual-first ordinary PDHG and its valid coupling bound, as specified in
profiling/research_execution_20260915/math/INEXACT_PROX_CONTRACT.md.  For every
actual returned feasible block y, the exact type-2 error is

    radius * ||Sigma^-1 (v-y)|| - <Sigma^-1 (v-y), y>.

Errors add over ALL inexact blocks. Equal-size blocks are batched here; callers
with several groups must split ONE global budget across those calls and sum
their errors. The global outer iteration must persist across JIT chunks and
restarts of execution; resetting it resets the summability contract. The error
budget is absolute, in objective units, and is not the relative KKT tolerance.

The fixed rounded factors define Sigma = Q diag(sigma) Q.T, including Q's small
orthogonality defect. A bisection root uses the idealized spectral formula; the
acceptance estimate instead evaluates the physical returned point and includes
an inverse-metric discrepancy estimate. Arithmetic allowances use the standard
normal-arithmetic gamma_n model, with small underflow guards, highest-precision
dot products and a conservative scaled norm allowance. They are deliberately
named *estimates*: XLA transformations, subnormal flushing and host preparation
are not enclosed by a validated interval-arithmetic implementation.

No budget is raised to a float32 floor. Unattainable active budgets return the
candidate and a resolution/cap/failure status. Exact zero-radius and definitely
inactive branches have algebraically zero type-2 error.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


ESTIMATED_BUDGET_MET = 0
ITERATION_LIMIT = 1
NUMERICAL_RESOLUTION = 2
INVALID_NUMERICS = 3
INFEASIBLE_CANDIDATE = 4
TOTAL_BUDGET_EXCEEDED = 5
STATUS_NAMES = {
    0: "estimated_budget_met", 1: "iteration_limit",
    2: "numerical_resolution", 3: "invalid_numerics",
    4: "infeasible_candidate", 5: "total_budget_exceeded",
}


class BallMetric(NamedTuple):
    Q: jax.Array
    sigma: jax.Array
    inverse_discrepancy: jax.Array
    orthogonality_estimate: jax.Array


class ErrorBudget(NamedTuple):
    total: jax.Array
    per_block: jax.Array
    requested_total: float
    requested_per_block: tuple[float, ...]


def _round_down(a, dtype):
    a64 = np.asarray(a, dtype=np.float64)
    out = a64.astype(dtype)
    return np.where(out.astype(np.float64) > a64,
                    np.nextafter(out, np.zeros_like(out)), out).astype(dtype)


def _round_up(a, dtype):
    a64 = np.asarray(a, dtype=np.float64)
    out = a64.astype(dtype)
    return np.nextafter(out, np.full_like(out, np.inf))


def prepare_metric(Q, sigma, *, dtype=np.float32,
                   orthogonality_limit=1e-3) -> BallMetric:
    """One-time host validation of the ACTUAL rounded fixed SPD factors.

    Shapes are (blocks, dimension, dimension), (blocks, dimension). The
    approximately orthogonal basis must be square, finite and full rank.
    The numerical inverse-discrepancy estimate is prepared in float64 and
    rounded outward into the runtime dtype. This is not an interval proof.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Only float32 and float64 are supported")
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before preparing a float64 metric")
    q = np.asarray(Q, dtype=dtype)
    s = np.asarray(sigma, dtype=dtype)
    if q.ndim != 3 or s.ndim != 2 or q.shape != (*s.shape, s.shape[-1]):
        raise ValueError("Expected Q=(B,d,d) and sigma=(B,d)")
    if not s.shape[0] or not s.shape[1]:
        raise ValueError("Empty products/blocks are not supported")
    if not np.isfinite(q).all() or not np.isfinite(s).all() or not (s > 0).all():
        raise ValueError("Finite Q and strictly positive finite eigenvalues required")
    if not 0 < orthogonality_limit < 1:
        raise ValueError("orthogonality_limit must lie strictly between 0 and 1")
    q64, s64 = q.astype(np.float64), s.astype(np.float64)
    d = s.shape[1]
    eps64 = np.finfo(np.float64).eps
    gamma = d * eps64 / (1 - d * eps64)
    gram = np.einsum("bji,bjk->bik", q64, q64)
    absolute_gram = np.einsum("bji,bjk->bik", abs(q64), abs(q64))
    beta = np.linalg.norm(gram - np.eye(d), axis=(1, 2))
    beta += (gamma + 32 * eps64) * np.linalg.norm(absolute_gram, axis=(1, 2))
    if (beta >= orthogonality_limit).any():
        raise ValueError("Rounded basis exceeds the declared orthogonality limit")
    # With G=Q.T Q, Dtrue=Q G^-1 diag(1/sigma) G^-1 Q.T.
    delta = ((1 + beta) * beta * (2 - beta) /
             ((1 - beta) ** 2 * np.min(s64, axis=1)))
    delta *= 1 + 32 * eps64
    if not np.isfinite(delta).all():
        raise ValueError("Inverse metric discrepancy estimate overflowed")
    cast_delta = _round_up(delta, dtype)
    if not np.isfinite(cast_delta).all():
        raise ValueError("Inverse metric estimate is not representable in runtime dtype")
    return BallMetric(jnp.asarray(q), jnp.asarray(s), jnp.asarray(cast_delta),
                      jnp.asarray(_round_up(beta, dtype)))


def prepare_budget(total, blocks, *, per_block=None, dtype=np.float32) -> ErrorBudget:
    """Validate requested budgets, then round/tighten them DOWN for reductions.

    Explicit per-block limits must sum to at most total. The effective limits
    are slightly smaller to account for the modeled positive-sum evaluation;
    both requested and effective budgets remain observable. No positive floor.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Only float32 and float64 are supported")
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before preparing float64 budgets")
    total = float(total)
    if blocks < 1 or not math.isfinite(total) or total < 0:
        raise ValueError("Nonnegative finite total and a positive block count required")
    requested = ([total / blocks] * blocks if per_block is None
                 else list(map(float, per_block)))
    if len(requested) != blocks or any(not math.isfinite(x) or x < 0 for x in requested):
        raise ValueError("One nonnegative finite budget per block required")
    sum_requested = math.fsum(requested)
    if per_block is not None and sum_requested > total:
        raise ValueError("Per-block allocations exceed the GLOBAL total budget")
    eps = np.finfo(dtype).eps
    if (blocks + 4) * eps >= .1:
        raise ValueError("Too many blocks for the declared reduction error model")
    gamma = (blocks + 4) * eps / (1 - (blocks + 4) * eps)
    shrink = min(1., total / (sum_requested * (1 + gamma))) if sum_requested else 1.
    effective = _round_down(np.asarray(requested) * shrink * (1 - 4 * eps), dtype)
    return ErrorBudget(jnp.asarray(_round_down(total, dtype)), jnp.asarray(effective),
                       total, tuple(requested))


def summable_budget(global_iteration: int, c: float, p: float) -> float:
    """Return c/(k+1)^(1+p); caller preserves global k across execution chunks.

    c,p are frozen positive parameters chosen before the solve. Underflow may
    return zero; it is deliberately not replaced by a representable floor.
    The caller must handle a later unmet inner budget as failure.
    """
    if not isinstance(global_iteration, (int, np.integer)) or global_iteration < 0:
        raise ValueError("global_iteration must be a persistent nonnegative integer")
    if not math.isfinite(c) or c <= 0 or not math.isfinite(p) or p <= 0:
        raise ValueError("c and p must be finite and positive")
    return math.exp(math.log(c) - (1 + p) * math.log(int(global_iteration) + 1))


def _norm(a):
    """Scaled Euclidean norm, with no divide-by-zero in the zero branch."""
    scale = jnp.max(jnp.abs(a), axis=-1)
    safe = jnp.where(scale > 0, scale, 1)
    return scale * jnp.sqrt(jnp.sum(jnp.square(a / safe[..., None]), axis=-1))


def _mv(Q, x):
    return jnp.einsum("bij,bj->bi", Q, x, precision=lax.Precision.HIGHEST)


def adaptive_ball_prox(metric: BallMetric, v, radius, total_budget,
                       per_block_budget, *, max_iterations=80):
    """Batched runtime helper; JIT with max_iterations static.

    Inputs v=(B,d), radius=(B,), and budgets=(scalar,(B,)) must match the
    prepared metric. A scalar radius is broadcast. Call prepare_budget first
    for validated allocations. Invalid runtime values fail without a success
    status. Return fields describe the same physical y supplied to PDHG.
    ``success`` means numerical-estimate acceptance only, including the SUM
    over all B blocks. It is never called ``certified``.
    """
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a finite nonnegative integer")
    Q, sigma, deltaD, _ = metric
    v = jnp.asarray(v, dtype=Q.dtype)
    B, d = sigma.shape
    if v.shape != (B, d):
        raise ValueError("v shape must match the fixed metric")
    radius = jnp.broadcast_to(jnp.asarray(radius, dtype=Q.dtype), (B,))
    total = jnp.asarray(total_budget, dtype=Q.dtype)
    budgets = jnp.asarray(per_block_budget, dtype=Q.dtype)
    if total.shape != () or budgets.shape != (B,):
        raise ValueError("Expected one scalar total and one budget per block")
    eps, tiny = jnp.finfo(Q.dtype).eps, jnp.finfo(Q.dtype).tiny
    gamma = lambda n: (n * eps) / (1 - n * eps)
    gn, gd, gb = gamma(4 * d + 16), gamma(d + 2), gamma(B + 4)
    valid = (jnp.all(jnp.isfinite(v), axis=1) & jnp.isfinite(radius) & (radius >= 0)
             & jnp.isfinite(budgets) & (budgets >= 0)
             & jnp.isfinite(total) & (total >= 0))
    # Inputs from prepare_budget have downward-rounded allocations. Runtime
    # callers bypassing preparation must still satisfy the total allocation.
    valid = valid & (jnp.sum(budgets) <= total)
    valid = valid & ((4 * d + 16) * eps < .1) & ((B + 4) * eps < .1)
    norm_v = _norm(v)
    norm_v_upper = norm_v * (1 + gn) + tiny
    zero = valid & (radius == 0)
    inactive = valid & (norm_v_upper <= radius)
    exact = zero | inactive
    QT, absQ = jnp.swapaxes(Q, 1, 2), jnp.abs(Q)
    absQT = jnp.swapaxes(absQ, 1, 2)
    w = _mv(QT, v)
    safe_radius = jnp.where(radius > 0, radius, 1)
    norm_w = _norm(w)
    # Exact-arithmetic analytic bracket, inflated under the disclosed norm model.
    hi0 = jnp.maximum(0, (norm_w * (1 + gn) / safe_radius - 1)) / jnp.min(sigma, axis=1)
    hi0 *= (1 + gn)
    bracket_valid = jnp.isfinite(hi0) & jnp.isfinite(norm_w)
    valid = valid & (exact | bracket_valid)
    hi0 = jnp.where(valid & ~exact, hi0, 0)

    def evaluate(eta):
        raw = _mv(Q, w / (1 + sigma * eta[:, None]))
        # Recheck and correct feasibility in physical coordinates, not eigenbasis.
        raw_upper = _norm(raw) * (1 + gn) + tiny
        factor = jnp.minimum(1, safe_radius / jnp.where(raw_upper > 0, raw_upper, 1))
        factor *= (1 - 2 * gn)
        y = raw * factor[:, None]
        y = jnp.where(inactive[:, None], v, y)
        y = jnp.where(zero[:, None], 0, y)
        yn = _norm(y)
        yn_upper = jnp.where(zero, 0, yn * (1 + gn) + tiny)
        feasible = valid & (yn_upper <= radius)
        u = v - y
        a = _mv(QT, u)
        b = a / sigma
        q = _mv(Q, b)
        # Modeled error in q_eval from subtraction, two matrix products and division.
        uerr = eps * (jnp.abs(v) + jnp.abs(y)) + tiny
        aerr = gd * _mv(absQT, jnp.abs(u)) + _mv(absQT, uerr) + tiny
        berr = (aerr + eps * jnp.abs(a)) / sigma + tiny
        qerr = gd * _mv(absQ, jnp.abs(b)) + _mv(absQ, berr) + tiny
        true_u_norm_upper = (_norm(u) + _norm(uerr)) * (1 + gn) + tiny
        xi = (_norm(qerr) + deltaD * true_u_norm_upper) * (1 + gn) + tiny
        qn = _norm(q)
        dot = jnp.sum(q * y, axis=1)
        gap = radius * qn - dot
        eval_allowance = (gn * radius * qn + gd * jnp.sum(jnp.abs(q * y), axis=1)
                          + eps * (radius * qn + jnp.abs(dot)) + tiny)
        correction = (radius + yn_upper) * xi
        # A negative rounded raw gap is never accepted without these allowances.
        error = (jnp.maximum(gap, 0) + eval_allowance + correction) * (1 + gn) + tiny
        error = jnp.where(exact, 0, error)
        gap = jnp.where(exact, 0, gap)
        finite = (jnp.all(jnp.isfinite(y), axis=1) & jnp.isfinite(error)
                  & jnp.isfinite(yn_upper))
        error = jnp.where(feasible & finite, error, jnp.inf)
        return y, error, gap, yn_upper, feasible, finite, xi, eval_allowance

    initial = evaluate(hi0)
    # Pending blocks stop only when their OWN limit is met. The positive-sum
    # error allowance is checked again globally below.
    pending0 = valid & ~exact & (initial[1] > budgets)
    state0 = (jnp.zeros_like(hi0), hi0, jnp.zeros(B, jnp.int32),
              jnp.zeros(B, bool), pending0, jnp.array(0, jnp.int32))

    def condition(state):
        return (state[-1] < max_iterations) & jnp.any(state[-2])

    def step(state):
        lo, hi, counts, stalled, pending, sweep = state
        mid = lo + (hi - lo) * .5
        resolution = (mid == lo) | (mid == hi)
        active = pending & ~resolution
        mid_norm = _norm(w / (1 + sigma * mid[:, None]))
        is_high = mid_norm * (1 + gn) <= radius
        new_hi = jnp.where(active & is_high, mid, hi)
        new_lo = jnp.where(active & ~is_high, mid, lo)
        info = evaluate(new_hi)
        new_stalled = stalled | (pending & resolution)
        new_pending = valid & ~exact & (info[1] > budgets) & ~new_stalled & info[5]
        return (new_lo, new_hi, counts + active.astype(jnp.int32), new_stalled,
                new_pending, sweep + 1)

    lo, hi, iterations, stalled, pending, sweeps = lax.while_loop(condition, step, state0)
    y, error, gap, yn_upper, feasible, finite, xi, allowance = evaluate(hi)
    per_block_met = valid & feasible & finite & (error <= budgets)
    total_estimate = jnp.sum(error) * (1 + gb)
    success = jnp.all(per_block_met) & jnp.isfinite(total_estimate) & (total_estimate <= total)
    status = jnp.where(per_block_met, ESTIMATED_BUDGET_MET,
             jnp.where(~valid | ~finite, INVALID_NUMERICS,
             jnp.where(~feasible, INFEASIBLE_CANDIDATE,
             jnp.where(stalled, NUMERICAL_RESOLUTION, ITERATION_LIMIT)))).astype(jnp.int32)
    global_status = jnp.where(success, ESTIMATED_BUDGET_MET,
                    jnp.where(jnp.all(per_block_met), TOTAL_BUDGET_EXCEEDED,
                              jnp.max(status))).astype(jnp.int32)
    return dict(y=y, success=success, block_status=status, per_block_budget_met=per_block_met,
                global_status=global_status,
                error_upper_estimate=error, total_error_upper_estimate=total_estimate,
                requested_runtime_total=total, effective_per_block_budget=budgets,
                raw_support_defect=gap, physical_norm_upper_estimate=yn_upper,
                feasible_estimate=feasible, inverse_action_error_estimate=xi,
                support_evaluation_allowance=allowance, iterations=iterations,
                eta_low=lo, eta_high=hi, sweeps=sweeps, exact_branch=exact)
