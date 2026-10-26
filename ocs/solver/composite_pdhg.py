"""The OCS engine: operator-pure, vmap-able adaptive-restart PDHG (float32).

Solves the saddle problem behind  min_x c^T x + g(x) + sum_i f_i(K_i x + l_i)
through matvec/rmatvec of the stacked operator K and the block proxes only --
never an entry of K.  Halpern anchoring + averaged restart + primal-weight
(omega) update.

The whole solve is a per-instance ``lax.while_loop`` with a per-instance
convergence test, so ``jax.vmap`` batches it and masks instances that have
converged.  This is the ONE solver in OCS; no other method is substituted.

Steps ``tau`` (n,) and ``sigma`` (m,) may be per-coordinate (diagonal matrix-free
equilibration).  Stopping uses a matrix-free primal + dual fixed-point residual
(``residual(x, y, data)``), each computed from a constant number of oracle calls.
Per-instance data enters through ``data = (l, c)`` (offset and cost).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .equilibrate import _power_norm
from .._operands import operand_jit


def _all_finite(*arrays):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in arrays]))


def _checked_while(check, advance, data, x0, y0, aux0, cap, tol, *,
                   aux_arrays=lambda aux: (), join=lambda y: y, split=lambda y: y):
    """The solve loop with ONE residual evaluation per check.

    ``check(x, y, data) -> (residual, y_checked)`` is evaluated once per check,
    at the iterate the loop holds.  When that iterate passes, the loop stops and
    returns it, with the checked dual and the residual that was tested, so the
    value that stops the run is the value reported: terminate when the
    candidate's residuals are small and emit that candidate.  Otherwise
    ``advance(x, y, aux, blocks) -> (x, y, aux)`` runs one block.  A point that
    is not finite, or whose residual is NaN, halts without a block.  At the
    budget the returned iterate is checked once, after the loop, and that one
    evaluation is reported.  ``join``/``split`` map a structured dual state to
    the flat dual the check takes.
    """
    def checked(x, flat):
        r, y_chk = check(x, flat, data)
        return jnp.asarray(r, jnp.float32), y_chk.astype(flat.dtype)

    def body(carry):
        x, y, aux, blocks, _res, _done = carry
        flat = join(y)
        r, y_chk = checked(x, flat)
        finite = _all_finite(x, flat, *aux_arrays(aux),
                             *jax.tree_util.tree_leaves(data))
        passed = finite & (r <= tol)
        halt = passed | ~finite | jnp.isnan(r)

        def stop(_):
            return x, split(jnp.where(passed, y_chk, flat)), aux, blocks

        def go(_):
            xn, yn, auxn = advance(x, y, aux, blocks)
            return xn, yn, auxn, blocks + 1

        xn, yn, auxn, bn = lax.cond(halt, stop, go, None)
        return (xn, yn, auxn, bn, jnp.where(finite, r, jnp.nan), passed)

    def cond(carry):
        return (carry[3] < cap) & ~carry[5] & ~jnp.isnan(carry[4])

    init = (x0, y0, aux0, jnp.asarray(0), jnp.asarray(jnp.inf, jnp.float32),
            jnp.asarray(False))
    x, y, _aux, blocks, res, done = lax.while_loop(cond, body, init)
    res_out, y_out = lax.cond(done, lambda _: (res, join(y)),
                              lambda _: checked(x, join(y)), None)
    return x, y_out, blocks, res_out


def make_solver(matvec, rmatvec, tau, sigma, prox_primal, prox_dual, residual,
                n, m, *, block=100, max_blocks=200, tol=1e-3, theta=0.5,
                adaptive_omega=True, variant="halpern", dual_slices=None, loop_unroll=1,
                batched=True, adaptive_blocks=None, budget_arg=False, check=None):
    """adaptive_blocks: when set, the primal weight adapts only during the
    first that many blocks and is then frozen, which restores the fixed metric
    for the tail; None adapts throughout.

    check: when given, ``check(x, y, data) -> (residual, y_checked)`` replaces
    ``residual`` as the stopping test and is evaluated once per check (see
    ``_checked_while``); the solver then returns the checked dual and the
    tested residual.  When None the loop stops on ``residual`` as before."""
    if variant == "standard":
        if adaptive_omega:
            raise ValueError("standard PDHG requires fixed steps")
        return _make_standard_solver(matvec, rmatvec, tau, sigma, prox_primal,
                                     prox_dual, residual, n, m, block=block,
                                     max_blocks=max_blocks, tol=tol, dual_slices=dual_slices,
                                     loop_unroll=loop_unroll, batched=batched,
                                     budget_arg=budget_arg, check=check)
    if not batched:
        raise ValueError("unbatched engine is available only for standard PDHG")
    if loop_unroll != 1:
        raise ValueError("nondefault loop_unroll requires standard PDHG")
    if variant != "halpern":
        raise ValueError("variant must be 'halpern' or 'standard'")
    def solve(data, x0, y0, cap=max_blocks):
        def run_block(x, y, omega):
            xa, ya = x, y
            sw = sigma * omega
            tw = tau / omega

            def step(k, st):
                x, y, xbar, xs, ys = st
                y_new = prox_dual(y + sw * matvec(xbar), sw, data)
                x_new = prox_primal(x - tw * rmatvec(y_new), tw, data)
                lam = 1.0 / (k.astype(jnp.float32) + 2.0)          # Halpern anchor
                xn = lam * xa + (1.0 - lam) * x_new
                yn = lam * ya + (1.0 - lam) * y_new
                return (xn, yn, 2.0 * xn - x, xs + xn, ys + yn)

            x, y, _xb, xs, ys = lax.fori_loop(
                0, block, step, (x, y, x, jnp.zeros(n), jnp.zeros(m)))
            return xs / block, ys / block                          # block average

        def advance(x, y, omega, blocks):
            xa, ya = x, y
            x_avg, y_avg = run_block(x, y, omega)
            if adaptive_omega:                      # PDLP-style primal-weight update
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                # Safeguards: at most a factor of two per block and a bounded
                # range, so a block with a tiny primal or dual move cannot
                # throw the weight to an extreme (measured: unbounded updates
                # diverged on the aggregated scheduling LP at 1e6 tasks).
                omega_new = (dy / dx) ** theta * omega ** (1.0 - theta)
                omega_new = jnp.clip(omega_new, 0.5 * omega, 2.0 * omega)
                omega_new = jnp.clip(omega_new, 1e-2, 1e2)
                if adaptive_blocks is not None:     # adapt early, then freeze
                    omega_new = jnp.where(blocks < adaptive_blocks, omega_new, omega)
                omega = omega_new
            # else: fixed omega (robust fallback for problems where the adaptive
            # weight destabilizes, e.g. pure least squares with g = 0)
            return x_avg, y_avg, omega                            # restart to avg

        if check is not None:
            return _checked_while(check, advance, data, x0, y0,
                                  jnp.asarray(1.0, jnp.float32), cap, tol)

        def body(carry):
            x, y, omega, blocks, _res = carry
            x_avg, y_avg, omega = advance(x, y, omega, blocks)
            return (x_avg, y_avg, omega, blocks + 1, _res)

        def cond(carry):
            # Evaluate the materialized carry in the condition. The exact point
            # which passes the test is returned, with no extra unchecked block.
            # This also avoids the fused-intermediate rematerialization issue.
            res = residual(carry[0], carry[1], data)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in
                         (carry[0], carry[1], *jax.tree_util.tree_leaves(data))]))
            return (carry[3] < cap) & finite & ~jnp.isnan(res) & (res > tol)

        init = (x0, y0, jnp.asarray(1.0, jnp.float32),
                jnp.asarray(0), jnp.asarray(jnp.inf, jnp.float32))
        x, y, _om, blocks, _res = lax.while_loop(cond, body, init)
        # report the residual AT the returned iterate, so the reported value
        # always certifies the point handed back to the caller
        return x, y, blocks, residual(x, y, data)

    if budget_arg:
        # The budget is an ARGUMENT, so one executable serves every segment
        # length the gap-driven criterion asks for.
        return operand_jit(jax.vmap(solve, in_axes=(0, 0, 0, None)))
    return operand_jit(jax.vmap(lambda d, a, b: solve(d, a, b)))


def _make_standard_solver(matvec, rmatvec, tau, sigma, prox_primal, prox_dual,
                          residual, n, m, *, block, max_blocks, tol, dual_slices=None, loop_unroll=1,
                          batched=True, budget_arg=False, check=None):
    """Classical fixed-step PDHG with theta=1 and last-iterate stopping.

    The extrapolated point persists across reporting blocks. No Halpern
    anchors or averaging accumulators are allocated. The metric, proximal
    contracts and certificate are supplied by the same compiler as usual.
    loop_unroll changes compiler scheduling, not the block length or checks.
    """
    if dual_slices is not None:
        return _make_block_standard_solver(
            matvec, rmatvec, tau, sigma, prox_primal, prox_dual, residual,
            dual_slices, block=block, max_blocks=max_blocks, tol=tol,
            loop_unroll=loop_unroll, batched=batched, budget_arg=budget_arg,
            check=check)

    def solve(data, x0, y0, cap=max_blocks):
        def step(_k, state):
            x, y, xbar = state
            yn = prox_dual(y + sigma * matvec(xbar), sigma, data)
            xn = prox_primal(x - tau * rmatvec(yn), tau, data)
            return xn, yn, 2.0*xn-x

        if check is not None:
            def advance(x, y, xbar, _blocks):
                return lax.fori_loop(0, block, step, (x, y, xbar),
                                     unroll=loop_unroll)
            return _checked_while(check, advance, data, x0, y0, x0, cap, tol,
                                  aux_arrays=lambda xbar: (xbar,))

        def body(state):
            x, y, xbar, blocks = state
            x, y, xbar = lax.fori_loop(0, block, step, (x, y, xbar),
                                         unroll=loop_unroll)
            return x, y, xbar, blocks+1

        def cond(state):
            x, y, xbar, blocks = state
            res = residual(x, y, data)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in
                (x, y, xbar, *jax.tree_util.tree_leaves(data))]))
            return ((blocks < cap) & finite & ~jnp.isnan(res)
                    & (res > tol))

        x, y, _xb, blocks = lax.while_loop(
            cond, body, (x0, y0, x0, jnp.asarray(0)))
        return x, y, blocks, residual(x, y, data)

    if budget_arg:
        return operand_jit(jax.vmap(solve, in_axes=(0, 0, 0, None)) if batched else solve)
    return operand_jit(jax.vmap(lambda d, a, b: solve(d, a, b)) if batched
                   else (lambda d, a, b: solve(d, a, b)))


def _make_block_standard_solver(matvec, rmatvec, tau, sigma, prox_primal,
                                prox_dual, residual, slices, *, block,
                                max_blocks, tol, loop_unroll=1, batched=True,
                                budget_arg=False, check=None):
    """Same PDHG arithmetic and flat public arrays; separate internal dual blocks.

    Keeping the block outputs separate lets XLA avoid repacking every dual
    vector in each numerical iteration. The existing flat operator/prox and
    certificate callbacks are unchanged. No user buffers are donated.
    """
    slices = tuple(slices)

    def split(y):
        return tuple(y[start:end] for start, end in slices)

    def join(parts):
        return jnp.concatenate(parts)

    def solve(data, x0, y0, cap=max_blocks):
        def step(_k, state):
            x, parts, xbar = state
            yn = prox_dual(join(parts) + sigma * matvec(xbar), sigma, data)
            xn = prox_primal(x - tau * rmatvec(yn), tau, data)
            return xn, split(yn), 2.0*xn-x

        if check is not None:
            def advance(x, parts, xbar, _blocks):
                return lax.fori_loop(0, block, step, (x, parts, xbar),
                                     unroll=loop_unroll)
            return _checked_while(check, advance, data, x0, split(y0), x0, cap,
                                  tol, aux_arrays=lambda xbar: (xbar,),
                                  join=join, split=split)

        def body(state):
            x, parts, xbar, blocks = state
            x, parts, xbar = lax.fori_loop(0, block, step, (x, parts, xbar),
                                             unroll=loop_unroll)
            return x, parts, xbar, blocks+1

        def cond(state):
            x, parts, xbar, blocks = state
            y = join(parts)
            res = residual(x, y, data)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in
                (x, y, xbar, *jax.tree_util.tree_leaves(data))]))
            return ((blocks < cap) & finite & ~jnp.isnan(res)
                    & (res > tol))

        x, parts, _xb, blocks = lax.while_loop(
            cond, body, (x0, split(y0), x0, jnp.asarray(0)))
        y = join(parts)
        return x, y, blocks, residual(x, y, data)

    if budget_arg:
        return operand_jit(jax.vmap(solve, in_axes=(0, 0, 0, None)) if batched else solve)
    return operand_jit(jax.vmap(lambda d, a, b: solve(d, a, b)) if batched
                   else (lambda d, a, b: solve(d, a, b)))


def make_online_solver(matvec, rmatvec, tau0, sigma0, prox_primal, prox_dual,
                       residual_p, n, m, *, block=100, max_blocks=200, tol=1e-3,
                       theta=0.5, adaptive_omega=True, warmup_blocks=3, safety=0.9):
    """Online equilibration: NO separate equilibration pass.  Start from the
    scalar (or supplied) metric tau0, sigma0; run a short warmup of ``warmup_blocks``
    real PDHG blocks that ALSO accumulate the free directional energies
    (K xbar)^2 (rows) and (K^T y)^2 (cols) from the products the iteration already
    computes; refine the diagonal metric once (inverse-energy Jacobi + a matrix-free
    global rescale to ||.||~safety); FREEZE it; then solve to tolerance in the fixed
    refined metric.  Operator-pure throughout: only matvec/rmatvec, no entry of K,
    no formed matrix.  The warmup blocks are genuine solve progress, so the
    equilibration is folded into the solve at (near) zero extra oracle cost."""
    def solve(data, x0, y0):
        # ---- phase 1: warmup with initial metric, accumulate free energies ----
        def wblock(x, y, omega):
            xa, ya = x, y
            sw = sigma0 * omega
            tw = tau0 / omega

            def step(k, st):
                x, y, xbar, xs, ys, re, ce = st
                Kxb = matvec(xbar)                         # free product
                y_new = prox_dual(y + sw * Kxb, sw, data)
                KTy = rmatvec(y_new)                       # free product
                x_new = prox_primal(x - tw * KTy, tw, data)
                lam = 1.0 / (k.astype(jnp.float32) + 2.0)
                xn = lam * xa + (1.0 - lam) * x_new
                yn = lam * ya + (1.0 - lam) * y_new
                return (xn, yn, 2.0 * xn - x, xs + xn, ys + yn,
                        re + Kxb * Kxb, ce + KTy * KTy)

            x, y, _xb, xs, ys, re, ce = lax.fori_loop(
                0, block, step, (x, y, x, jnp.zeros(n), jnp.zeros(m),
                                 jnp.zeros(m), jnp.zeros(n)))
            return xs / block, ys / block, re, ce

        def wbody(i, carry):
            x, y, omega, re, ce = carry
            xa, ya = x, y
            x_avg, y_avg, re_b, ce_b = wblock(x, y, omega)
            if adaptive_omega:
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                omega = jnp.clip((dy / dx) ** theta * omega ** (1.0 - theta), 1e-3, 1e3)
            return (x_avg, y_avg, omega, re + re_b, ce + ce_b)

        x, y, omega, re, ce = lax.fori_loop(
            0, warmup_blocks, wbody,
            (x0, y0, jnp.asarray(1.0, jnp.float32), jnp.zeros(m), jnp.zeros(n)))

        # ---- refine metric: CLIPPED log-space Jacobi + matrix-free global rescale ----
        # Free products are directional (biased, low-rank): many coordinates get ~0
        # energy, so a raw 1/energy scaling explodes.  Floor the energy at a fraction
        # of its max and clip the log-scaling to a bounded range, so unexcited
        # coordinates get a bounded (not infinite) step.  (This is the operator-pure
        # analog of SMFE's box constraint ||u||_inf <= M.)
        re = jnp.maximum(re, 1e-4 * jnp.max(re)) + 1e-30
        ce = jnp.maximum(ce, 1e-4 * jnp.max(ce)) + 1e-30
        u = jnp.clip(-0.5 * jnp.log(re / jnp.mean(re)), -3.0, 3.0)   # dual/row log-scale
        v = jnp.clip(-0.5 * jnp.log(ce / jnp.mean(ce)), -3.0, 3.0)   # primal/col log-scale
        sigma_s = jnp.exp(2.0 * u)             # dual per-row step  (downweight busy rows)
        tau_s = jnp.exp(2.0 * v)               # primal per-col step (downweight busy cols)
        st, sg = jnp.sqrt(tau_s), jnp.sqrt(sigma_s)
        nrm = _power_norm(lambda z: sg * matvec(st * z),
                          lambda w: st * rmatvec(sg * w), n, iters=15, seed=1)
        scale = safety / (nrm + 1e-30)
        tau = tau_s * scale
        sigma = sigma_s * scale

        # ---- phase 2: fixed refined metric to tolerance ----
        def run_block(x, y, omega):
            xa, ya = x, y
            sw = sigma * omega
            tw = tau / omega

            def step(k, st):
                x, y, xbar, xs, ys = st
                y_new = prox_dual(y + sw * matvec(xbar), sw, data)
                x_new = prox_primal(x - tw * rmatvec(y_new), tw, data)
                lam = 1.0 / (k.astype(jnp.float32) + 2.0)
                xn = lam * xa + (1.0 - lam) * x_new
                yn = lam * ya + (1.0 - lam) * y_new
                return (xn, yn, 2.0 * xn - x, xs + xn, ys + yn)

            x, y, _xb, xs, ys = lax.fori_loop(
                0, block, step, (x, y, x, jnp.zeros(n), jnp.zeros(m)))
            return xs / block, ys / block

        def body(carry):
            x, y, omega, blocks, _res = carry
            xa, ya = x, y
            x_avg, y_avg = run_block(x, y, omega)
            if adaptive_omega:
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                omega = jnp.clip((dy / dx) ** theta * omega ** (1.0 - theta), 1e-3, 1e3)
            return (x_avg, y_avg, omega, blocks + 1, _res)

        def cond(carry):
            res = residual_p(carry[0], carry[1], data, tau, sigma)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in
                         (carry[0], carry[1], *jax.tree_util.tree_leaves(data))]))
            return ((carry[3] < (max_blocks - warmup_blocks))
                    & finite & ~jnp.isnan(res) & (res > tol))

        init = (x, y, omega, jnp.asarray(0), jnp.asarray(jnp.inf, jnp.float32))
        x, y, _om, blk2, _res = lax.while_loop(cond, body, init)
        # residual AT the returned iterate -- see make_solver
        return (x, y, jnp.asarray(warmup_blocks) + blk2,
                residual_p(x, y, data, tau, sigma))

    return operand_jit(jax.vmap(solve))


def make_osgm_solver(matvec, rmatvec, tau0, sigma0, prox_primal, prox_dual,
                     residual_p, n, m, *, block=100, max_blocks=200, tol=1e-3,
                     theta=0.5, adaptive_omega=True, warmup_blocks=8,
                     eta=0.5, mclip=2.0):
    """OCS synthesis: SMFE cold start (tau0, sigma0) + OSGM-style online refinement
    of the diagonal metric, learned from OCS's OWN matrix-free residual directions
    (primal residual ~ K^T y, dual residual ~ K x -- the same products the block
    computes and the same quantities the termination residual uses), then FREEZE.

    Per warmup block, one online-gradient-descent step on per-coordinate log-refine-
    ments a (primal), b (dual): where a coordinate's residual direction ALIGNS across
    consecutive blocks the step is consistent and is accelerated; where it flips
    (oscillation) it is damped -- OSGM's hypergradient rule (Gao-Chu-Ye-Udell 2025),
    here driven by the operator-native residual and warm-started from SMFE.  After
    ``warmup_blocks`` the metric is frozen and the tail runs fixed.  Operator-pure:
    only matvec/rmatvec; a, b are clipped to a bounded box (a la SMFE's ||u||_inf<=M).
    """
    def solve(data, x0, y0):
        def run_block(x, y, omega, tau, sigma):
            xa, ya = x, y
            sw = sigma * omega
            tw = tau / omega

            def step(k, st):
                x, y, xbar, xs, ys = st
                y_new = prox_dual(y + sw * matvec(xbar), sw, data)
                x_new = prox_primal(x - tw * rmatvec(y_new), tw, data)
                lam = 1.0 / (k.astype(jnp.float32) + 2.0)
                xn = lam * xa + (1.0 - lam) * x_new
                yn = lam * ya + (1.0 - lam) * y_new
                return (xn, yn, 2.0 * xn - x, xs + xn, ys + yn)

            x, y, _xb, xs, ys = lax.fori_loop(
                0, block, step, (x, y, x, jnp.zeros(n), jnp.zeros(m)))
            return xs / block, ys / block

        def cosine(u, v):                      # per-coordinate aligned product, scale-free
            return (u * v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-12)

        # ---- phase 1: warmup with per-block OSGM updates on log-refinements a, b ----
        def wbody(i, carry):
            x, y, omega, a, b, gp_prev, gd_prev = carry
            tau = tau0 * jnp.exp(a)            # refined metric = SMFE base x online factor
            sigma = sigma0 * jnp.exp(b)
            xa, ya = x, y
            x_avg, y_avg = run_block(x, y, omega, tau, sigma)
            if adaptive_omega:
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                omega = jnp.clip((dy / dx) ** theta * omega ** (1.0 - theta), 1e-3, 1e3)
            gp = rmatvec(y_avg)                # primal residual direction K^T y
            gd = matvec(x_avg)                 # dual residual direction K x
            a = jnp.clip(a + eta * cosine(gp, gp_prev), -mclip, mclip)
            b = jnp.clip(b + eta * cosine(gd, gd_prev), -mclip, mclip)
            return (x_avg, y_avg, omega, a, b, gp, gd)

        a0 = jnp.zeros(n); b0 = jnp.zeros(m)
        init_w = (x0, y0, jnp.asarray(1.0, jnp.float32), a0, b0,
                  jnp.zeros(n), jnp.zeros(m))
        x, y, omega, a, b, _gp, _gd = lax.fori_loop(0, warmup_blocks, wbody, init_w)

        # ---- freeze the refined metric ----
        tau = tau0 * jnp.exp(a)
        sigma = sigma0 * jnp.exp(b)

        # ---- phase 2: fixed refined metric to tolerance ----
        def body(carry):
            x, y, omega, blocks, _res = carry
            xa, ya = x, y
            x_avg, y_avg = run_block(x, y, omega, tau, sigma)
            if adaptive_omega:
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                omega = jnp.clip((dy / dx) ** theta * omega ** (1.0 - theta), 1e-3, 1e3)
            return (x_avg, y_avg, omega, blocks + 1, _res)

        def cond(carry):
            res = residual_p(carry[0], carry[1], data, tau, sigma)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in
                         (carry[0], carry[1], *jax.tree_util.tree_leaves(data))]))
            return ((carry[3] < (max_blocks - warmup_blocks))
                    & finite & ~jnp.isnan(res) & (res > tol))

        init = (x, y, omega, jnp.asarray(0), jnp.asarray(jnp.inf, jnp.float32))
        x, y, _om, blk2, _res = lax.while_loop(cond, body, init)
        # residual AT the returned iterate -- see make_solver
        return (x, y, jnp.asarray(warmup_blocks) + blk2,
                residual_p(x, y, data, tau, sigma))

    return operand_jit(jax.vmap(solve))
