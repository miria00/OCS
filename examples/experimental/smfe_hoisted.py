"""The shipped SMFE equilibration with its jitted program HOISTED and cached.

`ocs/solver/equilibrate_sgd.py` defines `@jax.jit def run()` INSIDE
`smfe_equilibration`, so every call creates a new function object and JAX
recompiles the whole equilibration program.  In the compiled-family path that is
paid once per family; in one-shot solves and in any harness that compiles per
measurement it is paid on every call.  This module exists so that a timing
harness can separate compilation from execution WITHOUT editing `ocs/`.

The function body below is GENERATED from the shipped source text, not retyped:
`weight_init_twostage_timing.py --mode gate` reproduces the generation diff and
checks it.  Exactly two things differ from the shipped function:
  1. its name, `smfe_equilibration_hoisted`;
  2. the decorator on `run`: `@jax.jit` becomes `@_hoisted_jit(key, matvec,
     rmatvec)`, preceded by one line computing `key` from the static arguments.
     `_hoisted_jit` returns the already-compiled program when the key matches and
     the SAME matvec/rmatvec objects are passed, and otherwise jits and caches the
     new one.  The traced code is therefore identical to the shipped code, which is
     why the outputs are expected to be bitwise identical -- and that is checked,
     as a hard gate, before any timing uses this copy.
The diff against the shipped function is saved as `smfe_hoisted.diff` in the
results directory.

Callers must pass the SAME matvec/rmatvec objects to hit the cache; a new lambda
per call is a new key and recompiles, exactly like the shipped function.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ocs.solver.equilibrate import _power_norm
from ocs.solver.equilibrate_sgd import _scalar_span_layout

_RUN_CACHE = {}
STATS = {"builds": 0, "hits": 0}


def _hoisted_jit(key, matvec, rmatvec):
    def deco(fn):
        hit = _RUN_CACHE.get(key)
        if hit is not None and hit[0] is matvec and hit[1] is rmatvec:
            STATS["hits"] += 1
            return hit[2]
        jitted = jax.jit(fn)
        _RUN_CACHE[key] = (matvec, rmatvec, jitted)     # holds the objects, so ids cannot be reused
        STATS["builds"] += 1
        return jitted
    return deco


def smfe_equilibration_hoisted(matvec, rmatvec, n, m, scalar_spans=(), *,
                       iters=80, probes=2, gamma=1.0, M=6.0, safety=0.9, seed=0,
                       diagnostics=False, schedule="sgd"):
    """probes : Rademacher probes averaged per step (variance reduction).  Note:
    antithetic pairs s,-s give identical |DAEs|^2 (the estimator is even in s), so
    they do NOT reduce variance here -- multi-probe averaging is the right lever.

    diagnostics : if True, also return a dict with Bradley's normalized-variance
    equilibration diagnostic nvar(x) = d * var(x) / ||x||^2 (0 = perfectly
    equilibrated) of the row and column squared-norms of the operator BEFORE
    (identity scaling) and AFTER equilibration.  All estimated matrix-free with
    the same Rademacher probes, so the operator-only contract is preserved.  The
    global rescale is a uniform factor and leaves nvar invariant, so the AFTER
    numbers use the averaged log-scalings directly.
    """
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)
    row_layout, row_state_size = _scalar_span_layout(m, spans)
    r_tgt = jnp.sqrt(jnp.float32(n) / jnp.float32(m))   # row squared-norm target
    s_tgt = jnp.sqrt(jnp.float32(m) / jnp.float32(n))   # col squared-norm target (m r = n s)

    def expand_rows(u):
        if row_layout is None:
            return u
        return jnp.concatenate([
            jnp.broadcast_to(u[p:p + 1], (b - a,)) if scalar
            else u[p:p + b - a]
            for a, b, p, scalar in row_layout])

    def project_rows(u, w, M):
        """Project onto the box intersected with the tied-coordinate subspace.

        Free coordinates keep the value the caller already clipped. For a tied
        block the Euclidean projection onto {a*1 : |a| <= M} is the mean of the
        UNCLIPPED tentative point w, clipped afterwards. Averaging the clipped
        values instead is a different map and is not that projection: for
        M = 1 and w = (2, -1) it returns 0 rather than 1/2.
        """
        if row_layout is None:
            for a, b in spans:
                u = u.at[a:b].set(jnp.clip(jnp.mean(w[a:b]), -M, M))
            return u
        return jnp.concatenate([
            jnp.clip(jnp.mean(w[a:b], keepdims=True), -M, M) if scalar
            else u[a:b]
            for a, b, _p, scalar in row_layout])

    def rade(k, d):
        return jnp.where(jax.random.bernoulli(k, 0.5, (d,)), 1.0, -1.0).astype(jnp.float32)

    def probe_sum(key, eu, ev, count):
        # Keep the probe bodies in a loop: unrolling can retain independent FFT
        # and row-sized temporaries for every probe, including diagnostics.
        # The RNG splits and accumulation order match the unrolled estimator.
        def one_probe(_q, carry):
            key, row_e, col_e = carry
            key, k1, k2 = jax.random.split(key, 3)
            row_e = row_e + (eu * matvec(ev * rade(k1, n))) ** 2
            col_e = col_e + (ev * rmatvec(eu * rade(k2, m))) ** 2
            return key, row_e, col_e

        return jax.lax.fori_loop(
            0, count, one_probe,
            (key, jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32)))

    def body(t, carry):
        key, u, v, ubar, vbar, wsum = carry
        u = expand_rows(u)
        eu, ev = jnp.exp(u), jnp.exp(v)
        key, row_e, col_e = probe_sum(key, eu, ev, probes)
        tf = t.astype(jnp.float32)
        gu = row_e / probes - r_tgt + gamma * u
        gv = col_e / probes - s_tgt + gamma * v
        eta = 2.0 / (gamma * (tf + 2.0))
        w_u = u - eta * gu                        # tentative point, unclipped
        u = jnp.clip(w_u, -M, M)
        v = jnp.clip(v - eta * gv, -M, M)
        u = project_rows(u, w_u, M)
        wt = tf + 1.0
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    def body_bradley(t, carry):
        # Bradley SNBIN (thesis Ch.2, Fig.2.1, eqs.2.25-2.27): one (or few) probes
        # and a damped log-space fixed-point step toward equal norms, with a
        # decaying running-average weight omega that halves at k=2,4,8,16 (floor
        # 1/16) -- move fast early, settle late.  The running estimate (u,v) itself
        # is the output; no separate iterate-average.
        key, u, v, ubar, vbar, wsum = carry
        u = expand_rows(u)
        eu, ev = jnp.exp(u), jnp.exp(v)
        key, row_e, col_e = probe_sum(key, eu, ev, probes)
        row_e = row_e / probes; col_e = col_e / probes
        k = t.astype(jnp.float32) + 1.0
        w = jnp.maximum(2.0 ** (-jnp.floor(jnp.log2(jnp.maximum(k, 1.0)))), 1.0 / 16.0)
        w_u = u + 0.5 * w * (jnp.log(r_tgt) - jnp.log(row_e + 1e-30))
        u = jnp.clip(w_u, -M, M)
        v = jnp.clip(v + 0.5 * w * (jnp.log(s_tgt) - jnp.log(col_e + 1e-30)), -M, M)
        u = project_rows(u, w_u, M)
        wt = k
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    step_body = body_bradley if schedule == "bradley" else body

    def _rc_est(key, eu, ev, P):            # matrix-free row/col squared-norm estimate
        _key, row, col = probe_sum(key, eu, ev, P)
        return row / P, col / P

    def _nvar(x, d):                         # Bradley's normalized variance in [0,1)
        return d * jnp.var(x) / (jnp.sum(x * x) + 1e-30)

    _key = (id(matvec), id(rmatvec), int(n), int(m), spans, int(iters), int(probes),
            float(gamma), float(M), float(safety), int(seed), bool(diagnostics), str(schedule))
    @_hoisted_jit(_key, matvec, rmatvec)
    def run():                              # whole equilibration is ONE compiled program
        init = (jax.random.PRNGKey(seed),
                jnp.zeros(row_state_size, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.zeros(row_state_size, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.asarray(0.0, jnp.float32))
        _k, u, v, ubar, vbar, wsum = jax.lax.fori_loop(0, iters, step_body, init)
        # sgd: iterate-averaged log-scalings; bradley: the running estimate itself
        u_f = expand_rows(u if schedule == "bradley" else ubar / wsum)
        v_f = v if schedule == "bradley" else vbar / wsum
        tau = jnp.exp(2.0 * v_f)            # primal per-coordinate step
        sigma = jnp.exp(2.0 * u_f)          # dual per-coordinate step
        st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
        nrm = _power_norm(lambda x: sg * matvec(st * x),
                          lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        scale = safety / (nrm + 1e-30)
        if diagnostics:
            dk = jax.random.PRNGKey(seed + 99)
            r0, c0 = _rc_est(dk, jnp.ones(m, jnp.float32), jnp.ones(n, jnp.float32), 8)
            r1, c1 = _rc_est(jax.random.PRNGKey(seed + 131),
                             jnp.exp(u_f), jnp.exp(v_f), 8)
            # nu_row / nu_col are the imbalance factors of the block-scalar
            # bound (see the paper's block-sparsity theorem): the largest row
            # (column) 2-norm divided by the root-mean-square, which is 1 at
            # exact balance and degrades the bound linearly.
            _nu = lambda x: jnp.sqrt(jnp.max(x) / (jnp.mean(x) + 1e-30))
            diag = {"nvar_row_pre": _nvar(r0, m), "nvar_col_pre": _nvar(c0, n),
                    "nvar_row_post": _nvar(r1, m), "nvar_col_post": _nvar(c1, n),
                    "nu_row": _nu(r1), "nu_col": _nu(c1)}
            return tau * scale, sigma * scale, nrm, diag
        return tau * scale, sigma * scale, nrm

    if diagnostics:
        tau, sigma, nrm, diag = run()
        return tau, sigma, float(nrm), {k: float(x) for k, x in diag.items()}
    tau, sigma, nrm = run()
    return tau, sigma, float(nrm)
