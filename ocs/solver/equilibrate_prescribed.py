"""Prescribed-marginal stochastic matrix-free equilibration.

The Diamond-Boyd SMFE of ``equilibrate_sgd`` equilibrates the operator to
UNIFORM squared-norm targets (rows r = sqrt(n/m), columns s = sqrt(m/n)).
This module generalizes the targets to PRESCRIBED nonnegative marginals
rho (per row) and gam (per column): find diagonal D, E with

    row-norms^2(DKE) ~ rho_i,     col-norms^2(DKE) ~ gam_j,

which in the variables p = d^2, q = e^2 is exactly Rothblum-Schneider
(r, c)-scaling of M = |K|^{o2} -- the setting of the paper's rectangular
epsilon-scalability characterization (Proposition p-scal), now with weighted
targets.  The same projected-SGD objective applies verbatim with the scalar
targets replaced by vectors; totals are normalized to the compatible mass
sum(rho) = sum(gam) = sqrt(mn) internally.

Intended use: RESIDUAL-TARGETED re-equilibration at a restart boundary (the
"one-armed VM" of the CORGI solver, recast as equilibration): set rho
proportional to the per-row primal violation, or gam to the per-column dual
violation, so the metric concentrates on the coordinates that still block --
approximately equilibrating the ACTIVE submatrix, which is what the local
condition number H(B) = max_J 1/sigma_min^+(B_J) sees.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .equilibrate import _power_norm


def prescribed_smfe(matvec, rmatvec, n, m, rho, gam, scalar_spans=(), *,
                    iters=80, probes=2, gamma=1.0, M=6.0, safety=0.9, seed=0,
                    floor_frac=1e-3):
    """SMFE with prescribed row/column squared-norm targets.

    rho : (m,) nonnegative per-row targets (relative weights);
    gam : (n,) nonnegative per-column targets.  Both are floored at
    ``floor_frac`` of their mean (a zero target is never imposed -- a row that
    is currently feasible still keeps a small share of the mass) and rescaled
    so the totals match the compatible mass sqrt(m n).
    Returns (tau, sigma, scaled_norm) exactly like ``smfe_equilibration``.
    """
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)
    rho = jnp.asarray(rho, jnp.float32)
    gam = jnp.asarray(gam, jnp.float32)
    rho = jnp.maximum(rho, floor_frac * jnp.mean(rho))
    gam = jnp.maximum(gam, floor_frac * jnp.mean(gam))
    mass = jnp.sqrt(jnp.float32(m) * jnp.float32(n))
    r_tgt = rho * (mass / jnp.sum(rho))          # (m,) row squared-norm targets
    s_tgt = gam * (mass / jnp.sum(gam))          # (n,) col squared-norm targets

    def rade(k, d):
        return jnp.where(jax.random.bernoulli(k, 0.5, (d,)), 1.0, -1.0).astype(jnp.float32)

    def body(t, carry):
        key, u, v, ubar, vbar, wsum = carry
        eu, ev = jnp.exp(u), jnp.exp(v)
        row_e = jnp.zeros(m, jnp.float32); col_e = jnp.zeros(n, jnp.float32)
        for _q in range(probes):
            key, k1, k2 = jax.random.split(key, 3)
            row_e = row_e + (eu * matvec(ev * rade(k1, n))) ** 2
            col_e = col_e + (ev * rmatvec(eu * rade(k2, m))) ** 2
        tf = t.astype(jnp.float32)
        gu = row_e / probes - r_tgt + gamma * u
        gv = col_e / probes - s_tgt + gamma * v
        eta = 2.0 / (gamma * (tf + 2.0))
        w_u = u - eta * gu                        # tentative point, unclipped
        u = jnp.clip(w_u, -M, M)                  # free coordinates: clip each
        v = jnp.clip(v - eta * gv, -M, M)
        for (a, b) in spans:
            # projection onto the tied box: mean of the UNCLIPPED tentative
            # point, then clip that scalar. mean(clip(.)) is a different map.
            u = u.at[a:b].set(jnp.clip(jnp.mean(w_u[a:b]), -M, M))
        wt = tf + 1.0
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    @jax.jit
    def run():
        init = (jax.random.PRNGKey(seed),
                jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.asarray(0.0, jnp.float32))
        _k, u, v, ubar, vbar, wsum = jax.lax.fori_loop(0, iters, body, init)
        u_f, v_f = ubar / wsum, vbar / wsum
        tau = jnp.exp(2.0 * v_f)
        sigma = jnp.exp(2.0 * u_f)
        st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
        nrm = _power_norm(lambda x: sg * matvec(st * x),
                          lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        scale = safety / (nrm + 1e-30)
        return tau * scale, sigma * scale, nrm

    tau, sigma, nrm = run()
    return tau, sigma, float(nrm)
