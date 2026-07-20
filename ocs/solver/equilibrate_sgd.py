"""Stochastic matrix-free equilibration (Diamond & Boyd, JOTA 2017).

Instead of a one-shot estimate of per-coordinate steps, SOLVE the equilibration
problem itself by projected stochastic gradient over row/column log-scalings.
Minimize (their problem 5)

    f(u,v) = 1/2 sum_ij A_ij^2 e^{2u_i+2v_j} - a2 1^T u - b2 1^T v + (g/2)||(u,v)||^2
    s.t.  ||u||_inf <= M, ||v||_inf <= M,

using ONLY matvec/rmatvec.  The gradient in u is the row-norms^2 of DAE (D=diag e^u,
E=diag e^v) minus a2 plus g*u; its unbiased estimate is |D A E s|^2 for a random
sign vector s (one matvec).  Iterate-averaged projected SGD converges O(1/t).

The equilibrated scalings give the PDHG diagonal steps tau = e^{2 vbar} (primal,
per column) and sigma = e^{2 ubar} (dual, per row); a final power iteration on the
scaled operator applies the global rescale for the PDHG stability bound.

Rows of a coupled (scalar-class) block are averaged in log space each step, so the
dual scaling stays uniform over that block -- the block-admissible coercion done
INSIDE the optimization, not fighting it afterwards.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .equilibrate import _power_norm


def smfe_equilibration(matvec, rmatvec, n, m, scalar_spans=(), *,
                       iters=80, probes=2, gamma=1.0, M=6.0, safety=0.9, seed=0):
    """probes : Rademacher probes averaged per step (variance reduction).  Note:
    antithetic pairs s,-s give identical |DAEs|^2 (the estimator is even in s), so
    they do NOT reduce variance here -- multi-probe averaging is the right lever.
    """
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)

    def rade(k, d):
        return jnp.where(jax.random.bernoulli(k, 0.5, (d,)), 1.0, -1.0).astype(jnp.float32)

    def body(t, carry):
        key, u, v, ubar, vbar, wsum = carry
        eu, ev = jnp.exp(u), jnp.exp(v)
        row_e = jnp.zeros(m, jnp.float32); col_e = jnp.zeros(n, jnp.float32)
        for _q in range(probes):                                  # multi-probe average
            key, k1, k2 = jax.random.split(key, 3)
            row_e = row_e + (eu * matvec(ev * rade(k1, n))) ** 2  # est. row-norms^2(DAE)
            col_e = col_e + (ev * rmatvec(eu * rade(k2, m))) ** 2 # est. col-norms^2(DAE)
        tf = t.astype(jnp.float32)
        gu = row_e / probes - 1.0 + gamma * u
        gv = col_e / probes - 1.0 + gamma * v
        eta = 2.0 / (gamma * (tf + 2.0))
        u = jnp.clip(u - eta * gu, -M, M)
        v = jnp.clip(v - eta * gv, -M, M)
        for (a, b) in spans:                                      # block-admissible coercion
            u = u.at[a:b].set(jnp.mean(u[a:b]))
        wt = tf + 1.0
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    @jax.jit
    def run():                              # whole equilibration is ONE compiled program
        init = (jax.random.PRNGKey(seed),
                jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.asarray(0.0, jnp.float32))
        _k, u, v, ubar, vbar, wsum = jax.lax.fori_loop(0, iters, body, init)
        tau = jnp.exp(2.0 * vbar / wsum)    # primal per-coordinate step
        sigma = jnp.exp(2.0 * ubar / wsum)  # dual per-coordinate step
        st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
        nrm = _power_norm(lambda x: sg * matvec(st * x),
                          lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        scale = safety / (nrm + 1e-30)
        return tau * scale, sigma * scale, nrm

    tau, sigma, nrm = run()
    return tau, sigma, float(nrm)
