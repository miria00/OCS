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


def make_solver(matvec, rmatvec, tau, sigma, prox_primal, prox_dual, residual,
                n, m, *, block=100, max_blocks=200, tol=1e-3, theta=0.5,
                adaptive_omega=True):
    def solve(data, x0, y0):
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

        def body(carry):
            x, y, omega, blocks, _res = carry
            xa, ya = x, y
            x_avg, y_avg = run_block(x, y, omega)
            if adaptive_omega:                      # PDLP-style primal-weight update
                dx = jnp.linalg.norm(x_avg - xa) + 1e-12
                dy = jnp.linalg.norm(y_avg - ya) + 1e-12
                omega = jnp.clip((dy / dx) ** theta * omega ** (1.0 - theta),
                                 1e-3, 1e3)
            # else: fixed omega (robust fallback for problems where the adaptive
            # weight destabilizes, e.g. pure least squares with g = 0)
            res = residual(x_avg, y_avg, data)
            return (x_avg, y_avg, omega, blocks + 1, res)          # restart to avg

        def cond(carry):
            return (carry[3] < max_blocks) & (carry[4] > tol)

        init = (x0, y0, jnp.asarray(1.0, jnp.float32),
                jnp.asarray(0), jnp.asarray(jnp.inf, jnp.float32))
        x, y, _om, blocks, res = lax.while_loop(cond, body, init)
        return x, y, blocks, res

    return jax.jit(jax.vmap(solve))
