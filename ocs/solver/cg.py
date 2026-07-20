"""Matrix-free CGLS for the pure least-squares regime.

The restart-PDHG engine is the solver for the composite problem
``g(x) + sum_i f_i(K_i x + l_i)``.  But a *pure* least-squares problem

    minimize_x  1/2 || K x - b ||_2^2   (optionally + (rho/2)||x||^2)

with no other term is exactly the conjugate-gradient / LSQR regime Boyd flagged:
first-order splitting plateaus on it, while CG on the normal equations
``(K^T K + rho I) x = K^T b`` converges in O(sqrt(kappa)) matvecs.  This is NOT a
substitute for the PDHG engine on composite problems -- it is the specialised path
for the one case (a lone quadratic through an operator) where CG is the right tool.
Accessed through matvec/rmatvec only; the operator is never formed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def make_cgls_solver(matvec, rmatvec, n, m, *, iters=200, rho=0.0, tol=1e-8):
    """Return a jitted, vmap-able solver for  min 1/2||K x - b||^2 + (rho/2)||x||^2.

    Returns solve(b, x0) -> (x, k, rel_resid) where rel_resid is the relative
    normal-equation residual ||K^T(Kx-b) + rho x|| / ||K^T b||."""
    rho = float(rho)

    def solve(b, x0):
        Ktb = rmatvec(b)
        nrm0 = jnp.linalg.norm(Ktb) + 1e-30

        # CGLS (Bjorck): r = b - Kx (range space), s = K^T r - rho x (normal eq residual)
        x = x0
        r = b - matvec(x)
        s = rmatvec(r) - rho * x
        p = s
        gamma = jnp.dot(s, s)
        # In float32 the recurrence reaches an accuracy floor and then, if iterated
        # further, DIVERGES (denom -> 0, alpha blows up).  Keep the best iterate seen
        # and stop once the residual grows well past that best -- robust to any tol.
        x_best, g_best = x, gamma

        def cond(st):
            _x, _r, _s, _p, g, k, _xb, gb = st
            return (k < iters) & (jnp.sqrt(gb) / nrm0 > tol) & (g < 4.0 * gb + 1e-30)

        def body(st):
            x, r, s, p, gamma, k, x_best, g_best = st
            q = matvec(p)
            denom = jnp.dot(q, q) + rho * jnp.dot(p, p) + 1e-30
            alpha = gamma / denom
            x = x + alpha * p
            r = r - alpha * q
            s = rmatvec(r) - rho * x
            gamma_new = jnp.dot(s, s)
            beta = gamma_new / (gamma + 1e-30)
            p = s + beta * p
            better = gamma_new < g_best
            x_best = jnp.where(better, x, x_best)
            g_best = jnp.where(better, gamma_new, g_best)
            return (x, r, s, p, gamma_new, k + 1, x_best, g_best)

        x, r, s, p, gamma, k, x_best, g_best = lax.while_loop(
            cond, body, (x, r, s, p, gamma, 0, x_best, g_best))
        return x_best, k, jnp.sqrt(g_best) / nrm0

    return jax.jit(jax.vmap(solve))
