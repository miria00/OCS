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

from .stopping import stable_norm, relative_ratio


def make_cgls_solver(matvec, rmatvec, n, m, *, iters=200, rho=0.0, tol=1e-8):
    """Return a jitted, vmap-able solver for  min 1/2||K x - b||^2 + (rho/2)||x||^2.

    Returns solve(b, x0) -> (x, k, rel_resid) where rel_resid is the relative
    normal-equation residual divided by max(||K^T b||, ||K^T(Kx-b)||,
    rho*||x||), matching the free/smooth primal KKT normalization."""
    rho = float(rho)

    def solve(b, x0):
        Ktb = rmatvec(b)
        nrm0 = stable_norm(Ktb)

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
            return (k < iters) & (relative_ratio(jnp.sqrt(gb), nrm0) > tol) & (g < 4.0 * gb + 1e-30)

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
        # A recurrence residual can drift in float32. Report the optimality
        # residual of the point actually returned, using the operator again.
        gradient = rmatvec(matvec(x_best) - b)
        actual = gradient + rho * x_best
        actual_norm = stable_norm(actual)
        residual = relative_ratio(actual_norm, jnp.maximum(nrm0, jnp.maximum(
            stable_norm(gradient), stable_norm(rho * x_best))))
        valid = (jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(x0))
                 & jnp.all(jnp.isfinite(x_best)))
        return x_best, k, jnp.where(valid, residual, jnp.inf)

    return jax.jit(jax.vmap(solve))
