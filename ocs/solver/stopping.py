"""Matrix-free termination: a primal AND a dual fixed-point residual, each from a
constant number of oracle calls.

For min_x c^T x + g(x) + sum_i f_i(K_i x + l_i), a point (x, y) is optimal iff it
is a fixed point of the PDHG maps:

    x = prox_{tau g}( x - tau (K^T y + c) )          (primal)
    y = prox_{sigma f*}( y + sigma (K x + l) )        (dual)

So we report the two relative residuals

    r_p = ||x - prox_primal(x - tau K^T y)|| / (1 + ||x||)
    r_d = ||y - prox_dual (y + sigma K x)|| / (1 + ||y||)

(prox_primal folds in c; prox_dual folds in l).  Both -> 0 at optimality, both
use only one matvec + one rmatvec, and both are resolvent-based -- optimality is
certified without ever touching an entry of K.  Stopping uses max(r_p, r_d).

An optional duality gap (from block value / conj_value, when available) is
reported separately for the paper; it is not required for stopping.
"""
from __future__ import annotations

import jax.numpy as jnp


def make_residual(matvec, rmatvec, prox_primal, prox_dual, tau, sigma,
                  smooth_terms=(), offset_index=0):
    """smooth_terms: iterable of (a, b, grad_fn) for dual blocks whose f is smooth.
    For a smooth block the unique dual is y_i = grad f_i(s_i), s_i = (Kx+l)_i, so we
    substitute that exact gradient into the PRIMAL residual (making r_p the true
    primal-dual stationarity) and drop the block from the dual residual -- the dual
    ITERATE for a smooth block converges slowly and floors in float32, but its
    optimality is fully certified by primal stationarity."""
    smooth = tuple((int(a), int(b), g) for a, b, g in smooth_terms)

    def residual(x, y, data):
        Kx = matvec(x)
        s = Kx + data[offset_index]
        y_eff = y
        for (a, b, grad) in smooth:                    # exact dual for smooth blocks
            y_eff = y_eff.at[a:b].set(grad(s[a:b]))
        x_fp = prox_primal(x - tau * rmatvec(y_eff), tau, data)
        r_p = jnp.linalg.norm(x - x_fp) / (1.0 + jnp.linalg.norm(x))
        y_fp = prox_dual(y + sigma * Kx, sigma, data)
        dres = y - y_fp
        for (a, b, _g) in smooth:                       # smooth blocks certified by r_p
            dres = dres.at[a:b].set(0.0)
        r_d = jnp.linalg.norm(dres) / (1.0 + jnp.linalg.norm(y))
        return jnp.maximum(r_p, r_d)

    return residual
