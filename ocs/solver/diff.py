"""Differentiable operator-native solve via implicit differentiation.

The forward solve is a data-dependent fixed-point loop, so we cannot reverse-mode
through it.  Instead we attach a custom VJP that differentiates the OPTIMALITY map
by the implicit function theorem (the diffcp / cvxpylayers approach, made
operator-native).  At the solution z*=(x*,y*) the fixed-point residual
    R(z, (l,c)) = ( x - prox_f^T(x - T K^T y),  y - prox_{f*}^{Sigma^-1}(y + Sigma(Kx+l)) )
vanishes.  For a downstream loss with cotangent g on (x*,y*):
    solve   J_z^T w = g      (J_z = dR/dz, applied matrix-free via jvp/vjp of R),
    return  d(l,c) = - J_theta^T w     (J_theta = dR/d(l,c), via vjp of R in theta).
Only matvec/rmatvec and the block proxes' derivatives are used -- operator-pure,
nothing formed.  Applies to differentiable proxes (linear/quadratic, nonneg, box,
equality, second-order and semidefinite cones); not exp/power (while-loop prox).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def make_diff_solve(solver, matvec, rmatvec, prox_primal, prox_dual, tau, sigma, n, m):
    def raw(l, c):
        db = (l[None], c[None])
        x, y, _, _ = solver(db, jnp.zeros((1, n), jnp.float32), jnp.zeros((1, m), jnp.float32))
        return x[0], y[0]

    def residual(z, l, c):
        xx, yy = z[:n], z[n:]
        rp = xx - prox_primal(xx - tau * rmatvec(yy), tau, (l, c))
        rd = yy - prox_dual(yy + sigma * matvec(xx), sigma, (l, c))
        return jnp.concatenate([rp, rd])

    @jax.custom_vjp
    def solve_from_data(l, c):
        return raw(l, c)

    def fwd(l, c):
        x, y = raw(l, c)
        return (x, y), (x, y, l, c)

    def bwd(res, g):
        x, y, l, c = res
        gx, gy = g
        z = jnp.concatenate([x, y])
        gz = jnp.concatenate([gx, gy])
        Jt = lambda w: jax.vjp(lambda zz: residual(zz, l, c), z)[1](w)[0]   # J_z^T
        w, _ = jax.scipy.sparse.linalg.gmres(Jt, gz, tol=1e-9, atol=0.0,
                                             maxiter=400, restart=50)
        _, vjp_theta = jax.vjp(lambda lc: residual(z, lc[0], lc[1]), (l, c))
        (dl, dc), = vjp_theta(-w)
        return dl, dc

    solve_from_data.defvjp(fwd, bwd)
    return solve_from_data
