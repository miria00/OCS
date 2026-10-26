"""Solution refinement (Busseti, Moursi, Boyd, COAP 2019, adapted).

After the first-order solve reaches its modest tolerance, refine the point by a
few matrix-free Gauss-Newton steps on the fixed-point residual map
    R(z) = ( x - prox_primal(x - tau K^T y),  y - prox_dual(y + sigma K x) ),  z=(x,y),
whose zero is exactly an optimal primal-dual pair.  Each step solves the
Gauss-Newton normal equations  (J^T J + reg I) delta = -J^T R(z)  by conjugate
gradients, with J and J^T applied matrix-free through jax.jvp / jax.vjp of R -- so
the operator is still touched only through matvec/rmatvec (inside R) and the block
proxes' derivatives; nothing is formed.  This lifts a 1e-3 solve toward high
accuracy at a cost small compared to the solve.

Applies to blocks whose prox is differentiable (linear/quadratic, nonneg, box,
equality, second-order and semidefinite cones); the exponential and power cones,
whose projection is a while-loop Newton root-find, are not auto-differentiated and
are skipped by the caller.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _cg(Aop, b, iters):
    x = jnp.zeros_like(b)
    r = b
    p = r
    rs = jnp.vdot(r, r)

    def body(_, st):
        x, r, p, rs = st
        Ap = Aop(p)
        a = rs / (jnp.vdot(p, Ap) + 1e-30)
        x = x + a * p
        r = r - a * Ap
        rs2 = jnp.vdot(r, r)
        p = r + (rs2 / (rs + 1e-30)) * p
        return (x, r, p, rs2)

    x, _, _, _ = jax.lax.fori_loop(0, iters, body, (x, r, p, rs))
    return x


def gauss_newton_refine(R, z, steps=3, cg_iters=80, reg=1e-8):
    """Refine z toward a zero of R by matrix-free Gauss-Newton (CG on J^T J)."""
    def one(_, z):
        Rz = R(z)
        Jmv = lambda v: jax.jvp(R, (z,), (v,))[1]
        _, vjp = jax.vjp(R, z)
        JT = lambda w: vjp(w)[0]
        Aop = lambda v: JT(Jmv(v)) + reg * v
        delta = _cg(Aop, -JT(Rz), cg_iters)
        return z + delta
    return jax.lax.fori_loop(0, steps, one, z)
