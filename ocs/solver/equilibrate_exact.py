"""Exact matrix-free equilibration via the squared-operator oracle.

The stochastic equilibration (SMFE) exists because row/column norms of the
scaled operator are not directly readable through matvec/rmatvec: they are
estimated with Rademacher probes.  But many structured operators can apply the
ENTRYWISE-SQUARED operator M = |K|^{o2} exactly, at matvec cost:

    0/1 operators (marginals, incidence)  : M = K itself;
    diagonal / weighted                   : square the weights;
    dense (validation)                    : square the entries;
    stencils with entries in {+-1}        : the all-ones stencil.

With that oracle, row and column squared-norms of DKE are EXACT:
row_sums(D K E) = D^2 M (e^{o2}), col_sums = M^T (d^{o2}) E^2 -- so classical
RAS/Sinkhorn alternation solves the equilibration problem deterministically,
with linear convergence on epsilon-scalable operators (Proposition p-scal
gives the exact scalability condition).  Zero variance, no probes; the only
stochastic element left is the final power-iteration norm estimate for the
PDHG safety rescale, shared with SMFE.

This does not extend the diagonal class (the van der Sluis ceiling stands);
it reaches the same optimum exactly and cheaply when structure permits, and
falls back to SMFE otherwise.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .equilibrate import _power_norm


def exact_equilibration(sq_matvec, sq_rmatvec, matvec, rmatvec, n, m,
                        scalar_spans=(), *, iters=40, safety=0.9, seed=0,
                        M_clip=6.0):
    """Deterministic RAS equilibration in log space.

    sq_matvec / sq_rmatvec apply M = |K|^{o2} and M^T exactly.
    Returns (tau, sigma, scaled_norm) like ``smfe_equilibration``:
    tau = e^{2v} (primal, per column), sigma = e^{2u} (dual, per row),
    both multiplied by the shared spectral safety rescale.
    """
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)
    r_tgt = jnp.sqrt(jnp.float32(n) / jnp.float32(m))
    s_tgt = jnp.sqrt(jnp.float32(m) / jnp.float32(n))

    def body(_t, carry):
        u, v = carry
        # exact row sums of D M E in the squared variables
        row = jnp.exp(2.0 * u) * sq_matvec(jnp.exp(2.0 * v))
        w_u = u + 0.5 * (jnp.log(r_tgt) - jnp.log(row + 1e-30))   # unclipped
        u = jnp.clip(w_u, -M_clip, M_clip)
        for (a, b) in spans:                      # scalar-class blocks share one scale
            # mean of the unclipped tentative point, then clip the scalar
            u = u.at[a:b].set(jnp.clip(jnp.mean(w_u[a:b]), -M_clip, M_clip))
        col = jnp.exp(2.0 * v) * sq_rmatvec(jnp.exp(2.0 * u))
        v = jnp.clip(v + 0.5 * (jnp.log(s_tgt) - jnp.log(col + 1e-30)), -M_clip, M_clip)
        return (u, v)

    @jax.jit
    def run():
        u0 = jnp.zeros(m, jnp.float32)
        v0 = jnp.zeros(n, jnp.float32)
        u, v = jax.lax.fori_loop(0, iters, body, (u0, v0))
        tau = jnp.exp(2.0 * v)
        sigma = jnp.exp(2.0 * u)
        st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
        nrm = _power_norm(lambda x: sg * matvec(st * x),
                          lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        scale = safety / (nrm + 1e-30)
        return tau * scale, sigma * scale, nrm

    tau, sigma, nrm = run()
    return tau, sigma, float(nrm)
