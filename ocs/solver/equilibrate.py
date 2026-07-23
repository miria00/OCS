"""Matrix-free equilibration / preconditioning: estimate the conditioning of K
from operator probes only, never its entries.

Two levels:
  * ``scalar_steps``       : power iteration for ||K||_2, one scalar step.
  * ``diagonal_equilibration`` : Hutchinson probing for the diagonals of K^T K
    (column energies) and K K^T (row energies), giving per-coordinate steps
    tau_j ~ 1/||K_{:,j}||_2 and sigma_i ~ 1/||K_{i,:}||_2 -- Ruiz/Jacobi-style
    diagonal preconditioning WITHOUT the matrix.  Row groups that belong to a
    coupled (scalar-class) block are forced to share one uniform sigma, so the
    metric handed to each block stays inside its admissible scaling class.  A
    final power iteration on the SCALED operator rescales both steps so that
    the PDHG stability bound ||D_sigma^{1/2} K D_tau^{1/2}|| < 1 holds.

All probing is O(probes) matvec+rmatvec pairs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _power_norm(mv, rmv, n, *, iters=20, seed=0):
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n,), jnp.float32)
    v = v / (jnp.linalg.norm(v) + 1e-30)
    s = jnp.asarray(1.0, jnp.float32)

    def step(_, carry):
        v, _s = carry
        w = rmv(mv(v))
        s = jnp.linalg.norm(w) + 1e-30
        return (w / s, s)

    _v, s = jax.lax.fori_loop(0, iters, step, (v, s))
    return jnp.sqrt(s)


def operator_norm(matvec, rmatvec, n, *, iters=20, seed=0):
    return _power_norm(matvec, rmatvec, n, iters=iters, seed=seed)


def scalar_steps(matvec, rmatvec, n, *, safety=0.99, iters=20, seed=0):
    nrm = operator_norm(matvec, rmatvec, n, iters=iters, seed=seed)
    step = safety / (nrm + 1e-30)
    return step, step, float(nrm)


def diagonal_equilibration(matvec, rmatvec, n, m, scalar_spans=(), *,
                           probes=32, safety=0.9, seed=0):
    """Matrix-free diagonal (Jacobi) preconditioning.

    Hutchinson probes estimate the column energies diag(K^T K) and row energies
    diag(K K^T) from matvec/rmatvec only, giving per-coordinate steps
    tau_j ~ 1/||K_{:,j}||_2 and sigma_i ~ 1/||K_{i,:}||_2.  A modest clamp keeps a
    near-zero row/column (e.g. the imaginary part of a real FFT frequency) from
    producing an enormous step, and coupled (scalar-class) blocks share one
    uniform sigma so the metric stays inside their admissible tier.  A final power
    iteration on the scaled operator enforces the PDHG stability bound.

    This helps on moderately ill-conditioned operators (see
    examples/equilibration_ablation.py); at extreme conditioning the clamp bounds
    the correction.  Robust Ruiz iteration is future work.

    scalar_spans : iterable of (start, stop) row ranges sharing one uniform sigma.
    """
    key = jax.random.PRNGKey(seed)
    col = jnp.zeros(n, jnp.float32)
    row = jnp.zeros(m, jnp.float32)

    def rade(k, d):
        return jnp.where(jax.random.bernoulli(k, 0.5, (d,)), 1.0, -1.0).astype(jnp.float32)

    for _ in range(probes):
        key, ka, kb = jax.random.split(key, 3)
        xi = rade(ka, n)
        col = col + xi * rmatvec(matvec(xi))          # diag(K^T K)
        eta = rade(kb, m)
        row = row + eta * matvec(rmatvec(eta))        # diag(K K^T)
    col = col / probes
    row = row / probes
    col = jnp.clip(col, jnp.max(col) * 1e-4, jnp.max(col))
    row = jnp.clip(row, jnp.max(row) * 1e-4, jnp.max(row))

    tau = 1.0 / jnp.sqrt(col)
    sigma = 1.0 / jnp.sqrt(row)
    for (s, e) in scalar_spans:
        sigma = sigma.at[s:e].set(1.0 / jnp.sqrt(jnp.mean(row[s:e])))

    st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
    nrm = _power_norm(lambda x: sg * matvec(st * x),
                      lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 1)
    scale = safety / (nrm + 1e-30)
    return tau * scale, sigma * scale, float(nrm)
