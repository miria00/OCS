"""Equilibration bake-off: SMFE vs Remedy A (Diag++) vs Remedy B (optimal-diagonal)
vs the Bonus norm, judged on CONDITIONING PER MATVEC.

On operators where we can form A (only to MEASURE ground-truth condition number),
each method is run matrix-free (touching A only via matvec/rmatvec) and compared on
kappa(scaled operator) and the number of operator products it spent.

Remedy B here is a simplified matrix-free optimal-diagonal preconditioner
(Qu-Gao-Hinder-Ye-Zhou lineage): iterate extremal eigenpairs of the scaled Gram
(via scipy eigsh on a matrix-free LinearOperator -- operator-pure, nothing formed)
and descend the diagonal to shrink kappa.  Its cost driver is lambda_min, which is
slow matrix-free -- that is the point of the comparison.

Run:  PYTHONPATH=/home/miria/OCSolver/OCS /home/miria/jaxenv2/bin/python examples/equil_bakeoff.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, eigsh

from ocs.solver.equilibrate_sgd import smfe_equilibration
from ocs.solver.equilibrate_adv import diagpp_equilibration, sketch_norm
import jax


def kappa_scaled(A, tau, sigma):
    D = np.sqrt(np.asarray(sigma)); E = np.sqrt(np.asarray(tau))
    S = D[:, None] * A * E[None, :]
    sv = np.linalg.svd(S, compute_uv=False)
    return sv[0] / sv[-1]


def optdiag(A, steps=8, alpha=0.4):
    """Remedy B (simplified, matrix-free): symmetric optimal-diagonal on M=A^T A.
    Extremal eigenpairs via eigsh (matvec-only), log-diagonal descent on kappa."""
    n = A.shape[1]; d = np.ones(n, float); cnt = [0]
    Aj = A.astype(float)
    for _ in range(steps):
        Dm = d ** -0.5

        def mv(x):
            cnt[0] += 1
            return Dm * (Aj.T @ (Aj @ (Dm * x)))     # D^{-1/2} A^T A D^{-1/2} (2 products)
        M = LinearOperator((n, n), matvec=mv, dtype=float)
        lam_max, V = eigsh(M, k=1, which="LA", maxiter=2000, tol=1e-3)
        lam_min, W = eigsh(M, k=1, which="SA", maxiter=5000, tol=1e-3)
        umax = Dm * V[:, 0]; umin = Dm * W[:, 0]
        gmax = umax ** 2 / (umax ** 2).sum(); gmin = umin ** 2 / (umin ** 2).sum()
        d = d * np.exp(alpha * (gmax - gmin))        # up-weight top-mode coords, down-weight bottom
    return d, cnt[0] * 2                              # products (2 per matvec of M)


def run(name, A):
    n, m = A.shape[1], A.shape[0]
    mv = lambda x: jnp.asarray(A) @ x
    rmv = lambda y: jnp.asarray(A).T @ y
    svA = np.linalg.svd(A, compute_uv=False); k0 = svA[0] / svA[-1]
    print(f"\n== {name}:  raw kappa = {k0:.1f}  (shape {m}x{n}) ==")
    print(f"{'method':10s} {'kappa_scaled':>12s} {'products':>9s}")

    t0 = 0.9 / (svA[0]); tau = np.full(n, t0); sigma = np.full(m, t0)  # scalar ~ 1/||A||
    print(f"{'scalar':10s} {kappa_scaled(A, tau, sigma):12.1f} {'~30':>9s}")

    t, s, nr = smfe_equilibration(mv, rmv, n, m)
    print(f"{'SMFE':10s} {kappa_scaled(A, t, s):12.1f} {'~190':>9s}")

    for k, kh in [(6, 4), (40, 40)]:
        t, s, nr, dg = diagpp_equilibration(mv, rmv, n, m, k=k, kh=kh, diagnostics=True)
        print(f"{'diagpp'+str(k):10s} {kappa_scaled(A, t, s):12.1f} {dg['matvec_products']:9d}")

    try:
        d, prods = optdiag(A, steps=8)
        tau_b = 1.0 / d; sig_b = np.ones(m)            # symmetric one-sided D on columns
        print(f"{'optdiagB':10s} {kappa_scaled(A, tau_b, sig_b):12.1f} {prods:9d}")
    except Exception as e:
        print(f"{'optdiagB':10s} {'FAILED':>12s}  lambda_min matrix-free did not converge "
              f"({type(e).__name__})")


if __name__ == "__main__":
    G = np.random.default_rng(0).normal(size=(240, 180)).astype("f")
    s = (100.0 ** np.random.default_rng(7).uniform(0, 1, size=180)).astype("f")
    run("dense col-scaled R=100", (G * s[None, :]).astype("f"))

    # a two-sided ill-conditioned case
    G2 = np.random.default_rng(1).normal(size=(200, 200)).astype("f")
    rs = (30.0 ** np.random.default_rng(2).uniform(0, 1, size=200)).astype("f")
    cs = (30.0 ** np.random.default_rng(3).uniform(0, 1, size=200)).astype("f")
    run("dense two-sided R=30", (rs[:, None] * G2 * cs[None, :]).astype("f"))
