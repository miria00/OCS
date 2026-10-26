"""Sylvester accuracy against conditioning: the paper's fixed shift versus a
size-scaled shift, at the 512x512 size where the A100 audit found 7.16 percent
forward error.

The operator X -> AX + XB has eigenvalues lam_i(A) + lam_j(B), so kappa_2 is
exact and cheap. With A = 0.5 G + 5 I the Gaussian's spectral radius 0.5 sqrt(p)
overtakes the shift past p = 100 and eigenvalue sums approach zero. A shift of
0.5 sqrt(p) + 5 keeps kappa_2 below 8 at every size tested.

Forward error is measured against scipy.linalg.solve_sylvester in float64.
Accuracy is a numerical property, so this may run on CPU; the paper's timings
stay on the 5090.
"""
import os, sys, time, json
import numpy as np
from scipy.linalg import solve_sylvester
import jax, jax.numpy as jnp
import ocs

P = int(os.environ.get("SYL_P", 512))
TOL = float(os.environ.get("SYL_TOL", 1e-6))
BLK, MB = 100, int(os.environ.get("SYL_MAX_BLOCKS", 300))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots",
                   "sylvester_conditioning.json")


def instance(p, shift, seed=0):
    r = np.random.default_rng(seed)
    A = (0.5 * r.normal(size=(p, p)) + shift * np.eye(p)).astype(np.float32)
    B = (0.5 * r.normal(size=(p, p)) + shift * np.eye(p)).astype(np.float32)
    Xt = r.normal(size=(p, p)).astype(np.float32)
    C = (A.astype(np.float64) @ Xt + Xt @ B.astype(np.float64)).astype(np.float32)
    return A, B, Xt, C


def kappa2(A, B, iters=150, seed=1):
    """Spectral condition number of S(X) = AX + XB from its SINGULAR values.
    The eigenvalues lambda_i(A) + lambda_j(B) fix invertibility but not kappa_2
    for nonnormal A, B (a 2x2 counterexample has eigenvalue ratio 1 and kappa_2
    unbounded). Small sizes: dense SVD of the pq x pq representation. Large:
    power iteration on S*S for sigma_max and on (S*S)^{-1} for sigma_min, with
    S^{-1} and S^{-*} applied exactly by Bartels-Stewart (scipy solve_sylvester)."""
    A, B = A.astype(np.float64), B.astype(np.float64); p, q = A.shape[0], B.shape[0]
    if p * q <= 4096:
        M = np.kron(A, np.eye(q)) + np.kron(np.eye(p), B.T)
        sv = np.linalg.svd(M, compute_uv=False); return float(sv[0] / sv[-1])
    r = np.random.default_rng(seed)
    S = lambda X: A @ X + X @ B
    St = lambda Y: A.T @ Y + Y @ B.T
    X = r.normal(size=(p, q)); X /= np.linalg.norm(X)
    for _ in range(iters):
        Y = St(S(X)); lam_max = np.sum(X * Y); X = Y / np.linalg.norm(Y)
    X = r.normal(size=(p, q)); X /= np.linalg.norm(X)
    for _ in range(iters):
        W = solve_sylvester(A, B, solve_sylvester(A.T, B.T, X))
        lam_inv = np.sum(X * W); X = W / np.linalg.norm(W)
    return float(np.sqrt(lam_max * lam_inv))


def solve_ocs(A, B, C, p):
    x = ocs.Variable(p * p); cp_ = ocs.Parameter("c")
    comp = ocs.minimize([ocs.least_squares(ocs.Sylvester(A, B) @ x - cp_)]).compile(
        block=BLK, max_blocks=MB, tol=TOL, equilibrate="sgd")
    cj = jnp.asarray(C.reshape(-1))
    t0 = time.perf_counter(); s = comp.solve_instance({"c": cj}); jax.block_until_ready(s.x)
    return np.asarray(s.x, np.float64).reshape(p, p), float(s.residual), s.iters, s.status, time.perf_counter() - t0


if __name__ == "__main__":
    rows = []
    for label, shift in (("paper: shift 5", 5.0), ("scaled: 0.5*sqrt(p)+5", 0.5 * np.sqrt(P) + 5)):
        A, B, Xt, C = instance(P, shift)
        k = kappa2(A, B)
        Xr = solve_sylvester(A.astype(np.float64), B.astype(np.float64), C.astype(np.float64))
        Xo, res, it, st, dt = solve_ocs(A, B, C, P)
        fwd = float(np.linalg.norm(Xo - Xr) / np.linalg.norm(Xr))
        eqn = float(np.linalg.norm(A.astype(np.float64) @ Xo + Xo @ B.astype(np.float64) - C) / np.linalg.norm(C))
        rows.append(dict(recipe=label, p=P, kappa2=k, residual=res, iters=it, status=st,
                         forward_error=fwd, equation_residual=eqn, seconds=dt))
        print(f"  {label:24s} kappa2={k:9.1f}  residual={res:.2e} ({st}, {it} it)  "
              f"forward err={fwd:.3e}  eqn resid={eqn:.2e}  {dt:.0f}s", flush=True)
    json.dump(rows, open(OUT, "w"), indent=2); print("wrote", OUT)
