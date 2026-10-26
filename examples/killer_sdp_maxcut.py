"""Killer example: MaxCut SDP, operator-native.

    maximize  <L/4, X>   s.t.  diag(X) = 1,  X psd,      (Goemans-Williamson relaxation)

X is a k-by-k matrix variable, so the constraint matrix of the vectorized problem
is never formed: the only linear map is diagonal extraction diag: vec(X) -> diag(X),
which OCS applies matrix-free (a gather; its adjoint scatters a vector onto the
diagonal).  The PSD cone is handled by the block eigendecomposition projection.
"Nobody does SDP matrix-free"; here the data/constraint operator is.

We validate the SDP value against CVXPY/SCS on a small graph, then show it runs at
larger k where the vectorized constraint matrix (k x k^2) would be wasteful to form.

Run:  PYTHONPATH=/home/miria/OCSolver/OCS /home/miria/jaxenv2/bin/python examples/killer_sdp_maxcut.py
"""
import time
import numpy as np
import jax.numpy as jnp
import ocs


def diag_extract(k):
    """Operator-native diag: vec(X) (k*k) -> diagonal (k). Adjoint is free."""
    return ocs.LinearOperator.from_forward(
        lambda x: jnp.diag(x.reshape(k, k)), n=k * k, m=k)


def laplacian(k, seed):
    r = np.random.default_rng(seed)
    W = r.uniform(0.0, 1.0, size=(k, k)).astype("f")
    W = np.triu(W, 1); W = W + W.T                      # symmetric, zero diagonal
    L = np.diag(W.sum(1)) - W                           # graph Laplacian
    return L.astype("f")


def solve_ocs(L, max_blocks=4000, tol=1e-3):
    k = L.shape[0]
    D = diag_extract(k)
    x = ocs.Variable(k * k)
    ones = ocs.Parameter("ones"); c = ocs.Parameter("c")
    prob = ocs.minimize([ocs.equality(D @ x - ones), ocs.psd(x, k)], cost=c)
    comp = prob.compile(max_blocks=max_blocks, tol=tol)
    data = {"ones": np.ones(k, "f"), "c": (-L / 4.0).reshape(-1)}   # min <-L/4, X> = max <L/4, X>
    comp.solve_instance(data)                                        # warm
    t0 = time.perf_counter(); sol = comp.solve_instance(data); dt = time.perf_counter() - t0
    X = np.asarray(sol.x).reshape(k, k)
    val = float(np.tensordot(L / 4.0, X, 2))            # <L/4, X> = SDP objective (the cut bound)
    return val, sol.residual, sol.iters, dt, X


def solve_cvxpy(L):
    import cvxpy as cp
    k = L.shape[0]
    X = cp.Variable((k, k), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4.0), [cp.diag(X) == 1])
    prob.solve(solver=cp.SCS)
    return float(prob.value)


if __name__ == "__main__":
    print("== validation vs CVXPY/SCS ==")
    for k in (10, 20, 30):
        L = laplacian(k, seed=k)
        val, res, iters, dt, X = solve_ocs(L)
        ref = solve_cvxpy(L)
        relerr = abs(val - ref) / (1 + abs(ref))
        dmax = float(np.abs(np.diag(X) - 1).max())
        wmin = float(np.linalg.eigvalsh(0.5 * (X + X.T)).min())
        print(f"k={k:3d}  OCS={val:9.3f}  SCS={ref:9.3f}  relerr={relerr:.2e}  "
              f"res={res:.1e} iters={iters}  diag_err={dmax:.1e} min_eig={wmin:+.1e}  ({dt:.2f}s)")

    print("\n== scale (matrix-free diag; forming the k x k^2 constraint matrix avoided) ==")
    for k in (100, 200):
        L = laplacian(k, seed=k)
        val, res, iters, dt, X = solve_ocs(L, max_blocks=3000)
        print(f"k={k:3d}  vars={k*k:,}  (constraint matrix would be {k}x{k*k:,})  "
              f"SDP value={val:.1f}  res={res:.1e} iters={iters}  ({dt:.2f}s)")
