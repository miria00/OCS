"""Killer example at SCALE: optimal transport to N = 10^4 x 10^4 = 10^8 variables.

    minimize_X  <C, X> + (eps/2)||X||_F^2   s.t.  X 1 = p,  X^T 1 = q,  X >= 0.

X is an N x N transport plan (N^2 variables).  The marginal operator
K : vec(X) -> (row sums, col sums) is applied matrix-free; the explicit constraint
matrix would be (2N) x N^2 -- at N = 10^4 that is 2*10^12 entries, hopeless for any
matrix-forming solver.  We validate the transport cost against POT at small N
(exact network simplex) and, at scale, report feasibility + throughput and compare
to POT's Sinkhorn as the specialized-OT ballpark (per Boyd: be in the ballpark, not
winning).  A geometric cost C_ij = (a_i - b_j)^2 from two 1-D point clouds.

Run:  python examples/killer_ot_scale.py
"""
import time
import numpy as np
import jax, jax.numpy as jnp
import ocs

EPS = 0.1


def instance(N, seed=0):
    r = np.random.default_rng(seed)
    a = np.sort(r.uniform(0, 1, N)).astype(np.float32)
    b = np.sort(r.uniform(0, 1, N)).astype(np.float32)
    C = ((a[:, None] - b[None, :]) ** 2).astype(np.float32)     # geometric cost, N^2
    p = r.uniform(0.2, 1, N).astype(np.float32); p /= p.sum()
    q = r.uniform(0.2, 1, N).astype(np.float32); q /= q.sum()
    return C, p, q


def solve_ot(N, tol=1e-4, max_blocks=400):
    K = ocs.MarginalOperator(N, N)
    x = ocs.Variable(N * N); c = ocs.Parameter("c"); mrg = ocs.Parameter("m")
    se = jnp.full(N * N, np.sqrt(EPS), jnp.float32)
    comp = ocs.minimize([ocs.nonneg(x), ocs.equality(K @ x - mrg),
                         ocs.sum_squares(ocs.Diagonal(se) @ x)], cost=c).compile(max_blocks=max_blocks, tol=tol)
    C, p, q = instance(N)
    data = {"c": jnp.asarray(C.reshape(-1)), "m": jnp.asarray(np.concatenate([p, q]))}
    comp.solve_instance(data)                                   # warm (compile)
    t = time.perf_counter(); sol = comp.solve_instance(data); jax.block_until_ready(sol.x)
    dt = time.perf_counter() - t
    X = np.asarray(sol.x).reshape(N, N)
    return dict(N=N, dt=dt, res=sol.residual, obj=float((C * X).sum()),
                row_err=float(np.abs(X.sum(1) - p).max()), col_err=float(np.abs(X.sum(0) - q).max()),
                minX=float(X.min()), C=C, p=p, q=q, X=X)


print(f"{'N':>7} {'vars':>13} {'time(s)':>8} {'residual':>9} {'row err':>9} {'col err':>9}  reference")
print("-" * 88)
for N in [64, 256, 1024, 4096, 10000]:
    R = solve_ot(N)
    ref = ""
    try:
        import ot
        if N <= 512:                                           # exact network simplex
            e = float(ot.emd2(R["p"].astype(np.float64), R["q"].astype(np.float64), R["C"].astype(np.float64)))
            ref = f"POT emd2={e:.4f} (OCS w/o reg~{R['obj']:.4f}, rel {abs(R['obj']-e)/(1+abs(e)):.1e})"
        else:                                                  # Sinkhorn ballpark at scale
            e = float(ot.sinkhorn2(R["p"], R["q"], R["C"], reg=0.01, numItermax=200))
            ref = f"OCS obj={R['obj']:.4f} vs POT Sinkhorn={e:.4f}  (ballpark)"
    except Exception as ex:
        ref = f"(POT n/a: {type(ex).__name__})"
    print(f"{N:>7} {N*N:>13,} {R['dt']:>8.2f} {R['res']:>9.1e} {R['row_err']:>9.1e} {R['col_err']:>9.1e}  {ref}")

print(f"\nAt N=10^4 the transport plan has 10^8 variables and the explicit marginal")
print(f"constraint matrix would be 2*10^4 x 10^8 = 2*10^12 entries -- OCS never forms it.")
