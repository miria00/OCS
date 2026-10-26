"""Regime-3 head-to-head on multicommodity flow with piecewise-linear costs.

    minimize    sum_a phi( (sum_k f_k)_a )  +  (mu/2) ||x||_2^2
    subject to  N f_k = d_k,   f >= 0,

x = vec(F), F in R^{E x K} row-major.  OCS applies the block-diagonal incidence
(MultiCommodityFlow) and the arc coupling (CommoditySum) matrix-free.  The
competitor forms them: kron(N, I_K) and kron(I_E, 1_K^T).

The incidence matrix is genuinely sparse, so a sparse solver does not run out of
memory here.  This is therefore a time comparison on the competitor's own
ground, not a memory wedge, and it is reported as such.

Both sides run in this one process on one machine.

Emits plots/mcf_competitors.csv with schema
    K,vars,rows,nnz,ocs_s,ocs_res,ocs_status,ocs_obj,<solver>_s,<solver>_status,<solver>_obj
"""
import os, sys, time, csv
import numpy as np
import scipy.sparse as sp

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.path.join(PLOTS, "mcf_competitors.csv")

V, E, MU = 50, 200, 1e-3
SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float64)
_INTER = [0.0]
for _j, _b in enumerate([0.5, 1.0, 1.5]):
    _INTER.append(_INTER[-1] + (SLOPES[_j] - SLOPES[_j + 1]) * _b)
INTER = np.array(_INTER, np.float64)
SOLVERS = os.environ.get("MCF_SOLVERS", "CUOPT,HIGHS,SCS").split(",")
TIME_CAP = float(os.environ.get("MCF_TIME_CAP", 900))


def network(seed=1):
    r = np.random.default_rng(seed)
    N = np.zeros((V, E), np.float32)
    for e in range(E):
        a, b = r.choice(V, 2, replace=False)
        N[a, e] = 1.0
        N[b, e] = -1.0
    return N


def phi_np(y):
    return np.max(SLOPES[None, :] * np.asarray(y)[:, None] + INTER[None, :], axis=1)


def objective(x, K):
    """One evaluator for both sides, so the compared number is the same number."""
    x = np.asarray(x, dtype=np.float64)
    arc_total = x.reshape(E, K).sum(axis=1)
    return float(phi_np(arc_total).sum() + 0.5 * MU * float(x @ x))


def instance(N, K, seed=1):
    r = np.random.default_rng(seed)
    Ftrue = r.uniform(0.0, 0.6, (E, K)).astype("f")
    return (N @ Ftrue).reshape(-1).astype(np.float64)      # feasible demand


def run_ocs(N, K, d):
    import jax, jax.numpy as jnp, ocs
    n = E * K
    x = ocs.Variable(n)
    dp = ocs.Parameter("d")
    prob = ocs.minimize([
        ocs.nonneg(x),
        ocs.sum_squares(ocs.Diagonal(np.sqrt(MU) * np.ones(n, "f")) @ x),
        ocs.equality(ocs.MultiCommodityFlow(N, K) @ x - dp),
        ocs.pwl(ocs.CommoditySum(E, K) @ x, SLOPES.astype("f"), INTER.astype("f")),
    ])
    comp = prob.compile(block=100, max_blocks=200, tol=1e-6, equilibrate="sgd")
    dj = jnp.asarray(d.astype("f"))
    s = comp.solve_instance({"d": dj}); jax.block_until_ready(s.x)          # warm
    t0 = time.perf_counter()
    s = comp.solve_instance({"d": dj}); jax.block_until_ready(s.x)
    return time.perf_counter() - t0, float(s.residual), s.status, objective(s.x, K)


def run_competitor(N, K, d, solver):
    import cvxpy as cp
    n = E * K
    M_flow = sp.kron(sp.csr_matrix(N.astype(np.float64)), sp.eye(K, format="csr"),
                     format="csr")
    M_sum = sp.kron(sp.eye(E, format="csr"), np.ones((1, K)), format="csr")
    x = cp.Variable(n)
    arc = M_sum @ x
    cost = cp.sum(cp.maximum(*[SLOPES[j] * arc + INTER[j] for j in range(len(SLOPES))]))
    prob = cp.Problem(cp.Minimize(cost + 0.5 * MU * cp.sum_squares(x)),
                      [M_flow @ x == d, x >= 0])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    dt = time.perf_counter() - t0
    obj = objective(x.value, K) if x.value is not None else float("nan")
    return dt, prob.status, obj, M_flow.nnz


if __name__ == "__main__":
    Ks = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16, 32, 64, 128, 256, 512]
    N = network()
    alive = {s.strip(): True for s in SOLVERS}
    fields = ["K", "vars", "rows", "nnz", "ocs_s", "ocs_res", "ocs_status", "ocs_obj"]
    for s in alive:
        fields += [f"{s}_s", f"{s}_status", f"{s}_obj"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for K in Ks:
            d = instance(N, K)
            t, res, st, obj = run_ocs(N, K, d)
            row = {"K": K, "vars": E * K, "rows": V * K, "nnz": 2 * E * K,
                   "ocs_s": round(t, 5), "ocs_res": f"{res:.3e}",
                   "ocs_status": st, "ocs_obj": round(obj, 6)}
            line = f"K={K:5d} vars={E*K:8,d}  OCS {t:8.3f}s {st:11s} obj={obj:12.4f}"
            for s in list(alive):
                if not alive[s]:
                    row[f"{s}_s"], row[f"{s}_status"], row[f"{s}_obj"] = "", "not_attempted", ""
                    continue
                try:
                    dt, cst, cobj, _ = run_competitor(N, K, d, s)
                    row[f"{s}_s"], row[f"{s}_status"] = round(dt, 5), cst
                    row[f"{s}_obj"] = round(cobj, 6)
                    line += f" | {s} {dt:8.3f}s {cst:8s} obj={cobj:12.4f}"
                    if dt > TIME_CAP or cst != "optimal":
                        alive[s] = False
                except Exception as exc:
                    row[f"{s}_s"], row[f"{s}_obj"] = "", ""
                    row[f"{s}_status"] = f"failed:{type(exc).__name__}"
                    line += f" | {s} FAILED {type(exc).__name__}"
                    alive[s] = False
            w.writerow(row); fh.flush(); print(line, flush=True)
            if not any(alive.values()):
                print("every competitor has stopped; ladder ends here", flush=True)
                break
    print(f"\nwrote {OUT}")
