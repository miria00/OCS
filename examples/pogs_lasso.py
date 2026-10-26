"""Dense lasso: POGS against OCS, the problem class POGS was built for.

    minimize  (1/2)||A x - b||_2^2 + lambda ||x||_1

POGS's GPU path does not build against current CUDA: its source uses
thrust::unary_function, removed in CCCL 2.x which ships from CUDA 12.6, and the
machine has CUDA 13. So POGS runs on CPU here while OCS runs on the GPU of the
same machine. That is not an equal-hardware comparison and must be reported as
such.

Two halves, because POGS and OCS live in different environments on that one
machine. Run with --side pogs under the environment holding pogs_graph and
cvxpy, and with --side ocs under the one holding ocs. Each writes its own CSV.

Schema: size,solver,tol,time_s,objective,rel_obj_err,iters,status
"""
import os, sys, csv, time
import numpy as np
from guarded import run_guarded

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
SIZES = [(800, 1600), (2000, 4000), (4000, 8000)]
TOLS = [1e-3, 1e-4]
LAM_FRAC = 0.1                      # lambda = LAM_FRAC * ||A^T b||_inf
SOLVE_TIMEOUT_S = float(os.environ.get("POGS_TIMEOUT", 1800))


def instance(m, n, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(m, n)) / np.sqrt(m))
    xt = np.zeros(n)
    idx = rng.choice(n, max(1, n // 100), replace=False)
    xt[idx] = rng.normal(size=idx.size)
    b = A @ xt + 0.01 * rng.normal(size=m)
    lam = LAM_FRAC * np.abs(A.T @ b).max()
    return A, b, lam


def objective(A, b, lam, x):
    x = np.asarray(x, dtype=np.float64).ravel()
    r = A @ x - b
    return float(0.5 * r @ r + lam * np.abs(x).sum())


def reference(A, b, lam):
    """Clarabel in double precision, the accuracy yardstick for both sides."""
    import cvxpy as cp
    x = cp.Variable(A.shape[1])
    p = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lam * cp.norm1(x)))
    p.solve(solver=cp.CLARABEL)
    return float(p.value), np.asarray(x.value)


def pogs_solve(A, b, lam, tol):
    """POGS CPU through its own Python wrapper; module-level for the guard."""
    sys.path.insert(0, "/tmp/pogs/python")
    import pogs_graph as pg
    t0 = time.perf_counter()                    # timed here, inside the child,
    out = pg.solve_lasso(A, b, lam, abs_tol=tol, rel_tol=tol, max_iter=20000)
    dt = time.perf_counter() - t0                # so spawn and pickling are excluded
    return {"x": np.asarray(out["x"]), "iterations": out.get("iterations", ""),
            "status": out.get("status", ""), "seconds": dt}


def run_pogs(rows):
    sys.path.insert(0, "/tmp/pogs/python")
    import pogs_graph as pg
    print("POGS library:", pg._find_shared_library(), flush=True)
    for m, n in SIZES:
        A, b, lam = instance(m, n)
        g = run_guarded(reference, args=(A, b, lam), label=f"Clarabel {m}x{n}",
                        timeout_s=SOLVE_TIMEOUT_S)
        if g["status"] != "ok":
            rows.append([f"{m}x{n}", "REFERENCE", "", "", "", "", "", g["status"]])
            print(f"  {m}x{n} reference {g['status']}; skipping size", flush=True)
            continue
        ref_val, _ = g["value"]
        print(f"  {m}x{n} Clarabel reference objective {ref_val:.8f}", flush=True)
        for tol in TOLS:
            g = run_guarded(pogs_solve, args=(A, b, lam, tol), label=f"POGS {m}x{n} tol={tol:g}",
                            timeout_s=SOLVE_TIMEOUT_S)
            if g["status"] != "ok":
                rows.append([f"{m}x{n}", "POGS", tol, "", "", "", "", g["status"]])
                continue
            out, dt = g["value"], g["value"]["seconds"]
            obj = objective(A, b, lam, out["x"])
            rows.append([f"{m}x{n}", "POGS", tol, round(dt, 4), round(obj, 8),
                         f"{abs(obj-ref_val)/max(abs(ref_val),1e-12):.2e}",
                         out["iterations"], out["status"]])
            print(f"    POGS tol={tol:g} {dt:8.3f}s obj={obj:.6f} "
                  f"rel={rows[-1][5]} {rows[-1][7]}", flush=True)
        rows.append([f"{m}x{n}", "REFERENCE", "", "", round(ref_val, 8), "0", "", "clarabel"])


def run_ocs(rows):
    import jax, jax.numpy as jnp, ocs
    for m, n in SIZES:
        A, b, lam = instance(m, n)
        for tol in TOLS:
            x = ocs.Variable(n); bp = ocs.Parameter("b")
            comp = ocs.minimize([ocs.least_squares(ocs.Dense(A.astype("f")) @ x - bp),
                                 ocs.l1(ocs.Identity(n) @ x, float(lam))]).compile(
                block=100, max_blocks=400, tol=tol, equilibrate="sgd")
            bj = jnp.asarray(b.astype("f"))
            s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)     # warm
            t0 = time.perf_counter()
            s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
            dt = time.perf_counter() - t0
            obj = objective(A, b, lam, s.x)
            rows.append([f"{m}x{n}", "OCS", tol, round(dt, 4), round(obj, 8),
                         "", s.iters, s.status])
            print(f"    OCS  tol={tol:g} {dt:8.3f}s obj={obj:.6f} "
                  f"iters={s.iters} {s.status}", flush=True)


if __name__ == "__main__":
    side = sys.argv[sys.argv.index("--side") + 1] if "--side" in sys.argv else "pogs"
    rows = []
    (run_pogs if side == "pogs" else run_ocs)(rows)
    out = os.path.join(PLOTS, f"pogs_lasso_{side}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "solver", "tol", "time_s", "objective",
                    "rel_obj_err", "iters", "status"])
        w.writerows(rows)
    print(f"\nwrote {out}")
