"""Regime-1 measured wedge for the Kronecker / matrix-variable family.

Replaces the closed-form storage figure with an experiment.  Both sides solve
the same problem,

    minimize  (1/2) || (A kron B) vec(X) - b ||_2^2  +  lam || vec(X) ||_1,

with X of size S x T.  OCS applies A and B to the S x T block and never forms
the (ST) x (ST) Kronecker matrix.  The competitor must form it, so its ladder
ends where device or host memory ends.  We report measured solve time, measured
peak device memory, and the point at which forming the matrix stops working.

Emits plots/kron_wedge.csv with schema
    T,S,vars,ocs_s,ocs_peak_gb,ocs_res,ocs_status,dense_build_s,dense_solve_s,dense_gb,dense_status
"""
import os, sys, time, csv, gc
from guarded import run_guarded
import numpy as np
import jax, jax.numpy as jnp
import ocs

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.environ.get("KRON_OUT") or os.path.join(PLOTS, "kron_wedge.csv")

S = 64
LAM = 0.05
BLK, MB = 100, 30
# The explicit side is stopped by declared budgets rather than by thrashing the
# host: both numbers are reported with the figure.
DENSE_MAX_GB = float(os.environ.get("DENSE_MAX_GB", 100))
DENSE_TIME_BUDGET = float(os.environ.get("DENSE_TIME_BUDGET", 900))


def peak_gb():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return float("nan")


def make(T, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(S, S)).astype("f") / np.sqrt(S)
    B = rng.normal(size=(T, T)).astype("f") / np.sqrt(T)
    Xt = np.zeros((S, T), dtype="f")
    idx = rng.choice(S * T, size=max(1, (S * T) // 50), replace=False)
    Xt.reshape(-1)[idx] = rng.normal(size=idx.size)
    return A, B, Xt


def run_ocs(T):
    A, B, Xt = make(T)
    n = S * T
    K = ocs.Kronecker(A, B)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(Xt.reshape(-1))))
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    comp = ocs.minimize([ocs.least_squares(ocs.Kronecker(A, B) @ x - bp),
                         ocs.l1(ocs.Identity(n) @ x, LAM)]).compile(
        block=BLK, max_blocks=MB, tol=1e-6, equilibrate="sgd")   # held tighter than the 1e-3 bar, deliberately
    bj = jnp.asarray(b)
    s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)     # warm
    t0 = time.perf_counter()
    s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
    return time.perf_counter() - t0, peak_gb(), float(s.residual), s.status, A, B, b


def run_dense(T, A, B, b):
    """The competitor's unavoidable step: build the explicit (ST)x(ST) matrix.
    Reported in float32, the most generous choice for it."""
    n = S * T
    need = (n * n) * 4 / 1e9
    if need > DENSE_MAX_GB:
        return "", "", round(need, 4), "exceeds_memory_budget"
    try:
        t0 = time.perf_counter()
        Kd = np.kron(A.astype(np.float32), B.astype(np.float32))
        build = time.perf_counter() - t0
        assert Kd.shape == (n, n)
        t0 = time.perf_counter()
        # one dense normal-equation solve: the cheapest thing the explicit form
        # could possibly do, so the comparison is not stacked against it
        _ = np.linalg.lstsq(Kd, b, rcond=None)[0]
        solve = time.perf_counter() - t0
        del Kd; gc.collect()
        return round(build, 4), round(solve, 4), round(need, 4), "formed"
    except (MemoryError, np.core._exceptions._ArrayMemoryError):
        gc.collect()
        return "", "", round(need, 4), "out_of_memory"
    except Exception as exc:
        gc.collect()
        return "", "", round(need, 4), f"failed:{type(exc).__name__}"


if __name__ == "__main__":
    Ts = [int(a) for a in sys.argv[1:]] or [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    dense_alive = True
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T", "S", "vars", "ocs_s", "ocs_peak_gb", "ocs_res", "ocs_status",
                    "dense_build_s", "dense_solve_s", "dense_gb", "dense_status"])
        for T in Ts:
            t, pk, res, st, A, B, b = run_ocs(T)
            if dense_alive:
                need = (S * T) ** 2 * 4 / 1e9
                # the lstsq workspace is several times the matrix; cap the child
                # at that plus interpreter baseline, and bound its wall clock
                g = run_guarded(run_dense, args=(T, A, B, b), label=f"dense T={T}",
                                timeout_s=DENSE_TIME_BUDGET, mem_gb=need * 5 + 3)
                if g["status"] == "ok":
                    db, ds, dg, dst = g["value"]
                else:
                    db, ds, dg = "", "", round(need, 4)
                    dst = {"timeout": f"exceeded_{DENSE_TIME_BUDGET:.0f}s",
                           "out_of_memory": "out_of_memory",
                           "killed_out_of_memory": "killed_out_of_memory"}.get(
                               g["status"], g["status"])
                if dst != "formed":
                    dense_alive = False
                elif float(ds) > DENSE_TIME_BUDGET:
                    dense_alive = False
                    print(f"  explicit form exceeded the {DENSE_TIME_BUDGET:.0f}s budget; "
                          f"not attempted beyond T={T}", flush=True)
            else:
                db, ds, dg, dst = "", "", round((S * T) ** 2 * 4 / 1e9, 4), "not_attempted"
            w.writerow([T, S, S * T, round(t, 5), round(pk, 4), f"{res:.3e}", st,
                        db, ds, dg, dst]); fh.flush()
            print(f"T={T:6d}  vars={S*T:9,d}  OCS {t:8.3f}s peak {pk:6.3f}GB {st:12s} | "
                  f"dense {dg:10.3f}GB {dst}"
                  + (f" build {db}s solve {ds}s" if dst == "formed" else ""), flush=True)
    print(f"\nwrote {OUT}")
