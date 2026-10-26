"""Same-algorithm Kronecker comparison (companion to examples/kron_wedge.py).

Figure F compares OCS applying the factors against a numpy least-squares on
the formed matrix, a different computation.  Here the SAME OCS algorithm runs
on the SAME instance twice: once with the Kronecker factors, exactly as
kron_wedge.run_ocs does (S = 64, lambda = 0.05, its instance generator, its
compile keywords, tol 1e-6), and once with the stored dense matrix
ocs.Dense(np.kron(A, B)) as the operator, everything else identical.  The dense
matrix is formed deliberately: it is the comparison subject.

One process per (arm, T) so that the measured device peak belongs to that arm
alone and a stored-matrix failure cannot take the ladder down.

    python kron_samealg.py --driver --out kron_samealg.csv [--T 4 8 ... 512]
    python kron_samealg.py --arm factors|dense --T 64 --out kron_samealg.csv

Schema: arm,T,S,vars,matrix_gb,build_s,compile_s,first_call_s,warm_s,warm_runs,
        iters,residual,status,objective,peak_gb,note
"""
import os, sys, csv, time, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

FIELDS = ["arm", "T", "S", "vars", "matrix_gb", "build_s", "compile_s", "first_call_s",
          "warm_s", "warm_runs", "iters", "residual", "status", "objective", "peak_gb", "note"]
TOL = 1e-6


def objective(A, B, b, lam, x, S, T):
    X = np.asarray(x, np.float64).reshape(S, T)
    R = A.astype(np.float64) @ X @ B.astype(np.float64).T - b.astype(np.float64).reshape(S, T)
    return float(0.5 * np.sum(R * R) + lam * np.abs(X).sum())


def run(arm, T, reps):
    import jax, jax.numpy as jnp, ocs
    import kron_wedge as kw                                   # S, LAM, BLK, MB, make, peak_gb
    S = kw.S; n = S * T
    A, B, Xt = kw.make(T)
    K = ocs.Kronecker(A, B)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(Xt.reshape(-1))))   # identical b for both arms
    row = {"arm": arm, "T": T, "S": S, "vars": n, "warm_runs": reps, "note": ""}
    try:
        t0 = time.perf_counter()
        if arm == "factors":
            op = ocs.Kronecker(A, B); row["matrix_gb"] = round((S * S + T * T) * 4 / 1e9, 6)
        else:
            Kd = np.kron(A.astype(np.float32), B.astype(np.float32)); assert Kd.shape == (n, n)
            op = ocs.Dense(Kd); jax.block_until_ready(op.A)   # the stored matrix is the subject
            row["matrix_gb"] = round(n * n * 4 / 1e9, 6)
        row["build_s"] = round(time.perf_counter() - t0, 5)
        x = ocs.Variable(n); bp = ocs.Parameter("b")
        t0 = time.perf_counter()
        comp = ocs.minimize([ocs.least_squares(op @ x - bp),
                             ocs.l1(ocs.Identity(n) @ x, kw.LAM)]).compile(
            block=kw.BLK, max_blocks=kw.MB, tol=TOL, equilibrate="sgd")
        row["compile_s"] = round(time.perf_counter() - t0, 5)
        bj = jnp.asarray(b)
        t0 = time.perf_counter()
        s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
        row["first_call_s"] = round(time.perf_counter() - t0, 5)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
            ts.append(time.perf_counter() - t0)
        row.update(warm_s=round(float(np.median(ts)), 5), iters=s.iters,
                   residual=f"{s.residual:.3e}", status=s.status,
                   objective=f"{objective(A, B, b, kw.LAM, np.asarray(s.x), S, T):.8f}",
                   peak_gb=round(kw.peak_gb(), 4))
    except Exception as exc:  # noqa: BLE001
        row.update(status=f"failed:{type(exc).__name__}", note=str(exc).replace("\n", " ")[:300],
                   peak_gb=round(kw.peak_gb(), 4))
    for k in FIELDS:
        row.setdefault(k, "")
    return row


def append(out, row):
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)


def last_row(out):
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--arm", choices=["factors", "dense"])
    ap.add_argument("--T", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.driver:
        dense_alive = True
        for T in a.T:
            for arm in ("factors", "dense"):
                if arm == "dense" and not dense_alive:
                    append(a.out, {**{k: "" for k in FIELDS}, "arm": arm, "T": T, "vars": 64 * T,
                                   "matrix_gb": round((64 * T) ** 2 * 4 / 1e9, 6),
                                   "status": "not_attempted", "note": "stored-matrix arm stopped at the first failure"})
                    continue
                cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm, "--T", str(T),
                       "--reps", str(a.reps), "--out", a.out]
                r = subprocess.run(cmd, capture_output=True, text=True)
                row = last_row(a.out)
                if r.returncode != 0 or row is None or row["arm"] != arm or int(row["T"]) != T:
                    tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                    append(a.out, {**{k: "" for k in FIELDS}, "arm": arm, "T": T, "vars": 64 * T,
                                   "status": f"process_failed:rc={r.returncode}",
                                   "note": " | ".join(tail)[:300]})
                    row = last_row(a.out)
                print(f"{arm:8s} T={T:4d} vars={64*T:6d} warm {row['warm_s']:>9s}s compile {row['compile_s']:>8s}s "
                      f"first {row['first_call_s']:>8s}s iters {row['iters']:>5s} res {row['residual']:>9s} "
                      f"peak {row['peak_gb']:>7s}GB {row['status']} {row['note'][:80]}", flush=True)
                if arm == "dense" and not str(row["status"]).startswith(("solved", "inaccurate")):
                    dense_alive = False
        print("wrote", a.out)
    else:
        append(a.out, run(a.arm, a.T[0], a.reps))
