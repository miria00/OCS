"""Regime-1 competitor line for the SOCP-representable isotropic-TV MRI wedge.

OCS never forms a matrix.  A conic solver must.  This harness measures exactly
that cost: it densifies the same operators OCS accesses only through matvec,
hands the explicit problem to CVXPY/SCS, and records where that stops being
possible.  Emits plots/competitors_mri_time.csv, which make_plots.D_mri_scale
already reads.

Schema: cones,L,m,scs_solve_s,build_s,status,objective
"""
import os, sys, time, csv, json, signal, subprocess, traceback
import numpy as np
import jax, jax.numpy as jnp
import ocs

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.path.join(PLOTS, "competitors_mri_time.csv")
LAM, GS = 0.03, 2
TIME_CAP = float(os.environ.get("SCS_TIME_CAP", 1800))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# SCS's own time_limit_secs was not honoured in a measured run (it returned after
# 4918 s under an 1800 s setting), so the cap is enforced here as well.
signal.signal(signal.SIGALRM, _alarm)


def densify(op, n, chunk=256):
    """Materialize op as a dense array by probing it with identity columns.
    This is the cost the competitor cannot avoid; OCS never pays it."""
    cols = []
    mv = jax.jit(jax.vmap(op.matvec))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        I = jnp.zeros((e - s, n), dtype=jnp.float32).at[jnp.arange(e - s), jnp.arange(s, e)].set(1.0)
        cols.append(np.asarray(mv(I)))
    return np.concatenate(cols, axis=0).T          # (m, n)


def run(L):
    import cvxpy as cp
    n = L * L
    F = ocs.FFTMask.random(n, 0.35, seed=0)
    Gr = ocs.Gradient2D(L, L)
    y = np.random.default_rng(0).normal(size=F.shape[0]).astype(np.float64)

    t0 = time.perf_counter()
    Fm = densify(F, n).astype(np.float64)
    Gm = densify(Gr, n).astype(np.float64)
    build_s = time.perf_counter() - t0

    x = cp.Variable(n)
    groups = cp.reshape(Gm @ x, (Gm.shape[0] // GS, GS), order="C")
    obj = 0.5 * cp.sum_squares(Fm @ x - y) + LAM * cp.sum(cp.norm(groups, axis=1))
    prob = cp.Problem(cp.Minimize(obj))

    t0 = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, TIME_CAP)
    try:
        prob.solve(solver=cp.SCS, verbose=False, time_limit_secs=TIME_CAP)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    solve_s = time.perf_counter() - t0
    return dict(cones=n, L=L, m=Fm.shape[0], scs_solve_s=round(solve_s, 4),
                build_s=round(build_s, 4),
                status="solved" if prob.status == "optimal" else prob.status,
                objective=(round(float(prob.value), 6) if prob.value is not None else ""))


def _child(L):
    """Run one size and emit its row as JSON. Run in its own process so that an
    out-of-memory kill is observable to the parent instead of ending the ladder
    silently, which is exactly what happened at L=128 on a 62 GB machine."""
    try:
        r = run(L)
    except MemoryError:
        r = dict(cones=L * L, L=L, m="", scs_solve_s="", build_s="",
                 status="out_of_memory", objective="")
    except Timeout:
        r = dict(cones=L * L, L=L, m="", scs_solve_s="", build_s="",
                 status=f"exceeded_{TIME_CAP:.0f}s_cap", objective="")
    except Exception as exc:
        traceback.print_exc()
        r = dict(cones=L * L, L=L, m="", scs_solve_s="", build_s="",
                 status=f"failed:{type(exc).__name__}", objective="")
    print("ROW " + json.dumps(r), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        _child(int(sys.argv[2])); sys.exit(0)

    Ls = [int(a) for a in sys.argv[1:]] or [16, 24, 32, 48, 64, 96, 128]
    rows = []
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cones", "L", "m", "scs_solve_s",
                                           "build_s", "status", "objective"])
        w.writeheader()
        for L in Ls:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                   "--one", str(L)],
                                  capture_output=True, text=True)
            row = [ln for ln in proc.stdout.splitlines() if ln.startswith("ROW ")]
            if row:
                r = json.loads(row[-1][4:])
            elif proc.returncode < 0 or proc.returncode == 137:
                # killed by a signal: on this machine that means the kernel
                # out-of-memory killer took it, which is the datum we want
                r = dict(cones=L * L, L=L, m="", scs_solve_s="", build_s="",
                         status="killed_out_of_memory", objective="")
            else:
                r = dict(cones=L * L, L=L, m="", scs_solve_s="", build_s="",
                         status=f"failed:exit{proc.returncode}", objective="")
            rows.append(r); w.writerow(r); fh.flush()
            print(f"L={L:5d}  n={L*L:9,d}  build={r['build_s'] or '-':>9}s  "
                  f"SCS={r['scs_solve_s'] or '-':>10}s  {r['status']}", flush=True)
            if r["status"] != "solved":
                print("competitor stopped scaling; ladder ends here", flush=True); break
            if float(r["scs_solve_s"]) > TIME_CAP:
                print(f"competitor exceeded {TIME_CAP}s cap; ladder ends here", flush=True); break
    print(f"\nwrote {OUT} ({len(rows)} rows)")
