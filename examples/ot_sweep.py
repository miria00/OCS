"""Regime-1 OT wedge: OCS matrix-free marginal operator vs POT Sinkhorn.
Emits plots/ot_wedge.csv with the schema H_ot_wedge expects."""
import os, sys, time, csv
from guarded import run_guarded
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
from _quadratic_ot import family, instance, data, diagnostics, pot_reference

def pot_timed(c, p, q):
    """Solve time measured inside the guard's child, excluding spawn and pickling."""
    t0 = time.perf_counter(); pot_reference(c, p, q); return time.perf_counter() - t0


PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
SIZES = [int(a) for a in (sys.argv[1:] or [64, 128, 256, 512, 1024, 2048, 4096, 10000])]

def main():
    # Everything below runs only in the parent: the guard spawns children
    # that re-import this module, so no work may sit at module level.
    rows = []
    for N in SIZES:
        rng = np.random.default_rng(0)
        try:
            comp = family(N)
            c, p, q = instance(N, rng)
            d = data(c, p, q)
            out = comp.solve_instance(d); jax.block_until_ready(out.x)      # warm
            t0 = time.perf_counter(); out = comp.solve_instance(d); jax.block_until_ready(out.x)
            ocs_s = time.perf_counter() - t0
            diag = diagnostics(c, p, q, out.x, getattr(out, "y", None))
        except Exception as e:
            print(f"N={N}: OCS FAILED {type(e).__name__}: {e}", flush=True); break
        # the guard returns a row on every outcome, so no try/except is needed
        g = run_guarded(pot_timed, args=(c, p, q), label=f"POT N={N}",
                        timeout_s=float(os.environ.get("OT_COMPETITOR_TIMEOUT", 2400)))
        sink_s = g["value"] if g["status"] == "ok" else float("nan")   # solve time from inside the child
        sink_status = g["status"]
        rows.append(dict(N=N, vars=N*N, ocs_s=ocs_s, sinkhorn_s=sink_s, sinkhorn_status=sink_status,
                         status=getattr(out, "status", "?"), residual=float(getattr(out, "residual", float("nan")))))
        print(f"N={N:6d}  vars={N*N:.3e}  OCS {ocs_s:8.3f}s  POT {sink_s:8.3f}s  {rows[-1]['status']}  r={rows[-1]['residual']:.2e}", flush=True)
        with open(os.path.join(PLOTS, "ot_wedge.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(PLOTS, "ot_wedge.csv"))


if __name__ == "__main__":
    main()
