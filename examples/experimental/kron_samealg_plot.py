"""Render the same-algorithm Kronecker figure (companion to make_plots.F).

Reads the CSV written by kron_samealg.py (OCS with factors vs OCS with the
stored dense matrix, same algorithm and instance) and the numpy lstsq
reference column of plots/kron_wedge.csv, and draws the two panels of
Figure F: time and memory.  matplotlib only; no solver is imported.

    python kron_samealg_plot.py --csv kron_samealg.csv --wedge plots/kron_wedge.csv \
        --meta meta.json --out F_kronecker_wedge_samealg.png
"""
import os, csv, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, RED, PURP, ORANGE = "#1f77b4", "#76b900", "#d62728", "#9467bd", "#ff7f0e"


def fl(v):
    try: return float(v)
    except (TypeError, ValueError): return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--wedge", required=True)
    ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    devname, gpu_gb = meta["device"], meta["device_memory_gb"]
    rows = list(csv.DictReader(open(a.csv)))
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    fac = sorted(arms.get("factors", []), key=lambda r: int(r["T"]))
    den = sorted(arms.get("dense", []), key=lambda r: int(r["T"]))
    ok = lambda r: str(r["status"]).startswith(("solved", "inaccurate"))
    Vf = np.array([fl(r["vars"]) for r in fac]); tf = np.array([fl(r["warm_s"]) for r in fac])
    pf = np.array([fl(r["peak_gb"]) for r in fac])
    dok = [r for r in den if ok(r)]
    Vd = np.array([fl(r["vars"]) for r in dok]); td = np.array([fl(r["warm_s"]) for r in dok])
    pd_ = np.array([fl(r["peak_gb"]) for r in dok]); md = np.array([fl(r["matrix_gb"]) for r in dok])
    dfail = [r for r in den if not ok(r)]
    wedge = list(csv.DictReader(open(a.wedge)))
    Vw = np.array([fl(r["vars"]) for r in wedge if r["dense_status"] == "formed"])
    tw = np.array([fl(r["dense_solve_s"]) for r in wedge if r["dense_status"] == "formed"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax.loglog(Vf, tf, "o-", lw=2.8, ms=7, color=BLUE,
              label="OCS, Kronecker factors $A$, $B$ (same algorithm, warm, compile excluded)")
    if len(Vd):
        ax.loglog(Vd, td, "s-", lw=2.4, ms=7, color=ORANGE,
                  label="OCS, stored dense $A\\otimes B$ as ocs.Dense (same algorithm, warm, compile excluded)")
    if len(Vw):
        ax.loglog(Vw, tw, "^--", lw=1.8, ms=6, color=RED, alpha=.8,
                  label="reference: numpy float32 kron + one dense lstsq on the host (different computation, kron_wedge.csv)")
    if dfail:
        r = dfail[0]
        why = meta.get("dense_fail_note", r["status"])
        gb = fl(r["matrix_gb"])
        if np.isnan(gb):
            gb = (64 * int(r["T"])) ** 2 * 4 / 1e9
        ax.annotate(f"stored matrix stops at T={r['T']} ({gb:.1f} GB matrix):\n{why}",
                    (Vd[-1], td[-1]) if len(Vd) else (Vf[-1], tf[-1]),
                    xytext=(Vf[0] * 1.1, tf[-1] * 150), color=ORANGE, fontsize=8.0,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.set_xlabel("variables  $ST$  ($S=64$)"); ax.set_ylabel("solve time (s)")
    ax.set_title("Solve time, measured (median of 3 warm solves)")
    ax.grid(True, which="both", ls=":", alpha=.4); ax.legend(loc="upper left", fontsize=7.6)
    ax.set_ylim(top=max(ax.get_ylim()[1], 2e4))

    ax2.loglog(Vf, pf, "o-", lw=2.8, ms=7, color=BLUE, label="OCS with factors: peak device memory (measured)")
    if len(Vd):
        ax2.loglog(Vd, pd_, "s-", lw=2.4, ms=7, color=ORANGE, label="OCS with stored matrix: peak device memory (measured)")
        ax2.loglog(Vd, md, "s:", lw=1.6, ms=5, color=RED, label="stored $A\\otimes B$ alone ($S^2T^2$ float32)")
    ax2.axhline(gpu_gb, color=GREEN, ls=":", lw=1.8)
    ax2.text(Vf[-1], gpu_gb * 0.55, f"{gpu_gb:.1f} GB ({devname}, nominal; see README for this host's per-process cap)",
             color="#3f6000", fontsize=8, ha="right")
    ax2.set_ylim(top=max(gpu_gb, np.nanmax(pd_) if len(Vd) else 1) * 6)
    ax2.set_xlabel("variables  $ST$  ($S=64$)"); ax2.set_ylabel("memory (GB)")
    ax2.set_title("Peak device memory, measured for both OCS arms")
    ax2.grid(True, which="both", ls=":", alpha=.4); ax2.legend(loc="lower right", fontsize=7.6)
    from matplotlib.ticker import NullFormatter
    for axx in (ax, ax2):
        axx.xaxis.set_minor_formatter(NullFormatter()); axx.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.text(0.5, 0.965, "Kronecker-structured lasso, $\\min\\ \\frac{1}{2}\\|(A\\otimes B)\\mathrm{vec}(X)-b\\|_2^2+\\lambda\\|\\mathrm{vec}(X)\\|_1$, "
             "$X$ of size 64×T, same OCS algorithm on both operator forms", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.917, "Regime 1, scale: the stored-matrix form and the factor form run the identical solver, "
             "residual tolerance 1e-6, same instance and compile settings", ha="center", va="center", fontsize=8.8, color="#444444")
    fig.text(0.5, 0.012, f"Both OCS arms: one process per size, cold JIT and equilibration excluded from the warm time; "
             f"achieved residuals and iteration counts in the CSV.  {devname}, float32.\n"
             f"The red reference is the host numpy lstsq from Figure F, a different computation kept only for continuity.",
             ha="center", fontsize=7.5, color="dimgray")
    fig.savefig(a.out, dpi=150); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
