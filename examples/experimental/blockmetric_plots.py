"""Plots and summary CSV for the block-metric pilot (matplotlib only; no jax).

    python blockmetric_plots.py <results dir>

Reads f1_results.csv, f1_a4_results.csv, f2_results.csv, f3_results.csv,
f4_results.csv (whichever exist) written by blockmetric_pilot.py and writes the
PNGs and summary.csv into the same directory.
"""
import csv, os, sys, json, math

csv.field_size_limit(10 ** 9)   # the per-block residual history is a long field
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical palette (validated with the dataviz palette checker, light surface):
# hue follows the metric family; line style / marker distinguishes variants inside a family
ARM_STYLE = {
    "A0":    dict(color="#2a78d6", ls="-",  marker="o", label="A0 shipped engine (Halpern, restarts)"),
    "A0std": dict(color="#2a78d6", ls="--", marker="o", label="A0std shipped engine, standard PDHG"),
    "A1":    dict(color="#4a3aa7", ls="-",  marker="s", label="A1 loop, shipped scalar/tied metric"),
    "A2":    dict(color="#eb6834", ls="-",  marker="^", label="A2 diagonal metric (eta = 0)"),
    "A3":    dict(color="#1baf7a", ls="-",  marker="D", label="A3 full blocks, exact Grams"),
    "A3s1":  dict(color="#1baf7a", ls=":",  marker="v", label="A3s full blocks, sketched q = k"),
    "A3s4":  dict(color="#1baf7a", ls="--", marker="<", label="A3s full blocks, sketched q = 4k"),
    "A3s16": dict(color="#1baf7a", ls="-.", marker=">", label="A3s full blocks, sketched q = 16k"),
    "A4p1":  dict(color="#eda100", ls=":",  marker="p", label="A4 Liu-Xu-Yin p = 1"),
    "A4p2":  dict(color="#eda100", ls="--", marker="h", label="A4 Liu-Xu-Yin p = 2"),
    "A4p3":  dict(color="#eda100", ls="-",  marker="*", label="A4 Liu-Xu-Yin p = 3"),
    "A5":    dict(color="#e87ba4", ls="-",  marker="X", label="A5 optimal diagonal (SDP reference)"),
}
ORDER = list(ARM_STYLE)
TEXT = "#333"; MUTED = "#777"; GRID = "#e5e5e0"


DROPPED = []


def load(path, keep_contended=False):
    """Rows of one results CSV.  Rows marked `contended` were measured while another
    pilot process held the same host; they are DROPPED, never averaged in."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not keep_contended:
        bad = [r for r in rows if str(r.get("contended", "")).strip() in ("1", "true", "True")]
        if bad:
            DROPPED.append((os.path.basename(path), len(bad)))
        rows = [r for r in rows if r not in bad]
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def arm_of(r):
    return r.get("tag") or r.get("arm")


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def f1_lines(rows, outdir, ycol, fname, ylabel, title, variant="equal"):
    rows = [r for r in rows if r.get("family") == "F1" and r.get("variant") == variant and float(r.get("k", 4)) == 4]
    ns = sorted({int(r["n"]) for r in rows})
    if not rows:
        return
    rhos = sorted({float(r["rho"]) for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(6.2 * len(ns), 4.6), squeeze=False)
    for ax, n in zip(axes[0], ns):
        style_axes(ax)
        for arm in ORDER:
            pts = sorted([(float(r["rho"]), i, r) for i, r in enumerate(rows)
                          if int(r["n"]) == n and arm_of(r) == arm])   # index breaks ties, dicts are unorderable
            if not pts:
                continue
            st = ARM_STYLE[arm]
            xs = [rhos.index(p[0]) for p in pts]; ys = [float(p[2][ycol]) for p in pts]
            ok = [float(p[2].get("certified", 0)) == 1 for p in pts]
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=2, alpha=0.9)
            ax.scatter([x for x, o in zip(xs, ok) if o], [y for y, o in zip(ys, ok) if o], color=st["color"],
                       marker=st["marker"], s=42, zorder=3, edgecolor="white", linewidth=0.8)
            ax.scatter([x for x, o in zip(xs, ok) if not o], [y for y, o in zip(ys, ok) if not o], facecolor="white",
                       edgecolor=st["color"], marker=st["marker"], s=42, zorder=3, linewidth=1.6)
        ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_yscale("log"); ax.set_xlabel("within-block correlation rho", color=TEXT)
        if ax is axes[0][0]:
            ax.set_ylabel(ylabel, color=TEXT)
        ax.set_title(f"n = {n:,}", color=TEXT, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], color=ARM_STYLE[a]["color"], ls=ARM_STYLE[a]["ls"], marker=ARM_STYLE[a]["marker"],
                          lw=2, label=ARM_STYLE[a]["label"]) for a in ORDER if any(arm_of(r) == a for r in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "  (hollow marker: 1e-5 certificate NOT reached)", color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(os.path.join(outdir, fname), dpi=150); plt.close(fig)


def time_split_bars(rows, outdir, fname, title, group_key, group_label):
    """Stacked bars per arm within groups: shipped equilibration (or, for the shipped
    engine, its whole compile), metric setup (Grams, eigendecompositions, norm
    estimate), iterations excluding prox, prox, verification.  Warm timings."""
    groups = sorted({group_key(r) for r in rows}, key=lambda g: str(g))
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.8), squeeze=False)
    parts = [("equilibration / compile", "#4a3aa7"), ("metric setup", "#e87ba4"), ("iterations (excl. prox)", "#2a78d6"),
             ("prox", "#eb6834"), ("verification", "#1baf7a")]
    for ax, g in zip(axes[0], groups):
        style_axes(ax)
        arms = [a for a in ORDER if any(arm_of(r) == a and group_key(r) == g for r in rows)]
        for i, a in enumerate(arms):
            r = [r for r in rows if arm_of(r) == a and group_key(r) == g][0]
            setup = float(r.get("setup_s") or 0); it = float(r.get("iter_s") or 0)
            equil = r.get("equil_s"); equil = setup if equil in ("", None) else float(equil)
            prox = r.get("prox_s"); prox = 0.0 if prox in ("", None) or (isinstance(prox, float) and math.isnan(prox)) else float(prox)
            ver = float(r.get("verify_s") or 0)
            vals = [equil, max(setup - equil, 0.0), max(it - prox, 0.0), prox, ver]
            bottom = 0.0
            for (name, col), v in zip(parts, vals):
                ax.bar(i, v, bottom=bottom, color=col, width=0.72, edgecolor="white", linewidth=1.0)
                bottom += v
            cert = float(r.get("certified", 0)) == 1
            ax.text(i, bottom * 1.02, f"{int(float(r['iters']))} it" + ("" if cert else "\nNOT certified"),
                    ha="center", va="bottom", fontsize=7, color=TEXT if cert else "#e34948")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("seconds (warm)", color=TEXT); ax.set_title(group_label(g), color=TEXT, fontsize=10, loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    fig.legend(handles, [p[0] for p in parts], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title, color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(os.path.join(outdir, fname), dpi=150); plt.close(fig)


def f4_plot(rows, outdir):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); style_axes(ax)
    arms = [a for a in ORDER if any(arm_of(r) == a for r in rows)]
    for a in arms:
        rr = sorted([r for r in rows if arm_of(r) == a], key=lambda r: float(r["instance"]))
        Ns = [1, 10, 100]
        xs, ys = [], []
        for N in Ns:
            sub = [r for r in rr if float(r["instance"]) < N]
            if len(sub) == N:
                xs.append(N); ys.append(float(sub[-1]["cum_s"]) / N)
        st = ARM_STYLE[a]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"], lw=2, label=st["label"])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("instances N (same K, different b)", color=TEXT); ax.set_ylabel("mean seconds per instance, setup amortized", color=TEXT)
    ax.set_title("F4: repeated solves, F1 n = 200,000, rho = 0.99, tol 1e-5", color=TEXT, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "f4_mean_time_vs_N.png"), dpi=150); plt.close(fig)


def summary(all_rows, outdir):
    cols = ["family", "name", "variant", "note", "n", "k", "rho", "mu", "kappa_max", "tag", "eta", "q", "p", "tau_mult", "host", "iters", "status",
            "certified", "residual", "r_primal", "r_dual", "setup_s", "iter_s", "prox_s", "ops_s", "verify_s",
            "total_warm_s", "cold_solve_s", "equil_s", "gram_s", "eig_s", "norm_s",
            "norm_est", "norm_exact", "norm_upper", "factor", "peak_bytes", "contended",
            "block0_kappa_B0_kappa_global", "block0_kappa_new_kappa_global", "block0_kappa_new_kappa_block_median",
            "block0_gram_sketch_relerr", "block1_kappa_B0_kappa_global", "block1_kappa_new_kappa_global", "timed_run"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    outdir = sys.argv[1]
    f1 = load(os.path.join(outdir, "f1_results.csv")) + load(os.path.join(outdir, "f1_a4_results.csv")) + \
        load(os.path.join(outdir, "f1_a5_results.csv"))
    f2 = load(os.path.join(outdir, "f2_results.csv")); f3 = load(os.path.join(outdir, "f3_results.csv"))
    f4 = load(os.path.join(outdir, "f4_results.csv"))
    f1mu = load(os.path.join(outdir, "f1_mu0.1_results.csv"))
    f1k = load(os.path.join(outdir, "f1_kappa_results.csv"))
    if f1mu:
        f1_lines(f1mu, outdir, "total_warm_s", "f1_mu0.1_time_vs_rho.png", "seconds to the 1e-5 certificate (setup + loop + verification)",
                 "F1 with mu = 0.1 (away from the float32 floor): time to certificate")
        f1_lines(f1mu, outdir, "iters", "f1_mu0.1_iters_vs_rho.png", "PDHG iterations to the 1e-5 certificate",
                 "F1 with mu = 0.1: iterations")
        time_split_bars([r for r in f1mu if r.get("variant") == "imbalance"], outdir, "f1_mu0.1_imbalance_split.png",
                        "F1 magnitude-imbalance variant at mu = 0.1", lambda r: (r["variant"], float(r["rho"])),
                        lambda g: f"{g[0]}, rho = {g[1]:g}")
    if f1k:
        time_split_bars(f1k, outdir, "f1_kappa_max_split.png", "F1 full blocks: kappa_max clip 100 vs 1e4 (n = 1e6)",
                        lambda r: (float(r["rho"]), float(r.get("kappa_max", 100))), lambda g: f"rho = {g[0]:g}, kappa_max = {g[1]:g}")
    ov = load(os.path.join(outdir, "f1_overlap_results.csv"))
    if ov:
        f1_lines(ov, outdir, "total_warm_s", "f1_overlap_time_vs_rho.png",
                 "seconds to the 1e-5 certificate (setup + loop + verification)",
                 "F1 COUPLED (overlap, two stacked partitions): total time to certificate", variant="overlap")
        f1_lines(ov, outdir, "iter_s", "f1_overlap_looptime_vs_rho.png", "loop seconds only (iterations, setup excluded)",
                 "F1 COUPLED (overlap): loop time only, so setup-path differences cannot be read as method wins",
                 variant="overlap")
        f1_lines(ov, outdir, "iters", "f1_overlap_iters_vs_rho.png", "PDHG iterations to the 1e-5 certificate",
                 "F1 COUPLED (overlap): iterations", variant="overlap")
        for rr in (0.99, 0.999):
            sel = [r for r in ov if abs(float(r["rho"]) - rr) < 1e-9]
            if sel:
                time_split_bars(sel, outdir, f"f1_overlap_time_split_rho{rr:g}.png",
                                f"F1 COUPLED (overlap) rho = {rr:g}: wall time split",
                                lambda r: int(float(r["n"])), lambda n: f"n = {n:,}")
    if f1:
        f1_lines(f1, outdir, "total_warm_s", "f1_time_vs_rho.png", "seconds to the 1e-5 certificate (setup + loop + verification)",
                 "F1 SEPARABLE SANITY CHECK (one partition): time to certificate")
        f1_lines(f1, outdir, "iter_s", "f1_looptime_vs_rho.png", "loop seconds only (iterations, setup excluded)",
                 "F1 SEPARABLE SANITY CHECK: loop time only")
        f1_lines(f1, outdir, "iters", "f1_iters_vs_rho.png", "PDHG iterations to the 1e-5 certificate",
                 "F1 SEPARABLE SANITY CHECK (one partition): iterations")
        sel = [r for r in f1 if r.get("variant") == "equal" and float(r["rho"]) == 0.99 and float(r.get("k", 4)) == 4]
        time_split_bars(sel, outdir, "f1_time_split_rho0.99.png", "F1 rho = 0.99: wall time split (A0 has no prox split; its check is in-loop)",
                        lambda r: int(r["n"]), lambda n: f"n = {n:,}")
        extra = [r for r in f1 if r.get("variant") in ("imbalance",) or float(r.get("k", 4)) == 8]
        if extra:
            time_split_bars(extra, outdir, "f1_variants_time_split.png", "F1 variants: k = 8 and magnitude imbalance",
                            lambda r: (r["variant"], int(float(r["k"])), float(r["rho"])),
                            lambda g: f"{g[0]}, k = {g[1]}, rho = {g[2]:g}")
    if f2:
        time_split_bars(f2, outdir, "f2_bars.png", "F2 multichannel deblurring (3-channel blocks): wall time split",
                        lambda r: (r["variant"], int(round(math.sqrt(float(r["n"]) / 3)))), lambda g: f"mixing {g[0]}, {g[1]}x{g[1]}")
    if f3:
        time_split_bars(f3, outdir, "f3_bars.png", "F3 no-regression control: 3-D TV imaging at 64^3", lambda r: r["name"], lambda g: g)
    f4_plot(f4, outdir)
    summary(f1 + ov + f1mu + f1k + f2 + f3, outdir)
    if DROPPED:
        print("dropped contended rows:", DROPPED)
    print("wrote plots and summary.csv to", outdir)


if __name__ == "__main__":
    main()
