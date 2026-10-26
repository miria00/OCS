"""Figures and the summary table for the gate campaign (September 11, 2026).

Reads the per-run CSVs and *_summary.json files written by gated_weight_runner.py
(synced from the two RTX 5090 machines into profiling/gate_campaign_20260911/runs/<host>/)
and the competitors' recorded numbers per problem class, and writes one figure per
class plus a summary table.  No solver is run here: matplotlib and csv only.

    /home/miria/jaxenv2/bin/python examples/experimental/gate_campaign_plots.py

Outputs: profiling/gate_campaign_20260911/gate_campaign_<class>.png (+ copies in
OCS_paper/figs/), summary.csv, summary_table.tex, and readings.txt (the numbers the
rehearsal paragraphs quote).
"""
import os, sys, csv, json, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(ROOT, "profiling", "gate_campaign_20260911")
COMP = os.path.join(CAMP, "competitors")
PLOTS = os.path.join(ROOT, "plots")
FIGS = os.path.join(os.path.dirname(ROOT), "OCS_paper", "figs")

# fixed categorical order, never cycled: OCS fixed, OCS gated, then the competitors
BLUE, ORANGE, PURP, GREEN, RED, GREY, BROWN = "#1f77b4", "#ff7f0e", "#9467bd", "#76b900", "#d62728", "#555555", "#8c564b"
OFF = dict(color=BLUE, marker="o", ls="-", lw=2.0, ms=6, label="OCS, fixed weight (the paper's)")
ON = dict(color=ORANGE, marker="s", ls="-", lw=2.0, ms=6, label="OCS, restart-gated weight, budget 8")
COMPSTY = [dict(color=PURP, marker="^", ls="--", lw=1.6, ms=6),
           dict(color=GREEN, marker="v", ls="--", lw=1.6, ms=6),
           dict(color=GREY, marker="D", ls="-.", lw=1.4, ms=5),
           dict(color=RED, marker="x", ls=":", lw=1.4, ms=7),
           dict(color=BROWN, marker="P", ls=":", lw=1.4, ms=6)]
HOST_GPU = {"mazeppa": "Mazeppa RTX 5090", "kevin": "KEVIN RTX 5090"}


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- campaign runs
def load_runs():
    """One record per run: meta + summary + epoch rows, keyed by (host, stem, gate)."""
    runs = {}
    for host in sorted(os.listdir(os.path.join(CAMP, "runs"))) if os.path.isdir(os.path.join(CAMP, "runs")) else []:
        for js in sorted(glob.glob(os.path.join(CAMP, "runs", host, "*_summary.json"))):
            base = os.path.basename(js)[:-len("_summary.json")]
            if "_off_on" in base:                 # the misnamed first minutes of queue A, superseded
                continue
            if base.endswith("_off"):
                stem, gate = base[:-4], "off"
            elif base.endswith("_on"):
                stem, gate = base[:-3], "on"
            else:
                continue
            with open(js) as fh:
                d = json.load(fh)
            rows = read_csv(os.path.join(CAMP, "runs", host, base + ".csv"))
            runs[(host, stem, gate)] = dict(meta=d["meta"], summary=d["summary"], rows=rows, host=host, stem=stem, gate=gate)
    return runs


def pairs(runs, problem, stem_filter=lambda s: True):
    """(host, stem) -> {'off': run, 'on': run} for one problem class."""
    out = {}
    for (host, stem, gate), r in runs.items():
        if r["meta"]["problem"] == problem and stem_filter(stem):
            out.setdefault((host, stem), {})[gate] = r
    return out


def solve_wall(r):
    """Solve-only wall: the sum of the timed epochs (the first solve at each visited
    weight carries that weight's XLA compile; the count of those is reported)."""
    return fl(r["summary"].get("epochs_s", r["summary"].get("solve_s")))


def _rate(r):
    """Median solve-only seconds per iteration over the run's non-compile epochs, or None."""
    prev, rates = 0, []
    for row in r["rows"]:
        it = fl(row["iters"]) - prev; prev = fl(row["iters"])
        if not is_compile_epoch(row) and it > 0:
            rates.append(fl(row["epoch_s"]) / it)
    return float(np.median(rates)) if rates else None


def adj_wall(r, partner=None):
    """Solve-only wall with each compile epoch (the first solve at a weight, and epoch 2
    for the warm-start executable) replaced by the run's own median per-iteration rate
    times the epoch's iterations.  A run with no clean epoch borrows the rate of its
    pair partner (same problem, same machine); with neither, the raw clock stands.
    Reported next to the raw clock, never instead of it."""
    return adj_clock(r, partner)[-1] if r["rows"] else solve_wall(r)


def adj_clock(r, partner=None):
    """Cumulative adjusted solve clock per epoch (the rule of adj_wall)."""
    rows = r["rows"]; prev, items = 0, []
    for row in rows:
        it = fl(row["iters"]) - prev; prev = fl(row["iters"])
        items.append((fl(row["epoch_s"]), it, is_compile_epoch(row)))
    rate = _rate(r)
    if rate is None and partner is not None:
        rate = _rate(partner)
    out, acc = [], 0.0
    for e, it, first in items:
        acc += (min(e, rate * it) if (first and rate is not None) else e); out.append(acc)
    return out


def first_calls(r):
    return sum(1 for row in r["rows"] if is_compile_epoch(row))


def weight_path(r):
    """The weights in the order they were used.  The runner writes each epoch row after
    the gate decision, so a row's w is the weight of the NEXT epoch: the weights used
    are w0 followed by every row's w but the last."""
    used = [fl(r["meta"]["w0"])] + [fl(row.get("w")) for row in r["rows"][:-1]]
    path = []
    for w in used:
        if not path or path[-1] != w:
            path.append(w)
    return "->".join(f"{w:g}" for w in path)


def is_compile_epoch(row):
    """The first solve at a weight compiles its executable; the warm-start executable at
    the starting weight compiles on epoch 2."""
    return str(row.get("first_call", "")).lower() == "true" or int(fl(row["epoch"])) == 2


def size_of(r):
    m, p = r["meta"], r["meta"]["problem"]
    if p == "turf":
        return m["tasks"] * m["servers"] * (5 if m["mode"] == "shape" else (1 if m["mode"] == "global" else m["servers"]))
    if p == "imaging":
        return m["size"] ** 3
    if p == "ot":
        return m["size"] ** 2
    if p == "mri":
        return m["size"] ** 2
    if p == "kron":
        return 64 * m["size"]
    if p == "fft":
        return m["size"]
    return float("nan")


# ----------------------------------------------------------------------------- generic panel
def ladder_axes(pr, xkey, title, xlabel, comp_iters=(), comp_time=(), note=""):
    """Two panels: iterations to the tolerance and solve-only wall time, both log-log,
    against size.  comp_* are lists of (label, xs, ys, style)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for gate, sty in (("off", OFF), ("on", ON)):
        pts = sorted((xkey(v[gate]), fl(v[gate]["summary"]["iterations"]), adj_wall(v[gate], v.get("off" if gate == "on" else "on")),
                      v[gate]["summary"]["status"]) for v in pr.values() if gate in v)
        if not pts:
            continue
        xs = [p[0] for p in pts]; it = [p[1] for p in pts]; tt = [p[2] for p in pts]
        a1.loglog(xs, it, **sty); a2.loglog(xs, tt, **sty)
        for x, i, t, st in pts:                  # a hollow marker where the run stopped on its budget or cap
            if st != "solved":
                a1.plot([x], [i], marker=sty["marker"], ms=11, mfc="white", mec=sty["color"], mew=1.8, ls="none")
                a2.plot([x], [t], marker=sty["marker"], ms=11, mfc="white", mec=sty["color"], mew=1.8, ls="none")
    for lab, xs, ys, sty in comp_iters:
        if len(xs):
            a1.loglog(xs, ys, label=lab, **sty)
    for lab, xs, ys, sty in comp_time:
        if len(xs):
            a2.loglog(xs, ys, label=lab, **sty)
    a1.set_xlabel(xlabel); a1.set_ylabel("iterations to the tolerance"); a1.set_title("Iterations")
    a2.set_xlabel(xlabel); a2.set_ylabel("wall time (s)"); a2.set_title("Wall time (OCS: solve only, first-call compiles excluded)")
    for a in (a1, a2):
        a.grid(True, which="major", ls=":", alpha=0.4); a.legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=12)
    if note:
        footer(fig, note)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def footer(fig, note, width=190):
    import textwrap
    lines = textwrap.wrap(note, width)[:3]
    for i, ln in enumerate(lines):
        fig.text(0.5, 0.036 - 0.016 * i, ln, ha="center", fontsize=7.4, color="dimgray")


def save(fig, cls):
    os.makedirs(CAMP, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(CAMP, f"gate_campaign_{cls}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    import shutil; shutil.copy(out, os.path.join(FIGS, f"gate_campaign_{cls}.png"))
    print("wrote", out)


def hosts_of(pr):
    return ", ".join(sorted({HOST_GPU.get(h, h) for h, _ in pr}))


# ----------------------------------------------------------------------------- per class
def fig_ot(runs):
    pr = pairs(runs, "ot")
    if not pr:
        return
    wedge = read_csv(os.path.join(COMP, "mazeppa", "ot_wedge.csv"))
    pot = [(fl(r["vars"]), fl(r["sinkhorn_s"])) for r in wedge if r.get("sinkhorn_status", "ok") == "ok"]
    ocsH = [(fl(r["vars"]), fl(r["ocs_s"])) for r in wedge]
    fig = ladder_axes(pr, size_of, "Optimal transport (figure H family): tolerance 2e-6, weight 30 fixed against gated",
                      "transport plan size $N^2$ (variables)",
                      comp_time=[("POT quadratic semi-dual, CPU (Mazeppa host, figure H run)", [p[0] for p in pot], [p[1] for p in pot], COMPSTY[0]),
                                 ("OCS figure H run itself, warm call, Mazeppa 5090", [p[0] for p in ocsH], [p[1] for p in ocsH], COMPSTY[2])],
                      note=f"OCS runs on {hosts_of(pr)}; hollow markers stopped on the 200,000-iteration budget of the family. "
                           "POT time is the semi-dual solve of examples/_quadratic_ot.pot_reference on Mazeppa's CPU, as recorded in ot_wedge.csv.")
    save(fig, "ot")


def fig_fft(runs):
    pr = pairs(runs, "fft")
    if not pr:
        return
    scs = [(fl(r["N"]), fl(r["scs_solve_s"])) for r in read_csv(os.path.join(PLOTS, "competitors_fft_time.csv")) if r["status"] == "solved"]
    cu = [(fl(r["N"]), fl(r["cuopt_barrier_solve_s"])) for r in read_csv(os.path.join(PLOTS, "competitors_fft_cupdlp.csv")) if r["status"] == "solved"]
    fig = ladder_axes(pr, size_of, "Subsampled-FFT LASSO (figure C family): tolerance 1e-5, fixed against gated weight",
                      "signal length $N$ (variables)",
                      comp_time=[("CVXPY/SCS, dense operator, CPU (plots/competitors_fft_time.csv)", [p[0] for p in scs], [p[1] for p in scs], COMPSTY[0]),
                                 ("cuOpt barrier, dense QP, GPU as recorded in plots/README (RTX 4090)", [p[0] for p in cu], [p[1] for p in cu], COMPSTY[1])],
                      note=f"OCS runs on {hosts_of(pr)}. The paper's figure C is a fixed 500-iteration budget below the 1e-3 bar; here both arms run to 1e-5. "
                           "Competitor CSVs carry no device column; SCS is a CPU solver, cuOpt's GPU is the one plots/README names.")
    save(fig, "fft")


def fig_mri(runs):
    pr = pairs(runs, "mri")
    if not pr:
        return
    scs = [(fl(r["cones"]), fl(r["scs_solve_s"])) for r in read_csv(os.path.join(COMP, "mazeppa", "competitors_mri_time.csv")) if r["status"] == "solved"]
    fig = ladder_axes(pr, size_of, "Isotropic-TV MRI SOCP (figure D family): tolerance 1e-5, fixed against gated weight",
                      "pixels (one 2-D cone each)",
                      comp_time=[("CVXPY/SCS, dense operator, CPU (Mazeppa host, mri_competitor.py)", [p[0] for p in scs], [p[1] for p in scs], COMPSTY[0])],
                      note=f"OCS runs on {hosts_of(pr)}. The paper's figure D is a fixed 4,000-iteration budget below the 1e-3 bar; here both arms run to 1e-5.")
    save(fig, "mri")


def fig_kron(runs):
    pr = pairs(runs, "kron")
    if not pr:
        return
    kw = read_csv(os.path.join(COMP, "mazeppa", "kron_wedge.csv"))
    dense = [(fl(r["vars"]), fl(r["dense_solve_s"])) for r in kw if r["dense_status"] == "formed"]
    ocsF = [(fl(r["vars"]), fl(r["ocs_s"])) for r in kw]
    fig = ladder_axes(pr, size_of, "Kronecker-structured lasso (figure F family): tolerance 1e-6, fixed against gated weight",
                      "variables $ST$ ($S = 64$)",
                      comp_time=[("explicit $A \\otimes B$ formed and solved by lstsq, CPU (Mazeppa host)", [p[0] for p in dense], [p[1] for p in dense], COMPSTY[0]),
                                 ("OCS figure F run itself, warm call, Mazeppa 5090", [p[0] for p in ocsF], [p[1] for p in ocsF], COMPSTY[2])],
                      note=f"OCS runs on {hosts_of(pr)}. The explicit side is refused past T = 512 under the declared memory budget.")
    save(fig, "kron")


def fig_imaging(runs):
    pr = pairs(runs, "imaging")
    if not pr:
        return
    bw = read_csv(os.path.join(PLOTS, "billion_wedge_5090.csv"))
    ok = lambda r: str(r["status"]).lower().startswith(("solved", "optimal", "matched"))
    scs = sorted((fl(r["n"]), fl(r["solve_s"])) for r in bw if r["solver"] == "scs" and ok(r))
    cu = sorted((fl(r["n"]), fl(r["solve_s"])) for r in bw if r["solver"] == "cuopt" and ok(r))
    ocs_it = {int(fl(r["n"])): fl(r["iterations"]) for r in bw if r["solver"] == "ocs" and ok(r)}
    odl = sorted((fl(r["n"]), fl(r["solve_s"])) for r in bw if r["solver"] == "odl" and ok(r)
                 and int(fl(r["n"])) in ocs_it and abs(fl(r["iterations"]) - ocs_it[int(fl(r["n"]))]) < 0.5)
    ocsK = sorted((fl(r["n"]), fl(r["solve_s"]), fl(r["iterations"])) for r in bw if r["solver"] == "ocs" and ok(r))
    fig = ladder_axes(pr, size_of, "3-D imaging (figure K family): tolerance 1e-5, weight 1 fixed against gated",
                      "voxels (one 3-D cone each)",
                      comp_iters=[("OCS figure K run itself (Mazeppa 5090)", [p[0] for p in ocsK], [p[2] for p in ocsK], COMPSTY[2])],
                      comp_time=[("CVXPY/SCS, dense DFT rows, CPU (Mazeppa host)", [p[0] for p in scs], [p[1] for p in scs], COMPSTY[0]),
                                 ("cuOpt PDLP QP, anisotropic TV, RTX 5090", [p[0] for p in cu], [p[1] for p in cu], COMPSTY[1]),
                                 ("ODL PDHG at the OCS iteration count, CPU (Mazeppa host)", [p[0] for p in odl], [p[1] for p in odl], COMPSTY[3]),
                                 ("OCS figure K run itself (Mazeppa 5090)", [p[0] for p in ocsK], [p[1] for p in ocsK], COMPSTY[2])],
                      note=f"OCS runs on {hosts_of(pr)}. Competitors from plots/billion_wedge_5090.csv (same machine class, device column honored).")
    save(fig, "imaging")


def fig_turf(runs):
    pr = pairs(runs, "turf")
    if not pr:
        return
    turf = read_csv(os.path.join(COMP, "mazeppa", "cvxcluster_turf.csv"))
    OK = ("solved", "Optimal", "Optimization terminated successfully")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for ax, mode, xlabel in ((axes[0], "shape", "tasks $t$ (five shapes, $5t$ variables)"),
                             (axes[1], "server", "variables $s \\times t$ (individual servers)")):
        xk = (lambda r: r["meta"]["tasks"]) if mode == "shape" else size_of
        for tol, lsty in ((1e-3, "-"), (1e-5, ":")):
            for gate, sty in (("off", OFF), ("on", ON)):
                pts = sorted((xk(v[gate]), adj_wall(v[gate], v.get("off" if gate == "on" else "on")), v[gate]["summary"]["status"]) for v in pr.values()
                             if gate in v and v[gate]["meta"]["mode"] == mode and abs(v[gate]["meta"]["tol"] - tol) < 1e-12)
                if pts:
                    s = dict(sty); s["ls"] = lsty; s["label"] = sty["label"] + f", {tol:g}"
                    ax.loglog([p[0] for p in pts], [p[1] for p in pts], **s)
                    for x, t, st in pts:
                        if st != "solved":
                            ax.plot([x], [t], marker=sty["marker"], ms=11, mfc="white", mec=sty["color"], mew=1.8, ls="none")
        for k, (solver, lab) in enumerate((("HIGHS", "HiGHS IPM, CPU (Mazeppa host)"), ("CUOPT", "cuOpt PDLP, Mazeppa 5090"),
                                           ("SCS", "SCS, CPU (Mazeppa host)"))):
            last = {}
            for r in turf:
                if r["mode"] == mode and r["solver"] == solver and any(r["status"].startswith(o) for o in OK) and r["solve_s"]:
                    x = int(r["tasks"]) if mode == "shape" else int(r["variables"])
                    last[x] = fl(r["solve_s"])
            if last:
                xs = sorted(last); ax.loglog(xs, [last[x] for x in xs], label=lab, **COMPSTY[k])
        ax.set_xlabel(xlabel); ax.set_ylabel("wall time (s)"); ax.grid(True, which="major", ls=":", alpha=0.4)
        ax.set_title("Aggregated program (their sizes)" if mode == "shape" else "Per-server program")
        ax.legend(fontsize=7.2, loc="upper left")
    fig.suptitle("CvxCluster Stage-1 LP (figure J family): weight 10 fixed against gated, tolerances 1e-3 (solid) and 1e-5 (dotted)", fontsize=11.5)
    footer(fig, f"OCS runs on {hosts_of(pr)}, wall time is the solve-only clock with first-call compiles excluded. Competitor times from the figure J CSV measured on Mazeppa "
                "(HiGHS and SCS on its CPU, cuOpt on its 5090). Gurobi's Table 1 times are the authors' and are not drawn.")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save(fig, "turf")


def fig_sylvester(runs):
    pr = pairs(runs, "sylvester")
    if not pr:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    (host, stem), v = sorted(pr.items())[0]
    for gate, sty in (("off", OFF), ("on", ON)):
        if gate not in v:
            continue
        rows = v[gate]["rows"]
        a1.semilogy([fl(r["iters"]) for r in rows], [fl(r["residual"]) for r in rows], color=sty["color"], marker=sty["marker"], ms=3, lw=1.6, label=sty["label"])
        a2.semilogy([fl(r["epochs_s"]) for r in rows], [fl(r["residual"]) for r in rows], color=sty["color"], marker=sty["marker"], ms=3, lw=1.6, label=sty["label"])
    for a, xl in ((a1, "iterations"), (a2, "wall time (s), solve only")):
        a.axhline(1e-8, color="k", lw=1, ls="--", alpha=0.6); a.set_xlabel(xl); a.set_ylabel("certificate"); a.grid(True, which="major", ls=":", alpha=0.4); a.legend(fontsize=8)
    ref = os.path.join(CAMP, "runs", host, "killer_sylvester_reference.log")
    cg = ""
    if os.path.exists(ref):
        for ln in open(ref):
            if "CGLS" in ln and "matvecs" in ln:
                cg = ln.strip()
    fig.suptitle(f"Sylvester least squares, 60x50 (the paper's instance): PDHG at tolerance 1e-8, fixed against gated weight", fontsize=11.5)
    for a in (a1, a2):
        a.set_ylim(3e-9, 3e-6); a.text(0.02, 0.9, "both arms sit on a float32 certificate floor near 7e-7", transform=a.transAxes, fontsize=8)
    footer(fig, f"OCS runs on {HOST_GPU.get(host, host)}. The paper's alternative for this pure least-squares problem is matrix-free CGLS, measured in the same queue: "
                + (cg.strip() if cg else "131 operator products in the paper's text."))
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "sylvester")


def fig_rof(runs):
    pr = pairs(runs, "rof")
    if not pr:
        return
    (host, stem), v = sorted(pr.items())[0]
    rof = read_csv(os.path.join(COMP, "mazeppa", "rof_spdhg_vs_ocs.csv"))
    series = {}
    for r in rof:
        series.setdefault(r["method"], []).append((fl(r["wall_s"]), fl(r["objective"])))
    Js = [J for s in series.values() for _, J in s]
    for gate in ("off", "on"):
        if gate in v:
            Js += [fl(r["objective_J"]) for r in v[gate]["rows"] if r.get("objective_J")]
    best = min(Js)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sty = {"SPDHG-CPU": (COMPSTY[0], "SPDHG, authors' reference implementation, CPU (Mazeppa host)"),
           "SPDHG-GPU": (COMPSTY[1], "SPDHG, ODL pytorch backend, Mazeppa 5090"),
           "SplitBregman": (COMPSTY[3], "Split Bregman, scikit-image, CPU (Mazeppa host)"),
           "OCS": (COMPSTY[2], "OCS figure I budgets, Mazeppa 5090")}
    for name, pts in series.items():
        if name not in sty:
            continue
        t = np.array([p[0] for p in pts]); J = np.array([p[1] for p in pts])
        s = dict(sty[name][0]); s["ms"] = 3 if name.startswith("SPDHG") else s["ms"]
        ax.loglog(t, np.maximum((J - best) / abs(best), 1e-9), label=sty[name][1], **s)
    for gate, st in (("off", OFF), ("on", ON)):
        if gate in v:
            rows = [r for r in v[gate]["rows"] if r.get("objective_J")]
            t = np.array(adj_clock(v[gate], v.get("off" if gate == "on" else "on"))); J = np.array([fl(r["objective_J"]) for r in rows])
            ax.loglog(t, np.maximum((J - best) / abs(best), 1e-9), **st)
    ax.axhline(2e-4, color="k", lw=1.0, ls="--", alpha=0.6); ax.text(ax.get_xlim()[0] * 1.3, 2.6e-4, "common target 2e-4", fontsize=8)
    ax.set_xlabel("wall time (s), solve only"); ax.set_ylabel("relative objective gap to the best J seen")
    ax.set_title("ROF denoising (figure I): objective gap against time, fixed against gated weight, tolerance 1e-5")
    ax.grid(True, which="major", ls=":", alpha=0.4); ax.legend(fontsize=7.6, loc="lower left")
    footer(fig, f"OCS gate runs on {HOST_GPU.get(host, host)}, one point per 1,000-iteration epoch, solve-only clock with first-call compiles excluded. "
                "Competitor traces from the figure I CSV measured on Mazeppa.", width=150)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "rof")


def fig_featsel(runs):
    pr = pairs(runs, "featsel")
    if not pr:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for (host, stem), v in sorted(pr.items()):
        for gate, sty in (("off", OFF), ("on", ON)):
            if gate not in v:
                continue
            rows = v[gate]["rows"]
            lab = f"{sty['label']}: {stem.replace('featsel_', '')}"
            a1.semilogy([fl(r["iters"]) for r in rows], [fl(r["residual"]) for r in rows], color=sty["color"], marker=sty["marker"], ms=3, lw=1.4,
                        ls="-" if "10000" in stem else ("--" if "b100" in stem else ":"), label=lab)
            a2.semilogy([fl(r["epochs_s"]) for r in rows], [fl(r["residual"]) for r in rows], color=sty["color"], marker=sty["marker"], ms=3, lw=1.4,
                        ls="-" if "10000" in stem else ("--" if "b100" in stem else ":"), label=lab)
    for a, xl in ((a1, "iterations"), (a2, "wall time (s), solve only")):
        a.axhline(1e-3, color="k", lw=1, ls="--", alpha=0.6); a.set_xlabel(xl); a.set_ylabel("certificate"); a.grid(True, which="major", ls=":", alpha=0.4); a.legend(fontsize=6.8)
    fig.suptitle("Feature selection with a matrix variable (the paper's instances): tolerance 1e-3, fixed against gated weight", fontsize=11.5)
    footer(fig, f"OCS runs on {hosts_of(pr)}. The paper ran no stored-matrix solver on this instance, so there is no competitor line. "
                "b12 is the paper's 12,000-iteration budget, b100 a longer run at the smaller size. Every run met 1e-3 inside its first epoch, so the gate never fired.")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "featsel")


# ----------------------------------------------------------------------------- summary table
CLASS_ORDER = ["ot", "fft", "mri", "kron", "sylvester", "rof", "featsel", "turf", "imaging"]
CLASS_NAME = {"ot": "optimal transport", "fft": "FFT LASSO", "mri": "MRI SOCP", "kron": "Kronecker lasso", "sylvester": "Sylvester LS",
              "rof": "ROF denoising", "featsel": "feature selection", "turf": "cluster allocation", "imaging": "3-D imaging"}


def size_label(r):
    m, p = r["meta"], r["meta"]["problem"]
    if p == "turf":
        return f"{m['mode']} {m['tasks']}x{m['servers']}, tol {m['tol']:g}"
    if p in ("imaging",):
        return f"{m['size']}^3"
    if p in ("mri",):
        return f"{m['size']}^2"
    if p == "ot":
        return f"N={m['size']}"
    if p == "kron":
        return f"T={m['size']}"
    if p == "fft":
        return f"N={m['size']}"
    if p == "featsel":
        return r["stem"].replace("featsel_", "")
    if p == "sylvester":
        return "60x50"
    if p == "rof":
        return "442x331"
    return r["stem"]


def summary_table(runs):
    rows = []
    for cls in CLASS_ORDER:
        for (host, stem), v in sorted(pairs(runs, cls).items(), key=lambda kv: (kv[0][0], size_of(list(kv[1].values())[0]) if not math.isnan(size_of(list(kv[1].values())[0])) else 0, kv[0][1])):
            off, on = v.get("off"), v.get("on")
            r0 = off or on
            row = dict(cls=CLASS_NAME[cls], size=size_label(r0), host=HOST_GPU.get(host, host), tol=r0["meta"]["tol"], w0=r0["meta"]["w0"],
                       it_off=off["summary"]["iterations"] if off else "", it_on=on["summary"]["iterations"] if on else "",
                       wall_off=round(solve_wall(off), 2) if off else "", wall_on=round(solve_wall(on), 2) if on else "",
                       adj_off=round(adj_wall(off, on), 2) if off else "", adj_on=round(adj_wall(on, off), 2) if on else "",
                       ratio=(round(on["summary"]["iterations"] / off["summary"]["iterations"], 3) if off and on else ""),
                       status_off=off["summary"]["status"] if off else "", status_on=on["summary"]["status"] if on else "",
                       res_off=off["summary"]["residual"] if off else "", res_on=on["summary"]["residual"] if on else "",
                       weights=(weight_path(on) if on else ""),
                       compiles_on=first_calls(on) if on else "", compile_s_on=on["summary"]["compile_s"] if on else "")
            for k in ("objective", "objective_J", "rel_error_vs_phantom", "marginal_relative", "placement_rate", "rel_error_vs_direct", "true_positives", "rel_error_vs_truth"):
                if off and k in off["summary"]:
                    row["check_off"] = f"{k}={off['summary'][k]:.6g}" if isinstance(off["summary"][k], float) else f"{k}={off['summary'][k]}"
                if on and k in on["summary"]:
                    row["check_on"] = f"{k}={on['summary'][k]:.6g}" if isinstance(on["summary"][k], float) else f"{k}={on['summary'][k]}"
            rows.append(row)
    fields = ["cls", "size", "host", "tol", "w0", "it_off", "it_on", "ratio", "wall_off", "wall_on", "adj_off", "adj_on", "weights", "compiles_on", "compile_s_on",
              "status_off", "status_on", "res_off", "res_on", "check_off", "check_on"]
    with open(os.path.join(CAMP, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    # LaTeX: plain tabular, no booktabs, no dashes; split into two tables so each fits a page
    def tex(s):
        return str(s).replace("_", "\\_").replace("->", "$\\to$").replace("^", "\\^{}").replace("%", "\\%")
    groups = [["optimal transport", "FFT LASSO", "MRI SOCP", "Kronecker lasso"],
              ["Sylvester LS", "ROF denoising", "feature selection", "cluster allocation", "3-D imaging"]]
    lines = []
    for g in groups:
        lines += ["{\\footnotesize", "\\begin{tabular}{llrrrrrl}", "\\hline",
                  "class & size & it.\\ off & it.\\ on & ratio & wall off (s) & wall on (s) & weights used \\\\", "\\hline"]
        for r in rows:
            if r["cls"] not in g:
                continue
            mark = lambda st: "" if st == "solved" else "*"
            lines.append(f"{tex(r['cls'])} & {tex(r['size'])} & {r['it_off']}{mark(r['status_off'])} & {r['it_on']}{mark(r['status_on'])} & {r['ratio']} & "
                         f"{r['adj_off']} & {r['adj_on']} & {tex(r['weights'])} \\\\")
        lines += ["\\hline", "\\end{tabular}}", "", "\\vspace{6pt}", ""]
    with open(os.path.join(CAMP, "summary_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"summary: {len(rows)} pairs")
    return rows


if __name__ == "__main__":
    runs = load_runs()
    print(f"{len(runs)} runs loaded")
    rows = summary_table(runs)
    for f in (fig_ot, fig_fft, fig_mri, fig_kron, fig_sylvester, fig_rof, fig_featsel, fig_turf, fig_imaging):
        try:
            f(runs)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"{f.__name__} failed: {type(exc).__name__}: {exc}")
    with open(os.path.join(CAMP, "summary.txt"), "w") as fh:
        for r in rows:
            fh.write(f"{r['cls']:20s} {r['size']:32s} {r['host']:18s} off {r['it_off']:>8} ({r['status_off']}) on {r['it_on']:>8} ({r['status_on']}) "
                     f"ratio {r['ratio']:>6} wall {r['wall_off']:>8}/{r['wall_on']:>8} s (adj {r['adj_off']:>8}/{r['adj_on']:>8})  w {r['weights']}  {r.get('check_off','')} | {r.get('check_on','')}\n")
    print(open(os.path.join(CAMP, "summary.txt")).read())
