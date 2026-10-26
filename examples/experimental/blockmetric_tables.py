"""Verdict tables for the block-metric pilot (matplotlib-free; stdlib + numpy).

    python blockmetric_tables.py <results dir>

Writes verdict_tables.md next to the CSVs and prints it.  Per family and arm:
iterations to the 1e-5 certificate, setup time and loop time SEPARATELY (so a
setup-path difference such as A4 skipping the SGD equilibration can never be
read as a block-metric win), total warm time, cold solve time, the setup
breakdown (equilibration / Gram / eigendecomposition / norm), and peak bytes.
Contended rows are dropped.  The coupled (overlap) and separable (partition)
F1 variants are reported separately and never averaged together.
"""
import csv, os, sys, math

csv.field_size_limit(10 ** 9)   # the per-block residual history is a long field
from collections import OrderedDict

ORDER = ["A0", "A0std", "A1", "A2", "A3", "A3s1", "A3s4", "A3s16", "A4p1", "A4p2", "A4p3", "A5"]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if str(r.get("contended", "")).strip() not in ("1", "true", "True")]


def f(r, k):
    v = r.get(k, "")
    if v in ("", None):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def arm(r):
    return r.get("tag") or r.get("arm")


def fmt(x, nd=2):
    return "-" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ratio(a, b):
    return float("nan") if (b == 0 or math.isnan(a) or math.isnan(b)) else a / b


def table(rows, title, group_key, group_label, out):
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    for g, rs in groups.items():
        out.append(f"\n### {title} - {group_label(g)}\n")
        base = {arm(r): r for r in rs}
        a1 = base.get("A1"); a0 = base.get("A0"); a3 = base.get("A3")
        out.append("| arm | iters | certified | setup s | loop s | verify s | total warm s | cold s | equil s | gram s | eig s | norm s | peak MB | loop/A1 | total/A1 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in ORDER:
            r = base.get(a)
            if r is None:
                continue
            peak = f(r, "peak_bytes") / 1e6
            lr = ratio(f(r, "iter_s"), f(a1, "iter_s")) if a1 else float("nan")
            tr = ratio(f(r, "total_warm_s"), f(a1, "total_warm_s")) if a1 else float("nan")
            out.append(f"| {a} | {int(f(r,'iters'))} | {'yes' if f(r,'certified')==1 else 'NO'} | "
                       f"{fmt(f(r,'setup_s'))} | {fmt(f(r,'iter_s'),3)} | {fmt(f(r,'verify_s'),3)} | "
                       f"{fmt(f(r,'total_warm_s'))} | {fmt(f(r,'cold_solve_s'))} | {fmt(f(r,'equil_s'))} | "
                       f"{fmt(f(r,'gram_s'),3)} | {fmt(f(r,'eig_s'),3)} | {fmt(f(r,'norm_s'),3)} | "
                       f"{fmt(peak,0)} | {fmt(lr)} | {fmt(tr)} |")
        # explicit headline ratios
        if a3 and a1:
            out.append(f"\nA3 exact full blocks vs A1 tied-scalar diagonal: iterations "
                       f"{fmt(ratio(f(a1,'iters'), f(a3,'iters')))}x fewer, loop time "
                       f"{fmt(ratio(f(a3,'iter_s'), f(a1,'iter_s')))}x, total warm time "
                       f"{fmt(ratio(f(a3,'total_warm_s'), f(a1,'total_warm_s')))}x, peak memory "
                       f"{fmt(ratio(f(a3,'peak_bytes'), f(a1,'peak_bytes')))}x.")
        if a3 and a0:
            out.append(f"A3 vs A0 shipped engine: total warm time {fmt(ratio(f(a3,'total_warm_s'), f(a0,'total_warm_s')))}x, "
                       f"iterations {fmt(ratio(f(a0,'iters'), f(a3,'iters')))}x fewer.")
        for s in ("A3s1", "A3s4", "A3s16"):
            rs_ = base.get(s)
            if rs_ and a3:
                out.append(f"{s} sketched vs A3 exact: iterations {fmt(ratio(f(rs_,'iters'), f(a3,'iters')))}x, "
                           f"total warm time {fmt(ratio(f(rs_,'total_warm_s'), f(a3,'total_warm_s')))}x"
                           + (f", Gram relative error {f(rs_,'block0_gram_sketch_relerr'):.3g}."
                              if not math.isnan(f(rs_, "block0_gram_sketch_relerr")) else "."))
        for s in ("A4p1", "A4p2", "A4p3"):
            r4 = base.get(s)
            if r4 and a1:
                out.append(f"{s} Liu-Xu-Yin vs A1 on LOOP time only (its setup skips the SGD equilibration, "
                           f"{fmt(f(r4,'setup_s'))} s vs {fmt(f(a1,'setup_s'))} s, which is a setup-path difference, "
                           f"not a metric effect): {fmt(ratio(f(r4,'iter_s'), f(a1,'iter_s')))}x.")


def main():
    d = sys.argv[1]
    out = ["# Block-metric pilot: verdict tables",
           "",
           "Contended rows (measured while another pilot process held the same host) are excluded.",
           "Setup and loop time are separate columns throughout: arms differ in which setup they pay",
           "(A4 skips the SGD equilibration entirely), so only loop time compares the iterations themselves.",
           "The coupled overlap variant and the separable partition variants are reported separately."]
    ov = load(os.path.join(d, "f1_overlap_results.csv"))
    if ov:
        table(ov, "F1 COUPLED (overlap: two stacked partitions, every coordinate in 2 blocks)",
              lambda r: (int(float(r["n"])), float(r["rho"])),
              lambda g: f"n = {g[0]:,}, rho = {g[1]:g}", out)
    f1 = load(os.path.join(d, "f1_results.csv"))
    if f1:
        table(f1, "F1 SEPARABLE SANITY CHECK (one partition: the problem splits into n/k independent k-dim problems)",
              lambda r: (r["variant"], int(float(r["n"])), float(r["rho"]), int(float(r.get("k", 4)))),
              lambda g: f"{g[0]}, n = {g[1]:,}, rho = {g[2]:g}, k = {g[3]}", out)
    sc = load(os.path.join(d, "f2_scale_results.csv"))
    if sc:
        out.append("\n## Does the loop-time win survive as a total-time win when the loop dominates setup?\n")
        out.append("F2 strong mixing, pushing size and tolerance independently. `loop share` is loop/(setup+loop+verify).")
        table(sc, "F2 strong mixing, loop-dominance sweep",
              lambda r: (int(float(r["n"]) / 3) ** 0.5, float(r["tol"])),
              lambda g: f"L = {int(round(g[0]))}, tol = {g[1]:g}", out)
        out.append("\n| L | tol | loop share A1 | loop share A3 | A3/A1 loop | A3/A1 total | A3s16/A1 loop | A3s16/A1 total |")
        out.append("|---|---|---|---|---|---|---|---|")
        groups = OrderedDict()
        for r in sc:
            groups.setdefault((int(round((float(r["n"]) / 3) ** 0.5)), float(r["tol"])), []).append(r)
        for (L, tol), rs in groups.items():
            b = {arm(r): r for r in rs}
            a1, a3, a16 = b.get("A1"), b.get("A3"), b.get("A3s16")
            if not (a1 and a3):
                continue
            share = lambda r: f(r, "iter_s") / max(f(r, "total_warm_s"), 1e-9)
            out.append(f"| {L} | {tol:g} | {fmt(share(a1))} | {fmt(share(a3))} | "
                       f"{fmt(ratio(f(a3,'iter_s'), f(a1,'iter_s')))} | {fmt(ratio(f(a3,'total_warm_s'), f(a1,'total_warm_s')))} | "
                       f"{fmt(ratio(f(a16,'iter_s'), f(a1,'iter_s')) if a16 else float('nan'))} | "
                       f"{fmt(ratio(f(a16,'total_warm_s'), f(a1,'total_warm_s')) if a16 else float('nan'))} |")
    for name, title in (("f2_results.csv", "F2 multichannel deblurring (3-channel coupled blocks)"),
                        ("f2_mu0.1_results.csv", "F2 multichannel deblurring, mu = 0.1"),
                        ("f3_results.csv", "F3 no-regression control: 3-D TV imaging 64^3"),
                        ("f1_mu0.1_results.csv", "F1 at mu = 0.1"),
                        ("f1_kappa_results.csv", "F1 kappa_max sensitivity"),
                        ("f1_a5_results.csv", "F1 tiny instances with the A5 optimal-diagonal SDP reference")):
        rows = load(os.path.join(d, name))
        if rows:
            table(rows, title, lambda r: (r.get("variant", ""), int(float(r["n"])), float(r["rho"]),
                                          float(r.get("mu", 0.01)), float(r.get("kappa_max", 100))),
                  lambda g: f"{g[0]}, n = {g[1]:,}, rho = {g[2]:g}, mu = {g[3]:g}, kappa_max = {g[4]:g}", out)
    f4 = load(os.path.join(d, "f4_results.csv"))
    if f4:
        out.append("\n### F4 repeated solves: amortizing the setup over N instances\n")
        out.append(f"Model: {f4[0].get('name','?')} - this is the F1 SEPARABLE PARTITION model with fresh b per instance,")
        out.append("not an independent family and not a correlated-channel study.\n")
        out.append("| arm | N = 1 | N = 10 | N = 100 | setup s | iters (last) |")
        out.append("|---|---|---|---|---|---|")
        for a in ORDER:
            rs = sorted([r for r in f4 if arm(r) == a], key=lambda r: float(r["instance"]))
            if not rs:
                continue
            cells = []
            for N in (1, 10, 100):
                sub = [r for r in rs if float(r["instance"]) < N]
                cells.append(fmt(f(sub[-1], "cum_s") / N, 3) if len(sub) == N else "-")
            out.append(f"| {a} | {cells[0]} | {cells[1]} | {cells[2]} | {fmt(f(rs[0],'setup_s'))} | {int(f(rs[-1],'iters'))} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(d, "verdict_tables.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
