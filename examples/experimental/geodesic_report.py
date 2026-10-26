"""Render saved research results and reclassify with strict 1e-3 checks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def audit(rows):
    details = []
    for row in rows:
        independent = row["solve"]["independent"]
        residual = max(independent["rp"], independent["rd"])
        details.append({"case": row["case"], "alpha": row.get("alpha"),
                        "arm": row.get("arm", "sgd"), "budget": row.get("budget"),
                        "seed": row.get("seed"), "strict_residual": residual,
                        "strict_kkt_pass": residual <= 1e-3,
                        "strict_objective_pass": independent["relative_objective_error"] <= 1e-3,
                        "forward_error": independent["relative_forward_error"],
                        "forward_1e3_pass": independent["relative_forward_error"] <= 1e-3,
                        "additional_forward_1e2_pass": independent["relative_forward_error"] <= 1e-2})
    return {"rows": len(details), "strict_kkt_pass": sum(r["strict_kkt_pass"] for r in details),
            "strict_objective_pass": sum(r["strict_objective_pass"] for r in details),
            "strict_both_pass": sum(r["strict_kkt_pass"] and r["strict_objective_pass"] for r in details),
            "forward_1e3_pass": sum(r["forward_1e3_pass"] for r in details),
            "maximum_forward_error": max(r["forward_error"] for r in details), "details": details}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args(); folder = args.directory
    exact = json.loads((folder/"results/exact_gate_tol1e3.json").read_text())
    sgd = json.loads((folder/"results/sgd_reference.json").read_text())
    audits = {"exact": audit(exact["records"]), "sgd": audit(sgd["records"])}
    (folder/"results/strict_audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    out = folder/"plots"; out.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, case in zip(axs, ("rotated10", "rotated100")):
        subset = [r for r in sgd["records"] if r["case"] == case]
        for seed in sgd["config"]["seeds"]:
            rr = [r for r in subset if r["seed"] == seed]
            ax.loglog([r["budget"] for r in rr], [r["construction_objective_gap"] for r in rr],
                      marker="o", label=f"seed {seed}")
        ax.set_title(case); ax.set_xlabel("SGD updates (2 operator calls each)")
        ax.set_ylabel("Construction objective gap to exact reference")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("CPU float64 oracle reference; common boxed mixed-metric objective")
    fig.savefig(out/"construction_gap.png", dpi=180); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, case in zip(axs, ("rotated10", "rotated100")):
        subset = [r for r in sgd["records"] if r["case"] == case]
        for seed in sgd["config"]["seeds"]:
            rr = [r for r in subset if r["seed"] == seed]
            ax.semilogx([r["budget"] for r in rr], [r["solve"]["iterations"] for r in rr],
                        marker="o", label=f"SGD seed {seed}")
        ref = [r for r in exact["records"] if r["case"] == case and r["alpha"] > .001]
        for arm, color in (("diagonal", "black"), ("full", "tab:green")):
            r = next(r for r in ref if r["arm"] == arm)
            ax.axhline(r["solve"]["iterations"], color=color, ls="--", label=f"exact {arm}")
        ax.set_title(case); ax.set_xlabel("CPU reference construction updates")
        ax.set_ylabel("Float32 GPU PDHG updates to 1e-3 KKT")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Tiny research PDHG on KEVIN RTX 5090; not public OCS speed claims")
    fig.savefig(out/"downstream_iterations.png", dpi=180); plt.close(fig)
    lines = ["# Derived strict audit", "", "All numerical checks below use <=1e-3 with no allowance.", "",
             "| Dataset | Strict KKT | Objective error | Both | Forward error <=1e-3 |",
             "|---|---:|---:|---:|---:|"]
    for name, item in audits.items():
        n = item["rows"]
        lines.append(f"| {name} | {item['strict_kkt_pass']}/{n} | {item['strict_objective_pass']}/{n} | {item['strict_both_pass']}/{n} | {item['forward_1e3_pass']}/{n} |")
    lines += ["", "The original pilot also recorded a 1e-2 forward-error check; it is not a 1e-3 forward-accuracy claim.",
              "", "## CPU reference construction cost at 1,000 updates", "",
              "| Case | CPU seconds (min-max) | Solve updates | Construction objective gap |", "|---|---:|---:|---:|"]
    for case in ("isotropic", "aligned100", "rotated10", "rotated100"):
        rows = [r for r in sgd["records"] if r["case"] == case and r["budget"] == 1000]
        times = [r["constructor_seconds_cpu_f64"] for r in rows]
        it = [r["solve"]["iterations"] for r in rows]
        gaps = [r["construction_objective_gap"] for r in rows]
        lines.append(f"| {case} | {min(times):.3f}-{max(times):.3f} | {min(it)}-{max(it)} | {min(gaps):.3g}-{max(gaps):.3g} |")
    lines += ["", "This CPU correctness constructor is not a one-shot speed win. Its setup must be charged; a GPU/batched implementation has not been measured."]
    (folder/"STRICT_AUDIT.md").write_text("\n".join(lines)+"\n")
    family_files=list((folder/"results").glob("family_f[123].json"))
    if family_files:
        family_records=[]
        flines=["# Bounded nonsmooth metric gate", "", "All arms run the same existing float32 GPU research loop on KEVIN RTX 5090.",
                "Pinned SMFE is renormalized by the same exact tiny-matrix SVD used for the new metrics.",
                "New constructors are CPU float64 explicit references; this is not a scalable setup-time comparison.", "",
                "Acceptance: strict public relative KKT <=1e-3 and independent Clarabel objective error <=1e-3.", "",
                "| Family | Arm | Updates | KKT | Objective error | Constructor seconds | Warm updates estimate, ms |", "|---|---|---:|---:|---:|---:|---:|"]
        family_sets=[]
        for path in sorted(family_files):
            data=json.loads(path.read_text()); family_sets.append(data)
            for row in data["records"]:
                family_records.append({"family":data["config"]["family"],**row})
                flines.append(f"| {data['config']['family']} | {row['arm']} | {row['iterations']} | {row['public_residual']:.3g} | {row['relative_objective_error']:.3g} | {row['setup']['seconds']:.3f} | {1000*row['estimated_warm_update_seconds_excluding_checks']:.3f} |")
        flines += ["", "All times above exclude matrix materialization and common compilation/reference auditing unless explicitly named. Warm update estimates exclude stopping-check cost. They do not establish an end-to-end speedup.",
                   "", "## Setup amortization diagnostic", "", "For each new arm below, use constructor time plus N times the measured warm-update estimate; no precision conversion or GPU constructor performance is assumed.", "",
                   "| Family | New arm | Saves warm update time? | Setup break-even vs shipped (solves) |", "|---|---|---|---:|"]
        for data in family_sets:
            baseline=next(r for r in data["records"] if r["arm"]=="shipped_smfe")
            for row in data["records"]:
                if row["arm"]=="shipped_smfe":continue
                saving=baseline["estimated_warm_update_seconds_excluding_checks"]-row["estimated_warm_update_seconds_excluding_checks"]
                setup_difference=row["setup"]["seconds"]-baseline["setup"]["seconds"]
                threshold="none (updates slower)" if saving<=0 else str(max(1,int(np.ceil(max(setup_difference,0.)/saving))))
                flines.append(f"| {data['config']['family']} | {row['arm']} | {'yes' if saving>0 else 'no'} | {threshold} |")
        flines += ["", "These break-even figures are diagnostics, not a release gate: shipped setup includes a first-call GPU constructor, the new arms use tiny explicit CPU references, and shared compiled-kernel family setup has not been matched.",
                   "", "F3 uses a nonzero 4^3 phantom through the unchanged y parameter; the source fixture's fixed large-volume slices would be empty at this size.",
                   "", "The F2 stochastic constructor needs fewer updates than the more accurately optimized FULL constructor. This is direct evidence that the construction surrogate is not the downstream time objective."]
        (folder/"FAMILY_GATE.md").write_text("\n".join(flines)+"\n")
        (folder/"results/family_summary.json").write_text(json.dumps({"distinct_problem_instances":len(family_sets),"arms":len(family_records),
                   "strict_accepted":sum(r["accepted"] for r in family_records),"records":family_records},indent=2)+"\n")
        fig,axs=plt.subplots(1,len(family_sets),figsize=(11,3.4),constrained_layout=True)
        axs=np.atleast_1d(axs)
        for ax,data in zip(axs,family_sets):
            rr=data["records"]
            ax.bar(range(len(rr)),[r["iterations"] for r in rr],color=["gray","tab:blue","tab:green","tab:orange"])
            ax.set_xticks(range(len(rr)),["SMFE","Restricted","Full ref","Full SGD"],rotation=30,ha="right")
            ax.set_title(data["config"]["family"]);ax.set_ylabel("Updates to strict KKT + objective checks")
            ax.grid(axis="y",alpha=.25)
        fig.suptitle("Tiny nonsmooth controls, common float32 RTX 5090 research loop")
        fig.savefig(out/"nonsmooth_family_iterations.png",dpi=180);plt.close(fig)
    print(json.dumps({name: {k:v for k,v in item.items() if k!='details'} for name,item in audits.items()}, indent=2))


if __name__ == "__main__":
    main()
