"""Read archived remote experiment results and create small report artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import os
import argparse


def main():
    if os.uname().nodename not in ("Mazeppa","KEVIN-A51"):raise RuntimeError("Report runs on an authorized RTX 5090 host")
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path.home()/"ocs_research_20260914/weight")
    args=ap.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    root=args.root
    dest=root/"reports";dest.mkdir(exist_ok=True)
    audit_path=dest/"cross_family_independent_audit.json"
    audit={v["run"]:v for v in json.loads(audit_path.read_text())} if audit_path.exists() else {}
    rows=[];results={}
    for p in sorted((root/"results").glob("*/result.json")):
        d=json.loads(p.read_text())
        if d.get("stage")!="completed":continue
        a=d["args"];ind=d["independent"]
        archived_accepted=d["accepted"]
        gap=ind.get("certified_objective_relative_gap")
        if p.parent.name in audit:
            newer=audit[p.parent.name];gap=newer["best_relative_gap"]
            d["accepted"]=bool(d["accepted"] and newer["independent_gap_pass"])
        if gap is not None:d["accepted"]=bool(d["accepted"] and -1e-10<=gap<=1e-3)
        reference_error=ind.get("objective_relative_error")
        if "lower bound" in ind.get("reference_kind",""):reference_error=None
        inputs=np.load(p.parent/"inputs.npz")
        ih=hashlib.sha256()
        for key in sorted(inputs.files):
            arr=inputs[key]
            ih.update(f"{key}|{arr.shape}|{arr.dtype}|".encode());ih.update(arr.tobytes())
        row={"run":p.parent.name,"model":a["model"],"seed":a["seed"],"arm":a["arm"],
            "size":a["size"],"tol":a["tol"],"accepted":d["accepted"],"archived_accepted":archived_accepted,"status":d["status"],
            "iterations":d["iterations"],"residual":d["residual"],"weight":d["weight_final"],
            "reference_objective_relative_error":reference_error,
            "certified_objective_relative_gap":gap,
            "reference_kind":ind.get("reference_kind",d["model"].get("reference_status")),
            "feasibility_relative":ind.get("marginal_relative",ind.get("conservation_relative")),
            "warm_loop_seconds":d["warm_loop_seconds"],"certificate_seconds":d["certificate_seconds"],
            "run_wall_seconds":d["run_wall_seconds"],"setup_seconds":d["setup_seconds"],
            "compile_and_first_probe_seconds":d["compile_and_first_probe_seconds"],
            "reference_cpu_seconds":d["model"].get("reference_seconds",0),"total_seconds":d["total_seconds"],
            "source_unchanged":d["source"]["sha256"]==d["source_after"]["sha256"],
            "engine_sha256":d["source"]["sha256"],"pilot_sha256":d.get("pilot_sha256"),
            "inputs_sha256":ih.hexdigest()}
        rows.append(row);results[p.parent.name]=d
    with (dest/"summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (dest/"summary.json").write_text(json.dumps(rows,indent=2))
    lines=["# Weight policy results", "", "All solver runs: Mazeppa RTX 5090. All application references: CPU Clarabel, float64.",
        "Imaging uses independent NumPy float64 feasible-dual bounds. Feature selection additionally checks a feasible-dual bound because Clarabel reports optimal_inaccurate.",
        "Our primal-first PDHG policy port using frozen OCS operators/proxes/certificate; no author-code speed comparison.",
        "Warm loop includes adaptation and monitoring; cold total includes reference, setup, compilation and validation.",
        "Single timing sample per configuration; modest pilots do not establish large-scale speed.", "",
        "| Case | Arm | Requested residual | Accepted after audit | Updates | Warm loop (s) | Cold total (s) | Reference objective error | Certified objective gap | Feasibility |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "_gate" not in r["run"]:continue
        feas=r["feasibility_relative"]
        def fmt_optional(v):return "n/a" if v is None else f"{v:.3g}"
        lines.append(f"| {r['model']}, seed {r['seed']} | {r['arm']} | {r['tol']:.0e} | {r['accepted']} | {r['iterations']} | "
            f"{r['warm_loop_seconds']:.5f} | {r['total_seconds']:.4f} | {fmt_optional(r['reference_objective_relative_error'])} | "
            f"{fmt_optional(r['certified_objective_relative_gap'])} | {fmt_optional(feas)} |")
    lines += ["", "## Identity and quality audit", ""]
    application=[r for r in rows if "_gate" in r["run"]]
    families={}
    for r in application:families.setdefault((r['model'],r['size'],r['seed']),set()).add(r['inputs_sha256'])
    lines += [f"- Input arrays match across arms of every family: {all(len(v)==1 for v in families.values())}.",
              f"- Frozen engine unchanged before/after every completed run: {all(r['source_unchanged'] for r in rows)}.",
              f"- Engine hashes: {sorted(set(r['engine_sha256'] for r in rows))}.",
            "- Numerical tolerance and independent acceptance are separate columns; failed quality gates remain in the table."]
    lines += ["- Derived acceptance includes the independent feature audit; archived_accepted preserves the original pre-audit value without changing any raw record.",
              "- The initial feature fixed endpoint passed the Clarabel comparison but fails the stronger 1e-3 gap gate (1.42e-3). The common feature 1e-4 follow-up passes."]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained",sharey=True)
    colors={"fixed_fast":"#313695","fixed":"#777777","fercoq":"#d73027","rate_only":"#1a9850"}
    sizes={"ot":32,"mcf":120,"lasso":128}
    for seed in [0,1]:
        for j,model in enumerate(["ot","mcf","lasso"]):
            ax=axes[seed,j]
            for arm in ["fixed_fast","fercoq","rate_only"]:
                name=f"{model}{sizes[model]}_s{seed}_{arm}_gate1e3"
                strict=f"{model}{sizes[model]}_s{seed}_{arm}_gate1e4"
                if model=="mcf" and strict in results:name=strict
                if name not in results:continue
                d=results[name];t=d["trajectory"]
                ax.semilogy([r["loop_seconds"] for r in t],[r["residual"] for r in t],
                    color=colors[arm],label=arm,marker="o",markersize=2)
            ax.axhline(1e-4 if model=="mcf" else 1e-3,color="black",linestyle="--",linewidth=.7)
            ax.set_title(f"{model.upper()} size {sizes[model]}, seed {seed}")
            ax.set_xlabel("Warm loop seconds, adaptation included")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Public relative residual");axes[1,0].set_ylabel("Public relative residual")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("OCS policy ablation on one Mazeppa RTX 5090; not author-code solver timings")
    fig.savefig(dest/"application_residuals.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for j,r in enumerate([.001,1000]):
        ax=axes[j]
        for arm in ["fixed","fercoq","rate_only","oracle_fixed"]:
            name=f"q_r{r:g}_{arm}_centered"
            if name not in results:continue
            d=results[name];t=d["trajectory"]
            ax.semilogy([row["iteration"] for row in t],[row["residual"] for row in t],label=arm)
        ax.axhline(1e-3,color="black",linestyle="--",linewidth=.7)
        ax.set_title(f"Centered float32 quadratic, r={r:g}")
        ax.set_xlabel("Iterations");ax.set_ylabel("Distance to exact saddle, infinity norm");ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(dest/"quadratic_centered.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for j,(model,size,suffix) in enumerate([("feature",128,"1e4"),("imaging",32,"1e5")]):
        ax=axes[j]
        for arm in ["fixed_fast","fercoq","rate_only","goldstein"]:
            name=f"{model}{size}_s2_{arm}_gate{suffix}"
            if name not in results:continue
            d=results[name];t=d["trajectory"]
            ax.semilogy([row["loop_seconds"] for row in t],[row["residual"] for row in t],
                label=f"{arm} ({'accepted' if d['accepted'] else 'quality failed'})")
        ax.axhline(1e-4 if model=="feature" else 1e-5,color="black",linestyle="--",linewidth=.7)
        ax.set_title(f"{model}, size {size}, held-out seed 2")
        ax.set_xlabel("Warm loop seconds, adaptation included");ax.set_ylabel("Public relative residual")
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(dest/"cross_family_residuals.png",dpi=170)
    plt.close(fig)
    print(json.dumps({"completed_runs":len(rows),"source_unchanged":all(r["source_unchanged"] for r in rows),
                      "engine_hashes":sorted(set(r["engine_sha256"] for r in rows)),"report_dir":str(dest)}))


if __name__=="__main__":main()
