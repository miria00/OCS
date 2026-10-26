"""Separate KEVIN same-device repeated-process MCF campaign; no pooled timings."""
from pathlib import Path
import argparse
import json
import os
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--report",action="store_true");args=ap.parse_args()
    if os.uname().nodename!="KEVIN-A51":raise RuntimeError("KEVIN only")
    root=Path.home()/"ocs_research_20260914/weight_repeats"
    if args.report:
        import numpy as np
        import ast, hashlib
        current=(root/"scripts/rate_weight_pilot.py").read_text()
        previous=(root/"scripts/pilot_before_repeat_instrumentation.py").read_text()
        def core(text):
            tree=ast.parse(text)
            return {node.name:hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
                    for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ("route","make_model","build_runner")}
        policy=dict(current=core(current),previous=core(previous),unchanged=core(current)==core(previous))
        data=[json.loads(p.read_text()) for p in sorted((root/"results").glob("*/result.json"))]
        output=[]
        fields=["warm_loop_seconds","certificate_seconds","run_wall_seconds","setup_seconds",
                "compile_and_first_probe_seconds","total_seconds"]
        lines=["# MCF fresh-process repeats on KEVIN", "", "Same physical RTX 5090; no timings pooled with Mazeppa.",
            "Each configuration has three fresh processes, each with three additional same-data solves using the compiled executable.",
            "Adaptation state and initial point reset each time. Inclusive replays include monitoring, stopping, trace transfer and independent accuracy.",
            "These are repeated same-instance solves, not validation of varied data in a compiled family.", "",
            "| Seed | Arm | Accepted processes/replays | Updates | Warm loop median [min,max] s | Inclusive replay median [min,max] s | Cold total median [min,max] s |",
            "|---|---|---|---:|---|---|---|"]
        for seed in [0,1]:
            for arm in ["fixed_fast","fercoq"]:
                ds=[d for d in data if d["args"]["seed"]==seed and d["args"]["arm"]==arm]
                if not ds:continue
                stats={}
                for field in fields:
                    v=np.array([d[field] for d in ds]);stats[field]=dict(median=float(np.median(v)),min=float(v.min()),max=float(v.max()))
                replays=[r for d in ds for r in d["same_instance_replays"]]
                values=np.array([r["inclusive_seconds"] for r in replays])
                stats["inclusive_replays"]=dict(median=float(np.median(values)),min=float(values.min()),max=float(values.max()))
                counts=dict(processes=len(ds),accepted_processes=sum(d["accepted"] for d in ds),replays=len(replays),
                    accepted_replays=sum(r["accepted"] for r in replays),bit_identical_replays=sum(r["bit_identical_endpoint"] for r in replays))
                output.append(dict(seed=seed,arm=arm,counts=counts,stats=stats,iterations=[d["iterations"] for d in ds],
                    source_hashes=sorted(set(d["source"]["sha256"] for d in ds)),
                    source_unchanged=all(d["source"]["sha256"]==d["source_after"]["sha256"] for d in ds),
                    reference_statuses=sorted(set(d["model"]["reference_status"] for d in ds)),
                    max_objective_error=max(d["independent"]["objective_relative_error"] for d in ds),
                    max_conservation_error=max(d["independent"]["conservation_relative"] for d in ds)))
                def fmt(k):
                    q=stats[k];return f"{q['median']:.5f} [{q['min']:.5f}, {q['max']:.5f}]"
                lines.append(f"| {seed} | {arm} | {counts['accepted_processes']}/{counts['processes']}; {counts['accepted_replays']}/{counts['replays']} | "
                    f"{ds[0]['iterations']} | {fmt('warm_loop_seconds')} | {fmt('inclusive_replays')} | {fmt('total_seconds')} |")
        (root/"SUMMARY.json").write_text(json.dumps(output,indent=2));(root/"SUMMARY.md").write_text("\n".join(lines)+"\n")
        (root/"POLICY_RECEIPT.json").write_text(json.dumps(policy,indent=2))
        print(json.dumps(output),flush=True);return
    env=dict(os.environ,PYTHONPATH=str(root/"engine"),XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    manifest=[]
    for rep in range(3):
        for seed in [0,1]:
            # Alternate paired order to reduce a monotone warm-clock confound.
            for arm in (["fixed_fast","fercoq"] if (rep+seed)%2==0 else ["fercoq","fixed_fast"]):
                name=f"mcf120_s{seed}_{arm}_rep{rep}"
                command=["flock","-w","1800","/tmp/ocs_research_20260914.gpu.lock","timeout","300s",
                    str(Path.home()/"jaxenv/bin/python"),str(root/"scripts/rate_weight_pilot.py"),
                    "--model","mcf","--size","120","--seed",str(seed),"--arm",arm,"--tol","1e-4",
                    "--max-iters","30000","--seconds","120","--repeat-solves","3","--allowed-host","KEVIN-A51",
                    "--out",str(root/"results"/name)]
                with (root/"results"/f"{name}.log").open("w") as f:ret=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
                rec=dict(name=name,command=command,returncode=ret.returncode)
                p=root/"results"/name/"result.json"
                if p.exists():
                    d=json.loads(p.read_text());rec.update({k:d.get(k) for k in ["accepted","iterations","residual","warm_loop_seconds","total_seconds"]})
                manifest.append(rec);(root/"MANIFEST.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(rec),flush=True)


if __name__=="__main__":main()
