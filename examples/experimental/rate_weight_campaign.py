"""Serialized bounded remote-only driver for the weight policy experiment."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["quadratic","modest","holdout","quadratic_centered","quadratic_x64","mcf_accuracy","cross_family","goldstein_mcf","feature_accuracy"],required=True)
    args=parser.parse_args()
    if os.uname().nodename!="Mazeppa":
        raise RuntimeError("Campaign is restricted to Mazeppa")
    root=Path.home()/"ocs_research_20260914/weight"
    env=dict(os.environ,PYTHONPATH=str(root/"engine"),XLA_PYTHON_CLIENT_PREALLOCATE="false",
             OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    jobs=[]
    if args.phase.startswith("quadratic"):
        for curvature in [0.001,1,1000]:
            for arm in ["fixed","fercoq","rate_only","oracle_fixed"]:
                weight=curvature if arm=="oracle_fixed" else 1
                suffix="" if args.phase=="quadratic" else args.phase.removeprefix("quadratic")
                opts=["--model","quadratic","--curvature",str(curvature),"--arm","fixed" if arm=="oracle_fixed" else arm,
                      "--weight",str(weight),"--max-iters","10000"]
                if args.phase=="quadratic_centered":opts+=["--quadratic-centered","--tol","1e-3"]
                if args.phase=="quadratic_x64":opts+=["--quadratic-x64","--tol","1e-5"]
                jobs.append((f"q_r{curvature:g}_{arm}{suffix}",opts))
    elif args.phase=="feature_accuracy":
        for arm in ["fixed_fast","fercoq","rate_only","goldstein"]:
            jobs.append((f"feature128_s2_{arm}_gate1e4",["--model","feature","--size","128",
                "--seed","2","--arm",arm,"--max-iters","30000","--seconds","120","--tol","1e-4"]))
    elif args.phase=="cross_family":
        for model,size,tol in [("feature",128,"1e-3"),("imaging",32,"1e-5")]:
            for arm in ["fixed_fast","fercoq","rate_only","goldstein"]:
                jobs.append((f"{model}{size}_s2_{arm}_gate{tol.replace('-','')}",["--model",model,"--size",str(size),
                    "--seed","2","--arm",arm,"--max-iters","30000","--seconds","120","--tol",tol]))
    elif args.phase=="goldstein_mcf":
        for seed in [0,1]:
            jobs.append((f"mcf120_s{seed}_goldstein_gate1e4",["--model","mcf","--size","120",
                "--seed",str(seed),"--arm","goldstein","--max-iters","30000","--seconds","120","--tol","1e-4"]))
    elif args.phase=="mcf_accuracy":
        for seed in [0,1]:
            for arm in ["fixed","fixed_fast","fercoq","rate_only"]:
                jobs.append((f"mcf120_s{seed}_{arm}_gate1e4",["--model","mcf","--size","120",
                    "--seed",str(seed),"--arm",arm,"--max-iters","30000","--seconds","120","--tol","1e-4"]))
    else:
        seeds=[0] if args.phase=="modest" else [1]
        for seed in seeds:
            for model,size in [("ot",32),("mcf",120),("lasso",128)]:
                for arm in ["fixed","fixed_fast","fercoq","rate_only"]:
                    jobs.append((f"{model}{size}_s{seed}_{arm}_gate1e3",["--model",model,"--size",str(size),
                        "--seed",str(seed),"--arm",arm,"--max-iters","30000","--seconds","120","--tol","1e-3"]))
    manifest=[]
    for name,options in jobs:
        out=root/"results"/name
        command=["flock","-w","1800","/tmp/ocs_research_20260914.gpu.lock","timeout","300s",
                 str(Path.home()/"jaxenv/bin/python"),str(root/"scripts/rate_weight_pilot.py"),
                 *options,"--out",str(out)]
        entry={"name":name,"command":command}
        with (root/"results"/f"{name}.log").open("w") as log:
            ret=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        entry["returncode"]=ret.returncode
        result=out/"result.json"
        if result.exists():
            data=json.loads(result.read_text())
            entry.update({k:data.get(k) for k in ["status","accepted","iterations","residual","weight_final",
                "warm_loop_seconds","run_wall_seconds","total_seconds","independent"]})
        manifest.append(entry)
        (root/"results"/f"{args.phase}_manifest.json").write_text(json.dumps(manifest,indent=2))
        print(json.dumps(entry),flush=True)


if __name__=="__main__":main()
