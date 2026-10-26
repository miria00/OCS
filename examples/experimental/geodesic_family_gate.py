"""Bounded F1/F2/F3 metric gate with identical research loop and acceptance.

The shipped arm calls pinned SMFE and is exactly norm-renormalized along with
the other arms. New constructors here remain tiny CPU explicit references;
this file deliberately cannot establish a scalable setup-time advantage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from geodesic_metric_pilot import sym_function
from geodesic_f1_control import construction
from blockmetric_models import blockmetric_f1,blockmetric_f2,blockmetric_f3
from blockmetric_pilot import Loop,shipped_scaling


def build(family,seed):
    if family=="F1":
        return blockmetric_f1(n=16,k=4,rho=.9,variant="overlap",seed=seed,mu=.01),"original small F1-overlap fixture"
    if family=="F2":
        return blockmetric_f2(L=4,mixing="strong",seed=seed,mu=.01),"original small F2 channel-mixing fixture"
    problem,spec=blockmetric_f3(L=4,seed=seed)
    # The author's fixed slice indices generate an identically zero phantom
    # below L=13. Supply nonzero data to the SAME parameterized FFT/TV model;
    # no operator, regularizer, or constraint is changed.
    phantom=np.zeros((4,4,4),np.float32); phantom[1:3,1:3,1:3]=1.
    phantom[2:4,0:2,1:3]+=.4
    y=spec.dual_blocks[0]["op"].matvec(jnp.asarray(phantom.reshape(-1)))
    spec.data={"y":y}; spec.dual_blocks[0]["offset"]=-y
    spec.phantom=phantom.reshape(-1)
    return (problem,spec),"nonzero 4^3 phantom in unchanged FFT-mask/nonnegative/isotropic-TV model; original fixed slices are empty at this size"


def independent_objective(spec,matrices,x):
    x=np.asarray(x,np.float64)
    value=spec.mu/2*(x@x) if spec.primal=="ridge" else 0.
    for term,k in zip(spec.dual_blocks,matrices):
        r=k@x+np.asarray(term["offset"],np.float64)
        value+=.5*(r@r) if term["kind"]=="sum_squares" else term["lam"]*np.linalg.norm(r.reshape(-1,term["k"]),axis=1).sum()
    return float(value)


def solve_reference(spec,matrices):
    x=cp.Variable(spec.n)
    obj=spec.mu/2*cp.sum_squares(x) if spec.primal=="ridge" else 0
    for term,k in zip(spec.dual_blocks,matrices):
        r=k@x+np.asarray(term["offset"],np.float64)
        if term["kind"]=="sum_squares": obj+=cp.sum_squares(r)/2
        else: obj+=term["lam"]*cp.sum(cp.norm(cp.reshape(r,(-1,term["k"]),order="C"),axis=1))
    problem=cp.Problem(cp.Minimize(obj),[x>=0] if spec.primal=="nonneg" else [])
    t=time.perf_counter()
    problem.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=500)
    if problem.status not in ("optimal","optimal_inaccurate") or x.value is None:
        raise RuntimeError(f"Independent reference status {problem.status}")
    return {"status":problem.status,"objective":float(problem.value),
            "reevaluated_objective":independent_objective(spec,matrices,x.value),
            "seconds_cpu_f64":time.perf_counter()-t,"x":x.value.tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--family",choices=["F1","F2","F3"],required=True)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-blocks",type=int,default=200)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    if not any(d.platform=="gpu" for d in jax.devices()):raise RuntimeError("Authorized GPU required")
    (problem,spec),fixture_note=build(args.family,args.seed)
    matrices=[];sizes=[];term_slices=[];position=0
    materialize_start=time.perf_counter()
    for term in spec.dual_blocks:
        matrix=jax.jit(jax.vmap(term["op"].matvec))(jnp.eye(spec.n,dtype=jnp.float32)).T
        matrices.append(np.asarray(matrix,np.float64))
        rows=term["op"].shape[0];ks=1 if term["kind"]=="sum_squares" else term["k"]
        number=rows//ks;sizes.extend([ks]*number);term_slices.append((position,position+number));position+=number
    k=np.vstack(matrices);materialize_seconds=time.perf_counter()-materialize_start
    reference=solve_reference(spec,matrices)
    comp=problem.compile(tol=1e-3,max_blocks=args.max_blocks,block=50,pdhg_variant="standard")
    out={"config":{"family":args.family,"seed":args.seed,"host":platform.node(),"devices":[str(d) for d in jax.devices()],
                   "n":spec.n,"m":len(k),"fixture":fixture_note,"construction_mu":.1,"M":2.,"r":spec.n/len(k),"s":1.,
                   "safety":.95,"weight":1.,"matrix_materialization_seconds":materialize_seconds,
                   "all_iteration_dtype":"float32 GPU","new_construction_dtype":"float64 CPU explicit reference",
                   "shipped_arm":"pinned SMFE metric on identical research loop, with common exact SVD renormalization",
                   "acceptance":"public KKT<=1e-3 AND independent relative objective error<=1e-3; nonnegativity checked",
                   "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"reference":reference,"records":[]}
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    for name,structure in (("shipped_smfe","scalar"),("new_restricted","scalar"),("new_full","full"),("new_full_sgd200","full")):
        if name=="shipped_smfe":
            sc=shipped_scaling(spec);b=np.asarray(sc["tau"],np.float64)
            raw=np.asarray(sc["sigma"],np.float64)
            a=[np.diag(z) for z in np.split(raw,np.cumsum(sizes)[:-1])]
            setup={"seconds":sc["seconds"],"backend":"GPU float32", "probes":2,"updates":80}
        else:
            a,b,info=construction(k,structure,seed=args.seed,blocksizes=sizes,
                                 stochastic=name.endswith("sgd200"),budget=200 if name.endswith("sgd200") else 1500)
            setup={**info,"seconds":info["seconds_cpu_f64"],"backend":"CPU float64 tiny reference"}
        blocks=np.split(k,np.cumsum(sizes)[:-1])
        scaled=np.vstack([sym_function(ai,np.sqrt)@ki for ai,ki in zip(a,blocks)])*np.sqrt(b)
        norm=float(np.linalg.norm(scaled,2));factor=.95/norm
        sigma=[ai*factor for ai in a];tau=b*factor;metrics=[]
        for term,(start,end) in zip(spec.dual_blocks,term_slices):
            subset=np.asarray(sigma[start:end])
            if term["kind"]=="sum_squares":
                metrics.append(("vec",jnp.asarray(subset[:,0,0],jnp.float32)))
            elif structure=="scalar":
                metrics.append(("scalar",jnp.asarray(subset[:,0,0,None],jnp.float32)))
            else:
                eig,q=np.linalg.eigh(subset)
                metrics.append(("full",jnp.asarray(q,jnp.float32),jnp.asarray(eig,jnp.float32)))
        loop=Loop(spec,jnp.asarray(tau,jnp.float32),metrics,block=50)
        x,ys,xbar=loop.init_state();history=[];start=time.perf_counter()
        first_kkt=None
        for iteration in range(1,args.max_blocks+1):
            x,ys,xbar=loop._run_block(x,ys,xbar,loop.T,loop.marrays,loop.offs)
            jax.block_until_ready(x);y=loop.returned_pair(x,ys)
            certificate=comp.certificate(x,y,spec.data)
            residual=max(float(certificate["r_primal"]),float(certificate["r_dual"]))
            objective_value=independent_objective(spec,matrices,x)
            error=abs(objective_value-reference["objective"])/max(abs(reference["objective"]),1e-30)
            nonnegative=spec.primal!="nonneg" or float(jnp.min(x))>=0.
            if first_kkt is None and residual<=1e-3:first_kkt=iteration*50
            history.append([iteration*50,residual,error,time.perf_counter()-start])
            if residual<=1e-3 and error<=1e-3 and nonnegative:break
            if time.perf_counter()-start>45:break
        cold=time.perf_counter()-start
        bench=loop.bench(reps=3)
        row={"arm":name,"setup":setup,"iterations":iteration*50,"first_public_kkt_iteration":first_kkt,
             "public_residual":residual,"reference_objective":reference["objective"],"objective":objective_value,
             "relative_objective_error":error,"nonnegative":nonnegative,
             "accepted":bool(residual<=1e-3 and error<=1e-3 and nonnegative),"cold_solve_plus_checks_seconds":cold,
             "exact_coupling_norm":.95,"normalization_factor":factor,"warm_seconds_per_50_update_block":bench,
             "estimated_warm_update_seconds_excluding_checks":bench["block"]*iteration,
             "setup_plus_estimated_warm_updates_excluding_checks":setup["seconds"]+bench["block"]*iteration,
             "history":history,"x":np.asarray(x).tolist(),"y":np.asarray(y).tolist()}
        out["records"].append(row);args.output.write_text(json.dumps(out,indent=2)+"\n")
        print(json.dumps({"family":args.family,"arm":name,"iterations":iteration*50,"residual":residual,
                          "objective_error":error,"accepted":row["accepted"],"setup_seconds":setup["seconds"]}),flush=True)


if __name__=="__main__":main()
