"""Small coupled nonsmooth F1 control using the existing full-metric prox loop.

The tiny explicit operator is used ONLY for CPU float64 construction references
and independent dual bounds. Downstream operator/prox evaluation is the frozen
JAX float32 research loop. This is not a public-engine performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import jax
import jax.numpy as jnp

from geodesic_metric_pilot import sym_function, exp_step
from geodesic_sgd_pilot import geometric_mean
from blockmetric_models import blockmetric_f1
from blockmetric_pilot import Loop


def objective(a, b, k, mu, r):
    kk = np.split(k,np.cumsum([len(ai) for ai in a])[:-1])
    value = .5*sum(np.trace(ai @ (ki*b) @ ki.T) for ai,ki in zip(a,kk))
    la = [np.log(np.linalg.eigvalsh(ai)) for ai in a]
    lb = np.log(b)
    value -= r/2*sum(z.sum() for z in la)+.5*lb.sum()
    value += mu/8*(sum(z@z for z in la)+lb@lb)
    return float(value)


def construction(k, arm, mu=.1, box=2., budget=1500, seed=0, stochastic=False, blocksizes=None):
    m,n = k.shape; r=n/m
    sizes=blocksizes if blocksizes is not None else [4]*(m//4)
    if sum(sizes)!=m:
        raise ValueError("The declared disjoint blocks must cover every row")
    cuts=np.cumsum(sizes)[:-1]
    kk=np.split(k,cuts)
    a = [np.eye(size) for size in sizes]; b=np.ones(n)
    rng = np.random.default_rng(seed); avg_a=None; avg_logb=None; weights=0.
    start = time.perf_counter(); previous=objective(a,b,k,mu,r)
    eta = 1.; mappings=[]; calls=0
    for t in range(1,budget+1):
        if stochastic:
            ah=[sym_function(ai,np.sqrt) for ai in a]
            w=np.split(k @ (np.sqrt(b)*rng.choice([-1.,1.],n)),cuts)
            z=k.T @ np.concatenate([half @ rng.choice([-1.,1.],size) for half,size in zip(ah,sizes)])
            grams=[np.outer(wi,wi) for wi in w]; col=z*z; calls+=2
        else:
            grams=[(ki*b) @ ki.T for ki in kk]
            col=sum(np.sum(ki*(ai @ ki),axis=0) for ai,ki in zip(a,kk))
        ga=[.5*ai @ gi @ ai-r/2*ai+mu/4*ai @ sym_function(ai,np.log)
            for ai,gi in zip(a,grams)]
        if arm=="scalar":
            ga=[np.eye(size)*np.trace(gi)/size for gi,size in zip(ga,sizes)]
        elif arm=="diagonal":
            ga=[np.diag(np.diag(gi)) for gi in ga]
        bg_log=.5*b*col-.5+mu/4*np.log(b)
        if stochastic:
            wt=t+100-1; weights+=wt
            if avg_a is None:
                avg_a=[ai.copy() for ai in a]; avg_logb=np.log(b)
            else:
                avg_a=[geometric_mean(old,new,wt/weights) for old,new in zip(avg_a,a)]
                avg_logb=(1-wt/weights)*avg_logb+wt/weights*np.log(b)
            eta=2/((mu/4)*(t+100))
        for backtrack in range(35):
            an=[exp_step(ai,gi,eta,box) for ai,gi in zip(a,ga)]
            bn=np.exp(np.clip(np.log(b)-eta*bg_log,-2*box,2*box))
            next_value=objective(an,bn,k,mu,r)
            d2=np.sum((np.log(bn)-np.log(b))**2)
            for old,new in zip(a,an):
                inv=sym_function(old,lambda v:1/np.sqrt(v))
                d2+=np.sum(np.log(np.linalg.eigvalsh(inv@new@inv))**2)
            if stochastic or next_value <= previous-1e-4*d2/eta+1e-12:
                break
            eta*=.5
        else:
            raise RuntimeError("Deterministic manifold line search failed")
        a,b=an,bn; previous=next_value
        mapping=float(np.sqrt(d2)/eta)
        if not stochastic and (t%100==0 or t==budget):
            mappings.append([t,mapping,next_value])
        if not stochastic and mapping<1e-7:
            break
        if not stochastic:
            eta=min(eta*1.1,2.)
    if stochastic:
        a,b=avg_a,np.exp(avg_logb)
    return a,b,{"seconds_cpu_f64":time.perf_counter()-start,"updates":t,"operator_calls":calls,
                "objective":objective(a,b,k,mu,r),"last_mapping_norm":mapping,
                "mapping_applies_to_returned_average":not stochastic,"history":mappings,
                "reference_converged":bool(not stochastic and mapping<1e-6),
                "targets":{"r":r,"s":1.},"mu":mu,"M":box}


def independent(x,y,k,offset,ridge,lam,block):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    residual=(k @ x+offset).reshape(-1,block)
    primal=ridge/2*(x@x)+lam*np.linalg.norm(residual,axis=1).sum()
    yy=y.reshape(-1,block)
    yd=(yy/np.maximum(1.,np.linalg.norm(yy,axis=1,keepdims=True)/lam)).reshape(-1)
    dual=offset@yd-np.linalg.norm(k.T@yd)**2/(2*ridge)
    bracket=(primal-dual)/max(abs(primal),abs(dual),1e-30)
    stat=ridge*x+k.T@y
    stationarity=np.linalg.norm(stat)/max(np.linalg.norm(ridge*x)+np.linalg.norm(k.T@y),1e-30)
    return {"primal_objective":float(primal),"dual_lower_bound":float(dual),
            "relative_objective_bracket":float(bracket),"stationarity":float(stationarity),
            "dual_ball_violation":float(max(0.,np.linalg.norm(yy,axis=1).max()-lam)),
            "strict_objective_pass":bool(0. <= bracket <= 1e-3)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--rho",type=float,default=.9)
    p.add_argument("--max-blocks",type=int,default=200)
    args=p.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    if not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("Authorized remote GPU required")
    problem,spec=blockmetric_f1(n=16,k=4,rho=args.rho,variant="overlap",seed=args.seed,mu=.01)
    op=spec.dual_blocks[0]["op"]
    # Basis expansion is intentionally limited to this tiny 32x16 reference.
    dense=jax.jit(jax.vmap(op.matvec))(jnp.eye(spec.n,dtype=jnp.float32)).T
    k=np.asarray(dense,np.float64)
    offset=np.asarray(spec.dual_blocks[0]["offset"],np.float64)
    test=np.linspace(-1,1,spec.n).astype(np.float32)
    discrepancy=float(np.linalg.norm(k@test-np.asarray(op.matvec(jnp.asarray(test)),np.float64)))
    start=time.perf_counter()
    comp=problem.compile(tol=1e-3,max_blocks=args.max_blocks,block=50,pdhg_variant="standard")
    compile_s=time.perf_counter()-start
    out={"config":{"host":platform.node(),"devices":[str(d) for d in jax.devices()],
                   "family":"F1-overlap", "n":16,"m":32,"k":4,"rho":args.rho,"seed":args.seed,
                   "primal_ridge":.01,"construction_mu":.1,"M":2.,"safety":.95,"weight":1.,
                   "iteration":"existing dual-first research Loop; float32 GPU",
                   "construction":"CPU float64 tiny explicit-reference callbacks",
                   "public_compile_seconds":compile_s,"operator_expansion_error":discrepancy,
                   "acceptance":"public KKT<=1e-3 AND independent feasible-dual bracket<=1e-3",
                   "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"records":[]}
    for name,structure,budget,stochastic in (("scalar_ref","scalar",1500,False),
                ("diagonal_ref","diagonal",1500,False),("full_ref","full",1500,False),
                ("full_sgd200","full",200,True),("full_sgd1000","full",1000,True)):
        a,b,construction_info=construction(k,structure,budget=budget,seed=args.seed,stochastic=stochastic)
        ah=[sym_function(ai,np.sqrt) for ai in a]
        weighted=np.vstack([half @ rows for half,rows in zip(ah,k.reshape(-1,4,16))])*np.sqrt(b)
        norm=float(np.linalg.norm(weighted,2)); factor=.95/norm
        sigma=np.asarray(a)*factor; tau=b*factor
        if structure=="scalar":
            metrics=[("scalar",jnp.asarray(sigma[:,0,0,None],jnp.float32))]
        elif structure=="diagonal":
            metrics=[("diag",jnp.asarray(np.diagonal(sigma,axis1=1,axis2=2),jnp.float32))]
        else:
            eig,q=np.linalg.eigh(sigma)
            metrics=[("full",jnp.asarray(q,jnp.float32),jnp.asarray(eig,jnp.float32))]
        loop=Loop(spec,jnp.asarray(tau,jnp.float32),metrics,block=50)
        x,ys,xbar=loop.init_state()
        history=[]; t_start=time.perf_counter(); first_kkt=None
        for step in range(1,args.max_blocks+1):
            x,ys,xbar=loop._run_block(x,ys,xbar,loop.T,loop.marrays,loop.offs)
            jax.block_until_ready(x)
            y=loop.returned_pair(x,ys)
            cert=comp.certificate(x,y,spec.data)
            residual=max(float(cert["r_primal"]),float(cert["r_dual"]))
            check=independent(x,y,k,offset,spec.mu,spec.dual_blocks[0]["lam"],4)
            if first_kkt is None and residual<=1e-3:
                first_kkt=step*50
            history.append([step*50,residual,check["relative_objective_bracket"],time.perf_counter()-t_start])
            if residual<=1e-3 and check["strict_objective_pass"]:
                break
            if time.perf_counter()-t_start>45:
                break
        accepted=residual<=1e-3 and check["strict_objective_pass"]
        cold_seconds=time.perf_counter()-t_start
        bench=loop.bench(reps=3)
        row={"arm":name,"construction":construction_info,"A":np.asarray(a).tolist(),"B":b.tolist(),
             "exact_coupling_norm":.95,"metric_normalization_factor":factor,
             "iterations":step*50,"first_public_kkt_iteration":first_kkt,
             "public_residual":residual,"independent":check,"accepted":accepted,
             "cold_solve_plus_checks_seconds":cold_seconds,
             "warm_seconds_per_50_update_block":bench,
             "estimated_warm_update_seconds_excluding_checks":bench["block"]*step,
             "cpu_setup_plus_estimated_warm_update_seconds_excluding_checks":construction_info["seconds_cpu_f64"]+bench["block"]*step,
             "history":history,"x":np.asarray(x).tolist(),"y":np.asarray(y).tolist()}
        out["records"].append(row)
        args.output.write_text(json.dumps(out,indent=2)+"\n")
        print(json.dumps({"arm":name,"construction_objective":construction_info["objective"],
                          "reference_converged":construction_info["reference_converged"],
                          "iterations":step*50,"residual":residual,
                          "bracket":check["relative_objective_bracket"],"accepted":accepted}),flush=True)


if __name__=="__main__":
    main()
