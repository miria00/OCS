"""Bounded Fercoq rate-monitoring policy ablation on ordinary primal-first PDHG.

This is our JAX port of Algorithms 3--5 of Fercoq (2024), not the author's
solver and not a competitor timing. The primal-first ordering has metric
V=[[T^-1,-K.T],[-K,S^-1]], and Goldstein residuals use the corresponding minus
signs. OCS convention: T=T0/w, S=S0*w. The author implementation resolves the
printed Algorithm 5 gear typo: reverse when the new spectral radius is WORSE.
Adaptation is bounded and then frozen; no variable-metric theorem is claimed.
All numerical execution belongs on the designated remote RTX 5090.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np


def dump(path, value):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str))


def source_digest():
    import ocs
    root = Path(ocs.__file__).parent
    h = hashlib.sha256()
    files = {}
    for p in sorted(root.rglob("*.py")):
        s = hashlib.sha256(p.read_bytes()).hexdigest()
        files[str(p.relative_to(root))] = s
        h.update(f"{p.relative_to(root)} {s}\n".encode())
    return dict(path=str(root), sha256=h.hexdigest(), files=files)


def route(prob, data):
    import jax.numpy as jnp
    import ocs
    primal, dual = None, []
    for t in prob.terms:
        if primal is None and isinstance(t.op, ocs.Identity) and t.offset_fn is None:
            primal = t.block
        else:
            dual.append(t)
    ends = np.cumsum([0]+[t.op.shape[0] for t in dual])
    slices = list(zip(ends[:-1], ends[1:]))
    def mv(x):
        return jnp.concatenate([t.op.matvec(x) for t in dual])
    def rm(y):
        return sum(t.op.rmatvec(y[s:e]) for t, (s,e) in zip(dual,slices))
    offsets = [jnp.zeros(t.op.shape[0], jnp.float32) if t.offset_fn is None
               else jnp.asarray(t.offset_fn(data),jnp.float32) for t in dual]
    c = jnp.zeros(prob.n,jnp.float32) if prob.cost_fn is None else jnp.asarray(prob.cost_fn(data))
    def pp(v,tau):
        z = v-tau*c
        return z if primal is None else primal.prox(z,tau)
    from ocs.prox.groups import build_groups, grouped_moreau_dual
    apply_dual=grouped_moreau_dual(build_groups([t.block for t in dual],slices),int(ends[-1]))
    offset=jnp.concatenate(offsets)
    def pd(v,sigma):
        return apply_dual(v+sigma*offset,sigma)
    return primal, dual, slices, mv, rm, pp, pd


def make_model(args):
    import jax
    import jax.numpy as jnp
    import ocs
    import cvxpy as cp
    rng=np.random.default_rng(args.seed)
    if args.model == "quadratic":
        r=float(args.curvature)
        dtype=jnp.float64 if args.quadratic_x64 else jnp.float32
        def analytic_radius(w):
            t=r/w+w/r
            return (t+1.5+math.sqrt(t*t-3.75))/(2*(t+2))
        grid=np.geomspace(r/1000,r*1000,61)
        max_radius_error=0.
        for w in grid:
            tau=.5/w;sigma=.5*w;p=1/(1+2*r*tau);q=1/(1+2*sigma/r)
            J=np.array([[p,-p*tau],[q*sigma*(2*p-1),q*(1-2*sigma*p*tau)]])
            max_radius_error=max(max_radius_error,abs(max(abs(np.linalg.eigvals(J)))-analytic_radius(w)))
        mv=rm=lambda x:x
        pp=lambda v,t:(v+t*(2*r+1))/(1+2*r*t)
        pd=lambda v,s:(v+s*(2/r-1))/(1+2*s/r)
        if args.quadratic_centered:
            pp=lambda v,t:v/(1+2*r*t)
            pd=lambda v,s:v/(1+2*s/r)
        def check(x,y):
            xx,yy=float(x[0]),float(y[0])
            if args.quadratic_centered:xx+=1;yy+=1
            err=max(abs(xx-1),abs(yy-1))
            return {"objective":1.25*r*(xx-1)**2,"reference_objective":0.,
                    "primal_dual_distance_inf":err,"independent_pass":err<=1e-3,
                    "stationarity_abs":abs(2*r*(xx-1)-1+yy),
                    "dual_graph_abs":abs(xx-1-2*(yy-1)/r)}
        def cert(x,y):
            target=0 if args.quadratic_centered else 1
            return {"r_primal":float(jnp.abs(x[0]-target)),"r_dual":float(jnp.abs(y[0]-target))}
        return dict(n=1,m=1,mv=mv,rm=rm,pp=pp,pd=pd,tau=jnp.array([.5],dtype),
                    sigma=jnp.array([.5],dtype),certificate=cert,check=check,
                    initial_coordinate=-1 if args.quadratic_centered else 0,
                    meta={"r":r,"known_saddle":[1,1],"exact_best_weight":r,
                          "centered_error_arithmetic":args.quadratic_centered,
                          "radius_at_start":analytic_radius(args.weight),"radius_at_oracle":analytic_radius(r),
                          "analytic_radius_vs_eigenvalues_max_error_61_weights":max_radius_error,
                          "acceptance":"distance to exact saddle <= requested tolerance",
                          "not_public_certificate":True},inputs={"r":np.array(r)})
    x=ocs.Variable(args.size if args.model=="lasso" else 1)
    if args.model=="ot":
        n=args.size
        otsize=n
        C=rng.uniform(0,1,(n,n)).astype(np.float32)
        def dyadic(z):
            z=z/z.sum()*2**20;k=np.floor(z).astype(np.int64)
            k[np.argsort(-(z-k))[:2**20-int(k.sum())]]+=1
            return (k/2**20).astype(np.float32)
        p=dyadic(rng.uniform(.2,1,n));q=dyadic(rng.uniform(.2,1,n))
        eps=float(np.float32(.1));x=ocs.Variable(n*n)
        from ocs.ir import Term
        rhs=ocs.Parameter("m");cost=ocs.Parameter("c")
        prob=ocs.minimize([Term(ocs.Identity(n*n),ocs.NonNeg().with_linear_quadratic(e=eps)),
                          ocs.equality(ocs.MarginalOperator(n,n)@x-rhs)],cost=cost)
        data={"m":jnp.asarray(np.r_[p,q]),"c":jnp.asarray(C.reshape(-1))}
        z=cp.Variable((n,n),nonneg=True)
        ref=cp.Problem(cp.Minimize(cp.sum(cp.multiply(C.astype(float),z))+.5*eps*cp.sum_squares(z)),
                       [cp.sum(z,axis=1)==p.astype(float),cp.sum(z,axis=0)==q.astype(float)])
        rt=time.perf_counter();ref.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        rs=time.perf_counter()-rt; refval=float(ref.value)
        def check(xv,yv):
            X=np.asarray(xv,float).reshape(otsize,otsize);Y=np.asarray(yv,float)
            feas=float(np.linalg.norm(np.r_[X.sum(1),X.sum(0)]-np.r_[p,q])/np.linalg.norm(np.r_[p,q]))
            val=float(np.sum(C.astype(float)*X)+.5*eps*np.sum(X*X));err=abs(val-refval)/max(abs(refval),1e-30)
            return {"objective":val,"reference_objective":refval,"objective_relative_error":err,
                    "marginal_relative":feas,"min_x":float(X.min()),
                    "independent_pass":bool(err<=1e-3 and feas<=1e-3 and X.min()>=-1e-8)}
        inputs=dict(C=C,p=p,q=q);meta=dict(reference_status=ref.status,reference_seconds=rs,epsilon=eps)
    elif args.model=="mcf":
        V=20;E=args.size;K=8;n=E*K
        N=np.zeros((V,E),np.float32)
        # Exact dyadic feasible input avoids a floating-demand consistency confound.
        for e in range(E):
            a,b=rng.choice(V,2,replace=False);N[a,e]=1;N[b,e]=-1
        known=(rng.integers(0,9,(E,K))/16).astype(np.float32);d=N@known
        slopes=np.array([.5,1,2,4],np.float32);inter=np.array([0,-.25,-1.25,-4.25],np.float32)
        mu=float(np.float32(1e-3));x=ocs.Variable(n);rhs=ocs.Parameter("d")
        prob=ocs.minimize([ocs.nonneg(x),ocs.sum_squares(ocs.Diagonal(np.full(n,np.sqrt(mu),np.float32))@x),
                          ocs.equality(ocs.MultiCommodityFlow(N,K)@x-rhs),
                          ocs.pwl(ocs.CommoditySum(E,K)@x,slopes,inter)])
        # Actual model curvature is the float32 diagonal squared.
        mu=float(np.float32(np.sqrt(mu)))**2
        data={"d":jnp.asarray(d.reshape(-1))}
        z=cp.Variable((E,K),nonneg=True);tot=cp.sum(z,axis=1)
        phi=cp.maximum(*[float(s)*tot+float(b) for s,b in zip(slopes,inter)])
        ref=cp.Problem(cp.Minimize(cp.sum(phi)+.5*mu*cp.sum_squares(z)),[N.astype(float)@z==d.astype(float)])
        rt=time.perf_counter();ref.solve(solver="CLARABEL",tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
        rs=time.perf_counter()-rt;refval=float(ref.value)
        def check(xv,yv):
            X=np.asarray(xv,float).reshape(E,K);tot=X.sum(1)
            val=float(np.max(tot[:,None]*slopes+inter,axis=1).sum()+.5*mu*np.sum(X*X))
            feas=float(np.linalg.norm(N.astype(float)@X-d)/np.linalg.norm(d));err=abs(val-refval)/max(abs(refval),1e-30)
            return {"objective":val,"reference_objective":refval,"objective_relative_error":err,
                    "conservation_relative":feas,"min_x":float(X.min()),
                    "independent_pass":bool(err<=1e-3 and feas<=1e-3 and X.min()>=-1e-8)}
        inputs=dict(N=N,d=d,known=known,slopes=slopes,inter=inter);meta=dict(reference_status=ref.status,reference_seconds=rs,mu=mu,V=V,E=E,K=K)
    elif args.model=="lasso":
        n=args.size;m=n//2;A=(rng.normal(size=(m,n))/np.sqrt(m)).astype(np.float32)
        b=rng.normal(size=m).astype(np.float32);lam=float(np.float32(.1*np.max(abs(A.T@b))))
        x=ocs.Variable(n);rhs=ocs.Parameter("b")
        prob=ocs.minimize([ocs.l1(x,lam),ocs.least_squares(ocs.Dense(A)@x-rhs)])
        data={"b":jnp.asarray(b)};z=cp.Variable(n)
        ref=cp.Problem(cp.Minimize(.5*cp.sum_squares(A.astype(float)@z-b)+lam*cp.norm1(z)))
        rt=time.perf_counter();ref.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        rs=time.perf_counter()-rt;refval=float(ref.value)
        def check(xv,yv):
            X=np.asarray(xv,float);val=float(.5*np.sum((A.astype(float)@X-b)**2)+lam*np.sum(abs(X)))
            err=abs(val-refval)/max(abs(refval),1e-30)
            return {"objective":val,"reference_objective":refval,"objective_relative_error":err,"independent_pass":bool(err<=1e-3)}
        inputs=dict(A=A,b=b);meta=dict(reference_status=ref.status,reference_seconds=rs,lam=lam)
    elif args.model=="feature":
        Fdim=args.size;M=Fdim//2;T=8;LAM=float(np.float32(.5))
        A=rng.normal(size=(M,Fdim)).astype(np.float32)
        truth=np.zeros((Fdim,T),np.float32)
        active=rng.choice(Fdim,12,replace=False);truth[active]=rng.normal(size=(12,T))
        B=(A@truth+.01*rng.normal(size=(M,T))).astype(np.float32)
        x=ocs.Variable(Fdim*T);bp=ocs.Parameter("B")
        prob=ocs.minimize([ocs.group_l2(x,lam=LAM,group_size=T),
                          ocs.least_squares(ocs.LeftMatMul(A,T)@x-bp)])
        data={"B":jnp.asarray(B.reshape(-1))}
        Z=cp.Variable((Fdim,T));ref=cp.Problem(cp.Minimize(.5*cp.sum_squares(A.astype(float)@Z-B)+LAM*cp.sum(cp.norm(Z,2,axis=1))))
        rt=time.perf_counter();ref.solve(solver="CLARABEL",tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
        rs=time.perf_counter()-rt;refval=float(ref.value)
        def check(xv,yv):
            X=np.asarray(xv,float).reshape(Fdim,T)
            val=float(.5*np.sum((A.astype(float)@X-B)**2)+LAM*np.linalg.norm(X,axis=1).sum())
            err=abs(val-refval)/abs(refval)
            lower_bounds=[]
            for dual in [np.asarray(yv,float).reshape(M,T),A.astype(float)@X-B]:
                maximum=np.linalg.norm(A.astype(float).T@dual,axis=1).max()
                dual=dual*(min(1.,LAM/max(maximum,np.finfo(float).tiny))*(1-1e-12))
                if np.linalg.norm(A.astype(float).T@dual,axis=1).max()<=LAM:
                    lower_bounds.append(float(-.5*np.sum(dual*dual)-np.sum(B*dual)))
            lower=max(lower_bounds);gap=(val-lower)/max(abs(val),abs(lower))
            return {"objective":val,"reference_objective":refval,"objective_relative_error":err,
                    "feasible_dual_lower_bound":lower,"certified_objective_relative_gap":gap,
                    "independent_pass":bool(err<=1e-3 and -1e-12<=gap<=1e-3)}
        inputs=dict(A=A,B=B,truth=truth,active=active)
        meta=dict(reference_status=ref.status,reference_seconds=rs,lam=LAM,samples=M,features=Fdim,tasks=T)
    elif args.model=="imaging":
        L=args.size;nimg=L**3;lam=float(np.float32(.02))
        volume=np.zeros((L,L,L),np.float32)
        volume[8:28,10:34,12:30]=1;volume[20:36,6:20,18:38]+=.5
        phantom=volume.reshape(-1)
        ids=np.sort(np.r_[0,rng.choice(np.arange(1,nimg),int(.4*nimg)-1,replace=False)])
        mf=len(ids);Fop=ocs.FFTMask(ids,nimg);Gop=ocs.Gradient3D(L,L,L)
        def f64(xv):
            z=np.fft.fft(xv,norm="ortho")[ids]
            return np.r_[z.real,z.imag]
        def ft64(v):
            z=np.zeros(nimg,complex);z[ids]=v[:mf]+1j*v[mf:]
            return np.fft.ifft(z,norm="ortho").real
        def g64(xv):
            v=np.asarray(xv,float).reshape(L,L,L);z=np.zeros((L,L,L,3),float)
            z[:-1,:,:,0]=v[1:]-v[:-1];z[:,:-1,:,1]=v[:,1:]-v[:,:-1]
            z[:,:,:-1,2]=v[:,:,1:]-v[:,:,:-1]
            return z.reshape(-1,3)
        def gt64(q):
            q=q.reshape(L,L,L,3);z=np.zeros((L,L,L),float)
            z[1:]+=q[:-1,:,:,0];z[:-1]-=q[:-1,:,:,0]
            z[:,1:]+=q[:,:-1,:,1];z[:,:-1]-=q[:,:-1,:,1]
            z[:,:,1:]+=q[:,:,:-1,2];z[:,:,:-1]-=q[:,:,:-1,2]
            return z.reshape(-1)
        b=f64(phantom).astype(np.float32);x=ocs.Variable(nimg);bp=ocs.Parameter("b")
        prob=ocs.minimize([ocs.nonneg(x),ocs.least_squares(Fop@x-bp),ocs.group_l2(Gop@x,lam=lam,group_size=3)])
        data={"b":jnp.asarray(b)}
        def check(xv,yv):
            X=np.asarray(xv,float);Y=np.asarray(yv,float)
            q=Y[2*mf:].reshape(-1,3).copy();norms=np.linalg.norm(q,axis=1)
            # Stay strictly inside each TV dual ball despite roundoff.
            q*=np.minimum(1,np.nextafter(lam,0)/np.maximum(norms,np.finfo(float).tiny))[:,None]
            s=Y[:2*mf].copy();station=ft64(s)+gt64(q)
            shift=max(0.,-float(station.min()))+1e-12
            s[0]+=shift*np.sqrt(nimg)  # DC is the first observed frequency.
            repaired=ft64(s)+gt64(q)
            lower=float(-.5*np.dot(s,s)-np.dot(b.astype(float),s))
            val=float(.5*np.sum((f64(X)-b)**2)+lam*np.linalg.norm(g64(X),axis=1).sum())
            gap=(val-lower)/max(abs(val),abs(lower),np.finfo(float).tiny)
            feasible=bool(repaired.min()>=-1e-13 and np.linalg.norm(q,axis=1).max()<=lam*(1+1e-14) and X.min()>=0)
            return {"objective":val,"reference_objective":lower,"reference_kind":"feasible dual lower bound, not an optimum",
                    "objective_relative_error":gap,"certified_objective_relative_gap":gap,
                    "dual_stationarity_min_after_repair":float(repaired.min()),"dc_repair_shift":shift,
                    "tv_dual_max_norm":float(np.linalg.norm(q,axis=1).max()),"min_x":float(X.min()),
                    "independent_pass":bool(feasible and -1e-10<=gap<=1e-3)}
        inputs=dict(phantom=phantom,idx=ids,b=b)
        meta=dict(reference_status="independent float64 NumPy feasible-dual bound",reference_seconds=0.,
                  lam=lam,L=L,sampling_ratio=mf/nimg,dc_explicitly_included=True,
                  independent_bound="project TV dual groups; add observed DC to make F.T s+G.T q nonnegative; D=-.5||s||²-b.T s")
    else: raise ValueError(args.model)
    primal,dual,slices,mv,rm,pp,pd=route(prob,data)
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    n=prob.n;m=sum(t.op.shape[0] for t in dual)
    spans=[sl for t,sl in zip(dual,slices) if t.block.scaling_class=="scalar"]
    tau,sigma,norm,_=smfe_equilibration(mv,rm,n,m,spans,diagnostics=True)
    tau=jnp.broadcast_to(tau,(n,));sigma=jnp.broadcast_to(sigma,(m,))
    if primal is not None and primal.scaling_class=="scalar":
        tau=jnp.full((n,),jnp.min(tau),jnp.float32)
    comp=prob.compile(equilibrate=("given",tau,sigma,float(norm)),pdhg_variant="standard",block=100,max_blocks=2000,tol=args.tol)
    meta.update(scaled_norm=float(norm),public_compiled_meta={k:str(v) for k,v in comp.meta.items()})
    return dict(n=n,m=m,mv=mv,rm=rm,pp=pp,pd=pd,tau=tau,sigma=sigma,
                certificate=lambda x,y:comp.certificate(x,y,data),check=check,meta=meta,inputs=inputs)


def build_runner(model,args):
    import jax
    import jax.numpy as jnp
    from jax import lax
    mv,rm,pp,pd=[model[k] for k in ("mv","rm","pp","pd")]
    tau0,sigma0=model["tau"],model["sigma"]
    dtype=tau0.dtype.type
    histlen=args.adapt_iters+3
    # State: x,y,w,alpha,gear,last accepted rho,window anchor,last norm,
    # last two ratios,window count,iteration,extremum min/max indices,rates,changes.
    initial=(jnp.full(model["n"],model.get("initial_coordinate",0),dtype),
             jnp.full(model["m"],model.get("initial_coordinate",0),dtype),
             dtype(args.weight),dtype(.5 if args.arm in ("fercoq","goldstein") else 0),
             jnp.int32(1),dtype(1),dtype(0),dtype(0),
             dtype(0),dtype(0),jnp.int32(0),jnp.int32(0),
             jnp.int32(-1),jnp.int32(-1),jnp.zeros(histlen,dtype),jnp.int32(0))
    def step(st,_):
        x,y,w,alpha,gear,rhoref,anchor,prev,rprev,rprev2,count,it,imin,imax,rates,changes=st
        tau=tau0/w;sigma=sigma0*w
        xn=pp(x-tau*rm(y),tau)
        yn=pd(y+sigma*mv(2*xn-x),sigma)
        dx=xn-x;dy=yn-y
        if args.arm=="fixed_fast":
            norm=p1=d1=dtype(0)
        else:
            kdx=mv(dx)
            norm2=jnp.maximum(0,jnp.sum(dx*dx/tau)+jnp.sum(dy*dy/sigma)-2*jnp.vdot(kdx,dy))
            norm=jnp.sqrt(norm2)
            p1=jnp.sum(jnp.abs(dx/tau-rm(dy)));d1=jnp.sum(jnp.abs(dy/sigma-kdx))
        can=(args.arm not in ("fixed","fixed_fast")) & (it<args.adapt_iters)
        # This diagnostic is identical in all arms, so the kernel ablation
        # compares the policy and not whether monitoring was compiled away.
        balance=jnp.where(p1>1.5*d1,-1,jnp.where(d1>1.5*p1,1,0))
        gold=can & (alpha>1e-4) & (balance!=0)
        wg=w*jnp.power(1-alpha,-balance)
        ratio=jnp.where((count>0)&(prev>0),norm/prev,jnp.nan)
        rateidx=jnp.minimum(count,histlen-1);rates=rates.at[rateidx].set(ratio)
        minimum=(count>2)&(rprev<rprev2)&(rprev<ratio)
        maximum=(count>2)&(rprev>rprev2)&(rprev>ratio)
        imin=jnp.where((imin<0)&minimum,count-1,imin)
        imax=jnp.where((imax<0)&maximum,count-1,imax)
        den=1-rprev
        stable=(count>3)&(den>0)&(jnp.abs((1-ratio)/den-1)<=1e-3)&(jnp.abs(ratio-2*rprev+rprev2)/(den*den)<=1e-5)
        cycle=(imin>=0)&(imax>=0)
        midpoint=jnp.minimum((jnp.maximum(imin,0)+jnp.maximum(imax,0)+1)//2,histlen-1)
        rho=jnp.where(stable,rprev,rates[midpoint])
        decreased=(norm<=.6*anchor)&(norm>0)
        ratechange=can & (args.arm!="goldstein") & ~gold & decreased & (stable|cycle)&jnp.isfinite(rho)&(rho>0)&(rho<1)
        gearnew=jnp.where(ratechange&(rho>rhoref),-gear,gear)
        wr=w*jnp.power(dtype(1.5),gearnew)
        wn=jnp.clip(jnp.where(gold,wg,jnp.where(ratechange,wr,w)),1e-6,1e6)
        changed=can & (wn!=w)
        # Each window starts with its first norm in that fixed metric. A step
        # change invalidates within-window ratios; no cross-metric ratio used.
        anew=jnp.where(changed,0,jnp.where(count==0,norm,anchor))
        countnew=jnp.where(changed,0,jnp.minimum(count+1,histlen-1))
        out=(xn,yn,wn,jnp.where(gold,alpha*.95,alpha),gearnew,jnp.where(ratechange,rho,rhoref),
             anew,jnp.where(changed,0,norm),jnp.where(changed,0,ratio),jnp.where(changed,0,rprev),
             countnew,it+1,jnp.where(changed,-1,imin),jnp.where(changed,-1,imax),rates,changes+changed)
        return out,jnp.stack([it+1,wn,norm,jnp.where(ratechange,rho,0),gold,ratechange,p1,d1])
    return initial,jax.jit(lambda state:lax.scan(step,state,None,length=args.block))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["quadratic","mcf","ot","lasso","feature","imaging"],required=True)
    ap.add_argument("--size",type=int,default=64);ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--curvature",type=float,default=1);ap.add_argument("--arm",choices=["fixed","fixed_fast","fercoq","rate_only","goldstein"],default="fercoq")
    ap.add_argument("--weight",type=float,default=1);ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--max-iters",type=int,default=30000);ap.add_argument("--adapt-iters",type=int,default=4000)
    ap.add_argument("--block",type=int,default=100);ap.add_argument("--seconds",type=float,default=200)
    ap.add_argument("--quadratic-x64",action="store_true")
    ap.add_argument("--quadratic-centered",action="store_true")
    ap.add_argument("--allowed-host",choices=["Mazeppa","KEVIN-A51"],default="Mazeppa")
    ap.add_argument("--repeat-solves",type=int,default=0)
    ap.add_argument("--out",required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();import jax,ocs
    if args.quadratic_x64:
        if args.model!="quadratic":raise ValueError("x64 is a separate quadratic diagnostic only")
        jax.config.update("jax_enable_x64",True)
    if jax.devices()[0].device_kind!="NVIDIA GeForce RTX 5090" or os.uname().nodename!=args.allowed_host:
        raise RuntimeError("Numerical runs restricted to the explicitly selected host's RTX 5090")
    result={"args":vars(args),"stage":"setup","source":source_digest(),
            "pilot_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime":{"host":os.uname().nodename,"device":jax.devices()[0].device_kind,"jax":jax.__version__,"x64":jax.config.x64_enabled},
            "scope":"OCS policy ablation, our port; not author-code performance comparison",
            "paper":"https://arxiv.org/html/2403.19202v1","trajectory":[]}
    dump(out/"result.json",result)
    model=make_model(args);result["model"]=model["meta"]
    np.savez_compressed(out/"inputs.npz",**model["inputs"])
    initial,runblock=build_runner(model,args)
    setup=time.perf_counter()-start;tc=time.perf_counter()
    warm,hist=runblock(initial);jax.block_until_ready(warm[:2]);compilefirst=time.perf_counter()-tc
    result.update(setup_seconds=setup,compile_and_first_probe_seconds=compilefirst,stage="running")
    st=initial;loop=0.;checktime=0.;trace=[];ts=time.perf_counter();status="iteration_budget_exhausted"
    for block in range(math.ceil(args.max_iters/args.block)):
        tb=time.perf_counter();st,h=runblock(st);jax.block_until_ready(st[:2]);loop+=time.perf_counter()-tb
        h=np.asarray(h);trace.extend(h.tolist())
        tc=time.perf_counter();cert=model["certificate"](st[0],st[1]);checktime+=time.perf_counter()-tc
        cert={k:float(v) for k,v in cert.items()};r=max(cert["r_primal"],cert["r_dual"])
        row=dict(iteration=int(st[11]),weight=float(st[2]),residual=r,loop_seconds=loop,elapsed_seconds=time.perf_counter()-ts,
                 changes=int(st[15]));result["trajectory"].append(row)
        if block%10==0:print(json.dumps(row),flush=True);dump(out/"result.json",result)
        if not np.isfinite(r):status="numerical_failure";break
        if r<=args.tol:status="numerical_tolerance_met";break
        if time.perf_counter()-ts>=args.seconds:status="time_budget_exhausted";break
    xx,yy=np.asarray(st[0]),np.asarray(st[1]);ind=model["check"](xx,yy)
    result.update(stage="completed",status=status,iterations=int(st[11]),weight_final=float(st[2]),
                  residual=r,certificate=cert,independent=ind,accepted=bool(status=="numerical_tolerance_met" and ind["independent_pass"]),
                  adaptation_changes=int(st[15]),warm_loop_seconds=loop,certificate_seconds=checktime,
                  run_wall_seconds=time.perf_counter()-ts,total_seconds=time.perf_counter()-start,
                  peak_device_bytes=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use"),
                  source_after=source_digest())
    np.savez_compressed(out/"endpoint.npz",x=xx,y=yy)
    np.savez_compressed(out/"iteration_trace.npz",values=np.asarray(trace,np.float64),
                        columns=np.array(["iteration","weight_after","update_Vnorm","accepted_rho","goldstein_change","rate_change","p_l1","d_l1"]))
    repeats=[]
    for repeat in range(args.repeat_solves):
        replay=initial;replay_trace=[];replay_loop=0.;replay_checks=0.
        tr=time.perf_counter();replay_status="iteration_budget_exhausted"
        for block in range(math.ceil(args.max_iters/args.block)):
            tb=time.perf_counter();replay,h=runblock(replay);jax.block_until_ready(replay[:2])
            replay_loop+=time.perf_counter()-tb
            replay_trace.extend(np.asarray(h).tolist())
            tc=time.perf_counter();rc=model["certificate"](replay[0],replay[1])
            rr=max(float(rc["r_primal"]),float(rc["r_dual"]))
            replay_checks+=time.perf_counter()-tc
            if not np.isfinite(rr):replay_status="numerical_failure";break
            if rr<=args.tol:replay_status="numerical_tolerance_met";break
            if time.perf_counter()-tr>=args.seconds:replay_status="time_budget_exhausted";break
        quality=model["check"](np.asarray(replay[0]),np.asarray(replay[1]))
        repeats.append(dict(repeat=repeat,status=replay_status,iterations=int(replay[11]),residual=rr,
            accepted=bool(replay_status=="numerical_tolerance_met" and quality["independent_pass"]),
            warm_loop_seconds=replay_loop,certificate_seconds=replay_checks,
            inclusive_seconds=time.perf_counter()-tr,independent=quality,
            bit_identical_endpoint=bool(np.array_equal(np.asarray(replay[0]),xx) and np.array_equal(np.asarray(replay[1]),yy))))
    result["same_instance_replays"]=repeats
    result["replay_scope"]="Same data and compiled executable; initial point and adaptation state reset each solve. Not varied-data family validation."
    result["experiment_including_replays_seconds"]=time.perf_counter()-start
    dump(out/"result.json",result);print(json.dumps({k:result[k] for k in ["status","iterations","residual","weight_final","accepted","total_seconds"]}),flush=True)


if __name__=="__main__":main()
