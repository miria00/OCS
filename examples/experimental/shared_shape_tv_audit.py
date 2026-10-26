"""Bounded terminal-zero TV Gram/variance diagnostic on authorized 5090s.

This audits the derivative operator alone, not the full MRI/FFT model. Tiny
float64 analytic-Gram controls run on the same remote host. It makes no
billion-scale, solver-speed or finite stochastic optimality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from ocs.operators.structured import Gradient3D
from ocs.solver.equilibrate_sgd import smfe_equilibration
from mixed_metric_gpu import BlockGroup, initial_state, make_constructor


def layout(L):
    sites=np.arange(L**3).reshape(L,L,L)
    active=np.stack(np.broadcast_arrays(np.arange(L)[:,None,None]<L-1,
                       np.arange(L)[None,:,None]<L-1,np.arange(L)[None,None,:]<L-1),axis=-1)
    neighbors=[]
    for axis in range(3):
        neighbors.append(np.roll(sites,-1,axis=axis).reshape(-1))
    active=active.reshape(-1,3)
    groups=(active*np.array([1,2,4])).sum(axis=1)
    return active,np.stack(neighbors,axis=1),groups


def grams(b,active,neighbors):
    n=len(b)
    out=b[:,None,None]*(active[:,:,None]*active[:,None,:])
    out=out.astype(np.float64)
    for j in range(3):out[:,j,j]+=active[:,j]*b[neighbors[:,j]]
    return out


def column_energy(a,active,neighbors):
    mask=active.astype(np.float64)
    out=np.einsum('ni,nij,nj->n',mask,a,mask)
    for axis in range(3):
        idx=np.flatnonzero(active[:,axis])
        np.add.at(out,neighbors[idx,axis],a[idx,axis,axis])
    return out


def scalar_log_root(energy,target,box=2.,mu=1.):
    lo=np.full_like(energy,-2*box);hi=np.full_like(energy,2*box)
    for _ in range(60):
        mid=(lo+hi)/2
        positive=.5*energy*np.exp(mid)-target/2+mu/4*mid>0
        hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
    return (lo+hi)/2


def update_a(b,active,neighbors,groups,tied):
    g=grams(b,active,neighbors);r=1/np.sqrt(3)
    if tied:
        gg=np.stack([g[groups==i].mean(axis=0) for i in range(8)])
    else:gg=g
    eigen,q=np.linalg.eigh(gg)
    logs=scalar_log_root(np.maximum(eigen,0.),r)
    a=np.einsum('...ij,...j,...kj->...ik',q,np.exp(logs),q)
    return a[groups] if tied else a


def objective(a,b,active,neighbors):
    g=grams(b,active,neighbors);logs=np.log(np.linalg.eigvalsh(a));lb=np.log(b)
    return float(.5*np.einsum('nij,nji->',a,g)-.5/np.sqrt(3)*logs.sum()
        -.5*np.sqrt(3)*lb.sum()+.125*(np.sum(logs**2)+np.sum(lb**2)))


def deterministic_reference(active,neighbors,groups,tied,cap):
    n=len(groups);b=np.ones(n);a=np.broadcast_to(np.eye(3),(n,3,3)).copy();history=[]
    for sweep in range(cap):
        anew=update_a(b,active,neighbors,groups,tied)
        bnew=np.exp(scalar_log_root(column_energy(anew,active,neighbors),np.sqrt(3)))
        change=max(float(np.max(np.abs(np.log(bnew/b)))),
                   float(np.max(np.linalg.norm(anew-a,axis=(1,2))/np.maximum(1.,np.linalg.norm(a,axis=(1,2))))))
        a,b=anew,bnew
        if sweep%10==0 or change<1e-9 or sweep==cap-1:
            history.append({'sweep':sweep+1,'objective':objective(a,b,active,neighbors),'change':change})
        if change<1e-9:break
    astar=update_a(b,active,neighbors,groups,tied)
    residual=float(np.max(np.linalg.norm(astar-a,axis=(1,2))/np.maximum(1.,np.linalg.norm(a,axis=(1,2)))))
    return a,b,{'sweeps':sweep+1,'cap':cap,'history':history,'conditional_fixed_point_residual':residual,
                'reached_requested_change':bool(change<1e-9),'objective':objective(a,b,active,neighbors),
                'method':'deterministic exact-Gram conditional minimization; bounded diagnostic, no theorem claim'}


def group_report(b,active,neighbors,groups,probe_samples):
    exact=grams(b,active,neighbors);reports=[]
    for kind in range(8):
        ids=np.flatnonzero(groups==kind);gg=exact[ids];avg=gg.mean(axis=0)
        residual=np.linalg.norm(gg-avg,axis=(1,2));norm=np.linalg.norm(avg)
        samples=probe_samples[:,ids,:]
        traces=np.mean(np.sum(samples*samples,axis=2),axis=1)
        pooled=np.einsum('qni,qnj->ij',samples,samples)/(len(samples)*len(ids))
        # Every active oriented finite-difference edge appears once. For a
        # global Rademacher probe, trace variance is 4 sum_edges b_p*b_q/N².
        degree=np.zeros(len(b));edge_products=0.;edges=0
        for axis in range(3):
            ii=ids[active[ids,axis]];jj=neighbors[ii,axis]
            np.add.at(degree,ii,1);np.add.at(degree,jj,1)
            edge_products+=np.sum(b[ii]*b[jj]);edges+=len(ii)
        variance=4*edge_products/len(ids)**2
        trace=np.sum(b*degree)/len(ids)
        sqnorm=np.sum((b*degree/len(ids))**2)+2*edge_products/len(ids)**2
        rank=trace**2/sqnorm if sqnorm else 0.
        reports.append({'group':kind,'active_axes':[bool(kind&(1<<i)) for i in range(3)],
            'count':len(ids),'mean_gram':avg.tolist(),'max_relative_frobenius_variation':float(residual.max()/norm) if norm else 0.,
            'rms_relative_frobenius_variation':float(np.sqrt(np.mean(residual**2))/norm) if norm else 0.,
            'B_min':float(b[ids].min()),'B_max':float(b[ids].max()),
            'pooled_gram_estimation_relative_error':float(np.linalg.norm(pooled-avg)/norm) if norm else 0.,
            'pooled_trace_variance_empirical':float(np.var(traces,ddof=1)),
            'pooled_trace_variance_exact_global_probe':float(variance),
            'effective_rank_of_probe_quadratic_form':float(rank),'unique_edges':edges,
            'exact_trace_mean':float(trace),'empirical_trace_mean':float(traces.mean())})
    return reports,exact


def main():
    p=argparse.ArgumentParser();p.add_argument('--side',type=int,choices=[8,16],required=True)
    p.add_argument('--output',required=True);p.add_argument('--audit-probes',type=int,default=128)
    p.add_argument('--deterministic-sweeps',type=int,default=160)
    args=p.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='gpu' and '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64',False)
    started=time.perf_counter();L=args.side;n=L**3;active,neighbors,groups=layout(L);op=Gradient3D(L,L,L)
    report={'side':L,'voxels':n,'host':subprocess.check_output(['hostname'],text=True).strip(),
        'device':str(jax.devices()[0]),'jax':jax.__version__,'scope':'terminal-zero Gradient3D only; no FFT/MRI data blocks',
        'config':{'updates':80,'probes':2,'seed':0,'mu':1.,'box':2.,'shift':20,
                  'audit_probes':args.audit_probes,'audit_seed':7109,'deterministic_sweeps':args.deterministic_sweeps},
        'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),Path(__file__).with_name('mixed_metric_gpu.py')]},'arms':[]}
    saved={'active':active,'neighbors':neighbors,'groups':groups}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        out.write_text(json.dumps(report,indent=2));np.savez(out.with_suffix('.npz'),**saved)
    def record(name,b,a=None,extra=None):
        # Independent audit signs, reused across arms for a paired diagnostic.
        # They are not the constructor probes and do not tune any metric.
        key=jax.random.PRNGKey(7109)
        z=jax.random.rademacher(key,(args.audit_probes,n),dtype=jnp.float32)
        samples=jax.jit(jax.vmap(lambda v:op.matvec(jnp.sqrt(jnp.asarray(b,jnp.float32))*v)))(z)
        samples=np.asarray(samples,np.float64).reshape(args.audit_probes,n,3)
        summary,g=group_report(b,active,neighbors,groups,samples)
        arm={'name':name,'groups':summary,'B_range':[float(b.min()),float(b.max())],**(extra or {})}
        saved[name+'_B']=b;saved[name+'_weighted_grams']=g
        if a is not None:saved[name+'_A']=a
        report['arms'].append(arm);save()
        print(json.dumps({'arm':name,'interior_count':summary[7]['count'],
              'interior_gram_max_relative_variation':summary[7]['max_relative_frobenius_variation']}),flush=True)
    record('identity',np.ones(n))
    rows=(BlockGroup('full',3,n),);columns=(BlockGroup('diagonal',n),)
    run=make_constructor(op.matvec,op.rmatvec,rows,columns,iterations=80,probes=2,mu=1.,box=2.,shift=20,batch_size=4096)
    answer=run(jax.random.PRNGKey(0),initial_state(rows),initial_state(columns));jax.block_until_ready(answer)
    assert bool(answer['finite'])
    record('full_stochastic_80',np.exp(np.asarray(answer['B'][0],np.float64)).reshape(-1),
           np.asarray(answer['A'][0],np.float64),{'row_class':'independent full3 blocks','column_class':'free diagonal',
            'average':'pre-update weighted geodesic','operator_calls':320})
    tau,sigma,norm=smfe_equilibration(op.matvec,op.rmatvec,n,3*n,[(0,3*n)],iters=80,probes=2,gamma=1.,M=2.,seed=0)
    jax.block_until_ready((tau,sigma,norm));scale=.9/float(norm)
    record('configured_smfe_80',np.asarray(tau,np.float64)/scale,extra={
        'row_class':'one term-wide scalar','column_class':'free diagonal','average':'shipped post-update log mean',
        'operator_calls':320,'extra_norm_operator_calls':30,'undo_global_scale':scale,
        'warning':'same seed number but different sampling trajectory/schedule and row class from FULL'})
    if L==8:
        for tied in (False,True):
            a,b,details=deterministic_reference(active,neighbors,groups,tied,args.deterministic_sweeps)
            record('boundary_tied_deterministic' if tied else 'full_deterministic',b,a,details)
    report['memory']=jax.devices()[0].memory_stats()
    assert report['memory']['peak_bytes_in_use']<=8*2**30
    report['complete']=True;report['limits']=[
        'Within-group scaled-Gram variation is an exact diagnostic at each returned B; its stochastic B may contain sampling noise.',
        'L8 deterministic controls isolate boundary effects at a reported conditional fixed point; finite sweep endpoints are not certified optima.',
        'A shared boundary class may be useful despite restriction; no zero-loss claim without exact symmetry.',
        'This is a derivative-operator diagnostic, not a solver runtime or billion-voxel result.']
    save()


if __name__=='__main__':main()
