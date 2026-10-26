"""Independent tiny float64 references for the experimental GPU constructor."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp

from mixed_metric_gpu import (BlockGroup, initial_state, make_constructor,
                              update_from_samples, average_side, apply_power)


def symf(a, fn):
    values, vectors = np.linalg.eigh((a+a.T)/2)
    return (vectors*fn(values))@vectors.T


def reference_apply(groups, states, vector, power):
    result=[]; start=0
    for group,value in zip(groups,states):
        v=vector[start:start+group.dimension].reshape(group.count,group.size)
        if group.kind=='full':
            z=np.stack([symf(a,lambda d:d**power)@x for a,x in zip(value,v)])
        elif group.kind=='scalar':z=np.exp(power*value)[:,None]*v
        else:z=np.exp(power*value)*v
        result.append(z.reshape(-1));start+=group.dimension
    return np.concatenate(result)


def reference_step(groups,states,samples,target,mu,box,eta):
    result=[];start=0
    for group,value in zip(groups,states):
        sample=samples[:,start:start+group.dimension].reshape(-1,group.count,group.size)
        if group.kind=='full':
            result_group=[]
            for i,a in enumerate(value):
                gram=sample[:,i,:].T@sample[:,i,:]/len(sample)
                grad=.5*a@gram@a-target/2*a+mu/4*a@symf(a,np.log)
                ah=symf(a,np.sqrt);ai=symf(a,lambda d:1/np.sqrt(d))
                tentative=ah@symf(-eta*ai@grad@ai,np.exp)@ah
                result_group.append(symf(tentative,lambda d:np.clip(d,np.exp(-2*box),np.exp(2*box))))
            result.append(np.stack(result_group))
        else:
            energy=np.mean(sample**2,axis=0)
            if group.kind=='scalar':energy=energy.mean(axis=1)
            grad=.5*np.exp(value)*energy-target/2+mu/4*value
            result.append(np.clip(value-eta*grad,-2*box,2*box))
        start+=group.dimension
    return tuple(result)


def reference_average(groups,old,new,weight):
    result=[]
    for group,a,b in zip(groups,old,new):
        if group.kind=='full':
            item=[]
            for aa,bb in zip(a,b):
                half=symf(aa,np.sqrt);inv=symf(aa,lambda d:1/np.sqrt(d))
                item.append(half@symf(inv@bb@inv,lambda d:d**weight)@half)
            result.append(np.stack(item))
        else:result.append((1-weight)*a+weight*b)
    return tuple(result)


def error(a,b):
    return max(float(np.linalg.norm(np.asarray(x)-y)/max(1.,np.linalg.norm(y)))
               for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    assert '5090' in jax.devices()[0].device_kind and jax.devices()[0].platform=='gpu'
    jax.config.update('jax_enable_x64',False)
    started=time.perf_counter()
    report={'host':subprocess.check_output(['hostname'],text=True).strip(),
            'gpu':subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip(),
            'jax':jax.__version__,'main_dtype':'float32 GPU','reference_dtype':'float64 CPU',
            'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path(__file__).with_name('mixed_metric_gpu.py')]},'checks':[]}
    rng=np.random.default_rng(13)
    rows=(BlockGroup('scalar',2),BlockGroup('diagonal',2),BlockGroup('full',3,2))
    columns=(BlockGroup('full',2),BlockGroup('scalar',3),BlockGroup('diagonal',1))
    m,n=sum(g.dimension for g in rows),sum(g.dimension for g in columns)
    dense=rng.normal(size=(m,n)).astype(np.float32)*np.float32(.2)
    kj=jnp.asarray(dense)
    a0=initial_state(rows);b0=initial_state(columns)
    # Noncommuting initial FULL blocks and nonzero scalar/diagonal logs.
    def varied(groups):
        values=[]
        for g in groups:
            if g.kind=='full':
                mats=[]
                for _ in range(g.count):
                    q,_=np.linalg.qr(rng.normal(size=(g.size,g.size)))
                    mats.append((q*np.exp(rng.uniform(-.7,.7,g.size)))@q.T)
                values.append(jnp.asarray(mats,jnp.float32))
            else:values.append(jnp.asarray(rng.uniform(-.4,.4,(g.count,) if g.kind=='scalar' else (g.count,g.size)),jnp.float32))
        return tuple(values)
    a0,b0=varied(rows),varied(columns)
    ah,bh=tuple(np.asarray(z,np.float64) for z in a0),tuple(np.asarray(z,np.float64) for z in b0)
    f=rng.normal(size=(3,m)).astype(np.float32);r=rng.normal(size=(3,n)).astype(np.float32)
    config=dict(target_row=np.sqrt(n/m),target_col=np.sqrt(m/n),mu=1.,box=2.,eta=.17)
    one=jax.jit(lambda a,b,fw,rv:update_from_samples(rows,columns,a,b,fw,rv,**config))
    t=time.perf_counter();got=one(a0,b0,jnp.asarray(f),jnp.asarray(r));jax.block_until_ready(got)
    ref=(reference_step(rows,ah,f.astype(np.float64),config['target_row'],1.,2.,.17),
         reference_step(columns,bh,r.astype(np.float64),config['target_col'],1.,2.,.17))
    e=error(got,ref);assert e<3e-5,e
    report['checks'].append({'name':'mixed simultaneous step including scalar multiplicity','error':e,'passed':True,'cold_seconds':time.perf_counter()-t})
    vg=jnp.asarray(rng.normal(size=m),jnp.float32)
    for power in (-1.,.5,1.):
        got_apply=jax.jit(lambda v:apply_power(rows,a0,v,power))(vg)
        ref_apply=reference_apply(rows,ah,np.asarray(vg,np.float64),power)
        e=float(np.linalg.norm(np.asarray(got_apply)-ref_apply)/max(1,np.linalg.norm(ref_apply)))
        assert e<3e-5,e
        report['checks'].append({'name':f'mixed metric power {power}','error':e,'passed':True})
    # Exact same JAX signs replayed through a separate NumPy implementation.
    iterations,probes,shift=12,2,20
    key=jax.random.PRNGKey(4);keys=key
    ar,br=ah,bh;avg_a,avg_b=ah,bh;weights=0.
    for index in range(iterations):
        keys,kr,kc=jax.random.split(keys,3)
        xi=np.asarray(jax.random.rademacher(kr,(probes,n),dtype=jnp.float32),np.float64)
        ze=np.asarray(jax.random.rademacher(kc,(probes,m),dtype=jnp.float32),np.float64)
        fw=np.stack([dense.astype(np.float64)@reference_apply(columns,br,v,.5) for v in xi])
        rv=np.stack([dense.astype(np.float64).T@reference_apply(rows,ar,v,.5) for v in ze])
        weight=index+1+shift-1;weights+=weight
        if index==0:avg_a,avg_b=ar,br
        else:
            avg_a=reference_average(rows,avg_a,ar,weight/weights)
            avg_b=reference_average(columns,avg_b,br,weight/weights)
        eta=8/(index+1+shift)
        ar,br=(reference_step(rows,ar,fw,np.sqrt(n/m),1.,2.,eta),
               reference_step(columns,br,rv,np.sqrt(m/n),1.,2.,eta))
    constructor=make_constructor(lambda x:kj@x,lambda y:kj.T@y,rows,columns,
                                 iterations=iterations,probes=probes,shift=shift)
    t=time.perf_counter();answer=constructor(key,a0,b0);jax.block_until_ready(answer)
    e=error((answer['A'],answer['B']),(avg_a,avg_b));assert e<2e-4,e
    assert bool(answer['finite']) and int(answer['operator_calls'])==2*iterations*probes
    report['checks'].append({'name':'12-step identical-probe NumPy trajectory and pre-update mean','error':e,'passed':True,'cold_seconds':time.perf_counter()-t})
    # The zero-width box must return identity in every declared class.
    zero_box=make_constructor(lambda x:kj@x,lambda y:kj.T@y,rows,columns,iterations=3,box=0.)
    za,zb=initial_state(rows),initial_state(columns)
    z=zero_box(key,za,zb);jax.block_until_ready(z)
    e=error((z['A'],z['B']),(za,zb));assert e<3e-5,e
    report['checks'].append({'name':'zero-width box identity','error':e,'passed':True})
    # Matrix-free rank-zero operator checks simultaneous drift and all metrics.
    zero=make_constructor(lambda x:jnp.zeros(m,jnp.float32),lambda y:jnp.zeros(n,jnp.float32),rows,columns,iterations=12)
    z=zero(key,za,zb);jax.block_until_ready(z)
    assert bool(z['finite'])
    report['checks'].append({'name':'rank-zero mixed declarations finite','passed':True})
    # Force the chunked spectral implementation without a larger problem.
    chunk=make_constructor(lambda x:kj@x,lambda y:kj.T@y,rows,columns,iterations=12,probes=2,shift=20,batch_size=1)
    z=chunk(key,a0,b0);jax.block_until_ready(z)
    e=error((z['A'],z['B']),(answer['A'],answer['B']));assert e<3e-4,e
    report['checks'].append({'name':'chunked-vs-batched eigensolver','error':e,'passed':True})
    # Box projection must not hide an overflow in an unbounded oracle product.
    scalar=(BlockGroup('scalar',2),)
    huge=make_constructor(lambda x:jnp.float32(1e20)*x,lambda y:jnp.float32(1e20)*y,
                          scalar,scalar,iterations=2)
    si=initial_state(scalar);z=huge(key,si,si);jax.block_until_ready(z)
    assert not bool(z['finite'])
    report['checks'].append({'name':'overflow cannot be hidden by box clipping','passed':True})
    # Finite gradient with an overflowing step must also fail before clipping.
    tiny_ridge=make_constructor(lambda x:x,lambda y:y,scalar,scalar,iterations=2,mu=1e-44)
    z=tiny_ridge(key,si,si);jax.block_until_ready(z)
    assert not bool(z['finite'])
    report['checks'].append({'name':'overflowing log step cannot be hidden by box clipping','passed':True})
    report.update(passed=True,checks_count=len(report['checks']),total_seconds=time.perf_counter()-started,
                  row_groups=[asdict(g) for g in rows],column_groups=[asdict(g) for g in columns],
                  memory_stats=jax.devices()[0].memory_stats())
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
