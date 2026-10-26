"""Frozen metric expanded/compact endpoint and throughput control, remote only."""
import argparse
import json
from pathlib import Path
import time
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from blockmetric_models import blockmetric_f2
from blockmetric_pilot import Loop
from shared_shape_f2 import SharedF2Loop
from mixed_metric_f1 import host_operator, independent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--input',required=True);args=p.parse_args()
    assert jax.devices()[0].platform=='gpu' and '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64',False)
    source=Path(args.input);record=json.loads(source.read_text());side=record['arguments']['side']
    assert side<=256 and record['arm'] in ('D','F')
    problem,spec=blockmetric_f2(side,mixing='moderate',seed=0);op=spec.dual_blocks[0]['op']
    metricpath=source.with_name(source.stem+'_metrics.npz')
    with np.load(metricpath) as metric:sigma=metric['sigma_shared'];tau=metric['tau']
    with np.load(source.with_name(source.stem+'_case0.npz')) as endpoint:
        rhs=endpoint['b'];savedx=endpoint['x'];savedy=endpoint['y']
    values,q=np.linalg.eigh(sigma);k,count=op.c,op.N
    compact=SharedF2Loop(spec,tau.reshape(k,count)[:,0],q,values,fused=False)
    expanded=Loop(spec,jnp.asarray(tau,jnp.float32),[('full',
        jnp.asarray(np.broadcast_to(q,(count,k,k)),jnp.float32),
        jnp.asarray(np.broadcast_to(values,(count,k)),jnp.float32))],block=50)
    blocks=record['solves'][0]['iterations']//50
    c0=compact.init_state();e0=expanded.init_state();co=compact.offset(rhs);eo=(jnp.asarray(-rhs),)
    cj=jax.jit(lambda state:jax.lax.fori_loop(0,blocks,lambda _,s:compact.block_step(s,co),state))
    ej=jax.jit(lambda state:jax.lax.fori_loop(0,blocks,lambda _,s:expanded._run_block(*s,expanded.T,expanded.marrays,eo),state))
    ce=cj.lower(c0).compile();ee=ej.lower(e0).compile()
    # Warm both once, then alternate the order, always resetting the state.
    cs=ce(c0);es=ee(e0);jax.block_until_ready((cs,es));ct=[];et=[]
    for repetition in range(3):
        arms=['compact','expanded'] if repetition%2==0 else ['expanded','compact']
        for arm in arms:
            t=time.perf_counter()
            if arm=='compact':cs=ce(c0);jax.block_until_ready(cs);ct.append(time.perf_counter()-t)
            else:es=ee(e0);jax.block_until_ready(es);et.append(time.perf_counter()-t)
    cy=compact.assemble(cs[1],compact.q);ey=expanded.returned_pair(es[0],es[1])
    error=lambda a,b:float(np.linalg.norm(np.asarray(a,np.float64)-np.asarray(b,np.float64))/max(1.,np.linalg.norm(np.asarray(b,np.float64))))
    errors={'primal':error(cs[0],es[0]),'dual':error(cy,ey),
        'compact_vs_saved_primal':error(cs[0],savedx),'compact_vs_saved_dual':error(cy,savedy)}
    forward,reverse,_=host_operator(op)
    quality={name:independent(x,y,rhs,forward,reverse,k,spec.mu,1.) for name,x,y in
        [('compact',cs[0],cy),('expanded',es[0],ey)]}
    report={'source_run':str(source),'literal_metric_sha256':hashlib.sha256(metricpath.read_bytes()).hexdigest(),
        'arm':record['arm'],'side':side,'iterations':blocks*50,'relative_endpoint_errors':errors,
        'compact_seconds':ct,'expanded_seconds':et,'independent':quality,
        'scope':'same literal metric/data/initial state; fixed-count replay, CPU audits and compile outside timed interval',
        'all_passed':max(errors.values())<=1e-4 and all(x['passed'] for x in quality.values())}
    out=Path(args.output);out.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),compact_x=np.asarray(cs[0]),compact_y=np.asarray(cy),
        expanded_x=np.asarray(es[0]),expanded_y=np.asarray(ey),b=rhs)
    print(json.dumps(report),flush=True);assert report['all_passed']


if __name__=='__main__':main()
