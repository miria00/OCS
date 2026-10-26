"""Tiny exact construction-quality audit; all numerical work runs on KEVIN.

Only <=64-dimensional independent reference matrices are allowed. The intrinsic
gradient gives a conservative strong-convexity gap bound using normal v=0,
which is valid at every feasible point. No stochastic trace estimate is used.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp

from mixed_metric_gpu import BlockGroup,initial_state,make_constructor
from mixed_metric_f1 import make_problem,host_operator
from ocs.solver.equilibrate_sgd import smfe_equilibration


def spectral(a,fn):
    d,q=np.linalg.eigh((a+a.T)*.5)
    return (q*fn(d))@q.T


def matrices(groups,state):
    parts=[]
    for g,value in zip(groups,state):
        for i in range(g.count):
            parts.append(value[i] if g.kind=='full' else
                np.eye(g.size)*np.exp(value[i]) if g.kind=='scalar' else np.diag(np.exp(value[i])))
    return scipy.linalg.block_diag(*parts)


def exact(k,rows,cols,a,b,mu=1.):
    am,bm=matrices(rows,a),matrices(cols,b)
    m,n=k.shape; r,s=np.sqrt(n/m),np.sqrt(m/n)
    ka=k@bm@k.T;kb=k.T@am@k
    f=.5*np.trace(am@ka);tangents=[];norm2=0.;margin=np.inf
    for groups,state,gram,target in ((rows,a,ka,r),(cols,b,kb,s)):
        side=[];offset=0
        for g,vals in zip(groups,state):
            ts=[]
            for i in range(g.count):
                value=vals[i];lo,hi=offset,offset+g.size;offset=hi
                block=gram[lo:hi,lo:hi]
                if g.kind=='full':
                    log=spectral(value,np.log);half=spectral(value,np.sqrt)
                    logs=np.linalg.eigvalsh(log)
                    tangent=.5*half@block@half-target/2*np.eye(g.size)+mu/4*log
                    norm2+=np.sum(tangent*tangent)
                else:
                    logs=np.broadcast_to(value,(g.size,))
                    raw=.5*np.exp(logs)*np.diag(block)-target/2+mu/4*logs
                    tangent=raw.mean() if g.kind=='scalar' else raw
                    norm2+=g.size*tangent*tangent if g.kind=='scalar' else np.sum(tangent*tangent)
                margin=min(margin,float(4-np.max(np.abs(logs))))
                f+=-target/2*np.sum(logs)+mu/8*np.sum(logs*logs)
                ts.append(tangent)
            side.append(np.asarray(ts))
        tangents.append(tuple(side))
    if not np.isfinite(f) or margin<0:raise ArithmeticError('invalid/infeasible point; no boundary repair')
    return float(f),float(norm2),tangents,float(margin)


def step(groups,state,tangents,eta):
    out=[]
    for g,vals,ts in zip(groups,state,tangents):
        if g.kind=='full':
            values=[]
            for value,tangent in zip(vals,ts):
                half=spectral(value,np.sqrt)
                candidate=half@spectral(-eta*tangent,np.exp)@half
                values.append(spectral(candidate,lambda d:np.clip(d,np.exp(-4),np.exp(4))))
            out.append(np.asarray(values))
        else:out.append(np.clip(vals-eta*ts,-4,4))
    return tuple(out)


def reference(k,rows,cols):
    a=tuple(np.asarray(v,np.float64) for v in initial_state(rows))
    b=tuple(np.asarray(v,np.float64) for v in initial_state(cols))
    started=time.perf_counter();accepted=0
    for iteration in range(2000):
        f,norm2,tangent,margin=exact(k,rows,cols,a,b)
        if norm2/.5<=1e-10*sum(k.shape):break
        eta=1.
        for backtrack in range(45):
            an,bn=step(rows,a,tangent[0],eta),step(cols,b,tangent[1],eta)
            fn,_,_,_=exact(k,rows,cols,an,bn)
            if fn<=f-1e-4*eta*norm2:break
            eta*=.5
        else:break
        a,b=an,bn;accepted+=1
        if time.perf_counter()-started>20:break
    f,norm2,_,margin=exact(k,rows,cols,a,b)
    return {'value':f,'intrinsic_residual':float(np.sqrt(norm2)),
        'gap_upper_v_zero':norm2/.5,'box_log_margin':margin,'iterations':accepted,
        'gap_upper_per_dimension':norm2/.5/sum(k.shape),
        'seconds':time.perf_counter()-started,'reference_gate':bool(norm2/.5<=1e-10*sum(k.shape)),
        'method':'independent float64 exact gradient, exponential, Armijo, box projection'},a,b


def gradient_check(k,rows,cols,a,b):
    _,_,tangents,_=exact(k,rows,cols,a,b);rng=np.random.default_rng(273);errors=[]
    for trial in range(3):
        directions=[];predicted=0.
        for groups,ts in zip((rows,cols),tangents):
            side=[]
            for g,t in zip(groups,ts):
                z=rng.normal(size=t.shape)
                if g.kind=='full':z=(z+z.transpose(0,2,1))*.5
                z/=max(1.,np.linalg.norm(z));side.append(z)
                predicted+=float(np.sum(t*z))*(g.size if g.kind=='scalar' else 1)
            directions.append(tuple(side))
        h=1e-5
        plus=[step(g,x,v,-h) for g,x,v in zip((rows,cols),(a,b),directions)]
        minus=[step(g,x,v,h) for g,x,v in zip((rows,cols),(a,b),directions)]
        fp=exact(k,rows,cols,*plus)[0];fm=exact(k,rows,cols,*minus)[0]
        errors.append(abs((fp-fm)/(2*h)-predicted)/max(1.,abs(predicted)))
    assert max(errors)<2e-7,errors
    return {'geodesic_central_difference_errors':errors,'passed':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64',False)
    report={'dtype':'float32 GPU construction; independent float64 exact CPU reference',
        'mu':1.,'M':2.,'shift':20,'probes':2,'updates':80,
        'gap_bound':'||grad Psi||_g^2/(2 mu_g), mu_g=mu/4, normal v=0; numerical not interval',
        'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('mixed_metric_*.py')},'cases':[]}
    out=Path(args.output)
    rng=np.random.default_rng(13)
    mixed_rows=(BlockGroup('scalar',2),BlockGroup('diagonal',2),BlockGroup('full',3,2))
    mixed_cols=(BlockGroup('full',2),BlockGroup('scalar',3),BlockGroup('diagonal',1))
    # Match the existing validator's matrix fixture exactly.
    k=rng.normal(size=(10,6)).astype(np.float32)*np.float32(.2)
    fixtures=[('mixed10x6',k,mixed_rows,mixed_cols,None)]
    _,_,op,_=make_problem(16,0,.9);forward,_,_=host_operator(op)
    kd=np.column_stack([forward(np.eye(16)[:,i]) for i in range(16)])
    for kind,group in [('term_scalar',BlockGroup('scalar',32)),('block_scalar',BlockGroup('scalar',4,8)),('full',BlockGroup('full',4,8))]:
        fixtures.append(('F1_n16_'+kind,kd,(group,),(BlockGroup('diagonal',16),),op))
    for name,k,rows,cols,op in fixtures:
        assert max(k.shape)<=64
        kj=jnp.asarray(k,jnp.float32)
        mv=(lambda x:kj@x) if op is None else op.matvec
        rmv=(lambda y:kj.T@y) if op is None else op.rmatvec
        t=time.perf_counter();program=make_constructor(mv,rmv,rows,cols,iterations=80)
        answer=program(jax.random.PRNGKey(0),initial_state(rows),initial_state(cols));jax.block_until_ready(answer)
        assert bool(answer['finite'])
        construction_seconds=time.perf_counter()-t
        a,b=(tuple(np.asarray(v,np.float64) for v in answer[key]) for key in ('A','B'))
        f,norm2,_,margin=exact(k,rows,cols,a,b);ref,refa,refb=reference(k,rows,cols)
        row={'name':name,'rows':[asdict(g) for g in rows],'columns':[asdict(g) for g in cols],
            'matrix_sha256':hashlib.sha256(k.tobytes()).hexdigest(),'value':f,'intrinsic_residual':float(np.sqrt(norm2)),
            'gap_upper_v_zero':norm2/.5,'box_log_margin':margin,'reference':ref,
            'suboptimality_lower':max(0.,f-ref['value']),
            'suboptimality_upper':f-ref['value']+ref['gap_upper_v_zero'],
            'relative_suboptimality_upper':(f-ref['value']+ref['gap_upper_v_zero'])/max(1.,abs(ref['value'])),
            'construction_cold_seconds':construction_seconds}
        row['gradient_check']=gradient_check(k,rows,cols,a,b)
        row['gap_upper_per_dimension']=row['suboptimality_upper']/sum(k.shape)
        row['reference_interval_valid']=bool(row['suboptimality_upper']>=0 and ref['reference_gate'])
        assert row['reference_interval_valid']
        np.savez_compressed(out.with_name(out.stem+'_'+name+'.npz'),K=k,
            **{f'{prefix}{i}':v for prefix,state in [('A',a),('B',b),('ref_A',refa),('ref_B',refb)] for i,v in enumerate(state)})
        if name=='F1_n16_term_scalar':
            tau,sigma,nrm=smfe_equilibration(mv,rmv,16,32,[(0,32)],gamma=1.,M=2.,probes=2,iters=80)
            scale=.9/nrm;sa=(np.log(np.asarray(sigma[:1],np.float64)/scale),);sb=(np.log(np.asarray(tau,np.float64)/scale)[None],)
            sf,sn,_,sm=exact(k,rows,cols,sa,sb)
            row['configured_smfe']={'value':sf,'intrinsic_residual':float(np.sqrt(sn)),
                'gap_upper_v_zero':sn/.5,'box_log_margin':sm,
                'relative_suboptimality_upper':(sf-ref['value']+ref['gap_upper_v_zero'])/max(1.,abs(ref['value']))}
            longer=make_constructor(mv,rmv,rows,cols,iterations=160)
            ans=longer(jax.random.PRNGKey(0),initial_state(rows),initial_state(cols));jax.block_until_ready(ans)
            assert bool(ans['finite'])
            la,lb=(tuple(np.asarray(v,np.float64) for v in ans[key]) for key in ('A','B'))
            lf,ln,_,lm=exact(k,rows,cols,la,lb)
            lower_smfe=max(0.,sf-ref['value'])
            row['new_restricted_double_budget']={'updates':160,'operator_calls':640,
                'value':lf,'gap_upper':lf-ref['value']+ref['gap_upper_v_zero'],
                'same_tie_factor_two_quality_gate':bool(lf-ref['value']+ref['gap_upper_v_zero']<=2*max(lower_smfe,1e-10)),
                'comparison_floor':1e-10,'box_log_margin':lm}
        report['cases'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
    report['all_reference_gates']=all(c['reference']['reference_gate'] for c in report['cases'])
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
