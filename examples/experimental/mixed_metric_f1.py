"""Matched F1 constructor/solver gate; run only on KEVIN RTX 5090.

No global matrix is formed except an independent reference at n<=32. The
constructor only receives callbacks. Normalization separately uses the known
partition structure and small blocks; it is not a generic norm oracle.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp
import ocs
from ocs.ir import ProblemFamily, Term
from ocs.solver.equilibrate_sgd import smfe_equilibration
import ocs.solver.equilibrate_sgd as smfe_module
from blockmetric_models import BlockMixSelect, StackedBlockMix, Spec, ChannelBlur, blockmetric_f2
from blockmetric_pilot import Loop, prox_l2_diag_any, ball_project, PROX_IMPL
from mixed_metric_gpu import BlockGroup, initial_state, make_constructor, apply_power, state_storage_bytes

NVIDIA_SMI=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_problem(n, seed, rho):
    rng=np.random.default_rng(seed);k=4
    op1=BlockMixSelect(rng.permutation(n),rng.choice([-1.,1.],size=(n//k,k)),rho)
    op2=BlockMixSelect(rng.permutation(n),rng.choice([-1.,1.],size=(n//k,k)),rho)
    op=StackedBlockMix(op1,op2)
    truth=rng.normal(size=n)
    b=(np.asarray(jax.jit(op.matvec)(jnp.asarray(truth,jnp.float32)))+.3*rng.normal(size=2*n)).astype(np.float32)
    problem=ProblemFamily(n,[Term(ocs.Identity(n),ocs.SumSquares().scaled(.01)),
        Term(op,ocs.GroupL2(lam=1.,group_size=4),lambda data:-data['b'])])
    spec=Spec(n=n,mu=.01,primal='ridge',data={'b':jnp.asarray(b)},
              dual_blocks=[{'op':op,'kind':'group_l2','k':4,'lam':1.,'offset':jnp.asarray(-b)}])
    return problem,spec,op,b


def make_rof(side,seed):
    rng=np.random.default_rng(seed);grid=np.linspace(-1,1,side)
    z,y,x=np.meshgrid(grid,grid,grid,indexing='ij')
    phantom=((x*x+y*y+z*z)<.45).astype(np.float32)+.4*((x-.4)**2+(y+.2)**2+(z-.1)**2<.18)
    observed=(phantom+.08*rng.normal(size=phantom.shape)).astype(np.float32).reshape(-1)
    n=side**3;op=ocs.Gradient3D(side,side,side);rhs=-np.asarray(op.matvec(jnp.asarray(observed)))
    problem=ProblemFamily(n,[Term(ocs.Identity(n),ocs.SumSquares()),
        Term(op,ocs.GroupL2(lam=.02,group_size=3),lambda data:-data['b'])])
    spec=Spec(n=n,mu=1.,primal='ridge',data={'b':jnp.asarray(rhs)},observed=observed,
        dual_blocks=[{'op':op,'kind':'group_l2','k':3,'lam':.02,'offset':jnp.asarray(-rhs)}])
    return problem,spec,op,rhs


def host_operator(op):
    """Independent float64 evaluation of the same stored operator parameters."""
    if isinstance(op,ocs.Gradient3D):
        shape=(op.D,op.H,op.W)
        def forward(x):
            v=np.asarray(x,np.float64).reshape(shape);pieces=[]
            for axis in range(3):
                pad=[(0,0)]*3;pad[axis]=(0,1)
                pieces.append(np.pad(np.diff(v,axis=axis),pad))
            return np.stack(pieces,axis=-1).reshape(-1)
        def reverse(y):
            value=np.asarray(y,np.float64).reshape(*shape,3);out=np.zeros(shape)
            for axis in range(3):
                lower=[slice(None)]*3;upper=[slice(None)]*3
                lower[axis]=slice(0,-1);upper[axis]=slice(1,None)
                edge=value[...,axis][tuple(lower)]
                out[tuple(lower)]-=edge;out[tuple(upper)]+=edge
            return out.reshape(-1)
        return forward,reverse,{'kind':'gradient3d','shape':shape}
    if isinstance(op,ChannelBlur):
        matrix=np.asarray(op.M,np.float64);frequency=np.asarray(op.H,np.complex128)
        def blur(z,conjugate=False):
            h=np.conj(frequency) if conjugate else frequency
            return np.fft.ifft2(np.fft.fft2(z.reshape(op.c,op.L,op.L))*h[None]).real.reshape(op.c,op.N)
        def forward(x):return (matrix@blur(x)).T.reshape(-1)
        def reverse(y):return blur(matrix.T@y.reshape(op.N,op.c).T,True).reshape(-1)
        return forward,reverse,{'matrix':matrix,'frequency':frequency,'channels':op.c,'side':op.L}
    parts=[]
    for sub in (op.K1,op.K2):
        parts.append((np.asarray(sub.perm),np.asarray(sub.inv_perm),np.asarray(sub.a,np.float64),
                      float(np.float32(sub.c0)),float(np.float32(sub.rho)),
                      float(np.float32(1-sub.rho))))
    def forward(x):
        out=[]
        for perm,inv,a,c,r,s in parts:
            v=x[perm].reshape(-1,4);out.append((c*(s*v+r*a*np.sum(a*v,axis=1,keepdims=True))).reshape(-1))
        return np.concatenate(out)
    def reverse(y):
        out=np.zeros(op.shape[1],np.float64);start=0
        for perm,inv,a,c,r,s in parts:
            v=y[start:start+len(perm)].reshape(-1,4)
            z=c*(s*v+r*a*np.sum(a*v,axis=1,keepdims=True))
            out+=z.reshape(-1)[inv];start+=len(perm)
        return out
    return forward,reverse,parts


def decoded_metric(rows, state, m, k=4):
    group=rows[0];value=np.asarray(state[0],np.float64)
    if group.kind=='full':return value
    scalars=np.repeat(np.exp(value),group.size//k) if group.size!=k else np.exp(value)
    assert len(scalars)==m//k
    return scalars[:,None,None]*np.eye(k)


def construction_objective(a,b,forward,*,mu=1.,probes=64,seed=73):
    k=a.shape[1];m=len(a)*k;n=len(b);rng=np.random.default_rng(seed)
    energies=[]
    for _ in range(probes):
        z=forward(np.sqrt(b)*rng.choice([-1.,1.],n)).reshape(-1,k)
        energies.append(float(np.einsum('bi,bij,bj->',z,a,z)))
    logs=np.log(np.linalg.eigvalsh(a));logb=np.log(b)
    deterministic=-np.sqrt(n/m)/2*logs.sum()-np.sqrt(m/n)/2*logb.sum()+mu/8*(np.sum(logs**2)+np.sum(logb**2))
    if n<=32:
        dense=np.column_stack([forward(np.eye(n)[:,i]) for i in range(n)])
        blocks=dense.reshape(-1,k,n)
        trace=np.einsum('bik,k,bjk,bij->',blocks,b,blocks,a)
        return {'value':float(.5*trace+deterministic),'kind':'exact tiny dense reference in float64','matrix_shape':list(dense.shape)}
    return {'value':float(.5*np.mean(energies)+deterministic),
            'trace_standard_error':float(.5*np.std(energies,ddof=1)/np.sqrt(probes)),
            'kind':'independent Rademacher estimate; not certified optimum gap','audit_operator_calls':probes}


def normalization(a,b,parts):
    """Gershgorin bound for local scaled Gram; then sum the two partition bounds.

    ||stack K_l||² <= sum_l max_i ||A_i^1/2 C_i diag(sqrt(B_perm))||².
    Each local squared norm is bounded by max absolute row sum of its Gram.
    Float64 evaluation uses an explicit generous roundoff allowance; this is
    not interval arithmetic and no finite power estimate is called a bound.
    """
    if isinstance(parts,dict):
        if parts.get('kind')=='gradient3d':
            shape=tuple(parts['shape']);eigen,vectors=np.linalg.eigh(a)
            half=np.einsum('bij,bj,bkj->bik',vectors,np.sqrt(eigen),vectors).reshape(*shape,3,3)
            root=np.sqrt(b).reshape(shape);mask=np.ones((*shape,3))
            for axis in range(3):
                end=[slice(None)]*3;end[axis]=-1;mask[tuple(end)+(axis,)]=0
            coefficients=half*mask[...,None,:]
            center=np.abs(coefficients.sum(axis=-1))
            rows=root[...,None]*center;columns=root*center.sum(axis=-1)
            for axis in range(3):
                lower=[slice(None)]*3;upper=[slice(None)]*3
                lower[axis]=slice(0,-1);upper[axis]=slice(1,None)
                absolute=np.abs(coefficients[...,axis])
                rows[tuple(lower)]+=absolute[tuple(lower)]*root[tuple(upper)][...,None]
                columns[tuple(upper)]+=root[tuple(upper)]*absolute[tuple(lower)].sum(axis=-1)
            allowance=512*np.finfo(np.float64).eps*max(1.,np.max(np.abs(half)))*max(1.,root.max())
            rmax=float(rows.max()+allowance);cmax=float(columns.max()+allowance);upper=np.sqrt(rmax*cmax)
            return upper,{'kind':'actual local square-root gradient stencil induced1/infinity bound, float64',
                'row_sum_max':rmax,'column_sum_max':cmax,'upper':upper,'interval_certificate':False,
                'roundoff_allowance':float(allowance),'safety':.95}
        # Bhalf=D R, ||R||<=1, with D constant in each channel. D
        # commutes with the common circular blur. Bound each pixel's
        # Ahalf M D by its tiny Gram absolute row sum, then multiply Hmax.
        c=parts['channels'];dc=np.sqrt(b.reshape(c,-1).max(axis=1))
        scaled=parts['matrix']*dc[None,:]
        gram=np.einsum('ji,bjk,kl->bil',scaled,a,scaled)
        allowance=256*np.finfo(np.float64).eps*np.sum(np.abs(scaled))**2*np.max(np.abs(a),axis=(1,2))
        squared=float(np.max(np.abs(gram).sum(axis=2).max(axis=1)+allowance))
        hmax=float(np.abs(parts['frequency']).max());upper=np.sqrt(squared)*hmax
        if not np.isfinite(upper) or upper<=0:raise ArithmeticError('invalid blur norm bound')
        return upper,{'kind':'channel-maximum diagonal/commuting blur/Gershgorin analytical bound in float64',
            'channel_scaling_max':dc.tolist(),'pixel_gram_upper_squared':squared,'Hmax':hmax,'upper':upper,
            'roundoff_allowance':'256 eps64 ||M D||entry1^2 maxabs(A)',
            'interval_certificate':False,'safety':.95}
    bounds=[];start=0
    for perm,inv,signs,c,r,s in parts:
        count=len(perm)//4
        mixing=c*(s*np.eye(4)[None]+r*signs[:,:,None]*signs[:,None,:])
        scaled=mixing*np.sqrt(b[perm].reshape(-1,4))[:,None,:]
        gram=np.einsum('bji,bjk,bkl->bil',scaled,a[start:start+count],scaled)
        absolute=np.sum(np.abs(gram),axis=2)
        allowance=256*np.finfo(np.float64).eps*np.sum(np.abs(scaled),axis=(1,2))**2*np.max(np.abs(a[start:start+count]),axis=(1,2))
        bounds.append(float(np.max(np.max(absolute,axis=1)+allowance)))
        start+=count
    upper=float(np.sqrt(sum(bounds)))
    if not np.isfinite(upper) or upper<=0:raise ArithmeticError('invalid norm upper bound')
    return upper,{'kind':'partition/Gershgorin analytical bound evaluated in float64',
                  'squared_partition_bounds':bounds,'upper':upper,
                  'roundoff_allowance':'256 eps64 ||scaled block||entry1^2 maxabs(A)',
                  'interval_certificate':False,'safety':.95}


def prox_check(sigma,full=True,lam=1.):
    from scipy.optimize import brentq
    eigen=np.linalg.eigvalsh(sigma)
    condition=eigen[:,-1]/eigen[:,0]
    rng=np.random.default_rng(29)
    selected=np.unique(np.r_[np.argmax(condition),np.argmin(eigen[:,0]),np.argmax(eigen[:,-1]),
                              rng.integers(len(eigen),size=min(12,len(eigen)))])
    ds=[];vs=[]
    for i in selected:
        for magnitude in (0.,1e-7,.3,1.,2.,30.):
            vector=rng.normal(size=sigma.shape[1]);vector*=lam*magnitude/np.linalg.norm(vector)
            ds.append(eigen[i]);vs.append(vector)
    d=np.asarray(ds,np.float32);v=np.asarray(vs,np.float32)
    projector=(lambda vv,dd:vv-dd*prox_l2_diag_any(vv/dd,dd,lam)) if full else (lambda vv,dd:ball_project(vv,lam))
    actual=jax.jit(projector)(jnp.asarray(v),jnp.asarray(d))
    actual=np.asarray(actual,np.float64)
    references=[];worst_bound=0.;outside=0.
    for yy,vv,dd in zip(actual,v.astype(np.float64),d.astype(np.float64)):
        if np.linalg.norm(vv)<=lam:ref=vv;eta=0.
        else:
            f=lambda eta:np.linalg.norm(vv/(1+eta*dd))-lam
            upper=1.
            while f(upper)>0:upper*=2
            eta=brentq(f,0.,upper,xtol=1e-13)
            ref=vv/(1+eta*dd)
        references.append(ref)
        outside=max(outside,float(max(np.linalg.norm(yy)-lam,0.)/lam))
        if np.linalg.norm(yy)<=lam:
            residual=(yy-vv)/dd+eta*yy
            bound=.5*np.sum(residual**2/(1/dd+eta))+.5*eta*(lam*lam-yy@yy)
            worst_bound=max(worst_bound,float(bound/max(1.,.5*np.sum((ref-vv)**2/dd))))
    err=float(np.max(np.linalg.norm(actual-np.asarray(references),axis=1)/np.maximum(lam,np.linalg.norm(v,axis=1))))
    return {'cases':len(v),'relative_projection_error_max':err,'ball_violation_max':outside,
            'feasible_point_lagrangian_gap_max':worst_bound,
            'gap_not_claimed_for_points_outside_ball':True,
            'produced_metric_condition_max':float(condition.max()),
            'path':'component weighted projection in eigenbasis' if full else 'native scalar closed-form ball projection',
            'ball_violation_normalized_by_radius':True,'radius':lam,
            'passed':bool(err<=1e-5 and outside<=1e-5 and worst_bound<=1e-5),
            'scope':'bounded floating-point validation, not exact-prox or summable-error theorem'}


def independent(x,y,b,forward,reverse,k=4,mu=.01,lam=1.):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    residual=forward(x)-b
    primal=mu/2*(x@x)+lam*np.linalg.norm(residual.reshape(-1,k),axis=1).sum()
    grouped=y.reshape(-1,k)
    yd=(grouped/np.maximum(1.,np.linalg.norm(grouped,axis=1,keepdims=True)/lam)).reshape(-1)
    aty=reverse(yd);dual=-b@yd-(aty@aty)/(2*mu)
    gap=(primal-dual)/max(abs(primal),abs(dual),1e-30)
    stationarity=mu*x+reverse(y)
    return {'primal':float(primal),'dual_lower':float(dual),'relative_bracket':float(gap),
            'stationarity':float(np.linalg.norm(stationarity)/max(np.linalg.norm(mu*x)+np.linalg.norm(reverse(y)),1e-30)),
            'raw_dual_ball_violation':float(max(0.,np.linalg.norm(grouped,axis=1).max()-lam)),
            'dual_bound_uses_auxiliary_radially_repaired_dual':True,
            'passed':bool(np.isfinite(gap) and 0<=gap<=1e-3),
            'numerical_float64_not_interval':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=16)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--rho',type=float,default=.9)
    p.add_argument('--family',choices=['f1','f2','rof'],default='f1');p.add_argument('--side',type=int,default=32)
    p.add_argument('--mixing',default='strong')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--allocation-cap-gib',type=float,default=2.)
    p.add_argument('--arm',choices=['configured_smfe','new_term_scalar','new_block_scalar','new_full'],required=True)
    p.add_argument('--updates',type=int,default=80);p.add_argument('--solve-cap',type=int,default=10000)
    p.add_argument('--changed-data',type=int,default=0);args=p.parse_args()
    assert args.n%4==0 and args.n>0
    assert '5090' in jax.devices()[0].device_kind and jax.devices()[0].platform=='gpu'
    jax.config.update('jax_enable_x64',False)
    start_all=time.perf_counter();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    engine=Path(ocs.__file__).parent
    report={'arguments':vars(args),'host':subprocess.check_output(['hostname'],text=True).strip(),
            'gpu':subprocess.check_output([NVIDIA_SMI,'--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip(),
            'jax':jax.__version__,'iteration_dtype':'float32','constructor_dtype':'float32 GPU',
            'reference_dtype':'float64 CPU','prox_impl':PROX_IMPL,
            'sources':{f.name:digest(f) for f in Path(__file__).parent.glob('mixed_metric_*.py')},
            'engine_sha256':{str(f.relative_to(engine)):digest(f) for f in sorted(engine.rglob('*.py'))}}
    report['compilation_cache_context']={name:getattr(jax.config,name,None) for name in (
        'jax_enable_compilation_cache','jax_compilation_cache_dir',
        'jax_persistent_cache_min_compile_time_secs','jax_persistent_cache_min_entry_size_bytes')}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    t=time.perf_counter()
    if args.family=='f1':problem,spec,op,data_b=make_problem(args.n,args.seed,args.rho)
    elif args.family=='f2':
        problem,spec=blockmetric_f2(args.side,mixing=args.mixing,seed=args.seed)
        op=spec.dual_blocks[0]['op'];data_b=np.asarray(spec.data['b']);args.n=spec.n
    else:problem,spec,op,data_b=make_rof(args.side,args.seed);args.n=spec.n
    forward,reverse,parts=host_operator(op);m=op.shape[0];k=spec.dual_blocks[0]['k']
    lam=spec.dual_blocks[0]['lam']
    report['model_contract']={'family':args.family,'primal_ridge_mu':spec.mu,'group_lambda':lam,
        'variable':'u = physical reconstruction minus observed volume' if args.family=='rof' else 'x',
        'family_scope':'synthetic fixtures; no measured real-data claim'}
    report['model_seconds']=time.perf_counter()-t
    if isinstance(parts,dict):operator_hashes={name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name,value in parts.items()}
    else:operator_hashes={f'partition_{i}_{name}':hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
           for i,part in enumerate(parts) for name,value in zip(('perm','inv','signs','scale','rho','one_minus_rho'),part)}
    report['input_hashes']={'b':hashlib.sha256(data_b.tobytes()).hexdigest(),**operator_hashes}
    rows=(BlockGroup('full',k,m//k),) if args.arm=='new_full' else (
        BlockGroup('scalar',k,m//k) if args.arm=='new_block_scalar' else BlockGroup('scalar',m),)
    columns=(BlockGroup('diagonal',args.n),)
    report['construction_contract']={'mu':1.,'M':2.,'r':np.sqrt(args.n/m),'s':np.sqrt(m/args.n),
        'rows':[asdict(g) for g in rows],'columns':[asdict(g) for g in columns],
        'shift':20 if args.arm!='configured_smfe' else 'shipped Euclidean schedule',
        'averaging':'pre-update geodesic' if args.arm!='configured_smfe' else 'shipped post-update log average',
        'probes':2,'updates':args.updates,'gradient_operator_calls':4*args.updates,
        'extra_shipped_norm_operator_calls':30 if args.arm=='configured_smfe' else 0,
        'cap':'box only; no extra condition cap'}
    report['construction_contract']['spectral_batch_size']=args.batch_size
    a0,b0=initial_state(rows),initial_state(columns)
    assert all(np.isfinite(np.asarray(v)).all() for v in (*a0,*b0))
    assert all(np.linalg.eigvalsh(np.asarray(v)).min()>0 for g,v in zip(rows,a0) if g.kind=='full')
    if args.arm=='configured_smfe':
        public_start=time.perf_counter()
        native=smfe_equilibration(op.matvec,op.rmatvec,args.n,m,[(0,m)],
            gamma=1.,M=2.,probes=2,iters=args.updates,diagnostics=False)
        jax.block_until_ready(native);native_elapsed=time.perf_counter()-public_start
        # Instrument the exact shipped run body rather than copy its algorithm.
        # The custom callback deliberately bypasses its public cache; retaining
        # this executable permits matched construction-only replay timing.
        timings={};captured={};original_decorator=smfe_module._cached_smfe_jit
        def capture_decorator(operators,settings):
            def decorate(fn):
                def execute():
                    t=time.perf_counter();lowered=jax.jit(fn).lower();timings['lower']=time.perf_counter()-t
                    t=time.perf_counter();executable=lowered.compile();timings['compile']=time.perf_counter()-t
                    t=time.perf_counter();answer=executable();jax.block_until_ready(answer);timings['execute']=time.perf_counter()-t
                    captured.update(executable=executable,answer=answer)
                    return answer
                return execute
            return decorate
        smfe_module._cached_smfe_jit=capture_decorator
        public_call_start=time.perf_counter()
        try:
            tau,sigma,nrm=smfe_equilibration(op.matvec,op.rmatvec,args.n,m,[(0,m)],
                gamma=1.,M=2.,probes=2,iters=args.updates,diagnostics=False)
        finally:smfe_module._cached_smfe_jit=original_decorator
        public_call_elapsed=time.perf_counter()-public_call_start
        warm=[]
        for _ in range(2):
            t=time.perf_counter();repeat=captured['executable']();jax.block_until_ready(repeat);warm.append(time.perf_counter()-t)
            assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(captured['answer'],repeat))
        matched_error=max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64)))/max(1.,np.max(np.abs(np.asarray(x,np.float64)))))
                          for x,y in zip(native,(tau,sigma,nrm)))
        assert matched_error<=1e-6,matched_error
        tau,sigma,nrm=native
        scale=.9/(nrm+1e-30)
        state_a=(jnp.log(sigma[:1]/scale),);state_b=(jnp.log(tau/scale).reshape(1,-1),)
        report['construction_seconds']={**timings,'cold_setup_plus_execution':native_elapsed,
            'native_public_call_seconds':native_elapsed,'captured_setup_plus_execution':sum(timings.values()),
            'captured_vs_native_relative_error':matched_error,'numerical_metric_source':'uninstrumented public call',
            'warm_replays':warm,'warm_identical':True,'compile_separated':True,'undo_original_global_scale':scale,
            'instrumented_public_wrapper_elapsed':public_call_elapsed,
            'instrumentation':'capture unchanged shipped JIT body; custom callbacks otherwise bypass executable cache'}
    else:
        program=make_constructor(op.matvec,op.rmatvec,rows,columns,iterations=args.updates,probes=2,batch_size=args.batch_size)
        key=jax.random.PRNGKey(0)
        t=time.perf_counter();lowered=program.lower(key,a0,b0);lowering=time.perf_counter()-t
        t=time.perf_counter();executable=lowered.compile();compile_s=time.perf_counter()-t
        t=time.perf_counter();answer=executable(key,a0,b0);jax.block_until_ready(answer);execution=time.perf_counter()-t
        assert bool(answer['finite']),'nonfinite constructor; no downstream solve'
        warm=[]
        for _ in range(2):
            t=time.perf_counter();repeat=executable(key,a0,b0);jax.block_until_ready(repeat);warm.append(time.perf_counter()-t)
            assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree_util.tree_leaves(answer),jax.tree_util.tree_leaves(repeat)))
        state_a,state_b=answer['A'],answer['B']
        report['construction_seconds']={'lower':lowering,'compile':compile_s,'execute':execution,
            'cold_setup_plus_execution':lowering+compile_s+execution,'warm_replays':warm,'warm_identical':True}
    report['memory_after_construction']=jax.devices()[0].memory_stats()
    save()
    assert report['memory_after_construction']['peak_bytes_in_use']<=args.allocation_cap_gib*2**30,'allocator peak exceeds declared cap'
    t=time.perf_counter();a=decoded_metric(rows,state_a,m,k);b=np.exp(np.asarray(state_b[0],np.float64)).reshape(-1)
    eig=np.linalg.eigvalsh(a)
    report['returned_spectrum']={'A_min':float(eig.min()),'A_max':float(eig.max()),'B_min':float(b.min()),'B_max':float(b.max())}
    assert eig.min()>0 and b.min()>0 and eig.min()>=np.exp(-4)*(1-1e-3) and eig.max()<=np.exp(4)*(1+1e-3)
    assert b.min()>=np.exp(-4)*(1-1e-3) and b.max()<=np.exp(4)*(1+1e-3)
    report['construction_objective']=construction_objective(a,b,forward)
    report['construction_audit_seconds']=time.perf_counter()-t
    t=time.perf_counter();norm,note=normalization(a,b,parts);factor=.95/norm
    sigma=a*factor;tau=b*factor
    report['normalization']={**note,'factor':factor,'seconds':time.perf_counter()-t}
    t=time.perf_counter();report['prox_audit']=prox_check(sigma,args.arm=='new_full',lam);report['prox_audit_seconds']=time.perf_counter()-t
    save();assert report['prox_audit']['passed'],'actual-spectrum proximal validation failed'
    t=time.perf_counter();comp=problem.compile(equilibrate='scalar',block=50,max_blocks=args.solve_cap//50,
                                               tol=1e-3,pdhg_variant='standard');report['public_certificate_setup_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if args.arm=='new_full':
        values,vectors=np.linalg.eigh(sigma)
        metrics=[('full',jnp.asarray(vectors,jnp.float32),jnp.asarray(values,jnp.float32))]
    else:metrics=[('scalar',jnp.asarray(sigma[:,0,0,None],jnp.float32))]
    loop=Loop(spec,jnp.asarray(tau,jnp.float32),metrics,block=50)
    jax.block_until_ready((loop.T,loop.marrays))
    report['solver_metric_setup_seconds']=time.perf_counter()-t
    solve_records=[]
    for case in range(1+args.changed_data):
        # Only the offset changes: same operator and metric, genuinely new data.
        rhs=data_b if case==0 else (data_b+.05*np.random.default_rng(900+case).normal(size=m)).astype(np.float32)
        observed=None
        if args.family=='rof':
            observed=spec.observed if case==0 else (spec.observed+.05*np.random.default_rng(900+case).normal(size=args.n)).astype(np.float32)
            rhs=-np.asarray(op.matvec(jnp.asarray(observed)))
        offs=(jnp.asarray(-rhs),);raw={'b':jnp.asarray(rhs)}
        x,ys,xb=loop.init_state();history=[];iterate_s=check_s=0.;start=time.perf_counter();quality=None
        for index in range(args.solve_cap//50):
            t=time.perf_counter();x,ys,xb=loop._run_block(x,ys,xb,loop.T,loop.marrays,offs);jax.block_until_ready(x);iterate_s+=time.perf_counter()-t
            t=time.perf_counter();y=loop.returned_pair(x,ys);certificate=comp.certificate(x,y,raw)
            residual=max(certificate['r_primal'],certificate['r_dual'])
            quality=independent(x,y,rhs,forward,reverse,k,spec.mu,lam) if residual<=1e-3 else None
            check_s+=time.perf_counter()-t
            history.append([50*(index+1),residual,None if quality is None else quality['relative_bracket'],time.perf_counter()-start])
            if quality is not None and residual<=1e-3 and quality['passed']:break
            if not np.isfinite(residual) or time.perf_counter()-start>60:break
        # Always refresh at the saved final pair, including capped/failed solves.
        t=time.perf_counter();quality=independent(x,y,rhs,forward,reverse,k,spec.mu,lam);check_s+=time.perf_counter()-t
        row={'case':case,'input_sha256':hashlib.sha256(rhs.tobytes()).hexdigest(),
             'iterations':50*(index+1),'residual':float(residual),'independent':quality,
             'accepted':bool(residual<=1e-3 and quality['passed']),
             'iteration_seconds_including_first_compile':iterate_s,'check_and_transfer_seconds':check_s,
             'solve_plus_checks_seconds':time.perf_counter()-start,'history':history}
        if case==0:
            report['checked_cold_first_solution_seconds']=sum(report[name] for name in (
                'model_seconds','construction_audit_seconds','prox_audit_seconds',
                'public_certificate_setup_seconds','solver_metric_setup_seconds'))+report['construction_seconds']['cold_setup_plus_execution']+report['normalization']['seconds']+row['solve_plus_checks_seconds']
            report['cold_timing_scope']='sum measured model, constructor, exact/estimated construction audit, norm bound, prox audit, public certificate setup, solver metric setup, first solve+checks; excludes research replays, file I/O and metadata hashing'
        solve_records.append(row);report['solves']=solve_records;save()
        extra={} if observed is None else {'observed_volume':observed,'physical_reconstruction':np.asarray(x)+observed}
        np.savez_compressed(out.with_name(out.stem+f'_case{case}.npz'),x=np.asarray(x),y=np.asarray(y),b=rhs,**extra)
    report['kernel_breakdown']=loop.bench(reps=3)
    report['total_seconds']=time.perf_counter()-start_all
    report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['memory_final']=jax.devices()[0].memory_stats()
    report['allocator_cap_passed']=bool(report['memory_final']['peak_bytes_in_use']<=args.allocation_cap_gib*2**30)
    report['gpu_processes']=subprocess.check_output([NVIDIA_SMI,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    report['single_metric_state_bytes']=state_storage_bytes(rows)+state_storage_bytes(columns)
    np.savez_compressed(out.with_name(out.stem+'_metrics.npz'),A=a,B=b,tau=tau,sigma=sigma)
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('engine_sha256','solves')},indent=2),flush=True)
    print(json.dumps({'solves':[{k:v for k,v in row.items() if k!='history'} for row in solve_records]},indent=2),flush=True)


if __name__=='__main__':main()
