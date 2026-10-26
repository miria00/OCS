"""Gate-3 F1 fixture wrapper; all numerical helpers are reused unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import jax
import jax.numpy as jnp
import ocs
from ocs.solver.equilibrate_sgd import smfe_equilibration
import ocs.solver.equilibrate_sgd as smfe_module
from blockmetric_models import blockmetric_f2
from blockmetric_pilot import Loop, PROX_IMPL, NEWTON_ITERS
from mixed_metric_gpu import BlockGroup, initial_state, make_constructor, state_storage_bytes
from mixed_metric_f1 import (make_problem, make_rof, host_operator, decoded_metric,
    construction_objective, normalization, prox_check, independent, digest, NVIDIA_SMI)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=16)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--rho',type=float,default=.9)
    p.add_argument('--family',choices=['f1','f2','rof'],default='f1');p.add_argument('--side',type=int,default=32)
    p.add_argument('--mixing',default='strong')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--allocation-cap-gib',type=float,default=2.)
    p.add_argument('--arm',choices=['configured_smfe','new_term_scalar','new_block_scalar','new_full'],required=True)
    p.add_argument('--updates',type=int,default=80);p.add_argument('--solve-cap',type=int,default=10000)
    p.add_argument('--changed-data',type=int,default=0);p.add_argument('--fixtures',required=True);p.add_argument('--protocol',default='F1_CONTROL_V1.json');args=p.parse_args()
    assert args.family=='f1' and NEWTON_ITERS==20 and PROX_IMPL=='components'
    fixture_root=Path(args.fixtures);fixture_meta=json.loads((fixture_root/'manifest.json').read_text())
    assert fixture_meta['settings']==dict(n=args.n,seed=args.seed,rho=args.rho,changed_data=args.changed_data)
    assert fixture_meta['protocol_sha256']==digest(args.protocol)
    assert digest(fixture_root/'operator.npz')==fixture_meta['operator_file_sha256']
    fixture_rhs=[]
    for entry in fixture_meta['cases']:
        path=fixture_root/entry['file'];assert digest(path)==entry['file_sha256']
        with np.load(path,allow_pickle=False) as z:rhs=z['b'].copy()
        assert rhs.dtype==np.float32 and rhs.shape==(2*args.n,)
        assert hashlib.sha256(rhs.tobytes()).hexdigest()==entry['input_sha256']
        fixture_rhs.append(rhs)
    assert len(fixture_rhs)==1+args.changed_data
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
    if args.family=='f1':
        problem,spec,op,_=make_problem(args.n,args.seed,args.rho)
        data_b=fixture_rhs[0]
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
    assert operator_hashes==fixture_meta['operator_hashes'], 'literal coefficient hash mismatch'
    report['fixture_manifest_sha256']=digest(fixture_root/'manifest.json')
    report['fixture_protocol_sha256']=fixture_meta['protocol_sha256']
    report['fixture_scope']=fixture_meta['scope']
    report['declared_root_steps']=NEWTON_ITERS
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
        rhs=fixture_rhs[case]
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
        assert row['accepted'], 'quality failure; no further RHS or scaling'
    report['kernel_breakdown']=loop.bench(reps=3)
    report['total_seconds']=time.perf_counter()-start_all
    report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['memory_final']=jax.devices()[0].memory_stats()
    report['allocator_cap_passed']=bool(report['memory_final']['peak_bytes_in_use']<=args.allocation_cap_gib*2**30)
    report['all_accepted']=all(row['accepted'] for row in solve_records)
    report['checked_batch_solution_seconds']=report['checked_cold_first_solution_seconds']+sum(row['solve_plus_checks_seconds'] for row in solve_records[1:])
    report['engine_sha256_after']={str(f.relative_to(engine)):digest(f) for f in sorted(engine.rglob('*.py'))}
    report['engine_unchanged']=report['engine_sha256']==report['engine_sha256_after']
    report['gpu_processes']=subprocess.check_output([NVIDIA_SMI,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    report['single_metric_state_bytes']=state_storage_bytes(rows)+state_storage_bytes(columns)
    np.savez_compressed(out.with_name(out.stem+'_metrics.npz'),A=a,B=b,tau=tau,sigma=sigma)
    save();assert report['allocator_cap_passed'] and report['engine_unchanged'] and report['all_accepted']
    print(json.dumps({k:v for k,v in report.items() if k not in ('engine_sha256','solves')},indent=2),flush=True)
    print(json.dumps({'solves':[{k:v for k,v in row.items() if k!='history'} for row in solve_records]},indent=2),flush=True)


if __name__=='__main__':main()
