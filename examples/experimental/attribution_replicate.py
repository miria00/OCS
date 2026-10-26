"""Gate-2 frozen-fixture variant of attribution_run.py; solver/constructor unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import ocs

from attribution_constructor import construct
from blockmetric_models import blockmetric_f2
from blockmetric_pilot import Loop
from shared_shape_f2 import SharedF2Loop
from mixed_metric_f1 import host_operator, construction_objective, normalization, prox_check, independent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Adapter:
    def __init__(self,spec,tau,sigma,weighted):
        self.weighted=weighted;self.count=spec.dual_blocks[0]['op'].N;self.k=spec.dual_blocks[0]['op'].c
        self.q,self.values=None,None
        if weighted:
            self.values,self.q=np.linalg.eigh(sigma)
            self.loop=SharedF2Loop(spec,tau.reshape(self.k,self.count)[:,0],self.q,self.values,fused=False)
        else:
            self.loop=Loop(spec,jnp.asarray(tau,jnp.float32),
                [('scalar',jnp.full((self.count,1),float(sigma[0,0]),jnp.float32))],block=50)

    def init(self):
        if self.weighted:return self.loop.init_state()
        x,ys,xb=self.loop.init_state();return x,ys[0],xb

    def offset(self,rhs):return self.loop.offset(rhs) if self.weighted else jnp.asarray(-rhs)

    def block(self,state,off):
        if self.weighted:return self.loop.block_step(state,off)
        x,y,xb=state
        x,ys,xb=self.loop._run_block(x,(y,),xb,self.loop.T,self.loop.marrays,(off,))
        return x,ys[0],xb

    def physical(self,state):
        return self.loop.assemble(state[1],self.loop.q) if self.weighted else state[1].reshape(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=['S0','S1','S2','D','F'],required=True)
    p.add_argument('--side',type=int,default=16);p.add_argument('--changed-data',type=int,default=4)
    p.add_argument('--solve-cap',type=int,default=10000);p.add_argument('--loop-control',action='store_true')
    p.add_argument('--replay',action='store_true');p.add_argument('--online-control',action='store_true')
    p.add_argument('--fixtures',required=True)
    args=p.parse_args();jax.config.update('jax_enable_x64',False)
    assert jax.devices()[0].platform=='gpu' and '5090' in jax.devices()[0].device_kind
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);startall=time.perf_counter()
    ft=time.perf_counter();fixture_root=Path(args.fixtures)
    fixture_meta=json.loads((fixture_root/'manifest.json').read_text())
    assert fixture_meta['side']==args.side and len(fixture_meta['cases'])==1+args.changed_data
    fixture_rhs=[]
    for entry in fixture_meta['cases']:
        with np.load(fixture_root/entry['file']) as z: rhs=z['b']
        assert rhs.dtype==np.float32 and rhs.shape==(3*args.side*args.side,)
        assert hashlib.sha256(rhs.tobytes()).hexdigest()==entry['input_sha256']
        fixture_rhs.append(rhs)
    fixture_loading_seconds=time.perf_counter()-ft
    core=Path(ocs.__file__).parent
    enginehash={str(f.relative_to(core)):sha(f) for f in sorted(core.rglob('*.py'))}
    report={'arguments':vars(args),'arm':args.arm,'host':subprocess.check_output(['hostname'],text=True).strip(),
        'gpu':subprocess.check_output([shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip(),
        'jax':jax.__version__,'iteration_dtype':'float32','reference_dtype':'float64 on same host',
        'engine_sha256':enginehash,'sources':{f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        'prox_contract':'native scalar closed form' if args.arm.startswith('S') else 'historical fixed 20 steps',
        'initialization':'zero each RHS; same K and metric, five distinct offsets',
        'normalization_contract':'common conservative Gershgorin bound and .95 safety'}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    t=time.perf_counter();problem,spec=blockmetric_f2(args.side,mixing='moderate',seed=0)
    op=spec.dual_blocks[0]['op'];data=fixture_rhs[0];k,count,n=op.c,op.N,spec.n
    forward,reverse,parts=host_operator(op)
    report['model_seconds']=time.perf_counter()-t
    report['input_hashes']={'b':hashlib.sha256(data.tobytes()).hexdigest(),**{
        name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name,value in parts.items()}}
    assert all(report['input_hashes'][key]==value for key,value in fixture_meta['operator_hashes'].items()), 'operator coefficient hash mismatch'
    report['fixture_manifest_sha256']=sha(fixture_root/'manifest.json')
    report['fixture_loading_seconds']=fixture_loading_seconds
    report['fixture_scope']='literal archived RHS; regenerated operator coefficients accepted only by exact recorded SHA256 equality; fixture I/O outside assembled solver timings'
    A,B,times,contract,generated=construct(op,args.arm)
    report['construction_seconds']=times;report['construction_contract']=contract
    if generated:(out.parent/(out.stem+'_generated_smfe.py')).write_text(generated)
    report['memory_after_construction']=jax.devices()[0].memory_stats();save()
    assert report['memory_after_construction']['peak_bytes_in_use']<=8*2**30,'constructor allocator ceiling'
    t=time.perf_counter();ab=np.broadcast_to(A,(count,k,k))
    eig=np.linalg.eigvalsh(A);logs=np.log(eig)
    report['metric_spectrum']={'A_eigenvalues':eig.tolist(),'B_min':float(B.min()),'B_max':float(B.max())}
    assert np.all(np.isfinite(B)) and np.all(np.abs(logs)<=4.001) and np.max(np.abs(np.log(B)))<=4.001
    report['construction_objective']=construction_objective(ab,B,forward)
    report['construction_audit_seconds']=time.perf_counter()-t
    t=time.perf_counter();norm,note=normalization(ab,B,parts);factor=.95/norm
    report['normalization']={**note,'factor':factor,'seconds':time.perf_counter()-t}
    sigma,tau=A*factor,B*factor
    t=time.perf_counter();report['prox_audit']=prox_check(sigma[None],not args.arm.startswith('S'))
    report['prox_audit_seconds']=time.perf_counter()-t;save()
    assert report['prox_audit']['passed'],'prox audit failure'
    t=time.perf_counter();comp=problem.compile(equilibrate='scalar',block=50,max_blocks=args.solve_cap//50,
        tol=1e-3,pdhg_variant='standard');report['public_certificate_setup_seconds']=time.perf_counter()-t
    t=time.perf_counter();loop=Adapter(spec,tau,sigma,not args.arm.startswith('S'))
    jax.block_until_ready(loop.init());report['solver_metric_setup_seconds']=time.perf_counter()-t
    np.savez_compressed(out.with_name(out.stem+'_metrics.npz'),A_shared=A,B=B,sigma_shared=sigma,tau=tau)
    if args.loop_control and loop.weighted:
        assert args.side<=32,'expanded representation control is bounded to tiny instances'
        expanded=Loop(spec,jnp.asarray(tau,jnp.float32),[('full',
            jnp.asarray(np.broadcast_to(loop.q,(count,k,k)),jnp.float32),
            jnp.asarray(np.broadcast_to(loop.values,(count,k)),jnp.float32))],block=50)
        ex=expanded.init_state();cx=loop.init();off=loop.offset(data)
        ex=expanded._run_block(*ex,expanded.T,expanded.marrays,(jnp.asarray(-data),))
        cx=loop.block(cx,off)
        errors=[float(np.linalg.norm(np.asarray(a,np.float64)-np.asarray(b,np.float64))/max(1.,np.linalg.norm(np.asarray(b,np.float64))))
            for a,b in zip((cx[0],loop.physical(cx),cx[2]),(ex[0],expanded.returned_pair(ex[0],ex[1]),ex[2]))]
        report['literal_metric_expanded_compact_50update_errors']=errors;save()
        assert max(errors)<=1e-4,'representation algebra failure'
    report['solves']=[]
    for case in range(1+args.changed_data):
        rhs=fixture_rhs[case]
        start=time.perf_counter();t=time.perf_counter();off=loop.offset(rhs);jax.block_until_ready(off)
        prepare=time.perf_counter()-t;state=loop.init();raw={'b':jnp.asarray(rhs)}
        history=[];iterate_s=check_s=0.;quality=None
        for index in range(args.solve_cap//50):
            t=time.perf_counter();state=loop.block(state,off);jax.block_until_ready(state);iterate_s+=time.perf_counter()-t
            t=time.perf_counter();y=loop.physical(state);cert=comp.certificate(state[0],y,raw)
            residual=max(cert['r_primal'],cert['r_dual'])
            quality=independent(state[0],y,rhs,forward,reverse,k,spec.mu,1.) if residual<=1e-3 else None
            check_s+=time.perf_counter()-t
            history.append([50*(index+1),residual,None if quality is None else quality['relative_bracket'],time.perf_counter()-start])
            if quality is not None and quality['passed']:break
            if not np.isfinite(residual) or time.perf_counter()-start>60:break
        t=time.perf_counter();quality=independent(state[0],y,rhs,forward,reverse,k,spec.mu,1.)
        check_s+=time.perf_counter()-t
        row={'case':case,'input_sha256':hashlib.sha256(rhs.tobytes()).hexdigest(),'iterations':50*(index+1),
            'residual':float(residual),'independent':quality,'accepted':bool(residual<=1e-3 and quality['passed']),
            'offset_preparation_seconds':prepare,'iteration_seconds_including_first_compile':iterate_s,
            'check_and_transfer_seconds':check_s,'solve_plus_checks_seconds':time.perf_counter()-start,'history':history}
        if case==0:
            report['checked_cold_first_solution_seconds']=sum(report[x] for x in
                ('model_seconds','construction_audit_seconds','prox_audit_seconds','public_certificate_setup_seconds','solver_metric_setup_seconds'))+times['cold_setup_plus_execution']+report['normalization']['seconds']+row['solve_plus_checks_seconds']
        report['solves'].append(row);save()
        np.savez_compressed(out.with_name(out.stem+f'_case{case}.npz'),x=np.asarray(state[0]),y=np.asarray(y),b=rhs)
        assert row['accepted'],'quality failure stops this candidate before further cases/scaling'
        # Replay at the already accepted count, with audits strictly outside timing.
        if args.replay:
            blocks=row['iterations']//50
            replay=jax.jit(lambda st,ofs:jax.lax.fori_loop(0,blocks,lambda _,carry:loop.block(carry,ofs),st))
            initial=loop.init();t=time.perf_counter();executable=replay.lower(initial,off).compile();compile_s=time.perf_counter()-t
            replay_times=[]
            for _ in range(3):
                t=time.perf_counter();rs=executable(initial,off);jax.block_until_ready(rs);replay_times.append(time.perf_counter()-t)
            ry=loop.physical(rs);rq=independent(rs[0],ry,rhs,forward,reverse,k,spec.mu,1.)
            rc=comp.certificate(rs[0],ry,raw);rr=max(rc['r_primal'],rc['r_dual'])
            err=float(np.linalg.norm(np.asarray(rs[0],np.float64)-np.asarray(state[0],np.float64))/max(1.,np.linalg.norm(np.asarray(state[0],np.float64))))
            row['audit_free_fixedcount_replay']={'seconds':replay_times,'separate_compile_seconds':compile_s,
                'relative_primal_error':err,'residual':float(rr),'independent':rq,
                'scope':'throughput at preselected accepted count; not an online stopping solver'}
            save();assert rq['passed'] and rr<=1e-3 and err<=1e-4,'replay endpoint failure'
        # The same research loop with public certificate stopping alone. This
        # honestly exposes if that criterion stops before independent accuracy.
        if args.online_control and case==0:
            os=loop.init();online_start=time.perf_counter();phase=checks=0.
            for oi in range(args.solve_cap//50):
                t=time.perf_counter();os=loop.block(os,off);jax.block_until_ready(os);phase+=time.perf_counter()-t
                t=time.perf_counter();oy=loop.physical(os);oc=comp.certificate(os[0],oy,raw)
                ores=max(oc['r_primal'],oc['r_dual']);checks+=time.perf_counter()-t
                if ores<=1e-3 or not np.isfinite(ores) or time.perf_counter()-online_start>60:break
            online_s=time.perf_counter()-online_start
            oq=independent(os[0],oy,rhs,forward,reverse,k,spec.mu,1.)
            row['public_certificate_online_research_loop']={'iterations':50*(oi+1),'residual':float(ores),
                'iteration_seconds':phase,'check_seconds':checks,'total_seconds':online_s,
                'independent_final_check_outside_timer':oq,
                'public_full_metric_solver_api':'not supported; this is the research loop using the native certificate',
                'quality_gate_applied_during_stopping':False}
            np.savez_compressed(out.with_name(out.stem+f'_online_case{case}.npz'),x=np.asarray(os[0]),y=np.asarray(oy),b=rhs)
            save()
    report['engine_sha256_after']={str(f.relative_to(core)):sha(f) for f in sorted(core.rglob('*.py'))}
    report['engine_unchanged']=enginehash==report['engine_sha256_after']
    report['memory_final']=jax.devices()[0].memory_stats()
    report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['total_seconds']=time.perf_counter()-startall
    report['all_accepted']=all(s['accepted'] for s in report['solves'])
    report['checked_five_solution_seconds']=report['checked_cold_first_solution_seconds']+sum(s['solve_plus_checks_seconds'] for s in report['solves'][1:])
    report['timing_scope']='assembled checked research workflow; excludes replay/control/audits of replay, metadata and artifact IO; not public API latency'
    save();assert report['engine_unchanged'] and report['memory_final']['peak_bytes_in_use']<=8*2**30
    print(json.dumps({'arm':args.arm,'side':args.side,'all_accepted':report['all_accepted'],
        'updates':[s['iterations'] for s in report['solves']],
        'checked_total':report['checked_five_solution_seconds'],'total_process':report['total_seconds']}),flush=True)


if __name__=='__main__':main()
