"""Bounded constructor-only timing/Nsight capture; no solve or quality claim."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
from mixed_metric_f1 import make_problem
from blockmetric_models import blockmetric_f2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--source',required=True);p.add_argument('--family',choices=['f1','f2'],default='f2')
    p.add_argument('--n',type=int,default=8192);p.add_argument('--side',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--replays',type=int,default=3)
    p.add_argument('--capture',action='store_true');args=p.parse_args()
    assert '5090' in jax.devices()[0].device_kind
    spec=importlib.util.spec_from_file_location('selected_constructor',args.source)
    mod=importlib.util.module_from_spec(spec);sys.modules['selected_constructor']=mod;spec.loader.exec_module(mod)
    if args.family=='f1':_,problem,op,_=make_problem(args.n,0,.9)
    else:_,problem=blockmetric_f2(args.side,mixing='strong',seed=0);op=problem.dual_blocks[0]['op']
    k=problem.dual_blocks[0]['k'];m,n=op.shape
    rows=(mod.BlockGroup('full',k,m//k),);cols=(mod.BlockGroup('diagonal',n),)
    a,b=mod.initial_state(rows),mod.initial_state(cols);key=jax.random.PRNGKey(0)
    program=mod.make_constructor(op.matvec,op.rmatvec,rows,cols,iterations=80,probes=2,batch_size=args.batch_size)
    t=time.perf_counter();lowered=program.lower(key,a,b);lower_s=time.perf_counter()-t
    t=time.perf_counter();executable=lowered.compile();compile_s=time.perf_counter()-t
    t=time.perf_counter();first=executable(key,a,b);jax.block_until_ready(first);first_s=time.perf_counter()-t
    assert bool(first['finite'])
    runtime=None
    if args.capture:
        libs=list((Path(sys.prefix)/'lib').glob('python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.*'))
        assert libs,'CUDA profiler library unavailable'
        runtime=ctypes.CDLL(str(libs[0]))
        assert runtime.cudaProfilerStart()==0
    times=[];deltas=[]
    for _ in range(args.replays):
        t=time.perf_counter();answer=executable(key,a,b);jax.block_until_ready(answer);times.append(time.perf_counter()-t)
        assert bool(answer['finite'])
        deltas.append(max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(
            jax.tree_util.tree_leaves((first['A'],first['B'])),jax.tree_util.tree_leaves((answer['A'],answer['B'])))))
    if runtime is not None:assert runtime.cudaProfilerStop()==0
    assert max(deltas)==0
    memory=executable.memory_analysis()
    result={'arguments':vars(args),'device':str(jax.devices()[0]),'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        'lower_seconds':lower_s,'compile_seconds':compile_s,'first_execution_seconds':first_s,
        'warm_seconds':times,'warm_bitidentical':True,'capture_succeeded':runtime is not None,
        'allocator_memory':jax.devices()[0].memory_stats(),
        'executable_memory':{name:getattr(memory,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
        'metric_input_shape':{'n':n,'m':m,'k':k}}
    out=Path(args.output);out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),A=np.asarray(first['A'][0]),B=np.asarray(first['B'][0]))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
