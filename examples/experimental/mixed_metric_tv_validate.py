"""Independent tiny stencil/norm control, KEVIN only; no global large matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from mixed_metric_f1 import make_rof,host_operator,normalization


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert '5090' in jax.devices()[0].device_kind
    _,_,op,_=make_rof(3,0);forward,reverse,parts=host_operator(op);rng=np.random.default_rng(842)
    k=np.column_stack([forward(np.eye(27)[:,i]) for i in range(27)]);cases=[]
    for trial in range(3):
        x=rng.normal(size=27).astype(np.float32);y=rng.normal(size=81).astype(np.float32)
        ef=np.linalg.norm(np.asarray(op.matvec(jnp.asarray(x)))-forward(x))/max(1.,np.linalg.norm(forward(x)))
        er=np.linalg.norm(np.asarray(op.rmatvec(jnp.asarray(y)))-reverse(y))/max(1.,np.linalg.norm(reverse(y)))
        assert max(ef,er)<1e-6
        z=rng.normal(size=(27,3,3));a=np.einsum('bij,bkj->bik',z,z)+.2*np.eye(3);b=np.exp(rng.normal(size=27)*.3)
        if trial==0:a=np.broadcast_to(np.eye(3),(27,3,3)).copy();b=np.ones(27)
        bound,note=normalization(a,b,parts);v,q=np.linalg.eigh(a)
        half=np.einsum('bij,bj,bkj->bik',q,np.sqrt(v),q)
        scaled=np.einsum('bij,bjk->bik',half,k.reshape(27,3,27))*np.sqrt(b)[None,None,:]
        exact=float(np.linalg.norm(scaled.reshape(81,27),2));assert bound>=exact
        if trial==0:assert abs(bound-np.sqrt(12))<1e-10
        cases.append({'forward_relative_error':float(ef),'adjoint_relative_error':float(er),
            'analytic_bound':bound,'dense_tiny_reference_norm':exact,'passed':True})
    report={'matrix_shape':[81,27],'cases':cases,'passed':True,
        'scope':'dense matrix used only in independent tiny norm check',
        'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('mixed_metric_*.py')}}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
