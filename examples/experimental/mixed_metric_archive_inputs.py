"""Freeze final synthetic operator coefficients, verifying original receipt hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
from blockmetric_models import blockmetric_f2
from mixed_metric_f1 import host_operator


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert '5090' in jax.devices()[0].device_kind
    out=Path(args.output);records=[]
    for mixing in ('strong','moderate'):
        prefix='f2_L256_'+mixing;receipt=out.parent/(prefix+'_configured_smfe_final.json')
        saved=json.loads(receipt.read_text())
        _,spec=blockmetric_f2(256,mixing=mixing,seed=0)
        op=spec.dual_blocks[0]['op'];_,_,parts=host_operator(op)
        hashes={name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name,value in parts.items()}
        hashes['b']=hashlib.sha256(np.asarray(spec.data['b']).tobytes()).hexdigest()
        assert hashes==saved['input_hashes'],(mixing,hashes,saved['input_hashes'])
        dest=out.parent/(prefix+'_operator_inputs.npz')
        np.savez_compressed(dest,M=np.asarray(op.M),H=np.asarray(op.H),b=np.asarray(spec.data['b']),
            side=np.asarray(256),channels=np.asarray(3))
        records.append({'mixing':mixing,'input_hashes_verified':True,'saved':dest.name,
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    out.write_text(json.dumps({'records':records,'passed':True,'numpy':np.__version__,'jax':jax.__version__},indent=2)+'\n')


if __name__=='__main__':main()
