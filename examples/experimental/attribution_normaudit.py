"""Remote post-run saved-metric explanation, with no normalization changes."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from blockmetric_models import ChannelBlur, _mixing_matrix


def digest(value):return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--input-directory',default='results');args=p.parse_args()
    assert jax.devices()[0].platform=='gpu' and '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64',False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    op=ChannelBlur(1024,_mixing_matrix('moderate',0,3),.7)
    M32=np.asarray(op.M);H32=np.asarray(op.H)
    M=M32.astype(np.float64);H=H32.astype(np.complex128)
    verified={'matrix':digest(M),'frequency':digest(H)}
    arrays={'verified_M_float32':M32,'verified_H_stored_dtype':H32}
    rows=[]
    for source in sorted(Path(args.input_directory).glob('*L1024_repeat*_v1.json')):
        record=json.loads(source.read_text())
        if record['arm']=='S0':continue
        assert record['all_accepted']
        assert all(record['input_hashes'][k]==v for k,v in verified.items()),'coefficient reconstruction did not reproduce executed bytes'
        with np.load(source.with_name(source.stem+'_metrics.npz')) as data:
            sigma=data['sigma_shared'];tau=data['tau'].reshape(3,-1)
            A=data['A_shared'];B=data['B'].reshape(3,-1)
        assert np.max(np.ptp(tau,axis=1))==0 and np.max(np.ptp(B,axis=1))==0,'not spatially tied'
        tch=tau[:,0];bch=B[:,0]
        values,Q=np.linalg.eigh(sigma)
        if record['arm'].startswith('S'):
            actual_sigma=np.eye(3)*float(np.float32(sigma[0,0]));Q32=np.eye(3,dtype=np.float32)
            w32=np.full(3,np.float32(sigma[0,0]))
        else:
            Q32=Q.astype(np.float32);w32=values.astype(np.float32)
            actual_sigma=(Q32.astype(np.float64)*w32.astype(np.float64))@Q32.astype(np.float64).T
        actual_tau=tch.astype(np.float32).astype(np.float64)
        hmax=float(np.abs(H).max())
        def coupling(S,T):
            scaled=M*np.sqrt(T)[None,:]
            gram=scaled.T@S@scaled
            return float(np.sqrt(np.linalg.eigvalsh((gram+gram.T)/2)[-1])*hmax)
        unscaled=coupling(A,bch);intended=coupling(sigma,tch);rounded=coupling(actual_sigma,actual_tau)
        bound=record['normalization']['upper']
        row={'source':str(source),'arm':record['arm'],'unscaled_exact_structure_norm':unscaled,
            'executed_gershgorin_bound':bound,'bound_over_exact_structure_norm':bound/unscaled,
            'intended_double_steps_coupling_norm':intended,'reconstructed_rounded_steps_coupling_norm':rounded,
            'sigma_relative_frobenius_change':float(np.linalg.norm(actual_sigma-sigma)/np.linalg.norm(sigma)),
            'tau_relative_change':float(np.linalg.norm(actual_tau-tch)/np.linalg.norm(tch)),
            'norm_below_one':bool(rounded<1),
            'rounded_step_provenance':'source-faithful decomposition/casting reconstruction; original runtime Q/eigenvalue payload hashes were not saved'}
        rows.append(row)
        for name,value in [('intended_sigma',sigma),('intended_tau_channels',tch),('rounded_Q',Q32),
                           ('rounded_values',w32),('rounded_sigma_action',actual_sigma),('rounded_tau_channels',actual_tau)]:
            arrays[source.stem+'_'+name]=value
    report={'rows':rows,'coefficient_hashes_verified_against_every_run':verified,
        'coefficient_provenance':'recovered from the frozen generator, accepted only after exact recorded-byte SHA256 match; saved in NPZ',
        'fft_max_abs_imaginary':float(np.max(np.abs(H.imag))),
        'scope':'exact-structure numerical spectral calculation in float64, not a rigorous floating error upper bound; no new normalization, tuning, or solve',
        'excluded':'S0 has free spatial column scales; no k-by-k exact reduction claimed',
        'all_norms_below_one':all(row['norm_below_one'] for row in rows)}
    out.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**arrays)
    print(json.dumps(report,indent=2),flush=True);assert len(rows)==12 and report['all_norms_below_one']


if __name__=='__main__':main()
