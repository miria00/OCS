"""Freeze the prespecified bounded F1 control on an authorized remote 5090."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mixed_metric_f1 import make_problem, host_operator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--fixture-dir', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--section', choices=['development', 'scored'], required=True)
    args = p.parse_args()
    jax.config.update('jax_enable_x64', False)
    assert jax.devices()[0].platform == 'gpu' and '5090' in jax.devices()[0].device_kind
    protocol = json.loads(Path(args.protocol).read_text())
    cfg = protocol[args.section]
    root = Path(args.fixture_dir)
    assert not root.exists()
    root.mkdir(parents=True)
    _, _, op, data = make_problem(cfg['n'], cfg['seed'], cfg['rho'])
    _, _, parts = host_operator(op)
    fields = ('perm', 'inv', 'signs', 'scale', 'rho', 'one_minus_rho')
    coefficients = {f'partition_{i}_{name}': np.asarray(value)
        for i, part in enumerate(parts) for name, value in zip(fields, part)}
    np.savez_compressed(root / 'operator.npz', **coefficients)
    manifest = {'version': 1, 'section': args.section, 'settings': cfg,
        'protocol_sha256': digest(args.protocol),
        'operator_hashes': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in coefficients.items()},
        'operator_file_sha256': digest(root / 'operator.npz'), 'cases': [],
        'scope': 'existing overlapping-partition fixture; regenerated coefficients must match every archived hash; literal RHS loaded by every arm'}
    for case in range(1 + cfg['changed_data']):
        rhs = data if case == 0 else (data + .05*np.random.default_rng(900+case).normal(size=op.shape[0])).astype(np.float32)
        file = root / f'case{case}.npz'
        np.savez_compressed(file, b=rhs)
        manifest['cases'].append({'case': case, 'file': file.name, 'file_sha256': digest(file),
            'input_sha256': hashlib.sha256(rhs.tobytes()).hexdigest(),
            'additional_noise_seed': None if case == 0 else 900+case})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = {'fixture': str(root), 'manifest_sha256': digest(root / 'manifest.json'),
        'no_metric_or_solver_scoring': True, 'memory': jax.devices()[0].memory_stats()}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    assert result['memory']['peak_bytes_in_use'] <= 2*2**30


if __name__ == '__main__':
    main()
