"""Generate and freeze Gate-3 data before scoring, on the assigned RTX 5090."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from attribution_heldout_fixture import array_sha, make_operator, independent_image_noise
from mixed_metric_f1 import host_operator


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture-dir', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--operator-id', required=True)
    parser.add_argument('--side', type=int, required=True)
    parser.add_argument('--development-control', action='store_true')
    parser.add_argument('--natural-assets')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', False)
    assert jax.devices()[0].platform == 'gpu' and '5090' in jax.devices()[0].device_kind
    protocol = json.loads(Path(args.protocol).read_text())
    assert protocol['protocol_version'] == 2
    operator = dict(next(x for x in protocol['operators'] if x['id'] == args.operator_id))
    regularization = protocol['regularization']
    image_seeds, noise_seeds = protocol['image_seeds'], protocol['noise_seeds']
    scope = 'held-out prescribed synthetic corpus; generation before scoring'
    if args.development_control:
        assert args.side == 16 and not args.natural_assets
        operator['operator_seed'] = 0
        operator['id'] += '_development_seed0'
        image_seeds, noise_seeds = [0], [1]
        scope = 'development-only channel/source control; outside held-out corpus'
    else:
        assert args.side in (protocol['small_side'], protocol['large_side'], 512)
    root = Path(args.fixture_dir)
    assert not root.exists(), 'frozen fixture directory already exists'
    root.mkdir(parents=True)
    started = time.perf_counter()
    op = make_operator(operator, args.side)
    matrix, frequency = np.asarray(op.M), np.asarray(op.H)
    np.savez_compressed(root / 'operator.npz', matrix=matrix, frequency=frequency)
    _, _, host_parts = host_operator(op)
    manifest = {'fixture_version': 1, 'protocol_sha256': file_sha(args.protocol),
        'scope': scope, 'side': args.side, 'channels': operator['channels'],
        'operator': operator, 'regularization': regularization,
        'operator_file': 'operator.npz', 'operator_file_sha256': file_sha(root / 'operator.npz'),
        'execution_operator_hashes': {'matrix': array_sha(matrix), 'frequency': array_sha(frequency)},
        'operator_hashes': {name: array_sha(value) for name, value in host_parts.items()},
        'sources': {p.name: file_sha(p) for p in Path(__file__).parent.glob('*.py')},
        'host': subprocess.check_output(['hostname'], text=True).strip(),
        'rng_contract': 'independent operator/image/noise streams; constructor RNG is separate',
        'cases': []}
    natural = None
    if args.natural_assets:
        from PIL import Image
        asset_root = Path(args.natural_assets)
        receipt = json.loads((asset_root / 'ASSET_RECEIPT.json').read_text())
        asset = asset_root / 'astronaut.png'
        assert file_sha(asset) == receipt['sha256'] == receipt['expected_registry_sha256']
        decoded = np.asarray(Image.open(asset))
        assert decoded.dtype == np.uint8 and decoded.shape == (512, 512, 3)
        assert args.side == 512 and operator['id'] == protocol['natural_image']['operator_id']
        natural = (decoded.astype(np.float32) / np.float32(255.)).transpose(2, 0, 1).reshape(-1)
        manifest['natural_image'] = {'asset_receipt': receipt, 'decoded_sha256': array_sha(decoded),
            'shape': list(decoded.shape), 'transform': 'float32 / 255 then channel-major; no resize'}
        image_seeds, noise_seeds = [None], [protocol['natural_image']['noise_seed']]
    apply = jax.jit(op.matvec)
    for case, (image_seed, noise_seed) in enumerate(zip(image_seeds, noise_seeds)):
        if natural is None:
            image, noise = independent_image_noise(args.side, operator['channels'], image_seed, noise_seed)
        else:
            image = natural
            noise = np.random.default_rng(noise_seed).normal(size=image.size)
        b = (np.asarray(apply(jnp.asarray(image))) + regularization['noise_standard_deviation'] * noise).astype(np.float32)
        file = root / f'case{case}.npz'
        np.savez_compressed(file, b=b, image=image, noise=noise)
        manifest['cases'].append({'case': case, 'file': file.name, 'file_sha256': file_sha(file),
            'input_sha256': array_sha(b), 'image_sha256': array_sha(image), 'noise_sha256': array_sha(noise),
            'image_seed': image_seed, 'noise_seed': noise_seed})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'fixture': str(root), 'manifest_sha256': file_sha(root / 'manifest.json'),
        'cases': len(manifest['cases']), 'scope': scope, 'seconds': time.perf_counter()-started,
        'memory': jax.devices()[0].memory_stats(), 'no_solver_or_metric_scoring': True}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    assert report['memory']['peak_bytes_in_use'] <= 8 * 2**30
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
