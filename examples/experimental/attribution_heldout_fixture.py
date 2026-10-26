"""Frozen-input F2 fixtures for Gate 3; numerical use is remote-5090 only.

This module separates operator, image and noise RNG streams. It does not
estimate metrics or score solver outcomes. The solver loads literal operator
coefficients and RHS values, rather than regenerating data while benchmarking.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ocs
from ocs.ir import ProblemFamily, Term
from blockmetric_models import ChannelBlur, Spec, _mixing_matrix, _test_image


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def make_operator(configuration, side):
    k = configuration['channels']
    if k not in (3, 5, 6) or side < 2:
        raise ValueError('declared channel count and positive image side required')
    mixing = _mixing_matrix(configuration['mixing'], configuration['operator_seed'], k)
    return ChannelBlur(side, mixing, configuration['sigma_blur'])


def independent_image_noise(side, channels, image_seed, noise_seed):
    image = _test_image(side, np.random.default_rng(image_seed), channels).astype(np.float32).reshape(-1)
    noise = np.random.default_rng(noise_seed).normal(size=image.size)
    return image, noise


def load_fixture(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['fixture_version'] == 1
    k, side = manifest['channels'], manifest['side']
    n = k * side * side
    assert k in (3, 5, 6) and side >= 2
    assert [row['case'] for row in manifest['cases']] == list(range(len(manifest['cases'])))
    operator_path = root / manifest['operator_file']
    assert hashlib.sha256(operator_path.read_bytes()).hexdigest() == manifest['operator_file_sha256']
    with np.load(operator_path, allow_pickle=False) as values:
        matrix, frequency = values['matrix'].copy(), values['frequency'].copy()
    assert matrix.dtype == np.float32 and matrix.shape == (k, k)
    assert frequency.dtype == np.complex64 and frequency.shape == (side, side)
    assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(frequency))
    assert set(manifest['execution_operator_hashes']) == {'matrix', 'frequency'}
    assert {'matrix': array_sha(matrix), 'frequency': array_sha(frequency)} == manifest['execution_operator_hashes']
    rhs = []
    for row in manifest['cases']:
        file = root / row['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['file_sha256']
        with np.load(file, allow_pickle=False) as values:
            b = values['b'].copy()
        assert b.dtype == np.float32 and b.shape == (n,) and np.all(np.isfinite(b))
        assert array_sha(b) == row['input_sha256']
        rhs.append(b)
    return manifest, matrix, frequency, rhs


def problem_from_loaded(manifest, matrix, frequency, first_rhs):
    """Use literal M/H in both forward and adjoint; no exact-Gram callback."""
    k, side = manifest['channels'], manifest['side']
    regularization = manifest['regularization']
    mu, lam = regularization['mu'], regularization['lambda']
    op = ChannelBlur(side, matrix, manifest['operator']['sigma_blur'])
    op.M, op.H = jnp.asarray(matrix), jnp.asarray(frequency)
    n = k * side * side
    problem = ProblemFamily(n, [Term(ocs.Identity(n), ocs.SumSquares().scaled(mu)),
        Term(op, ocs.GroupL2(lam=lam, group_size=k), lambda data: -data['b'])])
    spec = Spec(n=n, mu=mu, primal='ridge', data={'b': jnp.asarray(first_rhs)},
        dual_blocks=[{'op': op, 'kind': 'group_l2', 'k': k, 'lam': lam,
                      'offset': jnp.asarray(-first_rhs)}],
        family='F2', name=manifest['operator']['id'], fixture_only=True)
    return problem, spec
