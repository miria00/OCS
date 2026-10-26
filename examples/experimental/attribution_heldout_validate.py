"""Unscored development controls for Gate-3 channel counts and literal inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from attribution_heldout_fixture import independent_image_noise, make_operator, load_fixture, problem_from_loaded, array_sha
from attribution_constructor import diagonal_row_update, make_diagonal_constructor
from blockmetric_models import blockmetric_f2
from blockmetric_pilot import NEWTON_ITERS, PROX_IMPL, prox_l2_diag_any
from mixed_metric_f1 import host_operator, prox_check
from mixed_metric_gpu import BlockGroup
import shared_shape_gpu as shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture-root', required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', False)
    assert jax.devices()[0].platform == 'gpu' and '5090' in jax.devices()[0].device_kind
    assert NEWTON_ITERS == 20 and PROX_IMPL == 'components'
    out = Path(args.output)
    report = {'checks': [], 'scope': 'development L16 seed0 only; no held-out scoring',
        'root_steps': NEWTON_ITERS, 'prox_impl': PROX_IMPL}
    arrays = {}
    started = time.perf_counter()
    def check(name, passed, **data):
        report['checks'].append({'name': name, 'passed': bool(passed), **data})
        report['seconds'] = time.perf_counter()-started
        out.write_text(json.dumps(report, indent=2)+'\n')
        np.savez(out.with_suffix('.npz'), **arrays)
        print(json.dumps(report['checks'][-1]), flush=True)
        assert passed, name
    for k in (3, 5, 6):
        config = {'channels': k, 'mixing': 'moderate', 'operator_seed': 0, 'sigma_blur': .7}
        op = make_operator(config, 16)
        image, noise = independent_image_noise(16, k, 0, 1)
        changed_image, same_noise = independent_image_noise(16, k, 2, 1)
        same_image, changed_noise = independent_image_noise(16, k, 0, 3)
        check(f'k{k} independent image/noise streams',
            np.array_equal(noise, same_noise) and np.array_equal(image, same_image)
            and not np.array_equal(image, changed_image) and not np.array_equal(noise, changed_noise))
        _, old = blockmetric_f2(16, mixing='moderate', seed=0, channels=k)
        oldop = old.dual_blocks[0]['op']
        check(f'k{k} generator preserves frozen operator coefficients',
            np.array_equal(op.M, oldop.M) and np.array_equal(op.H, oldop.H))
        root = Path(args.fixture_root) / f'k{k}_development'
        manifest, matrix, frequency, rhs = load_fixture(root)
        problem, spec = problem_from_loaded(manifest, matrix, frequency, rhs[0])
        frozen = spec.dual_blocks[0]['op']
        check(f'k{k} literal fixture coefficient load',
            np.array_equal(frozen.M, op.M) and np.array_equal(frozen.H, op.H)
            and manifest['cases'][0]['image_sha256'] == array_sha(image)
            and manifest['cases'][0]['noise_sha256'] == array_sha(noise))
        forward, reverse, _ = host_operator(frozen)
        rng = np.random.default_rng(41)
        x, y = rng.normal(size=op.shape[1]), rng.normal(size=op.shape[0])
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        ax = np.asarray(jax.jit(frozen.matvec)(jnp.asarray(xf)), np.float64)
        aty = np.asarray(jax.jit(frozen.rmatvec)(jnp.asarray(yf)), np.float64)
        ferror = float(np.linalg.norm(ax-forward(xf))/max(1., np.linalg.norm(forward(xf))))
        rerror = float(np.linalg.norm(aty-reverse(yf))/max(1., np.linalg.norm(reverse(yf))))
        adj = float(abs(ax @ yf - xf @ aty) / max(1., abs(ax @ yf), abs(xf @ aty)))
        check(f'k{k} forward adjoint versus rounded-data float64',
            max(ferror, rerror, adj) < 5e-6, forward_error=ferror, reverse_error=rerror, adjoint_error=adj)
        shiftx = np.roll(xf.reshape(k, 16, 16), 1, axis=1).reshape(-1)
        translated = np.asarray(jax.jit(frozen.matvec)(jnp.asarray(shiftx)))
        expected = np.roll(ax.reshape(16, 16, k), 1, axis=0).reshape(-1)
        error = float(np.linalg.norm(translated-expected)/max(1., np.linalg.norm(expected)))
        check(f'k{k} circular translation layout', error < 5e-6, relative_error=error)
        key = jax.random.PRNGKey(0)
        _, rk, _ = jax.random.split(key, 3)
        xi = jax.random.rademacher(rk, (2, op.shape[1]), dtype=jnp.float32)
        samples = jax.vmap(frozen.matvec)(xi)
        loga = np.linspace(-.1, .2, k).astype(np.float32)
        energy = np.mean(np.asarray(samples, np.float64).reshape(2, op.N, k)**2, axis=(0, 1))
        reference = np.clip(loga-.2*(.5*np.exp(loga.astype(np.float64))*energy-.5+loga/4), -4, 4)
        answer, finite = diagonal_row_update(jnp.asarray(loga), samples, eta=.2)
        error = float(np.max(np.abs(np.asarray(answer)-reference)))
        check(f'k{k} pooled constrained diagonal multiplicity', bool(finite) and error < 2e-6, max_error=error)
        rows, columns = (shared.SharedShapeGroup(k, op.N),), (BlockGroup('scalar', op.N, k),)
        full = shared.make_constructor(frozen.matvec, frozen.rmatvec, rows, columns, iterations=1)(
            key, shared.initial_state(rows), shared.initial_state(columns))
        diag = make_diagonal_constructor(frozen, iterations=1)(key, jnp.zeros(k), jnp.zeros(k))
        error = float(np.max(np.abs(np.asarray(full['last_B'][0])-np.asarray(diag['last_B'][0]))))
        check(f'k{k} D/F common probes and simultaneous column state', error < 3e-6, max_error=error)
        sigma = np.diag(np.geomspace(.4, 2., k))[None]
        prox = prox_check(sigma, True)
        check(f'k{k} fixed20 weighted prox independent root', prox['passed'], result=prox)
        rotation, _ = np.linalg.qr(rng.normal(size=(k,k)))
        coupled = (rotation*np.geomspace(.4,2.,k))@rotation.T
        coupled_audit = prox_check(coupled[None], True)
        offdiag = float(np.linalg.norm(coupled-np.diag(np.diag(coupled))))
        check(f'k{k} non-diagonal SPD fixed20 spectrum control',
            coupled_audit['passed'] and offdiag>1e-3, result=coupled_audit, offdiagonal_frobenius=offdiag)
        weights, q = np.linalg.eigh(coupled)
        vectors = rng.normal(size=(6,k))
        vectors *= np.asarray([0.,1e-7,.3,1.,2.,30.])[:,None]/np.linalg.norm(vectors,axis=1)[:,None]
        vectors = vectors.astype(np.float32)
        @jax.jit
        def physical_project(v, basis, d):
            z = jnp.matmul(v,basis,precision=jax.lax.Precision.HIGHEST)
            z = z-d*prox_l2_diag_any(z/d,d,1.)
            return jnp.matmul(z,basis.T,precision=jax.lax.Precision.HIGHEST)
        actual = np.asarray(physical_project(jnp.asarray(vectors),jnp.asarray(q,jnp.float32),
            jnp.asarray(weights[None],jnp.float32)),np.float64)
        reference = []
        for v in vectors.astype(np.float64):
            if np.linalg.norm(v)<=1.:
                reference.append(v)
            else:
                z=q.T@v
                fn=lambda t:np.linalg.norm(z/(1+t*weights))-1.
                upper=1.
                while fn(upper)>0:upper*=2
                t=brentq(fn,0.,upper,xtol=1e-13)
                reference.append(q@(z/(1+t*weights)))
        reference=np.asarray(reference)
        error=float(np.max(np.linalg.norm(actual-reference,axis=1)/np.maximum(1.,np.linalg.norm(vectors,axis=1))))
        violation=float(max(0.,np.linalg.norm(actual,axis=1).max()-1.))
        check(f'k{k} physical non-diagonal SPD fixed20 projection',
            error<=1e-5 and violation<=1e-5,relative_projection_error=error,ball_violation=violation,
            scope='float64 eigendecomposition reference versus float32 runtime Q/eigenvalues; bounded tolerance check, not an interval proof or exact orthogonality claim')
        arrays[f'k{k}_coupled_sigma']=coupled
        arrays[f'k{k}_coupled_v']=vectors
        arrays[f'k{k}_coupled_actual']=actual
        arrays[f'k{k}_coupled_reference']=reference
        arrays[f'k{k}_b'] = rhs[0]
        arrays[f'k{k}_M'] = matrix
        arrays[f'k{k}_H'] = frequency
    report['all_passed'] = True
    report['memory'] = jax.devices()[0].memory_stats()
    np.savez(out.with_suffix('.npz'), **arrays)
    out.write_text(json.dumps(report, indent=2)+'\n')
    assert report['memory']['peak_bytes_in_use'] <= 8*2**30


if __name__ == '__main__':
    main()
