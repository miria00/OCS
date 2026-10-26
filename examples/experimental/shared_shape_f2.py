"""Compact shared-metric F2 research gate, for the authorized RTX 5090 hosts.

The historical fixed-root proximal rule is deliberately retained for this
comparison. The adaptive outer-loop contract has its own named experiment.
No public OCS capability is added by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp
import ocs
from blockmetric_models import blockmetric_f2
from blockmetric_pilot import Loop, prox_l2_diag_any, PROX_IMPL
from mixed_metric_f1 import host_operator, construction_objective, normalization, prox_check, independent
from mixed_metric_gpu import BlockGroup
from shared_shape_gpu import SharedShapeGroup, initial_state, make_constructor


def mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


class SharedF2Loop:
    """Keep one eigenbasis and channel step vector, with optional mix fusion."""
    def __init__(self, spec, tau_channels, q, eigenvalues, *, fused=False, block=50):
        self.spec, self.block = spec, block
        op = spec.dual_blocks[0]['op']
        k, count, mu = op.c, op.N, spec.mu
        lam = float(spec.dual_blocks[0]['lam'])
        self.q = jnp.asarray(q, jnp.float32)
        self.values = jnp.asarray(eigenvalues, jnp.float32)
        self.tau = jnp.asarray(tau_channels, jnp.float32).reshape(k, 1)
        self.fused = fused
        self.mix = mm(self.q.T, op.M) if fused else op.M

        def step(state, off, q, values, tau, mix):
            x, y, xb = state
            if fused:
                r = mm(mix, op.blur(xb.reshape(k, op.L, op.L)).reshape(k, count)).T + off
            else:
                r = mm(op.matvec(xb).reshape(count, k) + off, q)
            v = y + values[None, :] * r
            yn = v - values[None, :] * prox_l2_diag_any(v / values[None, :], values[None, :], lam)
            if fused:
                aty = op.blur(mm(mix.T, yn.T).reshape(k, op.L, op.L)).reshape(-1)
            else:
                aty = op.rmatvec(mm(yn, q.T).reshape(-1))
            xn = ((x.reshape(k, count) - tau * aty.reshape(k, count)) / (1 + tau * mu)).reshape(-1)
            return xn, yn, 2 * xn - x

        self.run = jax.jit(lambda x, y, xb, off, q, w, tau, mix:
            jax.lax.fori_loop(0, block, lambda _, st: step(st, off, q, w, tau, mix), (x, y, xb)))
        self.assemble = jax.jit(lambda y, q: mm(y, q.T).reshape(-1))

    def offset(self, rhs):
        off = -jnp.asarray(rhs).reshape(-1, self.q.shape[0])
        return mm(off, self.q) if self.fused else off

    def init_state(self):
        x = jnp.zeros(self.spec.n, jnp.float32)
        return x, jnp.zeros((self.spec.n // self.q.shape[0], self.q.shape[0]), jnp.float32), x

    def block_step(self, state, off):
        return self.run(*state, off, self.q, self.values, self.tau, self.mix)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--side', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mixing', default='moderate')
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--changed-data', type=int, default=0)
    parser.add_argument('--solve-cap', type=int, default=10000)
    parser.add_argument('--allocation-cap-gib', type=float, default=8.)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--loop-control', action='store_true', help='Tiny-only expanded FULL and fusion equivalence checks')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', False)
    assert jax.devices()[0].platform == 'gpu' and '5090' in jax.devices()[0].device_kind
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    engine = Path(ocs.__file__).parent
    report = {'arguments': vars(args), 'host': subprocess.check_output(['hostname'], text=True).strip(),
        'gpu': subprocess.check_output([smi, '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip(),
        'jax': jax.__version__, 'iteration_dtype': 'float32', 'reference_dtype': 'host float64',
        'prox_impl': PROX_IMPL, 'arm': 'shared_fused' if args.fused else 'shared',
        'engine_sha256': {str(f.relative_to(engine)): sha(f) for f in sorted(engine.rglob('*.py'))},
        'sources': {f.name: sha(f) for f in Path(__file__).parent.glob('*.py')
                    if f.name.startswith(('shared_shape_', 'mixed_metric_', 'blockmetric_'))}}
    def save(): out.write_text(json.dumps(report, indent=2) + '\n')
    t = time.perf_counter()
    problem, spec = blockmetric_f2(args.side, mixing=args.mixing, seed=args.seed)
    op = spec.dual_blocks[0]['op']; data_b = np.asarray(spec.data['b'])
    forward, reverse, parts = host_operator(op)
    n, k, count = spec.n, op.c, op.N
    report['model_seconds'] = time.perf_counter() - t
    report['input_hashes'] = {'b': hashlib.sha256(data_b.tobytes()).hexdigest(),
        **{name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name, value in parts.items()}}
    rows = (SharedShapeGroup(k, count, amplitudes='tied'),)
    columns = (BlockGroup('scalar', count, k),)
    a0, b0 = initial_state(rows), initial_state(columns)
    program = make_constructor(op.matvec, op.rmatvec, rows, columns, iterations=args.updates,
                               probes=2, mu=1., box=2., shift=20)
    key = jax.random.PRNGKey(0)
    t = time.perf_counter(); lowered = program.lower(key, a0, b0); lowering = time.perf_counter() - t
    t = time.perf_counter(); executable = lowered.compile(); compiling = time.perf_counter() - t
    t = time.perf_counter(); answer = executable(key, a0, b0); jax.block_until_ready(answer); execution = time.perf_counter() - t
    report['constructor_finite'] = bool(answer['finite'])
    save(); assert report['constructor_finite'], 'constructor numerical failure'
    warm = []
    for _ in range(2):
        t = time.perf_counter(); repeat = executable(key, a0, b0); jax.block_until_ready(repeat); warm.append(time.perf_counter() - t)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
                   zip(jax.tree_util.tree_leaves(answer), jax.tree_util.tree_leaves(repeat)))
    report['construction_seconds'] = {'lower': lowering, 'compile': compiling, 'execute': execution,
        'cold_setup_plus_execution': lowering + compiling + execution, 'warm_replays': warm, 'warm_identical': True}
    report['construction_contract'] = {'mu': 1., 'M': 2., 'r': 1., 's': 1., 'updates': args.updates,
        'probes': 2, 'shift': 20, 'gradient_operator_calls': 4 * args.updates,
        'rows': 'one full matrix shared across all pixels', 'columns': 'one scalar per channel shared spatially',
        'averaging': 'pre-update geodesic', 'same_optimum_requires': 'joint circular-translation symmetry and positive ridge'}
    report['memory_after_construction'] = jax.devices()[0].memory_stats()
    assert report['memory_after_construction']['peak_bytes_in_use'] <= args.allocation_cap_gib * 2**30
    t = time.perf_counter()
    shared_a = np.asarray(answer['A'][0], np.float64)
    channel_b = np.exp(np.asarray(answer['B'][0], np.float64))
    # Host-only broadcast views for the SAME independent audit as existing arms.
    # No (N,k,k) metric is created on the GPU.
    a = np.broadcast_to(shared_a, (count, k, k))
    b = np.broadcast_to(channel_b[:, None], (k, count)).reshape(-1)
    logs = np.log(np.linalg.eigvalsh(shared_a))
    assert np.all(np.isfinite(logs)) and logs.min() >= -4.001 and logs.max() <= 4.001
    assert np.log(channel_b).min() >= -4.001 and np.log(channel_b).max() <= 4.001
    report['construction_objective'] = construction_objective(a, b, forward)
    report['construction_audit_seconds'] = time.perf_counter() - t
    t = time.perf_counter(); norm, note = normalization(a, b, parts); factor = .95 / norm
    report['normalization'] = {**note, 'factor': factor, 'seconds': time.perf_counter() - t}
    sigma, tau_channels = shared_a * factor, channel_b * factor
    t = time.perf_counter(); report['prox_audit'] = prox_check(sigma[None], True)
    report['prox_audit_seconds'] = time.perf_counter() - t
    save(); assert report['prox_audit']['passed'], 'actual-spectrum prox audit failed'
    t = time.perf_counter()
    comp = problem.compile(equilibrate='scalar', block=50, max_blocks=args.solve_cap // 50,
                           tol=1e-3, pdhg_variant='standard')
    report['public_certificate_setup_seconds'] = time.perf_counter() - t
    t = time.perf_counter(); values, q = np.linalg.eigh(sigma)
    loop = SharedF2Loop(spec, tau_channels, q, values, fused=args.fused)
    jax.block_until_ready((loop.q, loop.values, loop.tau, loop.mix))
    report['solver_metric_setup_seconds'] = time.perf_counter() - t
    if args.loop_control:
        assert args.side <= 32, 'expanded loop control is tiny-only'
        expanded = Loop(spec, jnp.asarray(np.repeat(tau_channels, count), jnp.float32),
            [('full', jnp.asarray(np.broadcast_to(q, (count, k, k)), jnp.float32),
              jnp.asarray(np.broadcast_to(values, (count, k)), jnp.float32))], block=50)
        compact = SharedF2Loop(spec, tau_channels, q, values, fused=False)
        fused = SharedF2Loop(spec, tau_channels, q, values, fused=True)
        x, ys, xb = expanded.init_state()
        ref = expanded._run_block(x, ys, xb, expanded.T, expanded.marrays, (jnp.asarray(-data_b),))
        ref = (ref[0], ref[1][0], ref[2])
        controls = {}
        for name, candidate in [('compact', compact), ('fused', fused)]:
            result = candidate.block_step(candidate.init_state(), candidate.offset(data_b))
            errors = [float(np.linalg.norm(np.asarray(v, np.float64) - np.asarray(w, np.float64)) /
                           max(1., np.linalg.norm(np.asarray(w, np.float64)))) for v, w in zip(result, ref)]
            controls[name] = errors
        report['loop_control_relative_l2_errors_after_50_updates'] = controls
        save(); assert max(max(v) for v in controls.values()) <= 1e-4, controls
    solves = []
    for case in range(1 + args.changed_data):
        rhs = data_b if case == 0 else (data_b + .05 * np.random.default_rng(900 + case).normal(size=n)).astype(np.float32)
        start = time.perf_counter(); t = time.perf_counter()
        off = loop.offset(rhs); jax.block_until_ready(off)
        prepare_s = time.perf_counter() - t
        state = loop.init_state(); raw = {'b': jnp.asarray(rhs)}
        history = []; iteration_s = check_s = 0.
        for index in range(args.solve_cap // 50):
            t = time.perf_counter(); state = loop.block_step(state, off); jax.block_until_ready(state)
            iteration_s += time.perf_counter() - t
            t = time.perf_counter(); y = loop.assemble(state[1], loop.q)
            cert = comp.certificate(state[0], y, raw); residual = max(cert['r_primal'], cert['r_dual'])
            quality = independent(state[0], y, rhs, forward, reverse, k, spec.mu, 1.) if residual <= 1e-3 else None
            check_s += time.perf_counter() - t
            history.append([50 * (index + 1), residual, None if quality is None else quality['relative_bracket'], time.perf_counter() - start])
            if quality is not None and quality['passed']: break
            if not np.isfinite(residual) or time.perf_counter() - start > 60: break
        t = time.perf_counter(); quality = independent(state[0], y, rhs, forward, reverse, k, spec.mu, 1.)
        check_s += time.perf_counter() - t
        row = {'case': case, 'input_sha256': hashlib.sha256(rhs.tobytes()).hexdigest(),
            'iterations': 50 * (index + 1), 'residual': float(residual), 'independent': quality,
            'accepted': bool(residual <= 1e-3 and quality['passed']), 'offset_preparation_seconds': prepare_s,
            'iteration_seconds_including_first_compile': iteration_s, 'check_and_transfer_seconds': check_s,
            'solve_plus_checks_seconds': time.perf_counter() - start, 'history': history}
        if case == 0:
            report['checked_cold_first_solution_seconds'] = sum(report[name] for name in (
                'model_seconds', 'construction_audit_seconds', 'prox_audit_seconds',
                'public_certificate_setup_seconds', 'solver_metric_setup_seconds')) + report['construction_seconds']['cold_setup_plus_execution'] + report['normalization']['seconds'] + row['solve_plus_checks_seconds']
        solves.append(row); report['solves'] = solves; save()
        np.savez_compressed(out.with_name(out.stem + f'_case{case}.npz'), x=np.asarray(state[0]), y=np.asarray(y), b=rhs)
    report['single_metric_state_bytes'] = 4 * (k * k + k)
    report['metric_storage_scope'] = 'constructor metric only, excluding averages/probes/workspaces/solver state'
    report['memory_final'] = jax.devices()[0].memory_stats()
    report['allocator_cap_passed'] = bool(report['memory_final']['peak_bytes_in_use'] <= args.allocation_cap_gib * 2**30)
    report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report['total_seconds'] = time.perf_counter() - started
    report['all_accepted'] = all(row['accepted'] for row in solves)
    report['timing_scope'] = 'checked research workflow; retained fixed-root prox; validation controls excluded from checked cold total'
    np.savez_compressed(out.with_name(out.stem + '_metrics.npz'), A_shared=shared_a, B_channels=channel_b,
                        sigma_shared=sigma, tau_channels=tau_channels)
    save(); print(json.dumps({'all_accepted': report['all_accepted'], 'solves': [
        {k: v for k, v in row.items() if k != 'history'} for row in solves]}, indent=2), flush=True)
    assert report['allocator_cap_passed'], 'allocator cap exceeded'
    if not report['all_accepted']: raise SystemExit(2)


if __name__ == '__main__':
    main()
