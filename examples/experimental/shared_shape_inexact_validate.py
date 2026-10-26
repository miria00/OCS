"""Bounded Mazeppa-only integration controls for adaptive inexact PDHG.

Float32 GPU updates, independent CPU float64 references on that same host.
No solver speed comparison and no finite-precision convergence theorem.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import ocs
from ocs.ir import ProblemFamily, Term

from inexact_full_ball_prox import prepare_metric
from shared_shape_inexact_loop import AcceptancePolicy, Group, InexactLoop


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def make_fixture():
    rng = np.random.default_rng(20260915)
    mats = [np.asarray([[1., .2, -.3, .1], [.1, .8, 0., -.2],
                        [-.2, .1, 1., .3], [.2, -.1, .1, .7]], np.float32) * .5,
            (rng.normal(size=(3, 4)) * .35).astype(np.float32)]
    rhs = [np.asarray([.7, -.4, .5, .3], np.float32),
           np.asarray([-.3, .6, .2], np.float32)]
    shapes = [(2, 2), (1, 3)]
    radii = [np.full(2, .3, np.float32), np.full(1, .25, np.float32)]
    q1 = []
    for angle in (.3, -.4):
        c, s = math.cos(angle), math.sin(angle)
        q1.append([[c, -s], [s, c]])
    qs = [np.asarray(q1, np.float32),
          np.linalg.qr(rng.normal(size=(3, 3)))[0][None].astype(np.float32)]
    sigmas = [np.asarray([[.7, 1.3], [.9, 1.1]], np.float32),
              np.asarray([[.6, 1., 1.4]], np.float32)]
    physical = [(q.astype(float) * s.astype(float)[:, None, :]) @ q.astype(float).transpose(0, 2, 1)
                for q, s in zip(qs, sigmas)]
    matrix = np.vstack(mats).astype(float)
    sigma = scipy.linalg.block_diag(*(S for group in physical for S in group))
    energy = float(np.trace(matrix.T @ sigma @ matrix))
    tau = np.full(4, np.float32(.81 / energy), np.float32)
    upper = float(math.sqrt(float(tau[0]) * energy) * (1 + 1e-6))
    measured_norm = float(math.sqrt(np.linalg.eigvalsh(float(tau[0]) * matrix.T @ sigma @ matrix).max()))
    assert measured_norm <= upper < 1
    groups = []
    for i, (K, b, shape, rad, q, s) in enumerate(zip(mats, rhs, shapes, radii, qs, sigmas)):
        kj = jnp.asarray(K)
        groups.append(Group(name=f"group{i}_dim{shape[1]}",
            matvec=lambda x, K=kj: K @ x, rmatvec=lambda y, K=kj: K.T @ y,
            metric=prepare_metric(q, s), radius=jnp.asarray(rad),
            offset=jnp.asarray(-b.reshape(shape)),
            operator_digest=hashlib.sha256(K.tobytes()).hexdigest()))
    return groups, tau, dict(K=mats, b=rhs, shapes=shapes, radius=radii,
        Q=qs, sigma=sigmas, physical_sigma=physical, mu=.7,
        coupling_upper=upper, exact_tiny_norm_float64=measured_norm,
        coupling_provenance="Analytical Frobenius upper bound evaluated in float64 on actual rounded Sigma/T, 1e-6 margin; not interval enclosure")


def declared_acceptance_identity():
    """Explicit evaluator provenance, not automatic semantic callback verification."""
    core = Path(ocs.__file__).parent
    declared = dict(harness=digest(__file__),
        ocs_source={str(p.relative_to(core)): digest(p) for p in sorted(core.rglob('*.py'))})
    return hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()


def build_loop(groups, tau, fixture, *, c=.03, p=.25, max_inner=80,
               policy_identity=None, check_every=5, implementation_version="shared-shape-inexact-v2"):
    return InexactLoop(groups, tau, mu=fixture['mu'], budget_c=c, budget_p=p,
        coupling_upper=fixture['coupling_upper'],
        coupling_provenance=fixture['coupling_provenance'], max_inner=max_inner,
        acceptance_policy_identity=policy_identity or declared_acceptance_identity(),
        check_every=check_every, implementation_version=implementation_version)


def make_evaluator(groups, fixture):
    terms = [Term(ocs.Identity(4), ocs.SumSquares().scaled(fixture['mu']))]
    for K, g, shape, radius in zip(fixture['K'], groups, fixture['shapes'], fixture['radius']):
        terms.append(Term(ocs.Dense(jnp.asarray(K)),
            ocs.GroupL2(lam=float(radius[0]), group_size=shape[1]),
            lambda data, off=g.offset: off.reshape(-1)))
    family = ProblemFamily(4, terms)
    compiled = family.compile(equilibrate='scalar', pdhg_variant='standard',
                              block=5, max_blocks=1, tol=1e-3)
    xx = cp.Variable(4)
    expression = fixture['mu'] / 2 * cp.sum_squares(xx)
    for K, b, shape, radius in zip(fixture['K'], fixture['b'], fixture['shapes'], fixture['radius']):
        for j in range(shape[0]):
            span = slice(j * shape[1], (j + 1) * shape[1])
            expression += float(radius[j]) * cp.norm(K[span].astype(float) @ xx - b[span].astype(float), 2)
    reference = cp.Problem(cp.Minimize(expression))
    reference.solve(solver='CLARABEL', tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                    tol_feas=1e-11, max_iter=200)
    if reference.status != 'optimal':
        raise RuntimeError(f"Independent reference failed: {reference.status}")

    def evaluate(x, ys):
        yy = jnp.concatenate([y.reshape(-1) for y in ys])
        certificate = compiled.certificate(x, yy, {})
        hostx, hostys = np.asarray(x, float), [np.asarray(y, float) for y in ys]
        primal = fixture['mu'] / 2 * float(hostx @ hostx)
        pullback = np.zeros_like(hostx)
        dual_linear = 0.
        feasible = True
        for K, b, shape, rad, y in zip(fixture['K'], fixture['b'], fixture['shapes'], fixture['radius'], hostys):
            value = (K.astype(float) @ hostx - b.astype(float)).reshape(shape)
            primal += float(rad.astype(float) @ np.linalg.norm(value, axis=1))
            pullback += K.astype(float).T @ y.reshape(-1)
            dual_linear -= float(b.astype(float) @ y.reshape(-1))
            feasible &= bool(np.all(np.linalg.norm(y, axis=1) <= rad.astype(float)))
        dual = dual_linear - float(pullback @ pullback) / (2 * fixture['mu'])
        gap = (primal - dual) / primal if primal > 0 else (0. if primal == dual == 0 else math.inf)
        return dict(r_primal=float(certificate['r_primal']), r_dual=float(certificate['r_dual']),
            primal=primal, dual_lower=dual, relative_bracket=gap,
            raw_dual_feasible=bool(feasible),
            relative_objective_error_to_clarabel=abs(primal-float(reference.value))/abs(float(reference.value)),
            reference_status=reference.status, reference_objective=float(reference.value),
            scope="same raw pair; public KKT plus independent numerical gap/P; no auxiliary-dual repair")
    return AcceptancePolicy(evaluate, declared_acceptance_identity()), dict(
        status=reference.status, objective=float(reference.value), x=np.asarray(xx.value))


def audit_records(records, fixture):
    audited = []
    for row in records:
        true_errors = []
        all_feasible = True
        covered = True
        block_met = True
        for v, y, estimates, budgets, matrices, radii in zip(row['inputs'], row['candidates'],
                row['block_estimates'], row['block_budgets'], fixture['physical_sigma'], fixture['radius']):
            for vv, yy, upper, budget, S, radius in zip(v, y, estimates, budgets, matrices, radii):
                vv, yy = vv.astype(float), yy.astype(float)
                q = np.linalg.solve(S, vv-yy)
                error = float(radius) * np.linalg.norm(q) - float(q @ yy)
                allowance = 64*np.finfo(float).eps*(float(radius)*np.linalg.norm(q)+abs(q@yy))
                true_errors.append(max(0., float(error)))
                all_feasible &= bool(np.linalg.norm(yy) <= float(radius))
                covered &= bool(error <= float(upper) + allowance)
                block_met &= bool(error <= float(budget) + allowance)
        actual_sum = math.fsum(true_errors)
        within = bool(actual_sum <= row['effective_global_total'])
        passed = bool(all_feasible and covered and (not row['committed'] or (block_met and within)))
        audited.append(dict(iteration=row['iteration'], committed=row['committed'],
            feasible=all_feasible, estimates_cover=covered, block_budget_met=block_met,
            actual_total=actual_sum, upper_total=row['estimated_total'],
            global_budget=row['effective_global_total'], within_global_budget=within, passed=passed))
    return dict(passed=all(a['passed'] for a in audited), updates=len(audited), rows=audited)


def save_run(output, name, loop, state, records, fixture):
    loop.save_state(output / (name + '_endpoint'), state)
    arrays = {}
    for i, row in enumerate(records):
        for field in ('inputs', 'candidates', 'block_estimates', 'block_budgets', 'inner_iterations'):
            for group, arr in enumerate(row[field]):
                arrays[f'k{i}_{field}_g{group}'] = arr
    np.savez_compressed(output / (name + '_trace.npz'), **arrays)
    audit = audit_records(records, fixture)
    serial_records = [{k: v for k, v in row.items() if k not in
        ('inputs','candidates','block_estimates','block_budgets','inner_iterations')} for row in records]
    result = dict(name=name, contract_hash=loop.contract_hash, status=state.status,
        iteration=state.iteration, reason=state.reason, last_check=state.last_check,
        budget_sum=state.budget_sum, estimate_sum=state.estimate_sum,
        trace=serial_records, independent_inner_audit=audit)
    (output / (name + '.json')).write_text(json.dumps(clean(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if platform.node() != 'Mazeppa':
        raise RuntimeError('Numerical work is authorized only on Mazeppa')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'], text=True).strip()
    if 'GPU-78b0d21c-fe02-946b-2e74-6930e848f338' not in gpu:
        raise RuntimeError('Wrong physical GPU')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; retain failed attempts')
    args.output.mkdir(parents=True)
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU runtime required')
    start = time.perf_counter()
    groups, tau, fixture = make_fixture()
    loop = build_loop(groups, tau, fixture)
    evaluate, reference = make_evaluator(groups, fixture)
    arrays = {'tau': np.asarray(tau), 'reference_x': reference['x']}
    for key in ('K','b','radius','Q','sigma'):
        for i, a in enumerate(fixture[key]):
            arrays[f'{key}_{i}'] = a
    np.savez_compressed(args.output / 'inputs.npz', **arrays)
    (args.output / 'contract.json').write_text(json.dumps(loop.contract, indent=2) + '\n')
    report = dict(host=platform.node(), gpu=gpu, jax=jax.__version__,
        dtype='float32 GPU updates; float64 CPU independent references',
        scope='tiny heterogeneous fixed-metric outer-loop integration, no speed or interval claim',
        sources={name: digest(Path(__file__).with_name(name)) for name in
            ('shared_shape_inexact_loop.py','shared_shape_inexact_validate.py','inexact_full_ball_prox.py')},
        fixture=clean({k:v for k,v in fixture.items() if k not in ('K','b','Q','sigma','physical_sigma','radius')}),
        reference=clean(reference), controls=[])

    def record_control(name, passed, **details):
        report['controls'].append(dict(name=name, passed=bool(passed), **details))
        (args.output / 'progress.json').write_text(json.dumps(clean(report), indent=2) + '\n')
        print(json.dumps(clean(report['controls'][-1])), flush=True)

    state, rows = loop.run_chunk(loop.initial_state(), 400, evaluate=evaluate)
    whole = save_run(args.output, 'uninterrupted', loop, state, rows, fixture)
    record_control('uninterrupted_accuracy', state.status == 'solved' and
        whole['independent_inner_audit']['passed'] and
        state.last_check['relative_objective_error_to_clarabel'] <= 1e-3,
        status=state.status, iteration=state.iteration, last_check=state.last_check)

    chunked, first = loop.run_chunk(loop.initial_state(), 7, evaluate=evaluate)
    loop.save_state(args.output/'resume_at_first_chunk', chunked)
    resumed_loop = build_loop(groups, tau, fixture)
    chunked = resumed_loop.load_state(args.output/'resume_at_first_chunk')
    chunked, second = resumed_loop.run_chunk(chunked, 11, evaluate=evaluate)
    resumed_loop.save_state(args.output/'resume_at_second_chunk', chunked)
    chunked = resumed_loop.load_state(args.output/'resume_at_second_chunk')
    chunked, third = resumed_loop.run_chunk(chunked, 400, evaluate=evaluate)
    chunk_rows = first + second + third
    split = save_run(args.output, 'chunked_resume', resumed_loop, chunked, chunk_rows, fixture)
    identical = (state.iteration == chunked.iteration and state.status == chunked.status
        and all(np.array_equal(np.asarray(a), np.asarray(b)) for a,b in
                zip((state.x,state.xprev,*state.ys),(chunked.x,chunked.xprev,*chunked.ys)))
        and [r['requested_total'] for r in rows] == [r['requested_total'] for r in chunk_rows]
        and [r['iteration'] for r in chunk_rows] == list(range(len(chunk_rows))))
    record_control('serialized_chunk_equivalence', identical and split['independent_inner_audit']['passed'],
        bitidentical=identical, iteration=chunked.iteration, cuts=[7,18],
        exact_same_budget_sequence=True if identical else False)

    for name, c, p, cap in [('zero_inner_cap', .03, .25, 0),
                           ('unattainable_budget', 1e-30, .25, 80),
                           ('rapid_decay_after_resume', .03, 6., 80)]:
        trial = build_loop(groups, tau, fixture, c=c, p=p, max_inner=cap)
        initial = trial.initial_state()
        before, prefix = trial.run_chunk(initial, 1, evaluate=evaluate)
        trial.save_state(args.output/(name+'_prefix'), before)
        restored = trial.load_state(args.output/(name+'_prefix'))
        failed, rest = trial.run_chunk(restored, 80, evaluate=evaluate)
        again, attempted = trial.run_chunk(failed, 10, evaluate=evaluate)
        result = save_run(args.output, name, trial, failed, prefix+rest, fixture)
        committed = [r for r in prefix+rest if r['committed']]
        expected_k = len(committed)
        sticky = (again is failed and attempted == [] and failed.iteration == expected_k)
        # The last attempted candidate is archived separately; committed state
        # has precisely the successful-prefix counter and no solved promotion.
        record_control(name, failed.status == 'inner_failure' and sticky and
                       result['independent_inner_audit']['passed'],
            status=failed.status, iteration=failed.iteration, sticky=sticky, reason=failed.reason)

    changed = replace(groups[0], offset=groups[0].offset + jnp.float32(.01))
    wrong_problem = build_loop((changed,*groups[1:]), tau, fixture)
    try:
        wrong_problem.load_state(args.output/'resume_at_first_chunk')
    except ValueError as error:
        record_control('restore_rejects_changed_problem', True, reason=str(error))
    else:
        record_control('restore_rejects_changed_problem', False)
    changed_metric = replace(groups[0], metric=prepare_metric(fixture['Q'][0], fixture['sigma'][0]*1.01))
    try:
        build_loop((changed_metric,*groups[1:]), tau, fixture).load_state(args.output/'resume_at_first_chunk')
    except ValueError as error:
        record_control('restore_rejects_changed_metric', True, reason=str(error))
    else:
        record_control('restore_rejects_changed_metric', False)
    try:
        build_loop(groups, tau, fixture, c=.04).load_state(args.output/'resume_at_first_chunk')
    except ValueError as error:
        record_control('restore_rejects_changed_budget', True, reason=str(error))
    else:
        record_control('restore_rejects_changed_budget', False)
    for name, kwargs in [
        ('restore_rejects_changed_implementation', dict(implementation_version='changed-version-control')),
        ('restore_rejects_changed_acceptance_policy', dict(policy_identity='changed-policy-control')),
        ('restore_rejects_changed_check_interval', dict(check_every=7)),
    ]:
        try:
            build_loop(groups, tau, fixture, **kwargs).load_state(args.output/'resume_at_first_chunk')
        except ValueError as error:
            record_control(name, True, reason=str(error))
        else:
            record_control(name, False)
    try:
        loop.run_chunk(loop.initial_state(), 1,
            evaluate=AcceptancePolicy(evaluate.evaluate, 'changed-policy-control'))
    except ValueError as error:
        record_control('runtime_rejects_changed_acceptance_policy', True, reason=str(error))
    else:
        record_control('runtime_rejects_changed_acceptance_policy', False)
    for name, kwargs in [('nonsummable_schedule',dict(budget_p=0.)),
                         ('invalid_coupling_bound',dict(coupling_upper=1.))]:
        options=dict(mu=fixture['mu'], budget_c=.03,budget_p=.25,
                     coupling_upper=fixture['coupling_upper'],coupling_provenance=fixture['coupling_provenance'],
                     acceptance_policy_identity=declared_acceptance_identity())
        options.update(kwargs)
        try:
            InexactLoop(groups,tau,**options)
        except ValueError as error:
            record_control(name,True,reason=str(error))
        else:
            record_control(name,False)
    stats = jax.devices()[0].memory_stats()
    report['allocator_peak_bytes'] = stats.get('peak_bytes_in_use')
    report['allocator_cap_passed'] = report['allocator_peak_bytes'] <= 8*1024**3
    report['elapsed_seconds'] = time.perf_counter()-start
    report['passed'] = all(c['passed'] for c in report['controls']) and report['allocator_cap_passed']
    report['files_sha256']={str(p.relative_to(args.output)):digest(p)
        for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'results.json').write_text(json.dumps(clean(report),indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
