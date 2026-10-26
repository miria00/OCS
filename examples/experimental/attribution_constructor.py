"""Gate-1 constructors; frozen public sources are reused without mutation.

S1 changes only the column projection of native SMFE. D implements the
constrained shared-diagonal gradient, never a truncated FULL matrix update.
All numerical execution belongs on the authorized remote RTX 5090 host.
"""
import inspect
import math
import numbers
import time

import jax
import jax.numpy as jnp
import numpy as np

import ocs.solver.equilibrate_sgd as native
from mixed_metric_gpu import BlockGroup, _update_side
import shared_shape_gpu as shared


def tied_smfe_function(channels, count):
    if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<=0 for v in (channels,count)):
        raise ValueError('positive integral channel/count required')
    source = inspect.getsource(native.smfe_equilibration)
    original = 'v = jnp.clip(v - eta * gv, -M, M)'
    replacement = ('v = jnp.repeat(jnp.clip(jnp.mean((v - eta * gv).reshape('
                   '_tie_channels, _tie_count), axis=1), -M, M), _tie_count)')
    if source.count(original) != 1:
        raise RuntimeError('frozen native source projection anchor changed')
    source = source.replace('def smfe_equilibration(', 'def tied_smfe_equilibration(', 1)
    source = source.replace(original, replacement)
    namespace = dict(vars(native), _tie_channels=channels, _tie_count=count)
    exec(compile(source, '<native-smfe-exact-column-tie-projection>', 'exec'), namespace)
    return namespace['tied_smfe_equilibration'], namespace, source


def diagonal_row_update(log_a, samples, *, target=1., mu=1., box=2., eta):
    """Multiplicity N cancels between pooled gradient and intrinsic metric."""
    k = log_a.shape[0]
    energy = jnp.mean(samples.reshape(samples.shape[0], -1, k)**2, axis=(0, 1))
    tangent = .5*jnp.exp(log_a)*energy - .5*target + mu/4*log_a
    tentative = log_a-eta*tangent
    return jnp.clip(tentative, -2*box, 2*box), jnp.all(jnp.isfinite(tentative))


def make_diagonal_constructor(op, *, iterations=80, probes=2, mu=1., box=2., shift=20):
    k, count, n = op.c, op.N, op.shape[1]
    if op.shape != (k*count,k*count):
        raise ValueError('square channel-major F2 operator required')
    if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<=0 for v in (k,count,iterations,probes,shift)):
        raise ValueError('positive integral dimensions and algorithm counts required')
    if not math.isfinite(mu) or mu<=0 or not math.isfinite(box) or box<0:
        raise ValueError('positive finite ridge and nonnegative finite box required')
    columns = (BlockGroup('scalar', count, k),)

    def run(key, a0, b0):
        if a0.shape!=(k,) or b0.shape!=(k,) or a0.dtype!=jnp.float32 or b0.dtype!=jnp.float32:
            raise ValueError('float32 row/column log states of shape (channels,) required')
        initial_valid=(jnp.all(jnp.isfinite(a0)) & jnp.all(jnp.isfinite(b0)) &
                       jnp.all(jnp.abs(a0)<=2*box) & jnp.all(jnp.abs(b0)<=2*box))
        def body(index, state):
            key, a, b, abar, bbar, total, finite = state
            key, rk, ck = jax.random.split(key, 3)
            xi = jax.random.rademacher(rk, (probes, n), dtype=jnp.float32)
            zeta = jax.random.rademacher(ck, (probes, n), dtype=jnp.float32)
            forward = jax.vmap(lambda x: op.matvec(
                (x.reshape(k, count)*jnp.exp(.5*b)[:, None]).reshape(-1)))(xi)
            reverse = jax.vmap(lambda y: op.rmatvec(
                (y.reshape(count, k)*jnp.exp(.5*a)[None, :]).reshape(-1)))(zeta)
            t = index+1
            weight = t+shift-1
            ratio = weight/(total+weight)
            avga = jnp.where(index == 0, a, (1-ratio)*abar+ratio*a)
            avgb = jnp.where(index == 0, b, (1-ratio)*bbar+ratio*b)
            eta = 2/((mu/4)*(t+shift))
            newa, fa = diagonal_row_update(a, forward, target=1., mu=mu, box=box, eta=eta)
            newb, fb = _update_side(columns, (b,), reverse, 1., mu, box, eta, 256)
            finite = finite & fa & fb & jnp.all(jnp.isfinite(avga)) & jnp.all(jnp.isfinite(avgb))
            return key, newa, newb[0], avga, avgb, total+weight, finite
        _, a, b, avga, avgb, weight, finite = jax.lax.fori_loop(0, iterations, body,
            (key, a0, b0, a0, b0, jnp.float32(0), initial_valid))
        return {'A': (avga,), 'B': (avgb,), 'last_A': (a,), 'last_B': (b,),
                'weight_sum': weight, 'finite': finite,
                'status':jnp.where(~initial_valid,1,jnp.where(finite,0,2)),
                'operator_calls': jnp.int32(2*iterations*probes)}
    return jax.jit(run)


def _time_program(program, arguments):
    t = time.perf_counter(); lowered = program.lower(*arguments); lower = time.perf_counter()-t
    t = time.perf_counter(); executable = lowered.compile(); compile_s = time.perf_counter()-t
    t = time.perf_counter(); answer = executable(*arguments); jax.block_until_ready(answer); execute = time.perf_counter()-t
    warm = []
    for _ in range(2):
        t = time.perf_counter(); repeat = executable(*arguments); jax.block_until_ready(repeat)
        warm.append(time.perf_counter()-t)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
            zip(jax.tree_util.tree_leaves(answer), jax.tree_util.tree_leaves(repeat)))
    return answer, {'lower': lower, 'compile': compile_s, 'execute': execute,
        'cold_setup_plus_execution': lower+compile_s+execute,
        'warm_replays': warm, 'warm_identical': True}


def construct(op, arm, iterations=80):
    """Return one row k-by-k matrix and primal column vector, before norm rescale."""
    k, count, n = op.c, op.N, op.shape[1]
    if arm not in ('S0','S1','S2','D','F'):
        raise ValueError('unknown attribution arm')
    if op.shape!=(k*count,k*count) or isinstance(iterations,bool) or not isinstance(iterations,numbers.Integral) or iterations<=0:
        raise ValueError('square F2 operator and positive integral iterations required')
    contract = {'arm': arm, 'mu': 1., 'M': 2., 'targets': [1., 1.],
        'updates': iterations, 'probes': 2, 'gradient_operator_calls': 4*iterations,
        'extra_norm_operator_calls': 30 if arm in ('S0', 'S1') else 0,
        'row_class': 'term scalar' if arm.startswith('S') else 'shared '+('diagonal' if arm == 'D' else 'full'),
        'column_class': 'free diagonal' if arm == 'S0' else 'spatially tied by channel',
        'schedule': 'native SGD/post-update weighted average' if arm in ('S0', 'S1') else 'shift20/pre-update weighted geodesic average',
        'probes_note': 'native sequential splits S0/S1; identical ambient array streams S2/D/F'}
    if arm in ('S0', 'S1'):
        fn = native.smfe_equilibration
        namespace = vars(native)
        generated = None
        if arm == 'S1': fn, namespace, generated = tied_smfe_function(k, count)
        original = namespace['_cached_smfe_jit']
        captured = {}
        def capture(_operators, _settings):
            def decorate(run):
                def execute():
                    answer, timings = _time_program(jax.jit(run), ())
                    captured.update(answer=answer, timings=timings)
                    return answer
                return execute
            return decorate
        namespace['_cached_smfe_jit'] = capture
        try:
            tau, sigma, norm = fn(op.matvec, op.rmatvec, n, n, [(0, n)],
                iters=iterations, probes=2, gamma=1., M=2., safety=.9, seed=0, schedule='sgd')
        finally:
            namespace['_cached_smfe_jit'] = original
        scale = .9/(norm+1e-30)
        a = np.eye(k)*float(np.asarray(sigma)[0]/scale)
        b = np.asarray(tau, np.float64)/scale
        contract['native_norm_removed_before_common_normalization'] = norm
        contract['native_unscaling_scope'] = 'divide rounded float32 scaled outputs; approximate recovery, independently checked'
        captured['timings']['instrumentation'] = 'unchanged native JIT body captured; S1 only exact tied column projection injected'
        return a, b, captured['timings'], contract, generated
    key = jax.random.PRNGKey(0)
    columns = (BlockGroup('scalar', count, k),)
    if arm == 'D':
        program = make_diagonal_constructor(op, iterations=iterations)
        arguments = (key, jnp.zeros(k, jnp.float32), jnp.zeros(k, jnp.float32))
    else:
        rows = (BlockGroup('scalar', n),) if arm == 'S2' else (shared.SharedShapeGroup(k, count),)
        program = shared.make_constructor(op.matvec, op.rmatvec, rows, columns, iterations=iterations)
        arguments = (key, shared.initial_state(rows), shared.initial_state(columns))
    answer, timings = _time_program(program, arguments)
    assert bool(answer['finite']), 'constructor failure'
    state = np.asarray(answer['A'][0], np.float64)
    a = np.eye(k)*np.exp(state[0]) if arm == 'S2' else np.diag(np.exp(state)) if arm == 'D' else state
    b = np.repeat(np.exp(np.asarray(answer['B'][0], np.float64)), count)
    return a, b, timings, contract, None
