"""Vectorized prepared OT execution using the installed scalar recurrence.

No competitor code, private input data, or benchmark configuration is imported.
The shared-endpoint policy amortizes GPU launches across independent problems.
"""
from collections.abc import Mapping
import time

import numpy as np

from ._prepared import QuadraticOTResult, _freeze


def _reports(owner, c, p, q):
    jax = owner._jax
    with jax.enable_x64():
        values = jax.device_get(owner._batch_assess(
            c, p, q, owner._state[0], -owner._state[1], -owner._state[2],
            np.float64(owner.regularization)))
    reports = []
    for i in range(owner.batch_size):
        report = {key: bool(value[i]) if key in ('gate_pass', 'arithmetic_valid')
                  else float(value[i]) for key, value in values.items()}
        report.update(numerical_float64=True, exact_arithmetic=False,
                      backend='GPU f64 common assessor')
        if not report['arithmetic_valid']:
            raise ArithmeticError(f"quadratic OT batch lane {i}: {report}")
        reports.append(report)
    return reports


def prepare_batch(owner):
    from ._first_order import centered_initial_state, make_segment, screening
    from ._assessment import _evaluate
    jax, jnp, cp = owner._jax, owner._jnp, owner._cp
    b, n = owner.batch_size, owner.n

    def pinned(shape):
        size = int(np.prod(shape))
        memory = cp.cuda.alloc_pinned_memory(size * 4)
        return np.frombuffer(memory, dtype=np.float32, count=size).reshape(shape)

    shapes = ((b, n, n), (b, n), (b, n))
    owner._hc, owner._hp, owner._hq = [pinned(s) for s in shapes]
    owner._hu = owner._hv = None
    owner._outputs = [tuple(pinned(s) for s in shapes)
                      for _ in range(owner.return_capacity)]
    owner._hc.fill(0)
    owner._hp.fill(np.float32(1/n))
    owner._hq[:] = owner._hp
    owner._dc, owner._dp, owner._dq = [cp.asarray(z) for z in
                                      (owner._hc, owner._hp, owner._hq)]
    cp.cuda.get_current_stream().synchronize()
    initialize = jax.jit(jax.vmap(centered_initial_state, in_axes=(0, 0, 0, None)))
    # Donate only the uniquely owned outer state. The inner scalar executable
    # must not independently donate references belonging to the enclosing vmap.
    owner._segment = jax.jit(jax.vmap(make_segment(True, 'pallas', False),
                                    in_axes=(0, 0, 0, 0, None, None)),
                             donate_argnums=(0,))
    owner._screen = jax.jit(jax.vmap(screening, in_axes=(0, 0, 0, 0, None)))
    owner._batch_assess = jax.jit(jax.vmap(_evaluate,
                                         in_axes=(0, 0, 0, 0, 0, 0, None)))
    owner._reset = jax.jit(lambda old, c, p, q, mu: initialize(c, p, q, mu),
                           donate_argnums=(0,), keep_unused=True)

    @jax.jit
    def validate(c, p, q):
        pm = jnp.sum(p.astype(jnp.float64), axis=1)
        qm = jnp.sum(q.astype(jnp.float64), axis=1)
        return (jnp.all(jnp.isfinite(c)) & (jnp.min(c) >= 0) &
                jnp.all(jnp.isfinite(p)) & jnp.all(jnp.isfinite(q)) &
                (jnp.min(p) >= 0) & (jnp.min(q) >= 0) &
                jnp.all(pm > 0) & jnp.all(pm == qm))

    owner._validate = validate
    c, p, q = [jax.dlpack.from_dlpack(z, copy=False)
               for z in (owner._dc, owner._dp, owner._dq)]
    with jax.enable_x64():
        if not bool(validate(c, p, q)):
            raise ArithmeticError('invalid batch compilation fixture')
    owner._state = initialize(c, p, q, owner.regularization)
    owner._state = owner._segment(owner._state, c, p, q, owner.regularization,
                                  jnp.int32(100))
    jax.block_until_ready(owner._state)
    jax.block_until_ready(owner._screen(owner._state, c, p, q, owner.regularization))
    _reports(owner, c, p, q)
    owner._state = owner._reset(owner._state, c, p, q, owner.regularization)
    jax.block_until_ready(owner._state)


def _inputs(owner, data):
    if not isinstance(data, Mapping) or set(data) != {'cost', 'p', 'q'}:
        raise ValueError("data must map exactly 'cost', 'p', 'q' to batch arrays")
    values = []
    for key, shape in (('cost', (owner.n, owner.n)), ('p', (owner.n,)),
                       ('q', (owner.n,))):
        group = data[key]
        if (not isinstance(group, (np.ndarray, list, tuple)) or
                (isinstance(group, np.ndarray) and group.ndim == 0) or
                len(group) != owner.batch_size):
            raise ValueError('inputs must have the prepared batch size')
        lanes = tuple(group)
        for value in lanes:
            if (not isinstance(value, np.ndarray) or value.shape != shape or
                    value.dtype != np.float32 or not value.flags.c_contiguous):
                raise ValueError('each lane must be a C-contiguous float32 array of the prepared shape')
        values.append(lanes)
    return values


def solve_many_results(owner, data):
    if not owner._lock.acquire(blocking=False):
        raise RuntimeError('session is busy')
    start = time.perf_counter()
    try:
        if owner._closed:
            raise RuntimeError('session is closed')
        inputs = _inputs(owner, data)
        if owner._used >= owner.return_capacity:
            raise RuntimeError('prepared output capacity exhausted')
        slot = owner._used
        owner._used += 1
        x, u, v = owner._outputs[slot]
        jax, jnp, cp = owner._jax, owner._jnp, owner._cp
        t = time.perf_counter()
        for target, lanes in zip((owner._hc, owner._hp, owner._hq), inputs):
            for i, value in enumerate(lanes):
                np.copyto(target[i], value)
        record = dict(host_staging_s=time.perf_counter()-t, events=[])
        t = time.perf_counter()
        for target, value in zip((owner._dc, owner._dp, owner._dq),
                                 (owner._hc, owner._hp, owner._hq)):
            target.set(value)
        cp.cuda.get_current_stream().synchronize()
        c, p, q = [jax.dlpack.from_dlpack(z, copy=False)
                   for z in (owner._dc, owner._dp, owner._dq)]
        record['input_transfer_s'] = time.perf_counter()-t
        t = time.perf_counter()
        with jax.enable_x64():
            if not bool(owner._validate(c, p, q)):
                raise ValueError('every lane must be finite, nonnegative, with positive exactly equal marginal mass')
        record['input_validation_s'] = time.perf_counter()-t
        t = time.perf_counter()
        owner._state = owner._reset(owner._state, c, p, q, owner.regularization)
        jax.block_until_ready(owner._state)
        record['fresh_initialize_s'] = time.perf_counter()-t
        target, reports = 0, []
        while target < owner.max_iterations:
            steps = min(100, owner.max_iterations-target)
            target += steps
            t = time.perf_counter()
            owner._state = owner._segment(owner._state, c, p, q, owner.regularization,
                                          jnp.int32(steps))
            jax.block_until_ready(owner._state)
            event = dict(iterations=target, update_s=time.perf_counter()-t)
            t = time.perf_counter()
            screen = np.stack(jax.device_get(owner._screen(
                owner._state, c, p, q, owner.regularization)), axis=-1)
            event.update(screen_s=time.perf_counter()-t, screen=screen.tolist())
            if not np.isfinite(screen).all():
                raise ArithmeticError('nonfinite quadratic OT batch screen')
            if (np.all((screen[:, 0] <= 1e-3) & (np.abs(screen[:, 1]) <= 1e-3)) or
                    target == owner.max_iterations):
                t = time.perf_counter()
                reports = _reports(owner, c, p, q)
                event.update(gates=reports, common_check_s=time.perf_counter()-t)
            record['events'].append(event)
            if 'gates' in event and all(r['gate_pass'] for r in reports):
                break
        t = time.perf_counter()
        for source, output in zip(owner._state[:3], (x, u, v)):
            cp.from_dlpack(source).get(out=output, blocking=True)
        np.negative(u, out=u)
        np.negative(v, out=v)
        for value in (x, u, v):
            value.setflags(write=False)
        record.update(device_return_and_finalize_s=time.perf_counter()-t,
                      iterations=target, output_slot=slot, batch_size=owner.batch_size,
                      batch_policy='vectorized_lockstep',
                      prepared_output_capacity=owner.return_capacity,
                      prepared_output_payload_bytes=sum(z.nbytes for out in owner._outputs for z in out))
        # Freeze shared history once: it is immutable and need not be copied B times.
        common = _freeze(record)
        rows = [QuadraticOTResult(x[i], u[i], v[i], _freeze(report),
                 _freeze(dict(common, lane=i, termination='criterion_met' if report['gate_pass']
                              else 'iteration_limit')))
                for i, report in enumerate(reports)]
        elapsed = time.perf_counter()-start
        return [QuadraticOTResult(row.plan, row.row_potential, row.column_potential,
                                  row.report, _freeze(dict(row.statistics, complete_warm_s=elapsed)))
                for row in rows]
    finally:
        owner._lock.release()
