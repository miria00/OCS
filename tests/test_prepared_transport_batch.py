"""Installed batch OT: numerical contract, mixed outcomes and owned lifetimes."""
import copy
import dataclasses
import pickle

import numpy as np
import pytest

from ocs.transport import PreparedQuadraticOT


@pytest.fixture
def gpu():
    import jax
    pytest.importorskip('cupy')
    if jax.devices()[0].platform != 'gpu':
        pytest.skip('NVIDIA GPU qualification')


def data(n=35, b=3):
    rng = np.random.RandomState(512)
    c = rng.uniform(size=(b, n, n)).astype(np.float32)
    # Exact dyadic masses, including zero coordinates and asymmetric marginals.
    p = np.zeros((b, n), np.float32)
    p[:, :4] = [0, .125, .375, .5]
    q = np.ascontiguousarray(p[:, ::-1])
    return dict(cost=c, p=p, q=q)


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5, '2', 2**31])
def test_invalid_batch_size_precedes_gpu(batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        PreparedQuadraticOT(4, regularization=1, batch_size=batch_size)


def test_owned_batch_accuracy_reuse_and_close(gpu):
    from ocs.transport._assessment import assess
    values = data()
    saved_inputs = {k: v.copy() for k, v in values.items()}
    session = PreparedQuadraticOT(35, regularization=.7, batch_size=3, return_capacity=2)
    assert session.batch_size == 3
    with pytest.raises(AttributeError): session.batch_size = 4
    with pytest.raises(TypeError): copy.copy(session)
    with pytest.raises(TypeError): copy.deepcopy(session)
    with pytest.raises(TypeError): pickle.dumps(session)
    with pytest.raises(ValueError, match='solve_many_results'):
        session.solve(values['cost'][0], values['p'][0], values['q'][0])
    session._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='busy'): session.solve_many_results(values)
        with pytest.raises(RuntimeError, match='busy'): session.close()
    finally: session._lock.release()
    first = session.solve_many_results(values)
    assert len(first) == 3 and all(r.solved for r in first)
    saved = [(r.plan.copy(), r.row_potential.copy(), r.column_potential.copy()) for r in first]
    for i, row in enumerate(first):
        check = assess(*(values[k][i] for k in ('cost', 'p', 'q')), *saved[i], np.float32(.7))
        assert check['gate_pass']
        assert row.statistics['lane'] == i and row.statistics['termination'] == 'criterion_met'
        with pytest.raises(TypeError): row.report['gate_pass'] = False
        with pytest.raises(TypeError): row.statistics['events'][0]['iterations'] = 0
        with pytest.raises(ValueError): row.plan[0, 0] = 2
        with pytest.raises(dataclasses.FrozenInstanceError): row.plan = values['cost'][i]
        for j in range(i): assert not np.shares_memory(row.plan, first[j].plan)
    # The sequence input form avoids an extra dense packing allocation.
    second = session.solve_many_results({k: list(v) for k, v in values.items()})
    assert all(r.solved for r in second)
    for one, two in zip(first, second):
        assert not np.shares_memory(one.plan, two.plan)
        np.testing.assert_array_equal(one.plan, two.plan)
    with pytest.raises(RuntimeError, match='capacity'): session.solve_many_results(values)
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match='closed'): session.solve_many_results(values)
    for row, old in zip(first, saved):
        for actual, wanted in zip((row.plan, row.row_potential, row.column_potential), old):
            np.testing.assert_array_equal(actual, wanted)
    for k in values: np.testing.assert_array_equal(values[k], saved_inputs[k])


@pytest.mark.parametrize('bad', ['missing_key', 'extra_key', 'count', 'dtype', 'transpose', 'scalar'])
def test_shape_refusal_does_not_consume_slot(gpu, bad):
    values = data(4, 2)
    if bad == 'missing_key': values.pop('q')
    if bad == 'extra_key': values['other'] = 1
    if bad == 'count': values['cost'] = values['cost'][:1]
    if bad == 'dtype': values['cost'] = values['cost'].astype(np.float64)
    if bad == 'transpose': values['cost'] = values['cost'].transpose(0, 2, 1)
    if bad == 'scalar': values['cost'] = np.array(1)
    with PreparedQuadraticOT(4, regularization=1, batch_size=2, return_capacity=1) as s:
        with pytest.raises(ValueError): s.solve_many_results(values)
        assert s._used == 0
        assert all(r.solved for r in s.solve_many_results(data(4, 2)))


@pytest.mark.parametrize('bad', ['nan', 'negative', 'mass', 'zero_mass'])
def test_one_invalid_lane_refuses_and_consumes_batch_slot(gpu, bad):
    values = data(4, 2)
    if bad == 'nan': values['cost'][1, 0, 0] = np.nan
    if bad == 'negative': values['cost'][1, 0, 0] = -1
    if bad == 'mass': values['q'][1, 0] += np.float32(.125)
    if bad == 'zero_mass': values['p'][1] = 0; values['q'][1] = 0
    with PreparedQuadraticOT(4, regularization=1, batch_size=2, return_capacity=1) as s:
        with pytest.raises(ValueError): s.solve_many_results(values)
        with pytest.raises(RuntimeError, match='capacity'): s.solve_many_results(data(4, 2))


def test_cap_returns_individual_reports_not_batch_success(gpu):
    values = data(4, 2)
    values['cost'][0] = .25
    values['p'][0] = .25
    values['q'][0] = .25
    with PreparedQuadraticOT(4, regularization=.01, batch_size=2,
                             return_capacity=1, max_iterations=100) as s:
        rows = s.solve_many_results(values)
    assert rows[0].solved and not rows[1].solved
    assert rows[0].statistics['termination'] == 'criterion_met'
    assert rows[1].statistics['termination'] == 'iteration_limit'
    assert all(r.statistics['iterations'] == 100 for r in rows)


def test_partial_final_block_and_b1_scalar_equivalence(gpu):
    from ocs.transport import OTConvergenceError
    values = data(65, 1)
    from ocs.transport._assessment import assess
    with PreparedQuadraticOT(65, regularization=1.3, batch_size=1,
                             return_capacity=1, max_iterations=101) as s:
        row, = s.solve_many_results(values)
    assert row.statistics['iterations'] == 101
    assert not row.solved
    check = assess(*(values[k][0] for k in ('cost', 'p', 'q')),
                   row.plan, row.row_potential, row.column_potential, np.float32(1.3))
    assert check['gate_pass'] == row.solved
    with PreparedQuadraticOT(65, regularization=1.3, return_capacity=1,
                             max_iterations=101) as s:
        with pytest.raises(ValueError, match='batch_size'): s.solve_many_results(values)
        with pytest.raises(OTConvergenceError) as error:
            s.solve(*(values[k][0] for k in ('cost', 'p', 'q')))
    np.testing.assert_allclose(row.report['upper'], error.value.report['upper'], rtol=3e-6)
    np.testing.assert_allclose(row.report['lower'], error.value.report['lower'], rtol=3e-6)


def test_vmap_recurrence_matches_separate_scalar_segments(gpu):
    import jax
    import jax.numpy as jnp
    from ocs.transport._first_order import centered_initial_state, make_segment
    values = data(65, 3)
    c, p, q = [jnp.asarray(values[k]) for k in ('cost', 'p', 'q')]
    mu = np.float32(1.3)
    with PreparedQuadraticOT(65, regularization=mu, batch_size=3,
                             return_capacity=1, max_iterations=7) as s:
        state = s._reset(s._state, c, p, q, mu)
        s._state = s._segment(state, c, p, q, mu, jnp.int32(7))
        actual = jax.device_get(s._state)
    segment = make_segment(True, 'pallas', True)
    for i in range(3):
        state = centered_initial_state(c[i], p[i], q[i], mu)
        expected = jax.device_get(segment(state, c[i], p[i], q[i], mu, jnp.int32(7)))
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a[i], e, rtol=3e-6, atol=3e-7)
