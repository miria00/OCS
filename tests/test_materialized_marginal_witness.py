"""Returned LP witness domain regression on original represented data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs


def case():
    x = ocs.Variable(1)
    f = ocs.minimize([ocs.nonneg(x), ocs.equality(
        ocs.MarginalOperator(1, 1) @ x - ocs.Parameter('m'))],
        cost=ocs.Parameter('c'))
    c = np.float32(3.3396772778360173e-6)
    # The exact offending entry of the preserved N1024 counterexample.
    y = -np.array([0.031172798946499825, -0.031169459223747253], np.float32)
    return f.compile(tol=1e-3, max_blocks=1), jnp.ones(1, jnp.float32), y, {
        'c': jnp.array([c]), 'm': jnp.ones(2, jnp.float32)}


@pytest.mark.parametrize('eager', [False, True])
def test_same_represented_invalid_witness_never_certifies(eager):
    before = jax.config.x64_enabled
    comp, x, y, data = case()
    exact_slack = float(data['c'][0]) + sum(float(v) for v in y)
    assert exact_slack < 0
    with jax.disable_jit(eager):
        b = comp.objective_bound(x, y, data, recover=False, return_dual=True)
    assert not b['objective_lower_bound_available']
    assert not comp.acceptance(b)['target_met']
    assert jax.config.x64_enabled == before


@pytest.mark.parametrize('eager', [False, True])
def test_inward_witness_is_useful_and_roundtrips(eager):
    comp, x, y, data = case()
    y[0] = np.nextafter(y[0], np.float32(np.inf))
    assert float(data['c'][0]) + sum(float(v) for v in y) > 0
    with jax.disable_jit(eager):
        b = comp.objective_bound(x, y, data, recover=False, return_dual=True)
        b2 = comp.objective_bound(x, b['certifying_dual'], data, recover=False)
    assert b['objective_lower_bound_available']
    assert comp.acceptance(b)['target_met']
    assert comp.acceptance(b2)['target_met']


def test_recovery_reports_actual_returned_witness_and_changed_cost():
    comp, x, y, data = case()
    b = comp.objective_bound(x, y, data, recover=True, return_dual=True)
    if b['objective_lower_bound_available']:
        witness = np.asarray(b['certifying_dual'])
        assert float(data['c'][0]) + sum(float(v) for v in witness) >= 0
        off = comp.objective_bound(x, witness, data, recover=False)
        assert off['objective_lower_bound_available']
        assert comp.acceptance(off)['target_met'] == comp.acceptance(b)['target_met']
    # A witness that is valid for the old cost is rejected for changed data.
    good = y.copy(); good[0] = np.nextafter(good[0], np.float32(np.inf))
    changed = dict(data, c=jnp.zeros(1, jnp.float32))
    assert not comp.objective_bound(x, good, changed, recover=False)['objective_lower_bound_available']


def test_zero_witness_and_positive_cost_remain_supported():
    comp, x, _, data = case()
    b = comp.objective_bound(x, jnp.zeros(2, jnp.float32), data, recover=False)
    assert b['objective_lower_bound_available']
    assert not comp.acceptance(b)['target_met']


@pytest.mark.parametrize('layout', ['dense', 'compact'])
def test_batch_and_changing_data_return_the_assessed_witness(layout):
    v = ocs.Variable(1)
    comp = ocs.minimize([ocs.nonneg(v), ocs.equality(
        ocs.MarginalOperator(1, 1) @ v - ocs.Parameter('m'))],
        cost=ocs.Parameter('c')).compile(
            tol=1e-3, block=10, max_blocks=100, data_layout=layout)
    data = {'m': jnp.ones((2, 2), jnp.float32),
            'c': jnp.asarray([[.25], [.75]], jnp.float32)}
    results = comp.solve_many_results(data, accuracy='objective_gap')
    report = comp.certificate_many(jnp.stack([s.x for s in results]),
                                   jnp.stack([s.y for s in results]), data)
    for i, result in enumerate(results):
        row = {k: val[i] for k, val in data.items()}
        bound = comp.objective_bound(result.x, result.y, row, recover=False)
        assert result.status == 'solved'
        assert comp.acceptance(bound)['target_met']
        assert float(row['c'][0]) + sum(map(float, np.asarray(result.y))) >= 0
        assert bool(report['objective_lower_bound_available'][i])
    # Reuse the compiled route with different costs; a formerly certifying
    # witness cannot retain a success attached to the prior instance.
    changed = {'m': data['m'][0], 'c': jnp.zeros(1, jnp.float32)}
    stale = comp.objective_bound(results[1].x, results[1].y, changed, recover=False)
    assert not stale['objective_lower_bound_available']
