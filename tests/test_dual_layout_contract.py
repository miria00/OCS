"""Independent analytic and public-buffer checks for the block dual candidate."""
import jax.numpy as jnp
import numpy as np
import pytest
import ocs

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# flat and blocked dual layouts agreeing.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def family():
    x = ocs.Variable(4)
    identity = ocs.Dense(jnp.eye(4, dtype=jnp.float32))
    selected = ocs.Dense(jnp.eye(4, dtype=jnp.float32)[jnp.array([0, 2])])
    # Two compatible L1 blocks have different parameters; a third has a
    # different output dimension. The exact minimizer is coordinatewise.
    return ocs.minimize([
        ocs.nonneg(x),
        ocs.least_squares(identity @ x - ocs.Parameter('b')),
        ocs.l1(identity @ x, lam=.2),
        ocs.l1(selected @ x, lam=.1),
        ocs.l1((-1.0 * identity) @ x, lam=.15),
    ], cost=ocs.Parameter('c'))


def compile_layout(layout, block=100, max_blocks=None, loop_unroll=4):
    return family().compile(pdhg_variant='standard', dual_layout=layout,
        equilibrate='scalar', tol=1e-5, block=block, loop_unroll=loop_unroll,
        max_blocks=max_blocks if max_blocks is not None else (5000+block-1)//block)


def data():
    return {'b': jnp.asarray([[-1., .5, 2., 3.], [2., 1., -3., .7],
                              [.3, .7, 1.2, -2.]], jnp.float32),
            'c': jnp.asarray([[.1, -.05, .02, .03], [-.1, .05, .1, -.03],
                              [.01, .03, -.05, .08]], jnp.float32)}


def exact_point(row):
    penalty = np.array([.45, .35, .45, .35])
    return np.maximum(np.asarray(row['b'], float)-np.asarray(row['c'], float)-penalty, 0.)


@pytest.mark.parametrize('block', [1, 3, 5, 100])
@pytest.mark.parametrize('loop_unroll', [1, 4])
def test_layout_preserves_grouped_functions_and_nondivisible_blocks(block, loop_unroll):
    batch = data()
    outputs = []
    for layout in ('flat', 'blocks'):
        compiled = compile_layout(layout, block, loop_unroll=loop_unroll)
        results = compiled.solve_many_results(batch)
        outputs.append(results)
        for i, result in enumerate(results):
            row = {k: v[i] for k, v in batch.items()}
            assert result.status == 'solved'
            assert result.iters % block == 0
            report = compiled.certificate(result.x, result.y, row)
            assert max(report['r_primal'], report['r_dual']) <= 1e-5
            assert result.y.shape == (14,)
            np.testing.assert_allclose(result.x, exact_point(row), rtol=2e-4, atol=2e-5)
    for flat, blocks in zip(*outputs):
        np.testing.assert_allclose(flat.x, blocks.x, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('layout', ['flat', 'blocks'])
def test_layout_keeps_flat_warm_starts_owned_by_the_caller(layout):
    compiled = compile_layout(layout)
    batch = data()
    row = {k: v[0] for k, v in batch.items()}
    first = compiled.solve_instance(row)
    x0, y0 = first.x, first.y
    before_x, before_y = np.array(x0), np.array(y0)
    second = compiled.solve_instance(row, warm_start=(x0, y0))
    assert second.status == 'solved'
    assert second.iters == 0
    np.testing.assert_array_equal(x0, before_x)
    np.testing.assert_array_equal(y0, before_y)
    np.testing.assert_allclose(second.x, exact_point(row), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('layout', ['flat', 'blocks'])
def test_layout_rejects_nonfinite_rows_without_contaminating_valid_instances(layout):
    compiled = compile_layout(layout)
    batch = data()
    batch['b'] = batch['b'].at[1, 0].set(jnp.nan)
    batch['c'] = batch['c'].at[2, 1].set(jnp.inf)
    results = compiled.solve_many_results(batch)
    assert results[0].status == 'solved'
    assert all(r.status == 'numerical_error' for r in results[1:])
    row = {k: v[0] for k, v in batch.items()}
    assert compiled.solve_instance(row, warm_start=(jnp.zeros(4), jnp.full(14, jnp.nan))).status == 'numerical_error'


def test_layout_preserves_zero_iteration_budget():
    compiled = compile_layout('blocks', block=3, max_blocks=0)
    row = {k: v[0] for k, v in data().items()}
    result = compiled.solve_instance(row)
    assert result.iters == 0 and result.status == 'inaccurate'
    assert result.y.shape == (14,)


@pytest.mark.parametrize('options', [{'dual_layout': 'other'},
                                   {'dual_layout': 'blocks', 'pdhg_variant': 'halpern'}])
def test_unsupported_layout_configuration_rejects(options):
    with pytest.raises(ValueError, match='layout'):
        family().compile(**options)
