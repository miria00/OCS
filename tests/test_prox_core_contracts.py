"""Independent references for core proximal metric and value contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox import (Box, Equality, ExpCone, GroupL2, L1, L2, NonNeg,
                      PSDCone, PowerCone, Quadratic, SecondOrderCone,
                      SumSquares, Zero)


def test_list_box_bounds_work_directly_and_in_repeated_instance_groups():
    from ocs.prox.library import Box
    from ocs.prox.groups import build_groups, grouped_moreau_dual
    block = Box([-1., 0.], [1., 2.])
    point = jnp.array([-3., 4.])
    np.testing.assert_allclose(block.prox(point, .5), [-1., 2.])
    np.testing.assert_allclose(jax.jit(block.prox)(point, .5), [-1., 2.])
    assert float(block.value(jnp.array([0., 1.]))) == 0.
    assert np.isinf(block.value(point))
    assert float(block.conj_value(jnp.array([-2., 3.]))) == 8.
    groups = build_groups([block, block], [(0, 2), (2, 4)])
    value = jnp.array([-3., 4., 2., -1.])
    steps = jnp.array([.5, .5, 1., 1.])
    actual = jax.jit(grouped_moreau_dual(groups, 4))(value, steps)
    np.testing.assert_allclose(actual, [-2.5, 3., 1., -1.])


def test_singular_quadratic_conjugate_checks_its_domain():
    # f(x)=x0^2+3*x0-2*x1. The conjugate is finite exactly at y1=-2.
    block = Quadratic(np.diag([2., 0.]), [3., -2.])
    assert float(block.conj_value(jnp.array([7., -2.]))) == pytest.approx(4.)
    assert np.isposinf(block.conj_value(jnp.array([7., -1.99])))
    assert float(Quadratic(np.zeros((2, 2))).conj_value(jnp.zeros(2))) == 0
    assert np.isposinf(Quadratic(np.zeros((2, 2))).conj_value(jnp.array([0., 1e-20])))


def test_small_positive_quadratic_curvature_is_preserved():
    block = Quadratic(np.diag([1., 1e-8]))
    assert float(block.conj_value(jnp.array([0., 1e-4]))) == pytest.approx(.5, rel=2e-6)


@pytest.mark.parametrize("units", [1e-6, 1., 1e6])
def test_singular_quadratic_fenchel_equality_in_objective_units(units):
    block = Quadratic(units*np.diag([2., 0.]), units*np.array([3., -2.]))
    point = jnp.array([2., 11.])
    dual = units*jnp.array([7., -2.])
    np.testing.assert_allclose(block.value(point)+block.conj_value(dual),
                               jnp.dot(point, dual), rtol=3e-6, atol=units*2e-6)


@pytest.mark.parametrize("P,q", [
    (np.ones((2, 3)), None), (np.array([[1., 1.], [0., 1.]]), None),
    (np.diag([-1., 2.]), None), (np.eye(2), [1.]),
    (np.diag([1., np.nan]), None), (np.eye(2), [0., np.inf]),
    (np.diag([1., -1e-6]), None),
    (np.array([[1., 1.000001], [1.000001, 1.]]), None),
])
def test_invalid_quadratic_data_rejected(P, q):
    with pytest.raises(ValueError):
        Quadratic(P, q)


def test_unbounded_box_support_handles_zero_multiplier():
    block = Box(jnp.array([-jnp.inf, 0., -2.]), jnp.array([jnp.inf, jnp.inf, 3.]))
    assert float(block.conj_value(jnp.array([0., -2., 4.]))) == 12.
    assert np.isposinf(block.conj_value(jnp.array([1., 0., 0.])))
    assert np.isposinf(block.conj_value(jnp.array([0., 1., 0.])))
    assert float(block.conj_value(jnp.zeros(3))) == 0


@pytest.mark.parametrize("lo,hi", [(1., 0.), (np.inf, np.inf),
                                    (-np.inf, -np.inf), (np.nan, 1.)])
def test_empty_or_invalid_box_rejected(lo, hi):
    with pytest.raises(ValueError):
        Box(lo, hi)


@pytest.mark.parametrize("block,point", [
    (L2(), [1., 2.]), (GroupL2(group_size=2), [1., 2.]),
    (SecondOrderCone(), [1., 2., 3.]), (PSDCone(2), [1., 0., 0., 2.]),
    (ExpCone(), [1., 2., 3.]), (PowerCone(), [1., 2., 3.]),
])
def test_coupled_blocks_reject_unsupported_diagonal_steps(block, point):
    point = jnp.array(point)
    steps = jnp.arange(1, len(point)+1, dtype=point.dtype)
    with pytest.raises(ValueError, match="uniform"):
        block.prox(point, steps)
    result = jax.jit(block.prox)(point, steps)
    assert not np.all(np.isfinite(result))
    np.testing.assert_allclose(block.prox(point, jnp.ones_like(point)),
                               block.prox(point, 1.), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("block", [L1(), NonNeg(), Equality(), Box(), Zero(), SumSquares()])
def test_invalid_traced_steps_cannot_return_valid_projection(block):
    point = jnp.array([1., 2.])
    with pytest.raises(ValueError):
        block.prox(point, 0.)
    result = jax.jit(block.prox)(point, jnp.array([1., -1.]))
    assert not np.all(np.isfinite(result))


@pytest.mark.parametrize("block", [Zero(), SumSquares(), Quadratic(np.eye(2))])
def test_matrix_metric_contract(block):
    point = jnp.array([1., 2.])
    metric = jnp.array([[2., .2], [.2, 3.]])
    expected = point if isinstance(block, Zero) else np.linalg.solve(
        np.eye(2)+np.asarray(metric), np.asarray(metric) @ np.asarray(point))
    np.testing.assert_allclose(block.prox_matrix(point, metric), expected, rtol=2e-6)
    with pytest.raises(ValueError, match="positive definite"):
        block.prox_matrix(point, jnp.diag(jnp.array([1., -1.])))
    result = jax.jit(block.prox_matrix)(point, jnp.diag(jnp.array([1., -1.])))
    assert not np.all(np.isfinite(result))
