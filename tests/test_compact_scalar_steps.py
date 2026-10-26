"""Verified scalar metric dispatch keeps block parameters and rejection rules."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.groups import (_verified_scalar_groups, build_groups,
                             grouped_moreau_dual)


@pytest.mark.parametrize("omega", [.1, 1., 4.])
def test_heterogeneous_coupled_wrappers_keep_distinct_uniform_steps(omega):
    values = np.array([[2., -1., .3, 1.2], [-.2, 3., -2., .4]])
    parameters = [(.7, 2., np.array([2., -2., 2., -2.]),
                   np.array([.2, -.5, .1, .8]), np.array([.1, -.3, .2, .4]), .3),
                  (1.3, .5, np.array([-1., 1., -1., 1.]),
                   np.array([-.4, .8, .2, -.3]), np.array([-.2, .4, .1, -.1]), 1.1)]
    blocks = [ocs.L2(lam).scaled(weight).precomposed(a, b).with_linear_quadratic(d, e)
              for lam, weight, a, b, d, e in parameters]
    groups = build_groups(blocks, [(0, 4), (4, 8)])
    assert len(groups) == 1
    sigma = jnp.repeat(jnp.array([.25, 1.5]), 4)
    assert _verified_scalar_groups(groups, 8, sigma) == (True,)
    actual = jax.jit(grouped_moreau_dual(groups, 8, static_sigma=sigma))(
        jnp.asarray(values.ravel(), jnp.float32), sigma * omega)
    expected = []
    for row, step, (lam, weight, a, b, d, e) in zip(
            values, np.array([.25, 1.5])*omega, parameters):
        center = b / a
        delta = (row-d)/(step+e)-center
        shrink = max(0., 1.-lam*weight*abs(a[0])/((step+e)*np.linalg.norm(delta)))
        expected.extend(row-step*(center+shrink*delta))
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=2e-6)


def test_mixed_metric_capabilities_keep_diagonal_rows_and_group_l2_radii():
    blocks = [ocs.SumSquares(), ocs.GroupL2(.2, 3), ocs.GroupL2(.9, 3)]
    groups = build_groups(blocks, [(0, 3), (3, 9), (9, 15)])
    sigma = jnp.asarray([.1, .4, 1.3] + [.2]*6 + [1.5]*6)
    assert _verified_scalar_groups(groups, 15, sigma) == (False, True)
    values = np.linspace(-2., 3., 15, dtype=np.float32)
    actual = jax.jit(grouped_moreau_dual(groups, 15, static_sigma=sigma))(
        jnp.asarray(values), sigma)
    expected = list(values[:3]/(1+np.asarray(sigma[:3])))
    for rows, radius in [(values[3:9].reshape(-1, 3), .2),
                         (values[9:15].reshape(-1, 3), .9)]:
        expected.extend((rows*np.minimum(1., radius/np.linalg.norm(
            rows, axis=1, keepdims=True))).ravel())
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-7)


@pytest.mark.parametrize("bad", [np.nextafter(np.float32(1), np.float32(2)),
                                 0., -1., np.inf, np.nan])
def test_invalid_member_disables_compaction_without_sanitizing_metric(bad):
    groups = build_groups([ocs.L2(.7), ocs.L2(1.3)], [(0, 2), (2, 4)])
    sigma = jnp.asarray([.5, .5, 1., bad])
    assert _verified_scalar_groups(groups, 4, sigma) == (False,)
    values = jnp.array([2., 1., -1., 3.])
    compact = jax.jit(grouped_moreau_dual(groups, 4, static_sigma=sigma))(values, sigma)
    ordinary = jax.jit(grouped_moreau_dual(groups, 4))(values, sigma)
    np.testing.assert_allclose(compact[:2], .7*np.array([2., 1.])/np.sqrt(5), atol=3e-7)
    assert np.isnan(np.asarray(compact[2:])).all()
    np.testing.assert_allclose(compact, ordinary, equal_nan=True, atol=0)


def test_standalone_default_retains_runtime_vector_validation():
    groups = build_groups([ocs.GroupL2(.7, 2)], [(0, 4)])
    # Enabling a separate specialized callable must not mutate these groups.
    grouped_moreau_dual(groups, 4, static_sigma=jnp.ones(4))
    result = jax.jit(grouped_moreau_dual(groups, 4))(
        jnp.ones(4), jnp.asarray([1., 1., 1., 2.]))
    assert np.isnan(np.asarray(result)).all()


@pytest.mark.parametrize("omega", [0., -1., np.inf, np.nan])
def test_invalid_scalar_multiplier_still_fails_prox_validation(omega):
    groups = build_groups([ocs.L2(.7).scaled(2.)], [(0, 4)])
    sigma = jnp.full(4, .5)
    result = jax.jit(grouped_moreau_dual(groups, 4, static_sigma=sigma))(
        jnp.ones(4), sigma*omega)
    assert np.isnan(np.asarray(result)).all()


@pytest.mark.parametrize("sigma_value,weight", [(1e-38, 1e10), (1e38, 1e-10)])
def test_calculus_transformed_metric_overflow_or_underflow_remains_invalid(sigma_value, weight):
    groups = build_groups([ocs.L2().scaled(weight)], [(0, 2)])
    sigma = jnp.full(2, sigma_value)
    compact = jax.jit(grouped_moreau_dual(groups, 2, static_sigma=sigma))(
        jnp.ones(2), sigma)
    ordinary = jax.jit(grouped_moreau_dual(groups, 2))(jnp.ones(2), sigma)
    assert np.isnan(np.asarray(compact)).all()
    assert np.isnan(np.asarray(ordinary)).all()


@pytest.mark.parametrize("bad", [np.nextafter(np.float32(.2), np.float32(1)),
                                 0., -1., np.inf, np.nan])
def test_given_invalid_scalar_metric_cannot_become_a_solved_program(bad):
    family = ProblemFamily(n=2, terms=[
        Term(ocs.Identity(2), ocs.SumSquares()),
        Term(ocs.Dense(np.eye(2)), ocs.GroupL2(.7, 2),
             lambda data: -jnp.ones(2)),
    ])
    # Formerly this metric reached the prox traced and returned NaN with a
    # numerical_error status. It is now refused eagerly at compile time, which
    # is the stronger form of "cannot become a solved program".
    with pytest.raises(ValueError, match="sigma"):
        family.compile(
            equilibrate=("given", jnp.full(2, .1), jnp.asarray([.2, bad])),
            block=2, max_blocks=1, tol=1e-4,
        )


def test_large_group_metric_validation_reduces_only_one_step_per_block():
    size = 3*2048
    groups = build_groups([ocs.GroupL2(.2, 3)], [(0, size)])
    sigma = jnp.full(size, .4)
    compact = grouped_moreau_dual(groups, size, static_sigma=sigma)
    ordinary = grouped_moreau_dual(groups, size)

    def validation_sizes(function):
        jaxpr = jax.make_jaxpr(function)(jnp.ones(size), sigma).jaxpr
        return [int(np.prod(eqn.invars[0].aval.shape)) for eqn in jaxpr.eqns
                if eqn.primitive.name == "reduce_and"]

    assert max(validation_sizes(ordinary)) == size
    assert max(validation_sizes(compact)) == 1
