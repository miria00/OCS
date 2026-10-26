"""Analytic domain-aware status certificates, with operator callbacks only."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.base import ProxBlock
from ocs.prox.cones import ExpCone, PSDCone, SecondOrderCone, SOCBatch
from ocs.prox.library import Box, Equality, NonNeg, SumSquares
from ocs.solver.certificates import make_status_checks


def _arr(values):
    return jnp.asarray(values, dtype=jnp.float32)


def test_nonnegative_infeasible_equality_needs_positive_adjoint():
    # x >= 0, sum(x) + 1 = 0.  d=1 is valid and K.T d is POSITIVE,
    # because the primal nonnegative constraint supplies the other normal.
    check = make_status_checks(
        lambda x: jnp.sum(x, keepdims=True),
        lambda y: jnp.full((3,), y[0]), NonNeg(), [Equality()], [(0, 1)], 3, 1,
    )
    result = check(_arr([0, 0, 0]), _arr([1]), (_arr([1]), _arr([0, 0, 0])))
    assert bool(result["primal_infeasible"])
    assert not bool(result["unbounded"])
    assert float(result["primal_infeasibility_residual"]) == 0
    np.testing.assert_allclose(result["dual_direction"], [1])


@pytest.mark.parametrize("g,sign,x", [(None, 1.0, -1.0), (NonNeg(), -1.0, 1.0)])
def test_feasible_large_dual_and_unit_cosine_is_not_farkas(g, sign, x):
    # Both systems have a feasible point.  <l,y>/(||l|| ||y||)=1 and the
    # dual is enormous, yet the coupled primal support condition fails.
    check = make_status_checks(lambda x: sign * x, lambda y: sign * y,
                               g, [Equality()], [(0, 1)], 1, 1)
    result = check(_arr([x]), _arr([1e12]), (_arr([1]), _arr([0])))
    assert not bool(result["primal_infeasible"])
    assert float(result["primal_infeasibility_residual"]) > 0.9


def test_unconstrained_primal_requires_adjoint_nullspace():
    # x=0 and x=1 cannot both hold.  d=(-1,1) has K.T d=0 and l.T d>0.
    check = make_status_checks(lambda x: jnp.repeat(x, 2),
                               lambda y: jnp.sum(y, keepdims=True),
                               None, [Equality()], [(0, 2)], 1, 2)
    result = check(_arr([0]), _arr([1, -1]), (_arr([0, -1]), _arr([0])))
    assert bool(result["primal_infeasible"])
    assert float(result["primal_infeasibility_residual"]) < 1e-6


def test_box_support_is_subtracted_from_separating_margin():
    # x in [0,1], x=2: d=-1 gives l.T d=2 and support_[0,1](1)=1.
    check = make_status_checks(lambda x: x, lambda y: y,
                               Box(0, 1), [Equality()], [(0, 1)], 1, 1)
    bad = check(_arr([1]), _arr([-1]), (_arr([-2]), _arr([0])))
    assert bool(bad["primal_infeasible"])
    np.testing.assert_allclose(bad["primal_infeasibility_margin"], 0.5)
    feasible = check(_arr([0.5]), _arr([-1e8]), (_arr([-0.5]), _arr([0])))
    assert not bool(feasible["primal_infeasible"])


def test_dual_box_domain_has_finite_nonzero_support():
    check = make_status_checks(lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
                               None, [Box(0, 1)], [(0, 1)], 1, 1)
    bad = check(_arr([0]), _arr([1]), (_arr([2]), _arr([0])))
    assert bool(bad["primal_infeasible"])
    good = check(_arr([0]), _arr([1]), (_arr([0.5]), _arr([0])))
    assert not bool(good["primal_infeasible"])


@pytest.mark.parametrize("cone,l,d", [
    (SecondOrderCone(), [-1, 0, 0], [-1, 0, 0]),
    (SOCBatch(2, 2), [-1, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]),
    (PSDCone(2), [-1, 0, 0, 1], [-1, 0, 0, 0]),
])
def test_conic_separation_uses_polar(cone, l, d):
    m = len(l)
    check = make_status_checks(lambda x: jnp.zeros(m), lambda y: jnp.zeros(1),
                               None, [cone], [(0, m)], 1, m)
    result = check(_arr([0]), _arr(d), (_arr(l), _arr([0])))
    assert bool(result["primal_infeasible"])
    assert float(result["primal_infeasibility_residual"]) < 1e-6


def test_full_domain_objective_rows_supply_no_separating_direction():
    check = make_status_checks(lambda x: jnp.concatenate((x, x)),
                               lambda y: y[:1] + y[1:], NonNeg(),
                               [Equality(), SumSquares()], [(0, 1), (1, 2)], 1, 2)
    result = check(_arr([0]), _arr([1, 1e8]), (_arr([1, -100]), _arr([0])))
    assert bool(result["primal_infeasible"])
    np.testing.assert_allclose(result["dual_direction"], [1, 0])
    assert not bool(result["recession_supported"])
    assert not bool(result["unbounded"])


def test_true_unbounded_lp_has_a_feasible_base_and_valid_recession_ray():
    check = make_status_checks(lambda x: x[:1] - x[1:],
                               lambda y: jnp.concatenate((y, -y)), NonNeg(),
                               [Equality()], [(0, 1)], 2, 1)
    result = check(_arr([3, 3]), _arr([0]), (_arr([0]), _arr([-1, -1])))
    assert bool(result["unbounded"])
    assert not bool(result["primal_infeasible"])
    np.testing.assert_allclose(result["feasible_base"], [0, 0])
    assert float(result["base_feasibility_residual"]) == 0
    assert float(result["recession_residual"]) == 0


def test_nonzero_feasible_base_can_certify_shifted_cone_problem():
    # min -x, x-1 >= 0.  Zero is infeasible but the returned x=1 is feasible.
    check = make_status_checks(lambda x: x, lambda y: y, None,
                               [NonNeg()], [(0, 1)], 1, 1)
    result = check(_arr([1]), _arr([0]), (_arr([-1]), _arr([-1])))
    assert bool(result["unbounded"])
    np.testing.assert_allclose(result["feasible_base"], [1])


def test_recession_direction_does_not_make_infeasible_problem_unbounded():
    # x >= 0, 0*x+1=0 has decreasing recession directions but NO feasible base.
    check = make_status_checks(lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
                               NonNeg(), [Equality()], [(0, 1)], 1, 1)
    result = check(_arr([100]), _arr([0]), (_arr([1]), _arr([-1])))
    assert float(result["recession_residual"]) == 0
    assert float(result["recession_margin"]) > 0
    assert float(result["base_feasibility_residual"]) == 1
    assert not bool(result["unbounded"])


def test_finite_box_has_no_nonzero_recession_direction():
    check = make_status_checks(lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
                               Box(0, 1), [Equality()], [(0, 1)], 1, 1)
    result = check(_arr([1]), _arr([0]), (_arr([0]), _arr([-1])))
    assert not bool(result["unbounded"])
    assert float(result["recession_residual"]) == 1


def test_recession_must_respect_primal_nonnegative_domain():
    check = make_status_checks(lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
                               NonNeg(), [Equality()], [(0, 1)], 1, 1)
    result = check(_arr([-100]), _arr([0]), (_arr([0]), _arr([1])))
    assert not bool(result["unbounded"])
    assert float(result["recession_residual"]) == 1


@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e5])
def test_equivalent_equation_scaling_preserves_certificate(scale):
    check = make_status_checks(lambda x: scale * x, lambda y: scale * y,
                               NonNeg(), [Equality()], [(0, 1)], 1, 1)
    result = check(_arr([0]), _arr([1 / scale]), (_arr([scale]), _arr([0])))
    assert bool(result["primal_infeasible"])
    np.testing.assert_allclose(result["primal_infeasibility_margin"], 1)


@pytest.mark.parametrize("block", [ProxBlock(), ExpCone().precomposed(a=2.0, b=1.0)])
def test_unsupported_domain_returns_false_without_oracle_calls(block):
    def unavailable(_):
        raise AssertionError("unsupported certificates must not inspect the operator")
    check = make_status_checks(unavailable, unavailable, None, [block], [(0, 3)], 1, 3)
    result = check(_arr([1]), _arr([1, 1, 1]), (_arr([1, 1, 1]), _arr([-1])))
    assert not bool(result["primal_infeasibility_supported"])
    assert not bool(result["primal_infeasible"])
    assert not bool(result["unbounded"])


def test_nan_inputs_never_certify_and_checks_can_be_vmapped():
    check = make_status_checks(lambda x: x, lambda y: y, NonNeg(),
                               [Equality()], [(0, 1)], 1, 1)
    results = jax.vmap(check)(_arr([[0], [jnp.nan]]), _arr([[1], [1]]),
                              (_arr([[1], [1]]), _arr([[0], [0]])))
    np.testing.assert_array_equal(results["primal_infeasible"], [True, False])
    np.testing.assert_array_equal(results["unbounded"], [False, False])
