"""Default relative stopping for a pure feasibility objective.

The returned zero dual is a valid witness once the independently measured
primal violation meets tolerance. Infeasible problems must retain separating
directions instead of having their duals erased by that recovery.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import ocs


SCALES = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
TOLERANCE = 1e-3


def _pure_equality(scale, max_blocks):
    diagonal = scale * jnp.asarray([1.0, 2.0])
    operator = ocs.LinearOperator.from_forward(lambda x: diagonal * x, n=2, m=2)
    variable = ocs.Variable(2)
    family = ocs.minimize([ocs.equality(operator @ variable - ocs.Parameter("b"))])
    compiled = family.compile(
        equilibrate="scalar", block=10, max_blocks=max_blocks, tol=TOLERANCE,
    )
    return compiled, {"b": diagonal}


@pytest.mark.parametrize("scale", SCALES)
def test_pure_equality_cold_solve_recovers_a_matching_zero_dual(scale):
    compiled, data = _pure_equality(scale, max_blocks=150)
    solution = compiled.solve_instance(data)
    report = compiled.certificate(solution.x, solution.y, data)

    assert compiled.meta["residual_norm"] == "relative"
    assert solution.status == "solved"
    assert solution.residual <= TOLERANCE
    assert report["r_domain"] <= TOLERANCE
    np.testing.assert_array_equal(np.asarray(solution.y), np.zeros(2))
    # Verify the original equalities independently in float64. A relative
    # residual tolerance does not imply the same coordinatewise solution error.
    actual = np.asarray(solution.x, dtype=np.float64)
    diagonal = np.asarray(data["b"], dtype=np.float64)
    image = diagonal * actual
    relative_violation = np.linalg.norm(image - diagonal) / max(
        np.linalg.norm(image), np.linalg.norm(diagonal),
    )
    assert relative_violation <= TOLERANCE * (1.0 + 2e-6)
    assert solution.residual == pytest.approx(max(report["r_primal"], report["r_dual"]), rel=2e-6)


@pytest.mark.parametrize("scale", SCALES)
def test_pure_equality_wrong_zero_is_not_accepted_without_iterations(scale):
    compiled, data = _pure_equality(scale, max_blocks=0)
    solution = compiled.solve_instance(data)
    report = compiled.certificate(solution.x, solution.y, data)

    assert solution.iters == 0
    assert solution.status == "inaccurate"
    assert solution.residual == pytest.approx(1.0, rel=2e-6)
    assert report["r_domain"] == pytest.approx(1.0, rel=2e-6)
    assert report["r_dual"] == pytest.approx(1.0, rel=2e-6)


def test_feasible_warm_start_can_recover_zero_dual_without_moving_primal():
    compiled, data = _pure_equality(1.0, max_blocks=0)
    point = jnp.ones(2)
    supplied_dual = jnp.asarray([100.0, -200.0])
    supplied_report = compiled.certificate(point, supplied_dual, data)
    solution = compiled.solve_instance(data, warm_start=(point, supplied_dual))
    returned_report = compiled.certificate(solution.x, solution.y, data)

    assert max(supplied_report["r_primal"], supplied_report["r_dual"]) > TOLERANCE
    assert solution.status == "solved"
    assert solution.iters == 0
    np.testing.assert_array_equal(np.asarray(solution.x), np.asarray(point))
    np.testing.assert_array_equal(np.asarray(solution.y), np.zeros(2))
    assert solution.residual == 0.0
    assert max(returned_report["r_primal"], returned_report["r_dual"]) == 0.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_feasibility_dual_recovery_does_not_hide_invalid_warm_dual(bad_value):
    compiled, data = _pure_equality(1.0, max_blocks=0)
    solution = compiled.solve_instance(
        data, warm_start=(jnp.ones(2), jnp.full(2, bad_value)),
    )
    assert solution.status == "numerical_error"
    assert not np.all(np.isfinite(np.asarray(solution.y)))


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("cold_start", [False, True])
def test_zero_cost_infeasible_problem_retains_a_valid_separating_direction(scale, cold_start):
    # x >= 0 cannot satisfy sum(x) = -1. A positive equality multiplier is a
    # Farkas direction: -A.T d <= 0 and l.T d > 0 for l=scale.
    operator = ocs.LinearOperator.from_forward(
        lambda x: scale * jnp.sum(x, keepdims=True), n=2, m=1,
    )
    variable = ocs.Variable(2)
    family = ocs.minimize([
        ocs.nonneg(variable), ocs.equality(operator @ variable - ocs.Parameter("b")),
    ])
    compiled = family.compile(
        equilibrate="scalar", block=10, max_blocks=10 if cold_start else 0,
        tol=TOLERANCE,
    )
    warm_start = None if cold_start else (jnp.zeros(2), jnp.ones(1))
    data = {"b": jnp.asarray([-scale])}
    solution = compiled.solve_instance(data, warm_start=warm_start)

    assert solution.status == "primal_infeasible"
    assert solution.certificate is not None
    assert np.linalg.norm(np.asarray(solution.y)) > 0.0
    direction = np.asarray(solution.certificate, dtype=np.float64)
    assert np.all(np.isfinite(direction))
    assert np.linalg.norm(direction) == pytest.approx(1.0, rel=2e-6)
    assert float(scale * direction[0]) > 0.0
    assert np.all(-scale * direction[0] * np.ones(2) <= 0.0)
    report = compiled.certificate(solution.x, solution.y, data)
    assert report["r_domain"] > TOLERANCE
