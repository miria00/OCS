"""Public-API requirements for default, data-relative termination.

Same-point tests exercise the certificate before any iteration. Separate solve
tests use equivalent safe iteration metrics under a change of data units; they
do not assume every preconditioner produces the same iteration count.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import ocs

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the relative residual and its scaling conventions.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


SCALES = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
DIAGONAL = np.asarray([1.0, 2.0], dtype=np.float32)
TOLERANCE = 1e-3


def _operator(diagonal):
    d = jnp.asarray(diagonal, dtype=jnp.float32)
    return ocs.LinearOperator.from_forward(lambda x: d * x, n=2, m=2)


def _make_lp(constraint_scale=1.0, objective_scale=1.0, *, target=None,
             steps=None, mode="relative", max_blocks=0, tol=TOLERANCE):
    """Nonnegative equality LP with known solution and transformed dual."""
    variable = ocs.Variable(2)
    rhs = ocs.Parameter("b")
    diagonal = constraint_scale * DIAGONAL
    cost = objective_scale * DIAGONAL
    family = ocs.minimize(
        [ocs.nonneg(variable), ocs.equality(_operator(diagonal) @ variable - rhs)],
        cost=jnp.asarray(cost),
    )
    if steps is None:
        steps = (0.1 / objective_scale, 0.1 * objective_scale / constraint_scale ** 2)
    tau, sigma = steps
    options = dict(
        block=20, max_blocks=max_blocks, tol=tol,
        equilibrate=("given", jnp.full(2, tau), jnp.full(2, sigma)),
    )
    if mode is not None:
        options["residual_norm"] = mode
    compiled = family.compile(**options)
    optimum = np.ones(2, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32)
    data = {"b": jnp.asarray(diagonal * optimum)}
    dual = jnp.asarray(-cost / diagonal)
    return compiled, data, jnp.asarray(optimum), dual


def _make_ls(scale=1.0, *, mode="relative", max_blocks=0):
    variable = ocs.Variable(2)
    rhs = ocs.Parameter("b")
    family = ocs.minimize([ocs.least_squares(_operator(scale * DIAGONAL) @ variable - rhs)])
    options = dict(
        block=20, max_blocks=max_blocks, tol=1e-4,
        equilibrate=("given", jnp.full(2, 0.05 / scale ** 2), jnp.full(2, 0.5)),
    )
    if mode is not None:
        options["residual_norm"] = mode
    compiled = family.compile(**options)
    return compiled, {"b": jnp.asarray(scale * DIAGONAL)}


def _reported_residual(report):
    return max(report["r_primal"], report["r_dual"])


def test_public_compile_defaults_to_relative_stopping():
    compiled, _, _, _ = _make_lp(mode=None)
    assert compiled.meta["residual_norm"] == "relative"


@pytest.mark.parametrize("scale", SCALES)
def test_default_does_not_accept_zero_for_a_nonzero_rescaled_rhs(scale):
    # These same steps are safe at every scale, including ||K|| <= 2e6.
    # A small algorithm step must not turn a unit relative violation into solved.
    compiled, data, _, _ = _make_lp(
        constraint_scale=scale, steps=(1e-14, 0.1), mode=None,
    )
    solution = compiled.solve_instance(data)
    report = compiled.certificate(solution.x, solution.y, data)

    assert solution.iters == 0
    assert solution.status != "solved"
    assert report["r_domain"] == pytest.approx(1.0, rel=2e-6)
    assert solution.residual == pytest.approx(_reported_residual(report), rel=2e-6)


@pytest.mark.parametrize("kind", ["constraint", "objective"])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("delta, accepted", [(1e-4, True), (1e-2, False)])
def test_same_point_acceptance_is_invariant_to_constraint_and_objective_units(kind, scale, delta, accepted):
    compiled, data, optimum, dual = _make_lp(
        constraint_scale=scale if kind == "constraint" else 1.0,
        objective_scale=scale if kind == "objective" else 1.0,
    )
    point = (1.0 + delta) * optimum
    report = compiled.certificate(point, dual, data)
    solution = compiled.solve_instance(data, warm_start=(point, dual))

    assert solution.iters == 0
    assert (solution.status == "solved") == accepted
    assert (_reported_residual(report) <= TOLERANCE) == accepted
    assert report["r_domain"] == pytest.approx(delta / (1.0 + delta), rel=0.01, abs=5e-7)
    assert solution.residual == pytest.approx(_reported_residual(report), rel=2e-6)


@pytest.mark.parametrize("steps", [(1e-8, 1e-3), (0.01, 1e-3), (1.0, 1e-3),
                                   (10.0, 1e-3), (0.01, 1e-8), (0.01, 0.1)])
def test_same_point_certificate_does_not_depend_on_iteration_metrics(steps):
    baseline, data, optimum, dual = _make_lp(steps=(0.01, 1e-3))
    candidate, _, _, _ = _make_lp(steps=steps)
    # This x is feasible, but y is deliberately not stationary. A large primal
    # algorithm step crosses the NonNeg boundary; it must not change certification.
    dual = 0.5 * dual
    reference = baseline.certificate(optimum, dual, data)
    report = candidate.certificate(optimum, dual, data)

    for name in ("r_primal", "r_dual", "r_domain", "r_polar", "r_complementarity"):
        assert report[name] == pytest.approx(reference[name], rel=2e-6, abs=2e-7), name
    assert _reported_residual(report) > TOLERANCE


@pytest.mark.parametrize("multiplier_scale", [1.0, 2.0 ** 25, 1e20])
def test_large_multiplier_does_not_hide_actual_constraint_violation(multiplier_scale):
    compiled, data, optimum, dual = _make_lp(
        objective_scale=multiplier_scale, steps=(0.01, 0.01),
    )
    point = 1.1 * optimum
    report = compiled.certificate(point, dual, data)

    assert report["r_domain"] == pytest.approx(0.1 / 1.1, rel=2e-6)
    assert _reported_residual(report) > TOLERANCE


def _make_two_block_lp(scales=(1.0, 1.0), target=(1.0, 1.0), *, mode="relative"):
    variable = ocs.Variable(2)
    terms = [ocs.nonneg(variable)]
    data = {}
    for coordinate, scale in enumerate(scales):
        operator = ocs.LinearOperator.from_forward(
            lambda x, coordinate=coordinate, scale=scale: scale * x[coordinate:coordinate + 1],
            n=2, m=1,
        )
        name = f"b{coordinate}"
        terms.append(ocs.equality(operator @ variable - ocs.Parameter(name)))
        data[name] = jnp.asarray([scale * target[coordinate]])
    family = ocs.minimize(terms, cost=jnp.ones(2))
    options = dict(max_blocks=0, tol=TOLERANCE,
                   equilibrate=("given", jnp.full(2, 0.1),
                                jnp.asarray([0.1 / scale ** 2 for scale in scales])))
    if mode is not None:
        options["residual_norm"] = mode
    return family.compile(**options), data, jnp.asarray(target), -1.0 / jnp.asarray(scales)


@pytest.mark.parametrize("scales", [(1.0, 1.0), (1e-6, 1.0), (1.0, 1e6),
                                    (1e-6, 1e6), (1e6, 1e-6)])
@pytest.mark.parametrize("delta, accepted", [(1e-4, True), (1e-2, False)])
def test_each_equality_block_can_change_units_independently(scales, delta, accepted):
    compiled, data, optimum, dual = _make_two_block_lp(scales)
    solution = compiled.solve_instance(data, warm_start=((1.0 + delta) * optimum, dual))
    report = compiled.certificate(solution.x, solution.y, data)

    assert (solution.status == "solved") == accepted
    assert report["r_domain"] == pytest.approx(delta / (1.0 + delta), rel=0.01, abs=5e-7)


def test_large_feasible_offset_cannot_mask_a_separate_violated_block():
    compiled, data, optimum, dual = _make_two_block_lp(target=(1e6, 1.0))
    point = optimum.at[1].set(1.01)
    solution = compiled.solve_instance(data, warm_start=(point, dual))
    report = compiled.certificate(solution.x, solution.y, data)

    assert solution.status != "solved"
    assert report["r_domain"] == pytest.approx(0.01 / 1.01, rel=1e-4)


@pytest.mark.parametrize("warm_value, accepted", [(0.0, True), (1.0, False)])
def test_zero_data_feasibility_checks_the_actual_warm_start(warm_value, accepted):
    compiled, data, _, dual = _make_lp(target=np.zeros(2))
    point = jnp.full(2, warm_value)
    solution = compiled.solve_instance(data, warm_start=(point, dual))

    assert solution.iters == 0
    assert (solution.status == "solved") == accepted
    if accepted:
        assert solution.residual == 0.0
    else:
        assert solution.residual > TOLERANCE


@pytest.mark.parametrize("scale", SCALES)
def test_known_least_squares_optimum_and_wrong_zero_are_distinguished(scale):
    compiled, data = _make_ls(scale)
    wrong = compiled.solve_instance(data)
    halfway = compiled.solve_instance(data, warm_start=(jnp.full(2, 0.5), jnp.zeros(2)))
    exact = compiled.solve_instance(data, warm_start=(jnp.ones(2), jnp.zeros(2)))

    assert wrong.status != "solved"
    assert wrong.residual == pytest.approx(1.0, rel=2e-6)
    assert halfway.status != "solved"
    assert halfway.residual == pytest.approx(0.5, rel=2e-6)
    assert exact.status == "solved"
    assert exact.residual <= 1e-4


@pytest.mark.parametrize("kind", ["constraint", "objective"])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_cold_start_lp_converges_with_equivalent_safe_iteration_metrics(kind, scale):
    compiled, data, optimum, _ = _make_lp(
        constraint_scale=scale if kind == "constraint" else 1.0,
        objective_scale=scale if kind == "objective" else 1.0,
        max_blocks=250,
    )
    solution = compiled.solve_instance(data)

    assert solution.status == "solved"
    assert solution.residual <= TOLERANCE
    np.testing.assert_allclose(np.asarray(solution.x), np.asarray(optimum), rtol=2e-3, atol=0)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_cold_start_least_squares_converges_with_equivalent_safe_iteration_metrics(scale):
    compiled, data = _make_ls(scale, max_blocks=250)
    solution = compiled.solve_instance(data)

    assert solution.status == "solved"
    assert solution.residual <= 1e-4
    np.testing.assert_allclose(np.asarray(solution.x), np.ones(2), rtol=5e-4, atol=0)


@pytest.mark.parametrize("fraction", [-10.0, 0.0, 0.5, 1.0, 2.0, 10.0])
def test_cg_and_pdhg_certify_the_same_least_squares_point_identically(fraction):
    variable = ocs.Variable(2)
    rhs = ocs.Parameter("b")
    family = ocs.minimize([ocs.least_squares(_operator(np.ones(2)) @ variable - rhs)])
    cg = family.compile(method="cg", cg_iters=0)
    pdhg = family.compile(method="pdhg", max_blocks=0, equilibrate="scalar", residual_norm="relative")
    data = {"b": jnp.asarray([1.0, 2.0])}
    point = fraction * data["b"]
    recovered_dual = point - data["b"]

    cg_report = cg.certificate(point, data=data)
    pdhg_report = pdhg.certificate(point, recovered_dual, data)
    cg_solution = cg.solve_instance(data, warm_start=(point, None))
    pdhg_solution = pdhg.solve_instance(data, warm_start=(point, recovered_dual))

    assert cg_report["r_primal"] == pytest.approx(_reported_residual(pdhg_report), rel=2e-6, abs=1e-7)
    assert cg_solution.residual == pytest.approx(pdhg_solution.residual, rel=2e-6, abs=1e-7)
    assert cg_solution.status == pdhg_solution.status
