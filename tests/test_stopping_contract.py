"""Termination reports must describe the returned operator-only solve.

Run on CPU while GPU scale experiments are active::

    JAX_PLATFORMS=cpu python -m pytest tests/test_stopping_contract.py
"""

import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.model.compile import CompiledFamily
from ocs.solver.composite_pdhg import make_online_solver, make_osgm_solver, make_solver
from ocs.solver.stopping import make_relative_components, relative_ratio

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the residual and termination reporting themselves.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def _diagonal_operator(diagonal=(1.0, 2.0, 4.0)):
    diagonal = jnp.asarray(diagonal, dtype=jnp.float32)
    return ocs.LinearOperator.from_forward(
        lambda x: diagonal * x, n=diagonal.size, m=diagonal.size
    )


def _least_squares_family():
    x = ocs.Variable(3)
    b = ocs.Parameter("b")
    return ocs.minimize([ocs.least_squares(_diagonal_operator() @ x - b)])


@pytest.mark.parametrize("normalization", ["iterate", "data"])
@pytest.mark.parametrize("smooth", [False, True])
def test_certificate_matches_returned_residual(normalization, smooth):
    x = ocs.Variable(3)
    b = ocs.Parameter("b")
    expression = _diagonal_operator() @ x - b
    loss = ocs.least_squares(expression) if smooth else ocs.l1(expression, 0.25)
    family = ocs.minimize(
        [ocs.box(x, lo=-2.0, hi=2.0), loss],
        cost=jnp.asarray([0.3, -0.1, 0.2], dtype=jnp.float32),
    )
    compiled = family.compile(
        block=3, max_blocks=2, tol=1e-10,
        equilibrate=("given", jnp.full(3, 0.1), jnp.full(3, 0.1), 0.4),
        residual_norm=normalization,
    )
    data = {"b": jnp.asarray([2.0, -1.0, 0.5], dtype=jnp.float32)}
    solution = compiled.solve_instance(data)
    report = compiled.certificate(solution.x, solution.y, data)
    reported = max(report["r_primal"], report["r_dual"])

    assert np.isfinite(solution.residual)
    assert reported == pytest.approx(solution.residual, rel=2e-6, abs=1e-8)


@pytest.mark.parametrize("normalization", ["iterate", "data"])
def test_smooth_certificate_uses_the_same_recovered_dual_as_stopping(normalization):
    compiled = _least_squares_family().compile(
        max_blocks=0, equilibrate="scalar", residual_norm=normalization,
    )
    b = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32)
    # x is the exact minimizer, even though the carried smooth dual is stale.
    solution = compiled.solve_instance(
        {"b": b}, warm_start=(jnp.ones(3), jnp.asarray([3.0, -2.0, 1.0]))
    )
    report = compiled.certificate(solution.x, solution.y, {"b": b})

    assert solution.residual == pytest.approx(0.0, abs=1e-7)
    assert max(report["r_primal"], report["r_dual"]) == pytest.approx(
        solution.residual, abs=1e-7
    )


@pytest.mark.parametrize("method", ["pdhg", "cg"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_instance_data_cannot_be_solved(method, bad_value):
    compiled = _least_squares_family().compile(
        method=method, equilibrate="scalar", block=2, max_blocks=2, cg_iters=2,
    )
    data = {"b": jnp.asarray([bad_value, 1.0, 2.0], dtype=jnp.float32)}
    try:
        solution = compiled.solve_instance(data)
    except ValueError:
        # Rejecting invalid input before entering the solver is also acceptable.
        return
    assert solution.status != "solved"


@pytest.mark.parametrize("invalid_part", ["x", "y", "data"])
def test_small_scalar_residual_cannot_certify_nonfinite_state(invalid_part):
    # Model a solver returning a misleading scalar residual. The public status
    # still has to reject nonfinite iterates/data rather than trusting that scalar.
    def solver(data, x0, y0):
        x = jnp.full_like(x0, jnp.nan) if invalid_part == "x" else x0
        y = jnp.full_like(y0, jnp.inf) if invalid_part == "y" else y0
        return x, y, jnp.asarray([0]), jnp.asarray([0.0])

    def resolve(_raw):
        offset = jnp.asarray([jnp.nan if invalid_part == "data" else 0.0])
        return offset, jnp.zeros(1)

    compiled = CompiledFamily(
        solver, resolve, None, n=1, m=1, meta={"block": 1}, tol=1e-3,
    )
    try:
        solution = compiled.solve_instance()
    except ValueError:
        return
    assert solution.status != "solved"


@pytest.mark.parametrize("iterations, expected_status", [(1, "inaccurate"), (8, "solved")])
def test_cg_status_reflects_achieved_accuracy(iterations, expected_status):
    compiled = _least_squares_family().compile(method="cg", cg_iters=iterations, tol=1e-5)
    data = {"b": jnp.ones(3)}
    solution = compiled.solve_instance(data)
    certificate = compiled.certificate(solution.x, data=data)

    assert solution.status == expected_status
    assert (certificate["r_primal"] <= 1e-5) == (expected_status == "solved")
    assert solution.residual == pytest.approx(certificate["r_primal"], abs=1e-6)


@pytest.mark.parametrize("cost", [0.0, 0.25])
def test_gap_is_not_exact_without_primal_stationarity_when_g_is_absent(cost):
    x = ocs.Variable(3)
    family = ocs.minimize(
        [ocs.least_squares(_diagonal_operator() @ x)],
        cost=jnp.full(3, cost),
    )
    compiled = family.compile(equilibrate="scalar", max_blocks=0)

    # All quadratic block values/conjugates are finite, but g=0 also requires
    # K^T y + c = 0 for dual feasibility. A finite surrogate alone is insufficient.
    report = compiled.certificate(jnp.zeros(3), jnp.ones(3))
    assert np.isfinite(report["gap"])
    assert not report["gap_exact"]

    dual_feasible = -jnp.full(3, cost) / jnp.asarray([1.0, 2.0, 4.0])
    feasible_report = compiled.certificate(jnp.zeros(3), dual_feasible)
    assert feasible_report["gap_exact"]


def test_termination_cannot_accept_a_residual_from_before_the_returned_block():
    # A controlled map isolates the loop contract: the initial point passes,
    # while every subsequent block moves away. The engine must retain a passing
    # point or exhaust its budget, not return a moved point as an early stop.
    identity = lambda v: v
    residual = lambda x, _y, _data: jnp.linalg.norm(x)
    solver = make_solver(
        identity, identity, jnp.ones(1), jnp.ones(1),
        lambda x, _tau, _data: x + 2.0,
        lambda y, _sigma, _data: jnp.zeros_like(y),
        residual, 1, 1, block=1, max_blocks=4, tol=1e-3,
        adaptive_omega=False,
    )
    x, y, blocks, reported = solver(
        (jnp.zeros((1, 1)), jnp.zeros((1, 1))),
        jnp.zeros((1, 1)), jnp.zeros((1, 1)),
    )
    assert float(reported[0]) == pytest.approx(float(jnp.linalg.norm(x[0])))
    assert float(reported[0]) <= 1e-3 or int(blocks[0]) == 4


def _equality_residual_components(scale=1.0, slices=((0, 1),)):
    return make_relative_components(
        lambda x: scale * x, lambda y: scale * y,
        None, [ocs.Equality() for _ in slices], slices,
    )


@pytest.mark.parametrize("scale", [2.0 ** -12, 1.0, 2.0 ** 12])
def test_relative_equality_violation_is_invariant_to_constraint_units(scale):
    x = jnp.asarray([1.5])
    target = jnp.asarray([1.0])
    # Scaling K,l by a positive alpha describes exactly the same constraint.
    # Its dual transforms contragrediently; no algorithm metric enters this test.
    evaluate = _equality_residual_components(scale=scale)
    report = evaluate(x, jnp.asarray([2.0 / scale]), (-scale * target, jnp.zeros(1)))
    distance = float(jnp.linalg.norm(x - target))
    relative_distance = distance / max(float(jnp.linalg.norm(x)), float(jnp.linalg.norm(target)))

    assert float(report["r_dual"]) == pytest.approx(relative_distance, rel=2e-6)
    assert float(report["r_dual"]) > 1e-3


@pytest.mark.parametrize("multiplier", [0.0, 32.0, 2.0 ** 20])
def test_relative_feasibility_is_not_diluted_by_a_large_multiplier(multiplier):
    report = _equality_residual_components()(
        jnp.asarray([1.5]), jnp.asarray([multiplier]),
        (jnp.asarray([-1.0]), jnp.zeros(1)),
    )
    # The distance to the feasible singleton {1} is 0.5 for every multiplier.
    assert float(report["r_dual"]) == pytest.approx(1.0 / 3.0, rel=2e-6)


@pytest.mark.parametrize("large_offset", [1.0, 2.0 ** 20])
def test_relative_feasibility_checks_each_block_independently(large_offset):
    report = _equality_residual_components(slices=((0, 1), (1, 2)))(
        jnp.asarray([large_offset, 1.5]), jnp.ones(2),
        (jnp.asarray([-large_offset, -1.0]), jnp.zeros(2)),
    )
    # Block one is exactly feasible. Its units must not conceal block two's
    # fixed 0.5 distance from its own equality constraint.
    assert float(report["r_dual"]) == pytest.approx(1.0 / 3.0, rel=2e-6)


@pytest.mark.parametrize("target", [2.0 ** -10, 1.0, 2.0 ** 10])
def test_relative_zero_cost_least_squares_tracks_distance_to_its_minimizer(target):
    evaluate = make_relative_components(
        lambda x: x, lambda y: y,
        None, [ocs.SumSquares()], ((0, 1),),
    )
    data = (jnp.asarray([-target]), jnp.zeros(1))
    residuals = []
    for fraction in (0.0, 0.5, 0.875, 1.0):
        point = jnp.asarray([fraction * target])
        exact_dual = point - target
        report = evaluate(point, exact_dual, data)
        # Here the normal-equation residual divided by |K^T b| is exactly
        # the relative distance to x*=target, despite c being identically zero.
        residuals.append(float(report["r_primal"]))
        assert float(report["r_primal"]) == pytest.approx(1.0 - fraction, abs=2e-7)
        assert float(report["r_dual"]) == pytest.approx(0.0, abs=2e-7)
    assert residuals == sorted(residuals, reverse=True)


@pytest.mark.parametrize(
    "numerator, denominator, expected",
    [(0.0, 0.0, 0.0), (1.0, 0.0, np.inf), (1e-12, 2e-12, 0.5),
     (np.nan, 1.0, np.inf), (1.0, np.nan, np.inf), (np.inf, 1.0, np.inf)],
)
def test_relative_ratio_zero_data_and_nonfinite_contract(numerator, denominator, expected):
    actual = float(relative_ratio(jnp.asarray(numerator), jnp.asarray(denominator)))
    if np.isinf(expected):
        assert actual == np.inf
    else:
        assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_smooth_dual_recovery_does_not_hide_nonfinite_warm_start(bad_value):
    compiled = _least_squares_family().compile(equilibrate="scalar", max_blocks=2)
    try:
        solution = compiled.solve_instance(
            {"b": jnp.asarray([1.0, 2.0, 4.0])},
            warm_start=(jnp.ones(3), jnp.full(3, bad_value)),
        )
    except ValueError:
        return
    assert solution.status != "solved"


@pytest.mark.parametrize("engine", [make_online_solver, make_osgm_solver])
def test_online_tail_does_not_move_a_point_after_it_passes_stopping(engine):
    # The forced warmup ends at x=1; the next update would move off that point.
    residual = lambda x, _y, _data, _tau, _sigma: jnp.linalg.norm(x - 1.0)
    solver = engine(
        lambda x: x, lambda y: y, jnp.ones(1), jnp.ones(1),
        lambda x, _tau, _data: x + 2.0,
        lambda y, _sigma, _data: jnp.zeros_like(y),
        residual, 1, 1, block=1, max_blocks=4, tol=1e-3,
        adaptive_omega=False, warmup_blocks=1,
    )
    x, _y, blocks, reported = solver(
        (jnp.zeros((1, 1)), jnp.zeros((1, 1))),
        jnp.zeros((1, 1)), jnp.zeros((1, 1)),
    )
    assert float(reported[0]) == pytest.approx(float(jnp.linalg.norm(x[0] - 1.0)))
    assert float(reported[0]) <= 1e-3 or int(blocks[0]) == 4


@pytest.mark.parametrize("online_method", ["jacobi", "osgm"])
@pytest.mark.parametrize("normalization", ["iterate", "data"])
def test_online_solve_reports_the_same_family_metric_as_its_certificate(online_method, normalization):
    compiled = _least_squares_family().compile(
        equilibrate="scalar", block=3, max_blocks=4, tol=1e-10,
        online_warmup=1, online_method=online_method, residual_norm=normalization,
    )
    data = {"b": jnp.asarray([2.0, -1.0, 0.5])}
    solution = compiled.solve_instance(data)
    certificate = compiled.certificate(solution.x, solution.y, data)
    assert solution.residual == pytest.approx(
        max(certificate["r_primal"], certificate["r_dual"]), rel=2e-6, abs=1e-8,
    )


@pytest.mark.parametrize("warm_value", [0.0, 1.0])
def test_cg_zero_data_only_accepts_a_stationary_returned_point(warm_value):
    compiled = _least_squares_family().compile(method="cg", cg_iters=0)
    solution = compiled.solve_instance(
        {"b": jnp.zeros(3)}, warm_start=(jnp.full(3, warm_value), None),
    )
    if warm_value == 0.0:
        assert solution.status == "solved"
        assert solution.residual == 0.0
    else:
        # CG and PDHG share normalization by the larger current/reference
        # gradient. A nonzero current gradient with zero data therefore gives 1.
        assert solution.status == "inaccurate"
        assert solution.residual == pytest.approx(1.0)


def test_cg_tiny_finite_rhs_is_not_mistaken_for_a_zero_residual():
    x = ocs.Variable(1)
    b = ocs.Parameter("b")
    family = ocs.minimize([ocs.least_squares(_diagonal_operator((1.0,)) @ x - b)])
    compiled = family.compile(method="cg", cg_iters=0, tol=1e-3)
    solution = compiled.solve_instance({"b": jnp.asarray([1e-25])})
    # With x=0 and K=I, the relative normal-equation residual is exactly one,
    # independent of the nonzero rhs magnitude. No iteration was authorized.
    assert solution.status != "solved"


def test_tolerant_indicator_values_do_not_claim_an_exact_gap():
    x = ocs.Variable(1)
    family = ocs.minimize([
        ocs.nonneg(x), ocs.least_squares(_diagonal_operator((1.0,)) @ x)])
    compiled = family.compile(equilibrate="scalar", max_blocks=0)
    # NonNeg.value admits a 1e-6 tolerance; the point is actually outside it.
    report = compiled.certificate(jnp.asarray([-5e-7]), jnp.zeros(1))
    assert report["gap_domains_finite"]
    assert not report["gap_exact"]


def test_relative_smooth_stationarity_survives_tiny_metric_cancellation():
    evaluate = make_relative_components(
        lambda x: x, lambda y: y,
        None, [ocs.SumSquares()], ((0, 1),))
    result = evaluate(jnp.ones(1), jnp.ones(1), (jnp.zeros(1), jnp.zeros(1)))
    assert float(result["r_primal"]) == pytest.approx(1.)
    x = ocs.Variable(1)
    family = ocs.minimize([ocs.least_squares(_diagonal_operator((1.0,)) @ x)])
    compiled = family.compile(
        max_blocks=0, residual_norm="relative",
        equilibrate=("given", jnp.asarray([1e-12]), jnp.ones(1)),
    )
    solution = compiled.solve_instance(warm_start=(jnp.ones(1), jnp.ones(1)))
    assert solution.status == "inaccurate"
    assert solution.residual == pytest.approx(1.)


def test_scalar_primal_psd_metric_produces_a_valid_dual_pair():
    x = ocs.Variable(9)
    diagonal = ocs.LinearOperator.from_forward(
        lambda z: jnp.diag(z.reshape(3, 3)), n=9, m=3)
    cost = (jnp.ones((3, 3)) - jnp.eye(3)).reshape(-1)
    family = ocs.minimize([ocs.psd(x, 3),
                           ocs.equality(diagonal @ x - jnp.ones(3))], cost=cost)
    metric = ("given", jnp.where(jnp.eye(3).reshape(-1) > 0, .1, 5.),
              jnp.full(3, .1), .1)
    compiled = family.compile(equilibrate=metric, block=50, max_blocks=100,
                              tol=1e-4, residual_norm="relative")
    solution = compiled.solve_instance()
    certificate = compiled.certificate(solution.x, solution.y)
    assert compiled.meta["primal_scalar_metric"]
    assert solution.status == "solved"
    assert float(cost @ solution.x) == pytest.approx(-3., abs=1e-3)
    assert certificate["r_complementarity"] <= 1e-4
    assert certificate["r_polar"] <= 1e-4
    with pytest.raises(ValueError, match="scalar primal metric"):
        family.compile(equilibrate=metric, online_warmup=1)
