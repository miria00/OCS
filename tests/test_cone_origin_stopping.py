"""Public stopping regressions at cone origins and zero normal multipliers.

These are supplied-point tests, not convergence or projector benchmarks.
The independent projection references use Clarabel at 1e-11 tolerances;
the alpha=.5 power reference used the equivalent SOC inequality while
retaining the original Euclidean objective.
"""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

import ocs


TOL = 1e-3
UNIT_SCALES = (1e-6, 1e-4, 1e-2, 1., 1e2, 1e4, 1e6)
# Vary data magnitude, constraint units, and objective units independently.
SCALE_CASES = [(1., 1., 1.)]
for _axis in range(3):
    for _value in UNIT_SCALES:
        if _value != 1.:
            _case = [1., 1., 1.]
            _case[_axis] = _value
            SCALE_CASES.append(tuple(_case))
SCALE_CASES += [(1e-6, 1e-6, 1e6), (1e6, 1e6, 1e-6)]
REJECTION_SCALES = [
    (1., 1., 1.), (1e-6, 1., 1.), (1e6, 1., 1.),
    (1., 1e-6, 1.), (1., 1e6, 1.),
    (1., 1., 1e-6), (1., 1., 1e6),
]


@lru_cache(maxsize=None)
def _family(kind, argument_units=1., objective_units=1.):
    """min gamma*(.5||x||²-v'x) subject to alpha*x in C."""
    a, gamma = argument_units, objective_units
    variable = ocs.Variable(3)
    quadratic = ocs.sum_squares(variable)
    # Exercise the public calculus wrapper on the identity-routed primal g.
    quadratic.block = ocs.SumSquares().scaled(gamma)
    operator = ocs.Diagonal(jnp.full(3, a))
    cone = (ocs.exp_cone(operator @ variable) if kind == "exp" else
            ocs.pow_cone(operator @ variable, alpha=.5))
    problem = ocs.minimize([quadratic, cone], cost=ocs.Parameter("c"))
    # No iteration is performed. These positive metrics remain admissible
    # under every tested unit transformation and do not define acceptance.
    # Exp and power cone blocks declare no exact conjugate-domain contract, so
    # no objective lower bound exists for this family and the default
    # criterion would report "unverified".  This file tests the cone KKT
    # acceptance itself, which is a statement about the residual criterion, so
    # it asks for that one explicitly.
    return problem.compile(
        max_blocks=0, tol=TOL, accuracy="residual",
        equilibrate=("given", jnp.full(3, 1e-3/gamma),
                     jnp.full(3, 1e-3*gamma/a**2)),
    )


def _point_report(kind, v, x, y, scales=(1., 1., 1.)):
    eta, a, gamma = scales
    compiled = _family(kind, a, gamma)
    data = {"c": jnp.asarray(-gamma*eta*np.asarray(v), dtype=jnp.float32)}
    point = jnp.asarray(eta*np.asarray(x), dtype=jnp.float32)
    dual = jnp.asarray((gamma*eta/a)*np.asarray(y), dtype=jnp.float32)
    return compiled, data, point, dual, compiled.certificate(point, dual, data)


def _residual(report):
    return max(report["r_primal"], report["r_dual"])


def _near_point(kind, error):
    if kind == "power":
        # v is strictly in the polar cone, so its exact projection is zero.
        v = np.asarray([-1., -1., 0.])
        return v, error*v, v.copy()
    # Battery seed100 is strictly interior; its exact normal multiplier is 0.
    v = np.random.default_rng(100).normal(size=3).astype("f").astype("d")
    return v, (1.+error)*v, error*np.linalg.norm(v)*np.asarray([0., 1., 0.])


@pytest.mark.parametrize("kind", ["power", "exp"])
@pytest.mark.parametrize("scales", SCALE_CASES)
def test_accurate_cone_origin_or_zero_normal_accepts_across_units(kind, scales):
    v, x, y = _near_point(kind, 1e-5)
    compiled, data, point, dual, report = _point_report(kind, v, x, y, scales)
    assert np.isfinite(_residual(report))
    assert _residual(report) < TOL

    solution = compiled.solve_instance(data, warm_start=(point, dual))
    assert solution.iters == 0
    assert solution.status == "solved"
    np.testing.assert_array_equal(np.asarray(solution.x), np.asarray(point))
    returned = compiled.certificate(solution.x, solution.y, data)
    assert solution.residual == pytest.approx(_residual(returned), rel=5e-5, abs=1e-7)

    # The same mapped candidate should have the same numerical acceptance
    # scale. Allow float32 oracle roundoff without accepting order-one drift.
    baseline = _point_report(kind, v, x, y)[-1]
    assert _residual(report) == pytest.approx(_residual(baseline), rel=.03, abs=3e-7)


@pytest.mark.parametrize("kind", ["power", "exp"])
@pytest.mark.parametrize("scales", REJECTION_SCALES)
def test_material_cone_or_stationarity_error_is_not_erased_by_anchor(kind, scales):
    v, x, y = _near_point(kind, .05)
    compiled, data, point, dual, report = _point_report(kind, v, x, y, scales)
    assert _residual(report) > TOL
    solution = compiled.solve_instance(data, warm_start=(point, dual))
    assert solution.status != "solved"
    assert solution.iters == 0
    if kind == "power":
        # This candidate violates x>=0 by 5% of the nonzero data scale.
        assert report["r_domain"] > .01
    else:
        # The cone remains strictly feasible, but x differs from the
        # unconstrained minimizer and the supplied multiplier is nonpolar.
        assert report["r_primal"] > .01 or report["r_polar"] > .01


def _battery_projection(kind, seed, v):
    if kind == "exp":
        if seed == 100:
            return v.copy()
        if seed == 101:
            return np.asarray([v[0], 0., v[2]])
        return np.asarray([.011251020507478454, 1.5542951167149301,
                           1.5655869568361287])
    if seed == 100:
        return np.asarray([.03076698707341977, .3847748580844707,
                           .1088042420350125])
    if seed == 101:
        return np.zeros(3)
    return v.copy()


@pytest.mark.parametrize("kind", ["power", "exp"])
@pytest.mark.parametrize("seed", [100, 101, 102])
def test_three_unique_battery_projection_kkt_pairs_accept(kind, seed):
    v = np.random.default_rng(seed).normal(size=3).astype("f").astype("d")
    point = _battery_projection(kind, seed, v)
    # Projection optimality gives v-point in the normal cone at point.
    compiled, data, x, y, report = _point_report(kind, v, point, v-point)
    assert _residual(report) < TOL
    solution = compiled.solve_instance(data, warm_start=(x, y))
    assert solution.status == "solved"
    assert solution.iters == 0


@pytest.mark.parametrize("kind", ["power", "exp"])
def test_zero_data_has_no_invented_absolute_acceptance_scale(kind):
    zero = np.zeros(3)
    compiled, data, x, y, report = _point_report(kind, zero, zero, zero)
    assert _residual(report) == 0.
    assert compiled.solve_instance(data, warm_start=(x, y)).status == "solved"

    # With v=0 the nonzero candidate is not relatively accurate merely
    # because its absolute magnitude is tiny.
    wrong = 1e-10*np.asarray([-1., -1., 0.] if kind == "power" else [-1., 1., 1.])
    compiled, data, x, y, report = _point_report(kind, zero, wrong, zero)
    assert _residual(report) > TOL
    assert compiled.solve_instance(data, warm_start=(x, y)).status != "solved"


@pytest.mark.parametrize("kind", ["power", "exp"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_tiny_candidate_does_not_allow_invalid_raw_warm_dual_recovery(kind, bad):
    v, x, _ = _near_point(kind, 1e-5)
    compiled, data, point, _, _ = _point_report(
        kind, v, x, np.zeros(3), (1e-6, 1., 1.))
    dual = jnp.asarray([bad, 1e-30, 0.], dtype=jnp.float32)
    report = compiled.certificate(point, dual, data)
    assert not np.isfinite(_residual(report))
    solution = compiled.solve_instance(data, warm_start=(point, dual))
    assert solution.status == "numerical_error"
    assert not np.all(np.isfinite(np.asarray(solution.y)))
