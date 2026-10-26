"""Independent objective and status checks for the small CVXPY harness."""
import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
if tuple(int(part) for part in cp.__version__.split(".")[:2]) < (1, 6):
    pytest.skip("validation contract requires CVXPY >= 1.6", allow_module_level=True)
from ocs.cvxpy_harness import validate_against_cvxpy
from ocs.cvxpy_harness.backend import _cvxpy_status

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the CVXPY validation harness against reference solvers.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def _validate(problem, **kwargs):
    return validate_against_cvxpy(
        problem, solver=cp.SCS, equilibrate="scalar", tol=1e-5,
        max_blocks=400, reference_options={"eps_abs": 1e-8, "eps_rel": 1e-8},
        **kwargs,
    )


@pytest.mark.parametrize("maximize", [False, True])
def test_linear_objective_preserves_offset_and_direction(maximize):
    x = cp.Variable()
    objective = cp.Maximize(2*x + 7) if maximize else cp.Minimize(2*x + 7)
    bounds = [x <= 3, x >= -5] if maximize else [x >= 3]
    report = _validate(cp.Problem(objective, bounds))
    assert report["ocs_status"] == "solved"
    assert report["reference_status"] == "optimal"
    assert report["objective_agreement"] and report["status_agreement"]
    assert report["ocs_obj"] == pytest.approx(13.0, abs=3e-4)
    assert report["ref_obj"] == pytest.approx(13.0, abs=1e-6)
    assert report["constraint_violation"] < 2e-4


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("lower_bound", [0.0, 2.0])
def test_dense_quadratic_P_is_solved_and_scored_with_constant(maximize, lower_bound):
    x = cp.Variable(2)
    P = np.asarray([[4.0, 1.0], [1.0, 2.0]])
    q = np.asarray([-5.0, -3.0])
    expression = 0.5 * cp.quad_form(x, P) + q @ x + 11
    objective = cp.Maximize(-expression) if maximize else cp.Minimize(expression)
    problem = cp.Problem(objective, [x >= lower_bound])
    canonical, _, _ = problem.get_problem_data(cp.SCS)
    assert "P" in canonical and canonical["P"].nnz > 0
    report = _validate(problem)
    # The inactive case has x=(1,1), value 7 and zero normal; the active
    # case has x=(2,2), value 11. Ignoring P makes either LP unbounded.
    assert report["ocs_status"] == "solved"
    point, value = ([1, 1], 7) if lower_bound == 0 else ([2, 2], 11)
    np.testing.assert_allclose(report["canonical_x"], point, atol=1e-3)
    assert report["ocs_obj"] == pytest.approx(-value if maximize else value, abs=1e-3)
    assert report["objective_agreement"]


def test_soc_reference_compares_the_original_objective():
    x = cp.Variable(2)
    report = _validate(cp.Problem(cp.Minimize(np.asarray([3., 4.]) @ x + 3),
                                  [cp.norm(x) <= 1]))
    assert report["ocs_status"] == "solved"
    assert report["ocs_obj"] == pytest.approx(-2.0, abs=2e-4)
    assert report["objective_agreement"]
    assert report["constraint_violation"] < 2e-4


@pytest.mark.parametrize("unbounded", [False, True])
def test_failure_statuses_cannot_count_as_objective_agreement(unbounded):
    x = cp.Variable()
    problem = (cp.Problem(cp.Minimize(-x), [x >= 0]) if unbounded else
               cp.Problem(cp.Minimize(0 * x), [x == -1, x >= 0]))
    report = _validate(problem)
    expected = "unbounded" if unbounded else "primal_infeasible"
    assert report["ocs_status"] == expected
    assert report["reference_status"] == ("unbounded" if unbounded else "infeasible")
    assert report["status_agreement"]
    assert not report["objective_comparable"]
    assert not report["objective_agreement"]
    assert np.isnan(report["rel_err"])


def test_budget_exhaustion_is_reported_even_with_a_finite_objective():
    x = cp.Variable()
    report = validate_against_cvxpy(cp.Problem(cp.Minimize(x + 4), [x >= 3]),
                                    max_blocks=0, equilibrate="scalar")
    assert report["ocs_status"] == "inaccurate"
    assert np.isfinite(report["candidate_obj"])
    assert not report["objective_comparable"]
    assert not report["objective_agreement"]
    assert np.isnan(report["rel_err"])


def test_reference_failure_is_visible(monkeypatch):
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 1])
    def fail(**kwargs):
        raise cp.error.SolverError("reference unavailable")
    monkeypatch.setattr(problem, "solve", fail)
    report = _validate(problem)
    assert report["reference_status"] == "solver_error"
    assert "reference unavailable" in report["reference_error"]
    assert not report["objective_agreement"]
    assert np.isnan(report["ref_obj"])


def test_incompatible_canonicalization_dependencies_have_an_actionable_error(monkeypatch):
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 1])
    def fail(*args, **kwargs):
        raise AttributeError("sparse object has no attribute A")
    monkeypatch.setattr(problem, "get_problem_data", fail)
    with pytest.raises(RuntimeError, match="CVXPY.*SciPy.*compatible"):
        validate_against_cvxpy(problem)


def test_power_cone_is_explicitly_outside_small_harness_scope():
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(x)), [cp.PowCone3D(x[0], x[1], x[2], .3)])
    with pytest.raises(NotImplementedError, match="p3d"):
        validate_against_cvxpy(problem)


@pytest.mark.parametrize("source,expected", [
    ("solved", "optimal"), ("inaccurate", "user_limit"),
    ("primal_infeasible", "infeasible"), ("unbounded", "unbounded"),
    ("numerical_error", "solver_error"),
])
def test_shared_cvxpy_status_mapping_does_not_claim_failed_points_optimal(source, expected):
    assert _cvxpy_status(source) == expected
