"""The default result returns the witness that actually supplied its bound.

All operators here are supplied through actions; no test requires the compiler
to form an operator matrix. Recovery-OFF validation is essential: another
recovery search could hide a wrong returned dual.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.model.compile import CompiledFamily, Solution


def _problem(name, layout="dense"):
    x = ocs.Variable(3)
    k = ocs.Diagonal(jnp.array([1., 2., 3.], jnp.float32))
    data = {"b": jnp.array([-1., 2., 3.])}
    if name == "primal_nonneg":
        family = ocs.minimize([ocs.nonneg(x),
                               ocs.least_squares(k @ x - ocs.Parameter("b"))])
    elif name == "lasso":
        family = ocs.minimize([ocs.l1(x, .1),
                               ocs.least_squares(k @ x - ocs.Parameter("b"))])
    elif name == "lp":
        family = ocs.minimize([ocs.nonneg(x),
                               ocs.equality(k @ x - ocs.Parameter("b"))],
                              cost=jnp.ones(3))
        data = {"b": jnp.array([1., 2., 3.])}
    else:
        family = ocs.minimize([
            Term(ocs.Identity(3), ocs.SumSquares()),
            Term(ocs.Diagonal(jnp.ones(3)), ocs.SecondOrderCone()),
        ], cost=jnp.array([0., -2., 0.]))
        data = {}
    return family.compile(equilibrate="scalar", block=10, max_blocks=80,
                          tol=1e-3, data_layout=layout), data


def _check_return(comp, sol, data, expected_status="solved"):
    assert sol.status == expected_status
    assert sol.dual_source == "certifying"
    assert sol.y_iterate is not None
    report = comp.validate(sol.x, sol.y, data)
    assert report["recovery_enabled"] is False
    assert report["source"] == "supplied_witness"
    assert report["objective_lower_bound_available"]
    assert report["objective_lower_bound"] == sol.objective_lower_bound
    assert comp.acceptance(report)["target_met"] == (expected_status == "solved")
    checked = comp.certificate(sol.x, sol.y, data)
    assert sol.residual == max(checked["r_primal"], checked["r_dual"])
    assert sol.strict_certificate == (sol.residual <= comp._tol)


@pytest.mark.parametrize("name", ["primal_nonneg", "lasso", "lp", "soc"])
@pytest.mark.parametrize("layout", ["dense", "compact"])
@pytest.mark.parametrize("route", ["single", "batch", "single_batched_drive", "no_drive"])
def test_actual_return_rechecks_without_recovery(name, layout, route):
    comp, data = _problem(name, layout)
    if route == "single_batched_drive":
        comp._single_drive = None
    if route == "no_drive":
        comp._single_drive = comp._drive = None
    if route == "batch":
        batch = ({key: jnp.stack([val, val]) for key, val in data.items()}
                 if data else {"unused": jnp.zeros((2, 1))})
        rows = comp.solve_many_results(batch)
    else:
        rows = [comp.solve_instance(data)]
    for sol in rows:
        # The pre-existing no-drive route scores a residual-stopped point;
        # on these two fixtures that point still misses objective accuracy.
        expected = ("inaccurate" if route == "no_drive"
                    and name in ("primal_nonneg", "lasso") else "solved")
        _check_return(comp, sol, data, expected)


def test_return_dual_retains_unmodified_supplied_witness_and_recovery_off():
    comp, data = _problem("primal_nonneg")
    x, y = jnp.array([0., 1., 1.]), jnp.array([1., 0., 0.])
    for recover in (False, True):
        report = comp._objective_bound(x, y, comp._resolve_dense(data),
                                        32, return_dual=True, recover=recover)
        assert report["source"] == "supplied_witness"
        np.testing.assert_array_equal(report["certifying_dual"], y)
        assert report["objective_lower_bound"] == .5


@pytest.mark.parametrize("homogeneous", [False, True])
def test_feasible_anchor_returns_its_selected_witness_including_ray_scaling(homogeneous):
    x = ocs.Variable(1)
    problem = ocs.minimize([
        ocs.nonneg(x),
        ocs.least_squares(ocs.Diagonal(jnp.ones(1)) @ x - ocs.Parameter("b")),
    ], cost=jnp.array([0. if homogeneous else -1.]))
    comp = problem.compile(equilibrate="scalar", max_blocks=0)
    data = {"b": jnp.array([-1. if homogeneous else 1.])}
    point = jnp.zeros(1)
    supplied = jnp.array([-1. if homogeneous else 0.])
    # With zero correction rounds, neither supplied ray is dual-feasible.
    # The homogeneous case also improves the anchor by scaling its ray.
    report = comp.objective_bound(point, supplied, data, rounds=0, return_dual=True)
    assert report["source"] == "feasible_anchor"
    witness = report["certifying_dual"]
    assert not np.array_equal(np.asarray(witness), np.asarray(supplied))
    checked = comp.validate(point, witness, data)
    assert checked["source"] == "supplied_witness"
    assert checked["objective_lower_bound_available"]
    assert checked["objective_lower_bound"] == report["objective_lower_bound"]
    if homogeneous:
        theta = report["anchor_theta"]
        unscaled = (1. - theta) * float(supplied[0]) + theta
        # The finite expansion cap need not reach the ray optimum. This
        # asserts that the returned candidate includes the chosen scaling.
        assert float(witness[0]) > 2. * unscaled > 0.
        assert report["objective_lower_bound"] > unscaled - .5 * unscaled**2


def _scripted():
    """Distinct iterate/witness exposes wrong restart and lane-freeze behavior."""
    def resolve(raw):
        return raw["kind"]

    def drive(data, x, y, cap):
        # y is accumulated, so accidentally restarting from y+100 changes
        # every later iterate and the final recorded residual.
        xx, yy = x + cap, y + cap + 1
        return xx, yy, jnp.full(x.shape[0], cap), jnp.full(x.shape[0], .01)

    def bound(x, y, kind, rounds, return_dual=False):
        gap = jnp.where(kind[0] == 1, .2 * jnp.exp2(-x[0]), .01)
        out = {
            "objective_lower_bound_available": jnp.asarray(True),
            "objective_lower_bound": 1. - gap,
            "objective_primal_value": jnp.asarray(1.),
            "objective_at_returned_point": jnp.asarray(1.),
            "relative_gap_certified": jnp.asarray(True),
            "relative_primal_infeasibility": jnp.asarray(0.),
        }
        if return_dual:
            out["certifying_dual"] = y + 100
        return out

    def many(x, y, data, rounds, return_dual=False):
        out = jax.vmap(bound, in_axes=(0, 0, 0, None))(x, y, data, rounds)
        return (out, y + 100) if return_dual else out

    def components(x, y, data):
        return {"r_primal": jnp.abs(y[0]), "r_dual": jnp.asarray(0.)}

    def unused(*args):
        raise AssertionError("unexpected residual solve")

    return CompiledFamily(unused, resolve, None, 1, 1, {"block": 1},
                          tol=1e-3, max_blocks=31, max_iters=31,
                          bound_capable=True, objective_bound=bound,
                          objective_bound_many=many, drive_solver=drive,
                          returned_components=components,
                          contract=lambda x, y, d: {"contract": jnp.abs(y[0])})


def test_witness_never_restarts_iteration_and_each_terminal_lane_is_frozen():
    comp = _scripted()
    data = {"kind": jnp.array([[0.], [1.]])}
    rows = comp.solve_many_results(data)
    assert [row.termination for row in rows] == ["stalled", "criterion"]
    assert [row.status for row in rows] == ["inaccurate", "solved"]
    assert [row.iters for row in rows] == [7, 15]
    for i, row in enumerate(rows):
        single = comp.solve_instance({"kind": data["kind"][i]})
        for field in ("x", "y", "y_iterate"):
            np.testing.assert_array_equal(getattr(row, field), getattr(single, field))
        assert row.gap_history == single.gap_history
        assert float(row.y_iterate[0]) == row.iters + len(row.gap_history)
        assert float(row.y[0]) == float(row.y_iterate[0]) + 100
        assert row.dual_source == "certifying"
        assert row.residual == float(row.y[0])
        assert row.contract["contract"] == float(row.y[0])
        assert row.contract_certificate is False
        assert all(h["legacy_residual"] == float(np.float32(.01))
                   for h in row.gap_history)


@pytest.mark.parametrize("status", ["primal_infeasible", "unbounded", "numerical_error"])
def test_certificate_status_never_replaces_its_dual_or_facts(status):
    comp = _scripted()
    y = jnp.array([4.])
    s = Solution(jnp.array([2.]), y, 7, .1, status=status,
                 certificate=jnp.array([3.]))
    out = comp._return_certifying_pair(
        s, {"objective_lower_bound_available": True,
            "certifying_dual": jnp.array([99.])}, jnp.array([0.]))
    assert out is s
    assert out.y is y
    assert out.dual_source == "iterate"


@pytest.mark.parametrize("kind", ["primal_infeasible", "unbounded"])
@pytest.mark.parametrize("batched", [False, True])
def test_actual_certificate_directions_survive_default_return(kind, batched):
    x = ocs.Variable(1)
    if kind == "primal_infeasible":
        op = ocs.LinearOperator.from_forward(lambda v: v, 1, 1)
        problem = ocs.minimize([ocs.nonneg(x),
                               ocs.equality(op @ x - ocs.Parameter("b"))])
        data = {"b": jnp.array([-1.])}
    else:
        op = ocs.LinearOperator.from_forward(lambda v: 0. * v, 1, 1)
        problem = ocs.minimize([ocs.nonneg(x), Term(op, ocs.Zero())],
                               cost=-jnp.ones(1))
        data = {}
    comp = problem.compile(equilibrate="scalar", block=10,
                           max_blocks=0 if kind == "unbounded" else 20)
    # Supply a nonzero valid recession direction for the zero-operator LP;
    # the test exercises certificate preservation, not discovery of a ray.
    start = (jnp.ones(1), jnp.zeros(1)) if kind == "unbounded" else None
    if batched:
        batch = ({key: jnp.stack([val, val]) for key, val in data.items()}
                 if data else {"unused": jnp.zeros((2, 1))})
        starts = (tuple(jnp.stack([v, v]) for v in start) if start is not None else None)
        rows = comp.solve_many_results(batch, warm_start=starts)
    else:
        rows = [comp.solve_instance(data, warm_start=start)]
    for sol in rows:
        assert sol.status == kind
        assert sol.certificate is not None
        assert sol.dual_source == "iterate"
        np.testing.assert_array_equal(sol.y, sol.y_iterate)
        status, direction = comp._status(sol.x, sol.y, comp._resolve(data),
                                          sol.iters // 10, sol.residual)
        assert status == kind
        np.testing.assert_array_equal(sol.certificate, direction)


@pytest.mark.parametrize("status", ["solved", "inaccurate", "unverified"])
def test_available_final_witness_is_returned_on_every_accuracy_status(status):
    comp = _scripted()
    s = Solution(jnp.array([2.]), jnp.array([4.]), 7, .1, status=status,
                 termination="iteration_limit")
    out = comp._return_certifying_pair(
        s, {"objective_lower_bound_available": True,
            "certifying_dual": jnp.array([99.])}, jnp.array([0.]))
    assert out.status == status and out.termination == "iteration_limit"
    assert out.dual_source == "certifying" and out.residual == 99.
    np.testing.assert_array_equal(out.y_iterate, s.y)


def test_unavailable_final_check_retains_iterate():
    comp = _scripted()
    s = Solution(jnp.array([2.]), jnp.array([4.]), 7, .1, status="unverified")
    assert comp._return_certifying_pair(
        s, {"objective_lower_bound_available": False,
            "certifying_dual": jnp.array([jnp.nan])}, jnp.array([0.])) is s


def test_batch_without_drive_preserves_unverified_fallback():
    comp, data = _problem("primal_nonneg")
    comp._drive = None
    results = comp.solve_many_results({"b": jnp.stack([data["b"], data["b"]])})
    assert all(r.status == "unverified" and r.dual_source == "iterate" for r in results)
    for r in results:
        np.testing.assert_array_equal(r.y, r.y_iterate)


def test_residual_opt_in_keeps_iteration_dual():
    comp, data = _problem("primal_nonneg")
    for r in [comp.solve_instance(data, accuracy="residual"),
              *comp.solve_many_results({"b": jnp.stack([data["b"], data["b"]])},
                                       accuracy="residual")]:
        assert r.accuracy_criterion == "fixed_point"
        assert r.dual_source == "iterate"
        np.testing.assert_array_equal(r.y, r.y_iterate)


def test_cg_preserves_explicit_dual_free_strong_convexity_exception():
    x = ocs.Variable(2)
    k = ocs.Diagonal(jnp.array([1., 2.]))
    comp = ocs.minimize([ocs.least_squares(k @ x - ocs.Parameter("b"))]).compile(
        method="cg", rho=1., cg_iters=40, tol=1e-3)
    data = {"b": jnp.array([1., 2.])}
    for r in [comp.solve_instance(data),
              *comp.solve_many_results({"b": jnp.stack([data["b"], data["b"]])})]:
        assert r.status == "solved"
        assert r.y is None and r.y_iterate is None
        assert r.dual_source == "none"
        assert r.objective_bound_source == "strong_convexity"
        assert comp.objective_bound(r.x, None, data)["source"] == "strong_convexity"


def test_coupled_operator_witness_rechecks_with_same_acceptance():
    # Cached a*(K.T@y) and a fresh K.T@(a*y) may round differently.
    # Compare bounds numerically, while keeping exact domain gates and the
    # original acceptance verdict. No tolerance is added to either gate.
    rng = np.random.default_rng(0)
    n = 8
    weights = jnp.asarray(rng.uniform(.2, .6, n), jnp.float32)
    op = ocs.LinearOperator.from_forward(
        lambda v: weights * v + .17 * jnp.roll(v, 1) - .09 * jnp.roll(v, -2),
        n, n)
    x = ocs.Variable(n)
    data = {"b": jnp.asarray(rng.normal(size=n), jnp.float32)}
    comp = ocs.minimize([
        Term(ocs.Identity(n), ocs.SumSquares().scaled(.1)),
        ocs.least_squares(op @ x - ocs.Parameter("b")),
    ]).compile(equilibrate="scalar", block=20, max_blocks=200, tol=1e-3)
    sol = comp.solve_instance(data)
    report = comp.validate(sol.x, sol.y, data)
    assert sol.status == "solved"
    assert report["source"] == "supplied_witness"
    assert report["objective_lower_bound_available"]
    assert comp.acceptance(report)["target_met"]
    np.testing.assert_array_max_ulp(np.float32(report["objective_lower_bound"]),
                                    np.float32(sol.objective_lower_bound), maxulp=8)
