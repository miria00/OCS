"""Opt-in scaled certificate contract (declared absolute budgets).

Three things are tested here and kept apart on purpose:

1. the default relative certificate is UNCHANGED and reported beside the
   contract, and with no contract declared nothing about the family changes;
2. the three separately reportable facts -- strict certificate, declared-budget
   certificate, termination reason -- and the documented CVXPY mapping built on
   them, including the rule that a budget-exhausted run is never relabelled;
3. the CORRECTED complementarity allowance, with the verified counterexample
   that refutes the expression it replaces pinned as a regression.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.prox.library import L1, NonNeg
from ocs.cvxpy_harness.backend import (_cvxpy_status, OCS_TO_CVXPY_STATUS,
                                       CONTRACT_STATUS_RULE)
from ocs.solver.contract import (ScaledContract, _budget_graph, _block_contract,
                                 budget_ratio, complementarity_allowance,
                                 STOPPING_MODES, EXPERIMENTAL_STOPPING)

TOL = 1e-6


def budgets(value):
    return {"primal_budget": value, "dual_budget": value,
            "variable_budget": value, "stationarity_budget": value}


def homogeneous(contract=None, **kw):
    x = ocs.Variable(2)
    return ocs.minimize([ocs.l1(x), ocs.equality(x)]).compile(
        residual_norm="relative", tol=TOL, certificate_contract=contract, **kw)


def report(compiled, x, y):
    return compiled.certificate(jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32))


# ---------------------------------------------------------------- defaults

def test_default_has_no_contract_fields():
    compiled = homogeneous()
    out = report(compiled, np.zeros(2), np.zeros(compiled.m))
    assert not any(k.startswith("contract") for k in out)
    # Absent, not None: an undeclared family's metadata is what it was before
    # this option existed.
    assert "certificate_contract" not in compiled.meta
    sol = compiled.solve_instance()
    assert sol.contract is None and sol.contract_certificate is None


def test_default_residual_is_unchanged_beside_contract():
    plain, scaled = homogeneous(), homogeneous(budgets(1e-6))
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)
    for k in range(0, 14, 3):
        a = report(plain, 10.0 ** -k * v, np.zeros(2))
        b = report(scaled, 10.0 ** -k * v, np.zeros(2))
        for key, value in a.items():
            np.testing.assert_allclose(b[key], value, rtol=1e-6, atol=0,
                                       equal_nan=True, err_msg=key)


def test_strict_only_is_the_default_and_keeps_the_default_iteration():
    assert ScaledContract(primal_budget=0.0, dual_budget=0.0).stopping == "strict_only"
    assert ScaledContract(primal_budget=0.0, dual_budget=0.0).stops_the_loop is False
    assert set(STOPPING_MODES) == {"strict_only", "strict_or_contract"}
    assert EXPERIMENTAL_STOPPING == ("strict_or_contract",)
    plain = homogeneous(block=50, max_blocks=6).solve_instance()
    for contract in (budgets(1e-6), dict(budgets(1e-6), stopping="strict_only")):
        kept = homogeneous(contract, block=50, max_blocks=6).solve_instance()
        assert kept.iters == plain.iters
        assert kept.status == plain.status
        np.testing.assert_allclose(kept.residual, plain.residual, rtol=0, atol=0)
        assert kept.termination == plain.termination
        assert kept.strict_certificate == plain.strict_certificate
    with pytest.raises(ValueError):
        homogeneous(dict(budgets(1e-6), stopping="contract_only"))


def test_declared_contract_never_changes_the_status_vocabulary():
    compiled = homogeneous(budgets(1e-6), block=50, max_blocks=6)
    sol = compiled.solve_instance()
    assert sol.status in ("solved", "inaccurate", "primal_infeasible",
                          "unbounded", "numerical_error")
    # (i) and (ii) are separate answers to separate questions.
    assert sol.strict_certificate == (sol.residual <= TOL)
    assert sol.contract_certificate == (sol.contract["contract"] <= 1.0)
    assert (sol.status == "solved") == sol.strict_certificate


# ------------------------------------------------- the three reported facts

def constrained(contract=None, **kw):
    """A family whose zero start is NOT already optimal."""
    x = ocs.Variable(3)
    shifted = ocs.Identity(3) @ x - np.ones(3, np.float32)
    return ocs.minimize([ocs.nonneg(shifted), ocs.l1(x)],
                        cost=np.ones(3, np.float32)).compile(
        residual_norm="relative", tol=TOL, certificate_contract=contract, **kw)


def test_three_facts_are_separately_reportable():
    converged = homogeneous(budgets(1e-6), block=200, max_blocks=400).solve_instance()
    starved = constrained(budgets(1e-9), block=1, max_blocks=1).solve_instance()
    for sol in (converged, starved):
        assert isinstance(sol.strict_certificate, bool)
        assert isinstance(sol.contract_certificate, bool)
        assert sol.termination in ("criterion", "iteration_limit",
                                   "time_limit", "numerical_failure")
    assert starved.iters == 1
    assert not starved.strict_certificate
    # A run that ran out of iterations says so, whatever any contract says,
    # and its status is not widened by the contract.
    assert starved.termination == "iteration_limit"
    assert starved.status == "inaccurate"


def test_budget_exhausted_point_keeps_user_limit_end_to_end():
    import cvxpy.settings as settings
    starved = constrained(budgets(1e9), block=1, max_blocks=1).solve_instance()
    # An absurd budget makes the contract pass at a point that ran out of
    # iterations.  It still is not a solve, and CVXPY still hears user_limit.
    assert starved.contract_certificate is True
    assert starved.status == "inaccurate"
    assert starved.termination == "iteration_limit"
    assert _cvxpy_status(starved.status,
                         contract_certificate=starved.contract_certificate,
                         termination=starved.termination) == settings.USER_LIMIT


def test_termination_is_criterion_when_the_loop_exits_early():
    sol = homogeneous(block=50, max_blocks=400).solve_instance()
    if sol.strict_certificate:
        assert sol.termination == "criterion"
        assert sol.iters < 50 * 400


def test_time_limit_is_reserved_and_never_produced():
    # This engine carries no wall-clock budget. The name exists so that a
    # reader can tell a time limit from an iteration limit if one is ever
    # added; nothing here may produce it.
    for blocks in (1, 4, 400):
        sol = homogeneous(block=25, max_blocks=blocks).solve_instance()
        assert sol.termination != "time_limit"


# --------------------------------------------------------- the CVXPY mapping

def test_cvxpy_mapping_table():
    import cvxpy.settings as settings
    assert OCS_TO_CVXPY_STATUS["solved"] == "optimal"
    assert "never relabelled a solve" in CONTRACT_STATUS_RULE
    # A strict pass is `optimal` and nothing else can be.
    assert _cvxpy_status("solved") == settings.OPTIMAL
    assert _cvxpy_status("solved", contract_certificate=False,
                         termination="iteration_limit") == settings.OPTIMAL
    # No contract declared: exactly the old mapping.
    for status in OCS_TO_CVXPY_STATUS:
        assert _cvxpy_status(status) == _cvxpy_status(
            status, contract_certificate=None, termination="criterion")
    # A strict failure is never `optimal`, whatever the contract says.
    for contract in (None, False, True):
        for termination in ("criterion", "iteration_limit", "time_limit",
                            "numerical_failure"):
            got = _cvxpy_status("inaccurate", contract_certificate=contract,
                                termination=termination)
            assert got != settings.OPTIMAL


def test_budget_exhausted_is_never_relabelled():
    import cvxpy.settings as settings
    # The whole point of condition 2: an extra contract passing does NOT turn a
    # run that ran out of iterations into a solve.
    assert _cvxpy_status("inaccurate", contract_certificate=True,
                         termination="iteration_limit") == settings.USER_LIMIT
    assert _cvxpy_status("inaccurate", contract_certificate=True,
                         termination="time_limit") == settings.USER_LIMIT
    # It is only a deliberate, declared contract met at a criterion stop that
    # earns CVXPY's lower-accuracy code, and even then never `optimal`.
    assert _cvxpy_status("inaccurate", contract_certificate=True,
                         termination="criterion") == settings.OPTIMAL_INACCURATE
    assert _cvxpy_status("inaccurate", contract_certificate=False,
                         termination="criterion") == settings.USER_LIMIT
    assert _cvxpy_status("inaccurate", contract_certificate=None,
                         termination="criterion") == settings.USER_LIMIT
    # Certificates and failures are untouched by the contract.
    for status, expected in (("primal_infeasible", settings.INFEASIBLE),
                             ("unbounded", settings.UNBOUNDED),
                             ("numerical_error", settings.SOLVER_ERROR)):
        assert _cvxpy_status(status, contract_certificate=True,
                             termination="criterion") == expected
    with pytest.raises(ValueError):
        _cvxpy_status("solved_scaled_contract")


# ------------------------------------------- corrected complementarity bound

def test_complementarity_counterexample():
    """The refuted allowance, pinned.

    Review packet, verified: one scalar block with A = 0.125, B = 1, an
    admissible witness z = 0.875, p = 0 at s = 1, y = 1 has defect 1.0, while
    the expression the paper claimed "follows from the witness bound",
    sum_i (a_i||y_i|| + b_i||s_i|| + a_i b_i), allows only 0.9475 at
    eps = 0.1 with P = Y = 1.  The bound the witness actually implies is
    A||y|| + B||s|| + AB = 1.25.
    """
    eps, P, Y = 0.1, 1.0, 1.0
    A, B = 0.125, 1.0
    a, b = A - eps * P, B - eps * Y                  # 0.025, 0.9
    s, y, z, p = 1.0, 1.0, 0.875, 0.0
    # The witness is admissible: it is exactly what the graph test permits.
    assert abs(s - z) <= A + 1e-12
    assert abs(y - p) <= B + 1e-12
    assert z * p == 0.0                              # an exact cone witness
    defect = abs(s * y - z * p)
    assert defect == 1.0
    refuted = a * abs(y) + b * abs(s) + a * b
    assert refuted == pytest.approx(0.9475, abs=1e-12)
    assert defect > refuted, "the counterexample must still be a counterexample"
    correct = float(complementarity_allowance(A, B, abs(s), abs(y)))
    assert correct == pytest.approx(1.25, abs=1e-12)
    assert defect <= correct


def test_shipped_allowance_uses_the_full_budgets():
    """The shipped evaluator must use A_i and B_i, not a_i and b_i."""
    block = NonNeg()
    s = jnp.asarray([1.0], jnp.float32)
    y = jnp.asarray([1.0], jnp.float32)
    A, B = jnp.asarray(0.125, jnp.float32), jnp.asarray(1.0, jnp.float32)
    eps, a, b = 0.1, jnp.asarray(0.025, jnp.float32), jnp.asarray(0.9, jnp.float32)
    _d, _p, comp, _sup, allowance = _block_contract(
        block, s, y, s, None, eps, a, b, A, B)
    assert float(comp) == pytest.approx(1.0, rel=1e-6)
    assert float(allowance) == pytest.approx(1.25, rel=1e-5)
    # The refuted expression would have rejected this pair.
    assert float(comp) / float(allowance) <= 1.0
    assert float(comp) > 0.025 * 1.0 + 0.9 * 1.0 + 0.025 * 0.9


def test_box_allowance_adds_the_support_lipschitz_term():
    from ocs.prox.library import Box
    block = Box(-2.0, 3.0)
    s = jnp.asarray([1.0, -1.0], jnp.float32)
    y = jnp.asarray([0.5, 0.25], jnp.float32)
    A, B = jnp.asarray(0.1, jnp.float32), jnp.asarray(0.2, jnp.float32)
    _d, _p, _c, _sup, allowance = _block_contract(
        block, s, y, s, None, 1e-3, jnp.asarray(0.0, jnp.float32),
        jnp.asarray(0.0, jnp.float32), A, B)
    bounds = math.sqrt(3.0 ** 2 + 3.0 ** 2)
    expected = (float(A) * float(np.linalg.norm(np.asarray(y)))
                + float(B) * float(np.linalg.norm(np.asarray(s)))
                + float(A) * float(B) + float(B) * bounds)
    assert float(allowance) == pytest.approx(expected, rel=1e-5)


def test_graph_pass_implies_complementarity_pass():
    """The implication the manuscript asserts, now measured end to end.

    With the refuted expression this failed: the graph test could accept a pair
    the complementarity term then rejected.
    """
    x = ocs.Variable(3)
    compiled = ocs.minimize([ocs.nonneg(x), ocs.equality(x)]).compile(
        residual_norm="relative", tol=1e-4, certificate_contract=budgets(1e-2))
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        xs = rng.normal(size=3).astype(np.float32) * rng.choice([1e-3, 1e-1, 1.0])
        ys = rng.normal(size=compiled.m).astype(np.float32) * rng.choice([1e-3, 1e-1, 1.0])
        out = report(compiled, xs, ys)
        if out["contract_graph"] <= 1.0 and np.isfinite(out["contract_complementarity"]):
            checked += 1
            assert out["contract_complementarity"] <= 1.0 + 1e-5
    assert checked > 0, "the sweep must actually exercise the implication"


# ----------------------------------------------------------- positive branch

def test_positive_branch_is_required_and_reported():
    """Clause (v) assumes A_i > 0 AND B_i > 0; the flag says when it held."""
    compiled = homogeneous(budgets(1e-6))
    out = report(compiled, np.zeros(2), np.zeros(2))
    assert out["contract_witness_active"] == 1.0
    zero = homogeneous(budgets(0.0))
    # At the exact zero point every scale vanishes, so A = B = 0: the witness
    # is not formed and the evaluator says so rather than claiming the bound.
    out0 = report(zero, np.zeros(2), np.zeros(2))
    assert out0["contract"] == 0.0
    assert out0["contract_witness_active"] == 0.0


def test_step_changes_with_budgets_and_bounds_dual_witness():
    block = L1()
    s = jnp.asarray([1e-7, -2e-7], jnp.float32)
    y = jnp.asarray([0.5, -1.0], jnp.float32)
    A, B = jnp.asarray(3e-6, jnp.float32), jnp.asarray(2e-6, jnp.float32)
    ratio, step, active = _budget_graph(block, s, y, A, B)
    assert bool(active)
    np.testing.assert_allclose(float(step), float(A / B), rtol=1e-6)
    z = np.asarray(block.prox(s + step * y, step), np.float64)
    p = np.asarray(y, np.float64) + (np.asarray(s, np.float64) - z) / float(step)
    if float(ratio) <= 1.0:
        assert np.linalg.norm(p - np.asarray(y, np.float64)) <= float(B) * (1 + 1e-5)


def test_inactive_branch_keeps_the_exact_zero_scale_rule():
    block = L1()
    s = jnp.asarray([1e-7, 0.0], jnp.float32)
    y = jnp.asarray([0.0, 0.0], jnp.float32)
    zero = jnp.asarray(0.0, jnp.float32)
    ratio, step, active = _budget_graph(block, s, y, zero, zero)
    assert not bool(active)
    assert float(step) == 1.0
    assert not np.isfinite(float(ratio))


# ---------------------------------------------------------- ratios and units

def test_near_zero_ladder_is_continuous_and_exact_pair_accepted():
    compiled = homogeneous(budgets(1e-6))
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)
    assert report(compiled, np.zeros(2), np.zeros(2))["contract"] == 0.0
    values = [report(compiled, 10.0 ** -k * v, np.zeros(2))["contract"] for k in range(14)]
    assert all(np.isfinite(values))
    assert all(b < a for a, b in zip(values, values[1:]))
    for k, value in enumerate(values):
        expected = 10.0 ** -k / (TOL * 10.0 ** -k + 1e-6)
        np.testing.assert_allclose(value, expected, rtol=1e-4)
    plain = homogeneous()
    assert all(not np.isfinite(max(report(plain, 10.0 ** -k * v, np.zeros(2))[f]
                                   for f in ("r_primal", "r_dual"))) for k in range(14))


@pytest.mark.parametrize("dual,threshold", [((5.0, 5.0), 4 * np.sqrt(2.0)), ((2.0, 0.0), 1.0)])
def test_invalid_duals_rejected_until_budget_covers_exhibited_perturbation(dual, threshold):
    for value in (1e-6, 1e-3, 0.9 * threshold):
        out = report(homogeneous(budgets(value)), np.zeros(2), np.array(dual))
        assert out["contract"] > 1.0, value
    out = report(homogeneous(budgets(1.1 * threshold)), np.zeros(2), np.array(dual))
    assert out["contract"] <= 1.0


def test_zero_budget_keeps_exact_zero_scale_rule():
    assert float(budget_ratio(jnp.float32(0.0), jnp.float32(0.0))) == 0.0
    assert float(budget_ratio(jnp.float32(1e-30), jnp.float32(0.0))) == np.inf
    assert float(budget_ratio(jnp.float32(np.nan), jnp.float32(1.0))) == np.inf
    compiled = homogeneous(budgets(0.0))
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)
    assert report(compiled, np.zeros(2), np.zeros(2))["contract"] == 0.0
    assert report(compiled, 1e-9 * v, np.zeros(2))["contract"] == np.inf


# --------------------------------------------------------- declared, not inferred

@pytest.mark.parametrize("bad", [
    {"primal_budget": -1.0, "dual_budget": 1.0},
    {"primal_budget": 1.0, "dual_budget": float("nan")},
    {"primal_budget": [1.0, 2.0], "dual_budget": 1.0},
    {"dual_budget": 1.0},
    {"primal_budget": 1.0, "dual_budget": 1.0, "unit": 1.0},
])
def test_budgets_are_validated_never_inferred(bad):
    with pytest.raises((ValueError, TypeError)):
        homogeneous(bad)


def test_contract_requires_relative_pdhg():
    x = ocs.Variable(2)
    with pytest.raises(ValueError):
        ocs.minimize([ocs.l1(x), ocs.equality(x)]).compile(
            residual_norm="iterate", certificate_contract=budgets(1e-6))


def test_scaled_contract_object_and_description():
    contract = ScaledContract(primal_budget=[1e-6], dual_budget=2e-6, stationarity_budget=3e-6)
    compiled = homogeneous(contract)
    described = compiled.meta["certificate_contract"]
    assert described["declared_by_caller"] is True
    assert described["primal_budget"] == [1e-6] and described["dual_budget"] == [2e-6]
    assert described["stationarity_budget"] == 3e-6 and described["variable_budget"] == 0.0
    assert described["stopping"] == "strict_only"
    assert described["experimental_stopping"] is False
    assert described["stops_the_loop"] is False
    experimental = ScaledContract(primal_budget=0.0, dual_budget=0.0,
                                  stopping="strict_or_contract")
    assert experimental.describe(1)["experimental_stopping"] is True
