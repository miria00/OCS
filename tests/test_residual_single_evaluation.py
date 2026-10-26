"""The residual path evaluates its stopping residual once and reports it.

SCS's rule: terminate when the candidate's residuals are small and emit that
candidate.  The loop's stopping test and the reported residual are one
evaluation at one point, so a run that stops on the criterion is solved, and the
dual it returns is the repaired dual the residual was measured at.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.solver.composite_pdhg import make_solver

MARK = 7.0   # the controlled check returns y + MARK as its "repaired" dual


def _controlled(prox_primal, *, variant="halpern", batched=True, dual_slices=None,
                max_blocks=6, tol=1e-3):
    # n = m = 2, identity operator, a dual prox that returns zero: the primal
    # prox alone decides where x goes, and the check measures ||x||_2.
    identity = lambda v: v
    check = lambda x, y, _data: (jnp.linalg.norm(x), y + MARK)
    return make_solver(
        identity, identity, jnp.ones(2), jnp.ones(2), prox_primal,
        lambda y, _sigma, _data: jnp.zeros_like(y),
        lambda x, _y, _data: jnp.linalg.norm(x), 2, 2, block=1,
        max_blocks=max_blocks, tol=tol, adaptive_omega=False, variant=variant,
        batched=batched, dual_slices=dual_slices, check=check)


def _call(solver, x0, batched):
    data = (jnp.zeros((1, 2)), jnp.zeros((1, 2))) if batched else (jnp.zeros(2), jnp.zeros(2))
    x0 = jnp.full((1, 2) if batched else (2,), x0, jnp.float32)
    y0 = jnp.zeros((1, 2) if batched else (2,), jnp.float32)
    x, y, blocks, res = solver(data, x0, y0)
    if batched:
        x, y, blocks, res = x[0], y[0], blocks[0], res[0]
    return np.asarray(x), np.asarray(y), int(blocks), float(res)


ENGINES = [dict(variant="halpern", batched=True),
           dict(variant="standard", batched=True),
           dict(variant="standard", batched=False),
           dict(variant="standard", batched=True, dual_slices=((0, 1), (1, 2)))]


@pytest.mark.parametrize("engine", ENGINES)
def test_criterion_stop_returns_the_checked_pair_and_the_tested_residual(engine):
    # x halves every block, so ||x|| crosses tol after a few blocks.
    solver = _controlled(lambda x, _tau, _data: 0.5 * x, max_blocks=40, **engine)
    x, y, blocks, res = _call(solver, 1.0, engine["batched"])
    assert blocks < 40
    assert res <= 1e-3
    assert res == pytest.approx(float(np.linalg.norm(x)), rel=1e-6)
    # The returned dual is the checked one, not the raw iterate.
    np.testing.assert_array_equal(y, np.full(2, MARK, np.float32))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_passing_start_point_is_returned_without_a_block(engine):
    solver = _controlled(lambda x, _tau, _data: x + 2.0, **engine)
    x, y, blocks, res = _call(solver, 0.0, engine["batched"])
    assert blocks == 0
    assert res == 0.0
    np.testing.assert_array_equal(x, np.zeros(2, np.float32))
    np.testing.assert_array_equal(y, np.full(2, MARK, np.float32))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_budget_exit_reports_one_check_of_the_returned_point(engine):
    # x moves away every block, so the test never passes.
    solver = _controlled(lambda x, _tau, _data: x + 2.0, max_blocks=4, **engine)
    x, y, blocks, res = _call(solver, 1.0, engine["batched"])
    assert blocks == 4
    assert res > 1e-3
    assert res == pytest.approx(float(np.linalg.norm(x)), rel=1e-6)
    np.testing.assert_array_equal(y, np.full(2, MARK, np.float32))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_non_finite_iterate_halts_without_passing(engine):
    solver = _controlled(lambda x, _tau, _data: x * jnp.nan, max_blocks=10, **engine)
    _x, _y, blocks, res = _call(solver, 1.0, engine["batched"])
    assert blocks <= 1
    assert np.isnan(res)


def test_batch_lanes_keep_their_own_checked_values():
    solver = _controlled(lambda x, _tau, _data: 0.5 * x, max_blocks=200)
    data = (jnp.zeros((3, 2)), jnp.zeros((3, 2)))
    x0 = jnp.asarray([[1.0, 1.0], [0.0, 0.0], [1e3, 1e3]], jnp.float32)
    x, y, blocks, res = solver(data, x0, jnp.zeros((3, 2), jnp.float32))
    blocks = np.asarray(blocks); res = np.asarray(res)
    assert blocks[1] == 0 and blocks[0] < blocks[2] < 200
    assert np.all(res <= 1e-3)
    np.testing.assert_allclose(res, np.linalg.norm(np.asarray(x), axis=1), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(y), np.full((3, 2), MARK, np.float32))


# ------------------------------------------------------------- real families
def _lp(seed, n=12):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n // 2, n)).astype(np.float32)
    b = A @ np.abs(rng.standard_normal(n)).astype(np.float32)
    x = ocs.Variable(n)
    return ocs.minimize([ocs.equality(ocs.Dense(A) @ x - b), ocs.nonneg(x)],
                        cost=np.abs(rng.standard_normal(n)).astype(np.float32)), {}


def _psd(seed, k=6):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((k, k)).astype(np.float32); C = C + C.T
    x = ocs.Variable(k * k)
    diag = np.zeros((k, k * k), np.float32)
    diag[np.arange(k), np.arange(k) * (k + 1)] = 1.0
    ident = ocs.LinearOperator.from_forward(lambda v: v, n=k * k, m=k * k)
    return ocs.minimize([ocs.equality(ocs.Dense(diag) @ x - np.ones(k, np.float32)),
                         ocs.psd(ident @ x, k)], cost=C.reshape(-1)), {}


def _lasso(seed, n=20):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32)
    x = ocs.Variable(n)
    b = ocs.Parameter("b")
    return (ocs.minimize([ocs.least_squares(ocs.Dense(A) @ x - b), ocs.l1(x, 0.1)]),
            {"b": rng.standard_normal(n).astype(np.float32)})


FAMILIES = {"lp": _lp, "psd": _psd, "lasso": _lasso}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("tol", [1e-3, 3e-4, 1e-4])
@pytest.mark.parametrize("variant", ["halpern", "standard"])
def test_a_criterion_stop_on_the_residual_path_is_solved(family, tol, variant):
    for seed in range(3):
        prob, data = FAMILIES[family](seed)
        kw = dict(tol=tol, max_blocks=4000, accuracy="residual")
        if variant == "standard":
            kw.update(pdhg_variant="standard", adaptive_omega=False)
        comp = prob.compile(**kw)
        sol = comp.solve_instance(data)
        if sol.termination == "criterion":
            assert sol.status == "solved", (family, tol, variant, seed, sol.residual)
            assert sol.residual <= tol
        else:
            assert sol.status != "solved" or sol.residual <= tol


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_batched_residual_solves_award_each_lane_on_its_tested_value(family):
    prob, data = FAMILIES[family](0)
    comp = prob.compile(tol=1e-4, max_blocks=4000, accuracy="residual")
    if data:
        rng = np.random.default_rng(5)
        batch = {"b": np.stack([data["b"], data["b"] * 10.0,
                                rng.standard_normal(data["b"].shape).astype(np.float32)])}
    else:
        batch = None
    rows = comp.solve_many_results(batch) if batch is not None else [comp.solve_instance(data)]
    for r in rows:
        if r.termination == "criterion":
            assert r.status == "solved" and r.residual <= 1e-4
