"""The stall exit of the objective-gap drive loop, on synthetic histories.

``CompiledFamily._stalled`` ends a run as ``inaccurate`` when, over a window
of ``window`` checks, neither the certified relative gap nor the legacy
residual improved by ``improvement`` (relative).  It never accepts and never
relaxes ``tol``; it only stops a loop that is not moving.

Before 0.3.1 it also demanded that every residual in the window lie at or
below an absolute 1e-7, so a gap that plateaued above ``tol`` while the
residual sat at 1e-4 to 1e-6 satisfied neither the acceptance nor the stall
and the loop ran its whole geometric schedule to ``max_blocks`` (the
subsampled Fourier lasso at N = 262,144, 2026-10-18, more than 50 minutes
where the residual criterion stops in under a second).  The last test here
covers that blind spot.
"""
import pytest

from ocs.model.compile import CompiledFamily

_stalled = CompiledFamily._stalled


def _history(gaps, residuals):
    assert len(gaps) == len(residuals)
    return [{"check": i, "segment_blocks": 2 ** i, "iterations": 10 * (2 ** (i + 1) - 1),
             "legacy_residual": r, "bound_available": True, "relative_gap": g,
             "relative_primal_infeasibility": 0.0}
            for i, (g, r) in enumerate(zip(gaps, residuals))]


def test_gap_plateau_with_residual_at_1e_minus_5_is_stalled():
    # (a) the gap moves by well under one percent and the residual sits at
    # 1e-5, far above the old absolute floor: this is a stall.
    h = _history([3.0e-4, 2.99e-4, 2.985e-4, 2.98e-4],
                 [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5])
    assert _stalled(h) is True


def test_gap_plateau_but_residual_still_falling_is_not_stalled():
    # (b) the residual improves ten percent per check: still moving.
    h = _history([3.0e-4, 2.99e-4, 2.985e-4, 2.98e-4],
                 [1.0e-5, 0.9e-5, 0.81e-5, 0.729e-5])
    assert _stalled(h) is False


def test_gap_improving_is_not_stalled():
    # (c) the gap closes by more than one percent over the window, even
    # though the residual is flat.
    h = _history([3.0e-4, 2.9e-4, 2.8e-4, 2.7e-4],
                 [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5])
    assert _stalled(h) is False
    # Exactly at the threshold counts as improvement, not as a stall.
    h = _history([1.0e-4, 1.0e-4, 1.0e-4, 0.99e-4], [1e-5] * 4)
    assert _stalled(h) is False


def test_none_gap_in_window_is_not_stalled():
    # (d) an uncertified bound anywhere in the window withholds the verdict.
    for where in range(4):
        gaps = [3.0e-4, 2.99e-4, 2.985e-4, 2.98e-4]
        gaps[where] = None
        h = _history(gaps, [1.0e-5] * 4)
        assert _stalled(h) is False
    # A None before the window does not matter: only the window is read.
    h = _history([None, 3.0e-4, 2.99e-4, 2.985e-4, 2.98e-4], [1.0e-5] * 5)
    assert _stalled(h) is True


def test_short_history_and_nonpositive_gap_are_not_stalled():
    h = _history([3.0e-4, 2.99e-4, 2.985e-4], [1.0e-5] * 3)
    assert _stalled(h) is False                      # fewer than window + 1 checks
    h = _history([0.0, 0.0, 0.0, 0.0], [1.0e-5] * 4)
    assert _stalled(h) is False                      # no positive gap to measure
    h = _history([-1e-6, -1e-6, -1e-6, -1e-6], [1.0e-5] * 4)
    assert _stalled(h) is False


def test_nonfinite_residual_withholds_stall_but_zero_plateau_does_not():
    h = _history([3.0e-4] * 4, [float("inf"), 1e-5, 1e-5, 1e-5])
    assert _stalled(h) is False
    h = _history([3.0e-4] * 4, [0.0, 0.0, 0.0, 0.0])
    assert _stalled(h) is True


@pytest.mark.parametrize("key", ["relative_gap", "legacy_residual", "relative_primal_infeasibility"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_measure_anywhere_withholds_stall(key, bad):
    h = _history([3e-4] * 4, [1e-5] * 4)
    h[2][key] = bad
    assert not _stalled(h)


def test_improving_feasibility_continues_even_with_gap_and_residual_plateaus():
    h = _history([3e-4] * 4, [1e-5] * 4)
    for row, value in zip(h, [0.2, 0.1, 0.05, 0.025]):
        row["relative_primal_infeasibility"] = value
    assert not _stalled(h)


def test_window_and_improvement_keywords_still_govern():
    h = _history([3.0e-4, 2.99e-4, 2.985e-4, 2.98e-4, 2.975e-4], [1.0e-5] * 5)
    assert _stalled(h, window=4) is True
    assert _stalled(h, window=5) is False            # not enough checks for that window
    assert _stalled(h, improvement=1e-4) is False    # the gap did move by 0.8 percent


@pytest.mark.parametrize("residual", [1.0e-4, 1.0e-5, 1.0e-6, 2.0e-7])
def test_old_blind_spot_residual_above_1e_minus_7_no_longer_blocks_a_stall(residual):
    # (e) The 0.3.0rc3 rule returned False here because residual > 1e-7, so
    # a plateaued gap with a flat residual never ended the run.  The defect
    # case had the gap at 2e-4 to 8.5e-4 and the residual at 1e-4 to 1e-6.
    h = _history([8.5e-4, 8.49e-4, 8.45e-4, 8.44e-4], [residual] * 4)
    assert _stalled(h) is True
    old_rule_rejects = any(r > 1e-7 for r in [residual] * 4)
    assert old_rule_rejects                          # the case the old rule could not end


# ---- a stall never awards: it only explains an "inaccurate" ----------------

def _compiled():
    import jax.numpy as jnp
    import ocs
    weights = jnp.array([1.0, 2.0, 3.0])
    K = ocs.LinearOperator.from_forward(lambda v: weights * v, n=3, m=3)
    x = ocs.Variable(3)
    b = ocs.Parameter("b")
    return ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - b)]).compile(
        tol=1e-4, block=10, max_blocks=40, equilibrate="scalar")


def _packet(gap, primal=1.5):
    lower = primal * (1.0 - gap)
    return {"objective_lower_bound_available": True, "relative_gap_certified": True,
            "objective_lower_bound": lower, "objective_primal_value": primal,
            "objective_at_returned_point": primal, "objective_primal_upper_bound": primal,
            "relative_primal_infeasibility": 0.0}


def test_stall_ends_as_inaccurate_and_never_as_solved():
    import jax.numpy as jnp
    from ocs.model.compile import Solution
    comp = _compiled()
    s = Solution(jnp.zeros(3), jnp.zeros(3), 0, 0.0, status="solved")
    out = comp._award(s, _packet(3.0e-3), None, stalled=True)
    assert out.status == "inaccurate"
    assert out.accuracy_criterion == "objective_gap"
    assert out.termination == "stalled"
    assert "insufficient relative improvement" in out.accuracy_detail
    assert "does not prove" in out.accuracy_detail
    out = comp._award(s, _packet(3.0e-3), None, stalled=False)
    assert out.status == "inaccurate"
    assert "numerically stationary" not in out.accuracy_detail
    # The stall flag changes the explanation only; the award itself is the
    # same `_gap_accepts` predicate, with or without a stall.
    assert comp._award(s, _packet(1.0e-5), None, stalled=True).status == "solved"
    assert comp._award(s, _packet(1.0e-5), None, stalled=False).status == "solved"


def _controller_fixture(max_blocks=127):
    """Prescribed trajectories test the real public controllers, not numerics."""
    import jax
    import jax.numpy as jnp

    def resolve(raw):
        return raw["kind"]

    def drive(data, x, y, cap):
        xx = x + cap
        residual = jnp.full(x.shape[0], 1e-5)
        # Kind 2 keeps improving its residual while the gap is fixed.
        residual = jnp.where(data[:, 0] == 2, 1.0 / (xx[:, 0] + 1), residual)
        return xx, -xx, jnp.full(x.shape[0], cap, jnp.int32), residual

    def bound(x, y, kind, rounds, return_dual=False):
        gap = jnp.where(kind[0] == 1, 0.1 * jnp.exp2(-x[0]), 3e-3)
        out = {"objective_lower_bound_available": jnp.asarray(True),
                "relative_gap_certified": jnp.asarray(True),
                "objective_lower_bound": 1.0 - gap,
                "objective_primal_value": jnp.asarray(1.0),
                "objective_at_returned_point": jnp.asarray(1.0),
                "relative_primal_infeasibility": jnp.asarray(0.0)}
        if return_dual:
            out["certifying_dual"] = y
        return out

    def bound_many(x, y, kind, rounds, return_dual=False):
        out = jax.vmap(bound, in_axes=(0, 0, 0, None))(x, y, kind, rounds)
        return (out, y) if return_dual else out

    def components(x, y, kind):
        residual = jnp.where(kind[0] == 2, 1.0 / (x[0] + 1), 1e-5)
        return {"r_primal": residual, "r_dual": residual}

    def unused(*args):
        raise AssertionError("gap controller must not use the residual solver")

    return CompiledFamily(unused, resolve, None, 1, 1, {"block": 1},
                          tol=1e-4, max_blocks=max_blocks, max_iters=max_blocks,
                          bound_capable=True, objective_bound=bound,
                          objective_bound_many=bound_many,
                          returned_components=components,
                          drive_solver=drive)


def test_scalar_stops_plateau_without_awarding_success_or_exhausting_budget():
    import jax.numpy as jnp
    comp = _controller_fixture()
    out = comp.solve_instance({"kind": jnp.array([0.0])})
    assert out.status == "inaccurate" and out.termination == "stalled"
    assert out.iters == 7 < 127
    assert len(out.gap_history) == 4
    assert float(out.x[0]) == out.iters and float(out.y[0]) == -out.iters


def test_batch_freezes_stalled_return_while_other_lanes_solve_or_reach_budget():
    import jax.numpy as jnp
    import numpy as np
    comp = _controller_fixture()
    kinds = jnp.array([[0.0], [1.0], [2.0]])
    rows = comp.solve_many_results({"kind": kinds})
    assert [r.termination for r in rows] == ["stalled", "criterion", "iteration_limit"]
    assert [r.status for r in rows] == ["inaccurate", "solved", "inaccurate"]
    assert [r.iters for r in rows] == [7, 15, 127]
    for i, row in enumerate(rows):
        single = comp.solve_instance({"kind": kinds[i]})
        assert (row.status, row.termination, row.iters) == (
            single.status, single.termination, single.iters)
        np.testing.assert_array_equal(row.x, single.x)
        np.testing.assert_array_equal(row.y, single.y)
        assert row.objective_gap == pytest.approx(single.objective_gap)
        assert row.gap_history == single.gap_history


def test_all_stalled_batch_ends_at_same_check_as_scalar():
    import jax.numpy as jnp
    comp = _controller_fixture()
    rows = comp.solve_many_results({"kind": jnp.zeros((2, 1))})
    assert all(r.termination == "stalled" and r.iters == 7 for r in rows)
