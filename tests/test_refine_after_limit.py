"""Controller contract: opt-in, honest trigger, one stage, preserved failure."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from ocs.model.compile import Solution
from ocs.solver.refine_auto import solve_with_refinement


def packet(status="inaccurate", termination="iteration_limit"):
    return Solution(np.array([1.], np.float32), np.array([2.], np.float32),
                    100, 0.1, status=status, termination=termination,
                    objective_lower_bound=1., objective_gap=.1,
                    accuracy_criterion="objective_gap",
                    y_iterate=np.array([3.], np.float32))


@pytest.mark.parametrize("termination", ["iteration_limit", "stalled"])
@pytest.mark.parametrize("accept", [True, False])
def test_opt_in_trigger_and_one_stage(termination, accept):
    first = packet(termination=termination)
    stage = replace(first, iters=0, status="solved" if accept else "inaccurate",
                    termination="criterion" if accept else "iteration_limit",
                    x=np.array([4.], np.float32), refinement={"outcome": "accepted" if accept else "refused"})
    calls = []
    def solve(data, **kw):
        assert kw == {"accuracy": "objective_gap", "refine": False}
        calls.append("solve")
        return first
    def refine(x, y, data):
        assert x is first.x and y is first.y
        calls.append("refine")
        return stage
    result = solve_with_refinement(SimpleNamespace(solve_instance=solve, refine_assessed=refine), {"b": 7})
    assert calls == ["solve", "refine"]
    assert result.iters == 100
    assert result.refinement["combined_trigger"] == termination
    assert result.refinement["first_order"]["termination"] == termination
    assert result.x is (stage.x if accept else first.x)
    assert result.termination == (stage.termination if accept else first.termination)


@pytest.mark.parametrize("status", ["solved", "unverified", "numerical_error", "unbounded", "primal_infeasible"])
def test_no_stage_on_success_or_unavailable_or_fault(status):
    first = packet(status=status)
    comp = SimpleNamespace(solve_instance=lambda *a, **kw: first,
                           refine_assessed=lambda *a: pytest.fail("stage must not run"))
    result = solve_with_refinement(comp)
    assert result.x is first.x and result.y is first.y
    assert result.status == status
    assert not result.refinement["stage_invoked"]
