"""The zero support-direction shortcut preserves status certificates."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.library import Equality, GroupL2, L1, NonNeg, SumSquares, Zero
from ocs.solver.certificates import make_status_checks


def _a(values):
    return jnp.asarray(values, jnp.float32)


def _unused(_):
    raise AssertionError("known zero support direction must not call the adjoint")


@pytest.mark.parametrize("objective", [
    SumSquares(), GroupL2(.2, group_size=2),
    L1(.5).scaled(2).precomposed(a=-3, b=1).with_linear_quadratic(d=.4, e=.2),
])
def test_full_domains_skip_operator_and_keep_all_analytic_diagnostics(objective):
    # With g=NonNeg and both other functions finite everywhere, d=0. The
    # complete expected certificate is independent of K and of the iterate.
    check = make_status_checks(_unused, _unused, NonNeg(),
        [SumSquares(), objective], [(0, 2), (2, 4)], 2, 4, operator_norm=3.)
    actual = check(_a([2, 3]), _a([1, -2, 4, -8]), (_a([1, 2, 3, 4]), _a([0, 0])))
    expected = {
        "primal_infeasible": False, "unbounded": False,
        "primal_infeasibility_supported": True, "recession_supported": False,
        "dual_direction": np.zeros(4), "primal_direction": np.zeros(2),
        "feasible_base": np.zeros(2), "primal_infeasibility_residual": 0.,
        "primal_infeasibility_margin": 0., "recession_residual": np.inf,
        "recession_margin": 0., "base_feasibility_residual": np.inf,
    }
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)


def test_mixed_domains_call_adjoint_and_reject_nonseparating_direction():
    # x+1=0 is feasible. Dropping its nonzero K.T d would falsely certify
    # infeasibility even though the other block has full domain.
    calls = []

    def adjoint(y):
        calls.append(True)
        return y[:1]+2*y[1:]

    check = make_status_checks(lambda x: jnp.concatenate((x, 2*x)), adjoint,
        None, [Equality(), SumSquares()], [(0, 1), (1, 2)], 1, 2, operator_norm=3.)
    result = check(_a([-1]), _a([1, 1e8]), (_a([1, -100]), _a([0])))
    assert calls
    assert not bool(result["primal_infeasible"])
    np.testing.assert_array_equal(result["dual_direction"], [1, 0])
    np.testing.assert_allclose(result["primal_infeasibility_residual"], 1/3)


def test_mixed_domains_still_certify_a_true_separating_ray():
    # x=0 and x=1 contradict one another; the full-domain objective must not
    # remove their separating direction (1,-1,0).
    check = make_status_checks(lambda x: jnp.concatenate((x, x, 2*x)),
        lambda y: y[:1]+y[1:2]+2*y[2:], None,
        [Equality(), SumSquares()], [(0, 2), (2, 3)], 1, 3, operator_norm=3.)
    result = check(_a([0]), _a([1, -1, 1e8]), (_a([0, -1, 10]), _a([0])))
    assert bool(result["primal_infeasible"])
    assert not bool(result["unbounded"])
    np.testing.assert_allclose(result["dual_direction"], np.array([1, -1, 0])/np.sqrt(2), rtol=1e-6)
    assert float(result["primal_infeasibility_residual"]) == 0.


def _zero_objective_check(operator_norm=2.):
    return make_status_checks(lambda x: 2*x, _unused, None,
        [Zero()], [(0, 1)], 1, 1, operator_norm=operator_norm)


def test_full_zero_objective_keeps_recession_checks_and_detects_unboundedness():
    # min -x with no domain restrictions: full dual domains do not permit an
    # early return, because the recession certificate still proves unbounded.
    result = _zero_objective_check()(_a([2]), _a([4]), (_a([3]), _a([-1])))
    assert bool(result["recession_supported"])
    assert bool(result["unbounded"])
    assert not bool(result["primal_infeasible"])
    np.testing.assert_array_equal(result["primal_direction"], [1])
    np.testing.assert_array_equal(result["feasible_base"], [0])
    assert float(result["recession_residual"]) == 0.
    assert float(result["recession_margin"]) == 1.


@pytest.mark.parametrize("slot", range(4), ids=["x", "y", "offset", "cost"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_full_domains_preserve_every_nonfinite_input_gate(slot, bad):
    values = [_a([2]), _a([4]), _a([3]), _a([-1])]
    values[slot] = _a([bad])
    result = _zero_objective_check()(values[0], values[1], (values[2], values[3]))
    assert not bool(result["primal_infeasible"])
    assert not bool(result["unbounded"])


@pytest.mark.parametrize("norm", [np.nan, np.inf, -1.])
def test_full_domains_preserve_invalid_operator_norm_gate(norm):
    result = _zero_objective_check(norm)(_a([2]), _a([4]), (_a([3]), _a([-1])))
    assert not bool(result["primal_infeasible"])
    assert not bool(result["unbounded"])


def test_zero_direction_shortcut_keeps_per_instance_batch_gates():
    result = jax.vmap(_zero_objective_check())(
        _a([[2], [2]]), _a([[4], [np.nan]]), (_a([[3], [3]]), _a([[-1], [-1]])))
    np.testing.assert_array_equal(result["unbounded"], [True, False])
