"""Scalar status classification preserves host semantics and checked rays."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.model.compile import CompiledFamily
from ocs.prox.library import Equality, NonNeg
from ocs.solver.certificates import make_status_checks


def _family(checks=None, *, tol=1e-3):
    return CompiledFamily(
        lambda data, x, y: (x, y, 0, 0), lambda data: data,
        None, 1, 1, {"block": 7}, infeas=checks, tol=tol,
    )


def _point():
    return jnp.zeros(1), jnp.zeros(1), (jnp.zeros(1), jnp.zeros(1))


@pytest.mark.parametrize("residual,expected", [
    (0.0, "solved"), (-0.0, "solved"),
    (float(np.nextafter(np.float32(1e-3), np.float32(-np.inf))), "solved"),
    (float(np.float32(1e-3)), "inaccurate"),
    (1e-3, "solved"), (1e-2, "inaccurate"),
    (np.inf, "inaccurate"), (-np.inf, "numerical_error"),
    (np.nan, "numerical_error"), (-1e-300, "numerical_error"),
])
def test_scalar_residual_host_semantics(residual, expected):
    compiled = _family()
    x, y, data = _point()
    status, ray = compiled._status(x, y, data, 0, residual)
    assert status == expected
    assert ray is None


@pytest.mark.parametrize("bad_leaf", range(4))
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_state_or_data_overrides_certificate_flags(bad_leaf, invalid):
    def both_flags(x, y, data):
        return {"primal_infeasible": True, "unbounded": True,
                "dual_direction": y, "primal_direction": x}

    compiled = _family(both_flags)
    leaves = [jnp.ones(1) for _ in range(4)]
    leaves[bad_leaf] = jnp.asarray([invalid], jnp.float32)
    status, ray = compiled._status(
        leaves[0], leaves[1], tuple(leaves[2:]), 0, 0.0,
    )
    assert status == "numerical_error"
    assert ray is None


def _nonnegative_equality_checker():
    return make_status_checks(
        lambda x: x, lambda y: y, NonNeg(), [Equality()], [(0, 1)],
        1, 1, operator_norm=1.0,
    )


def test_infinite_residual_still_returns_valid_farkas_ray():
    # x >= 0, x + 1 = 0 is infeasible. A positive ray has K.T d >= 0
    # and l.T d > 0. The returned residual need not be finite for this status.
    compiled = _family(_nonnegative_equality_checker())
    x, y = jnp.zeros(1), jnp.asarray([3.0])
    data = (jnp.ones(1), jnp.zeros(1))
    status, ray = compiled._status(x, y, data, 0, np.inf)
    assert status == "primal_infeasible"
    ray = np.asarray(ray)
    assert ray[0] > 0 and float(np.asarray(data[0]) @ ray) > 0
    np.testing.assert_allclose(np.linalg.norm(ray), 1.0)


def test_zero_dual_direction_never_becomes_farkas_certificate():
    compiled = _family(_nonnegative_equality_checker())
    x, y = jnp.zeros(1), jnp.zeros(1)
    status, ray = compiled._status(x, y, (jnp.ones(1), jnp.zeros(1)), 0, np.inf)
    assert status == "inaccurate"
    assert ray is None


def test_true_unbounded_ray_has_feasible_base_and_strict_descent():
    # min -x subject to x >= 0, 0*x = 0. The returned x proposes e=1.
    checker = make_status_checks(
        lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
        NonNeg(), [Equality()], [(0, 1)], 1, 1, operator_norm=0.0,
    )
    compiled = _family(checker)
    data = (jnp.zeros(1), -jnp.ones(1))
    status, ray = compiled._status(jnp.ones(1), jnp.zeros(1), data, 0, np.inf)
    assert status == "unbounded"
    assert np.all(np.asarray(ray) >= 0)
    assert float(np.asarray(data[1]) @ np.asarray(ray)) < 0
    assert bool(checker(jnp.ones(1), jnp.zeros(1), data)["unbounded"])


def test_scalar_revalidates_flags_before_returning_direction():
    # Simulate a flags-only compiler path landing just beyond a numerical
    # certificate boundary. The full checker must still authorize the ray.
    compiled = _family(_nonnegative_equality_checker())
    compiled._scalar_status_checks = lambda *args: 2
    x, y = jnp.ones(1), jnp.zeros(1)
    status, ray = compiled._status(x, y, (-jnp.ones(1), jnp.zeros(1)), 0, 0.0)
    assert status == "solved"
    assert ray is None


def test_primal_infeasibility_retains_precedence_over_unboundedness():
    def both_flags(x, y, data):
        return {"primal_infeasible": True, "unbounded": True,
                "dual_direction": jnp.ones_like(y),
                "primal_direction": jnp.full_like(x, 2)}

    compiled = _family(both_flags)
    x, y, data = _point()
    status, ray = compiled._status(x, y, data, 0, 0.0)
    assert status == "primal_infeasible"
    assert isinstance(ray, jax.Array)
    np.testing.assert_array_equal(ray, [1])
