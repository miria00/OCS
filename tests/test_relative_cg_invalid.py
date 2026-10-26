"""CG must reject invalid inputs even when the operator hides their values."""

import jax.numpy as jnp
import numpy as np
import pytest

import ocs


def _zero_operator_least_squares():
    # The zero map and its adjoint are valid linear operators. Neither needs
    # to inspect its input, so a derivative residual alone cannot validate b.
    operator = ocs.LinearOperator.from_forward(
        lambda x: jnp.zeros((1,), dtype=x.dtype), n=1, m=1,
    )
    variable = ocs.Variable(1)
    return ocs.minimize([
        ocs.least_squares(operator @ variable - ocs.Parameter("b")),
    ]).compile(method="cg", cg_iters=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_zero_operator_cg_rejects_invalid_rhs_in_status_and_certificate(invalid):
    compiled = _zero_operator_least_squares()
    data = {"b": jnp.asarray([invalid], dtype=jnp.float32)}

    solution = compiled.solve_instance(data)
    report = compiled.certificate(solution.x, data=data)

    assert solution.status == "numerical_error"
    assert np.isinf(solution.residual)
    assert report["r_primal"] == solution.residual


def test_zero_operator_cg_batch_marks_only_invalid_rhs_rows():
    compiled = _zero_operator_least_squares()
    data = {"b": jnp.asarray([[0.0], [jnp.nan], [jnp.inf], [-jnp.inf], [1.0]])}

    points, iterations, residuals = compiled.solve_many(data)

    np.testing.assert_array_equal(np.isinf(residuals), [False, True, True, True, False])
    # With a finite RHS, every x minimizes a constant objective for K=0.
    np.testing.assert_array_equal(np.asarray(residuals)[[0, 4]], [0.0, 0.0])
    np.testing.assert_array_equal(iterations, np.zeros(5))
    assert np.isfinite(points).all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_zero_operator_cg_rejects_nonfinite_warm_start(invalid):
    compiled = _zero_operator_least_squares()
    data = {"b": jnp.zeros(1)}
    point = jnp.asarray([invalid], dtype=jnp.float32)

    solution = compiled.solve_instance(data, warm_start=(point, None))
    report = compiled.certificate(point, data=data)
    _, _, batch_residuals = compiled.solve_many(
        {"b": jnp.zeros((1, 1))}, warm_start=(point[None], None),
    )

    assert solution.status == "numerical_error"
    assert np.isinf(solution.residual)
    assert report["r_primal"] == solution.residual
    assert np.isinf(batch_residuals[0])
