"""Private zero starts preserve numerical/public contracts and input ownership."""
import jax
import jax.numpy as jnp
import numpy as np
import ocs


def _family():
    x = ocs.Variable(2)
    a = ocs.Dense(np.asarray([[1., 2.], [3., -1.], [0., 1.]], np.float32))
    return ocs.minimize([
        ocs.nonneg(x), ocs.least_squares(a @ x - ocs.Parameter("b")),
    ]).compile(equilibrate="scalar", max_blocks=30, block=20, tol=1e-4)


def test_private_zero_start_matches_explicit_start_and_keeps_inputs_reusable():
    family = _family()
    data = {"b": jnp.asarray([1., 2., -1.])}
    starts = (jnp.zeros(2), jnp.zeros(3))
    cold = family.solve_instance(data)
    warm = family.solve_instance(data, warm_start=starts)
    again = family.solve_instance(data, warm_start=starts)
    assert cold.status == warm.status == again.status
    assert cold.iters == warm.iters == again.iters
    np.testing.assert_allclose(cold.x, warm.x, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(cold.y, warm.y, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(warm.x, again.x)
    np.testing.assert_array_equal(starts[0], np.zeros(2))
    np.testing.assert_array_equal(starts[1], np.zeros(3))
    np.testing.assert_array_equal(data["b"], [1., 2., -1.])
    certificate = family.certificate(cold.x, cold.y, data)
    assert abs(max(certificate["r_primal"], certificate["r_dual"])-cold.residual) < 1e-6


def test_batched_private_starts_preserve_rows_and_public_results():
    family = _family()
    data = {"b": jnp.asarray([[1., 2., -1.], [0., 0., 0.], [jnp.nan, 1., 0.]])}
    cold = family.solve_many(data, accuracy="residual")
    warm = family.solve_many(data, warm_start=(jnp.zeros((3, 2)), jnp.zeros((3, 3))), accuracy="residual")
    for a, b in zip(cold, warm):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7, equal_nan=True)
    # Exercise the explicitly selected residual path, including its numerical
    # error lane. Default tuple batching now follows the objective-gap policy;
    # that policy is covered independently in test_criterion7_contracts.
    results = family.solve_many_results(data, accuracy="residual")
    assert results[0].status == results[1].status == "solved"
    assert results[2].status == "numerical_error"
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result.x, cold[0][index])
