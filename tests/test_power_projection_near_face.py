"""Independent references where projected radius rounds to the input radius.

Uniform random triples seldom exercise this regime. The known normal pairs
below put the first coordinate near zero with a nonzero third coordinate.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import PowerConeBatch, proj_pow


@pytest.mark.parametrize("power", [2., 3., 4., 6.])
@pytest.mark.parametrize("radius", [.001, .005, .03])
def test_near_face_normal_pairs_preserve_the_nonzero_third_coordinate(power, radius):
    alpha = 1. / power
    point = np.asarray([radius**power, 1., radius])
    normal = np.asarray([-1., -(power-1.)*radius**power,
                         power*radius**(power-1.)])
    # This is an outward normal to |z| <= x**alpha*y**(1-alpha), and is
    # orthogonal to point. Projection optimality gives Pi(point+t*normal)=point.
    assert abs(np.dot(point, normal)) < 1e-15
    assert point[0]**alpha * point[1]**(1-alpha) == pytest.approx(radius)
    scales = np.repeat([1e-6, 1., 1e6], 3)
    multipliers = np.tile([.5, 1., 2.], 3) * point[0]
    inputs = scales[:, None] * (point + multipliers[:, None]*normal)
    actual = np.asarray(jax.jit(jax.vmap(lambda v: proj_pow(v, alpha)))(
        jnp.asarray(inputs, dtype=jnp.float32)), dtype=np.float64)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual/scales[:, None],
                               np.broadcast_to(point, actual.shape),
                               rtol=2e-6, atol=2e-7)


def test_archived_regression_points_are_no_farther_than_a_feasible_witness():
    # Actual triples164/180/168 from the m200,n60 cubic regression plateau.
    # The old radius parameter rounded to1, then its zero coefficient erased
    # the first coordinate and clipped the third to a distant face.
    inputs = np.asarray([[0., 1., -.004543781280517578],
                         [0., 1., .004541158676147461],
                         [1.1920928955078125e-7, 1., .007311820983886719]], "f")
    expected_feasible = inputs.astype("d")
    expected_feasible[:, 0] = abs(expected_feasible[:, 2])**3
    bound = np.linalg.norm(expected_feasible-inputs, axis=1)
    actual = np.asarray(jax.jit(jax.vmap(lambda v: proj_pow(v, 1/3)))(
        jnp.asarray(inputs)), dtype=np.float64)
    assert np.all(np.isfinite(actual))
    assert np.all(np.linalg.norm(actual-inputs, axis=1) <= bound+2e-8)
    np.testing.assert_allclose(actual, expected_feasible, atol=2e-8, rtol=1e-6)


def test_mixed_scale_batched_near_face_projection():
    radii = np.asarray([.001, -.004543781280517578, .007311820983886719])
    scales = np.asarray([1e-6, 1e6, 1.])
    triples = np.column_stack([np.zeros(3), np.ones(3), radii])
    packed = jnp.asarray((scales[:, None]*triples).T.reshape(-1), dtype=jnp.float32)
    projected = jax.jit(lambda v: PowerConeBatch(3, 1/3).prox(v, 1.))(packed)
    result = np.asarray(projected, dtype=np.float64).reshape(3, -1).T/scales[:, None]
    witnesses = np.column_stack([abs(radii)**3, np.ones(3), radii])
    np.testing.assert_allclose(result, witnesses, atol=2e-8, rtol=2e-6)
