"""3-D imaging differences preserve edge orientation, boundaries and adjoints.

Small independently assembled incidence matrices fix the mathematical
contract. The old scatter implementation additionally checks floating-point
and dtype compatibility; it is not the mathematical reference.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from ocs import Gradient3D


SHAPES = [(1, 1, 1), (1, 2, 3), (2, 1, 4), (3, 2, 1), (2, 3, 4), (5, 4, 3)]


def _incidence(shape):
    n = int(np.prod(shape))
    matrix = np.zeros((3*n, n), np.float64)
    for voxel in np.ndindex(shape):
        source = np.ravel_multi_index(voxel, shape)
        for axis in range(3):
            neighbor = list(voxel)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                target = np.ravel_multi_index(tuple(neighbor), shape)
                matrix[3*source+axis, source] = -1
                matrix[3*source+axis, target] = 1
    return matrix


def _old_forward(x, shape):
    v = x.reshape(shape)
    dx = jnp.zeros_like(v).at[:-1].set(v[1:]-v[:-1])
    dy = jnp.zeros_like(v).at[:, :-1].set(v[:, 1:]-v[:, :-1])
    dz = jnp.zeros_like(v).at[:, :, :-1].set(v[:, :, 1:]-v[:, :, :-1])
    return jnp.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], 1).reshape(-1)


def _old_adjoint(g, shape):
    g = g.reshape(-1, 3)
    dx, dy, dz = (g[:, i].reshape(shape) for i in range(3))
    result = jnp.zeros(shape, jnp.float32)
    result = result.at[1:].add(dx[:-1]).at[:-1].add(-dx[:-1])
    result = result.at[:, 1:].add(dy[:, :-1]).at[:, :-1].add(-dy[:, :-1])
    result = result.at[:, :, 1:].add(dz[:, :, :-1]).at[:, :, :-1].add(-dz[:, :, :-1])
    return result.reshape(-1)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_gradient3d_matches_independent_edges_and_adjoint(shape, compiled):
    operator = Gradient3D(*shape)
    matrix = _incidence(shape)
    rng = np.random.default_rng(206)
    x = rng.standard_normal(operator.shape[1]).astype(np.float32)
    y = rng.standard_normal(operator.shape[0]).astype(np.float32)
    forward = jax.jit(operator.matvec) if compiled else operator.matvec
    adjoint = jax.jit(operator.rmatvec) if compiled else operator.rmatvec
    actual_forward, actual_adjoint = np.asarray(forward(x)), np.asarray(adjoint(y))
    assert actual_forward.dtype == actual_adjoint.dtype == np.float32
    np.testing.assert_allclose(actual_forward, matrix@x, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual_adjoint, matrix.T@y, rtol=3e-7, atol=8e-7)
    lhs = np.dot(actual_forward.astype(np.float64), y)
    rhs = np.dot(x.astype(np.float64), actual_adjoint)
    np.testing.assert_allclose(lhs, rhs, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient3d_batched_values_and_legacy_order(shape):
    operator = Gradient3D(*shape)
    rng = np.random.default_rng(317)
    x = rng.standard_normal((4, operator.shape[1])).astype(np.float32)
    y = rng.standard_normal((4, operator.shape[0])).astype(np.float32)
    new_forward = np.asarray(jax.jit(jax.vmap(operator.matvec))(x))
    new_adjoint = np.asarray(jax.jit(jax.vmap(operator.rmatvec))(y))
    old_forward = np.asarray(jax.jit(jax.vmap(lambda v: _old_forward(v, shape)))(x))
    old_adjoint = np.asarray(jax.jit(jax.vmap(lambda v: _old_adjoint(v, shape)))(y))
    np.testing.assert_array_equal(new_forward, old_forward)
    np.testing.assert_allclose(new_adjoint, old_adjoint, rtol=3e-7, atol=8e-7)
    matrix = _incidence(shape)
    np.testing.assert_allclose(new_adjoint, y.astype(np.float64)@matrix,
                               rtol=3e-7, atol=8e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient3d_ignores_nonfinite_inactive_boundary_rows(shape):
    operator = Gradient3D(*shape)
    y = np.zeros((*shape, 3), np.float32)
    y[-1, :, :, 0] = np.nan
    y[:, -1, :, 1] = np.inf
    y[:, :, -1, 2] = -np.inf
    actual = np.asarray(jax.jit(operator.rmatvec)(y.reshape(-1)))
    np.testing.assert_array_equal(actual, np.zeros(operator.shape[1], np.float32))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_gradient3d_preserves_forward_dtype_and_float32_adjoint(dtype):
    shape = (2, 3, 4)
    operator = Gradient3D(*shape)
    rng = np.random.default_rng(493)
    x = jnp.asarray(rng.standard_normal(operator.shape[1]), dtype)
    y = jnp.asarray(rng.standard_normal(operator.shape[0]), dtype)
    new_forward = jax.jit(operator.matvec)(x)
    new_adjoint = jax.jit(operator.rmatvec)(y)
    assert new_forward.dtype == dtype
    assert new_adjoint.dtype == jnp.float32
    np.testing.assert_array_equal(new_forward, _old_forward(x, shape))
    np.testing.assert_allclose(new_adjoint, _old_adjoint(y, shape), rtol=3e-7, atol=8e-7)


def test_gradient3d_autodiff_transpose_matches_public_adjoint():
    operator = Gradient3D(2, 3, 4)
    rng = np.random.default_rng(716)
    x = jnp.asarray(rng.standard_normal(operator.shape[1]), jnp.float32)
    y = jnp.asarray(rng.standard_normal(operator.shape[0]), jnp.float32)
    expected, = jax.linear_transpose(operator.matvec, x)(y)
    np.testing.assert_allclose(operator.rmatvec(y), expected, rtol=3e-7, atol=8e-7)


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_gradient3d_preserves_empty_volume_operator(shape):
    operator = Gradient3D(*shape)
    empty = jnp.zeros(0, jnp.float32)
    assert operator.shape == (0, 0)
    for forward, adjoint in [(operator.matvec, operator.rmatvec),
                             (jax.jit(operator.matvec), jax.jit(operator.rmatvec))]:
        actual_forward, actual_adjoint = forward(empty), adjoint(empty)
        assert actual_forward.shape == actual_adjoint.shape == (0,)
        assert actual_forward.dtype == actual_adjoint.dtype == jnp.float32
        np.testing.assert_array_equal(actual_forward, _old_forward(empty, shape))
        np.testing.assert_array_equal(actual_adjoint, _old_adjoint(empty, shape))
    batch = jnp.zeros((2, 0), jnp.float32)
    assert jax.jit(jax.vmap(operator.matvec))(batch).shape == (2, 0)
    assert jax.jit(jax.vmap(operator.rmatvec))(batch).shape == (2, 0)
