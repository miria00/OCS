"""PSD projection must retain float32 accuracy under GPU matmul defaults.

Eigenvalue clipping is not sufficient if reconstructing U diag(w+) U.T uses
reduced multiplication precision. Compare against an independent float64
spectral projection and check idempotence, including an explicitly reduced
global matmul context. These tests run on whichever JAX backend is selected.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import PSDCone


CASES = [(10, 1000), (30, 3000)]
RELATIVE_TOLERANCE = 2e-5
N30_RELATIVE_TOLERANCE = 3e-6


def symmetric_input(size, seed):
    raw = np.random.default_rng(seed).normal(size=(size, size)).astype(np.float32)
    return np.asarray(.5*(raw + raw.T), dtype=np.float32)


def numpy_projection(matrix):
    source = np.asarray(matrix, dtype=np.float64)
    source = .5*(source + source.T)
    values, vectors = np.linalg.eigh(source)
    return (vectors * np.maximum(values, 0.)) @ vectors.T


def projection_errors(project, matrix):
    """Evaluate a flat-vector projector; all comparison arithmetic is float64."""
    size = matrix.shape[0]
    projected32 = np.asarray(project(jnp.asarray(matrix.reshape(-1))))
    projected = projected32.reshape(size, size).astype(np.float64)
    repeated = np.asarray(project(jnp.asarray(projected32))).reshape(size, size).astype(np.float64)
    reference = numpy_projection(matrix)
    reference_norm = np.linalg.norm(reference)
    projected_norm = np.linalg.norm(projected)
    assert reference_norm > 0 and projected_norm > 0
    return {
        "relative_projection_error": float(np.linalg.norm(projected-reference)/reference_norm),
        "relative_idempotence_error": float(np.linalg.norm(repeated-projected)/projected_norm),
        "relative_symmetry_error": float(np.linalg.norm(projected-projected.T)/projected_norm),
        "relative_negative_eigenvalue": float(max(
            0., -np.linalg.eigvalsh(.5*(projected+projected.T)).min())/projected_norm),
    }


@pytest.mark.parametrize("size,seed", CASES)
@pytest.mark.parametrize("global_precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_psd_spectral_projection_preserves_float32_accuracy(size, seed, global_precision):
    matrix = symmetric_input(size, seed)
    eigenvalues = np.linalg.eigvalsh(matrix.astype(np.float64))
    assert eigenvalues.min() < 0 < eigenvalues.max()
    cone = PSDCone(size)
    with jax.default_matmul_precision(global_precision):
        project = jax.jit(lambda v: cone.prox(v, jnp.asarray(1., dtype=v.dtype)))
        errors = projection_errors(project, matrix)
    for name, error in errors.items():
        assert error < RELATIVE_TOLERANCE, (
            f"{name}={error:.3e}, size={size}, seed={seed}, "
            f"backend={jax.default_backend()}, global_precision={global_precision!r}"
        )


@pytest.mark.parametrize("global_precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_n30_projection_does_not_accumulate_eigenvector_normalization_error(global_precision):
    # GPU eigenvector column norms in this case deviate from one enough that
    # full-precision reconstruction alone has ~6e-6 projection/idempotence
    # error. Dividing this persistent error by a small solver step blocks KKT
    # convergence. Normalized eigenvectors and Rayleigh values remove the bias.
    cone = PSDCone(30)
    matrix = symmetric_input(30, 3000)
    with jax.default_matmul_precision(global_precision):
        project = jax.jit(lambda v: cone.prox(v, jnp.asarray(1., dtype=v.dtype)))
        errors = projection_errors(project, matrix)
    for name in ("relative_projection_error", "relative_idempotence_error"):
        assert errors[name] < N30_RELATIVE_TOLERANCE, (
            f"{name}={errors[name]:.3e}, backend={jax.default_backend()}, "
            f"global_precision={global_precision!r}"
        )
