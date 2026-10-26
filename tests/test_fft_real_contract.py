"""Independent FFTMask references for the real-transform implementation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs import FFTMask


CASES = [(1, (0,)), (2, (0, 1))]
for _n in (3, 4, 8, 17, 32):
    CASES.extend((_n, idx) for idx in (
        tuple(range(_n)), (0,), (_n-1,), (1, _n-1), (2, 0, 1)))


@pytest.mark.parametrize('n,idx', CASES)
def test_real_fft_matches_independent_dft_and_adjoint(n, idx):
    rng = np.random.default_rng(1307+n+sum(idx))
    x = rng.normal(size=n).astype(np.float32)
    u = rng.normal(size=2*len(idx)).astype(np.float32)
    op = FFTMask(np.asarray(idx), n, real_fft=True)
    forward = np.asarray(jax.jit(op.matvec)(jnp.asarray(x)))
    reverse = np.asarray(jax.jit(op.rmatvec)(jnp.asarray(u)))
    spectrum = np.fft.fft(x.astype(np.float64), norm='ortho')[list(idx)]
    expected = np.concatenate([spectrum.real, spectrum.imag])
    full = np.zeros(n, np.complex128)
    full[list(idx)] = u[:len(idx)].astype(float)+1j*u[len(idx):].astype(float)
    expected_reverse = np.fft.ifft(full, norm='ortho').real
    np.testing.assert_allclose(forward, expected, rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(reverse, expected_reverse, rtol=2e-6, atol=5e-7)
    error = abs(np.dot(forward.astype(float), u)-np.dot(x.astype(float), reverse))
    assert error <= 2e-6*np.linalg.norm(x)*np.linalg.norm(u)


@pytest.mark.parametrize('idx', [(-1, 0), (0, 1, 1), (0, 20), (0, 2, 5)])
@pytest.mark.parametrize('complex_input', [False, True])
def test_fft_legacy_inputs_retain_full_transform_behavior(idx, complex_input):
    n, m = 8, len(idx)
    rng = np.random.default_rng(221)
    x, u = rng.normal(size=n).astype(np.float32), rng.normal(size=2*m).astype(np.float32)
    if complex_input:
        x = x+1j*rng.normal(size=n).astype(np.float32)
        u = u+1j*rng.normal(size=2*m).astype(np.float32)
    op = FFTMask(np.asarray(idx), n, real_fft=True)
    ji = jnp.asarray(idx)

    @jax.jit
    def previous_forward(point):
        c = jnp.fft.fft(point, norm='ortho')[ji]
        return jnp.concatenate([jnp.real(c), jnp.imag(c)])

    @jax.jit
    def previous_reverse(point):
        z = jnp.zeros(n, jnp.complex64).at[ji].set(point[:m]+1j*point[m:])
        return jnp.real(jnp.fft.ifft(z, norm='ortho'))

    np.testing.assert_allclose(np.asarray(jax.jit(op.matvec)(jnp.asarray(x))),
        np.asarray(previous_forward(jnp.asarray(x))), rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(np.asarray(jax.jit(op.rmatvec)(jnp.asarray(u))),
        np.asarray(previous_reverse(jnp.asarray(u))), rtol=2e-6, atol=5e-7)
    original = FFTMask(np.asarray(idx), n, real_fft=False)
    np.testing.assert_allclose(np.asarray(jax.jit(original.matvec)(jnp.asarray(x))),
        np.asarray(previous_forward(jnp.asarray(x))), rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(np.asarray(jax.jit(original.rmatvec)(jnp.asarray(u))),
        np.asarray(previous_reverse(jnp.asarray(u))), rtol=2e-6, atol=5e-7)


@pytest.mark.parametrize('n,idx', [(7, (0,)), (8, (0, 4))])
def test_imaginary_endpoint_dual_is_exactly_zero(n, idx):
    op = FFTMask(np.asarray(idx), n, real_fft=True)
    u = jnp.concatenate([jnp.zeros(len(idx)), jnp.arange(1, len(idx)+1, dtype=jnp.float32)])
    np.testing.assert_array_equal(np.asarray(jax.jit(op.rmatvec)(u)), np.zeros(n))


def test_random_constructor_forwards_transform_override_without_changing_subset():
    original = FFTMask.random(32, ratio=.4, seed=714, real_fft=False)
    real = FFTMask.random(32, ratio=.4, seed=714, real_fft=True)
    automatic = FFTMask.random(32, ratio=.4, seed=714)
    np.testing.assert_array_equal(np.asarray(original.idx), np.asarray(real.idx))
    point = jnp.linspace(-1, 1, 32)
    original_value = np.asarray(jax.jit(original.matvec)(point))
    np.testing.assert_array_equal(np.asarray(jax.jit(automatic.matvec)(point)), original_value)
    np.testing.assert_allclose(np.asarray(jax.jit(real.matvec)(point)), original_value,
                               rtol=2e-6, atol=5e-7)
