"""The primal block's intrinsic zero offset needs no vector norm reduction."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.solver.domain_checks import _block_checks


@pytest.mark.parametrize('kind', ['nonneg', 'box', 'equality', 'soc', 'psd', 'exp', 'power'])
@pytest.mark.parametrize('value', ['ordinary', 'zero', 'nan', 'inf'])
def test_implicit_zero_offset_matches_explicit_zero_geometry(kind, value):
    blocks = {'nonneg': ocs.NonNeg(), 'box': ocs.Box(-.2, 1.3),
              'equality': ocs.Equality(), 'soc': ocs.SecondOrderCone(),
              'psd': ocs.PSDCone(2), 'exp': ocs.ExpCone(),
              'power': ocs.PowerCone(.4)}
    block = blocks[kind]
    size = 4 if kind == 'psd' else 3
    s = jnp.asarray([.3, -.4, .2, .7][:size], jnp.float32)
    y = jnp.asarray([-.1, .2, .05, -.4][:size], jnp.float32)
    if value == 'zero':
        s, y = jnp.zeros_like(s), jnp.zeros_like(y)
    elif value == 'nan':
        s = s.at[0].set(jnp.nan)
    elif value == 'inf':
        y = y.at[0].set(jnp.inf)

    explicit = jax.jit(lambda x, dual: _block_checks(
        block, x, dual, x, jnp.zeros_like(x)))(s, y)
    implicit = jax.jit(lambda x, dual: _block_checks(
        block, x, dual, x, None))(s, y)
    for actual, expected in zip(implicit, explicit):
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_actual_affine_offset_keeps_its_independent_scale():
    block = ocs.NonNeg()
    kx = jnp.ones(4)
    l = -2*jnp.ones(4)
    result = jax.jit(lambda offset: _block_checks(
        block, kx+offset, jnp.zeros(4), kx, offset))(l)
    # ||negative part(Kx+l)||/max(||Kx||,||l||) = sqrt(4)/sqrt(16).
    assert float(result[0]) == .5
    wrong_zero = _block_checks(block, kx+l, jnp.zeros(4), kx, None)
    assert float(wrong_zero[0]) == 1.
