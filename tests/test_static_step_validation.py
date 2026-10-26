"""Validation contracts for concrete versus dynamic proximal steps."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs


def _block():
    return ocs.NonNeg().with_linear_quadratic(d=.03, e=.1)


BAD_STEPS = [0., -1., np.nan, np.inf, -np.inf,
             np.array(1e40, np.float64), np.array(1e-50, np.float64),
             np.array([.2, 0., .2, .2]),
             np.array([.2, np.nan, .2, .2])]


@pytest.mark.parametrize('step', BAD_STEPS)
@pytest.mark.parametrize('mode', ['eager', 'captured', 'captured_both', 'dynamic'])
def test_invalid_steps_keep_eager_exception_and_traced_nan(step, mode):
    block = _block()
    value = jnp.asarray([-.3, .4, .6, -.1], jnp.float32)
    if mode == 'eager':
        with pytest.raises(ValueError, match='finite and positive'):
            block.prox(value, step)
    elif mode == 'captured':
        result = jax.jit(lambda v: block.prox(v, step))(value)
        assert np.isnan(result).all()
    elif mode == 'captured_both':
        # The enclosing trace matters even when neither prox argument is traced.
        result, identity = jax.jit(lambda z: (block.prox(value, step), z+1.))(
            jnp.asarray(4., jnp.float32))
        assert float(identity) == 5.
        assert np.isnan(result).all()
    else:
        converted = jnp.asarray(step, dtype=jnp.float32)
        result = jax.jit(lambda v, t: block.prox(v, t))(value, converted)
        assert np.isnan(result).all()


@pytest.mark.parametrize('kind', ['linquad', 'scaled', 'affine'])
@pytest.mark.parametrize('step', [.2, np.array([.1, .2, .3, .4], np.float32)])
def test_concrete_and_dynamic_proxes_match_independent_formula(kind, step):
    value = np.array([-.3, .4, .6, -.1], np.float32)
    if kind == 'linquad':
        block = _block()
        expected = np.maximum((value.astype(float)-step*.03)/(1+step*.1), 0.)
    elif kind == 'scaled':
        block = ocs.L1(.4).scaled(1.3)
        expected = np.sign(value)*np.maximum(np.abs(value)-np.asarray(step)*.4*1.3, 0)
    else:
        block = ocs.SumSquares().precomposed(a=1.2, b=.07)
        expected = (value+np.asarray(step)*1.2*.07)/(1+np.asarray(step)*1.2**2)
    for result in (block.prox(jnp.asarray(value), step),
                   jax.jit(lambda v: block.prox(v, step))(jnp.asarray(value)),
                   jax.jit(block.prox)(jnp.asarray(value), jnp.asarray(step))):
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-8)


def test_scalar_tier_requires_exactly_uniform_step():
    block = ocs.L2(.7).scaled(1.3)
    value = jnp.asarray([.3, -.4, .8, -.7])
    good = jnp.full(4, .2)
    bad = good.at[3].set(jnp.nextafter(jnp.float32(.2), jnp.float32(1.)))
    expected = np.asarray(value)*max(1-.2*.7*1.3/np.linalg.norm(value), 0.)
    np.testing.assert_allclose(jax.jit(lambda v: block.prox(v, good))(value), expected,
                               rtol=2e-6, atol=2e-8)
    with pytest.raises(ValueError, match='uniform'):
        block.prox(value, bad)
    assert np.isnan(jax.jit(lambda v: block.prox(v, bad))(value)).all()


def test_dynamic_batched_steps_keep_rowwise_validation():
    block = _block()
    value = jnp.tile(jnp.asarray([-.3, .4, .6, -.1]), (4, 1))
    steps = jnp.asarray([[.2]*4, [.2, 0., .2, .2],
                         [.2]*4, [.2, np.nan, .2, .2]])
    result = jax.jit(jax.vmap(block.prox))(value, steps)
    expected = np.maximum((np.asarray(value[0])-.2*.03)/(1+.2*.1), 0.)
    np.testing.assert_allclose(result[[0, 2], :], np.stack([expected]*2), rtol=2e-6)
    assert np.isnan(result[[1, 3], :]).all()


def test_backend_subnormal_step_comparison_is_preserved():
    # Positive float32 subnormals may flush in the backend comparison. The
    # concrete fast path must preserve each backend evaluation context.
    # Use a direct indicator so no separate LinQuad metric division can flush
    # the step after its validity check.
    step = jnp.asarray(np.array(np.nextafter(np.float32(0.), np.float32(1.))))
    value = jnp.asarray([.1, -.2, .3, -.4])
    block = ocs.NonNeg()
    # The baseline already differs between these contexts on some backends.
    # Compare each context to its own backend's direct positivity test.
    for result, valid in (
        (jax.jit(block.prox)(value, step),
         jax.jit(lambda t: jnp.all(jnp.isfinite(t) & (t > 0)))(step)),
        (jax.jit(lambda v: block.prox(v, step))(value),
         jax.jit(lambda v: jnp.all(jnp.isfinite(step) & (step > 0)))(value))):
        if bool(valid):
            np.testing.assert_allclose(result, np.maximum(value, 0), rtol=0, atol=0)
        else:
            assert np.isnan(result).all()


@pytest.mark.parametrize('kind', ['list', 'tuple', 'nested', 'array_protocol'])
@pytest.mark.parametrize('step', [.2, 0., -1., np.inf, np.nan])
def test_dynamic_step_containers_preserve_validation(kind, step):
    class ArrayProtocol:
        def __init__(self, value):
            self.value = value

        def __jax_array__(self):
            return self.value

    value = jnp.asarray([-.3, .4, .6, -.1])
    if kind == 'nested':
        value = value.reshape(2, 2)
    block = _block()

    def solve(t):
        if kind == 'list':
            metric = [t, t, t, t]
        elif kind == 'tuple':
            metric = (t, t, t, t)
        elif kind == 'nested':
            metric = [[t, t], [t, t]]
        else:
            metric = ArrayProtocol(jnp.stack([t]*4))
        return block.prox(value, metric)

    result = jax.jit(solve)(jnp.asarray(step, jnp.float32))
    if np.isfinite(step) and step > 0:
        expected = np.maximum((np.asarray(value)-step*.03)/(1+step*.1), 0.)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-8)
    else:
        assert np.isnan(result).all()
