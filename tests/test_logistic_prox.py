"""Accuracy of the scalar logistic root under eager and fused execution."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from ocs.prox.library import Logistic


def _reference(v, t):
    return np.array([brentq(lambda u: u + ti * expit(u) - vi,
                            vi - ti, vi, xtol=1e-13)
                     if ti else vi for vi, ti in zip(v.astype(float), t.astype(float))])


@pytest.mark.parametrize("compiled", [False, True])
def test_logistic_prox_matches_independent_scalar_roots(compiled):
    rng = np.random.default_rng(702)
    v = rng.uniform(-40, 40, 512).astype(np.float32)
    t = np.exp(rng.uniform(-8, 14, 512)).astype(np.float32)
    # These expose rounded-root rejection and near-endpoint Newton cycling.
    v = np.concatenate((v, [-7.033399, 2.6514323, 2.6830127, 19.64188])).astype(np.float32)
    t = np.concatenate((t, [6467.438, 130.38269, 70.922615, 22.747025])).astype(np.float32)
    prox = jax.jit(Logistic().prox) if compiled else Logistic().prox
    got = np.asarray(prox(jnp.asarray(v), jnp.asarray(t)))
    expected = _reference(v, t)
    np.testing.assert_allclose(got, expected, rtol=8e-7, atol=2e-6)
    assert np.all(got >= v.astype(float) - t.astype(float) - 2e-6)
    assert np.all(got <= v + 2e-6)
    # Check optimality independently in float64, rather than only comparing
    # two implementations of the same safeguarded iteration.
    h = got.astype(float) - v + t * expit(got.astype(float))
    arithmetic_scale = np.abs(got) + np.abs(v) + np.abs(t * expit(got.astype(float)))
    assert np.all(np.abs(h) <= 4 * np.finfo(np.float32).eps * arithmetic_scale + 1e-7)


def test_logistic_prox_scalar_and_zero_metric():
    v = jnp.asarray([-100.0, -4.0, 0.0, 3.0, 100.0], jnp.float32)
    prox = jax.jit(Logistic().prox)
    np.testing.assert_array_equal(prox(v, 0.0), v)
    expected = _reference(np.asarray(v), np.full(v.shape, 50.0, np.float32))
    np.testing.assert_allclose(prox(v, 50.0), expected, rtol=8e-7, atol=2e-6)
    scalar_reference = _reference(np.array([2.6514323]), np.array([130.38269]))[0]
    np.testing.assert_allclose(Logistic().prox(2.6514323, 130.38269), scalar_reference,
                               rtol=8e-7, atol=2e-6)
