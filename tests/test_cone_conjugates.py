"""Cone indicator conjugates use the polar cone, including for self-dual cones.

The boundary pairs below are analytic: z is primal feasible, y is a normal
to the cone at z, and <z, y> = 0.  Therefore prox(z + y) = z and the Fenchel
equality holds.  They also fix the dual sign independently of the solver.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import (
    ExpCone,
    ExpConeBatch,
    PowerCone,
    PowerConeBatch,
    PSDCone,
    SecondOrderCone,
    SOCBatch,
)


def _stack_columns(rows):
    """CVXPY cone layout: each coordinate's entries are consecutive."""
    return np.asarray(rows, dtype=np.float32).T.reshape(-1)


def _cases():
    yield "soc", SecondOrderCone(), [1, 1, 0], [-1, 1, 0]
    yield "psd", PSDCone(2), [1, 0, 0, 0], [0, 0, 0, -2]
    # At (r,s,t)=(0,1,1), the exponential-cone outward normal is (1,1,-1).
    yield "exp", ExpCone(), [0, 1, 1], [1, 1, -1]
    for alpha in (0.25, 0.5, 0.75):
        # The outward normal to x^a y^(1-a) >= z at (1,1,1).
        yield (f"power-{alpha}", PowerCone(alpha), [1, 1, 1],
               [-alpha, -(1 - alpha), 1])
    yield (
        "soc-batch", SOCBatch(2, 2),
        _stack_columns([[1, 1, 0], [2, 0, 2]]),
        _stack_columns([[-1, 1, 0], [-0.5, 0, 0.5]]),
    )
    yield (
        "exp-batch", ExpConeBatch(2),
        _stack_columns([[0, 1, 1], [0, 2, 2]]),
        _stack_columns([[1, 1, -1], [0.5, 0.5, -0.5]]),
    )
    yield (
        "power-batch", PowerConeBatch(2, 0.25),
        _stack_columns([[1, 1, 1], [2, 2, 2]]),
        _stack_columns([[-0.25, -0.75, 1], [-0.125, -0.375, 0.5]]),
    )


CASES = list(_cases())


@pytest.mark.parametrize("name,cone,z,y", CASES, ids=[c[0] for c in CASES])
def test_conjugate_accepts_polar_and_rejects_positive_dual(name, cone, z, y):
    z = jnp.asarray(z, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    assert float(cone.value(z)) == 0.0
    assert float(cone.conj_value(y)) == 0.0
    assert np.isposinf(float(cone.conj_value(-y)))
    assert float(cone.conj_value(jnp.zeros_like(y))) == 0.0
    np.testing.assert_allclose(np.dot(z, y), 0.0, atol=1e-7)


@pytest.mark.parametrize("name,cone,z,y", CASES, ids=[c[0] for c in CASES])
def test_moreau_normal_satisfies_fenchel_equality(name, cone, z, y):
    z = jnp.asarray(z, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    projected = cone.prox(z + y, jnp.asarray(1.0, dtype=jnp.float32))
    np.testing.assert_allclose(projected, z, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(z + y - projected, y, rtol=2e-5, atol=2e-5)
    # Use the exact analytic normal for conjugate membership: projection roundoff
    # must not turn this sign regression into a test of membership tolerances.
    gap = cone.value(z) + cone.conj_value(y) - jnp.vdot(z, y)
    np.testing.assert_allclose(gap, 0.0, atol=1e-7)


@pytest.mark.parametrize("name", ["soc-batch", "exp-batch", "power-batch"])
def test_batched_conjugate_rejects_one_wrong_sign_cone(name):
    _, cone, _, y = next(case for case in CASES if case[0] == name)
    y = np.asarray(y, dtype=np.float32).reshape(3, 2).copy()
    y[:, 0] *= -1
    assert np.isposinf(float(cone.conj_value(jnp.asarray(y.reshape(-1)))))
