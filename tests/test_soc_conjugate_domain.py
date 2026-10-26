"""Exact polar-cone conjugate-domain contract on the second-order cone blocks.

dom f^* for the indicator of a self-dual cone K is the polar -K.  The test is
exact on the represented data: ||y[1:]|| <= -y[0], no slack, so the objective
bound is reachable on second-order cone families under the default acceptance.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from ocs.prox.cones import SecondOrderCone, SOCBatch


def _soc_members(rng, d, inside_polar):
    z = rng.normal(size=d)
    nz = np.linalg.norm(z)
    s = -(nz * (1.5 if inside_polar else 0.5))
    return np.concatenate([[s], z]).astype(np.float32)


def test_single_cone_contract_is_exact_and_moreau_holds():
    blk = SecondOrderCone()
    assert blk.bound_capable and blk.is_indicator
    rng = np.random.default_rng(0)
    for inside in (True, False):
        y = jnp.asarray(_soc_members(rng, 4, inside))
        assert bool(blk.conj_in_domain(y)) is inside
        assert float(blk.conj_domain_radius(y)) == (np.inf if inside else 0.0)
        p = blk.conj_domain_project(y)
        assert bool(blk.conj_in_domain(p))
        # Moreau: projection onto K plus projection onto -K recovers y
        np.testing.assert_allclose(np.asarray(p + blk._project(y)), np.asarray(y), rtol=4e-6, atol=4e-6)   # the proposal sits a few ulps inside the polar
    # boundary: equality is IN the domain, and one ulp outside is OUT
    z = np.array([3.0, 4.0], np.float32)
    assert bool(blk.conj_in_domain(jnp.asarray([-5.0, 3.0, 4.0], jnp.float32)))
    assert not bool(blk.conj_in_domain(jnp.asarray([np.nextafter(np.float32(-5.0), np.float32(0.0)), 3.0, 4.0], jnp.float32)))
    assert not bool(blk.conj_in_domain(jnp.asarray([np.nan, 0.0, 0.0], jnp.float32)))


def test_batched_cones_decide_every_cone():
    q, d = 5, 3
    blk = SOCBatch(q, d)
    assert blk.bound_capable and blk.is_indicator
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(d, q))
    nz = np.linalg.norm(Z, axis=0)
    t = -1.5 * nz
    y = np.concatenate([t, Z.reshape(-1, order="F")]).astype(np.float32)
    assert bool(blk.conj_in_domain(jnp.asarray(y)))
    y_bad = y.copy(); y_bad[2] = -0.5 * nz[2]          # one cone leaves the polar
    assert not bool(blk.conj_in_domain(jnp.asarray(y_bad)))
    assert float(blk.conj_domain_radius(jnp.asarray(y_bad))) == 0.0
    p = blk.conj_domain_project(jnp.asarray(y_bad))
    assert bool(blk.conj_in_domain(p))
    np.testing.assert_allclose(np.asarray(p + blk._project(jnp.asarray(y_bad))), y_bad, rtol=4e-6, atol=4e-6)


def test_rounded_norm_cannot_accept_an_outside_polar_witness():
    # sqrt(2) rounds downward in f32. The old f32 norm compared equal and
    # accepted this represented point, although its exact norm exceeds -y0.
    y=jnp.asarray([-np.float32(np.sqrt(2)),1.,1.],jnp.float32)
    assert np.linalg.norm(np.asarray(y[1:],dtype=np.float64)) > -float(y[0])
    assert not bool(SecondOrderCone().conj_in_domain(y))
    p=SecondOrderCone().conj_domain_project(y)
    assert bool(SecondOrderCone().conj_in_domain(p))
    assert np.linalg.norm(np.asarray(p[1:],dtype=np.float64)) <= -float(p[0])
