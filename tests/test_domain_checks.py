"""Independent numerical KKT guards at the returned point, using analytic cones."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import (
    ExpCone, ExpConeBatch, PowerCone, PowerConeBatch, PSDCone,
    SecondOrderCone, SOCBatch,
)
from ocs.prox.library import Box, Equality, NonNeg, SumSquares
from ocs.solver.domain_checks import make_domain_checks


def _check(cone, x, y, *, c=None, l=None):
    x, y = jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    check = make_domain_checks(lambda z: z, lambda z: z, None,
                               [cone], [(0, x.size)])
    c = -y if c is None else jnp.asarray(c, dtype=x.dtype)
    l = jnp.zeros_like(x) if l is None else jnp.asarray(l, dtype=x.dtype)
    return jax.jit(check)(x, y, (l, c))


@pytest.mark.parametrize("cone,x,y", [
    (NonNeg(), [0, 2], [-1, 0]),
    (Equality(), [0, 0], [2, -3]),
    (SecondOrderCone(), [1, 1, 0], [-1, 1, 0]),
    (SOCBatch(2, 2), [1, 2, 1, 0, 0, 2], [-1, -.5, 1, 0, 0, .5]),
    (PSDCone(2), [1, 0, 0, 0], [0, 0, 0, -2]),
    (ExpCone(), [0, 1, 1], [1, 1, -1]),
    (PowerCone(.25), [1, 1, 1], [-.25, -.75, 1]),
    (ExpConeBatch(2), [0, 0, 1, 2, 1, 2], [1, .5, 1, .5, -1, -.5]),
    (PowerConeBatch(2, .25), [1, 2, 1, 2, 1, 2],
     [-.25, -.125, -.75, -.375, 1, .5]),
])
def test_analytic_complementary_pairs(cone, x, y):
    result = _check(cone, x, y)
    assert float(result["r_domain"]) < 2e-6
    assert float(result["r_polar"]) < 2e-6
    assert float(result["r_complementarity"]) == 0.0


@pytest.mark.parametrize("cone,x,y", [
    (ExpCone(), [1, 1, 1], [1, 1, -1]),
    (PowerCone(.25), [1, 1, 2], [-.25, -.75, 1]),
])
def test_huge_multiplier_cannot_hide_primal_violation(cone, x, y):
    ordinary = _check(cone, x, y)
    huge = _check(cone, x, np.asarray(y) * 1e30)
    assert float(huge["r_domain"]) > .1
    np.testing.assert_allclose(huge["r_domain"], ordinary["r_domain"])


@pytest.mark.parametrize("cone,x", [
    (ExpConeBatch(2), [0, 1, 1e8, 1, 1e8, 1]),
    (PowerConeBatch(2, .25), [1e8, 1, 1e8, 1, 1e8, 2]),
])
def test_large_neighboring_cone_cannot_hide_violation(cone, x):
    result = _check(cone, x, np.zeros(6))
    assert float(result["r_domain"]) > .1


@pytest.mark.parametrize("cone,x,y", [
    (ExpCone(), [0, 1, 1], [-1, -1, 1]),
    (PowerCone(.25), [1, 1, 1], [.25, .75, -1]),
    (SecondOrderCone(), [1, 1, 0], [1, -1, 0]),
])
def test_positive_dual_is_not_polar(cone, x, y):
    assert float(_check(cone, x, y)["r_polar"]) > .1


def test_feasible_stationary_nonoptimal_point_fails_complementarity():
    # min x subject to x >= 0.  At x=100 the normal candidate is -c=-1,
    # so feasibility and stationarity alone must not certify optimality.
    check = make_domain_checks(lambda x: jnp.zeros(1), lambda y: jnp.zeros(1),
                               NonNeg(), [], [])
    args = (jnp.zeros(1), (jnp.zeros(1), jnp.ones(1)))
    bad = jax.jit(check)(jnp.array([100.]), *args)
    good = jax.jit(check)(jnp.array([0.]), *args)
    assert float(bad["r_domain"]) == float(bad["r_polar"]) == 0
    assert float(bad["r_complementarity"]) == 1
    assert float(good["r_complementarity"]) == 0


def test_box_support_checks_normal_at_boundary():
    interior = _check(Box(0, 1), [.5], [1])
    boundary = _check(Box(0, 1), [1], [1])
    assert float(interior["r_domain"]) == float(interior["r_polar"]) == 0
    assert float(interior["r_complementarity"]) == .5
    assert float(boundary["r_complementarity"]) == 0


@pytest.mark.parametrize("alpha", [1e-5, 1., 1e5])
def test_equivalent_equality_scaling(alpha):
    # x = 1 evaluated at x=1.1; rescale both A and b, with inverse dual scale.
    check = make_domain_checks(lambda x: alpha*x, lambda y: alpha*y,
                               None, [Equality()], [(0, 1)])
    result = jax.jit(check)(jnp.array([1.1]), jnp.array([2/alpha]),
                             (jnp.array([-alpha]), jnp.array([-2.])))
    np.testing.assert_allclose(result["r_domain"], .1/1.1, rtol=2e-6)
    np.testing.assert_allclose(result["r_complementarity"], .1/1.1, rtol=2e-6)


def test_unconstrained_fast_path_has_no_operator_calls():
    def forbidden(x):
        raise AssertionError("unconstrained helper should be a static no-op")
    check = make_domain_checks(forbidden, forbidden, None, [SumSquares()], [(0, 1)])
    assert not check.has_indicators
    assert not check.has_experimental_cones
    result = jax.jit(check)(jnp.ones(1), jnp.ones(1), (jnp.zeros(1), jnp.zeros(1)))
    assert all(float(result[key]) == 0 for key in ("r_domain", "r_polar", "r_complementarity"))


def test_metadata_vmap_and_nan_rejection():
    check = make_domain_checks(lambda x: x, lambda y: y, None,
                               [ExpCone()], [(0, 3)])
    assert check.has_indicators and check.has_experimental_cones
    xs = jnp.array([[0., 1., 1.], [jnp.nan, 1., 1.]])
    y = jnp.array([1., 1., -1.])
    result = jax.jit(jax.vmap(lambda x: check(x, y, (jnp.zeros(3), -y))))(xs)
    assert float(result["r_domain"][0]) == 0
    for key in ("r_domain", "r_polar", "r_complementarity"):
        assert np.isposinf(float(result[key][1]))
