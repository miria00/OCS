"""Independent feasible-witness bounds for primal and polar power domains."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import PowerCone, PowerConeBatch
from ocs.solver.domain_checks import _cone_violation, make_domain_checks


def _reference_witness_distance(value, alpha, polar=False):
    """Construct explicit feasible points in float64, not a cone projection."""
    weights = np.asarray([alpha, 1-alpha, 1.]) if polar else np.ones(3)
    v = -value/weights if polar else value
    x, y, z = v
    xp, yp, az = max(x, 0.), max(y, 0.), abs(z)
    witnesses = [np.asarray([xp, yp, np.sign(z)*min(az, xp**alpha*yp**(1-alpha))])]
    if yp > 0 or az == 0:
        needed = (az/yp**(1-alpha))**(1/alpha) if az else 0.
        witnesses.append(np.asarray([max(xp, needed), yp, z]))
    if xp > 0 or az == 0:
        needed = (az/xp**alpha)**(1/(1-alpha)) if az else 0.
        witnesses.append(np.asarray([xp, max(yp, needed), z]))
    for point in witnesses:
        assert point[0] >= 0 and point[1] >= 0
        assert abs(point[2]) <= point[0]**alpha*point[1]**(1-alpha) + 1e-12*max(1., np.linalg.norm(point))
    return min(np.linalg.norm(weights*(point-v)) for point in witnesses)


def _domain(rows, scale=1.):
    count = len(rows)
    block = PowerConeBatch(count, 1/3)
    check = make_domain_checks(lambda x: x, lambda y: y, None,
                               [block], [(0, 3*count)])
    point = jnp.asarray(scale*np.asarray(rows).T.reshape(-1), dtype=jnp.float32)
    return check(point, jnp.zeros_like(point),
                 (jnp.zeros_like(point), jnp.zeros_like(point)))["r_domain"]


@pytest.mark.parametrize("row", [
    [0., 1., -.0045437812805],
    [1.1920928955e-7, 1., .00731182098],
])
@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_near_face_epigraph_errors_have_small_independent_witness(row, scale):
    # The original forward cube-root inequality amplified a sub-micro
    # epigraph error to a residual around .0025 on the real regression model.
    reference = _reference_witness_distance(np.asarray(row), 1/3)/np.linalg.norm(row)
    actual = float(_domain([row], scale))
    assert actual < 1e-6
    assert actual == pytest.approx(reference, rel=5e-5, abs=1e-12)


@pytest.mark.parametrize("alpha", [.25, 1/3, .5, .75])
@pytest.mark.parametrize("polar", [False, True])
def test_witness_distance_bounds_known_projection_distance(alpha, polar):
    rows, distances = [], []
    for x, y, sign, multiplier in [(.2, 2., 1., .1), (2., .2, -1., 3.), (1., 1., 1., .7)]:
        z = x**alpha*y**(1-alpha)
        primal = np.asarray([x, y, sign*z])
        normal = multiplier*np.asarray([-alpha*z/x, -(1-alpha)*z/y, sign])
        # Moreau decomposition establishes both exact projections of v.
        assert abs(np.dot(primal, normal)) < 2e-14
        rows.append(primal+normal)
        distances.append(np.linalg.norm(primal if polar else normal))
    values = jnp.asarray(rows, dtype=jnp.float32)
    actual = np.asarray(jax.jit(lambda value: _cone_violation(PowerCone(alpha), value, polar))(values))
    reference = np.asarray([_reference_witness_distance(np.asarray(row), alpha, polar) for row in rows])
    np.testing.assert_allclose(actual, reference, rtol=4e-6, atol=3e-7)
    assert np.all(actual >= np.asarray(distances)-1e-6*np.linalg.norm(rows, axis=1))


@pytest.mark.parametrize("alpha", [.25, 1/3, .5, .75])
def test_polar_lift_distance_uses_original_coordinate_units(alpha):
    # Mapping to the primal cone divides coordinate0 by alpha. The witness
    # distance must multiply that displacement back before reporting it.
    z = .01
    point = jnp.asarray([[0., -(1-alpha), -z]], dtype=jnp.float32)
    actual = float(_cone_violation(PowerCone(alpha), point, polar=True)[0])
    assert actual == pytest.approx(alpha*z**(1/alpha), rel=5e-6)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_large_neighbor_does_not_hide_material_local_violation(scale):
    bad = [1., 1., 2.]
    alone = float(_domain([bad], scale))
    mixed = float(_domain([[1e9, 1e9, 0.], bad], scale))
    assert alone > .1
    assert mixed == pytest.approx(alone, rel=2e-6)


def test_nonzero_third_coordinate_at_zero_epigraph_is_not_ignored():
    assert float(_domain([[0., 1., 1.]])) > .5
    assert float(_domain([[0., 0., 1.]])) > .9
    assert float(_domain([[0., 0., 0.]])) == 0.


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_power_domain_data_remains_rejected(bad):
    assert not np.isfinite(float(_domain([[bad, 1., .01]])))
