"""Independent power-projection checks using exact normal pairs and Clarabel."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox._power_cone import proj_pow
from ocs.prox.cones import PowerCone, PowerConeBatch


ALPHAS = (0.25, 0.5, 0.75)
SCALES = (2.0**-30, 2.0**-12, 1.0, 2.0**12, 2.0**30)


def _normal_pairs(alpha):
    points, projections = [], []
    for x, y in ((.2, 2.), (2., .2), (1., 1.), (.01, 3.), (3., .01)):
        for sign in (-1., 1.):
            magnitude = x**alpha * y**(1-alpha)
            projection = np.asarray([x, y, sign*magnitude])
            normal = np.asarray([-alpha*magnitude/x,
                                 -(1-alpha)*magnitude/y, sign])
            assert abs(np.dot(projection, normal)) < 2e-14
            for multiplier in (1e-4, .1, 10.):
                points.append(projection + multiplier*normal)
                projections.append(projection)
    return np.asarray(points), np.asarray(projections)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("scale", SCALES)
def test_exact_normal_pairs_across_units(alpha, scale):
    inputs, expected = _normal_pairs(alpha)
    values = jnp.asarray(scale*inputs)
    project = jax.jit(jax.vmap(lambda value: proj_pow(value, alpha)))
    actual = np.asarray(project(values))/scale
    error = np.linalg.norm(actual-expected, axis=1)/np.linalg.norm(inputs, axis=1)
    assert np.max(error) < 3e-6
    repeated = np.asarray(project(project(values)))/scale
    idempotence = np.linalg.norm(repeated-actual, axis=1)/np.linalg.norm(inputs, axis=1)
    assert np.max(idempotence) < 1e-6


@lru_cache(maxsize=None)
def _clarabel_projection(seed, alpha):
    clarabel = pytest.importorskip("clarabel")
    sparse = pytest.importorskip("scipy.sparse")
    source = np.random.default_rng(seed).normal(size=3).astype(np.float32).astype(np.float64)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    settings.max_iter = 300
    # Direct conic QP: min .5*x.T*x-source.T*x, -x+s=0, s in K_power.
    # This reference does not use the projector's scalar reduction or CVXPY.
    solver = clarabel.DefaultSolver(sparse.eye(3, format="csc"), -source,
                                    -sparse.eye(3, format="csc"), np.zeros(3),
                                    [clarabel.PowerConeT(alpha)], settings)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved
    return source, np.asarray(solution.x)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("seed", (67, 70, 72))
def test_failed_breadth_seeds_match_independent_reference(alpha, seed):
    source, expected = _clarabel_projection(seed, alpha)
    actual = np.asarray(jax.jit(lambda value: proj_pow(value, alpha))(jnp.asarray(source)))
    assert np.linalg.norm(actual-expected)/np.linalg.norm(source) < 5e-6
    # These alpha=.5 cases previously returned zero despite a nonzero optimum.
    if alpha == .5:
        assert np.linalg.norm(actual) > .01


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exact_faces_interior_and_polar(alpha):
    inputs = np.asarray([[0., 0., 0.], [-2., 3., 0.], [3., -2., 0.],
                         [2., 3., .1], [-2., -3., .1]], dtype=np.float32)
    expected = np.asarray([[0., 0., 0.], [0., 3., 0.], [3., 0., 0.],
                           [2., 3., .1], [0., 0., 0.]], dtype=np.float32)
    actual = jax.jit(jax.vmap(lambda value: proj_pow(value, alpha)))(jnp.asarray(inputs))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=0.)
    np.testing.assert_array_equal(np.asarray(actual)[[0, 1, 2, 4]], expected[[0, 1, 2, 4]])


@pytest.mark.parametrize("alpha", ALPHAS)
def test_public_single_and_batch_use_same_accurate_projector(alpha):
    inputs, expected = _normal_pairs(alpha)
    # Independent magnitudes per cone catch accidental whole-batch scaling.
    magnitudes = np.resize(np.asarray(SCALES), inputs.shape[0])
    scaled = inputs*magnitudes[:, None]
    stacked = jnp.asarray(scaled.T.reshape(-1))
    count = scaled.shape[0]
    batch = np.asarray(jax.jit(lambda value: PowerConeBatch(count, alpha).prox(value, 1.))
                       (stacked)).reshape(3, count).T/magnitudes[:, None]
    single = np.asarray(jax.jit(jax.vmap(lambda value: PowerCone(alpha).prox(value, 1.)))
                        (jnp.asarray(scaled)))/magnitudes[:, None]
    for actual in (batch, single):
        error = np.linalg.norm(actual-expected, axis=1)/np.linalg.norm(inputs, axis=1)
        assert np.max(error) < 3e-6
    np.testing.assert_allclose(batch, single, rtol=3e-6, atol=2e-7)
