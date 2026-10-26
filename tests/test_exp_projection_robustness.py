"""Independent exponential projection references beyond three battery seeds.

For z=(rho*s,s,s*exp(rho)) and d proportional to
(exp(rho), exp(rho)*(1-rho), -1), z belongs to the cone, d to
its polar, and z dot d=0. Thus Pi(z+d)=z by projection optimality.
These tests do not use the implementation as their reference oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.prox.cones import ExpConeBatch, proj_exp

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# exponential cone projection, no conjugate-domain contract.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


SCALES = np.asarray([1e-6, 1., 1e6], dtype=np.float64)
PROJECTION_TOLERANCE = 3e-5


def _normal_pair(rho, multiplier):
    point = np.asarray([rho, 1., np.exp(rho)], dtype=np.float64)
    point /= np.max(np.abs(point))
    normal = np.asarray([np.exp(rho), np.exp(rho)*(1.-rho), -1.])
    normal *= multiplier / np.max(np.abs(normal))
    # Independent algebra checks that the generated expected projection is
    # primal feasible and the displacement is a complementary polar normal.
    assert point[1] > 0. and point[2] > 0. and normal[0] > 0.
    assert point[0]/point[1] == pytest.approx(rho)
    assert np.log(point[2]/point[1]) == pytest.approx(rho)
    assert normal[1]/normal[0] == pytest.approx(1.-rho)
    assert -normal[2]/normal[0] == pytest.approx(np.exp(-rho))
    assert abs(np.dot(point, normal)) <= 2e-14*max(1., multiplier)
    return point+normal, point


@pytest.mark.parametrize("rho", [-30., -25., -10., -4., 0., 4., 10.])
@pytest.mark.parametrize("normal_multiplier", [.01, 1., 100.])
def test_exponential_projection_preserves_known_complementary_normal_pair(
        rho, normal_multiplier):
    source, expected = _normal_pair(rho, normal_multiplier)
    inputs = jnp.asarray(SCALES[:, None]*source, dtype=jnp.float32)
    actual = np.asarray(jax.jit(jax.vmap(proj_exp))(inputs), dtype=np.float64)
    assert np.all(np.isfinite(actual))
    restored = actual / SCALES[:, None]
    np.testing.assert_allclose(
        restored, np.broadcast_to(expected, restored.shape),
        atol=PROJECTION_TOLERANCE, rtol=PROJECTION_TOLERANCE,
    )


def test_mixed_batch_includes_boundary_ratio_outside_old_fixed_interval():
    pairs = [_normal_pair(rho, multiplier) for rho, multiplier in
             [(-30., .01), (-25., .01), (-4., 1.), (0., 1.), (10., .01)]]
    scales = np.asarray([1e-6, 1e6, 1., 1e-3, 1e3])
    sources = np.asarray([pair[0] for pair in pairs])*scales[:, None]
    expected = np.asarray([pair[1] for pair in pairs])
    cone = ExpConeBatch(len(pairs))
    packed = jnp.asarray(sources.T.reshape(-1), dtype=jnp.float32)
    projected = jax.jit(lambda v: cone.prox(v, jnp.float32(1.)))(packed)
    actual = np.asarray(projected, dtype=np.float64).reshape(3, -1).T
    np.testing.assert_allclose(actual/scales[:, None], expected,
                               atol=PROJECTION_TOLERANCE,
                               rtol=PROJECTION_TOLERANCE)


@pytest.mark.parametrize("normal_multiplier", [.01, .7564633275546291, 1., 100.])
def test_seed_zero_projection_and_normal_ray_match_independent_reference(
        normal_multiplier):
    # Reference: SCS eps=1e-11 with the original Euclidean objective, using
    # float32-rounded seed-0 data promoted to float64. Clarabel at 1e-12
    # tolerances independently agrees within 5.6e-8 in Euclidean norm.
    source = np.random.default_rng(0).normal(size=3).astype("f").astype("d")
    expected = np.asarray([.06621562256783849, .01873414975309811,
                           .6421589030070316])
    normal_ray = expected+normal_multiplier*(source-expected)
    inputs = jnp.asarray(SCALES[:, None]*normal_ray, dtype=jnp.float32)
    actual = np.asarray(jax.jit(jax.vmap(proj_exp))(inputs), dtype=np.float64)
    restored = actual/SCALES[:, None]
    np.testing.assert_allclose(
        restored, np.broadcast_to(expected, restored.shape),
        atol=PROJECTION_TOLERANCE, rtol=PROJECTION_TOLERANCE,
    )


def test_seed_zero_normal_rays_never_jump_to_a_distant_flat_face():
    # Previously 16 of these rays returned (0,0,input_z): a reconstructed
    # curved candidate failed an exp(r/s) membership recheck by float32
    # roundoff, so the algorithm discarded its much better projection.
    # Varying the positive normal magnitude models different PDHG steps.
    source = np.random.default_rng(0).normal(size=3).astype("f").astype("d")
    expected = np.asarray([.06621562256783849, .01873414975309811,
                           .6421589030070316])
    multipliers = np.geomspace(.0001, 10000., 1024)
    rays = expected+multipliers[:, None]*(source-expected)
    projected = np.asarray(jax.jit(jax.vmap(proj_exp))(
        jnp.asarray(rays, dtype=jnp.float32)), dtype=np.float64)
    # Large normal multipliers amplify float32 input rounding. The error
    # allowance accounts for that unavoidable input perturbation; the old
    # erroneous flat projections were wrong by approximately .069.
    errors = np.linalg.norm(projected-expected, axis=1)
    bounds = PROJECTION_TOLERANCE+2e-7*np.linalg.norm(rays, axis=1)
    assert np.all(np.isfinite(projected))
    assert np.all(errors <= bounds), {
        "max_error": float(np.max(errors)),
        "bad_multipliers": multipliers[errors > bounds].tolist(),
    }


@pytest.mark.parametrize("source,expected", [
    # Seed33's actual archived returned x+y, not its well-behaved data v.
    ([.3984289765357971, -.5628044605255127, .58884197473526],
     [.0934358276351459, .03168195862307781, .6048184331735259]),
    # Seed77's direct data projection used to jump to (0,0,input_z).
    ([.4277699589729309, -.5708375573158264, 2.6544606685638428],
     [.21144916877242306, .05435549867323382, 2.658882753449232]),
], ids=["seed33-returned-balanced-input", "seed77-data"])
def test_archived_failed_sweep_inputs_keep_the_accurate_curved_candidate(
        source, expected):
    # Independent Clarabel projections with gap/feasibility tolerances 1e-12;
    # provenance: profiling/exp_seed_sweep_20260906/archived_failure_projection_checks.json.
    source, expected = np.asarray(source), np.asarray(expected)
    rays = jnp.asarray(SCALES[:, None]*source, dtype=jnp.float32)
    projected = np.asarray(jax.jit(jax.vmap(proj_exp))(rays), dtype=np.float64)
    np.testing.assert_allclose(
        projected/SCALES[:, None], np.broadcast_to(expected, projected.shape),
        atol=PROJECTION_TOLERANCE, rtol=PROJECTION_TOLERANCE,
    )


@pytest.mark.parametrize("seed,reference", [
    (0, [.06621562256783849, .01873414975309811, .6421589030070316]),
    (67, [-.5734452605360987, .019210867573004498, 1.4341089734414321e-10]),
])
def test_cold_start_exponential_projection_converges_to_independent_reference(
        seed, reference):
    # Seed0 used to encounter discontinuous curved-candidate rejection;
    # seed67 exposed both the old angular cutoff and a transient endpoint
    # sign-cancellation failure in the replacement root search. Direct prox
    # tests alone cannot verify all inputs encountered during these solves.
    # Seed0's reference is SCS eps=1e-11, crosschecked with Clarabel 1e-12.
    # Seed67's reference is Clarabel with 1e-10 gap/feasibility tolerances;
    # see profiling/exp_seed_sweep_20260906/baseline_cpu_cases/exp_067.json.
    tolerance, max_blocks = 1e-3, 20
    source = np.random.default_rng(seed).normal(size=3).astype("f")
    variable = ocs.Variable(3)
    compiled = ocs.minimize(
        [ocs.sum_squares(variable), ocs.exp_cone(variable)],
        cost=ocs.Parameter("nv"),
    ).compile(tol=tolerance, max_blocks=max_blocks)
    solution = compiled.solve_instance({"nv": -source})

    assert solution.status == "solved"
    assert 0 < solution.iters <= max_blocks*compiled.meta["block"]
    assert np.isfinite(solution.residual) and solution.residual <= tolerance
    assert np.all(np.isfinite(np.asarray(solution.y)))
    point = np.asarray(solution.x, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = np.linalg.norm(source.astype("d"))
    # Independent point accuracy also bounds distance to the cone, because
    # the reference is feasible. It catches seed67's formerly accepted
    # ~1.6% point error that an objective-only assertion would miss.
    assert np.linalg.norm(point-reference) <= 2*tolerance*scale
    objective = .5*np.sum((point-source)**2)
    reference_objective = .5*np.sum((reference-source)**2)
    assert abs(objective-reference_objective) <= 2*tolerance*scale**2
