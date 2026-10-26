"""Cone projections preserve units, with independent analytic references.

For a closed convex cone C, z in C and y in its polar with z'y=0 imply
Pi_C(z+y)=z. These normal pairs supply the curved-boundary references;
the test never uses the implementation's output as its accuracy oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.cones import ExpCone, ExpConeBatch, PowerCone, PowerConeBatch


SCALES = np.asarray([1e-12, 1e-8, 1e-4, 1., 1e4, 1e8, 1e12])
ALPHAS = (.25, .5, .75)
TOLERANCE = 2e-5


def _analytic_points(kind, alpha=.5):
    if kind == "exp":
        return [
            ("interior", [-1., 1., 2.], [-1., 1., 2.]),
            ("polar", [0., -1., -1.], [0., 0., 0.]),
            ("flat", [-1., -1., 2.], [-1., 0., 2.]),
            # z=(0,1,1), y=(1,1,-1) is an analytic outward normal.
            ("curved", [1., 2., 0.], [0., 1., 1.]),
        ]
    return [
        ("interior", [1., 1., 0.], [1., 1., 0.]),
        ("polar", [-1., -1., 0.], [0., 0., 0.]),
        ("flat-x", [-1., 2., 0.], [0., 2., 0.]),
        # z=(1,1,1), y=(-a,-(1-a),1), valid for every 0<a<1.
        ("curved", [1.-alpha, alpha, 2.], [1., 1., 1.]),
        ("flat-y", [2., -1., 0.], [2., 0., 0.]),
    ]


CASES = [
    (f"exp-{name}", ExpCone(), source, reference)
    for name, source, reference in _analytic_points("exp")
]
for _alpha in ALPHAS:
    CASES += [
        (f"power-{_alpha}-{name}", PowerCone(_alpha), source, reference)
        for name, source, reference in _analytic_points("power", _alpha)
    ]


@pytest.mark.parametrize("name,cone,source,reference", CASES,
                         ids=[case[0] for case in CASES])
def test_projection_matches_analytic_reference_across_input_scales(
        name, cone, source, reference):
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    inputs = jnp.asarray(SCALES[:, None]*source, dtype=jnp.float32)
    project = jax.jit(jax.vmap(lambda row: cone.prox(row, jnp.float32(1.))))
    projected = np.asarray(project(inputs))
    repeated = np.asarray(project(jnp.asarray(projected)))
    assert np.all(np.isfinite(projected))

    # Divide in float64 so tiny output magnitudes cannot make an absolute
    # tolerance hide a wrong projection. Also works for a zero projection.
    restored = projected.astype(np.float64)/SCALES[:, None]
    np.testing.assert_allclose(
        restored, np.broadcast_to(reference, restored.shape),
        rtol=TOLERANCE, atol=TOLERANCE, err_msg=name,
    )
    np.testing.assert_allclose(
        (repeated.astype(np.float64)-projected)/SCALES[:, None],
        0., atol=TOLERANCE, rtol=0., err_msg=f"{name}: idempotence",
    )


@pytest.mark.parametrize("kind,alpha", [("exp", .5)]+[
    ("power", alpha) for alpha in ALPHAS
])
def test_mixed_magnitude_batch_normalizes_every_cone_independently(kind, alpha):
    cases = _analytic_points(kind, alpha)
    # The smallest triple needs a curved projection; the largest is polar.
    # Normalizing the entire stacked vector instead of each triple would let
    # absolute membership tolerances misclassify several of the small cones.
    choices = [3, 1, 2, 0, 3, 2 if kind == "exp" else 4, 1]
    sources = np.asarray([cases[index][1] for index in choices], dtype=np.float64)
    expected = np.asarray([cases[index][2] for index in choices], dtype=np.float64)
    rows = SCALES[:, None]*sources
    rows = np.vstack([rows, np.zeros(3)])
    expected = np.vstack([expected, np.zeros(3)])
    scales = np.concatenate([SCALES, [1.]])
    count = len(rows)
    block = (ExpConeBatch(count) if kind == "exp" else
             PowerConeBatch(count, alpha))
    # CVXPY layout is [all first coordinates; all second; all third].
    packed = jnp.asarray(rows.T.reshape(-1), dtype=jnp.float32)
    projected = np.asarray(jax.jit(
        lambda value: block.prox(value, jnp.float32(1.)))(packed))
    actual = projected.reshape(3, count).T.astype(np.float64)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(
        actual/scales[:, None], expected,
        rtol=TOLERANCE, atol=TOLERANCE,
    )
    np.testing.assert_array_equal(actual[-1], np.zeros(3))


@pytest.mark.parametrize("cone", [ExpCone()]+[
    PowerCone(alpha) for alpha in ALPHAS
])
def test_exact_zero_is_an_exact_finite_projection(cone):
    result = jax.jit(lambda value: cone.prox(value, jnp.float32(1.)))(
        jnp.zeros(3, dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(result), np.zeros(3))
