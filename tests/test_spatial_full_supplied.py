"""The supplied path that carries a spatial-full cone metric (stage 3).

Gate for `SPEC_STAGE3.md` of `candidates/spatial_full_soc_20260930`, sha256
8f441194306374cbbe779a9540253d6117b7a051e8fe85fb6733938f87ca660c, frozen before
any of this was written. Stage 2's primitive is gated separately in
`tests/test_spatial_soc_prox.py` and nothing here re-tests it.

WHAT THIS STAGE IS. The class `M = diag(a, Q diag(b) Q^T)` is the one the archived
n = 2048 comparison used, and `METRIC_CLASS_BOUNDARY.md` established that the
released engine could apply NO such metric: `SecondOrderCone` and `SOCBatch`
declare `scaling_class = SCALAR`, and the only supplied cone carrier was
`ConeAutomorphism`, a class that meets this one only at its scalar members. This
stage makes the engine able to apply it, through the existing supplied path, and
that is all it does. Nothing here constructs a metric.

THE STEP CANCELS, which is why this is a small change. The dual update for
`Sigma = c M` is `v - M P_M(M^{-1} v)` with no `c` in it, because `P_{cM} = P_M`
and `P_M` is positively homogeneous. The first test measures that rather than
taking it on the derivation's word.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import Term
from ocs.prox import block_metric as bm
from ocs.prox import cone_metric as cm
from ocs.prox import cones
from ocs.prox import spatial_soc as ss
from ocs.prox.groups import build_groups, grouped_moreau_dual

F32 = np.float32


def factors(rng, cones_count, d, spread=8.0, shared=False):
    """A spatial-full metric whose spatial block is NOT a multiple of the identity."""
    count = 1 if shared else cones_count
    a = np.exp(rng.uniform(-0.5, 0.5, count)).astype(np.float64)
    b = np.exp(rng.uniform(-np.log(spread), np.log(spread), (count, d)))
    Q = np.stack([np.linalg.qr(rng.standard_normal((d, d)))[0]
                  for _ in range(count)])
    if shared:
        return a[0], b[0], Q[0]
    return a, b, Q


def one_cone_metric(rng, d, **kw):
    a, b, Q = factors(rng, 1, d, **kw)
    return ss.SpatialFullSOCMetric(np.reshape(a, (-1,))[:1] if np.ndim(a) else a,
                                   np.reshape(b, (1, d)) if np.ndim(b) > 1 else b,
                                   Q if np.ndim(Q) == 3 else Q)


# ---------------------------------------------------------------------------
# the carrier reaches the block through the existing seam
# ---------------------------------------------------------------------------
def test_the_carrier_is_accepted_on_a_second_order_cone_term():
    rng = np.random.default_rng(1)
    d = 5
    a, b, Q = factors(rng, 1, d, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    k, ranges = bm.normalize_term_metrics(carrier, 0, cones.SecondOrderCone(),
                                          d + 1)
    assert k == d + 1
    assert len(ranges) == 1 and ranges[0][:2] == (0, 1)
    applied = ranges[0][2]
    assert isinstance(applied, ss.SpatialFullSOCMetric)
    assert applied.layout == ("soc",)
    # NOT converted to the other carrier, which is the whole point of the class
    assert not isinstance(applied, cm.ConeAutomorphism)


def test_the_carrier_is_accepted_on_a_batched_term_with_its_layout():
    rng = np.random.default_rng(2)
    q, d = 4, 3
    a, b, Q = factors(rng, q, d)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    block = cones.SOCBatch(q, d)
    k, ranges = bm.normalize_term_metrics(carrier, 0, block, q * (1 + d))
    assert k == q * (1 + d)
    assert ranges[0][2].layout == ("socbatch", q, d)
    assert ranges[0][2].cones == q


@pytest.mark.parametrize("block,size,why", [
    (cones.PSDCone(3), 9, "SECOND-ORDER cone block"),
    (cones.ExpCone(), 3, "SECOND-ORDER cone block"),
    (ocs.SumSquares(), 4, "SECOND-ORDER cone block"),
])
def test_the_carrier_is_refused_on_every_other_block(block, size, why):
    rng = np.random.default_rng(3)
    a, b, Q = factors(rng, 1, size - 1, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    with pytest.raises(ValueError, match=why):
        bm.normalize_term_metrics(carrier, 0, block, size)


def test_a_list_of_cone_carriers_is_refused():
    rng = np.random.default_rng(4)
    a, b, Q = factors(rng, 1, 4, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    # The AUTOMORPHISM message is unchanged word for word by this stage, so the
    # new carrier has its own sentence and existing callers are told exactly
    # what they were told before.
    with pytest.raises(ValueError, match="exactly ONE SpatialFullSOCMetric"):
        bm.normalize_term_metrics([carrier, carrier], 0,
                                  cones.SecondOrderCone(), 5)


def test_a_mismatched_k_or_cone_count_is_refused():
    rng = np.random.default_rng(5)
    a, b, Q = factors(rng, 1, 4, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    with pytest.raises(ValueError, match="is for k = 5 and SecondOrderCone"):
        bm.normalize_term_metrics(carrier, 0, cones.SecondOrderCone(), 9)
    a3, b3, Q3 = factors(rng, 3, 3)
    per_cone = ss.SpatialFullSOCMetric(a3, b3, Q3)
    with pytest.raises(ValueError, match="carries 3 per-cone factors and the term stacks 5"):
        bm.normalize_term_metrics(per_cone, 0, cones.SOCBatch(5, 3), 5 * 4)


@pytest.mark.parametrize("spoil", ["a_zero", "a_negative", "b_zero", "b_nan",
                                   "q_inf"])
def test_a_factor_that_is_not_positive_and_finite_is_refused(spoil):
    rng = np.random.default_rng(6)
    d = 4
    a, b, Q = factors(rng, 2, d)
    if spoil == "a_zero":
        a[1] = 0.0
    elif spoil == "a_negative":
        a[1] = -1.0
    elif spoil == "b_zero":
        b[1, 2] = 0.0
    elif spoil == "b_nan":
        b[1, 2] = np.nan
    else:
        Q[1, 0, 0] = np.inf
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    with pytest.raises(ValueError, match="finite|non-finite"):
        bm.normalize_term_metrics(carrier, 0, cones.SOCBatch(2, d), 2 * (d + 1))


# ---------------------------------------------------------------------------
# the step cancels, and the update is the primitive's
# ---------------------------------------------------------------------------
def test_the_step_cancels_out_of_the_dual_update():
    """The derivation, measured. Any positive step gives the same point."""
    rng = np.random.default_rng(7)
    d = 6
    a, b, Q = factors(rng, 1, d, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",))
    ranges = ((0, 1, carrier),)
    v = jnp.asarray(rng.standard_normal(d + 1), jnp.float32)
    base = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0), ranges))
    for step in (1e-3, 0.25, 7.0, 1e3):
        got = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(step),
                                                ranges))
        assert np.array_equal(base, got), f"step {step} changed the update"


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_step_that_is_not_finite_and_positive_gives_nan(bad):
    rng = np.random.default_rng(8)
    d = 3
    a, b, Q = factors(rng, 1, d, shared=True)
    ranges = ((0, 1, ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",))),)
    v = jnp.asarray(rng.standard_normal(d + 1), jnp.float32)
    out = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(bad), ranges))
    assert np.all(np.isnan(out)), out


def test_a_term_with_several_distinct_steps_is_refused_not_averaged():
    rng = np.random.default_rng(9)
    d = 3
    a, b, Q = factors(rng, 1, d, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q)
    k, ranges = bm.normalize_term_metrics(carrier, 0, cones.SecondOrderCone(),
                                          d + 1)
    sigma = np.full(d + 1, 0.5, np.float64)
    bm.check_static_steps(sigma, 0, d + 1, k, ranges, 0)     # uniform: accepted
    sigma[2] = 0.7
    with pytest.raises(ValueError, match="ONE dual step for the whole term"):
        bm.check_static_steps(sigma, 0, d + 1, k, ranges, 0)


def test_the_update_is_the_primitive_and_its_other_exact_form():
    rng = np.random.default_rng(10)
    d = 8
    a, b, Q = factors(rng, 1, d, shared=True)
    carrier = ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",))
    ranges = ((0, 1, carrier),)
    v = jnp.asarray(rng.standard_normal(d + 1), jnp.float32)
    applied = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0),
                                                ranges))
    production, _s = ss.project_polar_step(jnp.reshape(v, (1, -1)), carrier)
    assert np.allclose(applied, np.asarray(production)[0], rtol=1e-6, atol=1e-7)
    other, _s2 = ss.project_polar_step_moreau(jnp.reshape(v, (1, -1)), carrier)
    rel = float(np.linalg.norm(applied - np.asarray(other)[0])
                / max(1e-30, np.linalg.norm(applied)))
    print(f"the two exact dual forms differ by {rel:.3e} relative")
    assert rel <= 1e-5


# ---------------------------------------------------------------------------
# the three mutation tests: nothing is flattened and nothing is dropped
# ---------------------------------------------------------------------------
def _update(carrier, v):
    return np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0),
                                             ((0, 1, carrier),)))


def test_the_applied_metric_depends_on_a_on_b_and_on_q():
    """Perturb each factor in turn. Each must move the dual update.

    This is the test that a conversion to `ConeAutomorphism`, a flattening of
    `b` to one value, or a dropped `Q` would fail, and those are exactly the
    three things `SPEC_STAGE3.md` forbids.
    """
    rng = np.random.default_rng(11)
    d = 6
    a, b, Q = factors(rng, 1, d, shared=True)
    v = jnp.asarray(rng.standard_normal(d + 1), jnp.float32)
    base = _update(ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",)), v)
    scale = max(1e-30, float(np.linalg.norm(base)))

    moved = {}
    a2 = np.array(a, np.float64) * 1.05
    moved["a"] = float(np.linalg.norm(
        _update(ss.SpatialFullSOCMetric(a2, b, Q, layout=("soc",)), v) - base))
    b2 = np.array(b, np.float64)
    b2[1] *= 1.05
    moved["one entry of b"] = float(np.linalg.norm(
        _update(ss.SpatialFullSOCMetric(a, b2, Q, layout=("soc",)), v) - base))
    theta = 0.05
    rot = np.eye(d)
    rot[0, 0] = rot[1, 1] = np.cos(theta)
    rot[0, 1], rot[1, 0] = -np.sin(theta), np.sin(theta)
    moved["Q"] = float(np.linalg.norm(
        _update(ss.SpatialFullSOCMetric(a, b, rot @ Q, layout=("soc",)), v)
        - base))
    for name, delta in moved.items():
        print(f"perturbing {name} moved the update by {delta / scale:.3e} relative")
        assert delta / scale > 1e-4, \
            f"the update does not depend on {name}, so it is being ignored"


def test_a_scalar_spatial_block_reduces_to_the_scalar_tier():
    """`b = a 1` IS the scalar tier, and must be recognised rather than routed.

    `diag(a, Q diag(b) Q^T) = a I` exactly when `b = a 1`, whatever `Q` is, so a
    declaration that reduces to the shipped metric should reach the shipped path.
    """
    rng = np.random.default_rng(12)
    d = 4
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    unit = ss.SpatialFullSOCMetric(np.float64(1.0), np.ones(d), Q)
    k, ranges = bm.normalize_term_metrics(unit, 0, cones.SecondOrderCone(), d + 1)
    checked = bm.check_static_steps(np.float64(0.5), 0, d + 1, k, ranges, 0)
    assert bm.equivalent_to_scalar_tier(np.float64(0.5), 0, d + 1, checked)
    not_unit = ss.SpatialFullSOCMetric(np.float64(1.0),
                                       np.concatenate([[2.0], np.ones(d - 1)]), Q)
    k2, r2 = bm.normalize_term_metrics(not_unit, 0, cones.SecondOrderCone(), d + 1)
    c2 = bm.check_static_steps(np.float64(0.5), 0, d + 1, k2, r2, 0)
    assert not bm.equivalent_to_scalar_tier(np.float64(0.5), 0, d + 1, c2)


def test_the_half_power_is_exact_and_any_other_is_refused():
    """`M^{1/2}` is in the same class, so the spectral check is not an estimate."""
    rng = np.random.default_rng(13)
    d = 5
    a, b, Q = factors(rng, 1, d, shared=True)
    host = rng.standard_normal(d + 1)
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        carrier = ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",))
        u = jnp.asarray(host, jnp.float64)
        half = ss.apply_spatial_full(carrier, u, power=0.5)
        twice = ss.apply_spatial_full(carrier, half, power=0.5)
        once = ss.apply_spatial_full(carrier, u, power=1)
        rel = float(np.linalg.norm(np.asarray(twice) - np.asarray(once))
                    / np.linalg.norm(np.asarray(once)))
    finally:
        jax.config.update("jax_enable_x64", previous)
    carrier = ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",))
    u = jnp.asarray(host, jnp.float32)
    print(f"(M^1/2)^2 against M: {rel:.3e} relative")
    assert rel <= 1e-6
    with pytest.raises(ValueError, match="power must be 1 or 0.5"):
        ss.apply_spatial_full(carrier, u, power=2)


def test_the_layout_puts_each_cones_metric_on_that_cone():
    """One cone given a far-off metric, and only that cone's output moves.

    A layout read wrongly would apply the right metric to the wrong cone, which
    no accuracy check on the whole vector would catch.
    """
    rng = np.random.default_rng(14)
    q, d = 5, 3
    a, b, Q = factors(rng, q, d, spread=1.0001)
    block = cones.SOCBatch(q, d)
    size = q * (1 + d)
    v = jnp.asarray(rng.standard_normal(size), jnp.float32)
    _k, ranges = bm.normalize_term_metrics(ss.SpatialFullSOCMetric(a, b, Q), 0,
                                           block, size)
    base = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0), ranges))
    target = 2
    a2 = np.array(a, np.float64)
    a2[target] *= 50.0
    _k2, r2 = bm.normalize_term_metrics(ss.SpatialFullSOCMetric(a2, b, Q), 0,
                                        block, size)
    moved = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0), r2))
    rowsb = ss.rows(("socbatch", q, d), base)
    rowsm = ss.rows(("socbatch", q, d), moved)
    deltas = [float(np.linalg.norm(np.asarray(rowsm)[j] - np.asarray(rowsb)[j]))
              for j in range(q)]
    print("per-cone movement:", [f"{x:.3e}" for x in deltas])
    assert deltas[target] > 1e-5, "the cone whose metric changed did not move"
    for j in range(q):
        if j != target:
            assert deltas[j] <= 1e-6, \
                f"cone {j} moved although only cone {target}'s metric changed"


# ---------------------------------------------------------------------------
# through the compiled public interface
# ---------------------------------------------------------------------------
def ball_family(c, A, mu, metric=None, tau=None, sigma=None):
    """The archived socp_ball shape, with an optional supplied cone metric."""
    n = c.size
    x = ocs.Variable(n)
    terms = [ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, F32)) @ x),
             ocs.soc(ocs.Dense(np.vstack([np.zeros((1, n), F32), A])) @ x
                     + np.concatenate([[1.0], np.zeros(n)]).astype(F32))]
    prob = ocs.minimize(terms, cost=jnp.asarray(c, F32))
    if metric is None:
        return prob.compile(tol=1e-5, max_blocks=200)
    return prob.compile(tol=1e-5, max_blocks=200,
                        equilibrate=("given", tau, sigma),
                        block_metrics={1: metric})


def test_a_supplied_spatial_full_metric_compiles_and_solves():
    """The reliability check of the specification, not a performance claim."""
    rng = np.random.default_rng(15)
    n = 24
    c = rng.standard_normal(n).astype(F32)
    A = rng.standard_normal((n, n)).astype(F32)
    mu = np.float32(1e-2)

    plain = ball_family(c, A, mu)
    base = plain.solve_instance({})
    assert np.all(np.isfinite(np.asarray(base.x)))

    d = n                                            # the cone's spatial size
    a1, b1, Q1 = factors(rng, 1, d, spread=4.0, shared=True)
    metric = ss.SpatialFullSOCMetric(a1, b1, Q1)
    low, high = ss.metric_amplitude(metric)
    # The caller normalizes: with Sigma = sigma M the step bound needs
    # sigma_max * tau * ||K||^2 * high <= 1, so a conservative pair is built from
    # the plain arm's own measured norm and the metric's exact top eigenvalue.
    scale = float(plain.meta.get("scaled_norm", 1.0)) or 1.0
    sigma_value = 0.25 / (high * max(scale, 1.0))
    tau = jnp.full((n,), np.float32(0.25 / max(scale, 1.0)))
    # MIGRATION (0.3.3). Only the scale is borrowed from the plain arm, never its dual
    # layout: a plain compile folds the diagonal sum_squares term into the primal step
    # (meta["routing"], FOLD_POLICY in ocs.model.compile), so base.y has n + 1 rows,
    # while this arm supplies steps and a metric, an explicit metric mapping under
    # which every non-identity term is a dual term as declared (2 n + 1 rows). Steps
    # are therefore bound to the dual terms by position: dual term 0 is the
    # sum_squares term, dual term 1 the cone, the same indices as block_metrics.
    assert plain.meta["routing"]["folded"] is True
    assert int(np.asarray(base.y).size) == n + 1
    sigma = {0: np.float32(0.25 / max(scale, 1.0)),      # the sum_squares rows
             1: np.float32(sigma_value)}                  # the cone rows, Sigma = sigma M
    comp = ball_family(c, A, mu, metric=metric, tau=tau, sigma=sigma)
    routing = comp.meta["routing"]
    assert routing["folded"] is False and routing["dual_rows"] == 2 * n + 1
    assert [t["role"] for t in routing["terms"]] == ["dual", "dual"]
    assert routing["terms"][0]["rows"] == [0, n] and routing["terms"][1]["rows"] == [n, 2 * n + 1]
    assert routing["terms"][1]["metric_source"] == "block_metrics[1]"
    out = comp.solve_instance({})
    x = np.asarray(out.x, np.float64)
    assert np.all(np.isfinite(x)), "the supplied metric produced a non-finite point"
    assert int(np.asarray(out.y).size) == 2 * n + 1
    record = comp.meta["block_metrics"]["terms"][0]
    assert record["block_family"] == "cone"
    assert record["sharing_groups"][0]["class"] == "spatial_full_soc"
    assert record["sharing_groups"][0]["spatial_is_scalar"] is False
    print(f"supplied arm: status {out.status}, residual {float(out.residual):.3e}, "
          f"metric eigenvalues [{low:.3e}, {high:.3e}], "
          f"measured scaled norm "
          f"{comp.meta['block_metrics']['measured_scaled_norm']:.6f}")
    # the independent quality check both arms must pass: the returned point is
    # finite, the ball is respected to the declared tolerance, and the objective
    # is not worse than the plain arm's by more than the tolerance allows
    norm_Ax = float(np.linalg.norm(np.asarray(A, np.float64) @ x))
    assert norm_Ax <= 1.0 + 1e-3, f"||A x|| = {norm_Ax}"


def test_the_record_names_the_class_and_whose_normalization_it_is():
    rng = np.random.default_rng(16)
    d = 4
    a, b, Q = factors(rng, 1, d, shared=True)
    entry = ss.describe_spatial_full(
        ss.SpatialFullSOCMetric(a, b, Q, layout=("soc",)))
    assert entry["class"] == "spatial_full_soc"
    assert entry["spatial_is_scalar"] is False
    assert entry["eigenvalue_min"] <= entry["eigenvalue_max"]
    assert "CALLER" in entry["normalization"]
    assert "no certified ceiling" in entry["normalization"]


def test_an_automorphism_and_a_spatial_full_carrier_are_not_interchanged():
    """The dispatch is keyed by carrier type, and the two are different metrics."""
    rng = np.random.default_rng(17)
    d = 4
    w = np.concatenate([[2.0], 0.1 * rng.standard_normal(d)])
    auto = cm.ConeAutomorphism.soc(w)
    a, b, Q = factors(rng, 1, d, shared=True)
    full = ss.SpatialFullSOCMetric(a, b, Q)
    block = cones.SecondOrderCone()
    _k1, r1 = bm.normalize_term_metrics(auto, 0, block, d + 1)
    _k2, r2 = bm.normalize_term_metrics(full, 0, block, d + 1)
    assert isinstance(r1[0][2], cm.ConeAutomorphism)
    assert isinstance(r2[0][2], ss.SpatialFullSOCMetric)
    assert not isinstance(r2[0][2], cm.ConeAutomorphism)
    v = jnp.asarray(rng.standard_normal(d + 1), jnp.float32)
    one = np.asarray(cm.automorphism_moreau(None, v, jnp.float32(1.0), r1))
    two = np.asarray(ss.spatial_full_moreau(None, v, jnp.float32(1.0), r2))
    rel = float(np.linalg.norm(one - two) / max(1e-30, np.linalg.norm(one)))
    print(f"the two carriers' updates differ by {rel:.3e} relative")
    assert rel > 1e-3, \
        "the two carriers give the same update, so this fixture cannot tell them apart"
