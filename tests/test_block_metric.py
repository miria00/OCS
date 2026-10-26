"""Row-side block metrics for rotation-invariant Euclidean-norm blocks."""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

import ocs
from ocs.ir import Term
from ocs.prox import block_metric as bm
from ocs.prox.groups import build_groups, grouped_moreau_dual


def reference_dual(v, sigma, lam):
    """float64 projection onto {||y|| <= lam} in the Sigma^{-1} norm, full solves only."""
    v = np.asarray(v, np.float64)
    if np.linalg.norm(v) <= lam:
        return v
    y = lambda mu: np.linalg.solve(np.eye(v.size) + mu * sigma, v)
    hi = 1.0
    while np.linalg.norm(y(hi)) > lam:
        hi *= 2
    return y(brentq(lambda mu: np.linalg.norm(y(mu)) - lam, 0.0, hi, xtol=1e-300, rtol=1e-15))


def rotation(seed, k=3):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(k, k)))[0]


@pytest.mark.parametrize("Q", [None, "full"])
def test_weighted_conjugate_prox_matches_float64_reference(Q):
    rng = np.random.default_rng(0)
    metric = bm.BlockMetric([0.5, 3.0, 40.0], None if Q is None else rotation(1))
    c = np.exp(rng.uniform(-1, 1, size=(50, 1))) * np.ones((50, 3))
    if Q is None:
        c = c * np.exp(rng.uniform(-1, 1, size=(50, 3)))
    v = rng.normal(size=(50, 3)) * np.exp(rng.uniform(-1, 2, size=(50, 1)))
    got = np.asarray(bm.moreau_dual_rows(jnp.asarray(v, jnp.float32), jnp.asarray(c, jnp.float32),
                                         metric, 0.8), np.float64)
    for i in range(50):
        sigma = np.diag(c[i] * metric.w_host) if Q is None else c[i, 0] * metric.matrix()
        np.testing.assert_allclose(got[i], reference_dual(v[i], sigma, 0.8), rtol=5e-5, atol=5e-6)


def test_uniform_metric_reproduces_shipped_closed_form():
    g = ocs.GroupL2(0.7, 3)
    v = jnp.asarray(np.random.default_rng(3).normal(size=60) * 2, jnp.float32)
    shipped = v - 0.37 * g.prox(v / 0.37, 1 / 0.37)
    new = bm.metric_moreau(g, v, jnp.full((60,), 0.37, jnp.float32),
                           ((0, 20, bm.BlockMetric(np.ones(3))),), 3)
    np.testing.assert_allclose(new, shipped, rtol=1e-5, atol=1e-6)


def test_sharing_groups_are_separate_mathematical_blocks_and_execution_groups():
    rng = np.random.default_rng(4)
    blocks = [ocs.GroupL2(0.8, 3), ocs.GroupL2(0.8, 3)]
    slices = [(0, 30), (30, 60)]
    ranges_a = bm.normalize_term_metrics([bm.BlockMetric([1., 7., 40.], rotation(2), groups=(0, 4)),
                                          bm.BlockMetric([2., .5, 9.], groups=(4, 10))], 0, blocks[0], 30)
    ranges_b = bm.normalize_term_metrics(bm.BlockMetric([5., 1., .2], rotation(3)), 1, blocks[1], 30)
    groups = build_groups(blocks, slices, metrics=[ranges_a, ranges_b])
    assert len(groups) == 2                    # different metrics never share one execution group
    assert len(build_groups(blocks, slices)) == 1
    sigma = np.repeat(np.exp(rng.uniform(-2, 2, size=20)), 3)
    sigma[12:30] *= np.exp(rng.uniform(-.5, .5, size=18))       # diagonal range: varies inside groups
    v = rng.normal(size=60) * 3
    got = np.asarray(jax.jit(grouped_moreau_dual(groups, 60, static_sigma=jnp.asarray(sigma, jnp.float32)))(
        jnp.asarray(v, jnp.float32), jnp.asarray(sigma, jnp.float32)))
    for term, (k, ranges) in enumerate((ranges_a, ranges_b)):
        for a, b, metric in ranges:
            for gi in range(a, b):
                lo = 30 * term + 3 * gi
                s = sigma[lo:lo + 3]
                Sig = np.diag(s * metric.w_host) if metric.Q is None else s[0] * metric.matrix()
                np.testing.assert_allclose(got[lo:lo + 3], reference_dual(v[lo:lo + 3], Sig, 0.8),
                                           rtol=5e-5, atol=5e-6)


def small_family(seed=0):
    rng = np.random.default_rng(seed)
    n = 8
    K1, K2 = rng.normal(size=(12, n)), rng.normal(size=(5, n))
    x = ocs.Variable(n)
    p1, p2 = ocs.Parameter("l1"), ocs.Parameter("l2")
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         ocs.group_l2(ocs.Dense(K1) @ x + p1, lam=0.9, group_size=3),
                         ocs.nonneg(ocs.Dense(K2) @ x + p2)])
    data = {"l1": jnp.asarray(rng.normal(size=12), jnp.float32),
            "l2": jnp.asarray(rng.normal(size=5), jnp.float32)}
    return prob, data, np.vstack([K1, K2]), n


def test_one_native_standard_step_matches_reference():
    prob, data, K, n = small_family()
    rng = np.random.default_rng(1)
    metric = bm.BlockMetric([1., 4., 20.], rotation(5))
    tau = np.full(n, 0.01)
    sigma = np.concatenate([np.repeat([0.01, 0.02, 0.005, 0.015], 3), np.full(5, 0.01)])
    comp = prob.compile(equilibrate=("given", tau.astype("f"), sigma.astype("f"), 0.5),
                        block_metrics={0: metric}, pdhg_variant="standard", block=1, max_blocks=1,
                        tol=1e-12)
    assert comp.meta["scaling_classes"] == ["block", "diagonal"]
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1
    x0, y0 = rng.normal(size=n), rng.normal(size=17) * 0.1
    y0[12:] = -abs(y0[12:])
    sol = comp.solve_instance(data, warm_start=(jnp.asarray(x0, "f"), jnp.asarray(y0, "f")))
    l = np.concatenate([np.asarray(data["l1"]), np.asarray(data["l2"])])
    u = K @ x0 + l
    y = np.empty(17)
    for g in range(4):
        Sig = sigma[3 * g] * metric.matrix()
        y[3 * g:3 * g + 3] = reference_dual(y0[3 * g:3 * g + 3] + Sig @ u[3 * g:3 * g + 3], Sig, 0.9)
    y[12:] = np.minimum(y0[12:] + sigma[12:] * u[12:], 0)
    x = (x0 - tau * (K.T @ y)) / (1 + 0.5 * tau)
    # This test compares the PDHG update, not the independently recovered
    # certifying witness now exposed by the default result.
    np.testing.assert_allclose(sol.y_iterate, y, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(sol.x, x, rtol=1e-4, atol=1e-6)


def test_certificate_is_metric_free():
    prob, data, K, n = small_family()
    tau, sigma = np.full(n, 0.01, "f"), np.full(17, 0.01, "f")
    declared = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=2,
                            block_metrics={0: bm.BlockMetric([1., 4., 20.], rotation(5))})
    plain = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=2)
    x, y = jnp.ones(n), jnp.full(17, 0.1)
    left, right = declared.certificate(x, y, data), plain.certificate(x, y, data)
    # NaN-aware: the report now carries fields that are legitimately NaN when a
    # bound is unavailable, and NaN != NaN would make a plain dict comparison
    # fail for reasons that have nothing to do with the metric.
    assert set(left) == set(right)
    assert all(value == right[key] or (np.isnan(value) and np.isnan(right[key]))
               for key, value in left.items())


def test_undeclared_compile_has_no_block_metric_record():
    prob, data, K, n = small_family()
    comp = prob.compile(max_blocks=2)
    assert "block_metrics" not in comp.meta
    assert comp.meta["scaling_classes"] == ["scalar", "diagonal"]


@pytest.mark.parametrize("case, needle", [
    ("nonneg", "not a supported rotation-invariant"),
    ("not_given", "require equilibrate=('given'"),
    ("wrong_k", "does not match"),
    ("gap", "partition"),
    ("nonuniform_full", "uniform inside every mathematical block"),
    ("unsafe", "violate the PDHG step bound"),
    ("iterate_norm", "residual_norm='relative'"),
])
def test_invalid_declarations_raise(case, needle):
    prob, data, K, n = small_family()
    tau, sigma = np.full(n, 0.01, "f"), np.full(17, 0.01, "f")
    given = ("given", tau, sigma, 0.5)
    full = bm.BlockMetric([1., 2., 3.], rotation(6))
    kwargs = {"nonneg": dict(equilibrate=given, block_metrics={1: bm.BlockMetric([1.])}),
              "not_given": dict(block_metrics={0: full}),
              "wrong_k": dict(equilibrate=given, block_metrics={0: bm.BlockMetric([1., 2.])}),
              "gap": dict(equilibrate=given, block_metrics={0: [bm.BlockMetric([1., 2., 3.], groups=(0, 1)),
                                                                bm.BlockMetric([1., 2., 3.], groups=(2, 4))]}),
              "nonuniform_full": dict(equilibrate=("given", tau, np.r_[0.02, sigma[1:]].astype("f"), .5),
                                      block_metrics={0: full}),
              "unsafe": dict(equilibrate=("given", tau * 1e3, sigma * 1e3, .5), block_metrics={0: full}),
              "iterate_norm": dict(equilibrate=given, residual_norm="iterate", block_metrics={0: full})}[case]
    with pytest.raises(ValueError, match=re.escape(needle)):
        prob.compile(**kwargs)


@pytest.mark.parametrize("block", [ocs.GroupL2(1., 3).precomposed(2., 0.), ocs.SecondOrderCone(), ocs.L1(1.)])
def test_non_rotation_invariant_blocks_are_rejected(block):
    with pytest.raises(ValueError, match="not a supported"):
        bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3.]), 0, block, 30)


def test_scaled_group_l2_and_l2_are_accepted():
    assert bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3.]), 0, ocs.GroupL2(1., 3).scaled(2.), 30)[0] == 3
    assert bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3., 4.]), 0, ocs.L2(1.), 4)[0] == 4


def test_metric_validation():
    with pytest.raises(ValueError, match="orthogonal"):
        bm.BlockMetric([1., 2., 3.], rotation(7) * 1.01)
    with pytest.raises(ValueError, match="finite positive"):
        bm.BlockMetric([1., 0., 3.])
    S = np.array([[2., .5], [.5, 1.]])
    metric = bm.BlockMetric.from_matrix(S)
    np.testing.assert_allclose(metric.matrix(), S, atol=1e-12)
    assert bm.BlockMetric.from_matrix(np.diag([1., 2.])).Q is None


def _declared(prob, **kwargs):
    tau, sigma = np.full(8, 0.01, "f"), np.full(17, 0.01, "f")
    sigma[:12] = np.repeat([0.01, 0.02, 0.005, 0.015], 3)
    return prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: bm.BlockMetric([1., 4., 20.], rotation(5))}, **kwargs)


@pytest.mark.parametrize("variant", ["halpern", "standard"])
def test_layouts_and_batches_carry_the_same_metric(variant):
    prob, data, K, n = small_family()
    dense = _declared(prob, pdhg_variant=variant, block=5).solve_instance(data)
    compact = _declared(prob, pdhg_variant=variant, block=5, data_layout="compact").solve_instance(data)
    np.testing.assert_allclose(compact.x, dense.x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(compact.y, dense.y, rtol=1e-5, atol=1e-6)
    batch = {k: jnp.stack([v, v]) for k, v in data.items()}
    # solve_many_results is the batched entry held to the SAME acceptance criterion as
    # solve_instance (the tuple-returning solve_many runs the residual loop and says so
    # in its docstring); with this family now bound-capable the two loops differ, so
    # the like-for-like comparison is lane 1 of solve_many_results.
    lanes = _declared(prob, pdhg_variant=variant, block=5).solve_many_results(batch)
    np.testing.assert_allclose(lanes[1].x, dense.x, rtol=1e-5, atol=1e-6)
    if variant == "standard":
        blocks = _declared(prob, pdhg_variant=variant, block=5, dual_layout="blocks").solve_instance(data)
        np.testing.assert_allclose(blocks.y, dense.y, rtol=1e-5, atol=1e-6)


def test_refine_uses_the_declared_metric_fixed_point_map():
    prob, data, K, n = small_family()
    comp = prob.compile(equilibrate=("given", np.full(8, 0.01, "f"), np.full(17, 0.01, "f"), 0.5),
                        block_metrics={0: bm.BlockMetric([1., 4., 20.], rotation(5))},
                        max_blocks=400, tol=1e-5)
    sol = comp.solve_instance(data)
    xr, yr = comp.refine(sol.x, sol.y, data, steps=1, cg_iters=10)
    assert np.all(np.isfinite(np.asarray(xr))) and np.all(np.isfinite(np.asarray(yr)))
    # the fixed point of the declared-metric map is the same optimum: refinement stays put
    assert np.linalg.norm(np.asarray(xr) - np.asarray(sol.x)) <= 1e-2 * (1 + np.linalg.norm(np.asarray(sol.x)))


def test_identity_declaration_with_uniform_steps_is_the_scalar_tier_code_path():
    prob, data, K, n = small_family()
    tau, sigma = np.full(n, 0.01, "f"), np.full(17, 0.01, "f")
    plain = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12).solve_instance(data)
    comp = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: bm.BlockMetric(np.ones(3))})
    assert comp.meta["block_metrics"]["canonicalized_to_scalar_tier"] == [0]
    assert comp.meta["scaling_classes"] == ["scalar", "diagonal"]
    declared = comp.solve_instance(data)
    assert np.asarray(declared.x).tobytes() == np.asarray(plain.x).tobytes()
    assert np.asarray(declared.y).tobytes() == np.asarray(plain.y).tobytes()
    # per-group steps are not the scalar tier: the weighted path is kept
    sigma[:12] = np.repeat([0.01, 0.02, 0.005, 0.015], 3)
    kept = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: bm.BlockMetric(np.ones(3))})
    assert kept.meta["scaling_classes"] == ["block", "diagonal"]
    assert "canonicalized_to_scalar_tier" not in kept.meta["block_metrics"]


def test_isotropic_effective_metric_uses_the_closed_form_and_matches_reference():
    rng = np.random.default_rng(9)
    block = ocs.GroupL2(0.6, 3)
    ranges = bm.normalize_term_metrics(bm.BlockMetric([2., 2., 2.], rotation(4)), 0, block, 60)[1]
    sigma = np.repeat(np.exp(rng.uniform(-2, 2, size=20)), 3)
    checked = bm.check_static_steps(sigma, 0, 60, 3, ranges, 0)
    assert checked[0][3] is True
    v = rng.normal(size=60) * 3
    got = np.asarray(bm.metric_moreau(block, jnp.asarray(v, "f"), jnp.asarray(sigma, "f"), checked, 3))
    newton = np.asarray(bm.metric_moreau(block, jnp.asarray(v, "f"), jnp.asarray(sigma, "f"), ranges, 3))
    for g in range(20):
        Sig = sigma[3 * g] * 2.0 * np.eye(3)
        np.testing.assert_allclose(got[3 * g:3 * g + 3], reference_dual(v[3 * g:3 * g + 3], Sig, 0.6),
                                   rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(got, newton, rtol=5e-5, atol=5e-6)
    varying = sigma.copy()
    varying[1] *= 1.5
    assert bm.check_static_steps(varying, 0, 60, 3, bm.normalize_term_metrics(
        bm.BlockMetric([2., 2., 2.]), 0, block, 60)[1], 0)[0][3] is False


# ------------------------------------------------ enforced numerical domain
def test_full_metric_outside_the_validated_range_is_rejected_not_clipped():
    """A rotated shape wider than MAX_FULL_COND raises; nothing is silently clipped."""
    w = np.array([1.0, 10.0, bm.MAX_FULL_COND * 1.01])
    with pytest.raises(ValueError, match="condition number at most"):
        bm.BlockMetric(w, rotation(11))
    S = rotation(11) @ np.diag(w) @ rotation(11).T
    # The matrix interface first requires exact represented symmetry. BLAS
    # products need not preserve it bitwise; exercise the condition guard here.
    S = 0.5 * (S + S.T)
    assert np.array_equal(S, S.T)
    assert np.linalg.cond(S) > bm.MAX_FULL_COND
    with pytest.raises(ValueError, match="condition number at most"):
        bm.BlockMetric.from_matrix(S)
    ok = bm.BlockMetric(np.array([1.0, 10.0, bm.MAX_FULL_COND]), rotation(11))
    assert ok.cond == pytest.approx(bm.MAX_FULL_COND)
    assert ok.describe()["cond_limit"] == bm.MAX_FULL_COND
    # the error names the measured value, the bound, and the explicit way out
    with pytest.raises(ValueError) as err:
        bm.BlockMetric(np.array([1.0, 3000.0]), rotation(12, 2))
    text = str(err.value)
    assert "3000" in text and "restrict_condition" in text and "never clipped" in text


def test_diagonal_metric_carries_only_the_loose_guard():
    """No rotation, so no growth with the spread: the diagonal bound is far wider."""
    assert bm.MAX_DIAGONAL_COND > bm.MAX_FULL_COND
    bm.BlockMetric(np.array([1.0, bm.MAX_FULL_COND * 50]))            # admitted
    with pytest.raises(ValueError, match="condition number at most"):
        bm.BlockMetric(np.array([1.0, bm.MAX_DIAGONAL_COND * 1.01]))


def test_effective_diagonal_metric_is_checked_against_the_steps():
    """diag(sigma_i) S, not S alone, is what the prox sees when steps vary in a block."""
    block = ocs.GroupL2(0.6, 3)
    ranges = bm.normalize_term_metrics(bm.BlockMetric([1., 1., 10.]), 0, block, 30)[1]
    sigma = np.ones(30)
    bm.check_static_steps(sigma, 0, 30, 3, ranges, 0)                 # effective cond 10
    sigma[0] = 1.0 / (bm.MAX_DIAGONAL_COND * 1.01 / 10.0)             # widen one block
    with pytest.raises(ValueError, match="effective metric"):
        bm.check_static_steps(sigma, 0, 30, 3, ranges, 0)


def test_restrict_condition_is_explicit_and_reports_what_it_changed():
    S = rotation(13) @ np.diag([1.0, 20.0, 5000.0]) @ rotation(13).T
    S = 0.5 * (S + S.T)
    restricted, info = bm.restrict_condition(S, bm.MAX_FULL_COND)
    assert info["restricted"] and info["eigenvalues_raised"] == 2
    assert info["cond_before"] == pytest.approx(5000.0)
    assert info["cond_after"] <= bm.MAX_FULL_COND * (1 + 1e-12)
    assert info["step_normalization"] == "must be recomputed against the restricted shape"
    assert np.array_equal(restricted, restricted.T)                    # from_matrix demands this
    metric = bm.BlockMetric.from_matrix(restricted)                    # now admissible
    assert metric.cond <= bm.MAX_FULL_COND * (1 + 1e-9)
    # S_restricted >= S in the positive semidefinite order (steps only shrink)
    assert np.min(np.linalg.eigvalsh(restricted - S)) >= -1e-9
    w_new, winfo = bm.restrict_condition(np.array([1.0, 2.0, 400.0]), 100.0)
    np.testing.assert_allclose(w_new, [4.0, 4.0, 400.0], rtol=1e-8)
    assert winfo["eigenvalue_floor"] == pytest.approx(4.0)
    # the restricted shape is accepted at the bound it was restricted to, not a hair outside it
    for seed in range(8):
        S = rotation(20 + seed) @ np.diag([1.0, 7.0, 9999.0]) @ rotation(20 + seed).T
        r, _ = bm.restrict_condition(0.5 * (S + S.T), bm.MAX_FULL_COND)
        assert bm.BlockMetric.from_matrix(r).cond <= bm.MAX_FULL_COND
    untouched, uinfo = bm.restrict_condition(np.diag([1.0, 5.0]), 100.0)
    assert not uinfo["restricted"]
    np.testing.assert_allclose(untouched, np.diag([1.0, 5.0]))
    assert bm.restrict_condition is ocs.restrict_condition


def test_compiled_meta_records_the_numerical_domain():
    prob, data, K, n = small_family()
    tau, sigma = np.full(n, 0.01, "f"), np.full(17, 0.01, "f")
    comp = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=2,
                        block_metrics={0: bm.BlockMetric([1., 4., 20.], rotation(5))})
    domain = comp.meta["block_metrics"]["numerical_domain"]
    assert domain["max_full_cond"] == bm.MAX_FULL_COND
    assert domain["max_diagonal_cond"] == bm.MAX_DIAGONAL_COND
    assert domain["prox_kkt_bar"] == bm.PROX_KKT_BAR
    assert comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"] == pytest.approx(20.0)
