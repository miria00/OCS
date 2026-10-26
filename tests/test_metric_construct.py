"""equilibrate="construct": the caller declares a class, the compiler builds the metric."""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import Term
from ocs.prox import block_metric as bm
from ocs.solver import metric_construct as mc

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# metric construction, not an acceptance claim.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def small_family(seed=0, n=8):
    rng = np.random.default_rng(seed)
    K1, K2 = rng.normal(size=(12, n)), rng.normal(size=(5, n))
    x = ocs.Variable(n)
    p1, p2 = ocs.Parameter("l1"), ocs.Parameter("l2")
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         ocs.group_l2(ocs.Dense(K1) @ x + p1, lam=0.9, group_size=3),
                         ocs.nonneg(ocs.Dense(K2) @ x + p2)])
    data = {"l1": jnp.asarray(rng.normal(size=12), jnp.float32),
            "l2": jnp.asarray(rng.normal(size=5), jnp.float32)}
    return prob, data, np.vstack([K1, K2]), n


def digest(a):
    a = np.asarray(jax.device_get(a))
    return a.dtype.str, a.shape, a.tobytes()


def endpoints(comp, data):
    sol = comp.solve_instance(data)
    cert = comp.certificate(sol.x, sol.y, data)
    return (digest(sol.x), digest(sol.y), int(sol.iters), str(sol.status),
            tuple(sorted((k, digest(v)) for k, v in cert.items())))


# ------------------------------------------------------- the Diamond-Boyd tier
def test_default_declaration_is_byte_identical_to_shipped_sgd():
    """The paper's claim, made literally true: with every block declaring the
    class it natively supports, the construction IS the shipped program."""
    prob, data, _K, _n = small_family()
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    built = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3)
    assert built.meta["metric_construction"]["regime"] == "smfe"
    assert built.meta["metric_construction"]["natural_declaration"] is True
    assert built.meta["metric_construction"]["bit_identical_to_sgd"] is True
    assert built.meta["scaling_classes"] == shipped.meta["scaling_classes"]
    assert "block_metrics" not in built.meta
    assert built.meta["scaled_norm"] == shipped.meta["scaled_norm"]
    assert endpoints(built, data) == endpoints(shipped, data)


def test_explicit_natural_classes_are_the_same_program():
    prob, data, _K, _n = small_family()
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    built = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                         metric_classes={0: ("scalar", "term"), 1: ("diagonal", "per_block")})
    assert built.meta["metric_construction"]["regime"] == "smfe"
    assert endpoints(built, data) == endpoints(shipped, data)


def test_free_rows_on_a_coupled_block_stay_the_shipped_program():
    """A diagonal declaration on a coupled block drops that term's tie: the
    numbers still come from the shipped SMFE, applied through the weighted prox."""
    prob, _data, _K, _n = small_family()
    built = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                         metric_classes={0: ("diagonal", "per_block")})
    report = built.meta["metric_construction"]
    assert report["regime"] == "smfe"
    assert report["bit_identical_to_sgd"] is False        # a different declared constraint set
    assert report["scalar_spans"] == ()
    assert built.meta["scaling_classes"] == ["block", "diagonal"]
    assert built.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["kind"] == "diagonal"


# ------------------------------------------------------------- the full tier
@pytest.mark.parametrize("declared, kind", [(("full", "term"), "full"),
                                            (("diagonal", "term"), "diagonal")])
def test_shared_shape_is_built_normalized_and_applied(declared, kind):
    prob, data, _K, _n = small_family()
    comp = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                        metric_classes={0: declared}, metric_updates=12)
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic"
    assert report["rate_claim"].startswith("NO convergence rate")
    assert report["constructor_status"] == 0
    group = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert group["kind"] == kind
    # ONE stored shape for all four mathematical blocks of the term
    record = comp.meta["effective_metric"]["terms"][0]
    assert record["metric_sharing_groups"] == [
        {"blocks": [0, 4], "kind": kind, "k": 3, "cond": group["cond"],
         "isotropic_closed_form": False}]
    assert record["distinct_dual_steps"] == 1        # Sigma_i = c S, c uniform
    # the compiler re-measures the metric it was handed and would refuse it >= 1
    measured = comp.meta["block_metrics"]["measured_scaled_norm"]
    assert 0 < measured < 1
    assert measured == pytest.approx(0.9, rel=2e-3)   # the reported safety, re-measured
    sol = comp.solve_instance(data)
    assert np.all(np.isfinite(np.asarray(sol.x)))


def test_operator_product_budget_is_reported_and_charged_in_full():
    """Every product is charged, including the shipped program's diagnostics."""
    prob, _data, _K, _n = small_family()
    built = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                         metric_classes={0: ("full", "term")}, metric_updates=80, metric_probes=2)
    shipped = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3)
    geodesic = built.meta["metric_construction"]["operator_products"]
    smfe = shipped.meta["metric_construction"]["operator_products"]
    assert geodesic == {"gradient": 320, "normalization": 30, "total": 350,
                        "note": geodesic["note"]}
    assert smfe == {"gradient": 320, "normalization": 30, "diagnostics": 32, "total": 382,
                    "note": smfe["note"]}
    # the same gradient budget: the richer metric is not bought with more probes
    assert geodesic["gradient"] == smfe["gradient"]


def test_cost_is_reported_with_compilation_separated_from_execution():
    prob, _data, _K, _n = small_family()
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("full", "term")}, metric_updates=12)
    seconds = comp.meta["metric_construction"]["seconds"]
    assert set(seconds) >= {"lower", "compile", "execute", "normalization", "cold_total"}
    assert all(seconds[k] >= 0 for k in ("lower", "compile", "execute", "normalization"))


# ------------------------------------------------- the enforced numerical domain
def test_out_of_domain_full_shape_is_refused_not_clipped():
    w = np.array([1.0, 1.0, 1.0])
    rotation = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    with pytest.raises(ValueError, match="condition number at most 100"):
        bm.BlockMetric(w * np.array([1.0, 1.0, 1000.0]), rotation)


def test_restrict_recomputes_and_reports_the_normalization(monkeypatch):
    """metric_domain='restrict' restricts EXPLICITLY, re-measures, and records both."""
    prob, _data, _K, _n = small_family()
    inside = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                          metric_classes={0: ("full", "term")}, metric_updates=12)
    built_cond = inside.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"]
    assert built_cond > 1.0
    limit = float(np.sqrt(built_cond))            # strictly between 1 and the built shape
    monkeypatch.setattr(bm, "MAX_FULL_COND", limit)
    with pytest.raises(ValueError, match="Nothing is clipped for you"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                     metric_classes={0: ("full", "term")}, metric_updates=12)
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("full", "term")}, metric_updates=12,
                        metric_domain="restrict")
    restriction = comp.meta["metric_construction"]["restriction"]
    applied = restriction["applied"][0]
    assert applied["restricted"] is True
    assert applied["cond_before"] == pytest.approx(built_cond, rel=1e-6)
    assert applied["cond_after"] <= limit * (1 + 1e-6)
    assert applied["eigenvalues_raised"] >= 1
    assert applied["eigenvalues_before"] != applied["eigenvalues_after"]
    # raising eigenvalues raises ||Sigma^1/2 K T^1/2||, so the normalization was
    # recomputed against the RESTRICTED shape, not carried over
    assert (restriction["measured_after_restriction"]
            > restriction["measured_before_restriction"])
    assert comp.meta["metric_construction"]["operator_products"]["normalization"] == 60
    assert 0 < comp.meta["block_metrics"]["measured_scaled_norm"] < 1


# ---------------------------------------------------------- refusals and record
@pytest.mark.parametrize("kwargs, needle", [
    (dict(metric_classes={1: "full"}), "only a valid proximal metric for a rotation-invariant"),
    (dict(metric_classes={0: ("full", "per_block")}), "must be shared"),
    (dict(metric_classes={0: "bogus"}), "class must be one of"),
    (dict(metric_classes={7: "scalar"}), "not a dual term index"),
    (dict(metric_classes={1: ("diagonal", "term")}), "SHARED diagonal shape needs"),
    (dict(metric_columns=[(0, 3)]), "must cover"),
])
def test_bad_declarations_are_refused(kwargs, needle):
    prob, _data, _K, _n = small_family()
    with pytest.raises(ValueError, match=re.escape(needle)):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, **kwargs)


def test_construct_arguments_require_the_construct_path():
    prob, _data, _K, _n = small_family()
    with pytest.raises(ValueError, match="require equilibrate='construct'"):
        prob.compile(max_blocks=2, metric_classes={0: "scalar"})
    with pytest.raises(ValueError, match="choose one of them"):
        prob.compile(max_blocks=2, equilibrate="construct",
                     block_metrics={0: bm.BlockMetric(np.ones(3))})


def test_effective_metric_is_recorded_on_every_compile():
    """Three groupings of the same rows, kept apart, written after compilation."""
    prob, _data, _K, n = small_family()
    plain = prob.compile(max_blocks=2, tol=1e-3)
    record = plain.meta["effective_metric"]
    assert [t["term"] for t in record["terms"]] == [0, 1]
    assert record["terms"][0]["mathematical_block"] == {"size_k": 3, "count": 4, "coupled": True}
    assert record["terms"][0]["metric_sharing_groups"] is None       # no shape declared
    assert record["terms"][0]["distinct_dual_steps"] == 1            # collapsed to one step
    assert record["n_execution_groups"] == len(record["execution_groups"])
    assert not any(g["carries_block_metric"] for g in record["execution_groups"])

    built = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                         metric_classes={0: ("full", "term")}, metric_updates=12)
    record = built.meta["effective_metric"]
    assert record["terms"][0]["mathematical_block"]["size_k"] == 3
    assert record["terms"][0]["metric_sharing_groups"][0]["blocks"] == [0, 4]
    assert record["terms"][0]["metric_sharing_groups"][0]["kind"] == "full"
    assert sum(g["carries_block_metric"] for g in record["execution_groups"]) == 1
    # batching is a throughput choice and never merges the two terms' metrics
    assert len(record["execution_groups"]) == 2


def test_column_tie_is_declarable_and_honoured():
    prob, _data, _K, n = small_family()
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("full", "term")},
                        metric_columns=[(0, n // 2), (n // 2, n)], metric_updates=12)
    assert comp.meta["metric_construction"]["regime"] == "geodesic"
    assert comp.meta["metric_construction"]["columns"]["metric_sharing_groups"] == 2
    assert comp.meta["effective_metric"]["primal"]["distinct_steps"] == 2


# ------------------------------------------------------ the declaration algebra
def test_declaration_canonicalization():
    blocks = [ocs.GroupL2(0.9, 3), ocs.NonNeg()]
    slices = [(0, 12), (12, 17)]
    decls = mc.normalize_declarations(None, blocks, slices)
    assert [(d.cls, d.sharing, d.k, d.count) for d in decls] == [
        ("scalar", "term", 3, 4), ("diagonal", "per_block", 1, 5)]
    assert mc.natural_declaration(decls)
    assert mc.shipped_expressible(decls, "diagonal")[0]
    # a scalar on single-coordinate blocks IS a free diagonal; say so once
    same = mc.normalize_declarations({1: ("scalar", "per_block")}, blocks, slices)
    assert (same[1].cls, same[1].sharing) == ("diagonal", "per_block")
    # one scalar per mathematical block is not a constraint the shipped program has
    per_block = mc.normalize_declarations({0: ("scalar", "per_block")}, blocks, slices)
    assert mc.shipped_expressible(per_block, "diagonal") == (
        False, "term 0 declares one scalar per mathematical block (4 of them); "
               "the shipped program ties whole spans only")


def test_operator_only_contract_of_the_geodesic_constructor():
    """The constructor may call matvec/rmatvec and nothing else on K."""
    rng = np.random.default_rng(0)
    K = rng.normal(size=(12, 8))
    calls = {"mv": 0, "rmv": 0}

    def matvec(x):
        calls["mv"] += 1
        return jnp.asarray(K, jnp.float32) @ x

    def rmatvec(y):
        calls["rmv"] += 1
        return jnp.asarray(K.T, jnp.float32) @ y

    rows = (mc._Group("shared_full", 3, 4),)
    columns = (mc._Group("diagonal", 8, 1),)
    program, info = mc.make_geodesic_constructor(matvec, rmatvec, rows, columns,
                                                 iterations=5, probes=2)
    answer = program(jax.random.PRNGKey(0), mc.initial_state(rows), mc.initial_state(columns))
    assert int(answer["status"]) == 0
    assert int(answer["operator_products"]) == 2 * 5 * 2
    assert info["targets"]["row"] == pytest.approx(np.sqrt(8 / 12))
    A = np.asarray(answer["A"][0], np.float64)
    np.testing.assert_allclose(A, A.T, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    assert mc.state_bytes(rows) == 4 * 9      # ONE 3x3 shape, not one per block


def test_diagonal_geodesic_step_is_the_shipped_step_under_the_parameter_map():
    """A measured statement, not an argued one.

    On a diagonal declaration the geodesic step and Diamond & Boyd's projected
    SGD step are the SAME map in the coordinate value = 2u, with mu = gamma,
    shift = 1 and box = M. This is why declaring diagonal is declaring their
    method. It is not byte identity -- the averaging and the probe streams still
    differ -- which is why the shipped program is invoked rather than rebuilt.
    """
    rng = np.random.default_rng(11)
    k, count, probes, gamma, M, target = 3, 5, 2, 1.0, 6.0, 0.8
    u = rng.uniform(-1.5, 1.5, size=(count, k))
    samples = jnp.asarray(rng.normal(size=(probes, count * k)), jnp.float32)
    group = mc._Group("diagonal", k, count)
    for t in range(4):
        eta_shipped = 2.0 / (gamma * (t + 2.0))
        eta_geodesic = 2.0 / ((gamma / 4) * ((t + 1) + 1))     # mu = gamma, shift = 1
        assert eta_geodesic / 4 == pytest.approx(eta_shipped, rel=1e-12)
        # shipped: row energy of the scaled operator, gradient, step, box clip
        w = np.asarray(samples, np.float64).reshape(probes, count, k)
        row_energy = np.exp(2 * u) * np.mean(w * w, axis=0)
        shipped = np.clip(u - eta_shipped * (row_energy - target + gamma * u), -M, M)
        # geodesic: the same step in value = 2u
        state = (jnp.asarray(2 * u, jnp.float32),)
        got, finite = mc.update_side((group,), state, samples, target, gamma, M, eta_geodesic)
        assert bool(finite)
        np.testing.assert_allclose(np.asarray(got[0], np.float64) / 2, shipped, rtol=2e-6, atol=2e-6)
        u = shipped


# =============================================================================
# CARRIED OVER from the peer session's suite (`tests/test_construct_metric.py`
# in `constructor_integration_on_current.patch`), compared by BEHAVIOUR and
# adapted to this patch's interface. Only cases this file did not already
# cover are here; what was skipped as already covered is listed in the merged
# receipt. Nothing below changes the constructor's design.
# =============================================================================

@pytest.mark.parametrize("kwargs", [
    dict(metric_updates=12),
    dict(metric_probes=2),
    dict(metric_safety=0.9),
    dict(metric_mu=1.0),
    dict(metric_box=2.0),
    dict(metric_shift=20),
    dict(metric_columns=[(0, 4), (4, 8)]),
    dict(metric_columns=("full", 2)),
    dict(metric_initial={0: None}),
])
def test_a_construct_only_option_is_refused_not_silently_ignored(kwargs):
    """Peer `test_construct_arguments_require_the_construct_mode` /
    `test_unknown_metric_option_raises`, on the settings rather than the class
    declaration: every construction setting offered off the construct path
    RAISES. It is never accepted and quietly dropped."""
    prob, _data, _K, _n = small_family()
    with pytest.raises(ValueError, match="require equilibrate='construct'"):
        prob.compile(max_blocks=2, tol=1e-3, **kwargs)


@pytest.mark.parametrize("kwargs", [dict(metric_seed=5), dict(metric_domain="restrict")])
def test_metric_seed_and_domain_outside_the_construct_path_are_refused(kwargs):
    """Peer `test_metric_seed_and_domain_outside_the_construct_path_are_refused`,
    carried whole: the same contract for the two settings whose default is not
    `None`.

    `metric_seed` defaults to `0` and `metric_domain` to `"reject"`, so neither
    can be detected by the `value is not None` test that catches the other
    seven. Before the repair carried with this case they were accepted and
    dropped on every non-construct compile: `prob.compile(metric_seed=5)`
    returned a compiled family and the caller believed they had seeded a
    construction that never ran. Silent acceptance is exactly what the case
    above refuses for the other settings, so it is refused here too.
    """
    prob, _data, _K, _n = small_family()
    with pytest.raises(ValueError, match="require equilibrate='construct'"):
        prob.compile(max_blocks=2, tol=1e-3, **kwargs)


def test_an_unknown_metric_argument_is_refused_by_the_signature():
    """The peer's `metric_options` dict can carry a typo; this interface spells
    each setting as a keyword, so a misspelling is refused by Python itself
    rather than read out of a dict and ignored. Asserted, not assumed."""
    prob, _data, _K, _n = small_family()
    with pytest.raises(TypeError):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, metric_iters=12)


def test_a_constructed_metric_requires_relative_stopping():
    """Peer `test_construct_requires_relative_stopping`. A constructed block
    metric changes the norm the iterates live in, so the metric-free relative
    stopping rule is required, exactly as for a hand-supplied one."""
    prob, _data, _K, _n = small_family()
    with pytest.raises(ValueError, match="residual_norm='relative'"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                     residual_norm="iterate", metric_classes={0: ("full", "term")},
                     metric_updates=12)


def test_the_constructed_metric_solves_the_same_problem():
    """Peer `test_constructed_metric_solves_the_same_problem`, named by them as
    one this suite might miss, and it did: the existing full-tier test only
    checked that the returned point is finite. A metric is a change of
    coordinates, so the solution must not move."""
    prob, data, _K, _n = small_family()
    plain = prob.compile(max_blocks=2000, tol=1e-5)
    built = prob.compile(equilibrate="construct", max_blocks=2000, tol=1e-5,
                         metric_classes={0: ("full", "term")}, metric_updates=12)
    a, b = plain.solve_instance(data), built.solve_instance(data)
    assert str(a.status) == str(b.status) == "solved"
    assert float(a.residual) <= 1e-5 and float(b.residual) <= 1e-5
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("cls", ["diagonal", "full"])
def test_the_effective_metric_is_the_declared_shape_times_the_rescale_factor(cls):
    """Peer `test_constructed_metric_is_recorded_and_applied`. The recorded
    spectral rescale factor is the one actually applied -- the dual step the
    loop receives IS that factor against the stored unit shape -- the effective
    condition number is inside the class's own limit, and a DIAGONAL
    declaration rotates nothing (no Q is stored)."""
    prob, _data, _K, _n = small_family()
    comp = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                        metric_classes={0: (cls, "term")}, metric_updates=12)
    shape = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    factor = comp.meta["metric_construction"]["normalization"]["factor"]
    record = comp.meta["metric_construction"]["effective_metric"][0]
    assert record["declared_class"] == cls
    assert record["dual_step_min"] == pytest.approx(factor, rel=1e-12)
    assert record["dual_step_max"] == pytest.approx(factor, rel=1e-12)
    assert shape["kind"] == cls
    assert shape["cond"] <= shape["cond_limit"]
    assert shape["cond_limit"] == (bm.MAX_FULL_COND if cls == "full" else bm.MAX_DIAGONAL_COND)
    assert (shape["Q"] is None) is (cls == "diagonal")   # a diagonal shape rotates nothing
    # factor = safety / measured, and the factor recorded is the factor used
    normalization = comp.meta["metric_construction"]["normalization"]
    assert factor == pytest.approx(0.9 / normalization["measured_before"], rel=1e-9)


def test_metric_sharing_does_not_merge_proximal_blocks_or_reach_the_certificate():
    """Peer `test_sharing_a_metric_does_not_merge_proximal_blocks`, named by
    them. The sharing-group half was already covered; the certificate half was
    not: a metric is a solver-side change of coordinates and the public
    certificate must expose exactly the same quantities as an undeclared
    compile."""
    prob, data, _K, n = small_family()
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("full", "term")}, metric_updates=12)
    term = comp.meta["block_metrics"]["terms"][0]
    assert term["k"] == 3 and term["mathematical_blocks"] == 4
    assert len(term["sharing_groups"]) == 1        # one shape, still four blocks
    plain = prob.compile(max_blocks=2, tol=1e-3)
    x, y = jnp.ones(n, jnp.float32), jnp.full(17, 0.1, jnp.float32)
    built_cert, plain_cert = comp.certificate(x, y, data), plain.certificate(x, y, data)
    assert built_cert.keys() == plain_cert.keys()
    for key in plain_cert:
        assert np.asarray(built_cert[key]).shape == np.asarray(plain_cert[key]).shape


def test_the_restricted_shape_dominates_the_constructed_one(monkeypatch):
    """Peer `test_explicit_restrict_policy_is_recorded_and_renormalized`, the
    half this suite did not assert: `restrict_condition` only RAISES small
    eigenvalues and leaves the eigenvectors alone, so the restricted shape
    dominates the constructed one in the positive semidefinite order. That is
    exactly why the norm rises and the normalization has to be recomputed."""
    prob, _data, _K, _n = small_family()
    inside = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                          metric_classes={0: ("full", "term")}, metric_updates=12)
    built_cond = inside.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"]
    monkeypatch.setattr(bm, "MAX_FULL_COND", float(np.sqrt(built_cond)))
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("full", "term")}, metric_updates=12,
                        metric_domain="restrict")
    applied = comp.meta["metric_construction"]["restriction"]["applied"][0]
    before = np.asarray(applied["eigenvalues_before"], np.float64)
    after = np.asarray(applied["eigenvalues_after"], np.float64)
    assert np.all(after >= before - 1e-12)                 # psd dominance
    assert np.all(after[after != before] > before[after != before])
    assert float(after.max()) == pytest.approx(float(before.max()), rel=1e-12)


def test_the_identity_state_applies_as_the_identity():
    """Peer `test_apply_power_of_the_identity_state_is_the_identity`. Every
    declared representation starts at the identity metric, and applying any
    power of it must be a no-op -- the property the byte-identity gate relies
    on when nothing is declared."""
    groups = (mc._Group("shared_full", 3, 4), mc._Group("shared_diagonal", 2, 3),
              mc._Group("scalar", 5, 2), mc._Group("diagonal", 4, 1))
    state = mc.initial_state(groups)
    v = jnp.asarray(np.random.default_rng(0).normal(size=sum(g.dimension for g in groups)),
                    jnp.float32)
    for power in (1.0, 0.5, -0.5):
        np.testing.assert_allclose(np.asarray(mc.apply_power(groups, state, v, power)),
                                   np.asarray(v), rtol=1e-6, atol=1e-6)
    assert mc.state_bytes(groups) == 4 * (3 * 3 + 2 + 2 + 1 * 4)


def test_the_geodesic_constructor_refuses_bad_settings():
    """Peer `test_constructor_rejects_bad_settings`: the constructor validates
    its own budget and declarations before it is handed an operator."""
    A = jnp.eye(4, dtype=jnp.float32)
    mv, rmv = (lambda x: A @ x), (lambda y: A.T @ y)
    with pytest.raises(ValueError, match="must be a positive integer"):
        mc.make_geodesic_constructor(mv, rmv, (mc._Group("diagonal", 4, 1),),
                                     (mc._Group("diagonal", 4, 1),), iterations=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        mc.make_geodesic_constructor(mv, rmv, (mc._Group("diagonal", 4, 1),),
                                     (mc._Group("diagonal", 4, 1),), probes=0)
    with pytest.raises(ValueError, match="nonempty row and column"):
        mc.make_geodesic_constructor(mv, rmv, (), (mc._Group("diagonal", 4, 1),))
    with pytest.raises(ValueError, match="positive finite mu"):
        mc.make_geodesic_constructor(mv, rmv, (mc._Group("diagonal", 4, 1),),
                                     (mc._Group("diagonal", 4, 1),), mu=0.0)


def test_declared_column_spans_must_not_overlap():
    """Peer `test_primal_tie_spans_must_be_disjoint`. This suite covered spans
    that fail to cover the primal space; overlapping spans are a different
    malformed declaration and are refused with their own reason."""
    prob, _data, _K, n = small_family()
    with pytest.raises(ValueError, match="without overlap"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                     metric_classes={0: ("full", "term")},
                     metric_columns=[(0, 5), (3, n)], metric_updates=12)
