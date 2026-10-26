"""A shared k x k metric routed to the quadratic blocks (SumSquares, Quadratic, Zero).

Every reference below is float64 numpy on the host, on k x k objects and the
dual block variable only. No test forms, densifies or reads an entry of the
stacked operator: the constructed metric is built through matvec/rmatvec by
the constructor, and the weighted prox is checked against an independent
solve of (H + Sigma) z = Sigma u - b on the block variable.
"""
import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import Term
from ocs.prox import block_metric as bm
from ocs.solver import metric_construct as mc
import ocs.model.compile as cm


# ------------------------------------------------------------------ helpers
def rotation(seed, k=3):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(k, k)))[0]


def spd(seed, k=3, eigenvalues=None):
    V = rotation(seed, k)
    e = np.asarray(eigenvalues if eigenvalues is not None else np.linspace(0.3, 1.0, k), np.float64)
    P = (V * e) @ V.T
    return 0.5 * (P + P.T)


def psd_float32(seed, k, rank=None):
    """A k x k float32 matrix that is EXACTLY symmetric positive semidefinite of the given rank.

    P = L L^T with L dyadic (multiples of 1/8 in [-1, 1], the last k - rank
    columns zero), so every entry of L L^T is a multiple of 1/64 of magnitude at
    most k and is exact in float32, then divided by a power of two so that
    ||P||_2 lies in (1/2, 1]. Rounding a float64 PSD matrix to float32 can leave
    a tiny negative eigenvalue that ocs.Quadratic rightly refuses. This does not.
    """
    rank = k if rank is None else rank
    for attempt in range(64):
        rng = np.random.default_rng([seed, attempt])
        L = rng.integers(-8, 9, size=(k, k)).astype(np.float64) / 8.0
        L[:, rank:] = 0.0
        P = L @ L.T
        top = float(np.linalg.eigvalsh(P).max())
        if top <= 0 or np.linalg.matrix_rank(P) != rank:
            continue
        P = P / 2.0 ** np.ceil(np.log2(top))
        P32 = P.astype(np.float32)
        assert np.array_equal(P32.astype(np.float64), P) and np.array_equal(P32, P32.T)
        return P32
    raise AssertionError("no dyadic PSD matrix of the requested rank in 64 draws")


def digest(a):
    a = np.asarray(jax.device_get(a))
    return a.dtype.str, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def endpoints(comp, data):
    sol = comp.solve_instance(data)
    cert = comp.certificate(sol.x, sol.y, data)
    return (digest(sol.x), digest(sol.y), int(sol.iters), str(sol.status),
            tuple(sorted((k, digest(v)) for k, v in cert.items())))


def hessian_of(block, k):
    """(H, b) of f(z) = 1/2 z^T H z + b^T z, float64, from the block's own parameters."""
    from ocs.prox import base
    a = 1.0
    while type(block) is base._Scaled:
        a, block = a * block.c, block.base
    if type(block) is ocs.SumSquares:
        return a * np.eye(k), np.zeros(k)
    if type(block) is ocs.Zero:
        return np.zeros((k, k)), np.zeros(k)
    P = np.asarray(jax.device_get(block.P), np.float64)
    q = np.asarray(jax.device_get(block.q), np.float64)
    return a * P, a * q


def reference_pair(v, Sigma, H, b):
    """float64: z solving (H + Sigma) z = v - b (so u = Sigma^{-1} v), y = v - Sigma z."""
    z = np.linalg.solve(H + Sigma, v - b)
    return z, v - Sigma @ z


def close(got, ref, rtol, atol):
    """Norm-wise agreement: the error scale of a block solve is set by its largest component."""
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    assert np.linalg.norm(got - ref) <= rtol * np.linalg.norm(ref) + atol, (got, ref)


def kkt_defect(z32, Sigma, u, H, b):
    """||(H + Sigma) z32 - (Sigma u - b)|| / (||Sigma u|| + ||b||), float64 on the float32 point."""
    rhs = Sigma @ u - b
    return float(np.linalg.norm((H + Sigma) @ np.asarray(z32, np.float64) - rhs)
                 / (np.linalg.norm(Sigma @ u) + np.linalg.norm(b)))


# ------------------------------------------------------------- the fixture
N = 8
ROWS = {"sumsquares": (0, 12), "quadratic": (12, 15), "zero": (15, 21), "nonneg": (21, 26)}
M = 26
P_QUAD = psd_float32(21, 3, rank=2)             # exactly PSD in float32, singular, ||P||_2 in (1/2, 1]
P_QUAD64 = np.asarray(P_QUAD, np.float64)
P_QUAD_NORM = float(np.linalg.eigvalsh(P_QUAD64).max())


def quadratic_family(seed=0):
    """SumSquares (12 rows, grouped by 3), Quadratic (3), Zero (6), nonneg (5).

    K1 is a Gaussian with row scales inside every block of 3, mixed by a
    fixed 3 x 3 rotation on every block, so the construction energy's optimum
    over full shapes is measurably rotated and anisotropic (asserted by
    test_the_fixture_separates_the_full_class_from_the_diagonal_one).
    """
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(12, N)) * np.tile(np.array([0.25, 1.0, 4.0]), 4)[:, None]
    R = rotation(3)
    K1 = np.kron(np.eye(4), R) @ G
    K2 = rng.normal(size=(3, N))
    K3 = rng.normal(size=(6, N))
    K4 = rng.normal(size=(5, N))
    x = ocs.Variable(N)
    p1, p4 = ocs.Parameter("l1"), ocs.Parameter("l4")
    prob = ocs.minimize([
        Term(ocs.Identity(N), ocs.SumSquares().scaled(0.5)),
        ocs.least_squares(ocs.Dense(K1) @ x + p1),
        Term(op=ocs.Dense(K2), block=ocs.Quadratic(P_QUAD),
             offset_fn=lambda d: d["l2"]),
        Term(op=ocs.Dense(K3), block=ocs.Zero(), offset_fn=lambda d: d["l3"]),
        ocs.nonneg(ocs.Dense(K4) @ x + p4)])
    data = {name: jnp.asarray(rng.normal(size=size), jnp.float32)
            for name, size in (("l1", 12), ("l2", 3), ("l3", 6), ("l4", 5))}
    return prob, data, np.vstack([K1, K2, K3, K4])


_CAPTURED = []


def _capturing_make_solver(matvec, rmatvec, tau, sigma, *args, **kwargs):
    _CAPTURED.append((np.asarray(jax.device_get(tau)), np.asarray(jax.device_get(sigma))))
    return _capturing_make_solver.wrapped(matvec, rmatvec, tau, sigma, *args, **kwargs)


@pytest.fixture(autouse=True)
def capture_applied_steps(monkeypatch):
    _CAPTURED.clear()
    _capturing_make_solver.wrapped = cm.make_solver
    monkeypatch.setattr(cm, "make_solver", _capturing_make_solver)
    yield


def applied_sigma(comp):
    return np.ascontiguousarray(np.broadcast_to(np.atleast_1d(_CAPTURED[-1][1]), (comp.m,)))


def applied_tau(comp):
    return np.ascontiguousarray(np.broadcast_to(np.atleast_1d(_CAPTURED[-1][0]), (comp.n,)))


def shaped(comp, index):
    """The block_metrics record of dual term ``index`` (the list holds shaped terms only)."""
    return next(t for t in comp.meta["block_metrics"]["terms"] if t["term"] == index)


def construct(prob, **kwargs):
    return prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3, metric_updates=12,
                        **kwargs)


# ============================================ 1. the independent k x k solve
CASES = [("sumsquares", None), ("sumsquares", "rot"), ("scaled_sumsquares", None),
         ("scaled_sumsquares", "rot"), ("quadratic_spd", None), ("quadratic_spd", "rot"),
         ("quadratic_singular", None), ("quadratic_singular", "rot"), ("zero", None), ("zero", "rot")]


def _block(case, k):
    if case == "sumsquares":
        return ocs.SumSquares()
    if case == "scaled_sumsquares":
        return ocs.SumSquares().scaled(0.5)
    if case == "quadratic_spd":
        return ocs.Quadratic(np.asarray(spd(11, k), np.float32), np.asarray(np.linspace(-1, 1, k), np.float32))
    if case == "quadratic_singular":
        return ocs.Quadratic(psd_float32(12, k, rank=k - 1), np.asarray(np.linspace(1, -1, k), np.float32))
    return ocs.Zero()


@pytest.mark.parametrize("case, Q", CASES)
@pytest.mark.parametrize("k", [3, 4])
def test_weighted_quadratic_prox_matches_the_independent_float64_solve(case, Q, k):
    if case.startswith("quadratic") and k != 3 and Q is None:
        pytest.skip("the diagonal-shape Quadratic case is covered at k = 3")
    rng = np.random.default_rng(k * 10 + len(case))
    w = np.array([0.5, 3.0, 40.0, 7.0][:k])
    metric = bm.BlockMetric(w, None if Q is None else rotation(1, k))
    block = _block(case, k)
    shape = bm.QuadraticShape(metric, block, k)
    assert isinstance(shape, bm.BlockMetric) and shape.k == k
    H, b = hessian_of(block, k)
    rows = 50
    c = np.exp(rng.uniform(-1, 1, size=(rows, 1)))
    if Q is None and not case.startswith("quadratic"):
        c = c * np.exp(rng.uniform(-1, 1, size=(rows, k)))          # steps may vary inside a block
    v = rng.normal(size=(rows, k)) * np.exp(rng.uniform(-1, 2, size=(rows, 1)))
    v32, c32 = np.asarray(v, np.float32), np.asarray(c, np.float32)
    v, c = np.asarray(v32, np.float64), np.asarray(c32, np.float64)
    y32, z32 = bm.quadratic_moreau_rows(jnp.asarray(v32), jnp.asarray(c32), shape, return_primal=True)
    y32, z32 = np.asarray(y32, np.float64), np.asarray(z32, np.float64)
    S = metric.matrix()
    u_rows = []
    for i in range(rows):
        Sigma = np.diag(c[i] * w) if Q is None else c[i, 0] * S
        z_ref, y_ref = reference_pair(v[i], Sigma, H, b)
        close(z32[i], z_ref, rtol=5e-5, atol=5e-6)
        close(y32[i], y_ref, rtol=5e-5, atol=5e-6)
        u = np.linalg.solve(Sigma, v[i])
        assert kkt_defect(z32[i], Sigma, u, H, b) < bm.PROX_KKT_BAR
        u_rows.append(u)
        if case == "zero":
            assert np.all(y32[i] == 0.0)
            np.testing.assert_allclose(z32[i], u, rtol=1e-5, atol=1e-7)
    # the primal prox entry point, from u rather than v, against the same reference
    u32 = np.asarray(np.stack(u_rows), np.float32)
    zp = np.asarray(bm.prox_quadratic_rows(jnp.asarray(u32), jnp.asarray(c32), shape), np.float64)
    for i in range(rows):
        Sigma = np.diag(c[i] * w) if Q is None else c[i, 0] * S
        z_ref = np.linalg.solve(H + Sigma, Sigma @ np.asarray(u32[i], np.float64) - b)
        close(zp[i], z_ref, rtol=1e-4, atol=1e-5)
    # the library's own float32 matrix prox (an independent implementation) agrees
    if case != "zero":
        for i in (0, 7, 23):
            Sigma = np.diag(c[i] * w) if Q is None else c[i, 0] * S
            lib = np.asarray(block.prox_matrix(jnp.asarray(u32[i]), jnp.asarray(Sigma, jnp.float32)), np.float64)
            close(zp[i], lib, rtol=2e-3, atol=2e-4)


def test_diagonal_sumsquares_shape_is_the_native_prox_with_the_expanded_step():
    rng = np.random.default_rng(5)
    shape = bm.QuadraticShape(bm.BlockMetric([2.0, 0.5, 8.0]), ocs.SumSquares(), 3)
    v = jnp.asarray(rng.normal(size=(20, 3)) * 3, jnp.float32)
    c = jnp.asarray(np.exp(rng.uniform(-1, 1, size=(20, 3))), jnp.float32)
    d = c * shape.w
    native = v - d * ocs.SumSquares().prox(v / d, 1.0 / d)
    np.testing.assert_allclose(np.asarray(bm.quadratic_moreau_rows(v, c, shape)), np.asarray(native),
                               rtol=1e-6, atol=1e-7)


def test_metric_moreau_dispatches_a_quadratic_shape_and_flags_bad_steps():
    rng = np.random.default_rng(6)
    block = ocs.SumSquares()
    k, ranges = bm.normalize_term_metrics(bm.BlockMetric([1.0, 4.0, 20.0], rotation(5)), 0, block, 12)
    assert k == 3 and isinstance(ranges[0][2], bm.QuadraticShape)
    v = rng.normal(size=12)
    sigma = np.repeat([0.01, 0.02, 0.005, 0.015], 3)
    got = np.asarray(bm.metric_moreau(block, jnp.asarray(v, "f"), jnp.asarray(sigma, "f"), ranges, 3))
    S = ranges[0][2].matrix()
    for g in range(4):
        _z, y = reference_pair(v[3 * g:3 * g + 3], sigma[3 * g] * S, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(got[3 * g:3 * g + 3], y, rtol=5e-5, atol=5e-6)
    varying = sigma.copy()
    varying[1] *= 2                                                   # not uniform in block 0
    bad = np.asarray(bm.metric_moreau(block, jnp.asarray(v, "f"), jnp.asarray(varying, "f"), ranges, 3))
    assert np.all(np.isnan(bad))


# ================================================== 2. the fixture separates
def test_the_fixture_separates_the_full_class_from_the_diagonal_one():
    prob, _data, _K = quadratic_family()
    comp = construct(prob, metric_classes={0: ("full", "term", 3)})
    group = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    Q = np.asarray(group["Q"])
    aligned = Q * np.sign(np.diag(Q))[None, :]
    assert np.max(np.abs(aligned - np.eye(3))) > 0.2
    assert group["cond"] > 1.5


# ================================= 3. constructed, applied, visible (done-when)
def test_full_shape_on_sumsquares_is_constructed_applied_and_visible():
    prob, data, _K = quadratic_family()
    comp = construct(prob, metric_classes={0: ("full", "term", 3)})
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic" and report["constructor_status"] == 0
    declared = report["declared"][0]
    assert declared["mathematical_block_size_k"] == 1 and declared["mathematical_blocks"] == 12
    assert declared["metric_block_size_k"] == 3 and declared["metric_blocks"] == 4
    assert declared["metric_block_size_source"] == "declared"
    assert declared["quadratic_block"] == "sumsquares"
    group = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert group["kind"] == "full" and group["k"] == 3 and group["groups"] == [0, 4]
    assert group["cond"] <= bm.MAX_FULL_COND
    assert group["quadratic"]["kind"] == "sumsquares" and group["quadratic"]["hessian"] == "a I"
    assert group["quadratic"]["stored_device_arrays"] == []
    assert comp.meta["block_metrics"]["terms"][0]["prox"].startswith("closed form in the eigenbasis of S")
    term = comp.meta["effective_metric"]["terms"][0]
    assert term["mathematical_block"] == {"size_k": 1, "count": 12, "coupled": False}
    assert term["metric_block_size_k"] == 3
    assert term["metric_sharing_groups"][0]["blocks"] == [0, 4]
    assert term["metric_sharing_groups"][0]["quadratic"]["kind"] == "sumsquares"
    assert term["distinct_dual_steps"] == 1
    factor = report["normalization"]["factor"]
    effective = report["effective_metric"][0]
    assert effective["dual_step_min"] == pytest.approx(factor, rel=1e-12)
    assert effective["dual_step_max"] == pytest.approx(factor, rel=1e-12)
    assert "quadratic block's closed form" in effective["effective"]
    assert comp.meta["scaling_classes"][0] == "block"
    assert comp.meta["eval_kinds"][0] == "closed_form"
    measured = comp.meta["block_metrics"]["measured_scaled_norm"]
    assert 0 < measured < 1 and measured == pytest.approx(0.9, rel=2e-3)
    sigma = applied_sigma(comp)
    np.testing.assert_array_equal(sigma[0:12], np.float32(factor))
    sol = comp.solve_instance(data)
    assert np.all(np.isfinite(np.asarray(sol.x)))


def test_full_shape_on_quadratic_is_constructed_applied_and_visible():
    prob, _data, _K = quadratic_family()
    comp = construct(prob, metric_classes={1: ("full", "term")})
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic"
    declared = report["declared"][1]
    assert declared["mathematical_block_size_k"] == 1 and declared["metric_block_size_k"] == 3
    assert declared["metric_block_size_source"] == "the quadratic's dimension"
    group = shaped(comp, 1)["sharing_groups"][0]
    assert group["kind"] == "full" and group["k"] == 3 and group["groups"] == [0, 1]
    quad = group["quadratic"]
    assert quad["kind"] == "quadratic" and quad["hessian"] == "a P, dense PSD"
    assert quad["pencil_eigenvalues_min"] >= 0 and quad["pencil_eigenvalues_max"] > 0
    assert quad["stored_device_arrays"] == ["G (k,k)", "lam (k,)", "b (k,)"]
    assert quad["hessian_to_metric_ratio"] is not None
    assert quad["hessian_to_metric_ratio"] <= quad["hessian_to_metric_ratio_limit"] == bm.MAX_QUADRATIC_HESSIAN_RATIO
    assert comp.meta["block_metrics"]["numerical_domain"]["max_quadratic_hessian_ratio"] == bm.MAX_QUADRATIC_HESSIAN_RATIO
    assert comp.meta["eval_kinds"][1] == "linear_solve" and comp.meta["scaling_classes"][1] == "block"
    assert comp.meta["effective_metric"]["terms"][1]["mathematical_block"] == {"size_k": 1, "count": 3, "coupled": False}
    np.testing.assert_array_equal(applied_sigma(comp)[12:15], np.float32(report["normalization"]["factor"]))
    # the same declaration spelled with its dimension is accepted, any other size is not
    construct(prob, metric_classes={1: ("full", "term", 3)})


def test_full_shape_on_zero_is_constructed_reported_and_inert_in_the_prox():
    prob, _data, _K = quadratic_family()
    comp = construct(prob, metric_classes={2: ("full", "term", 3)})
    group = shaped(comp, 2)["sharing_groups"][0]
    assert group["kind"] == "full" and group["groups"] == [0, 2]
    assert group["quadratic"]["kind"] == "zero" and group["quadratic"]["hessian"] == "0"
    assert comp.meta["eval_kinds"][2] == "closed_form"
    note = comp.meta["metric_construction"]["effective_metric"][2]["effective"]
    assert "never enters a dual prox" in note
    assert "changes the constructed metric of every other term" in note


def test_operator_products_are_the_same_budget_as_a_norm_block_full_shape():
    """The factorization is host k x k algebra: it charges no operator product."""
    prob, _data, _K = quadratic_family()
    quadratic = construct(prob, metric_classes={1: ("full", "term")}, metric_probes=2)
    sumsquares = construct(prob, metric_classes={0: ("full", "term", 3)}, metric_probes=2)
    for comp in (quadratic, sumsquares):
        products = comp.meta["metric_construction"]["operator_products"]
        assert products == {"gradient": 2 * 12 * 2, "normalization": 30, "total": 78,
                            "note": products["note"]}


# ==================================== 4. the loop applies the recorded shape
def test_one_native_standard_step_matches_the_float64_formulas():
    prob, data, K = quadratic_family()
    rng = np.random.default_rng(1)
    full3 = bm.BlockMetric([1.0, 4.0, 20.0], rotation(5))
    fullq = bm.BlockMetric([1.0, 1.5, 2.5], rotation(8))
    tau = np.full(N, 0.01)
    sigma = np.full(M, 0.01)
    sigma[0:12] = np.repeat([0.01, 0.02, 0.005, 0.015], 3)
    sigma[12:15] = 0.1
    # This test measures the ITERATION (one native standard step), so it asks for the
    # residual route explicitly, as test_the_constructed_metric_solves_the_same_problem
    # does. Under the default objective-gap acceptance this family has no dual-witness
    # contract (the singular Quadratic block declares none, see
    # meta["objective_bound_missing_contracts"]), and since 0.3.3 (CHANGES.md, v23:
    # "Unsupported objective-gap requests return unverified / unsupported with zero
    # iterations") such a request is refused BEFORE optimizer work: solve_instance
    # returns the warm start untouched with iters == 0. 0.3.2 fell through to the
    # residual loop and labelled the result "unverified". The expected answer below is
    # the float64 formula, unchanged; tests/test_given_routing_contract.py asserts the
    # zero-iteration refusal of the default route on this same family.
    comp = prob.compile(equilibrate=("given", tau.astype("f"), sigma.astype("f"), 0.5),
                        block_metrics={0: full3, 1: fullq}, pdhg_variant="standard", block=1,
                        max_blocks=1, tol=1e-12, accuracy="residual")
    assert comp.meta["scaling_classes"] == ["block", "block", "matrix", "diagonal"]
    assert comp.meta["eval_kinds"][:2] == ["closed_form", "linear_solve"]
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1
    x0, y0 = rng.normal(size=N), rng.normal(size=M) * 0.1
    y0[21:] = -abs(y0[21:])
    sol = comp.solve_instance(data, warm_start=(jnp.asarray(x0, "f"), jnp.asarray(y0, "f")))
    assert int(sol.iters) == 1 and sol.accuracy_criterion == "fixed_point"
    l = np.concatenate([np.asarray(data[k]) for k in ("l1", "l2", "l3", "l4")])
    u = K @ x0 + l
    y = np.empty(M)
    for g in range(4):
        Sig = sigma[3 * g] * full3.matrix()
        v = y0[3 * g:3 * g + 3] + Sig @ u[3 * g:3 * g + 3]
        y[3 * g:3 * g + 3] = reference_pair(v, Sig, np.eye(3), np.zeros(3))[1]
    Sig = sigma[12] * fullq.matrix()
    v = y0[12:15] + Sig @ u[12:15]
    y[12:15] = reference_pair(v, Sig, P_QUAD, np.zeros(3))[1]
    y[15:21] = 0.0
    y[21:] = np.minimum(y0[21:] + sigma[21:] * u[21:], 0)
    x = (x0 - tau * (K.T @ y)) / (1 + 0.5 * tau)
    # solve_instance returns the REPAIRED dual (ocs.solver.stopping.make_dual_repair): the rows
    # of every smooth block are overwritten with grad f(K x1 + l) at the returned x1. So the
    # SumSquares, Quadratic and Zero rows of sol.y are not the raw PDHG iterate, and the raw
    # shaped dual step is pinned through x1, which the loop computes from it. The nonneg rows
    # are the raw iterate.
    u1 = K @ x + l
    reported = y.copy()
    reported[0:12] = u1[0:12]                       # grad of 1/2 ||u||^2
    reported[12:15] = P_QUAD64 @ u1[12:15]          # grad of 1/2 u^T P u, q = 0
    reported[15:21] = 0.0                           # grad of 0
    np.testing.assert_allclose(sol.x, x, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(sol.y, reported, rtol=1e-4, atol=1e-6)
    # negative control: the same step with the shape left off (the native prox on the
    # SumSquares rows and the native diagonal solve on the Quadratic rows) lands elsewhere.
    y_flat = y.copy()
    for g in range(4):
        rows_g = slice(3 * g, 3 * g + 3)
        y_flat[rows_g] = (y0[rows_g] + sigma[rows_g] * u[rows_g]) / (1 + sigma[rows_g])
    v_flat = y0[12:15] + sigma[12:15] * u[12:15]
    y_flat[12:15] = v_flat - sigma[12] * np.linalg.solve(P_QUAD64 + sigma[12] * np.eye(3), v_flat)
    x_flat = (x0 - tau * (K.T @ y_flat)) / (1 + 0.5 * tau)
    assert np.max(np.abs(x_flat - x)) > 1e-3 * np.max(np.abs(x))
    assert not np.allclose(np.asarray(sol.x), x_flat, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("declaration", [{0: ("full", "term", 3)}, {1: ("full", "term")},
                                         {0: ("full", "term", 6), 1: ("full", "term"), 2: ("full", "term", 2)}])
def test_the_constructed_metric_solves_the_same_problem(declaration):
    prob, data, _K = quadratic_family()
    # metric-path test: this family has no exact dual witness under the default acceptance (g = 0 coupling with a cost, or a Zero block), so the explicit residual policy is requested here. The default objective-gap acceptance is tested in test_validate.py and test_stopping_contract.py.
    plain = prob.compile(max_blocks=2000, tol=1e-5, accuracy="residual")
    built = prob.compile(equilibrate="construct", max_blocks=2000, tol=1e-5, accuracy="residual",
                         metric_classes=declaration, metric_updates=12)
    a, b = plain.solve_instance(data), built.solve_instance(data)
    assert str(a.status) == str(b.status) == "solved"
    assert float(a.residual) <= 1e-5 and float(b.residual) <= 1e-5
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=5e-3, atol=5e-3)


# ====================================================== 5. Zero, precisely
def test_zero_shape_is_inert_in_the_prox_and_changes_every_other_terms_metric():
    """The dual rows of a Zero term are exactly zero under every metric, so the
    shape never enters a dual prox. It DOES enter the joint construction and the
    normalization, so the constructed tau and the sigma of the other terms move."""
    prob, data, _K = quadratic_family()
    without = construct(prob, metric_classes={0: ("full", "term", 3)})
    tau_a, sigma_a = applied_tau(without), applied_sigma(without)
    with_zero = construct(prob, metric_classes={0: ("full", "term", 3), 2: ("full", "term", 3)})
    tau_b, sigma_b = applied_tau(with_zero), applied_sigma(with_zero)
    assert without.meta["metric_construction"]["regime"] == with_zero.meta["metric_construction"]["regime"] == "geodesic"
    assert not np.array_equal(tau_a, tau_b)
    for start, end in (ROWS["sumsquares"], ROWS["quadratic"], ROWS["nonneg"]):
        assert not np.array_equal(sigma_a[start:end], sigma_b[start:end])
    a, b = without.solve_instance(data), with_zero.solve_instance(data)
    lo, hi = ROWS["zero"]
    assert np.all(np.asarray(b.y)[lo:hi] == 0.0)                     # the shaped Zero term: exactly zero
    assert np.max(np.abs(np.asarray(a.y)[lo:hi])) <= 1e-5           # the native Zero term: v - s (v / s)
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=2e-2, atol=2e-2)


# ============================================== 6. default parity, canonical
def test_default_declaration_is_byte_identical_to_shipped_sgd():
    prob, data, _K = quadratic_family()
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    shipped_steps = (digest(_CAPTURED[-1][0]), digest(_CAPTURED[-1][1]))
    built = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3)
    built_steps = (digest(_CAPTURED[-1][0]), digest(_CAPTURED[-1][1]))
    assert built_steps == shipped_steps
    report = built.meta["metric_construction"]
    assert report["regime"] == "smfe" and report["natural_declaration"] is True
    assert report["bit_identical_to_sgd"] is True
    assert "block_metrics" not in built.meta
    assert built.meta["scaling_classes"] == shipped.meta["scaling_classes"] == ["matrix", "matrix", "matrix", "diagonal"]
    assert built.meta["eval_kinds"] == shipped.meta["eval_kinds"]
    for d in report["declared"]:
        assert d["metric_block_size_k"] == d["mathematical_block_size_k"] == 1
        assert d["metric_block_size_source"] == "mathematical block"
    assert report["restriction_cache_identity"][1] == tuple(
        (i, "diagonal", "per_block", 1, size) for i, size in enumerate((12, 3, 6, 5)))
    assert endpoints(built, data) == endpoints(shipped, data)


def test_scalar_class_on_a_separable_quadratic_keeps_the_free_diagonal():
    """The scalar class is resolved against the mathematical block (one coordinate), so
    ('scalar', 'term') on SumSquares is the free diagonal it was before this change."""
    prob, data, _K = quadratic_family()
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    for spelling in (("scalar", "term"), ("scalar", "per_block")):
        comp = construct(prob, metric_classes={0: spelling})
        report = comp.meta["metric_construction"]
        assert report["regime"] == "smfe"
        assert (report["declared"][0]["declared_class"], report["declared"][0]["metric_sharing"]) == ("diagonal", "per_block")
        assert report["declared"][0]["metric_block_size_k"] == 1
        assert comp.meta["effective_metric"]["terms"][0]["distinct_dual_steps"] == 12
        assert endpoints(comp, data) == endpoints(shipped, data)


def test_shared_diagonal_on_a_quadratic_is_built_like_the_l2_row():
    """('diagonal', 'term') on a one-block quadratic term is not canonicalized away: as for
    L2 today it is a shared diagonal shape built by the geodesic tier."""
    prob, data, _K = quadratic_family()
    comp = construct(prob, metric_classes={1: ("diagonal", "term")})
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic"
    group = shaped(comp, 1)["sharing_groups"][0]
    assert group["kind"] == "diagonal" and group["Q"] is None and group["k"] == 3
    assert group["quadratic"]["kind"] == "quadratic"
    assert comp.meta["eval_kinds"][1] == "linear_solve"
    sol = comp.solve_instance(data)
    assert np.all(np.isfinite(np.asarray(sol.x)))
    grouped = construct(prob, metric_classes={0: ("diagonal", "term", 3)})
    group = shaped(grouped, 0)["sharing_groups"][0]
    assert group["kind"] == "diagonal" and group["groups"] == [0, 4] and group["quadratic"]["kind"] == "sumsquares"


def test_identity_shape_with_uniform_steps_is_the_scalar_tier_code_path():
    prob, data, _K = quadratic_family()
    tau, sigma = np.full(N, 0.01, "f"), np.full(M, 0.01, "f")
    plain = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12)
    comp = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: bm.BlockMetric(np.ones(3))})
    assert comp.meta["block_metrics"]["canonicalized_to_scalar_tier"] == [0]
    assert comp.meta["scaling_classes"] == plain.meta["scaling_classes"]
    assert endpoints(comp, data) == endpoints(plain, data)


# ============================================ 7. the certificate is untouched
def test_certificate_is_metric_free():
    prob, data, _K = quadratic_family()
    declared = construct(prob, metric_classes={0: ("full", "term", 3), 1: ("full", "term")})
    plain = prob.compile(max_blocks=4, tol=1e-3)
    x, y = jnp.ones(N, jnp.float32), jnp.full(M, 0.1, jnp.float32)
    left, right = declared.certificate(x, y, data), plain.certificate(x, y, data)
    assert set(left) == set(right)
    for key, value in left.items():
        other = right[key]
        assert np.asarray(value).shape == np.asarray(other).shape
        assert value == other or (np.isnan(value) and np.isnan(other)), key


# ============================================================ 8. refusals
def _groupl2_family():
    rng = np.random.default_rng(2)
    x = ocs.Variable(N)
    return ocs.minimize([Term(ocs.Identity(N), ocs.SumSquares().scaled(0.5)),
                         ocs.group_l2(ocs.Dense(rng.normal(size=(12, N))) @ x + ocs.Parameter("l"),
                                      lam=0.9, group_size=3)])


def _linquad_family():
    rng = np.random.default_rng(3)
    x = ocs.Variable(N)
    return ocs.minimize([Term(ocs.Identity(N), ocs.SumSquares().scaled(0.5)),
                         Term(op=ocs.Dense(rng.normal(size=(12, N))),
                              block=ocs.SumSquares().with_linear_quadratic(0.1, 0.2),
                              offset_fn=lambda d: d["l"])])


@pytest.mark.parametrize("family, kwargs, needle", [
    ("quad", {0: ("full", "term", 5)}, "not a multiple"),
    ("quad", {1: ("full", "term", 2)}, "fixes its block size at 3 (from the quadratic's dimension)"),
    ("quad", {3: ("full", "term")}, "is neither"),
    ("quad", {3: ("full", "term")}, "only a valid proximal metric for a rotation-invariant"),
    ("quad", {3: ("diagonal", "term")}, "SHARED diagonal shape needs"),
    ("quad", {3: ("full", "term", 2)}, "is neither"),
    ("quad", {3: ("diagonal", "per_block", 2)}, "fixes its block size at 1 (from one coordinate per block)"),
    ("quad", {0: ("full", "per_block", 3)}, "must be shared"),
    ("quad", {1: ("full", "per_block")}, "exactly one metric block of 3 rows"),
    ("quad", {0: ("scalar", "term", 3)}, "applies only to a shared shape"),
    ("quad", {0: ("diagonal", "per_block", 3)}, "applies only to a shared shape"),
    ("quad", {0: ("full", "term", 1)}, "metric blocks of size 1"),
    ("quad", {0: ("full", "term", 0)}, "positive integer"),
    ("quad", {0: ("full", "term", True)}, "positive integer"),
    ("quad", {0: ("full", "term", 3, 4)}, "(class, sharing, k) triple"),
    ("linquad", {0: ("full", "term", 3)}, "is neither"),
    ("groupl2", {0: ("full", "term", 2)}, "fixes its block size at 3 (from the norm's group size)"),
])
def test_bad_declarations_are_refused_with_their_reason(family, kwargs, needle):
    prob = {"quad": lambda: quadratic_family()[0], "linquad": _linquad_family,
            "groupl2": _groupl2_family}[family]()
    with pytest.raises(ValueError, match=re.escape(needle)):
        construct(prob, metric_classes=kwargs)


def test_explicit_route_refusals_name_their_reason():
    with pytest.raises(ValueError, match="same metric block size"):
        bm.normalize_term_metrics([bm.BlockMetric([1., 2., 3.], groups=(0, 2)),
                                   bm.BlockMetric([1., 2.], groups=(2, 3))], 0, ocs.SumSquares(), 12)
    with pytest.raises(ValueError, match=re.escape("accepts only a 3 x 3 shape")):
        bm.normalize_term_metrics(bm.BlockMetric([1., 2.]), 0, ocs.Quadratic(np.eye(3, dtype=np.float32)), 3)
    with pytest.raises(ValueError, match="not a multiple"):
        bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3., 4., 5.]), 0, ocs.SumSquares(), 12)
    for block in (ocs.SumSquares().precomposed(2.0, 0.0), ocs.SumSquares().with_linear_quadratic(0.1, 0.2),
                  ocs.L1(1.0), ocs.SecondOrderCone()):
        with pytest.raises(ValueError, match="not a supported"):
            bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3.]), 0, block, 6)
    overridden = ocs.SumSquares()
    overridden.prox = lambda v, t: v
    assert bm.quadratic_structure(overridden, 6) is None
    assert bm.quadratic_structure(ocs.SumSquares().scaled(2.0).scaled(0.25), 6)[:2] == ("sumsquares", 0.5)
    assert bm.quadratic_structure(ocs.Zero(), 6)[0] == "zero"
    assert bm.quadratic_structure(ocs.GroupL2(1.0, 3), 6) is None


def test_quadratic_pencil_requires_uniform_steps_and_a_bounded_hessian_ratio():
    block = ocs.Quadratic(P_QUAD)
    k, ranges = bm.normalize_term_metrics(bm.BlockMetric([1., 2., 4.]), 0, block, 3)
    assert k == 3
    with pytest.raises(ValueError, match="uniform inside every mathematical block"):
        bm.check_static_steps(np.array([0.1, 0.2, 0.1]), 0, 3, 3, ranges, 0)
    checked = bm.check_static_steps(np.array([0.1, 0.1, 0.1]), 0, 3, 3, ranges, 0)
    shape = checked[0][2]
    assert shape.hessian_norm == pytest.approx(P_QUAD_NORM, rel=1e-12)
    assert shape.hessian_ratio == pytest.approx(P_QUAD_NORM / (0.1 * 1.0))     # w_min = 1
    too_small = P_QUAD_NORM / (bm.MAX_QUADRATIC_HESSIAN_RATIO * 1.01)
    k, ranges = bm.normalize_term_metrics(bm.BlockMetric([1., 2., 4.]), 0, block, 3)
    with pytest.raises(ValueError, match="Hessian-to-metric ratio") as err:
        bm.check_static_steps(np.full(3, too_small), 0, 3, 3, ranges, 0)
    assert "two-parameter" in str(err.value) and "Nothing is rescaled" in str(err.value)
    with pytest.raises(ValueError, match="Hessian-to-metric ratio"):
        bm.check_static_steps(np.float64(too_small), 0, 3, 3, ranges, 0)      # the scalar-sigma branch


# =================================================== 9. the condition domain
def test_out_of_domain_full_shape_is_refused_not_clipped(monkeypatch):
    prob, _data, _K = quadratic_family()
    inside = construct(prob, metric_classes={0: ("full", "term", 3)})
    built_cond = inside.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"]
    assert built_cond > 1.0
    monkeypatch.setattr(bm, "MAX_FULL_COND", float(np.sqrt(built_cond)))
    with pytest.raises(ValueError, match="Nothing is clipped for you"):
        construct(prob, metric_classes={0: ("full", "term", 3)})


def test_restrict_recomputes_and_reports_the_normalization(monkeypatch):
    prob, _data, _K = quadratic_family()
    inside = construct(prob, metric_classes={0: ("full", "term", 3)})
    built_cond = inside.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"]
    limit = float(np.sqrt(built_cond))
    monkeypatch.setattr(bm, "MAX_FULL_COND", limit)
    comp = construct(prob, metric_classes={0: ("full", "term", 3)}, metric_domain="restrict")
    restriction = comp.meta["metric_construction"]["restriction"]
    applied = restriction["applied"][0]
    assert applied["restricted"] is True and applied["eigenvalues_raised"] >= 1
    assert applied["cond_before"] == pytest.approx(built_cond, rel=1e-6)
    assert applied["cond_after"] <= limit * (1 + 1e-6)
    assert restriction["measured_after_restriction"] > restriction["measured_before_restriction"]
    assert comp.meta["metric_construction"]["operator_products"]["normalization"] == 60
    assert 0 < comp.meta["block_metrics"]["measured_scaled_norm"] < 1
    before = np.asarray(applied["eigenvalues_before"])
    after = np.asarray(applied["eigenvalues_after"])
    assert np.all(after >= before - 1e-12)
    assert float(after.max()) == pytest.approx(float(before.max()), rel=1e-12)
    shape = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert shape["cond"] <= limit * (1 + 1e-6) and shape["quadratic"]["kind"] == "sumsquares"


def test_the_constructor_emits_a_plain_shape_and_the_compiler_wraps_it():
    """The constructor holds no Hessian: it emits a BlockMetric, the compiler's
    normalize_term_metrics wraps it in a QuadraticShape once per compile."""
    prob, _data, _K = quadratic_family()
    blocks = [t.block for t in prob.terms[1:]]
    slices = [(0, 12), (12, 15), (15, 21), (21, 26)]
    decls = mc.normalize_declarations({0: ("full", "term", 3), 2: ("diagonal", "term", 2)}, blocks, slices)
    assert [(d.cls, d.sharing, d.k, d.count, d.math_k, d.quadratic) for d in decls] == [
        ("full", "term", 3, 4, 1, "sumsquares"), ("diagonal", "per_block", 1, 3, 1, "quadratic"),
        ("diagonal", "term", 2, 3, 1, "zero"), ("diagonal", "per_block", 1, 5, 1, None)]
    groups, _owners = mc.row_groups(decls)
    assert [(g.kind, g.size, g.count) for g in groups] == [
        ("shared_full", 3, 4), ("diagonal", 1, 3), ("shared_diagonal", 2, 3), ("diagonal", 1, 5)]
    shape = bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3.], rotation(4)), 0, blocks[0], 12)[1][0][2]
    assert isinstance(shape, bm.QuadraticShape) and shape.describe()["quadratic"]["kind"] == "sumsquares"
