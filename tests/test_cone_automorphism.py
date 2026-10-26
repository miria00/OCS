"""The automorphism tier for symmetric-cone blocks (ocs.prox.cone_metric).

Every reference below is host float64 numpy written from the mathematics
(Jordan algebra of the second-order cone, congruence for PSD, projected
gradient as an independent solver, a solver-free KKT certificate). None of
it imports the code under test for the reference side. Tolerances follow
tests/test_block_metric.py (rtol 5e-5, atol 5e-6 on a float32 result).
"""
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
from ocs.prox.groups import build_groups, grouped_moreau_dual
from ocs.solver import metric_construct as mc
from ocs.solver.domain_checks import _rows as domain_rows
import ocs.model.compile as compile_module

EPS32 = float(np.finfo(np.float32).eps)


# ============================================================ float64 reference
def J_of(k):
    J = -np.eye(k)
    J[0, 0] = 1.0
    return J


def ref_det(w):
    return w[0] ** 2 - w[1:] @ w[1:]


def ref_P(w):
    return 2.0 * np.outer(w, w) - ref_det(w) * J_of(w.size)


def ref_spectral(w):
    nb = np.linalg.norm(w[1:])
    u = np.zeros(w.size - 1)
    if nb > 0:
        u = w[1:] / nb
    else:
        u[0] = 1.0
    c1 = 0.5 * np.concatenate([[1.0], u])
    c2 = 0.5 * np.concatenate([[1.0], -u])
    return w[0] + nb, w[0] - nb, c1, c2


def ref_pow(w, t):
    l1, l2, c1, c2 = ref_spectral(w)
    return l1 ** t * c1 + l2 ** t * c2


def ref_fn(w, fn):
    l1, l2, c1, c2 = ref_spectral(w)
    return fn(l1) * c1 + fn(l2) * c2


def ref_proj_soc(v):
    s, z = v[0], v[1:]
    nz = np.linalg.norm(z)
    if nz <= s:
        return v.copy()
    if nz <= -s:
        return np.zeros_like(v)
    a = 0.5 * (nz + s)
    return np.concatenate([[a], a * z / nz])


def ref_weighted_proj(Sigma, v):
    """W^{-1} Pi(W v) written from the Sigma matrix alone: W = Sigma^{1/2} by eigh."""
    lam, U = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    W = (U * np.sqrt(lam)) @ U.T
    Winv = (U / np.sqrt(lam)) @ U.T
    return Winv @ ref_proj_soc(W @ v)


def ref_weighted_proj_pgd(Sigma, v, iters=20000):
    """argmin_z (z-v)^T Sigma (z-v) over the SOC by projected gradient: never sees W."""
    step = 1.0 / np.linalg.eigvalsh(Sigma).max()
    z = ref_proj_soc(v)
    for _ in range(iters):
        z = ref_proj_soc(z - step * (Sigma @ (z - v)))
    return z


def soc_defect(x):
    return max(0.0, np.linalg.norm(x[1:]) - x[0])


def ref_kkt_soc(Sigma, v, z):
    """Three residuals of optimality for min 1/2 (z-v)^T Sigma (z-v), z in C.

    -grad = Sigma (v - z) is in the normal cone N_C(z) = (-C) cap z^perp, so
    y = Sigma (z - v) lies in C (not in the polar), z lies in C, and <y, z> = 0.
    """
    y = Sigma @ (z - v)
    scale = np.linalg.norm(Sigma @ v) + 1e-300
    return {"dist_z": soc_defect(z) / scale, "dist_y": soc_defect(y) / scale,
            "complementarity": abs(y @ z) / (scale * (np.linalg.norm(z) + 1e-300))}


def ref_proj_psd(M):
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    return (U * np.maximum(lam, 0.0)) @ U.T


def spd_pow(S, t):
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    return (U * lam ** t) @ U.T


def random_interior(rng, k, ratio_lo=0.2, ratio_hi=0.9, scale=1.0):
    xbar = rng.standard_normal(k - 1)
    x0 = np.linalg.norm(xbar) / rng.uniform(ratio_lo, ratio_hi)
    return scale * np.concatenate([[x0], xbar])


def random_w(rng, k, cond_max=100.0):
    """Interior w with cond P(w) = (lam1/lam2)^2 <= cond_max and lam1 lam2 ~ 1."""
    ratio = rng.uniform(1.05, np.sqrt(cond_max))
    l2 = 1.0 / np.sqrt(ratio)
    l1 = ratio * l2
    u = rng.standard_normal(k - 1)
    u /= np.linalg.norm(u)
    return 0.5 * (l1 + l2) * np.r_[1.0, np.zeros(k - 1)] + 0.5 * (l1 - l2) * np.r_[0.0, u]


def random_spd(rng, k, cond_max=100.0):
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    ratio = rng.uniform(1.05, np.sqrt(cond_max))
    lam = np.geomspace(1.0 / np.sqrt(ratio), np.sqrt(ratio), k)
    S = (Q * lam) @ Q.T
    return 0.5 * (S + S.T)


def digest(a):
    a = np.asarray(jax.device_get(a))
    return a.dtype.str, a.shape, a.tobytes()


def endpoints(comp, data):
    sol = comp.solve_instance(data)
    cert = comp.certificate(sol.x, sol.y, data)
    return (digest(sol.x), digest(sol.y), int(sol.iters), str(sol.status),
            tuple(sorted((k, digest(v)) for k, v in cert.items())))


def captured_solver(monkeypatch, sink):
    """Record the iteration matvec and the steps the compiled solver receives."""
    original = compile_module.make_solver

    def capture(matvec, rmatvec, tau, sigma, *args, **kwargs):
        sink.append({"matvec": matvec, "tau": digest(tau), "sigma": digest(sigma)})
        return original(matvec, rmatvec, tau, sigma, *args, **kwargs)
    monkeypatch.setattr(compile_module, "make_solver", capture)


# ================================================================== families
FAMILY_COSTS = {}        # (kind, seed) -> the linear cost c of that family, for hand-written steps


def cone_family(kind, seed=0, n=8):
    """One ridge primal term with a linear cost, one cone term, one nonneg term.

    Anisotropic rows. The offsets are chosen so that a random point x_star is
    STRICTLY feasible (every cone image interior, every nonneg row positive):
    the family is then feasible and bounded with a nontrivial optimum, which a
    fixture with random offsets did not guarantee (the undeclared compile of
    the earlier socbatch and psd fixtures ran 400,000 iterations to
    'inaccurate' on Mazeppa CPU)."""
    rng = np.random.default_rng(seed)
    if kind == "soc":
        block, rows = cones.SecondOrderCone(), 4
    elif kind == "socbatch":
        block, rows = cones.SOCBatch(3, 2), 9
    elif kind == "psd":
        block, rows = cones.PSDCone(3), 9
    else:
        raise ValueError(kind)
    K1 = rng.normal(size=(rows, n)) * np.geomspace(0.3, 3.0, rows)[:, None]
    K2 = rng.normal(size=(5, n))
    x = ocs.Variable(n)
    p2 = ocs.Parameter("l2")
    cost = rng.normal(size=n)
    FAMILY_COSTS[(kind, seed)] = cost
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         Term(ocs.Dense(K1), block, lambda d: d["l1"]),
                         ocs.nonneg(ocs.Dense(K2) @ x + p2)], cost=cost)
    x_star = rng.normal(size=n)
    if kind == "soc":
        interior = random_interior(rng, rows)
    elif kind == "socbatch":
        interior = cm.unrows(("socbatch", 3, 2), np.stack([random_interior(rng, 3) for _ in range(3)]), np)
    else:
        interior = random_spd(rng, 3, cond_max=9.0).reshape(-1)
    l1 = interior - K1 @ x_star
    l2 = rng.uniform(0.5, 1.5, 5) - K2 @ x_star
    data = {"l1": jnp.asarray(l1, jnp.float32), "l2": jnp.asarray(l2, jnp.float32)}
    return prob, data, np.vstack([K1, K2]), n, block, rows


# ===================================================================== T1
@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_jordan_operations_match_float64_reference(k):
    rng = np.random.default_rng(100 + k)
    e = np.zeros(k)
    e[0] = 1.0
    v = rng.standard_normal((16, k))
    # P(e) = I bitwise in float32
    Pe = np.asarray(cm.apply_soc(jnp.asarray(e[None], jnp.float32), cm.soc_det(jnp.asarray(e[None], jnp.float32)),
                                 jnp.asarray(v, jnp.float32)))
    assert Pe.tobytes() == v.astype(np.float32).tobytes()
    for _ in range(20):
        w = random_w(rng, k)
        w32 = jnp.asarray(w[None], jnp.float32)
        # spectral functions against the float64 Jordan decomposition
        for name, fn64, fn32 in (("sqrt", np.sqrt, jnp.sqrt), ("inv", lambda t: 1 / t, lambda t: 1 / t),
                                 ("log", np.log, jnp.log)):
            got = np.asarray(cm.soc_spectral(w32, fn32))[0]
            np.testing.assert_allclose(got, ref_fn(w, fn64), rtol=1e-5, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(float(cm.soc_det(w32)[0]), ref_det(w), rtol=1e-5)
        # P(w) v against the dense float64 matrix
        Pv = np.asarray(cm.apply_soc(w32, cm.soc_det(w32), jnp.asarray(v, jnp.float32)))
        np.testing.assert_allclose(Pv, v @ ref_P(w).T, rtol=1e-4, atol=1e-5)
        # P(w^{-1/2}) P(w^{1/2}) v = v and P(w^{1/2})^2 = P(w)
        half = cm.soc_spectral(w32, jnp.sqrt)
        invhalf = cm.soc_spectral(w32, lambda t: 1 / jnp.sqrt(t))
        u = cm.apply_soc(half, cm.soc_det(half), jnp.asarray(v, jnp.float32))
        back = np.asarray(cm.apply_soc(invhalf, cm.soc_det(invhalf), u))
        np.testing.assert_allclose(back, v, rtol=1e-4, atol=1e-5)
        twice = np.asarray(cm.apply_soc(half, cm.soc_det(half), u))
        np.testing.assert_allclose(twice, Pv, rtol=1e-4, atol=1e-5)
        # the carrier caches what the host computes
        carrier = cm.ConeAutomorphism.soc(w)
        np.testing.assert_allclose(np.asarray(carrier.half)[0], ref_pow(w, 0.5), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(carrier.invhalf)[0], ref_pow(w, -0.5), rtol=1e-6)
        assert carrier.cond == pytest.approx(np.linalg.cond(ref_P(w)), rel=1e-9)


# ===================================================================== T2
@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_weighted_soc_projection_matches_transformed_euclidean_reference(k):
    """SPEC Item 2 done-when: verified against a Euclidean projection in the
    transformed coordinates on random instances, membership checked, and the
    solver-free KKT certificate (z in C, Sigma (z - v) in C, complementarity)."""
    rng = np.random.default_rng(200 + k)
    worst = {"dist_z": 0.0, "dist_y": 0.0, "complementarity": 0.0}
    for _ in range(40):
        w = random_w(rng, k)
        carrier = cm.ConeAutomorphism.soc(w, layout=("soc",))
        Sigma = ref_P(w)
        for _ in range(5):
            v = rng.standard_normal(k) * rng.uniform(0.5, 3.0)
            got = np.asarray(cm.prox_automorphism(carrier, jnp.asarray(v, jnp.float32)), np.float64)
            ref = ref_weighted_proj(Sigma, v)
            np.testing.assert_allclose(got, ref, rtol=5e-5, atol=5e-6)
            np.testing.assert_allclose(ref, ref_weighted_proj_pgd(Sigma, v), rtol=1e-7, atol=1e-8)
            for key, value in ref_kkt_soc(Sigma, v, got).items():
                worst[key] = max(worst[key], value)
    # the certificate of the float32 point, in float64, under the declared bar
    assert max(worst.values()) < bm.PROX_KKT_BAR, worst


# ===================================================================== T3
def test_weighted_psd_projection_uses_the_congruence_order_S_half_first():
    rng = np.random.default_rng(20260920)
    k = 3
    worst_kkt = 0.0
    reversed_gap = 0.0
    for _ in range(12):
        S = random_spd(rng, k)
        carrier = cm.ConeAutomorphism.psd(S, layout=("psd", k))
        V = rng.standard_normal((k, k))
        V = 0.5 * (V + V.T)
        got = np.asarray(cm.prox_automorphism(carrier, jnp.asarray(V.reshape(-1), jnp.float32)),
                         np.float64).reshape(k, k)
        Sh, Smh = spd_pow(S, 0.5), spd_pow(S, -0.5)
        ref = Smh @ ref_proj_psd(Sh @ V @ Sh) @ Smh
        np.testing.assert_allclose(got, ref, rtol=5e-5, atol=5e-6)
        # KKT under the metric P(S): Y = S (Z - V) S psd, Z psd, <Y, Z> = 0
        Y = S @ (got - V) @ S
        scale = np.linalg.norm(S @ V @ S) + 1e-300
        worst_kkt = max(worst_kkt, -np.linalg.eigvalsh(0.5 * (got + got.T)).min() / scale,
                        -np.linalg.eigvalsh(0.5 * (Y + Y.T)).min() / scale,
                        abs(np.sum(Y * got)) / (scale * np.linalg.norm(got) + 1e-300))
        wrong = Sh @ ref_proj_psd(Smh @ V @ Smh) @ Sh        # the argmin under P(S^{-1}), refuted
        reversed_gap = max(reversed_gap, np.abs(wrong - ref).max() / np.abs(ref).max())
    assert worst_kkt < bm.PROX_KKT_BAR
    assert reversed_gap > 0.1          # the reversed order is a different point; pinned


# ===================================================================== T4
def test_dual_update_is_the_weighted_moreau_identity_and_ignores_the_scalar_step():
    rng = np.random.default_rng(4)
    k = 4
    w = random_w(rng, k)
    carrier = cm.ConeAutomorphism.soc(w, layout=("soc",))
    ranges = ((0, 1, carrier, False),)
    Pw = ref_P(w)
    results = []
    V = rng.standard_normal((6, k)) * 2.0          # the same six points under every step
    for c in (0.01, 1.0, 37.0):
        for v in V:
            got = np.asarray(cm.automorphism_moreau(cones.SecondOrderCone(), jnp.asarray(v, jnp.float32),
                                                    jnp.full((k,), c, jnp.float32), ranges), np.float64)
            Sigma = c * Pw
            ref = v - Sigma @ ref_weighted_proj(Sigma, np.linalg.solve(Sigma, v))
            np.testing.assert_allclose(got, ref, rtol=5e-5, atol=5e-6)
            results.append(got)
    per_c = np.array(results).reshape(3, 6, k)
    np.testing.assert_allclose(per_c[1], per_c[0], rtol=1e-6, atol=1e-7)   # the scalar cancels
    np.testing.assert_allclose(per_c[2], per_c[0], rtol=1e-6, atol=1e-7)
    bad = jnp.asarray([1.0, 1.0, 2.0, 1.0], jnp.float32)
    out = np.asarray(cm.automorphism_moreau(cones.SecondOrderCone(), jnp.ones(k, jnp.float32), bad, ranges))
    assert np.all(np.isnan(out))


# ===================================================================== T5
@pytest.mark.parametrize("k", [3, 4, 8])
def test_cone_membership_of_the_result_is_checked_without_slack(k):
    """Exactness beyond float32 rounding is not achievable for a boundary
    outcome by any float32-returning implementation, the shipped projection
    included. The bar is derived from the arithmetic of ``apply_soc``: the two
    terms have size about (1/lam2 + 1/lam1)|u| and det|u| while the result can
    be as small as |u|/lam1, so the rounding error relative to ||z|| scales like
    (2 + lam1/lam2) eps32, and the bar is four times that. Interior outcomes
    whose margin exceeds the round-trip rounding are exactly inside; zero
    outcomes are exactly zero. No value() slack (1e-5, -1e-4) is used."""
    rng = np.random.default_rng(500 + k)
    boundary = zero = interior = 0
    for _ in range(60):
        w = random_w(rng, k)
        carrier = cm.ConeAutomorphism.soc(w, layout=("soc",))
        l1, l2, _, _ = ref_spectral(w)
        ratio = l1 / l2
        Pw = ref_P(w)
        for _ in range(8):
            v = rng.standard_normal(k) * rng.uniform(0.5, 3.0)
            z = np.asarray(cm.prox_automorphism(carrier, jnp.asarray(v, jnp.float32)), np.float64)
            ref = ref_weighted_proj(Pw, v)
            defect = soc_defect(z)
            if np.linalg.norm(ref) == 0.0:
                zero += 1
                assert np.all(z == 0.0)                # exactly zero (a signed zero is zero)
            elif (v[0] - np.linalg.norm(v[1:])) > 64 * EPS32 * np.linalg.norm(v) * (1 + ratio):
                interior += 1
                assert defect == 0.0
            else:
                boundary += 1
                assert defect <= 4 * (2 + ratio) * EPS32 * np.linalg.norm(z), (defect, ratio)
    assert boundary > 0 and zero > 0 and interior > 0


def test_psd_result_is_psd_to_rounding_and_keeps_the_skew_part_exactly():
    rng = np.random.default_rng(55)
    k = 3
    for _ in range(20):
        S = random_spd(rng, k)
        carrier = cm.ConeAutomorphism.psd(S, layout=("psd", k))
        V = rng.standard_normal((k, k)) * 2.0
        v32 = jnp.asarray(V.reshape(-1), jnp.float32)
        z = np.asarray(cm.prox_automorphism(carrier, v32), np.float64).reshape(k, k)
        assert np.linalg.eigvalsh(0.5 * (z + z.T)).min() >= -4 * EPS32 * np.linalg.norm(z) * (1 + carrier.cond)
        # the dual update keeps the skew part of v: a symmetric congruence
        # preserves the skew subspace and the projection discards it
        y = np.asarray(cm.automorphism_moreau(cones.PSDCone(k), v32, jnp.full((k * k,), 0.7, jnp.float32),
                                              ((0, 1, carrier, False),)), np.float64).reshape(k, k)
        V32 = np.asarray(v32, np.float64).reshape(k, k)
        np.testing.assert_allclose(0.5 * (y - y.T), 0.5 * (V32 - V32.T), rtol=0, atol=2 * EPS32 * np.abs(V32).max())


# ===================================================================== T6
# test_factored_projections_are_bitwise_the_shipped_prox was REPLACED in the
# Gate I-b stage (candidates/gate1b_20260921/SPEC_FROZEN.md section 1). It
# pinned the jitted factored projections bitwise to the eager shipped classes
# on every backend and failed on the RTX 5090s (KEVIN 13 of 14 processes, the
# fusion autotuner's summation order at fixture point 688, and Mazeppa 1 of 5
# at the PSD pin). Its CPU byte claims, numerical parity under derived float32
# bounds, correctness, decision consistency and the boundary stratum are in
# tests/test_factored_projection_conformance.py.


# ===================================================================== T7
@pytest.mark.parametrize("q, d", [(1, 2), (3, 2), (5, 4), (7, 1)])
def test_socbatch_rows_and_unrows_match_domain_checks_rows(q, d):
    rng = np.random.default_rng(7)
    block = cones.SOCBatch(q, d)
    v = jnp.asarray(rng.standard_normal(q * (1 + d)), jnp.float32)
    layout = cm.cone_layout(block)
    R = cm.rows(layout, v)
    assert np.asarray(R).tobytes() == np.asarray(domain_rows(block, v)).tobytes()
    assert np.asarray(cm.unrows(layout, R)).tobytes() == np.asarray(v).tobytes()
    R64 = cm.rows(layout, np.asarray(v, np.float64), np)
    assert R64.tobytes() == np.asarray(R, np.float64).tobytes()


# ===================================================================== T8
@pytest.mark.parametrize("kind, declared", [
    ("soc", "automorphism"),
    ("socbatch", ("automorphism", "per_block")),
    ("socbatch", ("automorphism", "term")),
    ("psd", "automorphism"),
])
def test_the_constructor_builds_applies_and_records_the_automorphism_metric(kind, declared):
    prob, data, K, n, block, rows = cone_family(kind)
    comp = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                        metric_classes={0: declared}, metric_updates=12)
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic" and report["constructor_status"] == 0
    struct = cm.symmetric_cone_structure(block, rows)
    sharing = "term" if declared == ("automorphism", "term") else "per_block"
    elements = 1 if sharing == "term" else struct[2]
    record = comp.meta["effective_metric"]["terms"][0]
    group = record["metric_sharing_groups"][0]
    assert group["kind"] == "automorphism" and group["algebra"] == struct[0]
    assert group["k"] == struct[1] and group["count"] == elements and group["sharing"] == sharing
    assert group["blocks"] == [0, struct[2]] and group["cones"] == struct[2]
    assert group["isotropic_closed_form"] is False
    assert record["cone_structure"] == {"algebra": struct[0], "k": struct[1], "cones": struct[2],
                                        "scalar_tier_treats_term_as": "one coupled block"}
    assert record["mathematical_block"] == {"size_k": rows, "count": 1, "coupled": True}
    assert record["distinct_dual_steps"] == 1
    assert record["shape_applied"].startswith("Sigma_i = c P(w_i)")
    assert comp.meta["scaling_classes"] == ["automorphism", "diagonal"]
    assert comp.meta["eval_kinds"][0] == block.eval_kind          # never "newton"
    assert comp.meta["eval_kinds"][1] == prob.compile(max_blocks=2, tol=1e-3).meta["eval_kinds"][1]
    shape = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert shape["kind"] == "automorphism" and shape["cond"] <= shape["cond_limit"]
    assert shape["cond_min"] >= 1.0 and shape["degrees_of_freedom_per_element"] == cm.degrees_of_freedom(*struct[:2])
    assert comp.meta["block_metrics"]["terms"][0]["carrier"].startswith(
        "Jordan element" if struct[0] == "soc" else "S (count, k, k)")
    assert comp.meta["block_metrics"]["numerical_domain"]["max_automorphism_cond"] == cm.MAX_AUTOMORPHISM_COND
    measured = comp.meta["block_metrics"]["measured_scaled_norm"]
    assert 0 < measured < 1 and measured == pytest.approx(0.9, rel=2e-3)
    declared_record = report["declared"][0]
    assert declared_record["declared_class"] == "automorphism"
    assert declared_record["cone"]["algebra"] == struct[0] and declared_record["cone"]["cones"] == struct[2]
    assert declared_record["degrees_of_freedom"] == cm.degrees_of_freedom(*struct[:2]) * elements
    # the constructed elements are interior and inside the domain
    els = np.asarray(shape["elements"], np.float64)
    if struct[0] == "soc":
        assert els.shape == (elements, struct[1]) and np.all(els[:, 0] > np.linalg.norm(els[:, 1:], axis=1))
    else:
        assert els.shape == (elements, struct[1], struct[1]) and np.all(np.linalg.eigvalsh(els) > 0)
    sol = comp.solve_instance(data)
    assert np.all(np.isfinite(np.asarray(sol.x)))
    # a metric is a change of coordinates: the solution must not move
    # metric-path test: this family has no exact dual witness under the default acceptance (g = 0 coupling with a cost, or a Zero block), so the explicit residual policy is requested here. The default objective-gap acceptance is tested in test_validate.py and test_stopping_contract.py.
    plain = prob.compile(max_blocks=4000, tol=1e-4, accuracy="residual")
    built = prob.compile(equilibrate="construct", max_blocks=4000, tol=1e-4, accuracy="residual",
                         metric_classes={0: declared}, metric_updates=12)
    a, b = plain.solve_instance(data), built.solve_instance(data)
    assert str(a.status) == "solved", ("undeclared compile", str(a.status), int(a.iters))
    assert str(b.status) == "solved", ("automorphism compile", str(b.status), int(b.iters))
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=5e-3, atol=5e-3)


# ===================================================================== T9
@pytest.mark.parametrize("kind", ["soc", "socbatch", "psd"])
def test_the_certificate_is_metric_free_and_byte_identical(kind):
    prob, data, K, n, block, rows = cone_family(kind)
    built = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                         metric_classes={0: "automorphism"}, metric_updates=12)
    plain = prob.compile(max_blocks=2, tol=1e-3)
    x, y = jnp.asarray(np.random.default_rng(9).normal(size=n), jnp.float32), jnp.full(rows + 5, 0.1, jnp.float32)
    left, right = built.certificate(x, y, data), plain.certificate(x, y, data)
    assert set(left) == set(right)
    for key in right:
        lv, rv = np.asarray(left[key]), np.asarray(right[key])
        assert lv.tobytes() == rv.tobytes() or (np.all(np.isnan(lv)) and np.all(np.isnan(rv))), key
    assert built.meta["scaled_norm"] != plain.meta["scaled_norm"]      # the metric changed


# ===================================================================== T10
class _SubSOC(cones.SecondOrderCone):
    pass


def _overridden_soc():
    blk = cones.SecondOrderCone()
    blk.prox = lambda v, t: blk.__class__.prox(blk, v, t)
    return blk


@pytest.mark.parametrize("family, kwargs, needle", [
    ("exp", dict(metric_classes={0: "automorphism"}), "is not a symmetric cone"),
    ("pow", dict(metric_classes={0: "automorphism"}), "is not a symmetric cone"),
    ("expbatch", dict(metric_classes={0: "automorphism"}), "is not a symmetric cone"),
    ("powbatch", dict(metric_classes={0: "automorphism"}), "is not a symmetric cone"),
    ("nonneg", dict(metric_classes={0: "automorphism"}), "defined only for a symmetric-cone block"),
    ("groupl2", dict(metric_classes={0: "automorphism"}), "defined only for a symmetric-cone block"),
    ("soc", dict(metric_classes={0: ("diagonal", "per_block")}), "forces every diagonal entry to be equal"),
    ("soc", dict(metric_classes={0: ("diagonal", "term")}), "forces every diagonal entry to be equal"),
    ("socbatch", dict(metric_classes={0: ("diagonal", "per_block")}), "forces every diagonal entry to be equal"),
    ("psd", dict(metric_classes={0: "diagonal"}), "members of the automorphism class, not of the free diagonal class"),
    ("soc", dict(metric_classes={0: "full"}), "only a valid proximal metric for a rotation-invariant"),
    ("soc", dict(metric_classes={0: "full"}), "the structured full class is 'automorphism'. Declare that."),
    ("psd", dict(metric_classes={0: ("full", "term")}), "the structured full class is 'automorphism'. Declare that."),
    ("subclass", dict(metric_classes={0: "automorphism"}), "exact type"),
    ("override", dict(metric_classes={0: "automorphism"}), "exact type"),
    ("soc", dict(metric_classes={0: "automorphism"}, equilibrate="sgd"), "require equilibrate='construct'"),
    ("soc", dict(metric_initial={0: None}, equilibrate="sgd"), "require equilibrate='construct'"),
    ("soc", dict(metric_classes={0: "scalar"}, metric_initial={0: None}), "declares 'scalar', not 'automorphism'"),
    ("soc", dict(metric_classes={0: "automorphism"}, metric_initial={0: None}), "must be a ConeAutomorphism"),
    ("soc", dict(metric_classes={0: ("automorphism", "sideways")}), "sharing must be one of"),
])
def test_refusals_name_the_reason(family, kwargs, needle):
    rng = np.random.default_rng(10)
    n = 8
    blocks = {"exp": (cones.ExpCone(), 3), "pow": (cones.PowerCone(0.5), 3),
              "expbatch": (cones.ExpConeBatch(2), 6), "powbatch": (cones.PowerConeBatch(2, 0.5), 6),
              "nonneg": (ocs.NonNeg(), 6), "groupl2": (ocs.GroupL2(0.9, 3), 6),
              "soc": (cones.SecondOrderCone(), 4), "socbatch": (cones.SOCBatch(2, 2), 6),
              "psd": (cones.PSDCone(3), 9), "subclass": (_SubSOC(), 4), "override": (_overridden_soc(), 4)}
    block, rows = blocks[family]
    K = rng.normal(size=(rows, n))
    x = ocs.Variable(n)
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)), Term(ocs.Dense(K), block)])
    kwargs = {"equilibrate": "construct", **kwargs}
    with pytest.raises(ValueError, match=re.escape(needle)):
        prob.compile(max_blocks=2, tol=1e-3, metric_updates=4, **kwargs)


SOC_DIAGONAL_REFUSAL = (
    "metric_classes[0]: {block} admits no diagonal metric in closed form except a scalar one: a "
    "positive diagonal D is an automorphism of the second-order cone only when D J D = mu J, which "
    "forces every diagonal entry to be equal, so a diagonal weighted projection has no closed form on "
    "this block. Declare 'scalar' or 'automorphism'.")
PSD_DIAGONAL_REFUSAL = (
    "metric_classes[0]: PSDCone admits no free diagonal metric in closed form: an arbitrary positive "
    "diagonal on the k*k entries is not a congruence of the cone, so its weighted projection has no "
    "closed form. The diagonal congruences Z -> D Z D with D = diag(s) (entries s_i s_j, k degrees of "
    "freedom) are cone automorphisms and are members of the automorphism class, not of the free "
    "diagonal class. Declare 'scalar' or 'automorphism'.")
NONSYMMETRIC_REFUSAL = (
    "metric_classes[0]: {block} is not a symmetric cone. The automorphism tier is the closed-form "
    "weighted projection W^{{-1}} Pi_C(W v) for W in the automorphism group of C, which exists for the "
    "self-scaled (symmetric) cones only (Nesterov and Todd 1997, Hauser and Gueler 2002). The "
    "exponential and power cones have no self-scaled barrier and no such family of positive definite "
    "scalings, so this release keeps them scalar. Declare 'scalar'.")


@pytest.mark.parametrize("family, kwargs, expected", [
    ("soc", dict(metric_classes={0: ("diagonal", "per_block")}), SOC_DIAGONAL_REFUSAL.format(block="SecondOrderCone")),
    ("socbatch", dict(metric_classes={0: ("diagonal", "term")}), SOC_DIAGONAL_REFUSAL.format(block="SOCBatch")),
    ("psd", dict(metric_classes={0: "diagonal"}), PSD_DIAGONAL_REFUSAL),
    ("exp", dict(metric_classes={0: "automorphism"}), NONSYMMETRIC_REFUSAL.format(block="ExpCone")),
    ("pow", dict(metric_classes={0: "automorphism"}), NONSYMMETRIC_REFUSAL.format(block="PowerCone")),
    ("expbatch", dict(metric_classes={0: "automorphism"}), NONSYMMETRIC_REFUSAL.format(block="ExpConeBatch")),
    ("powbatch", dict(metric_classes={0: "automorphism"}), NONSYMMETRIC_REFUSAL.format(block="PowerConeBatch")),
])
def test_refusal_messages_are_asserted_exactly(family, kwargs, expected):
    """The sharp negative (SOC), the PSD diagonal-congruence statement and the
    non-symmetric refusal, asserted as the WHOLE message, not a needle."""
    rng = np.random.default_rng(24)
    n = 8
    blocks = {"exp": (cones.ExpCone(), 3), "pow": (cones.PowerCone(0.5), 3),
              "expbatch": (cones.ExpConeBatch(2), 6), "powbatch": (cones.PowerConeBatch(2, 0.5), 6),
              "soc": (cones.SecondOrderCone(), 4), "socbatch": (cones.SOCBatch(2, 2), 6),
              "psd": (cones.PSDCone(3), 9)}
    block, rows = blocks[family]
    K = rng.normal(size=(rows, n))
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)), Term(ocs.Dense(K), block)])
    with pytest.raises(ValueError) as err:
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, metric_updates=4, **kwargs)
    assert str(err.value) == expected


def test_explicit_carrier_refusals_name_the_reason():
    w = random_w(np.random.default_rng(11), 4)
    with pytest.raises(ValueError, match="is a metric for a symmetric-cone block"):
        bm.normalize_term_metrics(cm.ConeAutomorphism.soc(w), 0, ocs.NonNeg(), 4)
    with pytest.raises(ValueError, match="does not match"):
        bm.normalize_term_metrics(cm.ConeAutomorphism.soc(w), 0, cones.SecondOrderCone(), 5)
    with pytest.raises(ValueError, match="does not match"):
        bm.normalize_term_metrics(cm.ConeAutomorphism.soc(np.stack([w, w])), 0, cones.SOCBatch(3, 3), 12)
    with pytest.raises(ValueError, match="does not match"):
        bm.normalize_term_metrics(cm.ConeAutomorphism.psd(np.eye(3)), 0, cones.SecondOrderCone(), 9)
    with pytest.raises(ValueError, match=re.escape("not a supported rotation-invariant")) as err:
        bm.normalize_term_metrics(bm.BlockMetric([1., 2., 3., 4.]), 0, cones.SecondOrderCone(), 4)
    assert "declare a ConeAutomorphism instead" in str(err.value)
    with pytest.raises(ValueError, match="takes exactly ONE ConeAutomorphism"):
        bm.normalize_term_metrics([cm.ConeAutomorphism.soc(w)], 0, cones.SecondOrderCone(), 4)
    with pytest.raises(ValueError, match="takes exactly ONE ConeAutomorphism"):
        bm.normalize_term_metrics((cm.ConeAutomorphism.soc(w), cm.ConeAutomorphism.soc(w)), 0,
                                  cones.SOCBatch(2, 3), 8)
    with pytest.raises(ValueError, match="must be interior"):
        cm.ConeAutomorphism.soc([1.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="exactly symmetric"):
        cm.ConeAutomorphism.psd([[1.0, 0.1], [0.2, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        cm.ConeAutomorphism.psd([[1.0, 2.0], [2.0, 1.0]])
    # one step for the whole term, refused rather than averaged
    carrier = cm.ConeAutomorphism.soc(np.stack([w, w]))
    k, ranges = bm.normalize_term_metrics(carrier, 0, cones.SOCBatch(2, 3), 8)
    # One plumbing block for the whole term: the range bounds live in
    # [0, size // k) like every other carrier's, and the cone count is on the
    # carrier, never smuggled in the bound.
    assert k == 8 and ranges[0][:2] == (0, 1) and ranges[0][2].sharing == "per_block"
    assert ranges[0][2].cones == 2 and ranges[0][2].count == 2 and ranges[0][2].k == 4
    assert carrier.layout is None and ranges[0][2].layout == ("socbatch", 2, 3)   # the caller's object is untouched
    sigma = np.full(8, 0.1)
    sigma[3] = 0.2
    with pytest.raises(ValueError, match="requires ONE dual step for the whole term"):
        bm.check_static_steps(sigma, 0, 8, 8, ranges, 0)
    checked = bm.check_static_steps(np.full(8, 0.1), 0, 8, 8, ranges, 0)
    assert checked[0][3] is False
    assert bm.check_static_steps(np.float64(0.1), 0, 8, 8, ranges, 0)[0][3] is False   # D10: both branches


# =========================================================== sharp negative
def test_sharp_negative_witness_diagonal_projection_is_not_an_automorphism_projection():
    """Brief check 5 as an engine test: v = -D^{-1} c with c interior, -v outside
    C. The D-weighted projection of v is exactly 0 (KKT: D (v - 0) = -c in
    -C), while W^{-1} Pi(W v) is nonzero for every automorphism W because W v
    is not in -C. So no P(w) reproduces the diagonal metric D.

    The witness the verdict grafted (D = diag(1, 2, 3), c = (1, 0.3, 0.2))
    puts -v = D^{-1} c INSIDE C, where every cone projection of v is zero and
    the test could not fail; D = diag(3, 1, 1), c = (1, 0.5, 0.5) puts -v
    outside C (|(0.5, 0.5)| > 1/3) with c interior, which is the witness the
    argument needs. The reference script pins the same instance."""
    rng = np.random.default_rng(12)
    k = 3
    D = np.diag([3.0, 1.0, 1.0])
    c = np.array([1.0, 0.5, 0.5])
    v = -np.linalg.solve(D, c)
    assert soc_defect(-v) > 0.05 * np.linalg.norm(v)
    assert np.linalg.norm(ref_weighted_proj_pgd(D, v)) < 1e-9
    assert max(ref_kkt_soc(D, v, np.zeros(k)).values()) < 1e-12
    for _ in range(20):
        w = random_w(rng, k)
        carrier = cm.ConeAutomorphism.soc(w, layout=("soc",))
        z = np.asarray(cm.prox_automorphism(carrier, jnp.asarray(v, jnp.float32)), np.float64)
        assert np.linalg.norm(z) > 1e-3 * np.linalg.norm(v)


# ===================================================================== T11
def test_domain_is_enforced_never_clipped_and_restrict_recomputes(monkeypatch):
    rng = np.random.default_rng(13)
    e = np.array([1.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.6, 0.0, 0.8])
    wide = 0.5 * (20.0 + 1.0) * e + 0.5 * (20.0 - 1.0) * u          # lam1/lam2 = 20 -> cond 400
    with pytest.raises(ValueError, match="condition number at most 100") as err:
        cm.ConeAutomorphism.soc(wide)
    assert "restrict_cone_condition" in str(err.value) and "never clipped" in str(err.value)
    restricted, info = cm.restrict_cone_condition("soc", wide, 100.0)
    assert info["restricted"] and info["cond_before"] == pytest.approx(400.0)
    assert info["cond_after"] <= 100.0 and info["cones_restricted"] == [0]
    ok = cm.ConeAutomorphism.soc(restricted)
    assert ok.cond <= 100.0
    assert restricted.shape == wide.shape             # a (k,) element comes back as (k,)
    # P(w') dominates P(w) in the Loewner order
    assert np.linalg.eigvalsh(ref_P(restricted) - ref_P(wide)).min() >= -1e-9
    S = random_spd(rng, 3, cond_max=1e4)
    Sr, sinfo = cm.restrict_cone_condition("psd", S, 100.0)
    assert Sr.shape == S.shape
    assert sinfo["restricted"] and np.linalg.eigvalsh(np.kron(Sr, Sr) - np.kron(S, S)).min() >= -1e-9
    # the constructor path: reject (default) and restrict
    prob, data, K, n, block, rows = cone_family("socbatch")
    inside = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                          metric_classes={0: "automorphism"}, metric_updates=12)
    shape = inside.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert shape["cond"] > 1.0
    limit = float(np.sqrt(shape["cond"]))
    monkeypatch.setattr(cm, "MAX_AUTOMORPHISM_COND", limit)
    with pytest.raises(ValueError, match="Nothing is clipped for you"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                     metric_classes={0: "automorphism"}, metric_updates=12)
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: "automorphism"}, metric_updates=12, metric_domain="restrict")
    restriction = comp.meta["metric_construction"]["restriction"]
    applied = restriction["applied"][0]
    assert applied["restricted"] and applied["kind"] == "automorphism" and applied["cones_restricted"]
    assert applied["cond_before"] == pytest.approx(shape["cond"], rel=1e-6)
    assert applied["cond_after"] <= limit * (1 + 1e-6)
    assert restriction["measured_after_restriction"] >= restriction["measured_before_restriction"]
    assert comp.meta["metric_construction"]["operator_products"]["normalization"] == 60
    assert comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["cond"] <= limit * (1 + 1e-6)


# ===================================================================== T12
def test_identity_element_canonicalizes_to_the_scalar_tier_byte_identically():
    prob, data, K, n, block, rows = cone_family("soc")
    tau, sigma = np.full(n, 0.01, "f"), np.full(rows + 5, 0.01, "f")
    plain = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12)
    comp = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: cm.ConeAutomorphism.soc(np.r_[1.0, np.zeros(rows - 1)])})
    assert comp.meta["block_metrics"]["canonicalized_to_scalar_tier"] == [0]
    assert comp.meta["scaling_classes"] == ["scalar", "diagonal"]
    assert endpoints(comp, data) == endpoints(plain, data)
    # a non-identity element keeps the weighted path and the precise label
    kept = prob.compile(equilibrate=("given", tau, sigma, 0.5), max_blocks=3, tol=1e-12,
                        block_metrics={0: cm.ConeAutomorphism.soc(random_w(np.random.default_rng(1), rows))})
    assert kept.meta["scaling_classes"] == ["automorphism", "diagonal"]
    assert "canonicalized_to_scalar_tier" not in kept.meta["block_metrics"]


# ===================================================================== T13
@pytest.mark.parametrize("kind", ["soc", "socbatch", "psd"])
def test_default_declaration_on_a_cone_family_is_byte_identical_to_sgd(kind, monkeypatch):
    """The steps handed to the solver (tau, sigma digests) and the endpoints of
    an undeclared construct compile equal the shipped equilibrate='sgd' ones."""
    sink = []
    captured_solver(monkeypatch, sink)
    prob, data, K, n, block, rows = cone_family(kind)
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    built = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3)
    # each compile builds the iteration solver and its gap-drive solver with the same
    # steps, so the capture holds one pair per compile: compare the first of each.
    assert len(sink) in (2, 4) and len(sink) % 2 == 0
    half = len(sink) // 2
    assert sink[0]["tau"] == sink[half]["tau"] and sink[0]["sigma"] == sink[half]["sigma"]
    assert built.meta["metric_construction"]["regime"] == "smfe"
    assert built.meta["metric_construction"]["bit_identical_to_sgd"] is True
    assert "block_metrics" not in built.meta
    assert built.meta["scaling_classes"] == shipped.meta["scaling_classes"] == ["scalar", "diagonal"]
    assert built.meta["eval_kinds"] == shipped.meta["eval_kinds"]
    assert endpoints(built, data) == endpoints(shipped, data)
    # the cone structure is recorded beside the scalar tier's one coupled block on both
    struct = cm.symmetric_cone_structure(block, rows)
    for comp in (shipped, built):
        entry = comp.meta["effective_metric"]["terms"][0]
        assert entry["cone_structure"]["cones"] == struct[2] and entry["cone_structure"]["k"] == struct[1]
        assert entry["mathematical_block"] == {"size_k": rows, "count": 1, "coupled": True}


# ===================================================================== T14
def test_the_isotropic_w_update_is_the_scalar_kind_step_under_the_parameter_map():
    rng = np.random.default_rng(14)
    k, probes, mu, box, target, eta = 4, 6, 1.0, 2.0, 0.8, 0.3
    a = 1.7
    w = np.zeros(k)
    w[0] = a
    group = mc._Group("automorphism", k, 1, algebra="soc", shared=False, layout=("soc",))
    z = rng.normal(size=(probes, k))
    z_sym = np.concatenate([z, z * np.r_[1.0, -np.ones(k - 1)]])        # E[z0 zbar] = 0 exactly
    got, finite = mc.update_side((group,), (jnp.asarray(w[None], jnp.float32),),
                                 jnp.asarray(z_sym, jnp.float32), target, mu, box, eta)
    assert bool(finite)
    got = np.asarray(got[0], np.float64)[0]
    # the scalar kind's own step in value = log(a^2)
    energy = np.mean(z_sym * z_sym)
    value = np.log(a * a)
    tangent = 0.5 * np.exp(value) * energy - 0.5 * target + mu / 4 * value
    expected = np.exp(value - eta * tangent)
    assert abs(got[0] ** 2 - expected) <= 1e-5 * expected and np.linalg.norm(got[1:]) <= 1e-5 * got[0]
    # an unsymmetrized probe set moves wbar: the class is richer than scalar
    got2, _ = mc.update_side((group,), (jnp.asarray(w[None], jnp.float32),),
                             jnp.asarray(z, jnp.float32), target, mu, box, eta)
    assert np.linalg.norm(np.asarray(got2[0])[0, 1:]) > 1e-3


# ===================================================================== T15
def ref_projected_ambient_step_soc(w, Z, target, mu, eta):
    """The AMBIENT rule of update_side at A = P(w) with the tangent replaced by
    its Frobenius projection onto {L(g)}, computed in float64."""
    k = w.size
    A = ref_P(w)
    Ah = spd_pow(A, 0.5)
    G = Z.T @ Z / Z.shape[0]
    lam, U = np.linalg.eigh(A)
    logA = (U * np.log(lam)) @ U.T
    T = 0.5 * Ah @ G @ Ah - 0.5 * target * np.eye(k) + mu / 4 * logA
    g0 = np.trace(T) / k
    gbar = T[1:, 0]
    L = np.zeros((k, k))
    L[0, 0] = g0
    L[1:, 1:] = g0 * np.eye(k - 1)
    L[1:, 0] = gbar
    L[0, 1:] = gbar
    return Ah @ spd_pow(np.eye(k) * 0 + (-eta) * L + 0 * np.eye(k), 1.0) if False else Ah @ _expm_sym(-eta * L) @ Ah


def _expm_sym(M):
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    return (U * np.exp(lam)) @ U.T


def _logm_sym(M):
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    return (U * np.log(lam)) @ U.T


def test_the_w_update_is_the_projected_ambient_geodesic_step():
    rng = np.random.default_rng(15)
    k, probes, mu, box, target, eta = 3, 8, 1.0, 2.0, 0.8, 0.2
    group = mc._Group("automorphism", k, 1, algebra="soc", shared=False, layout=("soc",))
    for _ in range(10):
        w = random_w(rng, k, cond_max=16.0)      # inside the box with room, so no clip fires (D7)
        Z = rng.normal(size=(probes, k)) * 0.7
        got, finite = mc.update_side((group,), (jnp.asarray(w[None], jnp.float32),),
                                     jnp.asarray(Z, jnp.float32), target, mu, box, eta)
        assert bool(finite)
        w_new = np.asarray(got[0], np.float64)[0]
        l1, l2, _, _ = ref_spectral(w_new)
        assert np.exp(-box) < l2 and l1 < np.exp(box)                   # the clip did not fire
        A_ref = ref_projected_ambient_step_soc(w, Z, target, mu, eta)
        np.testing.assert_allclose(ref_P(w_new), A_ref, rtol=1e-4, atol=1e-5)


def test_the_psd_update_is_the_bounded_partial_trace_projected_ambient_step():
    rng = np.random.default_rng(16)
    k, probes, mu, box, target, eta = 3, 8, 1.0, 2.0, 0.8, 0.2
    group = mc._Group("automorphism", k, 1, algebra="psd", shared=False, layout=("psd", k))
    for _ in range(6):
        S = random_spd(rng, k, cond_max=16.0)
        Z = rng.normal(size=(probes, k * k)) * 0.5
        got, finite = mc.update_side((group,), (jnp.asarray(S[None], jnp.float32),),
                                     jnp.asarray(Z, jnp.float32), target, mu, box, eta)
        assert bool(finite)
        S_new = np.asarray(got[0], np.float64)[0]
        # ambient: A = S kron S on R^{k^2}, T projected onto {Y kron I + I kron Y}
        A = np.kron(S, S)
        Ah = spd_pow(A, 0.5)
        G = Z.T @ Z / probes
        T = 0.5 * Ah @ G @ Ah - 0.5 * target * np.eye(k * k) + mu / 4 * _logm_sym(A)
        # least squares fit of T by Y kron I + I kron Y over symmetric Y (brute force)
        basis = []
        for i in range(k):
            for j in range(i, k):
                Y = np.zeros((k, k))
                Y[i, j] = Y[j, i] = 1.0
                basis.append((np.kron(Y, np.eye(k)) + np.kron(np.eye(k), Y)).reshape(-1))
        B = np.stack(basis, axis=1)
        coef, *_ = np.linalg.lstsq(B, T.reshape(-1), rcond=None)
        Y = np.zeros((k, k))
        idx = 0
        for i in range(k):
            for j in range(i, k):
                Y[i, j] = Y[j, i] = coef[idx]
                idx += 1
        # The successor explicitly bounds the carrier step, preserving the
        # independently fitted ambient tangent rather than its old step length.
        eta_ref = min(eta, .5 / np.linalg.norm(Y, 'fro'))
        A_ref = Ah @ _expm_sym(-eta_ref * (np.kron(Y, np.eye(k)) + np.kron(np.eye(k), Y))) @ Ah
        np.testing.assert_allclose(np.kron(S_new, S_new), A_ref, rtol=2e-4, atol=2e-5)


# ===================================================================== T16
def test_nt_scaling_point_solves_P_w_s_equals_z_and_seeds_the_run():
    rng = np.random.default_rng(17)
    block = cones.SOCBatch(3, 3)
    S = np.stack([random_interior(rng, 4, scale=rng.uniform(0.5, 2.0)) for _ in range(3)])
    Z = np.stack([random_interior(rng, 4, scale=rng.uniform(0.5, 2.0)) for _ in range(3)])
    layout = cm.cone_layout(block)
    s, z = cm.unrows(layout, S, np), cm.unrows(layout, Z, np)
    nt = cm.nt_scaling_point(block, s, z)
    assert nt.algebra == "soc" and nt.count == 3 and nt.origin["direction"] == "P(w) s = z"
    for j in range(3):
        w = nt.w_host[j] * nt.origin["det_scale_removed"][j]
        np.testing.assert_allclose(ref_P(w) @ S[j], Z[j], rtol=1e-10, atol=1e-12)
        assert abs(ref_det(nt.w_host[j]) - 1.0) < 1e-12                    # det-normalized
    # the reversed convention gives the inverse element (shape only, up to scale)
    rev = cm.nt_scaling_point(block, z, s)
    for j in range(3):
        np.testing.assert_allclose(ref_pow(rev.w_host[j], -1.0), nt.w_host[j], rtol=1e-9)
    psd = cones.PSDCone(3)
    Sm, Zm = random_spd(rng, 3), random_spd(rng, 3)
    ntp = cm.nt_scaling_point(psd, Sm.reshape(-1), Zm.reshape(-1))
    Wp = ntp.S_host[0] * ntp.origin["det_scale_removed"][0]
    np.testing.assert_allclose(Wp @ Sm @ Wp, Zm, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="is not interior on cone 1"):
        bad = S.copy()
        bad[1, 0] = 0.5 * np.linalg.norm(bad[1, 1:])
        cm.nt_scaling_point(block, cm.unrows(layout, bad, np), z)
    assert ocs.nt_scaling_point is cm.nt_scaling_point and ocs.ConeAutomorphism is cm.ConeAutomorphism


def test_metric_initial_seeds_the_run_and_updates_1_returns_the_seed_itself():
    """D2 of the refutation, stated plainly: the returned metric is the mean of
    the pre-update states, so with metric_updates=1 it IS the seed."""
    rng = np.random.default_rng(18)
    prob, data, K, n, block, rows = cone_family("socbatch")
    seed = cm.ConeAutomorphism.soc(np.stack([random_w(rng, 3, cond_max=9.0) for _ in range(3)]))
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: "automorphism"}, metric_updates=1, metric_initial={0: seed})
    init = comp.meta["metric_construction"]["initializer"][0]
    assert init["source"] == "metric_initial" and init["term"] == 0
    assert "with metric_updates=1 the returned element is the initial element itself" in init["returned_metric"]
    got = np.asarray(comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["elements"], np.float64)
    assert got.astype(np.float32).tobytes() == seed.w_host.astype(np.float32).tobytes()
    # an out-of-box seed is refused, not clipped and not status 1
    with pytest.raises(ValueError, match="outside the spectral box"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, metric_classes={0: "automorphism"},
                     metric_updates=1, metric_initial={0: cm.ConeAutomorphism.soc(20.0 * seed.w_host)})
    # a seed of the wrong count for the declared sharing is refused
    with pytest.raises(ValueError, match="does not match term 0"):
        prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, metric_classes={0: ("automorphism", "term")},
                     metric_updates=1, metric_initial={0: seed})
    identity = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                            metric_classes={0: "automorphism"}, metric_updates=12)
    assert identity.meta["metric_construction"]["initializer"][0]["source"] == "identity"


def test_two_updates_return_the_geodesic_mean_of_the_seed_and_its_one_update():
    """A test that can fail (D2): with iterations=2 the returned element is
    P(a_0) #_ratio P(a_1) with a_1 the seed's one update, replicated on the host
    in float64 from the same Rademacher probes."""
    rng = np.random.default_rng(19)
    k, n, m, probes, shift, mu, box = 3, 6, 3, 2, 20, 1.0, 2.0
    K = rng.normal(size=(m, n))
    K32 = jnp.asarray(K, jnp.float32)
    rows = (mc._Group("automorphism", k, 1, algebra="soc", shared=False, layout=("soc",)),)
    columns = (mc._Group("diagonal", n, 1),)
    program, info = mc.make_geodesic_constructor(lambda x: K32 @ x, lambda y: K32.T @ y, rows, columns,
                                                 iterations=2, probes=probes, mu=mu, box=box, shift=shift)
    w0 = random_w(rng, k, cond_max=9.0)
    key = jax.random.PRNGKey(3)
    answer = program(key, (jnp.asarray(w0[None], jnp.float32),), mc.initial_state(columns))
    assert int(answer["status"]) == 0
    got = np.asarray(answer["A"][0], np.float64)[0]
    # replicate update 1 on the host: the probes of the first iteration
    key1, row_key, _col_key = jax.random.split(key, 3)
    xi = np.asarray(jax.random.rademacher(row_key, (probes, n), dtype=jnp.float32), np.float64)
    Z = xi @ K.T                                   # B = I at the identity column state
    eta1 = 2 / ((mu / 4) * (1 + shift))
    A1 = ref_projected_ambient_step_soc(w0, Z, info["targets"]["row"], mu, eta1)
    ratio = (2 + shift - 1) / (shift + (2 + shift - 1))
    A0 = ref_P(w0)
    A0h, A0mh = spd_pow(A0, 0.5), spd_pow(A0, -0.5)
    mean = A0h @ spd_pow(A0mh @ A1 @ A0mh, ratio) @ A0h
    np.testing.assert_allclose(ref_P(got), mean, rtol=1e-4, atol=1e-5)
    assert not np.allclose(got, w0, rtol=1e-6)      # it moved: not the seed


# ===================================================================== T17
class _OperatorOnly(ocs.LinearOperator):
    """Exposes matvec and rmatvec and nothing else; any other access raises."""

    def __init__(self, A, calls):
        super().__init__(A.shape)
        object.__setattr__(self, "_hidden", jnp.asarray(A, jnp.float32))
        object.__setattr__(self, "_calls", calls)

    def matvec(self, x):
        self._calls["mv"] += 1
        return jnp.matmul(self._hidden, x, precision=jax.lax.Precision.HIGHEST)

    def rmatvec(self, y):
        self._calls["rmv"] += 1
        return jnp.matmul(self._hidden.T, y, precision=jax.lax.Precision.HIGHEST)


def test_operator_only_contract_of_the_automorphism_construction():
    """K is reached through matvec/rmatvec only. The recorded operator-product
    count is a DECLARED constant (2 * updates * probes + 30 * norm_calls), not a
    host-side count: under jit the Python callables are traced, not called per
    product, so a Python counter sees traces. What is asserted here is that the
    compile succeeds on an operator that exposes nothing else, that the
    callables were traced, and that the declared budget is the formula."""
    rng = np.random.default_rng(20)
    n, rows = 8, 6
    calls = {"mv": 0, "rmv": 0}
    op = _OperatorOnly(rng.normal(size=(rows, n)), calls)
    x = ocs.Variable(n)
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         Term(op, cones.SOCBatch(2, 2)),
                         ocs.nonneg(ocs.Dense(rng.normal(size=(3, n))) @ x)])
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: "automorphism"}, metric_updates=5, metric_probes=2)
    assert calls["mv"] > 0 and calls["rmv"] > 0
    products = comp.meta["metric_construction"]["operator_products"]
    assert products == {"gradient": 2 * 5 * 2, "normalization": 30, "total": 2 * 5 * 2 + 30, "note": products["note"]}
    # the prox path never touches the operator: the dual update reads the block only
    calls_before = dict(calls)
    carrier = comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]
    assert carrier["kind"] == "automorphism"
    v = jnp.asarray(rng.normal(size=rows), jnp.float32)
    cm.automorphism_moreau(cones.SOCBatch(2, 2), v, jnp.full((rows,), 0.3, jnp.float32),
                           ((0, 1, cm.ConeAutomorphism.soc(np.asarray(carrier["elements"]), layout=("socbatch", 2, 2)), False),))
    assert calls == calls_before


# ====================================================== the applied metric
def test_the_applied_forward_product_is_P_w_times_K(monkeypatch):
    """Graft 5 / B T11: capture the iteration matvec the solver receives and
    check P(w)(K e_j) against the host from the recorded elements."""
    sink = []
    captured_solver(monkeypatch, sink)
    prob, data, K, n, block, rows = cone_family("socbatch")
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: "automorphism"}, metric_updates=12)
    elements = np.asarray(comp.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["elements"], np.float64)
    assert elements.shape == (3, 3)
    # the fixture is anisotropic enough: the constructed elements are not all e
    assert np.linalg.norm(elements[:, 1:]) > 1e-3
    matvec = sink[-1]["matvec"]
    layout = ("socbatch", 3, 2)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        got = np.asarray(matvec(jnp.asarray(ej, jnp.float32)), np.float64)
        Kx = K @ ej
        R = cm.rows(layout, Kx[:rows], np)
        expected = np.stack([ref_P(elements[i]) @ R[i] for i in range(3)])
        np.testing.assert_allclose(got[:rows], cm.unrows(layout, expected, np), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got[rows:], Kx[rows:], rtol=1e-6, atol=1e-6)    # the nonneg term is untouched


def test_socbatch_per_cone_elements_equal_per_cone_second_order_cone_results():
    rng = np.random.default_rng(21)
    q, k = 4, 3
    W = np.stack([random_w(rng, k) for _ in range(q)])
    batch = cm.ConeAutomorphism.soc(W, layout=("socbatch", q, k - 1))
    R = rng.normal(size=(q, k)) * 2
    v = cm.unrows(("socbatch", q, k - 1), R, np)
    got = np.asarray(cm.automorphism_moreau(cones.SOCBatch(q, k - 1), jnp.asarray(v, jnp.float32),
                                            jnp.full((q * k,), 0.2, jnp.float32), ((0, 1, batch, False),)))
    got_rows = cm.rows(("socbatch", q, k - 1), got, np)
    for j in range(q):
        single = cm.ConeAutomorphism.soc(W[j], layout=("soc",))
        one = np.asarray(cm.automorphism_moreau(cones.SecondOrderCone(), jnp.asarray(R[j], jnp.float32),
                                                jnp.full((k,), 0.2, jnp.float32), ((0, 1, single, False),)))
        np.testing.assert_allclose(got_rows[j], one, rtol=1e-6, atol=1e-7)


def test_sharing_semantics_term_records_one_element_and_per_block_records_count():
    prob, data, K, n, block, rows = cone_family("socbatch")
    term = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                        metric_classes={0: ("automorphism", "term")}, metric_updates=12)
    per_block = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                             metric_classes={0: ("automorphism", "per_block")}, metric_updates=12)
    t, p = (c.meta["effective_metric"]["terms"][0]["metric_sharing_groups"][0] for c in (term, per_block))
    assert t["count"] == 1 and t["sharing"] == "term" and t["degrees_of_freedom"] == 3
    assert p["count"] == 3 and p["sharing"] == "per_block" and p["degrees_of_freedom"] == 9
    # state_bytes["rows"] also counts the nonneg term's diagonal group, so the
    # sharing shows as the difference: (count - 1) extra elements of k floats
    t_bytes = term.meta["metric_construction"]["state_bytes"]["rows"]
    p_bytes = per_block.meta["metric_construction"]["state_bytes"]["rows"]
    assert p_bytes - t_bytes == 4 * 3 * (3 - 1)
    assert term.meta["metric_construction"]["declared"][0]["metric_sharing_groups"] == 1
    assert per_block.meta["metric_construction"]["declared"][0]["metric_sharing_groups"] == 3
    for comp in (term, per_block):
        assert np.all(np.isfinite(np.asarray(comp.solve_instance(data).x)))
    # the two constructions differ: the class was applied, not relabelled
    te = np.asarray(term.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["elements"])
    pe = np.asarray(per_block.meta["block_metrics"]["terms"][0]["sharing_groups"][0]["elements"])
    assert te.shape == (1, 3) and pe.shape == (3, 3)


def test_explicit_carrier_through_block_metrics_matches_the_float64_reference_step():
    """One standard PDHG step with an explicit per-cone carrier, against the
    float64 weighted Moreau update computed from the P(w) matrices."""
    prob, data, K, n, block, rows = cone_family("socbatch")
    rng = np.random.default_rng(22)
    W = np.stack([random_w(rng, 3) for _ in range(3)])
    carrier = cm.ConeAutomorphism.soc(W)
    tau = np.full(n, 0.01)
    sigma = np.full(rows + 5, 0.01)
    comp = prob.compile(equilibrate=("given", tau.astype("f"), sigma.astype("f"), 0.5),
                        block_metrics={0: carrier}, pdhg_variant="standard", block=1, max_blocks=1, tol=1e-12)
    assert comp.meta["scaling_classes"] == ["automorphism", "diagonal"]
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1
    x0, y0 = rng.normal(size=n), rng.normal(size=rows + 5) * 0.1
    y0[rows:] = -abs(y0[rows:])
    sol = comp.solve_instance(data, warm_start=(jnp.asarray(x0, "f"), jnp.asarray(y0, "f")))
    l = np.concatenate([np.asarray(data["l1"], np.float64), np.asarray(data["l2"], np.float64)])
    u = K @ x0 + l
    layout = ("socbatch", 3, 2)
    Ru, Ry = cm.rows(layout, u[:rows], np), cm.rows(layout, y0[:rows], np)
    Y = np.empty((3, 3))
    for j in range(3):
        Sigma = sigma[0] * ref_P(W[j])
        v = Ry[j] + Sigma @ Ru[j]
        Y[j] = v - Sigma @ ref_weighted_proj(Sigma, np.linalg.solve(Sigma, v))
    y = np.concatenate([cm.unrows(layout, Y, np), np.minimum(y0[rows:] + sigma[rows:] * u[rows:], 0)])
    x = (x0 - tau * (K.T @ y + FAMILY_COSTS[("socbatch", 0)])) / (1 + 0.5 * tau)
    # This reference is one iteration step, before certifying-dual recovery.
    np.testing.assert_allclose(np.asarray(sol.y_iterate), y, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sol.x), x, rtol=1e-4, atol=1e-6)


def test_group_dispatch_keys_cone_carriers_apart_and_batches_them_with_metric():
    rng = np.random.default_rng(23)
    blocks = [cones.SecondOrderCone(), cones.SecondOrderCone()]
    slices = [(0, 4), (4, 8)]
    ra = bm.normalize_term_metrics(cm.ConeAutomorphism.soc(random_w(rng, 4)), 0, blocks[0], 4)
    rb = bm.normalize_term_metrics(cm.ConeAutomorphism.soc(random_w(rng, 4)), 1, blocks[1], 4)
    groups = build_groups(blocks, slices, metrics=[ra, rb])
    assert len(groups) == 2 and len(build_groups(blocks, slices)) == 1
    sigma = np.full(8, 0.3)
    v = rng.normal(size=8) * 2
    got = np.asarray(jax.jit(grouped_moreau_dual(groups, 8, static_sigma=jnp.asarray(sigma, jnp.float32)))(
        jnp.asarray(v, jnp.float32), jnp.asarray(sigma, jnp.float32)), np.float64)
    for (k, ranges), (a, b) in zip((ra, rb), slices):
        Sigma = 0.3 * ref_P(ranges[0][2].w_host[0])
        vv = v[a:b]
        np.testing.assert_allclose(got[a:b], vv - Sigma @ ref_weighted_proj(Sigma, np.linalg.solve(Sigma, vv)),
                                   rtol=5e-5, atol=5e-6)


# ==================================================== the CVXPY-bridge route
@pytest.mark.parametrize("q, n, d", [(1, 2, 3), (3, 6, 3)])
def test_cvxpy_bridge_soc_terms_take_the_automorphism_tier(q, n, d):
    """ocs/cvxpy_mf/solve.py routes a CVXPY SOC constraint to SecondOrderCone
    (one cone) or SOCBatch(q, d) (several); the automorphism class is declared
    on that term through the bridge's compile kwargs and the recovered
    objective agrees with the undeclared solve and with CVXPY's own solver."""
    cp = pytest.importorskip("cvxpy")
    from ocs.cvxpy_mf.solve import compile_mf
    rng = np.random.default_rng(30 + q)
    A = rng.normal(size=(d * q, n)) * np.geomspace(0.5, 2.0, d * q)[:, None]
    c = rng.normal(size=n)
    x = cp.Variable(n)
    t = cp.Variable(q)
    X = cp.reshape(A @ x, (d, q), order="F")
    problem = cp.Problem(cp.Minimize(c @ x), [cp.SOC(t, X, axis=0), t <= 1.0])
    plain, recover_plain, _ = compile_mf(problem, max_blocks=2000, tol=1e-5, accuracy="residual")
    terms = plain.meta["effective_metric"]["terms"]
    soc = [e["term"] for e in terms if e["block"] in ("SecondOrderCone", "SOCBatch")]
    assert len(soc) == 1
    assert terms[soc[0]]["block"] == ("SecondOrderCone" if q == 1 else "SOCBatch")
    assert terms[soc[0]]["cone_structure"] == {"algebra": "soc", "k": d + 1, "cones": q,
                                               "scalar_tier_treats_term_as": "one coupled block"}
    built, recover_built, _ = compile_mf(problem, equilibrate="construct",
                                         metric_classes={soc[0]: "automorphism"}, max_blocks=2000, tol=1e-5, accuracy="residual")
    group = built.meta["effective_metric"]["terms"][soc[0]]["metric_sharing_groups"][0]
    assert group["kind"] == "automorphism" and group["count"] == q and group["k"] == d + 1
    assert built.meta["scaling_classes"][soc[0]] == "automorphism"
    assert built.meta["metric_construction"]["regime"] == "geodesic"
    a, b = plain.solve_instance({}), built.solve_instance({})
    assert str(a.status) == str(b.status) == "solved"
    va, vb = float(recover_plain(a)), float(recover_built(b))
    assert abs(va - vb) <= 2e-3 * max(1.0, abs(va))
    reference = next((s for s in ("CLARABEL", "SCS") if s in cp.installed_solvers()), None)
    if reference is not None:
        ref = float(problem.solve(solver=reference))
        assert abs(vb - ref) <= 2e-3 * max(1.0, abs(ref)), (vb, ref, reference)


def test_describe_truncates_large_batches_and_the_identity_state_applies_as_identity():
    w = np.zeros((20, 3))
    w[:, 0] = 1.0
    big = cm.ConeAutomorphism.soc(w)
    d = big.describe()
    assert "elements" not in d and len(d["elements_sha256"]) == 64 and big.is_identity
    groups = (mc._Group("automorphism", 3, 4, algebra="soc", shared=False, layout=("socbatch", 4, 2)),
              mc._Group("automorphism", 3, 1, algebra="psd", shared=False, layout=("psd", 3)),
              mc._Group("scalar", 5, 2))
    state = mc.initial_state(groups)
    v = jnp.asarray(np.random.default_rng(0).normal(size=sum(g.dimension for g in groups)), jnp.float32)
    for power in (1.0, 0.5, -0.5):
        np.testing.assert_allclose(np.asarray(mc.apply_power(groups, state, v, power)), np.asarray(v),
                                   rtol=1e-6, atol=1e-6)
    assert mc.state_bytes(groups) == 4 * (4 * 3 + 9 + 2)
