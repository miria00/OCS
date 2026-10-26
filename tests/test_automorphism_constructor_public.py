"""Public symmetric-cone construction and unchanged supplied-metric safety.

The former defective-constructor refusal is replaced by public-path checks.
Malformed declarations and the supplied-carrier condition cap stay enforced.
"""
import inspect
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
from ocs.solver import metric_construct as mc
import ocs.model.compile as compile_module

EPS32 = float(np.finfo(np.float32).eps)
CAP = 100.0
AT_CAP = 1.0 - 1e-9          # lam1/lam2 = 10 * AT_CAP: cond P(w) = 100 (1 - 2e-9), inside the cap


def _two_term(block, rows, n=8, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(rows, n))
    return ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)), Term(ocs.Dense(K), block)])


SYMMETRIC = {"soc": (cones.SecondOrderCone, (), 4), "socbatch": (cones.SOCBatch, (2, 2), 6),
             "psd": (cones.PSDCone, (3,), 9)}


def _block(kind):
    cls, args, rows = SYMMETRIC[kind]
    return cls(*args), rows


@pytest.mark.parametrize("kind", ["soc", "socbatch", "psd"])
@pytest.mark.parametrize("declared", ["automorphism", ("automorphism", "per_block"), ("automorphism", "term")])
def test_public_construct_automorphism_has_stable_arithmetic_record(kind, declared):
    block, rows = _block(kind)
    comp = _two_term(block, rows).compile(equilibrate="construct", max_blocks=2,
        tol=1e-3, metric_updates=4, metric_classes={0: declared})
    rec = comp.meta["metric_construction"]["automorphism_arithmetic"]
    assert rec["identity"] == "stable_soc_psd_bounded_v1"
    assert len(rec["eta_history"]["values"]) == 4
    assert comp.meta["scaling_classes"][0] == "automorphism"


def test_public_constructor_needs_no_private_switch():
    assert not hasattr(mc, "_RESEARCH_ONLY_AUTOMORPHISM_CONSTRUCTOR")
    assert not any("research" in k for k in inspect.signature(compile_module.compile_family).parameters)


def test_metric_initial_uses_the_stable_constructor():
    block, rows = _block("soc")
    seed = cm.ConeAutomorphism.soc(np.array([1.2, 0.3, 0.2, 0.1]))
    comp = _two_term(block, rows).compile(equilibrate="construct", max_blocks=2,
        tol=1e-3, metric_updates=4, metric_classes={0: "automorphism"}, metric_initial={0: seed})
    assert comp.meta["scaling_classes"][0] == "automorphism"


@pytest.mark.parametrize("block, rows, kwargs, needle", [
    (cones.ExpCone(), 3, dict(metric_classes={0: "automorphism"}), "is not a symmetric cone"),
    (ocs.NonNeg(), 6, dict(metric_classes={0: "automorphism"}), "defined only for a symmetric-cone block"),
    (cones.SecondOrderCone(), 4, dict(metric_classes={0: ("automorphism", "term", 4)}),
     "the automorphism tier takes no metric block size"),
    (cones.SecondOrderCone(), 4, dict(metric_classes={0: ("automorphism", "sideways")}), "sharing must be one of"),
    (cones.SecondOrderCone(), 4, dict(metric_classes={0: "automorphism"}, equilibrate="sgd"),
     "require equilibrate='construct'"),
])
def test_a_malformed_declaration_still_names_its_own_fault(block, rows, kwargs, needle):
    prob = _two_term(block, rows)
    kwargs = {"equilibrate": "construct", **kwargs}
    with pytest.raises(ValueError) as err:
        prob.compile(max_blocks=2, tol=1e-3, metric_updates=4, **kwargs)
    assert needle in str(err.value) and "research-only" not in str(err.value)


def test_the_other_classes_on_a_cone_are_unchanged():
    block, rows = _block("soc")
    prob = _two_term(block, rows)
    comp = prob.compile(equilibrate="construct", max_blocks=2, tol=1e-3, metric_updates=4,
                        metric_classes={0: "scalar"})
    assert comp.meta["scaling_classes"][0] != "automorphism"


def test_the_cvxpy_bridge_reaches_the_public_constructor():
    cp = pytest.importorskip("cvxpy")
    from ocs.cvxpy_mf.solve import compile_mf
    rng = np.random.default_rng(31)
    q, n, d = 3, 6, 3
    A = rng.normal(size=(d * q, n))
    x = cp.Variable(n)
    t = cp.Variable(q)
    problem = cp.Problem(cp.Minimize(rng.normal(size=n) @ x),
                         [cp.SOC(t, cp.reshape(A @ x, (d, q), order="F"), axis=0), t <= 1.0])
    plain, _, _ = compile_mf(problem, max_blocks=2, tol=1e-3)
    soc = [e["term"] for e in plain.meta["effective_metric"]["terms"] if e["block"] == "SOCBatch"]
    assert len(soc) == 1
    compiled, _, _ = compile_mf(problem, equilibrate="construct",
        metric_classes={soc[0]: "automorphism"}, max_blocks=2, tol=1e-3)
    assert compiled.meta["scaling_classes"][soc[0]] == "automorphism"


# ================================ the caller-supplied element, at the cap
def soc_at_cap(k, scale, rng, ratio=10.0 * AT_CAP):
    """w with lam1/lam2 = ratio (cond P(w) = ratio^2) and lam1 lam2 = scale^2."""
    lam2 = scale / np.sqrt(ratio)
    lam1 = ratio * lam2
    u = rng.standard_normal(k - 1)
    u /= np.linalg.norm(u)
    return np.r_[0.5 * (lam1 + lam2), 0.5 * (lam1 - lam2) * u]


def psd_at_cap(k, scale, rng, ratio=10.0 * AT_CAP):
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    lam = scale * np.geomspace(1.0 / np.sqrt(ratio), np.sqrt(ratio), k)
    S = (Q * lam) @ Q.T
    return 0.5 * (S + S.T)


def _host_eig(w):
    nb = np.linalg.norm(w[1:])
    return w[0] + nb, w[0] - nb


@pytest.mark.parametrize("k", [2, 3, 9, 1025])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_an_element_at_the_cap_is_accepted_and_just_above_it_is_refused(k, scale):
    rng = np.random.default_rng(40 + k)
    w = soc_at_cap(k, scale, rng)
    carrier = cm.ConeAutomorphism.soc(w)
    assert CAP * (1 - 1e-8) <= carrier.cond <= CAP
    with pytest.raises(ValueError, match="condition number at most 100"):
        cm.ConeAutomorphism.soc(soc_at_cap(k, scale, rng, ratio=10.0 * (1 + 1e-6)))


@pytest.mark.parametrize("k", [2, 3, 9, 1025])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_even_a_float32_subtraction_on_a_stored_element_at_the_cap_cannot_cancel(k, scale):
    """Reason 2. The float32 ``a0 - ||abar||`` of every stored device element
    (``w``, ``w^{1/2}``, ``w^{-1/2}``) against the exact float64 Jordan
    eigenvalue. The bound is the rounding of the stored entries (``eps32/2``
    times ``lam1``), the float32 sum of squares and square root (``(k/2 + 1)
    eps32`` relative on ``||abar||``) and the subtraction, relative to the
    small eigenvalue: ``(k/2 + 2) eps32 lam1/lam2``. The element whose
    reconstruction failed in the constructor had ``lam1/lam2 = e^{30.45}``
    and is refused by the cap."""
    rng = np.random.default_rng(50 + k)
    w = soc_at_cap(k, scale, rng)
    carrier = cm.ConeAutomorphism.soc(w)
    lam1, lam2 = _host_eig(w)
    exact = {"w": (lam1, lam2), "half": (np.sqrt(lam1), np.sqrt(lam2)),
             "invhalf": (1 / np.sqrt(lam2), 1 / np.sqrt(lam1))}
    for name, (big, small) in exact.items():
        a = getattr(carrier, name)[0]
        small32 = float(jax.jit(lambda v: v[0] - jnp.sqrt(jnp.sum(v[1:] * v[1:])))(a))
        bound = (k / 2 + 2) * EPS32 * (big / small)
        assert abs(small32 - small) <= bound * small, (name, small32, small, bound)
        assert big / small <= 10.0
    failing = soc_at_cap(k, 1.0, rng, ratio=float(np.exp(30.45)))
    with pytest.raises(ValueError, match="condition number at most 100"):
        cm.ConeAutomorphism.soc(failing)


@pytest.fixture
def no_device_jordan_spectral(monkeypatch):
    """Reason 1. Any device reconstruction of a Jordan eigenvalue raises."""
    def forbidden(name):
        def fn(*args, **kwargs):
            raise AssertionError(f"cone_metric.{name} was called on the caller-supplied route")
        return fn
    for name in ("soc_eig", "soc_spectral", "soc_det", "psd_spectral"):
        monkeypatch.setattr(cm, name, forbidden(name))


def test_control_the_forbidding_fixture_does_intercept_the_constructor(no_device_jordan_spectral, monkeypatch):
    """Negative control for reason 1: the research constructor DOES call the
    forbidden functions, so their absence from the supplied route is a finding
    and not a fixture that never fires."""
    block, rows = _block("soc")
    with pytest.raises(AssertionError, match="was called on the caller-supplied route"):
        _two_term(block, rows).compile(equilibrate="construct", max_blocks=2, tol=1e-3,
                                       metric_updates=4, metric_classes={0: "automorphism"})


def _ref_P(w):
    k = w.size
    J = -np.eye(k)
    J[0, 0] = 1.0
    return 2.0 * np.outer(w, w) - (w[0] ** 2 - w[1:] @ w[1:]) * J


def _ref_proj_soc(v):
    s, z = v[0], v[1:]
    nz = np.linalg.norm(z)
    if nz <= s:
        return v.copy()
    if nz <= -s:
        return np.zeros_like(v)
    a = 0.5 * (nz + s)
    return np.concatenate([[a], a * z / nz])


def _soc_kkt(Sigma, v, z):
    """max of dist(z, C), dist(Sigma (z - v), C) and complementarity, each normalized."""
    def defect(x):
        return max(0.0, np.linalg.norm(x[1:]) - x[0])
    y = Sigma @ (z - v)
    scale = np.linalg.norm(Sigma @ v) + 1e-300
    return max(defect(z) / scale, defect(y) / scale,
               abs(y @ z) / (scale * (np.linalg.norm(z) + 1e-300)))


@pytest.mark.parametrize("k", [3, 9, 65])
def test_the_supplied_route_never_reconstructs_and_meets_the_prox_bar_at_the_cap(k, no_device_jordan_spectral):
    """At the cap the float32 weighted projection, evaluated in float64, meets
    the declared proximal optimality bar ``PROX_KKT_BAR`` on inputs inside,
    below and across the cone, with the device spectral functions forbidden."""
    rng = np.random.default_rng(60 + k)
    w = soc_at_cap(k, 1.0, rng)
    carrier = cm.ConeAutomorphism.soc(w, layout=("soc",))
    Sigma = _ref_P(w)
    prox = jax.jit(lambda v: cm.prox_automorphism(carrier, v))
    worst = 0.0
    for trial in range(60):
        v = rng.standard_normal(k) * 10.0 ** rng.uniform(-3, 3)
        if trial % 3 == 1:
            v[0] = abs(v[0]) + np.linalg.norm(v[1:])            # inside
        z = np.asarray(prox(jnp.asarray(v, jnp.float32)), np.float64)
        assert np.all(np.isfinite(z))
        worst = max(worst, _soc_kkt(Sigma, v.astype(np.float32).astype(np.float64), z))
    print("worst float64 KKT defect of the float32 weighted projection at cond P(w) = 100:", worst)
    assert worst <= bm.PROX_KKT_BAR


def test_psd_supplied_route_never_reconstructs_at_the_cap(no_device_jordan_spectral):
    rng = np.random.default_rng(70)
    k = 3
    S = psd_at_cap(k, 1.0, rng)
    carrier = cm.ConeAutomorphism.psd(S, layout=("psd", k))
    assert CAP * (1 - 1e-8) <= carrier.cond <= CAP
    lam, U = np.linalg.eigh(S)
    half, invhalf = (U * np.sqrt(lam)) @ U.T, (U / np.sqrt(lam)) @ U.T
    prox = jax.jit(lambda v: cm.prox_automorphism(carrier, v))
    for _ in range(30):
        M = rng.standard_normal((k, k))
        M = 0.5 * (M + M.T)
        got = np.asarray(prox(jnp.asarray(M.reshape(-1), jnp.float32)), np.float64).reshape(k, k)
        Mh = half @ M @ half
        ev, V = np.linalg.eigh(Mh)
        ref = invhalf @ ((V * np.maximum(ev, 0.0)) @ V.T) @ invhalf
        assert np.all(np.isfinite(got))
        # float32 congruences and eigh around a cond(S) = 10 element
        assert np.linalg.norm(got - ref) <= 1e-4 * max(1.0, np.linalg.norm(ref))


def test_a_whole_compile_and_solve_through_block_metrics_at_the_cap_never_reconstructs(no_device_jordan_spectral):
    rng = np.random.default_rng(80)
    n, q, d = 8, 3, 2
    rows = q * (d + 1)
    K1 = rng.normal(size=(rows, n))
    K2 = rng.normal(size=(5, n))
    x = ocs.Variable(n)
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         Term(ocs.Dense(K1), cones.SOCBatch(q, d), lambda dd: dd["l1"]),
                         ocs.nonneg(ocs.Dense(K2) @ x + ocs.Parameter("l2"))], cost=rng.normal(size=n))
    W = np.stack([soc_at_cap(d + 1, 1.0, rng) for _ in range(q)])
    carrier = cm.ConeAutomorphism.soc(W)
    assert CAP * (1 - 1e-8) <= carrier.cond <= CAP
    tau, sigma = np.full(n, 0.01, "f"), np.full(rows + 5, 0.01, "f")
    comp = prob.compile(equilibrate=("given", tau, sigma, 0.5), block_metrics={0: carrier},
                        max_blocks=3, tol=1e-3)
    assert comp.meta["scaling_classes"][0] == "automorphism"
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1
    data = {"l1": jnp.asarray(rng.uniform(1.0, 2.0, rows), jnp.float32),
            "l2": jnp.asarray(rng.uniform(0.5, 1.5, 5), jnp.float32)}
    sol = comp.solve_instance(data)
    assert np.all(np.isfinite(np.asarray(sol.x))) and np.all(np.isfinite(np.asarray(sol.y)))
