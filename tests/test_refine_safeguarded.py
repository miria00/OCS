"""The safeguarded refinement never returns a worse assessed pair, refuses
invalid or worse proposals, and enforces every budget.  Small FFT lasso."""
import numpy as np
import jax.numpy as jnp
import pytest

import ocs
from ocs.solver.refine_safeguarded import (RefineBudget, safeguarded_refine, gauss_newton_proposal,
                                           LassoSupportCorrection, criterion7_assessor)

N, TOL = 1024, 1e-2


@pytest.fixture(scope="module")
def problem():
    K = ocs.FFTMask.random(N, ratio=0.25, seed=0)
    idx = np.asarray(K.idx); m = len(idx); lam = 1e-3
    rng = np.random.default_rng(1)
    xt = np.zeros(N); xt[rng.choice(N, N // 50, replace=False)] = rng.normal(size=N // 50)
    c = np.fft.fft(xt, norm="ortho")[idx]; b = np.r_[c.real, c.imag].astype(np.float32)
    v = ocs.Variable(N)
    comp = ocs.minimize([ocs.l1(v, lam), ocs.least_squares(ocs.FFTMask(idx, N) @ v - ocs.Parameter("y"))]).compile(
        tol=TOL, max_blocks=8)
    data = {"y": jnp.asarray(b)}
    s = comp.solve_instance(data)                       # a rough pair at tol 1e-2 and a tiny budget

    def forward(x):
        cc = np.fft.fft(np.asarray(x, np.float64), norm="ortho")[idx]
        return np.r_[cc.real, cc.imag]

    def adjoint(u):
        u = np.asarray(u, np.float64); f = np.zeros(N, np.complex128); f[idx] = u[:m] + 1j * u[m:]
        return np.fft.ifft(f, norm="ortho").real
    return dict(comp=comp, data=data, x=s.x, y=s.y, forward=forward, adjoint=adjoint, b=b.astype(np.float64), lam=lam)


def _score(pb, x, y):
    return criterion7_assessor(pb["comp"], pb["data"], 1e-6)(x, y)["score"]


def test_fixed_point_residual_is_public_and_refine_is_unchanged(problem):
    comp, data = problem["comp"], problem["data"]
    r = comp.fixed_point_residual(problem["x"], problem["y"], data)
    assert r.shape == (comp.n + comp.m,) and bool(jnp.all(jnp.isfinite(r)))
    xr, yr = comp.refine(problem["x"], problem["y"], data, steps=1, cg_iters=5)
    assert xr.shape == (comp.n,) and yr.shape == (comp.m,)


@pytest.mark.parametrize("kind", ["gn", "pcg", "lsqr"])
def test_step_never_worsens_the_assessed_pair(problem, kind):
    pb = problem; comp, data = pb["comp"], pb["data"]
    if kind == "gn":
        prop = gauss_newton_proposal(comp, data, cg_iters=30, max_line_search=6)
    else:
        prop = LassoSupportCorrection(pb["forward"], pb["adjoint"], pb["b"], pb["lam"], solver=kind, max_iters=100)
    xb, wb, ledger = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                        budget=RefineBudget(max_steps=3, cg_iters=30, max_assessments=8))
    assert ledger["final"]["score"] <= ledger["start"]["score"]
    # the returned pair IS the assessed pair: re-assessing it reproduces the ledger's final score
    assert _score(pb, xb, wb) == pytest.approx(ledger["final"]["score"], rel=1e-6, abs=1e-12)
    for row in ledger["steps"]:
        if row["decision"].startswith("accepted"):
            assert row["score_after"] < row["score_before"]
        elif "score_after" in row:
            assert not (row["valid"] and row["score_after"] < row["score_before"])
    assert ledger["products"] > 0 and ledger["assessments"] <= 8


def test_invalid_proposals_are_refused_and_the_start_pair_kept(problem):
    pb = problem; comp, data = pb["comp"], pb["data"]
    nan = lambda x, y, ledger: (jnp.full(comp.n, jnp.nan), y, {"kind": "nan", "products": 1})
    wrong_shape = lambda x, y, ledger: (x[:-1], y, {"kind": "shape", "products": 1})
    raises = lambda x, y, ledger: (_ for _ in ()).throw(RuntimeError("boom"))
    withdrawn = lambda x, y, ledger: (None, None, {"kind": "withdrawn", "products": 0})
    for prop in (nan, wrong_shape, raises, withdrawn):
        xb, wb, ledger = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                            budget=RefineBudget(max_steps=2))
        assert np.array_equal(np.asarray(xb), np.asarray(pb["x"]))
        assert ledger["final"]["score"] == ledger["start"]["score"] and not ledger["improved"]
        assert ledger["steps"][0]["decision"].startswith("refused")
        assert ledger["assessments"] == 1                   # an invalid proposal is never assessed
        assert ledger["stop_reason"] == "proposal refused"


def test_worse_proposal_is_refused(problem):
    pb = problem; comp, data = pb["comp"], pb["data"]
    zero = lambda x, y, ledger: (jnp.zeros_like(x), jnp.zeros_like(y), {"kind": "zero", "products": 0})
    xb, wb, ledger = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=zero, tol=1e-6,
                                        budget=RefineBudget(max_steps=3))
    row = ledger["steps"][0]
    assert row["decision"].startswith("refused") and ledger["assessments"] == 2
    assert np.array_equal(np.asarray(xb), np.asarray(pb["x"]))
    assert _score(pb, xb, wb) == pytest.approx(ledger["start"]["score"], rel=1e-6, abs=1e-12)


def test_budgets_are_enforced(problem):
    pb = problem; comp, data = pb["comp"], pb["data"]
    prop = gauss_newton_proposal(comp, data, cg_iters=10, max_line_search=3)
    _, _, l0 = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6, budget=RefineBudget(max_steps=0))
    assert l0["steps"] == [] and l0["assessments"] == 1 and l0["stop_reason"] == "max_steps"
    _, _, l1 = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                  budget=RefineBudget(max_steps=5, max_products=1))
    assert l1["steps"] == [] and l1["stop_reason"] == "max_products"
    _, _, l2 = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                  budget=RefineBudget(max_steps=5, max_assessments=1))
    assert l2["steps"] == [] and l2["stop_reason"] == "max_assessments"
    _, _, l3 = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                  budget=RefineBudget(max_steps=5, max_seconds=0.0))
    assert l3["steps"] == [] and l3["stop_reason"] == "max_seconds"
    _, _, l4 = safeguarded_refine(comp, pb["x"], pb["y"], data, proposals=prop, tol=1e-6,
                                  budget=RefineBudget(max_steps=5, max_assessments=3, cg_iters=10))
    assert l4["assessments"] <= 3 and len(l4["steps"]) <= 2


def test_lasso_support_diagnostics_and_empty_support_withdrawn(problem):
    pb = problem
    corr = LassoSupportCorrection(pb["forward"], pb["adjoint"], pb["b"], pb["lam"], solver="pcg")
    xn, yn, info = corr(pb["x"], pb["y"])
    for key in ("support_size", "sign_flips", "inactive_max_before", "inactive_max_after",
                "inactive_violations_after", "active_deviation_after", "support_valid", "linear_solve", "products"):
        assert key in info
    assert info["linear_solve"]["min_rayleigh"] is None or info["linear_solve"]["min_rayleigh"] > 0
    empty = LassoSupportCorrection(pb["forward"], pb["adjoint"], pb["b"], pb["lam"], threshold=1e9)
    assert empty(pb["x"], pb["y"])[0] is None and empty.rounds[-1]["support_valid"] is False
