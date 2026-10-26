"""The bounded refinement policy, ``compile(refine="auto")``: every row of its
table, on the public return, with the ordinary packet preserved byte for byte
whenever the stage does not produce a pair that passes the unchanged
criterion (7).  Small subsampled-Fourier lasso (N = 1024), which at tol 1e-5
reaches its float32 fixed point and stops on the stall guard at 25,500
iterations with the request unmet, and the small legacy quadratic OT."""
import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.library import Logistic
from ocs.solver.refine_auto import (RefineBudget, ROUTE_PRIMAL_DUAL_REFINEMENT,
                                    ROUTE_ACTIVE_SUPPORT_SOLVER, WITNESS_RECOVERY, detect_route,
                                    workspace_bytes)

N, LAM = 1024, 1e-3
TOL_STALL, TOL_PASS = 1e-5, 1e-4
BLOCKS = 4000

PACKET_FIELDS = ("x", "y", "y_iterate", "iters", "residual", "status", "termination",
                 "accuracy_criterion", "objective_gap", "objective_lower_bound",
                 "objective_primal_value", "relative_primal_infeasibility", "accuracy_detail",
                 "dual_source", "objective_bound_source", "strict_certificate",
                 "contract_certificate", "contract", "certificate")


def _same_packet(a, b):
    """Every public field equal, arrays byte for byte; only ``refinement`` may differ."""
    for name in PACKET_FIELDS:
        va, vb = getattr(a, name), getattr(b, name)
        if hasattr(va, "shape") or hasattr(vb, "shape"):
            assert np.asarray(va).dtype == np.asarray(vb).dtype, name
            assert np.asarray(va).tobytes() == np.asarray(vb).tobytes(), name
        elif isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb):
            continue
        else:
            assert va == vb, (name, va, vb)
    assert a.gap_history == b.gap_history
    return True


def _zero_work(work):
    return all(work[k] == 0 for k in ("operator_applications", "cg_iterations", "assessments",
                                       "proposals", "pdhg_iterations"))


def _fft_rhs(idx, seed, k=20):
    rng = np.random.default_rng(seed)
    xt = np.zeros(N)
    xt[rng.choice(N, k, replace=False)] = rng.normal(size=k)
    c = np.fft.fft(xt, norm="ortho")[idx]
    return np.r_[c.real, c.imag].astype(np.float32)


@pytest.fixture(scope="module")
def lasso():
    K = ocs.FFTMask.random(N, ratio=0.25, seed=0)
    idx = np.asarray(K.idx)

    def compile(tol, **kw):
        v = ocs.Variable(N)
        return ocs.minimize([ocs.l1(v, LAM),
                             ocs.least_squares(ocs.FFTMask(idx, N) @ v - ocs.Parameter("y"))]
                            ).compile(tol=tol, max_blocks=kw.pop("max_blocks", BLOCKS), **kw)

    comp = compile(TOL_STALL)
    data = {"y": jnp.asarray(_fft_rhs(idx, 1))}
    base = comp.solve_instance(data)                      # the ordinary packet, policy off
    assert base.status == "inaccurate" and base.termination == "stalled", base.accuracy_detail
    assert base.refinement is None

    def f64_discrepancy(x, y):
        x = np.asarray(x, np.float64); u = np.asarray(y, np.float64); b = np.asarray(data["y"], np.float64)
        cc = np.fft.fft(x, norm="ortho")[idx]; Fx = np.r_[cc.real, cc.imag]
        J = 0.5 * np.sum((Fx - b) ** 2) + LAM * np.abs(x).sum()
        d = -b @ u - 0.5 * np.sum(u * u)
        f = np.zeros(N, np.complex128); f[idx] = u[:len(idx)] + 1j * u[len(idx):]
        return abs(J - d) / max(abs(J), abs(d)), float(np.max(np.abs(np.fft.ifft(f, norm="ortho").real)))
    return dict(comp=comp, data=data, base=base, idx=idx, compile=compile, f64=f64_discrepancy)


@pytest.fixture
def no_hook(lasso):
    yield
    lasso["comp"]._refine_test_hook = None


# ---- the option -----------------------------------------------------------
def test_option_validation_and_default_off(lasso):
    assert lasso["comp"]._refine_policy is False
    with pytest.raises(ValueError, match="ambiguous"):
        lasso["compile"](TOL_PASS, refine=True)
    with pytest.raises(ValueError, match="refine must be"):
        lasso["compile"](TOL_PASS, refine="always")
    with pytest.raises(ValueError, match="refine_memory_bytes"):
        lasso["compile"](TOL_PASS, refine="auto", refine_memory_bytes=-1)
    with pytest.raises(TypeError):
        lasso["compile"](TOL_PASS, refine="auto", refine_budget={"max_cg_iters": 3})
    with pytest.raises(ValueError, match="refine must be"):
        lasso["comp"].solve_instance(lasso["data"], refine="yes")
    v = ocs.Variable(8)
    with pytest.raises(ValueError, match="PDHG"):
        ocs.minimize([ocs.least_squares(ocs.Dense(jnp.eye(8)) @ v - ocs.Parameter("b"))]).compile(
            method="cg", refine="auto")


def test_route_detection_names_the_two_structures(lasso):
    route, why = detect_route(lasso["comp"]._refine_structure)
    assert route == ROUTE_PRIMAL_DUAL_REFINEMENT and why is None
    n = 16
    fam = ProblemFamily(n * n, [Term(ocs.Identity(n * n), ocs.NonNeg().with_linear_quadratic(e=0.1)),
                                Term(ocs.MarginalOperator(n, n), ocs.Equality(), lambda d: -d["m"])],
                        cost_fn=lambda d: d["c"])
    comp = fam.compile(max_blocks=1, pdhg_variant="standard", primal_weight=30.)
    assert detect_route(comp._refine_structure) == (ROUTE_ACTIVE_SUPPORT_SOLVER, None)
    v = ocs.Variable(N)
    other = ocs.minimize([ocs.l2(v, LAM), ocs.least_squares(ocs.FFTMask(lasso["idx"], N) @ v - ocs.Parameter("y"))]
                         ).compile(tol=TOL_STALL, max_blocks=1)
    route, why = detect_route(other._refine_structure)
    assert route is None and "L2" in why and "SumSquares" in why
    assert workspace_bytes(ROUTE_PRIMAL_DUAL_REFINEMENT, N, 512) > 0
    assert len({ROUTE_PRIMAL_DUAL_REFINEMENT, ROUTE_ACTIVE_SUPPORT_SOLVER, WITNESS_RECOVERY}) == 3


# ---- row 1: already passing -----------------------------------------------
def test_already_passing_call_does_zero_refinement_work(lasso):
    off = lasso["compile"](TOL_PASS).solve_instance(lasso["data"])
    on = lasso["compile"](TOL_PASS, refine="auto").solve_instance(lasso["data"])
    assert off.status == "solved" and off.refinement is None
    assert on.status == "solved"
    r = on.refinement
    assert r["policy"] == "auto" and r["outcome"] == "not_needed" and r["route"] is None
    assert _zero_work(r["work"]) and r["work"]["seconds"] == 0.0
    assert r["started_from"]["iterations"] == off.iters and r["started_from"]["status"] == "solved"
    _same_packet(on, off)


# ---- row 2: assessment unavailable ----------------------------------------
def test_missing_contract_family_reports_the_capability_and_does_no_work():
    class Unspecified(Logistic):
        bound_capable = False
    fam = ProblemFamily(2, [Term(ocs.Diagonal(jnp.ones(2)), Unspecified(), lambda d: d["b"])])
    comp = fam.compile(max_blocks=1000, refine="auto")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported request dispatched solver or refinement work")
    comp._single_solver = comp._single_cold_solver = comp._cold_solver = comp._run = forbidden
    comp._drive = comp._single_drive = forbidden
    comp._refine_test_hook = {"proposals": lambda x, y: forbidden()}
    single = comp.solve_instance({"b": jnp.zeros(2)})
    batch = comp.solve_many_results({"b": jnp.zeros((2, 2))})
    for sol in [single] + batch:
        assert sol.status == "unverified" and sol.termination == "unsupported" and sol.iters == 0
        r = sol.refinement
        assert r["outcome"] == "unavailable" and r["route"] is None
        assert "Unspecified" in r["capability_missing"]
        assert _zero_work(r["work"])
    refined = comp.refine_assessed(jnp.zeros(2), jnp.zeros(2), {"b": jnp.zeros(2)})
    assert refined.refinement["outcome"] == "unavailable" and refined.iters == 0


# ---- row 3 and 5: the fixed trigger and a passing refined pair -------------
def test_stall_triggers_one_stage_whose_refined_pair_passes(lasso):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    s = comp.solve_instance(data, refine="auto")
    assert s.status == "solved" and s.termination == "criterion" and s.accuracy_criterion == "objective_gap"
    assert s.iters == base.iters                           # zero PDHG iterations were added
    assert abs(s.objective_gap) <= TOL_STALL and s.relative_primal_infeasibility <= TOL_STALL
    r = s.refinement
    assert r["outcome"] == "accepted" and r["route"] == ROUTE_PRIMAL_DUAL_REFINEMENT
    assert r["trigger"] == "stall_guard" and r["started_from"]["stall_guard_fired"] is True
    st = r["started_from"]
    assert st["iterations"] == base.iters and st["status"] == "inaccurate" and st["termination"] == "stalled"
    assert st["objective_gap"] == base.objective_gap and st["witness_source"] == base.objective_bound_source
    w = r["work"]
    assert w["cg_iterations"] > 0 and w["assessments"] == 1 and w["proposals"] == 1
    assert w["pdhg_iterations"] == 0 and w["seconds"] > 0 and w["arithmetic"] == "float64"
    assert w["operator_applications"] == (w["correction_operator_applications"]
                                          + w["assessment_operator_applications_upper_bound"] + 2)
    assert w["correction_operator_applications"] >= 2 * w["cg_iterations"] + 4
    cand = r["candidates"][0]
    assert cand["decision"].startswith("accepted") and cand["proposal"]["support_valid_f64"] is True
    assert cand["proposal"]["sign_flips"] == 0 and cand["proposal"]["support_size"] == int(jnp.count_nonzero(base.x))
    assert r["assessment"]["witness_source"] == s.objective_bound_source
    # the returned pair IS the assessed pair: the public evaluator with recovery OFF
    # re-derives the award, and an independent float64 same-pair check agrees
    report = comp.objective_bound(s.x, s.y, data, recover=False)
    acc = comp.acceptance(report)
    assert acc["target_met"] and acc["primal_feasible"]
    disc, adjoint_peak = lasso["f64"](s.x, s.y)
    assert disc <= TOL_STALL and adjoint_peak <= LAM * (1 + 1e-6)
    assert int(jnp.count_nonzero(s.x)) == int(jnp.count_nonzero(base.x))   # the support was kept
    # one stage per solve: the record is one stage with one proposal, and the
    # refined pair, assessed as returned, needs no further work
    assert r["work"]["proposals"] == 1 and len(r["candidates"]) == 1
    direct = comp.refine_assessed(s.x, s.y_iterate, data)
    assert direct.status == "solved" and direct.refinement["outcome"] == "not_needed"
    # A warm-started CONTINUATION is a new solve: the drive executable repairs
    # the iterate dual before its first check, that check assesses the repaired
    # pair (not the returned one), and the run may stall and refine again; what
    # the policy guarantees is one stage within that solve.
    again = comp.solve_instance(data, warm_start=(s.x, s.y_iterate), refine="auto")
    assert again.status == "solved" and again.refinement["work"]["proposals"] <= 1


def test_requested_stage_on_a_supplied_pair(lasso):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    s = comp.refine_assessed(base.x, base.y_iterate, data)
    assert s.status == "solved" and s.iters == 0
    assert s.refinement["outcome"] == "accepted" and s.refinement["trigger"] == "requested"
    assert s.refinement["started_from"]["iterations"] == 0


# ---- row 6: invalid, nonfinite, unsuccessful, over budget -> rollback -------
def test_failed_correction_rolls_back_byte_for_byte(lasso, no_hook):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    comp._refine_test_hook = {"proposals": lambda x, y: [
        (jnp.zeros_like(x), jnp.zeros_like(y), {"kind": "zero", "products": 3, "cg_iterations": 0})]}
    s = comp.solve_instance(data, refine="auto")
    r = s.refinement
    assert r["outcome"] == "refused: no proposal passed criterion (7)"
    assert r["candidates"][0]["decision"].startswith("refused")
    assert r["work"]["assessments"] == 1 and r["work"]["proposals"] == 1
    assert r["work"]["correction_operator_applications"] == 3
    assert r["started_from"]["iterations"] == base.iters
    _same_packet(s, base)


def test_wrong_support_guess_is_refused_and_the_packet_kept(lasso, no_hook):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    wrong = np.roll(np.asarray(base.x) != 0, 7)           # same size, wrong indices
    assert wrong.sum() == int(jnp.count_nonzero(base.x))
    comp._refine_test_hook = {"support": wrong}
    s = comp.solve_instance(data, refine="auto")
    r = s.refinement
    assert r["outcome"].startswith("refused")
    row = r["candidates"][0]
    assert row["proposal"]["support_source"].startswith("override")
    assert row["proposal"]["support_valid_f64"] is False or row["decision"].startswith("refused")
    assert row["decision"].startswith("refused")
    _same_packet(s, base)


def test_nonfinite_or_malformed_corrections_are_refused_without_assessment(lasso, no_hook):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    nan = lambda x, y: [(jnp.full_like(x, jnp.nan), y, {"kind": "nan", "products": 1})]
    inf = lambda x, y: [(x, jnp.full_like(y, jnp.inf), {"kind": "inf", "products": 1})]
    shape = lambda x, y: [(x[:-1], y, {"kind": "shape", "products": 1})]
    none = lambda x, y: [(None, None, {"kind": "withdrawn", "products": 0})]
    empty = lambda x, y: []
    raises = lambda x, y: (_ for _ in ()).throw(RuntimeError("boom"))
    for name, hook in (("nan", nan), ("inf", inf), ("shape", shape), ("none", none),
                       ("empty", empty), ("raises", raises)):
        comp._refine_test_hook = {"proposals": hook}
        s = comp.solve_instance(data, refine="auto")
        r = s.refinement
        assert r["outcome"].startswith("refused"), name
        assert r["work"]["assessments"] == 0, name
        if name in ("nan", "inf", "shape", "none"):
            assert r["candidates"][0]["decision"].startswith("refused: nonfinite or invalid"), name
        if name == "raises":
            assert "RuntimeError" in r["outcome"]
        if name == "empty":
            assert "withdrawn" in r["outcome"]
        _same_packet(s, base)


def test_over_budget_corrections_are_refused(lasso, no_hook):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    # (a) a correction that over-spends operator applications is not assessed
    comp._refine_test_hook = {"proposals": lambda x, y: [
        (x, y, {"kind": "spent", "products": 10 ** 9}), (x, y, {"kind": "never", "products": 0})]}
    s = comp.solve_instance(data, refine="auto")
    r = s.refinement
    assert r["outcome"] == "refused: over budget (max_operator_applications)"
    assert r["candidates"][0]["decision"] == "not assessed: over budget (max_operator_applications)"
    assert r["work"]["assessments"] == 0 and len(r["candidates"]) == 1
    _same_packet(s, base)
    comp._refine_test_hook = None
    # (b) a wall-clock budget of zero stops the stage before any proposal
    zero = lasso["compile"](TOL_STALL, refine="auto", refine_budget=RefineBudget(max_seconds=0.0))
    s = zero.solve_instance(data)
    r = s.refinement
    assert r["outcome"] == "refused: over budget (max_seconds) before the next proposal"
    assert _zero_work(r["work"]) and r["budget"]["max_seconds"] == 0.0
    _same_packet(s, base)
    # (c) one CG iteration is not enough: the assessed proposal fails and is refused
    short = lasso["compile"](TOL_STALL, refine="auto", refine_budget=RefineBudget(max_cg_iters=1))
    s = short.solve_instance(data)
    r = s.refinement
    assert r["outcome"].startswith("refused") and r["work"]["cg_iterations"] == 1
    assert r["candidates"][0]["proposal"]["linear_solve"]["converged"] is False
    _same_packet(s, base)


# ---- row 4: memory admission ----------------------------------------------
def test_memory_admission_refusal_preserves_the_ordinary_path(lasso):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    tight = lasso["compile"](TOL_STALL, refine="auto", refine_memory_bytes=1)
    s = tight.solve_instance(data)
    r = s.refinement
    assert r["outcome"] == "skipped: workspace does not fit the admitted memory budget"
    assert r["memory"]["admitted"] is False and r["memory"]["admitted_bytes"] == 1
    assert r["memory"]["workspace_bytes_estimate"] == workspace_bytes(ROUTE_PRIMAL_DUAL_REFINEMENT, N, comp.m)
    assert r["route"] == ROUTE_PRIMAL_DUAL_REFINEMENT and _zero_work(r["work"])
    _same_packet(s, base)
    roomy = lasso["compile"](TOL_STALL, refine="auto",
                             refine_memory_bytes=workspace_bytes(ROUTE_PRIMAL_DUAL_REFINEMENT, N, comp.m))
    assert roomy.solve_instance(data).refinement["outcome"] == "accepted"


# ---- the trigger is the stall guard, nothing else --------------------------
def test_budget_exit_does_not_trigger_refinement(lasso):
    data = lasso["data"]
    off = lasso["compile"](TOL_STALL, max_blocks=4).solve_instance(data)
    on = lasso["compile"](TOL_STALL, max_blocks=4, refine="auto").solve_instance(data)
    assert off.termination == "iteration_limit" and off.status == "inaccurate"
    r = on.refinement
    assert r["outcome"] == "not_triggered" and "iteration_limit" in r["reason"]
    assert _zero_work(r["work"])
    _same_packet(on, off)


def test_residual_path_never_refines(lasso):
    s = lasso["comp"].solve_instance(lasso["data"], accuracy="residual", refine="auto")
    assert s.accuracy_criterion == "fixed_point" and s.refinement is None


def test_unsupported_structure_is_reported_without_work(lasso):
    v = ocs.Variable(N)
    other = ocs.minimize([ocs.l2(v, LAM), ocs.least_squares(ocs.FFTMask(lasso["idx"], N) @ v - ocs.Parameter("y"))]
                         ).compile(tol=TOL_STALL, max_blocks=1, refine="auto")
    s = other.refine_assessed(jnp.zeros(N), jnp.zeros(other.m), lasso["data"])
    assert s.status != "solved"
    r = s.refinement
    assert r["outcome"] == "unsupported" and "L2" in r["reason"] and _zero_work(r["work"])


# ---- batches ----------------------------------------------------------------
def test_batch_lanes_refine_independently(lasso, no_hook):
    comp, idx = lasso["comp"], lasso["idx"]
    batch = {"y": jnp.asarray(np.stack([_fft_rhs(idx, seed) for seed in (1, 2, 3)]))}
    comp.solve_many_results(batch)                        # warm the batched executables
    t = time.perf_counter(); off = comp.solve_many_results(batch); t_off = time.perf_counter() - t
    assert all(s.refinement is None for s in off)
    stalled = [i for i, s in enumerate(off) if s.termination == "stalled" and s.status == "inaccurate"]
    assert len(stalled) >= 2, [(s.status, s.termination) for s in off]
    # (1) a failing lane and a slow lane: no other lane's packet moves
    calls = []

    def hook(x, y):
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("lane failure")
        time.sleep(1.0)                                   # a slow refining lane
        return [(jnp.zeros_like(x), jnp.zeros_like(y), {"kind": "slow zero", "products": 1})]
    comp._refine_test_hook = {"proposals": hook}
    t = time.perf_counter(); on = comp.solve_many_results(batch, refine="auto"); t_on = time.perf_counter() - t
    assert len(calls) == len(stalled)
    for i, (a, b) in enumerate(zip(on, off)):
        _same_packet(a, b)
        r = a.refinement
        assert r["batch_lane"] == i
        if i in stalled:
            assert r["outcome"].startswith("refused")
        else:
            assert r["outcome"] in ("not_needed", "not_triggered")
    # the slow lane's wall time lands on the whole call's return (the rows come
    # back together), and is recorded per lane; both calls are warm here
    slow = [on[i].refinement["work"]["seconds"] for i in stalled[1:]]
    assert all(sec >= 1.0 for sec in slow)
    assert t_on - t_off >= 0.5 * sum(slow)              # a measurement, with slack for noise
    print("batch wall: off %.2fs, auto with one failing and %d slow lane(s) %.2fs" % (t_off, len(slow), t_on))
    comp._refine_test_hook = None
    # (2) the real stage, lane by lane
    t = time.perf_counter(); real = comp.solve_many_results(batch, refine="auto"); t_real = time.perf_counter() - t
    for i, (a, b) in enumerate(zip(real, off)):
        if i in stalled:
            assert a.status == "solved" and a.refinement["outcome"] == "accepted"
            assert a.iters == b.iters and a.refinement["started_from"]["iterations"] == b.iters
            assert comp.acceptance(comp.objective_bound(a.x, a.y, {"y": batch["y"][i]}, recover=False))["target_met"]
        else:
            _same_packet(a, b)
    print("batch wall: off %.2fs, auto refining %d lanes %.2fs" % (t_off, len(stalled), t_real))


# ---- the active-support route on the small legacy quadratic OT -------------
def _ot(n, seed=0):
    rng = np.random.default_rng(seed)
    u = np.sort(rng.uniform(0, 1, n)).astype(np.float32); v = np.sort(rng.uniform(0, 1, n)).astype(np.float32)
    c = (u[:, None] - v[None, :]) ** 2

    def dyadic(a):
        z = np.asarray(a, np.float64); z = z / z.sum() * 2 ** 20
        k = np.floor(z).astype(np.int64); k[np.argsort(-(z - k))[:2 ** 20 - int(k.sum())]] += 1
        return (k / 2 ** 20).astype(np.float32)
    p = rng.uniform(.2, 1, n).astype(np.float32); p /= p.sum()
    q = rng.uniform(.2, 1, n).astype(np.float32); q /= q.sum()
    return c, dyadic(p), dyadic(q)


@pytest.fixture(scope="module")
def ot():
    n = 256
    c, p, q = _ot(n)
    fam = ProblemFamily(n * n, [Term(ocs.Identity(n * n), ocs.NonNeg().with_linear_quadratic(e=np.float32(.1))),
                                Term(ocs.MarginalOperator(n, n), ocs.Equality(), lambda d: -d["m"])],
                        cost_fn=lambda d: d["c"])
    kw = dict(equilibrate="sgd", block=100, adaptive_omega=False, primal_weight=30.,
              pdhg_variant="standard", loop_unroll=1)
    data = {"c": jnp.asarray(c.ravel()), "m": jnp.asarray(np.r_[p, q])}
    return dict(n=n, fam=fam, kw=kw, data=data, c=c, p=p, q=q)


def _ot_f64(ot, x, y):
    n, mu = ot["n"], float(np.float32(.1))
    C = np.asarray(ot["c"], np.float64); X = np.asarray(x, np.float64).reshape(n, n); Y = np.asarray(y, np.float64)
    p, q = np.asarray(ot["p"], np.float64), np.asarray(ot["q"], np.float64)
    R = -Y[:n, None] - Y[None, n:] - C
    J = np.sum(C * X) + .5 * mu * np.sum(X * X)
    d = -(p @ Y[:n] + q @ Y[n:]) - np.sum(np.maximum(R, 0) ** 2) / (2 * mu)
    feas = np.linalg.norm(np.r_[X.sum(1) - p, X.sum(0) - q]) / np.linalg.norm(np.r_[p, q])
    return abs(J - d) / max(abs(J), abs(d)), feas


def test_active_support_route_passes_when_solved_without_it(ot):
    comp = ot["fam"].compile(tol=1e-3, max_blocks=4000, refine="auto", **ot["kw"])
    s = comp.solve_instance(ot["data"])
    assert s.status == "solved" and s.refinement["outcome"] == "not_needed"


def test_active_support_stage_from_a_coarse_solve_meets_a_finer_request(ot):
    """The route needs a settled support: the pair that passes at 1e-3 (the
    legacy acceptance) is handed to the stage at a finer request, where one
    bounded correction must pass or the packet must be preserved."""
    coarse = ot["fam"].compile(tol=1e-3, max_blocks=4000, **ot["kw"]).solve_instance(ot["data"])
    assert coarse.status == "solved"
    tol = 1e-4
    comp = ot["fam"].compile(tol=tol, max_blocks=4000, refine="auto", **ot["kw"])
    s = comp.refine_assessed(coarse.x, coarse.y_iterate, ot["data"])
    r = s.refinement
    assert r["route"] == ROUTE_ACTIVE_SUPPORT_SOLVER and r["trigger"] == "requested" and s.iters == 0
    assert r["started_from"]["status"] == "inaccurate"
    assert r["outcome"] == "accepted", r["outcome"]
    assert s.status == "solved" and s.termination == "criterion"
    w = r["work"]
    assert w["cg_iterations"] > 0 and w["arithmetic"] == "float64" and w["pdhg_iterations"] == 0
    prop = r["candidates"][-1]["proposal"]
    assert prop["nullspace"]["n_components"] >= 1 and prop["candidate"] in ("restricted", "full")
    disc, feas = _ot_f64(ot, s.x, s.y)
    assert disc <= tol and feas <= tol
    off = ot["fam"].compile(tol=tol, max_blocks=1, **ot["kw"])
    assert off.acceptance(off.objective_bound(s.x, s.y, ot["data"], recover=False))["target_met"]


def test_active_support_stage_from_a_rough_pair_is_refused_and_kept(ot):
    rough = ot["fam"].compile(tol=1e-3, max_blocks=20, **ot["kw"]).solve_instance(ot["data"])
    assert rough.status == "inaccurate"
    comp = ot["fam"].compile(tol=1e-3, max_blocks=20, refine="auto", **ot["kw"])
    s = comp.refine_assessed(rough.x, rough.y_iterate, ot["data"])
    r = s.refinement
    assert r["route"] == ROUTE_ACTIVE_SUPPORT_SOLVER
    if r["outcome"] == "accepted":
        assert s.status == "solved"
        return
    assert r["outcome"].startswith("refused") and r["work"]["assessments"] >= 1
    assert s.status == "inaccurate" and s.iters == 0
    assert np.asarray(s.x).tobytes() == np.asarray(rough.x).tobytes()
