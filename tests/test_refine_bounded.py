"""Bounded execution of the ``refine="auto"`` stage: every limit, small sizes.

Three exits are distinguished on the record and here: (1) an admission
refusal (``exit == "admission_refusal"``, nothing constructed or compiled,
the estimate and the limit named), (2) a bounded stop at a segment boundary
(``exit == "bounded_stop"``: the deadline, the operator budget, the memory
budget, the Krylov allowance), and (3) the external watchdog, which kills
the process and leaves a marker with no numerical verdict.  The previous
assessed packet is preserved byte for byte on every refusal and stop.

The deadline tests use an FFT mask whose FLOAT64 products sleep in a host
callback, standing in for an operator whose float64 application is slow
(the 262,144-point complex128 scatter measured at 2.9 s per adjoint on the
5090); its float32 products, which the solve and the assessment use, are
unchanged."""
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.solver import refine_auto
from ocs.solver.refine_auto import (RefineBudget, ROUTE_PRIMAL_DUAL_REFINEMENT, workspace_bytes,
                                    EXIT_ADMISSION_REFUSAL, EXIT_BOUNDED_STOP, EXIT_COMPLETED,
                                    EXIT_NO_STAGE, process_watchdog)

N, LAM = 1024, 1e-3
TOL_STALL, TOL_PASS = 1e-5, 1e-4
BLOCKS = 4000
DELAY = 0.05                     # seconds per float64 product of the slow operator
MAX_S = 4.0                      # the slow stage's deadline

PACKET_FIELDS = ("x", "y", "y_iterate", "iters", "residual", "status", "termination",
                 "accuracy_criterion", "objective_gap", "objective_lower_bound",
                 "objective_primal_value", "relative_primal_infeasibility", "accuracy_detail",
                 "dual_source", "objective_bound_source", "strict_certificate",
                 "contract_certificate", "contract", "certificate")


def _same_packet(a, b):
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


class SlowF64Mask(ocs.FFTMask):
    """An FFT mask whose float64 products sleep ``DELAY`` seconds in a host
    callback; float32 products are the plain FFT mask's."""

    def _slow(self, v):
        if v.dtype != jnp.float64:
            return v

        def cb(a):
            time.sleep(DELAY)
            return np.asarray(a)
        return jax.pure_callback(cb, jax.ShapeDtypeStruct(v.shape, v.dtype), v)

    def matvec(self, x):
        return super().matvec(self._slow(x))

    def rmatvec(self, u):
        return super().rmatvec(self._slow(u))


@pytest.fixture(scope="module")
def lasso():
    K = ocs.FFTMask.random(N, ratio=0.25, seed=0)
    idx = np.asarray(K.idx)

    def compile(tol=TOL_STALL, op=ocs.FFTMask, **kw):
        v = ocs.Variable(N)
        return ocs.minimize([ocs.l1(v, LAM), ocs.least_squares(op(idx, N) @ v - ocs.Parameter("y"))]
                            ).compile(tol=tol, max_blocks=kw.pop("max_blocks", BLOCKS), **kw)

    comp = compile()
    data = {"y": jnp.asarray(_fft_rhs(idx, 1))}
    base = comp.solve_instance(data)                      # the ordinary packet, policy off
    assert base.status == "inaccurate" and base.termination == "stalled", base.accuracy_detail
    return dict(comp=comp, data=data, base=base, idx=idx, compile=compile)


def _stage(lasso, budget=None, **kw):
    return lasso["compile"](refine="auto", refine_budget=budget, **kw).solve_instance(lasso["data"])


DENSE = np.ones(N, bool)                                  # the 46-minute case in miniature


# ---- (0) the budget itself ---------------------------------------------------
def test_budget_fields_are_validated():
    for bad in (dict(max_cg_iters=-1), dict(max_corrections=-1), dict(max_trials=1.5),
                dict(cg_chunk_iters=0), dict(segment_seconds=0.0), dict(segment_seconds=float("inf")),
                dict(max_seconds=float("nan")), dict(max_seconds=-1.0), dict(max_support_fraction=1.5),
                dict(max_extra_bytes=-1), dict(support_within_rows=1), dict(max_assessments=True)):
        with pytest.raises(ValueError):
            RefineBudget(**bad)
    RefineBudget(max_seconds=0.0)
    RefineBudget(max_seconds=float("inf"), max_extra_bytes=0, max_support_fraction=1.0)


# ---- (1) immediate return when the request already passes --------------------
def test_passing_pair_returns_before_any_budget_check(lasso):
    strict = RefineBudget(max_seconds=0.0, max_corrections=0, max_operator_applications=0,
                          max_support_fraction=0.0, max_extra_bytes=0)
    on = _stage(lasso, strict, tol=TOL_PASS)
    off = lasso["compile"](tol=TOL_PASS).solve_instance(lasso["data"])
    r = on.refinement
    assert on.status == "solved" and r["outcome"] == "not_needed" and r["exit"] == EXIT_NO_STAGE
    assert _zero_work(r["work"]) and r["work"]["seconds"] == 0.0
    assert "admission" not in r and "execution" not in r
    _same_packet(on, off)
    comp = lasso["compile"](tol=TOL_PASS, refine="auto", refine_budget=strict)
    direct = comp.refine_assessed(off.x, off.y_iterate, lasso["data"])
    assert direct.status == "solved" and direct.refinement["outcome"] == "not_needed"
    assert _zero_work(direct.refinement["work"])


# ---- (2) admission refusals: nothing constructed, nothing compiled ------------
def _admission_refused(r, name):
    assert r["exit"] == EXIT_ADMISSION_REFUSAL, r["outcome"]
    assert r["refusal"]["name"] == name and r["refusal"]["exit"] == EXIT_ADMISSION_REFUSAL
    assert "estimate" in r["refusal"] and "limit" in r["refusal"]
    assert _zero_work(r["work"])
    if "execution" in r:                                  # the route's admission ran
        assert r["execution"]["compiled_executables"] == 0 and r["execution"]["segments"] == 0
    return True


def test_dense_support_is_refused_up_front(lasso, no_hook):
    """The 46-minute case in miniature: a support of all N entries (> the 512
    rows) is refused before any product or compilation, by name."""
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    comp._refine_test_hook = {"support": DENSE}
    t = time.perf_counter()
    s = comp.solve_instance(data, refine="auto")
    r = s.refinement
    assert r["outcome"] == "skipped: ineligible (support_density)"
    _admission_refused(r, "support_density")
    assert r["refusal"]["estimate"] == 1.0 and r["refusal"]["limit"] == 0.5
    assert "support_exceeds_rows" in r["refusal"]["also_failed"]
    assert r["admission"]["products_before_admission"] == 0 and r["admission"]["admitted"] is False
    assert r["work"]["seconds"] < 1.0
    _same_packet(s, base)
    print("dense support refused in %.3f s (solve included)" % (time.perf_counter() - t))


def test_support_larger_than_the_rows_is_refused(lasso, no_hook):
    comp, data, base = lasso["comp"], lasso["data"], lasso["base"]
    over = np.zeros(N, bool)
    over[:comp.m + 1] = True                              # one more than the 512 rows
    loose = lasso["compile"](refine="auto", refine_budget=RefineBudget(max_support_fraction=1.0))
    loose._refine_test_hook = {"support": over}
    s = loose.solve_instance(data)
    r = s.refinement
    assert r["outcome"] == "skipped: ineligible (support_exceeds_rows)"
    _admission_refused(r, "support_exceeds_rows")
    assert r["refusal"]["estimate"] == comp.m + 1 and r["refusal"]["limit"] == comp.m
    _same_packet(s, base)


def test_workspace_over_the_extra_bytes_cap_is_refused(lasso):
    s = _stage(lasso, RefineBudget(max_extra_bytes=1))
    r = s.refinement
    assert r["outcome"] == "skipped: workspace does not fit the admitted memory budget"
    _admission_refused(r, "workspace_device")
    assert r["memory"]["admitted_bytes"] == 1 and "max_extra_bytes" in r["memory"]["admitted_from"]
    _same_packet(s, lasso["base"])


def test_operator_work_estimate_over_budget_is_refused(lasso):
    s = _stage(lasso, RefineBudget(max_operator_applications=10))
    r = s.refinement
    assert r["outcome"] == "skipped: ineligible (operator_applications)"
    _admission_refused(r, "operator_applications")
    assert r["refusal"]["limit"] == 10 and r["refusal"]["estimate"] > 10
    _same_packet(s, lasso["base"])


@pytest.mark.parametrize("field,name", [("max_corrections", "max_corrections"),
                                        ("max_assessments", "max_assessments")])
def test_zero_corrections_or_assessments_admit_no_stage(lasso, field, name):
    s = _stage(lasso, RefineBudget(**{field: 0}))
    r = s.refinement
    assert r["outcome"] == "skipped: ineligible (%s)" % name
    _admission_refused(r, name)
    _same_packet(s, lasso["base"])


# ---- (3) hard limits during execution -----------------------------------------
def test_krylov_chunks_do_not_change_the_arithmetic(lasso):
    """Chunks of one iteration and the default chunking return the same pair."""
    one = _stage(lasso, RefineBudget(cg_chunk_iters=1))
    many = _stage(lasso)
    for s in (one, many):
        assert s.status == "solved" and s.refinement["outcome"] == "accepted"
        assert s.refinement["exit"] == EXIT_COMPLETED
    l1 = one.refinement["candidates"][0]["proposal"]["linear_solve"]
    lm = many.refinement["candidates"][0]["proposal"]["linear_solve"]
    assert l1["iterations"] == lm["iterations"] and l1["stopped_by"] == "converged"
    assert l1["chunks"] == l1["iterations"] and l1["largest_chunk_iterations"] == 1
    assert lm["chunks"] < l1["chunks"]
    assert np.asarray(one.x).tobytes() == np.asarray(many.x).tobytes()
    assert np.asarray(one.y).tobytes() == np.asarray(many.y).tobytes()
    ex = one.refinement["execution"]
    assert ex["synchronizations"] >= l1["chunks"] and ex["by_segment"]["Krylov chunk"]["count"] == l1["chunks"]
    w = many.refinement["work"]
    assert w["operator_applications"] == (w["correction_operator_applications"]
                                          + w["assessment_operator_applications_upper_bound"] + 2)
    assert w["correction_operator_applications"] == 2 * w["cg_iterations"] + 5


def test_krylov_iteration_limit_stops_at_a_chunk_boundary(lasso):
    s = _stage(lasso, RefineBudget(max_cg_iters=3, cg_chunk_iters=2))
    r = s.refinement
    lin = r["candidates"][0]["proposal"]["linear_solve"]
    assert lin["iterations"] == 3 and lin["stopped_by"] == "max_cg_iters" and lin["chunks"] == 2
    assert r["exit"] == EXIT_BOUNDED_STOP and r["bounded_stop"]["limit"] == "max_cg_iters"
    assert r["work"]["assessments"] == 1                  # the truncated proposal is assessed
    if r["outcome"] != "accepted":
        _same_packet(s, lasso["base"])


def test_operator_budget_clamps_the_krylov_allowance(lasso):
    assessment = lasso["comp"]._objective_bound.operator_applications_bound(
        lasso["comp"]._bound_rounds)
    cap = 5 + assessment + 2 + 2 * 4      # room for 4 iterations after all guards
    s = _stage(lasso, RefineBudget(max_operator_applications=cap))
    r = s.refinement
    assert r["admission"]["planned"]["krylov_iteration_allowance"] == 4
    lin = r["candidates"][0]["proposal"]["linear_solve"]
    assert lin["iterations"] == 4 and lin["stopped_by"] == "max_operator_applications"
    assert r["exit"] == EXIT_BOUNDED_STOP and r["bounded_stop"]["limit"] == "max_operator_applications"
    assert r["work"]["operator_applications"] <= cap
    if r["outcome"] != "accepted":
        _same_packet(s, lasso["base"])


def test_legacy_operator_cap_is_not_overspent_after_guard_repair(lasso):
    # The historical cap 90 incorrectly admitted four iterations and spent 93
    # even before the sensitive arithmetic actions were charged.
    cap = 90
    s = _stage(lasso, RefineBudget(max_operator_applications=cap))
    r = s.refinement
    assert r["work"]["operator_applications"] <= cap
    if r["outcome"] != "accepted":
        _same_packet(s, lasso["base"])


def test_memory_watch_stops_when_the_stage_holds_more_than_admitted():
    watch = refine_auto._MemoryWatch(admitted=1024)
    if not watch.available:
        pytest.skip("the backend reports no memory_stats")
    big = jnp.zeros(1 << 20, jnp.float32).block_until_ready()
    with pytest.raises(refine_auto._Stop) as e:
        watch.check("test segment")
    assert e.value.limit == "max_extra_bytes" and watch.extra_max >= big.nbytes
    del big


def test_admitted_workspace_bounds_the_observed_device_bytes(lasso):
    est = workspace_bytes(ROUTE_PRIMAL_DUAL_REFINEMENT, N, lasso["comp"].m)
    s = _stage(lasso, RefineBudget(max_extra_bytes=est))
    r = s.refinement
    assert r["outcome"] == "accepted" and r["memory"]["admitted_bytes"] == est
    obs = r["memory"]["observed_extra_bytes_max"]
    assert obs is None or obs <= est, (obs, est)
    print("workspace estimate %d bytes, observed extra %s bytes" % (est, obs))


# ---- (4)/(5) the deadline, segment by segment ----------------------------------
@pytest.fixture(scope="module")
def slow(lasso):
    loose = dict(max_support_fraction=1.0, support_within_rows=False, segment_seconds=0.25)
    base = lasso["base"]
    comp = lasso["compile"](op=SlowF64Mask, refine="auto", refine_budget=RefineBudget(max_corrections=0))
    # An admission refusal on the same family: it compiles the family's own
    # evaluator (outside the timed calls below) and is the packet to keep.
    kept = comp.refine_assessed(base.x, base.y_iterate, lasso["data"])
    assert kept.refinement["exit"] == EXIT_ADMISSION_REFUSAL
    comp._refine_budget = RefineBudget(max_seconds=MAX_S, **loose)
    return dict(comp=comp, kept=kept, loose=loose)


def test_deadline_stops_a_slow_dense_stage_at_a_segment_boundary(lasso, slow):
    comp, base = slow["comp"], lasso["base"]
    comp._refine_test_hook = {"support": DENSE}
    t = time.perf_counter()
    s = comp.refine_assessed(base.x, base.y_iterate, lasso["data"])
    wall = time.perf_counter() - t
    comp._refine_test_hook = None
    r = s.refinement
    assert r["outcome"] == "refused: over budget (max_seconds) before the next proposal"
    assert r["exit"] == EXIT_BOUNDED_STOP and r["bounded_stop"]["limit"] == "max_seconds"
    assert r["bounded_stop"]["segment"] == "Krylov chunk"
    assert r["deadline"]["clock_started"].startswith("refine_assessed entry")
    ex = r["execution"]
    chunk = ex["by_segment"]["Krylov chunk"]
    assert chunk["count"] >= 2 and chunk["max_seconds"] < 1.0      # no compiled call near the deadline
    assert r["work"]["cg_iterations"] >= 2 and r["work"]["assessments"] == 0
    assert r["deadline"]["elapsed_seconds"] <= MAX_S + ex["longest_segment_seconds"] + 0.5
    assert wall < MAX_S + ex["longest_segment_seconds"] + 1.0
    _same_packet(s, slow["kept"])
    print("slow dense stage: stopped after %.2f s (%d Krylov chunks, longest segment %.3f s %r)"
          % (r["deadline"]["elapsed_seconds"], chunk["count"], ex["longest_segment_seconds"],
             ex["longest_segment"]))


def test_projected_work_over_the_deadline_is_refused_before_compiling(lasso, slow):
    """A later stage on the same family has measured product times: a
    deadline the smallest useful stage cannot meet is an admission refusal."""
    comp, base = slow["comp"], lasso["base"]
    assert "forward_seconds" in comp._refine_cache[("timing", ROUTE_PRIMAL_DUAL_REFINEMENT, N, comp.m)]
    saved = comp._refine_budget
    comp._refine_budget = RefineBudget(max_seconds=0.2, **slow["loose"])
    comp._refine_test_hook = {"support": DENSE}
    try:
        s = comp.refine_assessed(base.x, base.y_iterate, lasso["data"])
    finally:
        comp._refine_budget, comp._refine_test_hook = saved, None
    r = s.refinement
    assert r["outcome"] == "skipped: ineligible (projected_seconds)"
    _admission_refused(r, "projected_seconds")
    assert r["refusal"]["estimate"] > 0.2 and r["refusal"]["limit"] == 0.2
    assert r["admission"]["planned"]["projected_seconds_minimum"] >= 5 * DELAY
    _same_packet(s, slow["kept"])


def test_deadline_counts_from_the_requested_call_entry(lasso, slow):
    comp, base = slow["comp"], lasso["base"]
    saved = comp._refine_budget
    comp._refine_budget = RefineBudget(max_seconds=1e-9, **slow["loose"])
    try:
        s = comp.refine_assessed(base.x, base.y_iterate, lasso["data"])
    finally:
        comp._refine_budget = saved
    r = s.refinement
    assert r["deadline"]["seconds_before_stage_entry"] > 0
    assert r["exit"] in (EXIT_BOUNDED_STOP, EXIT_ADMISSION_REFUSAL)
    assert _zero_work(r["work"])
    _same_packet(s, slow["kept"])


# ---- (6) the external watchdog: process-level, no verdict invented -------------
def _child(code, tmp_path, timeout=300):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(code))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    t = time.perf_counter()
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout,
                       cwd=str(tmp_path), env=env)
    return p, time.perf_counter() - t


def test_watchdog_kills_the_process_and_leaves_no_verdict(tmp_path):
    marker = tmp_path / "wd.json"
    p, wall = _child("""
        import time
        from ocs.solver.refine_auto import process_watchdog
        with process_watchdog(1.0, marker_path=%r):
            print("ARMED", flush=True)
            time.sleep(60)
        print("NOT REACHED", flush=True)
    """ % str(marker), tmp_path)
    assert p.returncode == -9, (p.returncode, p.stderr[-2000:])
    assert "ARMED" in p.stdout and "NOT REACHED" not in p.stdout
    assert wall < 30
    rec = json.loads(marker.read_text())
    assert rec["exit"] == refine_auto.EXIT_EXTERNAL_WATCHDOG and rec["verdict"] is None
    assert rec["limit_seconds"] == 1.0


def test_watchdog_disarms_when_the_block_exits(tmp_path):
    marker = tmp_path / "wd.json"
    p, _ = _child("""
        import time
        from ocs.solver.refine_auto import process_watchdog
        with process_watchdog(1.0, marker_path=%r):
            pass
        time.sleep(2.5)
        print("RETURNED", flush=True)
    """ % str(marker), tmp_path)
    assert p.returncode == 0 and "RETURNED" in p.stdout, p.stderr[-2000:]
    assert not marker.exists()


def test_watchdog_terminates_a_stage_running_past_it(tmp_path):
    """A stage with a long deadline on the slow operator is cut by the
    watchdog: the process is killed, nothing is returned, the marker says so."""
    marker = tmp_path / "wd.json"
    out = tmp_path / "result.json"
    p, wall = _child("""
        import json, time
        import numpy as np, jax, jax.numpy as jnp, ocs
        from ocs.solver.refine_auto import RefineBudget, process_watchdog
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("trb", %r)
        trb = importlib.util.module_from_spec(spec); spec.loader.exec_module(trb)
        idx = np.asarray(ocs.FFTMask.random(trb.N, ratio=0.25, seed=0).idx)
        v = ocs.Variable(trb.N)
        comp = ocs.minimize([ocs.l1(v, trb.LAM), ocs.least_squares(trb.SlowF64Mask(idx, trb.N) @ v
                             - ocs.Parameter("y"))]).compile(
            tol=1e-5, max_blocks=4, refine="auto",
            refine_budget=RefineBudget(max_seconds=600.0, max_support_fraction=1.0, support_within_rows=False))
        data = {"y": jnp.asarray(trb._fft_rhs(idx, 1))}
        x = jnp.full(trb.N, 1e-3, jnp.float32); y = jnp.zeros(comp.m, jnp.float32)
        comp._refine_test_hook = {"support": np.ones(trb.N, bool)}
        with process_watchdog(8.0, marker_path=%r):
            print("CALLING", flush=True)
            s = comp.refine_assessed(x, y, data)
            json.dump({"outcome": s.refinement["outcome"]}, open(%r, "w"))
        print("RETURNED", flush=True)
    """ % (os.path.abspath(__file__), str(marker), str(out)), tmp_path)
    assert p.returncode == -9, (p.returncode, p.stdout[-1000:], p.stderr[-3000:])
    assert "CALLING" in p.stdout and "RETURNED" not in p.stdout
    assert not out.exists()                               # no numerical verdict was produced
    rec = json.loads(marker.read_text())
    assert rec["exit"] == "external_watchdog" and rec["verdict"] is None


@pytest.fixture
def no_hook(lasso):
    yield
    lasso["comp"]._refine_test_hook = None
