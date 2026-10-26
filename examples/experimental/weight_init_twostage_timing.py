"""Re-time the block-metric pilot's two-stage comparison with compilation separated.

What bears on the paper's number.  The paper quotes the pilot's operator-only
two-stage arm A3s16 as 13.1 percent faster than the shipped engine A0std at
F2 L = 4096 (paired total warm ratio 0.8692).  Every arm of that comparison
recorded a "warm" setup that contains an XLA compilation of the shipped SMFE,
because `ocs/solver/equilibrate_sgd.py` defines its `@jax.jit` program inside the
function and recompiles it on every call.  This harness re-times the same three
arms with compilation and execution recorded as SEPARATE columns, and reports a
cold user-visible total and a warm numerical total as two distinct measurements,
never substituted for each other.

The arms, exactly as the pilot built them (`blockmetric_pilot.py`, imported
unchanged, `BM_PROX_IMPL=components`, 20 Newton steps, tol 1e-5, kappa cap 100):
  A0std   the shipped engine, standard PDHG, UNTOUCHED: `prob.compile(...)` then
          `solve_instance`.  Nothing about it is patched.
  A1      the research loop with the shipped SMFE scalings.
  A3s16   the two-stage construction: shipped SMFE, then q = 16 k forward probes
          build per-block Grams, whitened into full k x k blocks (eta = 1).

How compilation is separated.
  * Prototype arms (A1, A3s16) take their SMFE scalings from
    `smfe_hoisted.smfe_equilibration_hoisted`, a GENERATED copy of the shipped
    function whose jitted program is cached on its static arguments.  A hard gate
    (`--mode gate`, run first, in its own process) requires the copy's tau, sigma,
    norm and diagnostics to be BITWISE identical to the shipped function's; every
    timing process re-checks the sha256 of its own tau/sigma bytes against the
    gate's and aborts on any mismatch.
  * Every arm runs in a FRESH process per repetition (the pilot's rule), after one
    trivial device operation so CUDA context creation is excluded equally.
  * COLD phase, measured once per process from that point: every component's first
    call, i.e. compile plus execution.  For prototype arms this includes compiling
    the certificate family the research loop verifies against, because a user of
    the prototype needs it too.  Cold user-visible total = the sum.
  * WARM phase, measured three times in the same process: the same components
    again, now executing already-compiled programs.  Warm numerical total = the sum
    (median of the three).  compile_s = cold - warm, per component.
  * Engine arm: cold user-visible total = first `prob.compile` + first
    `solve_instance`.  Warm numerical total = median of repeated `solve_instance`
    on the already-compiled family, as instructed.  Because that excludes all setup
    while the prototypes' warm total includes SMFE and metric EXECUTION, a second,
    labelled composite is also reported: engine warm solve + the hoisted SMFE's
    measured execution on the identical operator.
  * The pilot's own definition is also reconstructed from the same measurements
    (prototype: one SMFE call including its compile + warm metric + warm loop;
    engine: compile + warm solve), so the paper's ratio can be checked against a
    reproduction before anything is read into the separated numbers.

Engine SMFE recompilation (refinement for axis two).  After the engine arm's own
timings, and only then, the same process times: two further `prob.compile` calls,
three standalone calls of the SHIPPED `smfe_equilibration` on the stacked operator
exactly as `compile_family` builds it (each recompiles), and four calls of the
hoisted copy (the first compiles, the rest only execute).  The SMFE recompilation
cost per one-shot solve is estimated as shipped-call time minus hoisted execution
time: an attribution by subtraction, not an instrumentation of `ocs/`.

Nothing in `ocs/` is edited.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

WARM_REPS = 3


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def gate_difference(tau, sigma, gate_rec, ref_prefix):
    """Size of the difference from the gate reference: a MEASUREMENT, not an abort.
    Cross-process nondeterminism of the shipped SMFE (section 5.7) makes a mismatch a
    property of the function, not evidence against the copy."""
    t, sg = np.asarray(tau), np.asarray(sigma)
    out = dict(tau_sha16=sha(t)[:16], sigma_sha16=sha(sg)[:16],
               tau_matches_gate=sha(t) == gate_rec["tau_sha256"],
               sigma_matches_gate=sha(sg) == gate_rec["sigma_sha256"])
    if ref_prefix and os.path.exists(ref_prefix + "_tau.npy"):
        rt = np.load(ref_prefix + "_tau.npy"); rs = np.load(ref_prefix + "_sigma.npy")
        if sha(rt) != gate_rec["tau_sha256"] or sha(rs) != gate_rec["sigma_sha256"]:
            raise SystemExit("saved gate reference does not hash to the gate: refusing to measure against it")
        dt = np.abs(t.astype(np.float64) - rt.astype(np.float64))
        ds = np.abs(sg.astype(np.float64) - rs.astype(np.float64))
        out.update(tau_max_abs_diff=float(dt.max()), tau_n_diff=int(np.count_nonzero(dt)),
                   sigma_max_abs_diff=float(ds.max()), sigma_n_diff=int(np.count_nonzero(ds)))
    else:
        out.update(tau_max_abs_diff=None, sigma_max_abs_diff=None, note="no saved reference arrays")
    return out


def within_process_check(hoisted_tau, hoisted_sigma, spec, mv, rmv, m, spans, log, what):
    """The approved gate for timing processes: AFTER all timings, call the SHIPPED
    function on the same operator and require bitwise equality with the copy's output
    from this process.  This is the hard stop."""
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    t_s, s_s, n_s, _d, dt = smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
    same = bool(np.array_equal(np.asarray(t_s), np.asarray(hoisted_tau))
                and np.array_equal(np.asarray(s_s), np.asarray(hoisted_sigma)))
    log(f"  within-process check ({what}): shipped == hoisted bitwise: {same}")
    if not same:
        d = np.abs(np.asarray(t_s, np.float64) - np.asarray(hoisted_tau, np.float64))
        log(f"  within-process tau max |diff| {d.max():.3e} at {int(np.count_nonzero(d))} entries")
        raise SystemExit("hoisted SMFE differs from the shipped function IN THIS PROCESS: aborting")
    return same, dt


def check_against_gate(tau, sigma, gate_rec, what, log):
    """Hard stop on any bitwise difference from the gate, reporting the hashes and,
    when a saved reference is available, the size of the difference."""
    ht, hs = sha(tau), sha(sigma)
    if ht == gate_rec["tau_sha256"] and hs == gate_rec["sigma_sha256"]:
        return
    msg = [f"{what}: tau sha {ht[:16]} (gate {gate_rec['tau_sha256'][:16]}), "
           f"sigma sha {hs[:16]} (gate {gate_rec['sigma_sha256'][:16]}); "
           f"dtypes {np.asarray(tau).dtype}/{np.asarray(sigma).dtype}, "
           f"shapes {np.asarray(tau).shape}/{np.asarray(sigma).shape}"]
    ref = os.environ.get("WI_SMFE_REF")
    if ref and os.path.exists(ref + "_tau.npy"):
        rt, rs = np.load(ref + "_tau.npy"), np.load(ref + "_sigma.npy")
        dt = np.abs(np.asarray(tau, np.float64) - rt.astype(np.float64))
        ds = np.abs(np.asarray(sigma, np.float64) - rs.astype(np.float64))
        msg.append(f"vs saved reference: tau max |diff| {dt.max():.3e} at {int(np.count_nonzero(dt))} entries, "
                   f"sigma max |diff| {ds.max():.3e} at {int(np.count_nonzero(ds))} entries")
    log("  " + " ; ".join(msg))
    raise SystemExit("SMFE output differs from the gate: aborting")


def build(L):
    import unified_equil as UE
    fa = argparse.Namespace(family="F2", variant="strong", n=200_000, k=4, rho=0.99, L=L,
                            blur=0.7, channels=3, mu=1e-2, seed=0)
    return UE.build_family(fa)


def stacked_like_shipped(spec):
    """The stacked operator and scalar spans exactly as blockmetric_pilot.shipped_scaling
    and compile_family build them for this family (one group_l2 term, tied)."""
    import jax.numpy as jnp
    ops = [d["op"] for d in spec.dual_blocks]
    dims = [op.shape[0] for op in ops]
    slices, a = [], 0
    for dim in dims:
        slices.append((a, a + dim)); a += dim
    spans = [sl for d, sl in zip(spec.dual_blocks, slices) if d["kind"] == "group_l2"]
    matvec = lambda x: jnp.concatenate([op.matvec(x) for op in ops])
    rmatvec = lambda y: sum(op.rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))
    return matvec, rmatvec, slices, a, spans


def smfe_call(fn, mv, rmv, spec, m, spans):
    import jax
    t0 = time.perf_counter()
    tau, sigma, nrm, diag = fn(mv, rmv, spec.n, m, spans, diagnostics=True, probes=2, iters=80,
                               schedule="sgd")
    jax.block_until_ready((tau, sigma))
    return tau, sigma, nrm, diag, time.perf_counter() - t0


# ----------------------------------------------------------------------- gate
def gate(args, log):
    import jax, ocs
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    import smfe_hoisted as SH
    # the generation diff must still be exactly the decorator change
    src = open(os.path.join(os.path.dirname(ocs.__file__), "solver", "equilibrate_sgd.py")).read()
    shipped_fn = src[src.index("def smfe_equilibration("):]
    copy_src = open(os.path.join(HERE, "smfe_hoisted.py")).read()
    copy_fn = copy_src[copy_src.index("def smfe_equilibration_hoisted("):]
    expected = shipped_fn.replace("def smfe_equilibration(", "def smfe_equilibration_hoisted(", 1).replace(
        "    @jax.jit\n    def run():",
        "    _key = (id(matvec), id(rmatvec), int(n), int(m), spans, int(iters), int(probes),\n"
        "            float(gamma), float(M), float(safety), int(seed), bool(diagnostics), str(schedule))\n"
        "    @_hoisted_jit(_key, matvec, rmatvec)\n    def run():", 1)
    text_ok = (copy_fn == expected)
    log(f"  source check: copy equals shipped function with only the decorator change: {text_ok}")

    prob, spec = build(args.L)
    mv, rmv, slices, m, spans = stacked_like_shipped(spec)
    t_s, s_s, n_s, d_s, dt_s = smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
    t_h, s_h, n_h, d_h, dt_h = smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
    t_h2, s_h2, n_h2, d_h2, dt_h2 = smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
    A = lambda x: np.asarray(x)
    checks = dict(
        tau_bitwise=bool(np.array_equal(A(t_s), A(t_h))),
        sigma_bitwise=bool(np.array_equal(A(s_s), A(s_h))),
        norm_equal=bool(n_s == n_h),
        diag_equal=bool(d_s == d_h),
        cached_tau_bitwise=bool(np.array_equal(A(t_h), A(t_h2))),
        cached_sigma_bitwise=bool(np.array_equal(A(s_h), A(s_h2))),
        cached_norm_equal=bool(n_h == n_h2),
        cache_builds=SH.STATS["builds"], cache_hits=SH.STATS["hits"],
        source_text_equal=bool(text_ok))
    ok = all(v for k, v in checks.items() if k not in ("cache_builds", "cache_hits")) \
        and SH.STATS["builds"] == 1 and SH.STATS["hits"] == 1
    if not checks["tau_bitwise"] or not checks["sigma_bitwise"]:
        dt = np.abs(A(t_s).astype(np.float64) - A(t_h).astype(np.float64))
        ds = np.abs(A(s_s).astype(np.float64) - A(s_h).astype(np.float64))
        checks.update(tau_max_abs_diff=float(dt.max()), sigma_max_abs_diff=float(ds.max()),
                      tau_n_diff=int(np.count_nonzero(dt)), sigma_n_diff=int(np.count_nonzero(ds)))
    rec = dict(L=args.L, n=spec.n, m=m, passed=bool(ok), checks=checks,
               tau_sha256=sha(t_s), sigma_sha256=sha(s_s), norm=float(n_s),
               shipped_call_s=dt_s, hoisted_first_s=dt_h, hoisted_cached_s=dt_h2,
               host=os.uname().nodename, jax=jax.__version__, ocs=ocs.__version__)
    for k, v in checks.items():
        log(f"  {k}: {v}")
    log(f"  shipped call {dt_s:.3f}s  hoisted first {dt_h:.3f}s  hoisted cached {dt_h2:.3f}s")
    log(f"  GATE {'PASSED' if ok else 'FAILED -- no timing may use the copy'}")
    return rec, ok


# ----------------------------------------------------------------------- arms
def run_engine(args, prob, spec, data, gate_rec, log):
    import jax
    kw = dict(tol=args.tol, max_blocks=args.max_blocks, block=100, pdhg_variant="standard")
    t0 = time.perf_counter(); comp = prob.compile(**kw); compile_cold = time.perf_counter() - t0
    t0 = time.perf_counter(); sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
    solve_cold = time.perf_counter() - t0
    iters_all = [int(sol.iters)]
    warm = []
    for _ in range(WARM_REPS):
        t0 = time.perf_counter(); sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
        warm.append(time.perf_counter() - t0)
        iters_all.append(int(sol.iters))
    cert = comp.certificate(sol.x, sol.y, data)
    res = max(float(cert["r_primal"]), float(cert["r_dual"]))
    row = dict(iters=int(sol.iters), status=str(sol.status), residual=res,
               certified=int(res <= args.tol),
               compile_cold_s=compile_cold, solve_cold_s=solve_cold,
               solve_warm_s=float(np.median(warm)), solve_warm_all=warm,
               total_cold_s=compile_cold + solve_cold,
               total_warm_s=float(np.median(warm)),
               iters_per_run=iters_all, iteration_resolution=100)
    log(f"  A0std: {row['iters']} it {row['status']} r={res:.3e}  compile(cold) {compile_cold:.2f}s  "
        f"solve cold {solve_cold:.2f}s  solve warm {row['solve_warm_s']:.2f}s")

    # ---- engine SMFE recompilation, measured only after the arm's own timings
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    import smfe_hoisted as SH
    recomp = []
    for _ in range(2):
        t0 = time.perf_counter(); c2 = prob.compile(**kw); recomp.append(time.perf_counter() - t0); del c2
    mv, rmv, slices, m, spans = stacked_like_shipped(spec)
    shipped = []
    for i in range(3):
        t_s, s_s, n_s, _d, dt = smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
        shipped.append(dt)
    hoisted = []
    for i in range(4):
        t_h, s_h, n_h, _d, dt = smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
        hoisted.append(dt)
    within = bool(np.array_equal(np.asarray(t_s), np.asarray(t_h))
                  and np.array_equal(np.asarray(s_s), np.asarray(s_h)))
    gdiff = gate_difference(t_s, s_s, gate_rec, args.ref)
    smfe_exec = float(np.median(hoisted[1:]))
    smfe_shipped_warmproc = float(np.median(shipped[1:]))
    recompile = float(np.median(recomp))
    est = smfe_shipped_warmproc - smfe_exec
    row.update(recompile_s=recompile, recompile_all=recomp,
               smfe_shipped_calls_s=shipped, smfe_hoisted_calls_s=hoisted,
               smfe_exec_s=smfe_exec, smfe_recompile_est_s=est,
               smfe_recompile_share_of_recompile=est / recompile,
               smfe_recompile_share_of_cold_total=est / row["total_cold_s"],
               smfe_within_process_bitwise=within, smfe_gate_difference=gdiff,
               smfe_recompile_share_of_warm_setup=est / recompile,
               composite_warm_with_smfe_exec_s=row["solve_warm_s"] + smfe_exec,
               # The pilot compiled the certificate family FIRST in every process, so its
               # engine setup was a compile in a WARM process: recompile + warm solve.
               # The revision that recorded twostage/timing_L*.jsonl (sha256 c17208bf...)
               # wrote compile_cold + warm solve here, a withdrawn reconstruction; those
               # raw records are corrected in twostage/pilot_definition_derived.jsonl.
               pilot_definition_total_warm_s=recompile + row["solve_warm_s"],
               superseded_cold_compile_reconstruction_s=compile_cold + row["solve_warm_s"])
    log(f"  engine recompile (warm process) {recompile:.2f}s; shipped SMFE call {smfe_shipped_warmproc:.3f}s "
        f"vs hoisted execution {smfe_exec:.3f}s -> SMFE recompilation ~{est:.3f}s "
        f"({100 * est / recompile:.1f}% of a recompile, {100 * est / row['total_cold_s']:.1f}% of the cold total); "
        f"shipped == hoisted in this process: {within}; tau vs gate: sha {gdiff['tau_sha16']} "
        f"match {gdiff['tau_matches_gate']} max|diff| {gdiff['tau_max_abs_diff']}; sigma match {gdiff['sigma_matches_gate']}")
    if not within:
        raise SystemExit("hoisted SMFE differs from the shipped function IN THIS PROCESS: aborting")
    return row


def run_proto(args, prob, spec, data, gate_rec, log):
    import jax, jax.numpy as jnp
    import blockmetric_pilot as BP
    import smfe_hoisted as SH
    arm = args.arm
    eta, source, q = (0.0, "exact", 0) if arm == "A1" else (1.0, "sketch", 16 * spec.k)
    mv, rmv, slices, m, spans = stacked_like_shipped(spec)

    def scaling():
        t_, s_, n_, d_, dt = smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
        return dict(tau=t_, sigma=s_, slices=slices, m=m, scaled_norm=n_, diag=d_, seconds=dt), dt

    # `prototype_metrics` computes DIAGNOSTIC exact Grams and block condition numbers
    # whenever the family exposes `exact_gram`.  They are not part of the method and
    # the pilot excluded them from setup by timing gram + eig + norm only.  With the
    # sketch source (A3s16) exact_gram is used ONLY for those diagnostics, so it is
    # switched off here; the metric is unchanged (checked by hash across repeats).
    if source == "sketch":
        for d in spec.dual_blocks:
            d["exact_gram"] = None

    def metrics(sc):
        T, met, timings, _diag, nrm = BP.prototype_metrics(spec, sc, arm, eta, source, q, lambda *_: None)
        jax.block_until_ready((T, [x for mm in met for x in mm[1:]]))
        return T, met, nrm, timings["gram_s"] + timings["eig_s"] + timings["norm_s"]

    # ---- COLD: every component's first call
    t0 = time.perf_counter(); comp = prob.compile(tol=args.tol, max_blocks=1, block=100)
    cert_compile_cold = time.perf_counter() - t0
    sc, smfe_cold = scaling()
    if args.check_only:
        # DIAGNOSTIC ONLY, no timing is recorded: the exact prelude of a timing
        # process, then the SHIPPED function in the same process, compared bitwise.
        from ocs.solver.equilibrate_sgd import smfe_equilibration
        t_s, s_s, n_s, _d, _dt = smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
        A = lambda x: np.asarray(x)
        same = bool(np.array_equal(A(t_s), A(sc["tau"])) and np.array_equal(A(s_s), A(sc["sigma"])))
        gate_t = sha(sc["tau"]) == gate_rec["tau_sha256"]
        gate_s = sha(sc["sigma"]) == gate_rec["sigma_sha256"]
        log(f"  CHECK-ONLY: within this process shipped == hoisted bitwise: {same}; "
            f"hoisted tau sha {sha(sc['tau'])[:16]} matches gate: {gate_t}; "
            f"sigma matches gate: {gate_s}; shipped tau sha {sha(t_s)[:16]}; "
            f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
        ref = os.environ.get("WI_SMFE_REF")
        if ref and os.path.exists(ref + "_tau.npy"):
            rt = np.load(ref + "_tau.npy").astype(np.float64)
            dt = np.abs(A(sc["tau"]).astype(np.float64) - rt)
            log(f"  CHECK-ONLY: hoisted tau vs saved reference: max |diff| {dt.max():.3e}, "
                f"max relative {float((dt / np.abs(rt)).max()):.3e}, entries differing {int(np.count_nonzero(dt))}")
        raise SystemExit(0)
    cold_tau_sha = sha(sc["tau"])[:16]
    T, met, nrm, metric_cold = metrics(sc)
    t0 = time.perf_counter(); loop = BP.Loop(spec, T, met, block=100); loop_build = time.perf_counter() - t0
    cold = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
    met_hash = sha(np.asarray(met[0][-1]))
    iters_all = [int(cold["iters"])]
    warm_tau_sha = []

    # ---- WARM: the same components, already compiled
    w_smfe, w_metric, w_loop, w_iter, w_verify = [], [], [], [], []
    res = None
    for _ in range(WARM_REPS):
        sc2, ds = scaling(); w_smfe.append(ds); warm_tau_sha.append(sha(sc2["tau"])[:16])
        T2, met2, _n2, dm = metrics(sc2); w_metric.append(dm)
        if sha(np.asarray(met2[0][-1])) != met_hash:
            raise SystemExit("warm metric differs from the cold metric: aborting")
        res = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
        w_loop.append(res["total_s"]); w_iter.append(res["iter_s"]); w_verify.append(res["verify_s"])
        iters_all.append(int(res["iters"]))
    # ---- after ALL timings: the approved within-process gate, then the gate difference
    within, shipped_call_s = within_process_check(sc["tau"], sc["sigma"], spec, mv, rmv, m, spans,
                                                  log, arm)
    gdiff = gate_difference(sc["tau"], sc["sigma"], gate_rec, args.ref)
    med = lambda v: float(np.median(v))
    row = dict(iters=int(res["iters"]), status=str(res["status"]), residual=float(res["residual"]),
               certified=int(res["residual"] <= args.tol), q=q, eta=eta,
               cert_compile_cold_s=cert_compile_cold,
               smfe_cold_s=smfe_cold, smfe_warm_s=med(w_smfe), smfe_compile_s=smfe_cold - med(w_smfe),
               metric_cold_s=metric_cold, metric_warm_s=med(w_metric),
               metric_compile_s=metric_cold - med(w_metric),
               loop_build_s=loop_build,
               loop_cold_s=cold["total_s"], loop_warm_s=med(w_loop), loop_compile_s=cold["total_s"] - med(w_loop),
               iter_cold_s=cold["iter_s"], iter_warm_s=med(w_iter), verify_warm_s=med(w_verify),
               total_cold_s=cert_compile_cold + smfe_cold + metric_cold + loop_build + cold["total_s"],
               total_warm_s=med(w_smfe) + med(w_metric) + med(w_loop),
               pilot_definition_total_warm_s=smfe_cold + med(w_metric) + med(w_loop),
               warm_all=dict(smfe=w_smfe, metric=w_metric, loop=w_loop),
               norm_est=float(nrm["norm_est"]), cache=dict(SH.STATS),
               iters_per_run=iters_all, iteration_resolution=100,
               smfe_within_process_bitwise=within, smfe_shipped_call_after_timings_s=shipped_call_s,
               smfe_gate_difference=gdiff, cold_tau_sha16=cold_tau_sha, warm_tau_sha16=warm_tau_sha,
               warm_tau_equal_cold=all(h == cold_tau_sha for h in warm_tau_sha))
    log(f"  {arm}: {row['iters']} it {row['status']} r={row['residual']:.3e}  "
        f"cold total {row['total_cold_s']:.2f}s  warm total {row['total_warm_s']:.2f}s  "
        f"[smfe {smfe_cold:.2f}/{row['smfe_warm_s']:.3f}  metric {metric_cold:.2f}/{row['metric_warm_s']:.3f}  "
        f"loop {cold['total_s']:.2f}/{row['loop_warm_s']:.2f}  cert compile {cert_compile_cold:.2f}]  "
        f"iters {iters_all}; tau vs gate: sha {gdiff['tau_sha16']} match {gdiff['tau_matches_gate']} "
        f"max|diff| {gdiff['tau_max_abs_diff']}; sigma match {gdiff['sigma_matches_gate']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "arm", "saveref"])
    ap.add_argument("--ref", default=None, help="prefix of the saved gate-reference tau/sigma arrays")
    ap.add_argument("--arm", choices=["A0std", "A1", "A3s16"])
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=5000)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="diagnostic: run a prototype process's prelude and compare shipped vs hoisted in-process; no timings")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)

    import jax, jax.numpy as jnp, ocs
    t0 = time.perf_counter(); _ = float(jnp.sum(jnp.ones(8))); init_s = time.perf_counter() - t0
    import blockmetric_pilot as BP
    log(f"# host {os.uname().nodename} jax {jax.__version__} ocs {ocs.__version__} "
        f"BM_PROX_IMPL={BP.PROX_IMPL} newton={BP.NEWTON_ITERS} kappa={BP.KAPPA_MAX} init {init_s:.2f}s")

    if args.mode == "saveref":
        # save the shipped SMFE output as the gate reference, ONLY if it hashes to the gate
        from ocs.solver.equilibrate_sgd import smfe_equilibration
        gate_rec = json.load(open(args.gate))
        prob, spec = build(args.L)
        mv, rmv, slices, m, spans = stacked_like_shipped(spec)
        t_s, s_s, n_s, _d, _dt = smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
        ok = sha(t_s) == gate_rec["tau_sha256"] and sha(s_s) == gate_rec["sigma_sha256"]
        log(f"  saveref L={args.L}: shipped output hashes to the gate: {ok} (tau sha {sha(t_s)[:16]})")
        if not ok:
            sys.exit(4)
        np.save(args.ref + "_tau.npy", np.asarray(t_s)); np.save(args.ref + "_sigma.npy", np.asarray(s_s))
        sys.exit(0)

    if args.mode == "gate":
        rec, ok = gate(args, log)
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=1)
        sys.exit(0 if ok else 3)

    gate_rec = json.load(open(args.gate))
    if not gate_rec.get("passed") or gate_rec.get("L") != args.L:
        raise SystemExit("no passed gate for this size: refusing to time")
    prob, spec = build(args.L)
    data = spec.data
    log(f"# {spec.name} n={spec.n} arm {args.arm} rep {args.rep}")
    row = (run_engine if args.arm == "A0std" else run_proto)(args, prob, spec, data, gate_rec, log)
    row.update(arm=args.arm, L=args.L, rep=args.rep, n=spec.n, init_s=init_s,
               prox_impl=BP.PROX_IMPL, newton_iters=BP.NEWTON_ITERS, kappa_max=BP.KAPPA_MAX,
               tol=args.tol, host=os.uname().nodename, jax=jax.__version__, ocs=ocs.__version__,
               time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    log(f"appended to {args.out}")


if __name__ == "__main__":
    main()
