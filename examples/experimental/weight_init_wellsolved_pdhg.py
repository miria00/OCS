"""PDHG at a WELL-SOLVED restricted ridged metric, weight-matched to the shipped SMFE.

Why.  Section 5.6 of profiling/weight_init_20260913/README.md found that the
restricted (tied) ridged constructor used in the three-arm comparison was
under-solved -- 8x further than SMFE from the optimum of their shared objective --
because of its 16-probe column budget, and that the under-solved points also sat
off the reciprocal ray.  So its PDHG loss said nothing about the formulation.  This
runs PDHG at restricted metrics that reach a BETTER objective than SMFE, to see
whether the iteration gap closes.

These metrics are NOT practical constructors: at 256 column probes the restricted
constructor costs 3,172 to 5,476 operator applications (3 rounds) or 30,820 to
53,860 (30 rounds) against SMFE's 350, so any PDHG result here is a statement about
the FORMULATION, not a deployable method.

One normalization and one weight, so the comparison cannot re-confound.
  * NORMALIZATION.  Every arm gets the same stability rule: 0.9 divided by a
    50-iteration power estimate of ||Sigma^{1/2} K T^{1/2}||.  The shipped SMFE
    applies 0.9 over its OWN 15-iteration estimate internally, so two shipped arms
    are run: `S` exactly as the three-arm comparison used it, and `Sn`, the same
    SMFE scalings with that internal rescale undone and the common 50-iteration rule
    applied instead.  Weight-matched arms are compared against `Sn`.
  * WEIGHT.  The primal/dual weight of a metric is taken as
        L(T, Sigma) = mean_i log sigma_i  -  mean_j log tau_j ,
    the log of the ratio of the geometric means of the dual and primal steps, with
    each tied scalar counted once per row it covers.  A common rescale of both steps
    leaves L unchanged, and a move along the reciprocal ray (sigma e^{2c},
    tau e^{-2c}) changes it by 4c, while leaving the preconditioned operator, and so
    the stability rescale, unchanged.  Matching sets c = (L_S - L_U) / 4, which
    shifts the metric's effective PDHG weight by e^{2c}.  The matched values of L
    and c are recorded per arm.

Arms (all tied-class, shipped-admissible):
  S       shipped SMFE, as in the three-arm comparison
  Sn      shipped SMFE, common normalization
  U3      ridged restricted, 3 rounds, q_col = 16 (as run), raw weight
  U3m     the same metric, weight-matched to Sn
  U3bigm  3 rounds, q_col = 256, weight-matched
  Ubigm   30 rounds, q_col = 256, weight-matched
Iterations are checked every 100, so iteration counts have 100-iteration
resolution.

Nothing in ocs/ is edited.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

SAFETY, POWER = 0.9, 50


def weight_coordinate(spec, T, metrics):
    """L = mean_i log sigma_i - mean_j log tau_j over every dual row and primal coordinate."""
    tot, cnt = 0.0, 0
    for d, met in zip(spec.dual_blocks, metrics):
        rows = d["op"].shape[0]
        if met[0] == "scalar":
            tot += rows * math.log(float(np.asarray(met[1], np.float64))); cnt += rows
        else:
            a = np.asarray(met[1], np.float64).reshape(-1)
            tot += float(np.sum(np.log(a))); cnt += a.size
    tau = np.asarray(np.broadcast_to(np.asarray(T, np.float64), (spec.n,)))
    return tot / cnt - float(np.mean(np.log(tau)))


def normalize(spec, T, metrics):
    import blockmetric_pilot as BP
    nu = BP.power_norm_metric(spec, T, metrics, iters=POWER)
    met, Tn = BP.scale_metrics(metrics, T, SAFETY / nu)
    return Tn, list(met), float(nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=1000)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import jax, jax.numpy as jnp, ocs
    import unified_equil as UE
    import blockmetric_pilot as BP
    import weight_init_ridge as WR
    log = lambda s: print(s, flush=True)

    fa = argparse.Namespace(family=args.family, variant="overlap", n=200_000, k=4, rho=args.rho, L=args.L,
                            blur=0.7, channels=3, mu=1e-2, seed=0)
    if args.family == "F2":
        fa.variant = "strong"
    prob, spec = UE.build_family(fa)
    data = spec.data
    k = spec.k
    classes = UE.resolve_classes(spec, "tied")
    comp = prob.compile(tol=args.tol, max_blocks=1, block=100)
    log(f"# host {os.uname().nodename} jax {jax.__version__} ocs {ocs.__version__} BM_PROX_IMPL={BP.PROX_IMPL}")
    log(f"# {spec.name}: n={spec.n} tol {args.tol} budget {args.max_blocks * 100} iterations; "
        f"iteration resolution 100")
    out = dict(meta=dict(family=args.family, spec=spec.name, n=spec.n, tol=args.tol,
                         budget_iters=args.max_blocks * 100, repeats=args.repeats, safety=SAFETY,
                         power_iters=POWER, iteration_resolution=100, prox_impl=BP.PROX_IMPL,
                         host=os.uname().nodename, time=time.strftime("%Y-%m-%d %H:%M:%S %Z")),
               arms={})

    def save():
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    arms = {}
    sc = BP.shipped_scaling(spec)
    T_S, met_S = UE.shipped_metrics(spec, sc, classes)
    arms["S"] = (T_S, list(met_S), dict(normalization="SMFE internal: 0.9 / its 15-iteration power estimate",
                                        nu=float(sc["scaled_norm"])))
    undo = (sc["scaled_norm"] + 1e-30) / SAFETY
    met_pre, T_pre = BP.scale_metrics(met_S, T_S, undo)
    T_Sn, met_Sn, nu_Sn = normalize(spec, T_pre, list(met_pre))
    arms["Sn"] = (T_Sn, met_Sn, dict(normalization="common: 0.9 / 50-iteration power estimate", nu=nu_Sn))
    L_S = weight_coordinate(spec, T_Sn, met_Sn)
    L_S_raw = weight_coordinate(spec, T_S, met_S)
    log(f"  weight coordinate L of the shipped SMFE: {L_S:.6f} (common normalization), "
        f"{L_S_raw:.6f} (as shipped) -- equal up to rounding, as a common rescale must leave it")
    out["L_shipped"] = dict(common=L_S, as_shipped=L_S_raw)

    for tag, rounds, qg, qc, match in (("U3", 3, 16 * k, 16, False), ("U3m", 3, 16 * k, 16, True),
                                       ("U3bigm", 3, 256 * k, 256, True), ("Ubigm", 30, 256 * k, 256, True)):
        T, met, _h, cnt, _tm, _r, _s = WR.ridged_equilibration(spec, classes, rounds, qg, qc,
                                                                log=lambda *_: None, record=False)
        L_before = weight_coordinate(spec, T, met)
        c = (L_S - L_before) / 4.0 if match else 0.0
        if match:
            met, T = WR.slide_ray(met, T, c)
            met = list(met)
        Tn, metn, nu = normalize(spec, T, met)
        L_after = weight_coordinate(spec, Tn, metn)
        info = dict(rounds=rounds, q_gram=qg, q_col=qc, applications=cnt["fwd"] + cnt["adj"] + POWER * 2,
                    weight_matched=match, L_before=L_before, ray_shift_c=c,
                    effective_weight_factor=math.exp(2 * c), L_after=L_after, L_target=L_S if match else None,
                    normalization="common: 0.9 / 50-iteration power estimate", nu=nu)
        arms[tag] = (Tn, metn, info)
        log(f"  {tag}: L before {L_before:.6f}" + (f" -> ray shift c = {c:+.6f} (weight x{math.exp(2 * c):.4f}) "
                                                   f"-> L after {L_after:.6f} (target {L_S:.6f})" if match else
                                                   f" (not matched; differs from SMFE by weight x{math.exp((L_before - L_S) / 2):.4f})"))

    for tag in ("S", "Sn", "U3", "U3m", "U3bigm", "Ubigm"):
        T, met, info = arms[tag]
        loop = BP.Loop(spec, T, met, block=100)
        loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)     # compile
        reps = []
        for r in range(args.repeats):
            rr = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
            reps.append(dict(iters=int(rr["iters"]), status=str(rr["status"]), residual=float(rr["residual"]),
                             iter_s=float(rr["iter_s"]), verify_s=float(rr["verify_s"]), total_s=float(rr["total_s"])))
        obj, minx = UE.float64_objective(spec, rr["x"])
        rec = dict(info, iters_per_repeat=[x["iters"] for x in reps], status=reps[-1]["status"],
                   residual=reps[-1]["residual"], certified=int(reps[-1]["residual"] <= args.tol),
                   loop_s_median=float(np.median([x["iter_s"] for x in reps])),
                   loop_s_min=float(min(x["iter_s"] for x in reps)), loop_s_max=float(max(x["iter_s"] for x in reps)),
                   objective_f64=obj, reps=reps)
        out["arms"][tag] = rec
        log(f"  {tag:7s} iters {rec['iters_per_repeat']} {rec['status']} r={rec['residual']:.3e} "
            f"loop {rec['loop_s_median']:.3f}s [{rec['loop_s_min']:.3f}, {rec['loop_s_max']:.3f}] "
            f"objective {obj:.6f}")
        save()
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
