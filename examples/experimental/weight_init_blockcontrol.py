"""The control the unified block-metric study is missing: the SAME construction
restricted to the classes the shipped engine can actually consume.

Why this exists.  `profiling/unified_equil_20260913/` compares a unified
block-metric equilibration against the shipped diagonal one, and reports both a
metric difference and a large primal/dual weight effect.  Those two are
confounded twice over:

  1. the unified arm uses RICHER metrics (full k x k blocks) than the shipped
     engine can consume, so an advantage could be the richness or could be the
     construction, and
  2. it sits at a different point on the primal/dual weight ray.

This runs the unified constructor with per-block classes restricted to what the
shipped proxes admit -- a coupled `group_l2` term takes ONE tied scalar, a
`sum_squares` term takes a per-row diagonal -- against the shipped SMFE and
against the unrestricted full-block version, each at the SAME normalization
(the canonical gauge) and each under the SAME weight policy as every other arm
in `profiling/weight_init_20260913/`: the weight estimated once at setup by the
rules in `weight_init.py`, applied as tau <- tau/w, sigma <- sigma*w.

A note on framing, because the earlier brief for this work had it wrong.  The
reciprocal scaling ambiguity (D -> e^a D, E -> e^-a E) is a KNOWN property of
the unregularized equilibration problem, stated by Diamond and Boyd in section
3.3 of "Stochastic Matrix-Free Equilibration" (papers/mf_equil.pdf), and they
fix it in the same section with a ridge and a box.  The SHIPPED code implements
that fix: `ocs/solver/equilibrate_sgd.py` carries `gamma=1.0` and `M=6.0`, its
gradients are `row_e - r_tgt + gamma*u` and `col_e - s_tgt + gamma*v`, and the
iterates are clipped to [-M, M].  So the shipped diagonal equilibration DOES
pin the scale.  What is measured here is not a gap the equilibration cannot
fill; it is that the point the ridge pins is a symmetric-scale convention, and
the primal/dual balance PDHG wants is a separate parameter, which is what PDLP
and its relatives select explicitly.

Budget accounting is reported separately for construction, weight selection,
solve and checking, because the unified constructor's saving (228 applications
against the shipped 450 at one round) is only about 111 iterations of headroom
and a weight SEARCH would erase it.

Nothing in ocs/ is edited, and nothing in `unified_equil.py` or
`blockmetric_pilot.py` is edited either: both are imported unchanged so that the
earlier study's recorded checksums stay valid.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def apply_weight(metrics, T, w):
    """Move the SAME metric along the primal/dual weight ray: sigma <- sigma*w,
    tau <- tau/w.  This is exactly what `compile(primal_weight=w)` does to the
    shipped steps, so the research loop and the shipped engine are moved by the
    same reparametrization.  The preconditioned operator norm is invariant under
    it, so the stability rescale already applied stays valid."""
    out = []
    for met in metrics:
        if met[0] in ("scalar", "diag", "vec"):
            out.append((met[0], met[1] * w))
        else:
            out.append(("full", met[1], met[2] * w))
    return out, T / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="F1", choices=["F1", "F2", "F3"])
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--arms", default="S,U:tied,U:full",
                    help="comma list of <arm>[:classes]; arm in S,TS,U")
    ap.add_argument("--weights", default="1.0,est:objscale",
                    help="comma list; a number, or est:<estimator name> from weight_init.py")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=1000)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax, ocs
    import unified_equil as UE
    import blockmetric_pilot as BP
    import weight_init as WI
    log = lambda s: print(s, flush=True)

    # the family, built exactly as the unified study builds it
    fa = argparse.Namespace(family=args.family, variant="overlap", n=200_000, k=4,
                            rho=args.rho, L=args.L, blur=0.7, channels=3, mu=1e-2, seed=0)
    if args.family == "F2":
        fa.variant = "strong"                 # build_family reads mixing from .variant
    prob, spec = UE.build_family(fa)
    data = spec.data
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} "
        f"jax {jax.__version__} ocs {ocs.__version__}")
    log(f"# {spec.name}: n={n} m={m} tol {args.tol} budget {args.max_blocks * 100} iterations")

    # ---- the weight policy, identical to every other arm in this study
    ck = {"pdhg_variant": "standard"}
    t0 = time.perf_counter()
    est, ediag = WI.setup_estimates(prob, data, ck)
    pe, pdiag = WI.probe_estimates(prob, data, ck, args.tol)
    est.update(pe); ediag.update(pdiag)
    for k in WI.ESTIMATORS:
        est[k]["w"], est[k]["clip"] = WI._clip(est[k]["raw"])
        log(f"  estimator {k:18s} raw={est[k]['raw']:.6g}  w={est[k]['w']:.6g} "
            f"[{est[k]['cost_ops']} ops, {est[k]['cost_s']:.3f}s]")
    est_s = time.perf_counter() - t0

    weights = []
    for tok in args.weights.split(","):
        tok = tok.strip()
        if tok.startswith("est:"):
            name = tok[4:]
            weights.append((f"est:{name}", est[name]["w"], est[name]["cost_s"],
                            est[name]["cost_ops"]))
        else:
            weights.append((f"fixed:{tok}", float(tok), 0.0, 0))

    comp = prob.compile(tol=args.tol, max_blocks=1, block=100)   # public certificate only
    out = dict(meta=dict(family=args.family, spec=spec.name, n=n, m=m, tol=args.tol,
                         rho=args.rho, L=args.L, rounds=args.rounds,
                         max_blocks=args.max_blocks, budget_iters=args.max_blocks * 100,
                         max_seconds=args.max_seconds, repeats=args.repeats,
                         host=os.uname().nodename, device=jax.devices()[0].device_kind,
                         jax=jax.__version__, ocs=ocs.__version__,
                         estimator_setup_s=est_s,
                         time=time.strftime("%Y-%m-%d %H:%M:%S %Z")))
    out["estimates"] = est
    out["diagnostics"] = ediag
    out["rows"] = []

    for spec_arm in args.arms.split(","):
        arm, _, cls = spec_arm.strip().partition(":")
        cls = cls or "full"
        ba = argparse.Namespace(arm=arm, classes=cls, gauge="canonical", rounds=args.rounds,
                                qgram=16, qcol=16, probe_mode="fixed", source="probe",
                                seed=0, record_alt=False, delta_mode="fixed",
                                regauge_shipped=False, kappa_max=100.0)
        ba.qgram_blocks = ba.qgram * spec.k
        classes = UE.resolve_classes(spec, cls)
        log(f"\n## arm {arm} classes={cls} "
            f"({'|'.join(UE.class_tag(c) for c in classes)})")
        UE.build_arm(ba, spec, classes, lambda *_: None)          # pay XLA compilation
        t0 = time.perf_counter()
        built = UE.build_arm(ba, spec, classes, log)
        construct_s = time.perf_counter() - t0
        c = built["counts"]
        apps = c["equil_fwd"] + c["equil_adj"] + c["norm_fwd"] + c["norm_adj"]
        log(f"  construction: {construct_s:.2f}s, {apps} operator applications "
            f"(equil {c['equil_fwd'] + c['equil_adj']}, power {c['norm_fwd'] + c['norm_adj']})")
        for wtag, w, wcost_s, wcost_ops in weights:
            metrics, T = apply_weight(built["metrics"], built["T"], w)
            loop = BP.Loop(spec, T, metrics, block=100)
            times, res = [], None
            for r in range(args.repeats):
                rr = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
                times.append(rr["total_s"])
                res = rr
                log(f"    {wtag:<18s} w={w:<10.5g} rep{r}: {rr['iters']:7d} it  "
                    f"{rr['status']:12s} r={rr['residual']:.4e}  {rr['total_s']:7.2f}s")
                if sum(times) > args.max_seconds and r + 1 < args.repeats:
                    log(f"    (stopping repeats at {r + 1}: wall budget)"); break
            obj, minx = UE.float64_objective(spec, res["x"])
            warm = times[1:] or times
            out["rows"].append(dict(
                arm=arm, classes=cls, class_tag="|".join(UE.class_tag(x) for x in classes),
                weight_tag=wtag, w=float(w), iters=res["iters"], status=res["status"],
                residual=res["residual"], r_primal=res["cert"]["r_primal"],
                r_dual=res["cert"]["r_dual"], certified=int(res["residual"] <= args.tol),
                objective_f64=obj, min_x=minx,
                construct_s=construct_s, construct_apps=apps,
                construct_equil_apps=c["equil_fwd"] + c["equil_adj"],
                construct_power_apps=c["norm_fwd"] + c["norm_adj"],
                weight_select_s=wcost_s, weight_select_apps=wcost_ops,
                solve_s=res["iter_s"], verify_s=res["verify_s"],
                cold_total_s=times[0], warm_med_s=float(np.median(warm)),
                warm_min_s=float(min(warm)), warm_max_s=float(max(warm)),
                repeats=len(times), gauge_a=built["extra"].get("gauge_a"),
                norm_est=built["extra"].get("norm_est")))
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1, default=str)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
