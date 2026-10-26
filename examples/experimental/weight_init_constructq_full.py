"""Construction quality of the FULL-BLOCK ridged constructor (refinement 3).

`weight_init_constructq.py` showed that the restricted (tied) ridged constructor,
at the 16-column-probe budget the three-arm comparison used, was under-solved.  The
full-block arm of that comparison, and the whole F2 size ladder, used the same
budget, and its construction quality was not measured.  This measures it.  No PDHG
solve is run.

The objective is the same regularized equilibration objective, written over the
eigen-decomposition of each block metric Sigma_i = Q_i diag(w_i) Q_i^T, with the log
variables u = (1/2) log w counted once per eigenvalue (the multiplicity convention
of the tied case, where a tied scalar stands for every row it covers):

    F = 1/2 sum_j tau_j c_j(Sigma) - r sum u - s sum_j v_j + (gamma/2)(sum u^2 + sum_j v_j^2),
    c_j(Sigma) = (K^T Sigma K)_jj,   v = (1/2) log tau.

EXACT evaluation, per point, is one vector c(Sigma):
  * F1, F3: a pure sweep over the standard basis.  Column j of K is K e_j (one
    forward application); for a full block its contribution to c_j is
    sum_k w_k (q_k^T (K e_j)_b)^2, for a per-row block sum_i sigma_i (K e_j)_i^2.
    Nothing but matvec is called.
  * F2: the unified study's closed-form `exact_coldiag` (it reads the blur kernel),
    spot-checked against the pure formula on 64 random columns.

What can and cannot be certified.  A tied metric (one scalar per term) is a special
case of the full-block class under this objective: Sigma_i = e^{2a} I gives u = a
at every eigenvalue, the same trace term, and the same multiplicity-weighted target
and ridge.  So the TIED optimum is a certified UPPER bound on the full-block optimum:
F*_full <= F*_tied.  The full-block objective is NOT jointly convex in its log
variables, so there is no certified lower bound and no optimality certificate.
What IS certified: the as-run construction's suboptimality is at least
F(as run) - min(F*_tied, best evaluated full point).

Points (full-block ridged constructor, 3 or 30 rounds, 16 or 256 column probes, as
in weight_init_constructq.py): Fu3 = as run, Fu30, Fu3big, Fubig.  At the F2 ladder
sizes L = 2048 and 4096 only Fu3 and Fu3big are evaluated.

Nothing in ocs/ is edited.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

GAMMA, M_BOX = 1.0, 6.0


def device_metric(met):
    import jax.numpy as jnp
    if met[0] == "full":
        return ("full", jnp.asarray(met[1], jnp.float32), jnp.asarray(met[2], jnp.float32))
    return (met[0], jnp.asarray(met[1], jnp.float32))


def c_contrib(seg, kind, arrs, k):
    """sum over this term's rows of the metric quadratic form of each column in `seg` (b, rows).
    `kind` is static (closed over); only arrays pass through jit."""
    import jax.numpy as jnp
    if kind == "vec":
        return (seg * seg) @ arrs[0]
    if kind == "scalar":
        return arrs[0] * jnp.sum(seg * seg, axis=1)
    R = seg.reshape(seg.shape[0], -1, k)
    if kind == "diag":
        return jnp.sum(arrs[0] * R * R, axis=(1, 2))
    P = jnp.einsum("blk,jbl->jbk", arrs[0], R)
    return jnp.sum(arrs[1] * P * P, axis=(1, 2))


def coldiag_sweep(spec, metrics, batch):
    import jax, jax.numpy as jnp
    import weight_init_constructq as CQ
    matvec, slices, m = CQ.stacked(spec)
    n = spec.n
    dms = [device_metric(met) for met in metrics]
    kinds = tuple(dm[0] for dm in dms)
    arrays = tuple(tuple(dm[1:]) for dm in dms)
    ks = [d["k"] if d["kind"] != "sum_squares" else 1 for d in spec.dual_blocks]

    @jax.jit
    def one(J, arrays):
        X = jnp.zeros((J.shape[0], n), jnp.float32).at[jnp.arange(J.shape[0]), J].set(1.0)
        cols = jax.vmap(matvec)(X)
        c = jnp.zeros(J.shape[0], jnp.float32)
        for (s, e), kind, arrs, k in zip(slices, kinds, arrays, ks):
            c = c + c_contrib(cols[:, s:e], kind, arrs, k)
        return c
    c = np.zeros(n, np.float64)
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        J = np.arange(start, min(start + batch, n)); real = J.size
        if real < batch:
            J = np.concatenate([J, np.zeros(batch - real, J.dtype)])
        c[start:start + real] = np.asarray(one(jnp.asarray(J), arrays), np.float64)[:real]
    return c, time.perf_counter() - t0


def coldiag_spot(spec, metrics, J, chunk=8):
    import jax, jax.numpy as jnp
    import weight_init_constructq as CQ
    matvec, slices, m = CQ.stacked(spec)
    dms = [device_metric(met) for met in metrics]
    kinds = tuple(dm[0] for dm in dms)
    arrays = tuple(tuple(dm[1:]) for dm in dms)
    ks = [d["k"] if d["kind"] != "sum_squares" else 1 for d in spec.dual_blocks]

    @jax.jit
    def one(Jc, arrays):
        X = jnp.zeros((Jc.shape[0], spec.n), jnp.float32).at[jnp.arange(Jc.shape[0]), Jc].set(1.0)
        cols = jax.vmap(matvec)(X)
        c = jnp.zeros(Jc.shape[0], jnp.float32)
        for (s, e), kind, arrs, k in zip(slices, kinds, arrays, ks):
            c = c + c_contrib(cols[:, s:e], kind, arrs, k)
        return c
    out = []
    for a in range(0, len(J), chunk):
        Jc = np.asarray(J[a:a + chunk]); pad = chunk - Jc.size
        Jp = np.concatenate([Jc, np.zeros(pad, Jc.dtype)]) if pad else Jc
        out.append(np.asarray(one(jnp.asarray(Jp), arrays), np.float64)[:Jc.size])
    return np.concatenate(out)


def log_eigs(spec, metrics):
    parts = []
    for d, met in zip(spec.dual_blocks, metrics):
        rows = d["op"].shape[0]
        if met[0] == "scalar":
            parts.append(np.full(rows, 0.5 * math.log(float(np.asarray(met[1], np.float64)))))
        elif met[0] in ("vec", "diag"):
            parts.append(0.5 * np.log(np.asarray(met[1], np.float64).reshape(-1)))
        else:
            parts.append(0.5 * np.log(np.asarray(met[2], np.float64).reshape(-1)))
    u = np.concatenate(parts)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--points", default="Fu3,Fu30,Fu3big,Fubig")
    ap.add_argument("--tied-json", default=None, help="step-1 cq_*.json for the tied optimum, when available")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import jax, jax.numpy as jnp, ocs
    import unified_equil as UE
    import weight_init_ridge as WR
    import weight_init_constructq as CQ
    log = lambda s: print(s, flush=True)

    fa = argparse.Namespace(family=args.family, variant="overlap", n=200_000, k=4, rho=args.rho, L=args.L,
                            blur=0.7, channels=3, mu=1e-2, seed=0)
    if args.family == "F2":
        fa.variant = "strong"
    prob, spec = UE.build_family(fa)
    n = spec.n; m = sum(d["op"].shape[0] for d in spec.dual_blocks); k = spec.k
    r_tgt, s_tgt = math.sqrt(n / m), math.sqrt(m / n)
    classes = UE.resolve_classes(spec, "full")
    log(f"# host {os.uname().nodename} jax {jax.__version__} ocs {ocs.__version__}")
    log(f"# {spec.name}: n={n} m={m} classes {'|'.join(UE.class_tag(c) for c in classes)}")
    out = dict(meta=dict(family=args.family, spec=spec.name, n=n, m=m, k=k, rho=args.rho, L=args.L,
                         gamma=GAMMA, M=M_BOX, r_tgt=r_tgt, s_tgt=s_tgt, host=os.uname().nodename,
                         time=time.strftime("%Y-%m-%d %H:%M:%S %Z")), points={})

    def save():
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    # ---- the tied optimum: a certified upper bound on the full-block optimum
    tied = None
    if args.tied_json and os.path.exists(args.tied_json):
        tj = json.load(open(args.tied_json))
        opt = tj["optimum"]
        tied = dict(source=os.path.basename(args.tied_json),
                    value=opt.get("F_star", opt.get("F_star_upper")),
                    kind="exact" if "F_star" in opt else "certified upper end of the tied bracket")
    elif args.family == "F2":
        B = spec.dual_blocks[0]["op"].shape[0] // k
        d = np.asarray(UE.exact_coldiag(spec, [("diag", np.ones((B, k)))]), np.float64)
        opt = CQ.tied_single_optimum(d, m, n, r_tgt, s_tgt, log=log)
        tied = dict(source="exact block coordinate descent on d = diag(K^T K)", value=opt["F_star"],
                    kind="exact", one_d_check=opt["F_star_1d"])
    out["tied_reference"] = tied
    log(f"  tied optimum (certified upper bound on the full-block optimum): {tied}")

    budgets = dict(Fu3=(3, 16 * k, 16), Fu30=(30, 16 * k, 16), Fu3big=(3, 256 * k, 256), Fubig=(30, 256 * k, 256))
    for tag in args.points.split(","):
        rounds, qg, qc = budgets[tag]
        t0 = time.perf_counter()
        T, met, _h, cnt, _tm, _r, _s = WR.ridged_equilibration(spec, classes, rounds, qg, qc,
                                                                log=lambda *_: None, record=False)
        build_s = time.perf_counter() - t0
        tau = np.asarray(jnp.broadcast_to(jnp.asarray(T), (n,)), np.float64)
        if args.family == "F2":
            c = np.asarray(UE.exact_coldiag(spec, [met[0]]), np.float64)
            J = np.random.default_rng(11).choice(n, 64, replace=False)
            cs = coldiag_spot(spec, met, J)
            spot = float(np.max(np.abs(cs - c[J]) / np.maximum(np.abs(c[J]), 1e-300)))
            route, sweep_s = f"closed form, 64-column pure spot check max rel {spot:.2e}", 0.0
        else:
            c, sweep_s = coldiag_sweep(spec, met, args.batch)
            route, spot = "pure basis sweep", None
        u = log_eigs(spec, met)
        v = 0.5 * np.log(tau)
        tr = float(tau @ c)
        F = 0.5 * tr - r_tgt * u.sum() - s_tgt * v.sum() + 0.5 * GAMMA * (u @ u + v @ v)
        rec = dict(F=F, trace=tr, route=route, spot_rel=spot, sweep_s=sweep_s, build_s=build_s,
                   rounds=rounds, q_gram=qg, q_col=qc, applications=cnt["fwd"] + cnt["adj"] + 100,
                   ray_a=float((v.sum() - u.sum()) / (u.size + n)),
                   kappa_max=float(max((np.asarray(x[2]).max(axis=1) / np.asarray(x[2]).min(axis=1)).max()
                                       for x in met if x[0] == "full")),
                   box_binding=int(np.sum(np.abs(u) >= M_BOX - 1e-6) + np.sum(np.abs(v) >= M_BOX - 1e-6)))
        out["points"][tag] = rec
        log(f"  {tag:7s} F = {F:.6f}  ({route}; build {build_s:.1f}s, sweep {sweep_s:.1f}s)  "
            f"applications {rec['applications']}  ray a {rec['ray_a']:+.4e}  kappa max {rec['kappa_max']:.3g}")
        save()

    best = min(p["F"] for p in out["points"].values())
    ub = min(best, tied["value"]) if tied else best
    out["full_optimum_upper_bound"] = ub
    for tag, p in out["points"].items():
        p["certified_suboptimality_at_least"] = p["F"] - ub
        if tied:
            p["F_minus_tied_optimum"] = p["F"] - tied["value"]
    log(f"  certified upper bound on the full-block optimum: {ub:.6f}")
    for tag, p in out["points"].items():
        log(f"  {tag:7s} suboptimality at least {p['certified_suboptimality_at_least']:.3f}"
            + (f"; F - F*_tied = {p['F_minus_tied_optimum']:+.3f}" if tied else ""))
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
