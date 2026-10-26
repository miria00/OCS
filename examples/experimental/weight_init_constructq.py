"""Construction quality: do the shipped SMFE and the restricted ridged block
constructor reach the same value of the objective they both optimize?

Why this exists.  `weight_init_ridge.py` verified that the ridged block
constructor, restricted to the classes the shipped proxes admit (one tied scalar
per coupled group_l2 term, per-row diagonal for sum_squares), optimizes EXACTLY
the shipped SMFE objective: the stationarity of the block equation reduces to
the shipped gradient `row_e e^{2u} - r_tgt + gamma u` to 3.7e-7.  The two are
therefore different finite-budget algorithms for ONE problem.  The three-arm
comparison found the restricted arm needing more PDHG iterations than the
shipped SMFE on every family, and the README first worded that as "the
construction brings nothing".  That wording is only earned if the restricted
constructor reached the optimum at least as well as SMFE did.  If it reached a
WORSE objective value, the loss is an under-solved constructor, not evidence
about the formulation.  This script measures which.  No PDHG solve is run.

The objective, written once, in the shipped variables (sigma = e^{2u} dual,
tau = e^{2v} primal, a scalar per tied span counted with its row multiplicity):

    F(u, v) = 1/2 sum_ij K_ij^2 e^{2u_i + 2v_j} - r sum_i u_i - s sum_j v_j
              + (gamma/2) (sum_i u_i^2 + sum_j v_j^2),     |u|, |v| <= M,

with r = sqrt(n/m), s = sqrt(m/n), gamma = 1, M = 6: Diamond and Boyd's
regularized problem (5), and exactly what ocs/solver/equilibrate_sgd.py
minimizes.  Both constructors are evaluated BEFORE their common stability
rescale (0.9 / power-iteration norm): that rescale moves both scalings by one
factor, which is not a move along the reciprocal ray and does change F, and it
is applied identically afterwards to every arm.  The shipped function returns
the rescaled steps and the norm it used, so the pre-rescale point is recovered
exactly as out * norm / 0.9.

HOW F IS EVALUATED EXACTLY, operator-purely.  F needs only two vectors: the row
energies rho(tau) = (K o K) tau and the column energies c(sigma) = (K o K)^T
sigma.  Both come from ONE sweep over the standard basis: column j of K is
K e_j, one forward application, and its entrywise square gives the j-th column
of K o K.  n forward applications per point, batched with vmap, and nothing but
matvec is called.  The trace is then computed two independent ways, tau^T c and
sigma^T rho, and their disagreement is reported as the numerical floor.

Where a family has a single tied term (F1, F2) the dependence on tau collapses
to d = diag(K^T K), so a single sweep suffices for every point and the
exact minimizer F* can be computed to machine precision by exact block
coordinate descent in float64, cross-checked by a one-dimensional convex
minimization over the tied scalar.  F2's sweep at n = 3.1M would take tens of
minutes, so F2 uses the unified study's closed-form `exact_coldiag` (it reads
the blur kernel) for d, SPOT-CHECKED against pure forward applications on 256
random columns; F1 uses both and reports their agreement.

A certified bound needs no reference point.  F is gamma-strongly convex in the
norm weighted by the row multiplicities (the exponential term is convex, the
ridge contributes gamma W with W = diag(multiplicity)), so over ALL of R^{m'+n}
    F(z) - min F  <=  (1 / 2 gamma) grad F(z)^T W^{-1} grad F(z)  =:  B(z),
and the box-constrained optimum is no smaller than the unconstrained one, so
B(z) certifies the suboptimality of z whether or not the box is active.  On
F1/F2 B(z) >= F(z) - F* is also a consistency test of the exact evaluation.

Points evaluated (all restricted to the shipped-admissible classes):
    S       the shipped SMFE, as the three-arm comparison used it
    U3      the ridged constructor exactly as the comparison used it:
            3 rounds, q_gram = 16 k, q_col = 16
    U30     30 rounds, same probes          (does more rounds close the gap?)
    U3big   3 rounds, q_gram = 256 k, q_col = 256   (do more probes?)
    Ubig    30 rounds, 256 probes           (both)

Nothing in ocs/ is edited; weight_init_ridge.py, unified_equil.py and
blockmetric_pilot.py are imported unchanged.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

GAMMA, M_BOX, SAFETY = 1.0, 6.0, 0.9


# ------------------------------------------------------------- float64 roots
def root64(lam, target, gamma=GAMMA, M=M_BOX, iters=100):
    """Box-constrained minimizer of 1/2 e^{2mu} lam - target mu + gamma/2 mu^2,
    float64, vectorized.  The stationarity map is strictly increasing, so
    bisection on [-M, M] is exact to 12 / 2^100."""
    lam = np.maximum(np.asarray(lam, np.float64), 0.0)
    lo = np.full(lam.shape, -M); hi = np.full(lam.shape, M)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        h = np.exp(2 * mid) * lam + gamma * mid - target
        neg = h < 0
        lo = np.where(neg, mid, lo); hi = np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- the sweep
def stacked(spec):
    import jax.numpy as jnp
    ops = [d["op"] for d in spec.dual_blocks]
    dims = [op.shape[0] for op in ops]
    slices, a = [], 0
    for dim in dims:
        slices.append((a, a + dim)); a += dim

    def matvec(x):
        return jnp.concatenate([op.matvec(x) for op in ops])
    return matvec, slices, a


def column_sweep(spec, sigma_rows, tau, batch, log=print):
    """rho = (K o K) tau  and  c = (K o K)^T sigma, from n forward applications
    on basis vectors.  Returns float64 host vectors and the sweep time."""
    import jax, jax.numpy as jnp
    matvec, _slices, m = stacked(spec)
    n = spec.n
    sig = jnp.asarray(np.asarray(sigma_rows, np.float32))
    tau_d = jnp.asarray(np.asarray(tau, np.float32))

    @jax.jit
    def one(J, wt):
        X = jnp.zeros((J.shape[0], n), jnp.float32).at[jnp.arange(J.shape[0]), J].set(1.0)
        cols = jax.vmap(matvec)(X)                 # (b, m): column j of K
        sq = cols * cols
        return sq.T @ (tau_d[J] * wt), sq @ sig    # rho contribution (m,), c_J (b,)

    rho = np.zeros(m, np.float64); c = np.zeros(n, np.float64)
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        J = np.arange(start, min(start + batch, n))
        real = J.size
        if real < batch:
            J = np.concatenate([J, np.zeros(batch - real, J.dtype)])
        wt = np.zeros(batch, np.float32); wt[:real] = 1.0
        rc, cj = one(jnp.asarray(J), jnp.asarray(wt))
        rho += np.asarray(rc, np.float64)
        c[start:start + real] = np.asarray(cj, np.float64)[:real]
    return rho, c, time.perf_counter() - t0


def spot_columns(spec, J, chunk=8):
    """||K e_j||^2 for the listed columns, by pure forward applications, in small
    vmapped chunks (a 256-wide vmap of the F2 blur at n = 3.1M exhausts a 32 GB card)."""
    import jax, jax.numpy as jnp
    matvec, _s, _m = stacked(spec)
    J = np.asarray(J)

    @jax.jit
    def one(Jc):
        X = jnp.zeros((Jc.shape[0], spec.n), jnp.float32).at[jnp.arange(Jc.shape[0]), Jc].set(1.0)
        cols = jax.vmap(matvec)(X)
        return jnp.sum(cols * cols, axis=1)
    out = []
    for a in range(0, J.size, chunk):
        Jc = J[a:a + chunk]
        pad = chunk - Jc.size
        Jp = np.concatenate([Jc, np.zeros(pad, Jc.dtype)]) if pad else Jc
        out.append(np.asarray(one(jnp.asarray(Jp)), np.float64)[:Jc.size])
    return np.concatenate(out)


# ------------------------------------------------------------- points
def point_rows(spec, T, metrics):
    """Per-row sigma over the whole stacked dual, and per-term multiplicity."""
    parts, terms = [], []
    for j, (d, met) in enumerate(zip(spec.dual_blocks, metrics)):
        rows = d["op"].shape[0]
        if met[0] == "scalar":
            w = float(np.asarray(met[1], np.float64))
            parts.append(np.full(rows, w)); terms.append(("tied", rows, w))
        elif met[0] in ("vec", "diag"):
            v = np.asarray(met[1], np.float64).reshape(-1)
            if v.size != rows:
                raise RuntimeError(f"term {j}: {v.size} entries for {rows} rows")
            parts.append(v); terms.append(("rows", rows, None))
        else:
            raise RuntimeError("full metrics are outside the restricted comparison")
    tau = np.asarray(np.broadcast_to(np.asarray(T, np.float64), (spec.n,)), np.float64)
    return np.concatenate(parts), terms, tau


def objective_from(spec, sigma_rows, terms, tau, rho, c, r_tgt, s_tgt):
    """Exact F, the certified bound B, and the ray position, in float64."""
    u = 0.5 * np.log(np.maximum(sigma_rows, 1e-300))
    v = 0.5 * np.log(np.maximum(tau, 1e-300))
    tr_a, tr_b = float(tau @ c), float(sigma_rows @ rho)
    F = 0.5 * tr_a - r_tgt * u.sum() - s_tgt * v.sum() + 0.5 * GAMMA * (u @ u + v @ v)
    gv = tau * c - s_tgt + GAMMA * v
    Bsum = float(gv @ gv)
    a = 0
    for kind, rows, w in terms:
        ur, rr, sr = u[a:a + rows], rho[a:a + rows], sigma_rows[a:a + rows]
        if kind == "tied":
            g = float(np.sum(sr * rr)) - r_tgt * rows + GAMMA * rows * float(ur[0])
            Bsum += g * g / rows                     # W^{-1} weight of a tied scalar
        else:
            g = sr * rr - r_tgt + GAMMA * ur
            Bsum += float(g @ g)
        a += rows
    m = sigma_rows.size
    return dict(F=float(F), trace_tau_c=tr_a, trace_sigma_rho=tr_b,
                trace_rel_disagreement=abs(tr_a - tr_b) / max(abs(tr_a), 1e-300),
                B=Bsum / (2 * GAMMA), grad_v_inf=float(np.max(np.abs(gv))),
                ray_a=float((v.sum() - u.sum()) / (m + spec.n)),
                box_binding=int(np.sum(np.abs(u) >= M_BOX - 1e-6) + np.sum(np.abs(v) >= M_BOX - 1e-6)))


def tied_single_objective(d, w, tau, m, n, r_tgt, s_tgt):
    """F, B and ray position for a single tied term, from d = diag(K^T K)."""
    a = 0.5 * math.log(w)
    v = 0.5 * np.log(np.maximum(tau, 1e-300))
    S = float(tau @ d)
    F = 0.5 * w * S - r_tgt * m * a - s_tgt * v.sum() + 0.5 * GAMMA * (m * a * a + v @ v)
    ga = w * S - r_tgt * m + GAMMA * m * a
    gv = w * tau * d - s_tgt + GAMMA * v
    B = (ga * ga / m + float(gv @ gv)) / (2 * GAMMA)
    return dict(F=float(F), B=float(B), grad_a=float(ga), grad_v_inf=float(np.max(np.abs(gv))),
                ray_a=float((v.sum() - m * a) / (m + n)), tied_w=float(w))


def tied_single_optimum(d, m, n, r_tgt, s_tgt, log=print):
    """F* for a single tied term by exact block coordinate descent in float64,
    cross-checked by convex 1-D minimization over the tied log-scalar.  v depends
    on j only through d_j, so the v-step is solved on the unique values of d."""
    du, inv = np.unique(d, return_inverse=True)

    def v_of(w):
        return root64(w * du, s_tgt)[inv]

    def F_of(a, v):
        w = math.exp(2 * a); tau = np.exp(2 * v)
        return 0.5 * w * float(tau @ d) - r_tgt * m * a - s_tgt * v.sum() + 0.5 * GAMMA * (m * a * a + v @ v)

    v = np.zeros(n); a = 0.0; F_prev = F_of(a, v); rounds = 0
    for rounds in range(1, 20001):
        S = float(np.exp(2 * v) @ d)
        a_new = float(root64(np.array([S / m]), r_tgt)[0])
        v_new = v_of(math.exp(2 * a_new))
        step = max(abs(a_new - a), float(np.max(np.abs(v_new - v))))
        a, v = a_new, v_new
        if step < 1e-14:
            break
    F_bcd = F_of(a, v)
    lo, hi = -M_BOX, M_BOX
    for _ in range(300):                             # ternary search on a convex function
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if F_of(m1, v_of(math.exp(2 * m1))) <= F_of(m2, v_of(math.exp(2 * m2))):
            hi = m2
        else:
            lo = m1
    a_t = 0.5 * (lo + hi)
    F_ter = F_of(a_t, v_of(math.exp(2 * a_t)))
    log(f"    F* by exact BCD = {F_bcd:.10f} ({rounds} rounds, a = {a:+.10f}); "
        f"by 1-D convex search = {F_ter:.10f} (a = {a_t:+.10f}); "
        f"unique d values: {du.size}")
    return dict(F_star=F_bcd, F_star_1d=F_ter, a_star_opt=a, rounds=rounds,
                unique_d=int(du.size), agree=abs(F_bcd - F_ter))


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax, jax.numpy as jnp, ocs
    import unified_equil as UE
    import blockmetric_pilot as BP
    import weight_init_ridge as WR
    log = lambda s: print(s, flush=True)

    fa = argparse.Namespace(family=args.family, variant="overlap", n=200_000, k=4,
                            rho=args.rho, L=args.L, blur=0.7, channels=3, mu=1e-2, seed=0)
    if args.family == "F2":
        fa.variant = "strong"
    prob, spec = UE.build_family(fa)
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    r_tgt, s_tgt = math.sqrt(n / m), math.sqrt(m / n)
    k = spec.k
    classes = UE.resolve_classes(spec, "tied")
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} jax {jax.__version__} ocs {ocs.__version__}")
    log(f"# {spec.name}: n={n} m={m} r={r_tgt:.6f} s={s_tgt:.6f} gamma={GAMMA} M={M_BOX}; "
        f"classes {'|'.join(UE.class_tag(c) for c in classes)}")
    out = dict(meta=dict(family=args.family, spec=spec.name, n=n, m=m, k=k, rho=args.rho, L=args.L,
                         gamma=GAMMA, M=M_BOX, r_tgt=r_tgt, s_tgt=s_tgt, safety=SAFETY,
                         host=os.uname().nodename, jax=jax.__version__, ocs=ocs.__version__,
                         time=time.strftime("%Y-%m-%d %H:%M:%S %Z")), points={})

    def save():
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    # ---- the points, all pre-rescale
    pts = {}
    t0 = time.perf_counter()
    sc = BP.shipped_scaling(spec)
    T_S, met_S = UE.shipped_metrics(spec, sc, classes)
    undo = (sc["scaled_norm"] + 1e-30) / SAFETY
    met_S, T_S = BP.scale_metrics(met_S, T_S, undo)
    pts["S"] = (T_S, met_S, dict(builder="shipped SMFE (80 SGD steps x 2 probes)",
                                 scaled_norm=sc["scaled_norm"], undo_factor=undo,
                                 seconds=time.perf_counter() - t0))
    for tag, rounds, qg, qc in (("U3", 3, 16 * k, 16), ("U30", 30, 16 * k, 16),
                                ("U3big", 3, 256 * k, 256), ("Ubig", 30, 256 * k, 256)):
        t0 = time.perf_counter()
        T, met, _h, cnt, _tm, _r, _s = WR.ridged_equilibration(
            spec, classes, rounds, qg, qc, log=lambda *_: None, record=False)
        pts[tag] = (T, met, dict(builder=f"ridged restricted, {rounds} rounds, q_gram={qg}, q_col={qc}",
                                 applications=cnt["fwd"] + cnt["adj"],
                                 seconds=time.perf_counter() - t0))
        log(f"  built {tag}: {pts[tag][2]['builder']}")

    single_tied = len(spec.dual_blocks) == 1 and classes[0] == "tied"

    if single_tied:
        # ---- d = diag(K^T K), exactly
        B = spec.dual_blocks[0]["op"].shape[0] // k
        if args.family == "F1":
            rho1, d_sweep, sweep_s = column_sweep(spec, np.ones(m), np.ones(n), args.batch)
            d_ref = np.asarray(UE.exact_coldiag(spec, [("diag", np.ones((B, k)))]), np.float64)
            rel = float(np.max(np.abs(d_sweep - d_ref) / np.maximum(np.abs(d_ref), 1e-300)))
            log(f"  d by pure basis sweep ({sweep_s:.1f} s) vs closed-form exact_coldiag: "
                f"max relative difference {rel:.3e}")
            d = d_ref
            out["meta"]["d_check"] = dict(route="full basis sweep vs exact_coldiag",
                                          max_rel=rel, sweep_s=sweep_s)
        else:
            d = np.asarray(UE.exact_coldiag(spec, [("diag", np.ones((B, k)))]), np.float64)
            J = np.random.default_rng(7).choice(n, 256, replace=False)
            d_spot = spot_columns(spec, J)
            rel = float(np.max(np.abs(d_spot - d[J]) / np.maximum(np.abs(d[J]), 1e-300)))
            log(f"  d by closed-form exact_coldiag, spot-checked on 256 random columns by "
                f"pure forward applications: max relative difference {rel:.3e}")
            out["meta"]["d_check"] = dict(route="exact_coldiag, 256-column pure spot check", max_rel=rel)
        opt = tied_single_optimum(d, m, n, r_tgt, s_tgt, log=log)
        out["optimum"] = opt
        for tag, (T, met, info) in pts.items():
            w = float(np.asarray(met[0][1], np.float64))
            tau = np.asarray(np.broadcast_to(np.asarray(T, np.float64), (n,)), np.float64)
            res = tied_single_objective(d, w, tau, m, n, r_tgt, s_tgt)
            res.update(info)
            res["gap"] = res["F"] - opt["F_star"]
            res["gap_rel"] = res["gap"] / abs(opt["F_star"])
            res["bound_holds"] = bool(res["B"] >= res["gap"] - 1e-9 * abs(opt["F_star"]))
            out["points"][tag] = res
            log(f"  {tag:6s} F = {res['F']:.6f}  F - F* = {res['gap']:.6e} "
                f"({res['gap_rel']:.3e} relative)  certified bound B = {res['B']:.6e} "
                f"[B >= gap: {res['bound_holds']}]  ray a = {res['ray_a']:+.4e}")
            save()
    else:
        # ---- general case: one pure sweep per point
        for tag, (T, met, info) in pts.items():
            sig_rows, terms, tau = point_rows(spec, T, met)
            rho, c, sweep_s = column_sweep(spec, sig_rows, tau, args.batch)
            res = objective_from(spec, sig_rows, terms, tau, rho, c, r_tgt, s_tgt)
            res.update(info); res["sweep_s"] = sweep_s
            out["points"][tag] = res
            log(f"  {tag:6s} F = {res['F']:.6f}  certified bound B = {res['B']:.6e}  "
                f"trace two ways agree to {res['trace_rel_disagreement']:.2e}  "
                f"ray a = {res['ray_a']:+.4e}  (sweep {sweep_s:.1f} s)")
            save()
        Fs = {t: p["F"] for t, p in out["points"].items()}
        lower = max(p["F"] - p["B"] for p in out["points"].values())
        upper = min(Fs.values())
        out["optimum"] = dict(F_star_lower=lower, F_star_upper=upper,
                              note="certified bracket: upper = best evaluated point, "
                                   "lower = max over points of F - B")
        for tag, p in out["points"].items():
            p["gap_upper"] = p["F"] - lower           # certified: F - F* <= this
            p["gap_lower"] = p["F"] - upper           # certified: F - F* >= this
        log(f"  F* bracket: [{lower:.6f}, {upper:.6f}]")
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
