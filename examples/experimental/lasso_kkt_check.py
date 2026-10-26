"""Residual-floor diagnostic for the dense lasso of examples/pogs_lasso.py.

Appendix H reports that at a requested 1e-5 OCS's float32 residual floors near
2.7e-5 (800x1600) and 1e-4 (4000x8000) while the objective error is ~1e-9.
Objective agreement cannot tell whether that plateau is roundoff in the
float32 residual evaluation or a real KKT error at the returned point.  This
script decides it in two separate processes:

  --side ocs     (GPU, jax + ocs)  solve both sizes exactly as pogs_lasso.run_ocs
                 does (same instance(), same compile keywords, tol 1e-5, the
                 script's block=100 / max_blocks=400 budget), save the returned
                 float32 primal-dual pair and the certificate OCS reports.
  --side check   (numpy only)      re-evaluate the lasso KKT residual at the
                 saved pair in float64 and, with the same formulas, in float32.

The lasso   minimize (1/2)||Ax-b||_2^2 + lam ||x||_1   has KKT conditions
    y = Ax - b            (least-squares block dual: the gradient of the data term)
    -A^T y in lam d||x||_1   (stationarity; OCS routes the identity-block l1 term
                             out of the dual stack, so its dual is -A^T y)
The stationarity distance per coordinate is
    |(A^T y)_i + lam sign(x_i)|   for x_i != 0,   max(0, |(A^T y)_i| - lam)  for x_i == 0.

Scales: OCS (ocs/solver/stopping.py, make_relative_components) divides the
stationarity error by max(||A^T y||_2, ||A^T b||_2) (the pullback scale and the
pulled-back anchor grad f(l) = -b; the cost is zero) and the smooth-block error
||y - (Ax-b)||_2 by max(||y||_2, ||Ax-b||_2, ||b||_2).  Those are reported as
"ocs_scale"; the simpler 1+||A^T y||_inf and 1+||b||_2 scales are reported too.
OCS's own l1 check is a balanced graph distance, not this subgradient
distance, so the two are the same quantity only up to that choice of metric.

Outputs (all under --out):
    lasso_pair_<m>x<n>.npz      x, y (float32), residual, iters, status, certificate json
    lasso_kkt_check.csv         one row per (size, precision, dual candidate)
"""
import os, sys, csv, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # examples/: pogs_lasso, guarded
import pogs_lasso                                   # numpy-only at import time

SIZES = [(800, 1600), (4000, 8000)]
TOL = float(os.environ.get("LASSO_TOL", 1e-5))     # 1e-5 is the appendix setting; an unreachable value forces the float32 floor
TAG = os.environ.get("LASSO_TAG", "")
BLOCK, MAX_BLOCKS = 100, 400                        # the script's budget: 40,000 iterations


def side_ocs(out):
    import jax, jax.numpy as jnp, ocs
    for m, n in SIZES:
        A, b, lam = pogs_lasso.instance(m, n)
        x = ocs.Variable(n); bp = ocs.Parameter("b")
        t0 = time.perf_counter()
        comp = ocs.minimize([ocs.least_squares(ocs.Dense(A.astype("f")) @ x - bp),
                             ocs.l1(ocs.Identity(n) @ x, float(lam))]).compile(
            block=BLOCK, max_blocks=MAX_BLOCKS, tol=TOL, equilibrate="sgd")
        compile_s = time.perf_counter() - t0
        bj = jnp.asarray(b.astype("f"))
        s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)     # warm (identical cold start)
        t0 = time.perf_counter()
        s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
        dt = time.perf_counter() - t0
        cert = comp.certificate(s.x, s.y, {"b": bj})
        xs = np.asarray(s.x, np.float32); ys = np.asarray(s.y, np.float32)
        obj = pogs_lasso.objective(A, b, lam, xs)
        print(f"{m}x{n}: tol {TOL:g} {dt:.3f}s iters={s.iters} residual={s.residual:.4e} "
              f"status={s.status} obj={obj:.10f} y.shape={ys.shape} cert={cert}", flush=True)
        np.savez(os.path.join(out, f"lasso_pair_{m}x{n}{TAG}.npz"), x=xs, y=ys, m=m, n=n, lam=lam,
                 residual=s.residual, iters=s.iters, status=s.status, solve_s=dt,
                 compile_s=compile_s, objective=obj,
                 certificate=json.dumps({k: float(v) for k, v in cert.items()}),
                 jax_version=jax.__version__, device=str(jax.devices()[0]))


def graph_distance(t, s, lam, a):
    """Per-coordinate Euclidean distance from (t/a, s) to the graph of
    lam*d|.| drawn in the same coordinates: the segment {(0, s): |s| <= lam}
    and the two rays {(t/a, +lam): t >= 0}, {(t/a, -lam): t <= 0}.  With a = 1
    this is the plain graph distance; a = ||x||/||w|| balances the two axes the
    way OCS's balanced graph check does (up to its exact step choice)."""
    tp = t / a
    d_seg = np.sqrt(tp ** 2 + np.maximum(0, np.abs(s) - lam) ** 2)
    d_pos = np.sqrt(np.maximum(0, -tp) ** 2 + (s - lam) ** 2)
    d_neg = np.sqrt(np.maximum(0, tp) ** 2 + (s + lam) ** 2)
    return np.minimum(d_seg, np.minimum(d_pos, d_neg))


def kkt(A, b, lam, x, y_ls, dtype):
    """KKT residuals at (x, y_ls) in the given numpy dtype; y_ls is the
    candidate dual of the least-squares block."""
    A = A.astype(dtype); b = b.astype(dtype); x = x.astype(dtype); y = y_ls.astype(dtype)
    lam = dtype(lam)
    r = A @ x - b
    g = A.T @ y
    # (1) the sign-based subgradient distance exactly as specified: any nonzero
    #     x_i, however tiny, is charged |g_i + lam sign(x_i)|
    sg = np.where(x != 0, np.abs(g + lam * np.sign(x)), np.maximum(0, np.abs(g) - lam))
    # (2) the same with |x_i| <= 1e-6 max|x| treated as zero (float32 residue)
    tiny = np.abs(x) <= dtype(1e-6) * np.abs(x).max()
    sg_thr = np.where(~tiny, np.abs(g + lam * np.sign(x)), np.maximum(0, np.abs(g) - lam))
    # (3) the distance from (x_i, -g_i) to the graph of lam*d|.|, plain and balanced
    w = -g
    gd = graph_distance(x, w, lam, dtype(1))
    a = dtype(np.linalg.norm(x.astype(np.float64)) / max(np.linalg.norm(w.astype(np.float64)), 1e-300))
    gd_bal = graph_distance(x, w, lam, a)
    Atb = A.T @ b
    nrm = lambda v: float(np.linalg.norm(v.astype(np.float64)))
    stat_scale = max(nrm(g), nrm(Atb))
    worst = int(np.argmax(sg))
    return {
        "stat_l2": nrm(sg), "stat_inf": float(np.abs(sg).max()),
        "stat_ocs_scale": nrm(sg) / stat_scale,
        "stat_inf_scale": float(np.abs(sg).max()) / (1.0 + float(np.abs(g).max())),
        "stat_thr_ocs_scale": nrm(sg_thr) / stat_scale,
        "stat_thr_inf_scale": float(np.abs(sg_thr).max()) / (1.0 + float(np.abs(g).max())),
        "graph_dist_ocs_scale": nrm(gd) / stat_scale,
        "graph_dist_balanced_ocs_scale": nrm(gd_bal) / stat_scale,
        "n_tiny_nonzero": int((tiny & (x != 0)).sum()),
        "worst_coord": worst, "worst_x": float(x[worst]), "worst_g": float(g[worst]), "lam": float(lam),
        "prim_l2": nrm(y - r),
        "prim_ocs_scale": nrm(y - r) / max(nrm(y), nrm(r), nrm(b)),
        "prim_1pb_scale": nrm(y - r) / (1.0 + nrm(b)),
        "norm_Aty": nrm(g), "norm_Atb": nrm(Atb), "norm_r": nrm(r), "norm_y": nrm(y),
        "nnz_x": int(np.count_nonzero(x)),
    }


def side_check(out):
    rows = []
    for m, n in SIZES:
        f = os.path.join(out, f"lasso_pair_{m}x{n}{TAG}.npz")
        if not os.path.exists(f):
            print(f"missing {f}"); continue
        z = np.load(f, allow_pickle=False)
        A, b, lam = pogs_lasso.instance(m, n)
        assert abs(float(z["lam"]) - lam) < 1e-12 * lam
        x = np.asarray(z["x"]); y = np.asarray(z["y"])
        cert = json.loads(str(z["certificate"]))
        # y holds the least-squares block dual (m entries) and, if the l1 term
        # was kept in the dual stack, n further entries; record which.
        y_ls = y[:m]
        extra = y[m:] if y.shape[0] > m else None
        obj = pogs_lasso.objective(A, b, lam, x)
        ref = ""
        cache = os.path.join(out, f"clarabel_reference_{m}x{n}.json")
        try:
            if os.path.exists(cache):
                ref_val = json.load(open(cache))["objective"]
            else:
                t0 = time.perf_counter()
                ref_val, _ = pogs_lasso.reference(A, b, lam)
                json.dump({"objective": ref_val, "seconds": time.perf_counter() - t0}, open(cache, "w"))
            ref = f"{abs(obj - ref_val) / abs(ref_val):.3e}"
            print(f"{m}x{n}: Clarabel reference objective {ref_val:.10f}, OCS objective {obj:.10f}, rel err {ref}")
        except Exception as exc:  # noqa: BLE001
            print(f"{m}x{n}: no Clarabel reference ({type(exc).__name__}: {exc})")
        for prec, dtype in (("float64", np.float64), ("float32", np.float32)):
            for cand, yc in (("saved_y", y_ls), ("r=Ax-b", (A @ x.astype(np.float64) - b))):
                k = kkt(A, b, lam, x, yc, dtype)
                if extra is not None:
                    k["extra_dual_stat_ocs_scale"] = float(np.linalg.norm(
                        (A.T @ y_ls.astype(np.float64)) + extra.astype(np.float64))) / max(k["norm_Aty"], k["norm_Atb"])
                rows.append({"size": f"{m}x{n}", "precision": prec, "dual_candidate": cand,
                             "ocs_residual": float(z["residual"]), "ocs_iters": int(z["iters"]),
                             "ocs_status": str(z["status"]), "ocs_r_primal": cert.get("r_primal"),
                             "ocs_r_dual": cert.get("r_dual"), "ocs_gap": cert.get("gap"),
                             "objective": obj, "rel_obj_err_vs_clarabel": ref,
                             "y_len": int(y.shape[0]), **k})
                print(f"  {prec:7s} {cand:7s} stat/ocs_scale={k['stat_ocs_scale']:.3e} "
                      f"stat_inf/(1+|A^Ty|_inf)={k['stat_inf_scale']:.3e} "
                      f"thresholded={k['stat_thr_ocs_scale']:.3e} graph={k['graph_dist_ocs_scale']:.3e} "
                      f"graph_bal={k['graph_dist_balanced_ocs_scale']:.3e} tiny_nonzero={k['n_tiny_nonzero']} "
                      f"worst(i={k['worst_coord']}, x={k['worst_x']:.3e}, g={k['worst_g']:.4e}, lam={k['lam']:.4e}) "
                      f"prim/ocs_scale={k['prim_ocs_scale']:.3e} prim/(1+|b|)={k['prim_1pb_scale']:.3e}"
                      + (f" extra_dual_stat={k['extra_dual_stat_ocs_scale']:.3e}" if extra is not None else ""),
                      flush=True)
    if rows:
        f = os.path.join(out, f"lasso_kkt_check{TAG}.csv")
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["ocs", "check"], required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "plots"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (side_ocs if a.side == "ocs" else side_check)(a.out)
