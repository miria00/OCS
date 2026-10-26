"""Reliability battery: sweep OCS across problem CLASSES and SIZES, report a
solve-rate table (fraction that reach tolerance and agree with a reference).

Each class is modeled NATIVELY in both OCS and a reference solver (scipy for
LP/QP, CVXPY/SCS for the cones and the regression losses) and compared on the
objective — so the cone blocks (SOC, PSD, exp, power) and the new regression
blocks (pinball, logistic) are numerically diffed against SCS, not just asserted.

A trial "solves" when its matrix-free residual is below `TOL` AND its objective
relative error vs the reference is below `RELERR`.

Run:  python examples/reliability_battery.py            # modest sweep
      OCS_BATTERY_FULL=1 python examples/reliability_battery.py   # larger sweep

Exp/power use 14 distinct targeted regression seeds by default, or all
103 distinct seeds 0 through 102 in the full sweep.
"""
import os, time
import numpy as np
import scipy.sparse as _sp
import jax.numpy as jnp
import ocs

# Newer scipy removed the deprecated sparse ``.A`` attribute that older CVXPY
# builds still use during SCS canonicalization; restore it so the reference
# solves run under any environment.
for _cls in (_sp.csc_matrix, _sp.csr_matrix, _sp.coo_matrix):
    if not hasattr(_cls, "A"):
        _cls.A = property(lambda self: self.toarray())

TOL, RELERR = 1e-3, 1e-2         # reached-tolerance bar; objective-agreement bar
FULL = bool(os.environ.get("OCS_BATTERY_FULL"))
REFINE = os.environ.get("OCS_BATTERY_REFINE", "1") != "0"   # auto refinement post-step


def _maybe_refine(comp, sol, data, obj_fn, ref):
    """Objective error after the matrix-free Gauss-Newton refinement post-step
    (None when refinement is unavailable for the family or disabled)."""
    if not REFINE:
        return None
    try:
        xr, _yr = comp.refine(sol.x, sol.y, data, steps=5, cg_iters=200)
        return abs(obj_fn(np.asarray(xr)) - ref) / (1 + abs(ref))
    except Exception:
        return None


# ---- per-class generators: (size, seed) -> (residual, rel_err) --------------
def run_lp(sz, seed):
    from scipy.optimize import linprog
    r = np.random.default_rng(seed); m, n = sz, 2 * sz
    A = r.normal(size=(m, n)).astype("f"); xt = np.abs(r.normal(size=n)).astype("f")
    b = A @ xt; c = np.abs(r.normal(size=n)).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b"); cc = ocs.Parameter("c")
    comp = ocs.minimize([ocs.nonneg(x), ocs.equality(ocs.Dense(A) @ x - bp)], cost=cc).compile(max_blocks=1500)
    sol = comp.solve_instance({"b": b, "c": c}); xo = np.asarray(sol.x)
    ref = float(linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * n).fun)
    obj = lambda z: float(c @ z)
    return (sol.residual, abs(obj(xo) - ref) / (1 + abs(ref)),
            _maybe_refine(comp, sol, {"b": b, "c": c}, obj, ref))


def run_qp(sz, seed):
    from scipy.optimize import lsq_linear
    r = np.random.default_rng(seed); m, n = 2 * sz, sz
    A = r.normal(size=(m, n)).astype("f"); b = r.normal(size=m).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    comp = ocs.minimize([ocs.box(x, 0, 1), ocs.least_squares(ocs.Dense(A) @ x - bp)]).compile(max_blocks=2500)
    sol = comp.solve_instance({"b": b}); xo = np.asarray(sol.x)
    obj = lambda z: 0.5 * float(np.sum((A @ z - b) ** 2))
    xr = lsq_linear(A.astype("d"), b.astype("d"), bounds=(0, 1)).x
    ref = 0.5 * float(np.sum((A @ xr.astype("f") - b) ** 2))
    return (sol.residual, abs(obj(xo) - ref) / (1 + abs(ref)),
            _maybe_refine(comp, sol, {"b": b}, obj, ref))


def run_socp(sz, seed, mu=1e-2):
    import cvxpy as cp
    r = np.random.default_rng(seed); n = sz
    c = r.normal(size=n).astype("f"); A = r.normal(size=(n, n)).astype("f")
    x = ocs.Variable(n); cc = ocs.Parameter("c")
    # min c^T x + (mu/2)||x||^2  s.t. ||A x||_2 <= 1  ==  (1, A x) in SOC
    comp = ocs.minimize([ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, "f")) @ x),
                         ocs.soc(ocs.Dense(np.vstack([np.zeros((1, n), "f"), A])) @ x + np.concatenate([[1.0], np.zeros(n)]).astype("f"))],
                        cost=cc).compile(max_blocks=8000, tol=1e-5, adaptive_omega=False)
    sol = comp.solve_instance({"c": c}); xo = np.asarray(sol.x)
    f = lambda z: float(c @ z) + 0.5 * mu * np.sum(z * z)
    xv = cp.Variable(n); cp.Problem(cp.Minimize(c @ xv + 0.5 * mu * cp.sum_squares(xv)), [cp.norm(A @ xv, 2) <= 1]).solve(solver=cp.CLARABEL)
    ref = f(xv.value)
    return (sol.residual, abs(f(xo) - ref) / (1 + abs(ref)),
            _maybe_refine(comp, sol, {"c": c}, f, ref))


def run_psd(sz, seed):
    # PSD-CONE PROJECTION (block validation): min 1/2||X - V||_F^2 s.t. X >= 0.
    # Validates the PSDCone prox against the EXACT projection (clip negative
    # eigenvalues of sym(V)). This block validation is separate from the
    # supported end-to-end MaxCut SDP cases below.
    r = np.random.default_rng(seed); k = sz
    V = r.normal(size=(k, k)).astype("f"); V = (V + V.T) / 2
    x = ocs.Variable(k * k)
    # 1/2||X||^2 (=g) + cost (-V)^T x  ==  1/2||X - V||^2 - const, plus X >= 0
    comp = ocs.minimize([ocs.sum_squares(x), ocs.psd(x, k)],
                        cost=ocs.Parameter("nv")).compile(max_blocks=4000, tol=1e-5)
    sol = comp.solve_instance({"nv": -V.reshape(-1)}); Xo = np.asarray(sol.x).reshape(k, k)
    w, U = np.linalg.eigh(V); Xr = (U * np.maximum(w, 0.0)) @ U.T      # exact PSD projection
    f = lambda X: 0.5 * float(np.sum((X - V) ** 2))
    fx = lambda z: f(np.asarray(z).reshape(k, k))
    ref = f(Xr)
    return (sol.residual, abs(f(Xo) - ref) / (1 + abs(ref)),
            _maybe_refine(comp, sol, {"nv": -V.reshape(-1)}, fx, ref))


def _proj_cone(term_builder, v, seed):
    """min 1/2||x - v||^2 s.t. x in cone;  return OCS (res, obj) with x = proj(v)."""
    n = v.shape[0]; x = ocs.Variable(n); vp = ocs.Parameter("v")
    # 1/2||x||^2 (=g) + cost (-v)^T x  ==  1/2||x-v||^2 - const
    comp = ocs.minimize([ocs.sum_squares(x), term_builder(x)], cost=ocs.Parameter("nv")).compile(max_blocks=4000)
    sol = comp.solve_instance({"nv": -v}); xo = np.asarray(sol.x)
    return sol.residual, 0.5 * float(np.sum((xo - np.asarray(v)) ** 2))


def run_expproj(sz, seed):
    import cvxpy as cp
    r = np.random.default_rng(seed); v = r.normal(size=3).astype("f")
    res, o = _proj_cone(lambda x: ocs.exp_cone(x), v, seed)
    z = cp.Variable(3); cp.Problem(cp.Minimize(0.5 * cp.sum_squares(z - v)), [cp.constraints.ExpCone(z[0], z[1], z[2])]).solve(solver=cp.SCS)
    ref = 0.5 * float(np.sum((z.value - v) ** 2))
    return res, abs(o - ref) / (1 + abs(ref))


def run_powproj(sz, seed):
    import cvxpy as cp
    r = np.random.default_rng(seed); v = r.normal(size=3).astype("f"); a = 0.5
    res, o = _proj_cone(lambda x: ocs.pow_cone(x, a), v, seed)
    z = cp.Variable(3); cp.Problem(cp.Minimize(0.5 * cp.sum_squares(z - v)), [cp.constraints.PowCone3D(z[0], z[1], z[2], a)]).solve(solver=cp.SCS)
    ref = 0.5 * float(np.sum((z.value - v) ** 2))
    return res, abs(o - ref) / (1 + abs(ref))


def run_quantile(sz, seed):
    import cvxpy as cp
    r = np.random.default_rng(seed); m, n, tau, mu = 20 * sz, sz, 0.75, 1e-2
    X = r.normal(size=(m, n)).astype("f"); tt = r.normal(size=n).astype("f")
    y = (X @ tt + 0.3 * r.standard_t(3, size=m)).astype("f")
    th = ocs.Variable(n); yp = ocs.Parameter("y")
    comp = ocs.minimize([ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, "f")) @ th),
                         ocs.pinball(ocs.Dense(-X) @ th + yp, tau)]).compile(max_blocks=2000, tol=1e-6)
    sol = comp.solve_instance({"y": y}); f = lambda t: np.sum(np.maximum(tau * (y - X @ t), (tau - 1) * (y - X @ t))) + 0.5 * mu * np.sum(t * t)
    tv = cp.Variable(n); rr = y - X @ tv
    cp.Problem(cp.Minimize(cp.sum(cp.maximum(tau * rr, (tau - 1) * rr)) + 0.5 * mu * cp.sum_squares(tv))).solve(solver=cp.SCS)
    o = f(np.asarray(sol.x)); ref = f(tv.value)
    return (sol.residual, abs(o - ref) / (1 + abs(ref)),
            _maybe_refine(comp, sol, {"y": y}, f, ref))


def run_logistic(sz, seed):
    import cvxpy as cp
    r = np.random.default_rng(seed); m, n, mu = 20 * sz, sz, 0.1
    X = r.normal(size=(m, n)).astype("f"); tt = r.normal(size=n).astype("f")
    b = np.where(X @ tt + 0.5 * r.normal(size=m) > 0, 1.0, -1.0).astype("f")
    K = (-b[:, None] * X).astype("f"); th = ocs.Variable(n)
    comp = ocs.minimize([ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, "f")) @ th),
                         ocs.logistic(ocs.Dense(K) @ th)]).compile(max_blocks=2500, tol=1e-6)
    sol = comp.solve_instance({}); g = lambda t: np.sum(np.log1p(np.exp(K @ t))) + 0.5 * mu * np.sum(t * t)
    tv = cp.Variable(n); cp.Problem(cp.Minimize(cp.sum(cp.logistic(K @ tv)) + 0.5 * mu * cp.sum_squares(tv))).solve(solver=cp.SCS)
    o = g(np.asarray(sol.x)); ref = g(tv.value)
    # refinement excluded: the logistic prox is an inner Newton loop, and
    # differentiating through its unrolled iterations blows up XLA compilation
    return sol.residual, abs(o - ref) / (1 + abs(ref)), None


def run_sdp_e2e(sz, seed):
    """END-TO-END semidefinite program: the MaxCut relaxation
    max <L,X> s.t. X psd, diag(X) = 1, solved as a full conic program and
    compared with CVXPY/SCS.  The k x k^2 diagonal-extraction operator is
    applied matrix-free (never formed)."""
    import cvxpy as cp
    k = sz
    r = np.random.default_rng(seed)
    W = r.normal(size=(k, k)).astype("f"); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    L = (np.diag(W.sum(1)) - W).astype("f") / 4.0

    class _Diag(ocs.LinearOperator):
        def __init__(self, k):
            super().__init__((k, k * k)); self.k = k
        def matvec(self, v):
            return jnp.diag(v.reshape(self.k, self.k))
        def rmatvec(self, u):
            return jnp.diag(u).reshape(-1)

    X = ocs.Variable(k * k)
    comp = ocs.minimize([ocs.psd(X, k),
                         ocs.equality(_Diag(k) @ X - ocs.Parameter("d"))],
                        cost=(-L).reshape(-1)).compile(max_blocks=3000, tol=1e-3)
    sol = comp.solve_instance({"d": np.ones(k, np.float32)})
    Xs = np.asarray(sol.x).reshape(k, k); Xs = (Xs + Xs.T) / 2
    o = float((L * Xs).sum())
    Z = cp.Variable((k, k), symmetric=True)
    cp.Problem(cp.Maximize(cp.sum(cp.multiply(L, Z))),
               [Z >> 0, cp.diag(Z) == 1]).solve(solver=cp.SCS)
    ref = float((L * Z.value).sum())
    return sol.residual, abs(o - ref) / (1 + abs(ref)), None


# Supported END-TO-END families: OCS solves the whole program and agrees with the ref.
SUPPORTED = [
    ("LP   min cᵀx, Ax=b, x≥0",     run_lp,       [8, 16, 32] if not FULL else [8, 16, 32, 64, 128]),
    ("QP   box least-squares",       run_qp,       [8, 16, 32] if not FULL else [8, 16, 32, 64, 128]),
    ("SOCP  min cᵀx, ‖Ax‖₂≤1",       run_socp,     [6, 12, 24] if not FULL else [6, 12, 24, 48]),
    ("Quantile regression",          run_quantile, [4, 8, 16]  if not FULL else [4, 8, 16, 32]),
    ("Logistic regression",          run_logistic, [4, 8, 16]  if not FULL else [4, 8, 16, 32]),
    ("SDP  MaxCut relaxation",       run_sdp_e2e,  [10, 20, 30] if not FULL else [10, 20, 30, 60]),
]
# Additional cone-block projection validation. Exp/power remain experimental;
# the PSD projection checks supplement the supported MaxCut SDP cases above.
# Exp/power use actual seed identities, including failures found by the
# independent 103-seed sweep. PSD retains its historical size/seed schedule.
EXPERIMENTAL = [
    ("PSD-cone projection",          run_psd,      [3, 5, 8]   if not FULL else [3, 5, 8, 12]),
    ("Exp-cone projection",          run_expproj,  [1]),
    ("Power-cone projection",        run_powproj,  [1]),
]
SEEDS = range(3) if not FULL else range(6)
CONE_SEEDS = range(103) if FULL else (0, 1, 2, 33, 50, 67, 69, 70, 72, 77, 93, 100, 101, 102)


def _case_schedule(name, sizes):
    # Use the stable group label so instrumentation can wrap its generator.
    if name in ("Exp-cone projection", "Power-cone projection"):
        return [(sz, seed) for sz in sizes for seed in CONE_SEEDS]
    return [(sz, 100 * sz + sd) for sz in sizes for sd in SEEDS]


def _run_group(group):
    rows = []
    for name, fn, sizes in group:
        res_l, err_l, ref_l = [], [], []
        cases = _case_schedule(name, sizes)
        for sz, seed in cases:
            try:
                out = fn(sz, seed)
                res, err = out[0], out[1]
                eref = out[2] if len(out) > 2 else None
            except Exception:
                res, err, eref = np.inf, np.inf, None
            res_l.append(float(res)); err_l.append(float(err))
            ref_l.append(np.nan if eref is None else float(eref))
        res_a, err_a, ref_a = np.array(res_l), np.array(err_l), np.array(ref_l)
        n = len(res_a)
        unique = len(set(cases))
        ok_r = np.isfinite(res_a); ok_e = np.isfinite(err_a)      # non-finite = not solved
        reached = int(np.sum(res_a[ok_r] < TOL))
        agree = int(np.sum(ok_e & (err_a < RELERR)))
        joint = int(np.sum(ok_r & ok_e & (res_a < TOL) & (err_a < RELERR)))
        # effective error after the refinement post-step (falls back to the raw error)
        eff = np.where(np.isfinite(ref_a), np.minimum(err_a, ref_a), err_a)
        joint2 = int(np.sum(ok_r & np.isfinite(eff) & (res_a < TOL) & (eff < RELERR)))
        rfin_e = ref_a[np.isfinite(ref_a)]
        wref = float(np.max(rfin_e)) if rfin_e.size else float("nan")
        efin = err_a[ok_e]; rfin = res_a[ok_r]
        me = float(np.median(efin) if efin.size else np.inf); we = float(np.max(efin) if efin.size else np.inf)
        mr = float(np.median(rfin) if rfin.size else np.inf)
        wref_s = f"{wref:9.1e}" if np.isfinite(wref) else "        —"
        print(f"{name:30s} {n:6d} {unique:6d} {reached:4d}/{n:<4d} "
              f"{agree:4d}/{n:<4d} {joint:4d}/{n:<4d} "
              f"{me:9.1e} {we:10.1e} {mr:9.1e} {joint2:4d}/{n:<4d} {wref_s}")
        # Keep the historical joint-count indices for downstream summaries;
        # append the independent objective count and distinct-instance count.
        rows.append((name, n, reached, joint, me, we, joint2, wref, agree, unique))
        # refinement doubles the per-instance XLA compilation churn; drop the
        # accumulated compiled graphs between families to keep memory bounded
        import jax; jax.clear_caches()
    return rows


def _figure(sup, exp, *, device_label=None):
    """Render the two-group solve-rate table as plots/A2_reliability_battery.png."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    cols = ["problem class", "trials / unique", "residual bar", "objective agrees",
            "joint pass", "med rel-err", "worst rel-err", "raw res + best obj",
            "worst refined"]

    def block(rows, header):
        out = [[header] + [""] * (len(cols) - 1)]
        out += [[nm, f"{n} / {uq}", f"{rc}/{n} ({rc/n:.0%})", f"{ag}/{n} ({ag/n:.0%})",
                 f"{jt}/{n} ({jt/n:.0%})", f"{me:.1e}",
                 f"{we:.1e}", f"{a2}/{n} ({a2/n:.0%})",
                 (f"{wr:.1e}" if np.isfinite(wr) else "—")]
                for (nm, n, rc, jt, me, we, a2, wr, ag, uq) in rows]
        return out

    cell = block(sup, "SUPPORTED  (end-to-end solve)") \
        + block(exp, "CONE PROJECTION CHECKS")
    fig, ax = plt.subplots(figsize=(17.5, 4.8)); ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, cellLoc="center", loc="center",
                   colWidths=[0.22, 0.08, 0.10, 0.11, 0.09, 0.09, 0.10, 0.12, 0.09])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for j in range(len(cols)): tbl[0, j].set_facecolor("#e8e8e8")
    hdr_rows = [1, 2 + len(sup)]
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_text_props(ha="left")
        if i in hdr_rows:
            for j in range(len(cols)):
                tbl[i, j].set_facecolor("#d9e6f2"); tbl[i, j].set_text_props(weight="bold", ha="left")
    Ts = sum(r[1] for r in sup); Js = sum(r[3] for r in sup)
    if device_label is None:
        import jax
        device_label = ", ".join(str(d) for d in jax.devices())
    ax.set_title(f"OCS reliability battery  (float32, {device_label})\n"
                 f"supported families: {Js}/{Ts} pass both checks ({Js/Ts:.0%}); "
                 f"residual < {TOL:g}, objective relative error < {RELERR:g}.\n"
                 "Individual requested solve tolerances can be tighter than the residual bar.\n"
                 "Exp/power remain experimental. Unique counts record distinct instances; "
                 "'raw res + best obj' does not certify the refined point.", fontsize=10.5, pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(plots, "A2_reliability_battery.png"), dpi=150); plt.close(fig)
    print(f"-> wrote {os.path.join(plots, 'A2_reliability_battery.png')}")


def main():
    print("OCS reliability battery")
    print(f"  'res bar'    = returned matrix-free residual < {TOL:g}; individual solve tolerances can be tighter")
    print(f"  'obj agree'  = objective relative error < {RELERR:g}, independently of residual")
    print("  'joint'      = both residual and objective checks pass")
    print("  'unique'     = distinct (size, seed) instances")
    print("  exp/power   = " + ("full distinct sweep: seeds 0 through 102" if FULL else
          "targeted regression coverage: 14 distinct seeds, including prior failures"))
    print(f"  'raw+best'   = raw residual passes and best raw/refined objective agrees; no refined-point certificate"
          + ("" if REFINE else "  [DISABLED: OCS_BATTERY_REFINE=0]"))
    print(f"{'class':30s} {'trials':>6s} {'unique':>6s} {'res bar':>9s} {'obj agree':>9s} {'joint':>9s} "
          f"{'med err':>9s} {'worst err':>10s} {'med res':>9s} {'raw+best':>9s} {'worst ref':>9s}")
    print("=" * 134)
    print("SUPPORTED — end-to-end solve:")
    sup = _run_group(SUPPORTED)
    print("CONE PROJECTIONS — PSD block checks; exp/power remain experimental:")
    exp = _run_group(EXPERIMENTAL)
    Ts, Rs, Js, J2s, As, Us = (sum(r[i] for r in sup) for i in (1, 2, 3, 6, 8, 9))
    print("=" * 134)
    print(f"{'SUPPORTED TOTAL':30s} {Ts:6d} {Us:6d} {Rs:4d}/{Ts:<4d} "
          f"{As:4d}/{Ts:<4d} {Js:4d}/{Ts:<4d}{'':31s} {J2s:4d}/{Ts:<4d}")
    _figure(sup, exp)


if __name__ == "__main__":
    t = time.perf_counter(); main(); print(f"[{time.perf_counter()-t:.0f}s]")
