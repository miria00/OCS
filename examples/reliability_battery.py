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
    return sol.residual, abs(float(c @ xo) - ref) / (1 + abs(ref))


def run_qp(sz, seed):
    from scipy.optimize import lsq_linear
    r = np.random.default_rng(seed); m, n = 2 * sz, sz
    A = r.normal(size=(m, n)).astype("f"); b = r.normal(size=m).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    comp = ocs.minimize([ocs.box(x, 0, 1), ocs.least_squares(ocs.Dense(A) @ x - bp)]).compile(max_blocks=2500)
    sol = comp.solve_instance({"b": b}); xo = np.asarray(sol.x)
    o = 0.5 * np.sum((A @ xo - b) ** 2)
    xr = lsq_linear(A.astype("d"), b.astype("d"), bounds=(0, 1)).x
    ref = 0.5 * np.sum((A @ xr.astype("f") - b) ** 2)
    return sol.residual, abs(o - ref) / (1 + abs(ref))


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
    return sol.residual, abs(f(xo) - ref) / (1 + abs(ref))


def run_psd(sz, seed):
    # PSD-CONE PROJECTION (block validation): min 1/2||X - V||_F^2 s.t. X >= 0.
    # Validates the PSDCone prox against the EXACT projection (clip negative
    # eigenvalues of sym(V)).  This is block validation, not a full affine SDP --
    # OCS's end-to-end SDP path is not yet a supported feature.
    r = np.random.default_rng(seed); k = sz
    V = r.normal(size=(k, k)).astype("f"); V = (V + V.T) / 2
    x = ocs.Variable(k * k)
    # 1/2||X||^2 (=g) + cost (-V)^T x  ==  1/2||X - V||^2 - const, plus X >= 0
    comp = ocs.minimize([ocs.sum_squares(x), ocs.psd(x, k)],
                        cost=ocs.Parameter("nv")).compile(max_blocks=4000, tol=1e-5)
    sol = comp.solve_instance({"nv": -V.reshape(-1)}); Xo = np.asarray(sol.x).reshape(k, k)
    w, U = np.linalg.eigh(V); Xr = (U * np.maximum(w, 0.0)) @ U.T      # exact PSD projection
    f = lambda X: 0.5 * float(np.sum((X - V) ** 2))
    return sol.residual, abs(f(Xo) - f(Xr)) / (1 + abs(f(Xr)))


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
    return sol.residual, abs(o - ref) / (1 + abs(ref))


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
    return sol.residual, abs(o - ref) / (1 + abs(ref))


# Supported END-TO-END families: OCS solves the whole program and agrees with the ref.
SUPPORTED = [
    ("LP   min cᵀx, Ax=b, x≥0",     run_lp,       [8, 16, 32] if not FULL else [8, 16, 32, 64, 128]),
    ("QP   box least-squares",       run_qp,       [8, 16, 32] if not FULL else [8, 16, 32, 64, 128]),
    ("SOCP  min cᵀx, ‖Ax‖₂≤1",       run_socp,     [6, 12, 24] if not FULL else [6, 12, 24, 48]),
    ("Quantile regression",          run_quantile, [4, 8, 16]  if not FULL else [4, 8, 16, 32]),
    ("Logistic regression",          run_logistic, [4, 8, 16]  if not FULL else [4, 8, 16, 32]),
]
# EXPERIMENTAL: cone-block projection validation (the prox is validated by projecting
# onto the cone; the full end-to-end conic-program path is not yet a supported feature).
EXPERIMENTAL = [
    ("PSD-cone projection",          run_psd,      [3, 5, 8]   if not FULL else [3, 5, 8, 12]),
    ("Exp-cone projection",          run_expproj,  [1, 1, 1]),
    ("Power-cone projection",        run_powproj,  [1, 1, 1]),
]
SEEDS = range(3) if not FULL else range(6)


def _run_group(group):
    rows = []
    for name, fn, sizes in group:
        res_l, err_l = [], []
        for sz in sizes:
            for sd in SEEDS:
                try:
                    res, err = fn(sz, 100 * sz + sd)
                except Exception:
                    res, err = np.inf, np.inf
                res_l.append(float(res)); err_l.append(float(err))
        res_a, err_a = np.array(res_l), np.array(err_l); n = len(res_a)
        ok_r = np.isfinite(res_a); ok_e = np.isfinite(err_a)      # non-finite = not solved
        reached = int(np.sum(res_a[ok_r] < TOL)); agree = int(np.sum((res_a < TOL) & (err_a < RELERR)))
        efin = err_a[ok_e]; rfin = res_a[ok_r]
        me = float(np.median(efin) if efin.size else np.inf); we = float(np.max(efin) if efin.size else np.inf)
        mr = float(np.median(rfin) if rfin.size else np.inf)
        print(f"{name:30s} {n:6d} {reached:5d}/{n:<4d} {reached/n:>3.0%} {agree:5d}/{n:<4d} {agree/n:>3.0%} "
              f"{me:9.1e} {we:10.1e} {mr:9.1e}")
        rows.append((name, n, reached, agree, me, we))
    return rows


def _figure(sup, exp):
    """Render the two-group solve-rate table as plots/A2_reliability_battery.png."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    cols = ["problem class", "trials", "reached tol", "agrees w/ ref", "med rel-err", "worst rel-err"]

    def block(rows, header):
        out = [[header, "", "", "", "", ""]]
        out += [[nm, str(n), f"{rc}/{n} ({rc/n:.0%})", f"{ag}/{n} ({ag/n:.0%})", f"{me:.1e}", f"{we:.1e}"]
                for (nm, n, rc, ag, me, we) in rows]
        return out

    cell = block(sup, "SUPPORTED  (end-to-end solve)") \
        + block(exp, "EXPERIMENTAL  (cone-block projection)")
    fig, ax = plt.subplots(figsize=(12.2, 4.3)); ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, cellLoc="center", loc="center",
                   colWidths=[0.30, 0.08, 0.16, 0.16, 0.13, 0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for j in range(len(cols)): tbl[0, j].set_facecolor("#e8e8e8")
    hdr_rows = [1, 2 + len(sup)]
    for i in range(1, len(cell) + 1):
        tbl[i, 0].set_text_props(ha="left")
        if i in hdr_rows:
            for j in range(len(cols)):
                tbl[i, j].set_facecolor("#d9e6f2"); tbl[i, j].set_text_props(weight="bold", ha="left")
    Ts = sum(r[1] for r in sup); Rs = sum(r[2] for r in sup); As = sum(r[3] for r in sup)
    ax.set_title(f"OCS reliability battery  (float32, RTX 4090)\n"
                 f"supported families: {As}/{Ts} agree with the reference ({As/Ts:.0%}).  "
                 f"'reached tol' = residual < {TOL:g} (OCS termination); 'agrees w/ ref' = obj rel-err < {RELERR:g} "
                 f"vs CVXPY / exact projection", fontsize=10.5, pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(plots, "A2_reliability_battery.png"), dpi=150); plt.close(fig)
    print(f"-> wrote {os.path.join(plots, 'A2_reliability_battery.png')}")


def main():
    print("OCS reliability battery")
    print(f"  'reached tol'  = matrix-free residual < {TOL:g}  (OCS's own termination)")
    print(f"  'agrees w/ref' = objective relative error < {RELERR:g}  (vs CVXPY / exact projection)")
    print(f"{'class':30s} {'trials':>6s} {'reached tol':>11s} {'agrees w/ref':>12s} {'med err':>9s} {'worst err':>10s} {'med res':>9s}")
    print("=" * 96)
    print("SUPPORTED — end-to-end solve:")
    sup = _run_group(SUPPORTED)
    print("EXPERIMENTAL — cone-block projection validation:")
    exp = _run_group(EXPERIMENTAL)
    Ts, Rs, As = (sum(r[i] for r in sup) for i in (1, 2, 3))
    print("=" * 96)
    print(f"{'SUPPORTED TOTAL':30s} {Ts:6d} {Rs:5d}/{Ts:<4d} {Rs/Ts:>3.0%} {As:5d}/{Ts:<4d} {As/Ts:>3.0%}")
    _figure(sup, exp)


if __name__ == "__main__":
    t = time.perf_counter(); main(); print(f"[{time.perf_counter()-t:.0f}s]")
