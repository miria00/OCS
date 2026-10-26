"""Equilibration measurement harness (baseline instrument).

For each problem and each equilibrator kind (scalar vs SMFE), report:
  iters-to-tol, residual, compile time (includes equilibration), solve time,
  and the nvar equilibration diagnostic (0 = perfectly equilibrated) pre/post.

This is the a/b measurement bench that every equilibration change (one-probe
SMFE, Jacobi tier, online warmup-refine) is judged on.  Run BEFORE changing
anything to capture the baseline.

Run:  /home/miria/jaxenv2/bin/python examples/equil_harness.py
"""
import os, time, json
import numpy as np
import ocs

TOL, MAXB = 1e-3, 400          # iters-to-tol bar; budget = MAXB * block(=100)


def bench(name, make, kind):
    prob, data = make()
    t0 = time.perf_counter()
    comp = prob.compile(max_blocks=MAXB, tol=TOL, equilibrate=kind)
    t_comp = time.perf_counter() - t0                 # includes equilibration
    comp.solve_instance(data)                         # warm (JIT compile)
    t0 = time.perf_counter()
    sol = comp.solve_instance(data)
    t_solve = time.perf_counter() - t0
    d = comp.meta.get("equil_diag", {}) or {}
    return {"name": name, "kind": kind, "iters": sol.iters,
            "residual": sol.residual, "t_compile": t_comp, "t_solve": t_solve,
            "nvar_row_pre": d.get("nvar_row_pre"), "nvar_col_pre": d.get("nvar_col_pre"),
            "nvar_row_post": d.get("nvar_row_post"), "nvar_col_post": d.get("nvar_col_post")}


# ---- problem generators: () -> (prob, data) --------------------------------
_GA = np.random.default_rng(0).normal(size=(240, 180)).astype(np.float32)

def make_dense_nnls(R):
    def _f():
        r = np.random.default_rng(7)
        s = (R ** r.uniform(0.0, 1.0, size=180)).astype(np.float32)   # column ill-cond
        A = _GA * s[None, :]
        b = (A @ np.abs(r.normal(size=180)).astype(np.float32)).astype(np.float32)
        x = ocs.Variable(180); bp = ocs.Parameter("b")
        return ocs.minimize([ocs.nonneg(x), ocs.least_squares(ocs.Dense(A) @ x - bp)]), {"b": b}
    return _f

_FB = ocs.FFTMask.random(1024, ratio=0.5, seed=0)
_MB = _FB.shape[0]

def make_fft_nnls(R):
    def _f():
        r = np.random.default_rng(9)
        w = (R ** r.uniform(0.0, 1.0, size=_MB)).astype(np.float32)   # row ill-cond
        K = ocs.Weighted(_FB, w)
        xt = np.zeros(1024, np.float32)
        xt[r.choice(1024, 51, replace=False)] = np.abs(r.normal(size=51)).astype(np.float32)
        c = np.asarray(K.matvec(xt)).astype(np.float32)
        x = ocs.Variable(1024); cp = ocs.Parameter("c")
        return ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - cp)]), {"c": c}
    return _f

def make_lp(sz=120):
    def _f():
        r = np.random.default_rng(3); m, n = sz, 2 * sz
        A = r.normal(size=(m, n)).astype("f"); xt = np.abs(r.normal(size=n)).astype("f")
        b = A @ xt; c = np.abs(r.normal(size=n)).astype("f")
        x = ocs.Variable(n); bp = ocs.Parameter("b"); cc = ocs.Parameter("c")
        return ocs.minimize([ocs.nonneg(x), ocs.equality(ocs.Dense(A) @ x - bp)],
                            cost=cc), {"b": b, "c": c}
    return _f

def make_qp(sz=120):
    def _f():
        r = np.random.default_rng(5); m, n = 2 * sz, sz
        A = r.normal(size=(m, n)).astype("f"); b = r.normal(size=m).astype("f")
        x = ocs.Variable(n); bp = ocs.Parameter("b")
        return ocs.minimize([ocs.box(x, 0, 1), ocs.least_squares(ocs.Dense(A) @ x - bp)]), {"b": b}
    return _f


PROBLEMS = [
    ("dense-nnls R=30",  make_dense_nnls(30.0)),
    ("dense-nnls R=100", make_dense_nnls(100.0)),
    ("fft-nnls R=30",    make_fft_nnls(30.0)),
    ("fft-nnls R=100",   make_fft_nnls(100.0)),
    ("lp",               make_lp()),
    ("qp",               make_qp()),
]

if __name__ == "__main__":
    rows = []
    hdr = (f"{'problem':18s} {'kind':7s} {'iters':>7s} {'residual':>11s} "
           f"{'t_comp(s)':>10s} {'t_solve(s)':>11s} {'nvar_post(r,c)':>18s}")
    print(hdr); print("-" * len(hdr))
    for name, make in PROBLEMS:
        for kind in ("scalar", "sgd"):
            row = bench(name, make, kind)
            rows.append(row)
            nvp = ("" if row["nvar_row_post"] is None
                   else f"{row['nvar_row_post']:.3f},{row['nvar_col_post']:.3f}")
            print(f"{row['name']:18s} {row['kind']:7s} {row['iters']:7d} "
                  f"{row['residual']:11.2e} {row['t_compile']:10.3f} "
                  f"{row['t_solve']:11.4f} {nvp:>18s}")
    out = os.path.join(os.path.dirname(__file__), "output", "equil_harness_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("-> wrote", out)
