"""Step 3 experiment: online equilibration (zero-setup) vs SMFE.

Four arms, all judged on iters-to-1e-3 and final residual on the same bench:
  (a) none         equilibrate=scalar                      (no equilibration)
  (b) SMFE         equilibrate=sgd                         (separate ~160-product pass)
  (c) SMFE+warmup  equilibrate=sgd,  online_warmup=W       (SMFE base, refined online)
  (d) online-only  equilibrate=scalar, online_warmup=W     (ZERO separate pass -- the goal:
                    the metric is refined from the free Kx, K^T y of a short warmup, then
                    frozen)

The question: does (d) match (b)'s iters-to-tol with NO separate equilibration pass?
If yes, online equilibration earns its place.  Operator-pure throughout.

Run:  PYTHONPATH=/home/miria/OCSolver/OCS /home/miria/jaxenv2/bin/python examples/equil_online_ab.py
"""
import os, json
import ocs
from equil_harness import (make_dense_nnls, make_fft_nnls, make_lp, make_qp)

TOL, MAXB, W = 1e-3, 400, 3

ARMS = [
    ("a none",        dict(equilibrate="scalar", online_warmup=0)),
    ("b SMFE",        dict(equilibrate="sgd",    online_warmup=0)),
    ("c SMFE+warmup", dict(equilibrate="sgd",    online_warmup=W)),
    ("d online-only", dict(equilibrate="scalar", online_warmup=W)),
]

PROBLEMS = [
    ("dense-nnls R=30",  make_dense_nnls(30.0)),
    ("dense-nnls R=100", make_dense_nnls(100.0)),
    ("fft-nnls R=30",    make_fft_nnls(30.0)),
    ("fft-nnls R=100",   make_fft_nnls(100.0)),
    ("lp",               make_lp()),
    ("qp",               make_qp()),
]


def bench(make, kw):
    prob, data = make()
    comp = prob.compile(max_blocks=MAXB, tol=TOL, **kw)
    comp.solve_instance(data)                              # warm
    sol = comp.solve_instance(data)
    return {"iters": sol.iters, "residual": sol.residual}


if __name__ == "__main__":
    rows = []
    hdr = f"{'problem':18s} {'arm':14s} {'iters':>7s} {'residual':>11s}"
    print(hdr); print("-" * len(hdr))
    for pname, make in PROBLEMS:
        for aname, kw in ARMS:
            try:
                r = bench(make, kw)
            except Exception as e:
                r = {"iters": -1, "residual": float("nan"), "error": repr(e)[:80]}
            r["problem"] = pname; r["arm"] = aname; rows.append(r)
            print(f"{pname:18s} {aname:14s} {r['iters']:7d} {r['residual']:11.2e}"
                  + (f"  ERR {r['error']}" if "error" in r else ""))
        print()
    out = os.path.join(os.path.dirname(__file__), "output", "equil_online_ab.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print("-> wrote", out)
