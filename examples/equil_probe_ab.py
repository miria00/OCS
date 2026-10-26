"""Step 2 A/B: can we halve SMFE setup cost without losing quality?

Compares SMFE variants on the ill-conditioned operators where equilibration
actually matters (dense-nnls column-imbalanced, fft-nnls row-imbalanced):

  sgd 2x80      current baseline           (setup units = probes*iters = 160)
  sgd 1x80      one probe, half the cost    (80)
  sgd 1x160     one probe, equal cost, more averaging  (160)
  bradley 1x80  Bradley SNBIN running-average, half cost (80)
  bradley 1x160 Bradley SNBIN, equal cost   (160)

"Bradley's philosophy: one probe, let the iteration do the averaging."  We want
a variant that matches the baseline's iters-to-1e-3 and nvar reduction at <= its
setup cost.

Run:  PYTHONPATH=/home/miria/OCSolver/OCS /home/miria/jaxenv2/bin/python examples/equil_probe_ab.py
"""
import os, json
import numpy as np
import ocs
from equil_harness import make_dense_nnls, make_fft_nnls

TOL, MAXB = 1e-3, 400

VARIANTS = [
    ("sgd 2x80",     dict(smfe_schedule="sgd",     smfe_probes=2, smfe_iters=80)),
    ("sgd 1x80",     dict(smfe_schedule="sgd",     smfe_probes=1, smfe_iters=80)),
    ("sgd 1x160",    dict(smfe_schedule="sgd",     smfe_probes=1, smfe_iters=160)),
    ("bradley 1x80", dict(smfe_schedule="bradley", smfe_probes=1, smfe_iters=80)),
    ("bradley 1x160",dict(smfe_schedule="bradley", smfe_probes=1, smfe_iters=160)),
]

PROBLEMS = [
    ("dense-nnls R=30",  make_dense_nnls(30.0)),
    ("dense-nnls R=100", make_dense_nnls(100.0)),
    ("fft-nnls R=30",    make_fft_nnls(30.0)),
    ("fft-nnls R=100",   make_fft_nnls(100.0)),
]


def bench(make, kw):
    prob, data = make()
    comp = prob.compile(max_blocks=MAXB, tol=TOL, equilibrate="sgd", **kw)
    comp.solve_instance(data)                             # warm
    sol = comp.solve_instance(data)
    d = comp.meta.get("equil_diag", {}) or {}
    units = kw["smfe_probes"] * kw["smfe_iters"]
    return dict(iters=sol.iters, residual=sol.residual, units=units,
                nvar_row_post=d.get("nvar_row_post"), nvar_col_post=d.get("nvar_col_post"))


if __name__ == "__main__":
    rows = []
    hdr = f"{'problem':18s} {'variant':14s} {'units':>6s} {'iters':>7s} {'residual':>11s} {'nvar_post(r,c)':>16s}"
    print(hdr); print("-" * len(hdr))
    for pname, make in PROBLEMS:
        for vname, kw in VARIANTS:
            r = bench(make, kw); r["problem"] = pname; r["variant"] = vname; rows.append(r)
            nv = f"{r['nvar_row_post']:.3f},{r['nvar_col_post']:.3f}"
            print(f"{pname:18s} {vname:14s} {r['units']:6d} {r['iters']:7d} "
                  f"{r['residual']:11.2e} {nv:>16s}")
        print()
    out = os.path.join(os.path.dirname(__file__), "output", "equil_probe_ab.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print("-> wrote", out)
