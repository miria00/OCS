"""Independently re-audit archived SOCP endpoints against an SCS reference.

No OCS solve or refinement occurs. Original case JSON and endpoints are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cvxpy as cp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cases", nargs="+", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
args = p.parse_args()
assert platform.node() == "Mazeppa" and not args.output.exists()
rows = []
for path in args.cases:
    raw = json.loads(path.read_text())
    assert raw["config"]["family"] == "socp"
    inputs = path.with_suffix(".inputs.npz")
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == raw["inputs_sha256"]
    with np.load(inputs) as z:
        A, c, ridge = [z[k].astype(np.float64) for k in ("A", "c", "ridge")]
    x = cp.Variable(len(c))
    problem = cp.Problem(cp.Minimize(c @ x + .5 * cp.sum_squares(cp.multiply(ridge, x))),
                         [cp.norm(A @ x) <= 1])
    start = time.perf_counter()
    problem.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
    elapsed = time.perf_counter() - start
    rx = np.asarray(x.value, np.float64)
    objective = lambda xx: float(c @ xx + .5 * np.sum((ridge * xx)**2))
    feasibility = lambda xx: max(0., float(np.linalg.norm(A @ xx) - 1))
    ref_obj, ref_feas = objective(rx), feasibility(rx)
    reference_ok = problem.status == "optimal" and np.isfinite(ref_obj) and ref_feas <= 1e-7
    record = {"case": str(path), "case_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "input_sha256": raw["inputs_sha256"], "reference_solver": "SCS",
              "reference_status": problem.status, "reference_objective": ref_obj,
              "reference_feasibility": ref_feas, "reference_seconds": elapsed,
              "reference_accepted": bool(reference_ok), "reference_x": rx.tolist(),
              "prior_reference": raw["reference"], "points": []}
    for old in raw["points"]:
        endpoint = path.with_name(path.stem + "." + old["point"] + ".npz")
        assert hashlib.sha256(endpoint.read_bytes()).hexdigest() == old["endpoint_sha256"]
        with np.load(endpoint) as z:
            xx = z["x"]
        value, feas = objective(xx), feasibility(xx)
        error = abs(value - ref_obj) / abs(ref_obj)
        accepted = (reference_ok and old["finite"] and old["solver_status"] == "solved"
                    and old["recomputed_residual"] <= 1e-3 and error <= 1e-3 and feas <= 1e-3)
        record["points"].append({"point": old["point"], "endpoint_sha256": old["endpoint_sha256"],
                                 "prior_accepted": old["accepted"], "objective": value,
                                 "objective_error": error, "feasibility": feas,
                                 "recomputed_residual_on_same_pair": old["recomputed_residual"],
                                 "accepted_against_independent_scs_reference": bool(accepted)})
    rows.append(record)
out = {"host": platform.node(), "cvxpy": cp.__version__,
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "scope": "Independent SCS reference and immutable returned-point re-audit; no OCS rerun/refinement",
       "rows": rows}
args.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
