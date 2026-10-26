"""Regime-2 competitor line: sequential throughput on the CvxCluster family.

This is the Stage-1 relaxation of the CvxCluster scheduler, whose structure is a
fractional bin-packing problem. It is not generic bin packing, and the figure
must not call it that.

OCS compiles the family once and solves many instances in a batched call.  A
conventional solver has no such notion: it re-forms and re-solves each instance.
This harness measures that, on exactly the instances make_plots.B builds, so the
crossover in the figure is between measured numbers rather than a batched
against an idealized sequential rate.

The model is the one OCS solves, written out explicitly:

    minimize  c^T x + 0.05 ||x||_2^2
    subject to  K x <= rhs,   x >= 0,

with K the BinPack operator, X of shape (I items x B bins), and
K x = [ X 1_B ; vec(Rmat X) ].

Emits plots/competitors_binpack.csv with schema
    solver,instances,total_s,throughput_per_s,per_solve_s,status,objective_median,mode

Fairness note: OCS compiles the family once and then solves instances.  The
competitor is given the same courtesy through CVXPY parameters, so its
canonicalization is also paid once and the loop is solver time.  Charging it a
fresh canonicalization per instance would flatter us and would not reflect what
a careful user would write.
"""
import os, sys, time, csv
import numpy as np
from guarded import run_guarded

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.environ.get("BINPACK_OUT") or os.path.join(PLOTS, "competitors_binpack.csv")

I, B, R = 120, 40, 3
REG = 0.05                                   # 0.5 * ||sqrt(0.1) x||^2
SOLVERS = os.environ.get("BINPACK_SOLVERS", "CUOPT,HIGHS,SCS").split(",")
# Names that bypass CVXPY and call cuOpt's own Python API on the QP directly:
# the CVXPY interface lifts the quadratic to a second-order cone and then
# forces cuOpt's barrier method for any SOC problem (cuopt_conif.py), so the
# PDLP method can only be exercised through the native quadratic objective.
NATIVE_CUOPT = {"CUOPT_QP_PDLP": "PDLP", "CUOPT_QP_BARRIER": "Barrier"}
NINST = int(os.environ.get("BINPACK_N", 8))
SOLVER_TIMEOUT_S = float(os.environ.get("BINPACK_TIMEOUT", 900))


def binpack_matrix(Rmat):
    """Explicit K, which the competitor cannot avoid forming.  Rows are the I
    per-item assignment rows then the R*B per-resource-per-bin usage rows."""
    n = I * B
    K = np.zeros((I + R * B, n), dtype=np.float64)
    for i in range(I):                                   # X 1_B
        K[i, i * B:(i + 1) * B] = 1.0
    for r in range(R):                                   # vec(Rmat X)
        for b in range(B):
            K[I + r * B + b, b::B] = Rmat[r, :]
    return K


def instances(n):
    """Byte-for-byte the instances make_plots.B_binpack_throughput generates."""
    Rmat = np.random.default_rng(1).uniform(.1, 1, (R, I)).astype("f")
    rng = np.random.default_rng(4)
    per = 0.6 * Rmat.sum(1) / B
    out = []
    for _ in range(n):
        C = (per[:, None] * rng.uniform(.7, 1.3, (R, B))).astype("f")
        w = rng.uniform(.5, 1.5, I).astype("f")
        out.append((np.concatenate([np.ones(I, "f"), C.reshape(-1)]).astype(np.float64),
                    (-np.repeat(w, B)).astype(np.float64)))
    return Rmat, out


def solve_one(name, K, insts):
    """One competitor, compile-once parametric, timed over its solves.
    Runs inside the guard's child process."""
    import cvxpy as cp
    n = K.shape[1]
    x = cp.Variable(n)
    p_rhs = cp.Parameter(K.shape[0]); p_c = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(p_c @ x + REG * cp.sum_squares(x)),
                      [K @ x <= p_rhs, x >= 0])
    mode = "parametric(DPP)" if prob.is_dcp(dpp=True) else "parametric(non-DPP)"
    p_rhs.value, p_c.value = insts[0]
    prob.solve(solver=name)                       # untimed: canonicalize + warm
    objs, status, t0 = [], "optimal", time.perf_counter()
    for rhs, c in insts:
        p_rhs.value, p_c.value = rhs, c
        prob.solve(solver=name)
        objs.append(float(prob.value) if prob.value is not None else float("nan"))
        if prob.status != "optimal":
            status = prob.status
    total = time.perf_counter() - t0
    n_inst = len(insts)
    print(f"  {name:8s} {total:8.3f}s for {n_inst} -> {n_inst/total:8.4f}/s  "
          f"({total/n_inst:.3f}s each)  {status}  {mode}  "
          f"median obj {np.median(objs):.6f}", flush=True)
    return [name, n_inst, round(total, 4), round(n_inst / total, 4),
            round(total / n_inst, 4), status, round(float(np.median(objs)), 6), mode]


def solve_one_cuopt_native(name, K, insts):
    """cuOpt through its own Python API on the QP as written: quadratic
    objective REG*||x||^2 set directly, method chosen by name (PDLP or
    Barrier), default tolerances.  The DataModel is built once and only the
    rhs and cost vectors change per instance, the same compile-once courtesy.
    Runs inside the guard's child process."""
    import scipy.sparse as sp
    import cuopt.linear_programming as lp
    method = getattr(lp.SolverMethod, NATIVE_CUOPT[name])
    Ks = sp.csr_matrix(K)
    m, n = Ks.shape
    dm = lp.DataModel()
    dm.set_csr_constraint_matrix(Ks.data.astype(np.float64), Ks.indices.astype(np.int32),
                                 Ks.indptr.astype(np.int32))
    dm.set_constraint_lower_bounds(np.full(m, -np.inf))
    dm.set_variable_lower_bounds(np.zeros(n)); dm.set_variable_upper_bounds(np.full(n, np.inf))
    # cuOpt symmetrizes Q as Q + Q^T and minimizes c^T x + (1/2) x^T (Q + Q^T) x,
    # so REG on the diagonal gives REG ||x||^2; checked against the CVXPY arms
    # through the median-objective agreement test below.
    dm.set_quadratic_objective_matrix(np.full(n, REG), np.arange(n, dtype=np.int32),
                                      np.arange(n + 1, dtype=np.int32))
    dm.set_maximize(False)
    ss = lp.SolverSettings(); ss.set_parameter("method", int(method))
    ss.set_parameter("log_to_console", False)

    def solve(rhs, c):
        dm.set_constraint_upper_bounds(np.asarray(rhs, np.float64))
        dm.set_objective_coefficients(np.asarray(c, np.float64))
        sol = lp.Solve(dm, ss)
        x = np.asarray(sol.get_primal_solution(), np.float64)
        return sol, float(c @ x + REG * (x @ x))

    solve(*insts[0])                                  # untimed warm-up
    objs, status, t0 = [], "optimal", time.perf_counter()
    for rhs, c in insts:
        sol, obj = solve(rhs, c)
        objs.append(obj)
        term = str(sol.get_termination_reason())
        if "ptimal" not in term:
            status = term
    total = time.perf_counter() - t0
    n_inst = len(insts)
    mode = f"native-QP({NATIVE_CUOPT[name]}, solved_by={sol.get_solved_by()})"
    print(f"  {name:8s} {total:8.3f}s for {n_inst} -> {n_inst/total:8.4f}/s  "
          f"({total/n_inst:.3f}s each)  {status}  {mode}  "
          f"median obj {np.median(objs):.6f}", flush=True)
    return [name, n_inst, round(total, 4), round(n_inst / total, 4),
            round(total / n_inst, 4), status, round(float(np.median(objs)), 6), mode]


def main():
    import cvxpy as cp
    Rmat, insts = instances(NINST)
    K = binpack_matrix(Rmat)
    n = I * B
    print(f"CvxCluster Stage-1: {I} tasks x {B} server shapes = {n:,} variables, "
          f"{K.shape[0]:,} constraint rows, {NINST} instances", flush=True)

    rows = []
    for name in SOLVERS:
        name = name.strip()
        if name in NATIVE_CUOPT:
            try:
                import cuopt.linear_programming  # noqa: F401
            except ImportError:
                print(f"  {name:8s} cuopt not importable, skipped", flush=True); continue
            target = solve_one_cuopt_native
        elif name not in cp.installed_solvers():
            print(f"  {name:8s} not installed, skipped", flush=True); continue
        else:
            target = solve_one
        g = run_guarded(target, args=(name, K, insts), label=name,
                        timeout_s=SOLVER_TIMEOUT_S)
        if g["status"] == "ok":
            rows.append(g["value"])
        else:
            rows.append([name, NINST, "", "", "", g["status"], "", "guarded"])
            print(f"  {name:8s} {g['status']} after {g['seconds']:.1f}s", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "instances", "total_s", "throughput_per_s",
                    "per_solve_s", "status", "objective_median", "mode"])
        w.writerows(rows)
    print(f"\nwrote {OUT}")
    objs = [float(r[6]) for r in rows if r[6] != ""]
    if objs:
        spread = (max(objs) - min(objs)) / max(1.0, abs(np.median(objs)))
        if spread > 1e-4:
            print(f"WARNING: solvers disagree on the objective by {spread:.2e} "
                  f"relative; check the model")
        else:
            print(f"solvers agree on the objective to {spread:.1e} relative")


if __name__ == "__main__":
    main()
