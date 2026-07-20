"""Run NVIDIA cuOpt as two competitors and write CSVs for the hero plots:

  * cuPDLP        = cuOpt method=PDLP        (the GPU first-order LP/QP solver)
  * cuOpt(simplex)= cuOpt method=DualSimplex (the classical exact method)

Both are STORED-MATRIX solvers: they need the constraint / quadratic matrix in
memory, so on the operator problems (FFT) they hit a materialization wall.  A hard
memory guard prevents ever pushing the host/GPU to a crash.

Run with the opt env:
    /home/miria/envs/optenv/bin/python examples/run_cupdlp_cuopt.py
"""
import os, csv, time
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
import cuopt.linear_programming as lp

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
MEM_LIMIT_GB = 8.0                                   # never allocate a matrix past this


def cuopt_lp(A, b, c, lo, hi, method, tol=1e-4):
    dm = lp.DataModel()
    dm.set_csr_constraint_matrix(A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32))
    dm.set_constraint_lower_bounds(np.full(A.shape[0], -np.inf)); dm.set_constraint_upper_bounds(b)
    dm.set_variable_lower_bounds(lo); dm.set_variable_upper_bounds(hi)
    dm.set_objective_coefficients(c); dm.set_maximize(False)
    ss = lp.SolverSettings(); ss.set_optimality_tolerance(tol)
    ss.set_parameter("method", int(method))
    t = time.perf_counter(); sol = lp.Solve(dm, ss); dt = time.perf_counter() - t
    return dt, int(sol.get_termination_status())


# ============ (B) bin packing LP: cuPDLP vs cuOpt(simplex) vs HiGHS ============
I, B, R = 120, 40, 3
Rmat = np.random.default_rng(1).uniform(.1, 1, (R, I))
A = sp.vstack([sp.kron(sp.eye(I), np.ones((1, B))),
               sp.kron(sp.csr_matrix(Rmat), sp.eye(B))]).tocsr()
rng = np.random.default_rng(4); per = 0.6 * Rmat.sum(1) / B
def inst():
    C = (per[:, None] * rng.uniform(.7, 1.3, (R, B))); w = rng.uniform(.5, 1.5, I)
    return np.concatenate([np.ones(I), C.reshape(-1)]), -np.repeat(w, B)

lo, hi = np.zeros(I * B), np.ones(I * B)
tp = {"cuPDLP": [], "cuOpt(simplex)": [], "HiGHS": []}
for _ in range(10):
    b, c = inst()
    tp["cuPDLP"].append(cuopt_lp(A, b, c, lo, hi, lp.SolverMethod.PDLP)[0])
    tp["cuOpt(simplex)"].append(cuopt_lp(A, b, c, lo, hi, lp.SolverMethod.DualSimplex)[0])
    t = time.perf_counter(); linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * (I * B), method="highs"); tp["HiGHS"].append(time.perf_counter() - t)
with open(os.path.join(PLOTS, "competitors_binpack.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["solver", "median_solve_s", "throughput_per_s"])
    for name in ["cuPDLP", "cuOpt(simplex)", "HiGHS"]:
        med = float(np.median(tp[name])); w.writerow([name, med, 1.0 / med])
        print(f"binpack {name:16s}: {med*1e3:6.1f} ms  ({1/med:5.1f} inst/s)")

# ============ (C) FFT LASSO as a QP: cuPDLP hits the K^T K materialization wall ============
# min 1/2||Kx-y||^2 + lam||x||_1 (K = subsampled DFT, dense).  Posed for a stored-matrix
# QP solver: split x = u - v >= 0, quadratic P = [[G,-G],[-G,G]] with G = K^T K (DENSE
# N x N -- the FFT operator has no sparse form).  We STOP before P exceeds the guard.
def fft_operator(N, ratio=0.25, seed=0):
    r = np.random.default_rng(seed)
    m = max(1, int(ratio * N)); rows = r.choice(N, m, replace=False)
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(rows, n) / N) / np.sqrt(N)     # m x N subsampled DFT
    return np.vstack([F.real, F.imag]).astype(np.float32)            # (2m) x N real

rows = []
for N in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    P_gb = (2 * N) ** 2 * 4 / 1e9                                    # dense 2N x 2N quadratic
    if P_gb > MEM_LIMIT_GB:
        rows.append([N, None, f"skipped (dense QP {P_gb:.1f} GB > {MEM_LIMIT_GB} GB guard)"]);
        print(f"FFT N={N}: skip (QP {P_gb:.1f} GB)"); continue
    try:
        K = fft_operator(N); y = np.random.default_rng(1).normal(size=K.shape[0]).astype(np.float32)
        G = (K.T @ K).astype(np.float64); Kty = (K.T @ y).astype(np.float64); lam = 1e-3
        n2 = 2 * N
        Pmat = np.block([[G, -G], [-G, G]])                          # 2N x 2N dense
        q = np.concatenate([lam - Kty, lam + Kty])
        dm = lp.DataModel()
        A0 = sp.csr_matrix((0, n2))
        dm.set_csr_constraint_matrix(np.array([], np.float64), np.array([], np.int32), np.zeros(1, np.int32))
        dm.set_variable_lower_bounds(np.zeros(n2)); dm.set_variable_upper_bounds(np.full(n2, np.inf))
        dm.set_objective_coefficients(q)
        Psp = sp.csr_matrix(Pmat)
        dm.set_quadratic_objective_matrix(Psp.data, Psp.indices.astype(np.int32), Psp.indptr.astype(np.int32))
        ss = lp.SolverSettings(); ss.set_optimality_tolerance(1e-3); ss.set_parameter("method", int(lp.SolverMethod.PDLP))
        t = time.perf_counter(); sol = lp.Solve(dm, ss); dt = time.perf_counter() - t
        rows.append([N, dt, "solved"]); print(f"FFT N={N}: cuPDLP QP {dt:.2f}s (dense P {P_gb:.2f} GB)")
    except Exception as e:
        rows.append([N, None, f"OOM/{type(e).__name__}"]); print(f"FFT N={N}: {type(e).__name__} {str(e)[:50]}")
with open(os.path.join(PLOTS, "competitors_fft_cupdlp.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N", "cupdlp_solve_s", "status"]); w.writerows(rows)
print("wrote competitors_binpack.csv + competitors_fft_cupdlp.csv")
print("note: MRI-SOCP is a second-order-cone program -- cuOpt/cuPDLP are LP/QP only and cannot represent it.")
