"""Run competitor solvers (optenv: cuOpt, HiGHS) and write CSVs into ../plots/ for
overlay on the hero plots.  Run with the optenv python:
    /home/miria/envs/optenv/bin/python examples/run_competitors.py
"""
import os, csv, time
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
import cuopt.linear_programming as lp

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)

# ---------- (B) bin packing: same config as the OCS throughput plot ----------
I, B, R = 120, 40, 3
Rmat = np.random.default_rng(1).uniform(.1, 1, (R, I))
A = sp.vstack([sp.kron(sp.eye(I), np.ones((1, B))),
               sp.kron(sp.csr_matrix(Rmat), sp.eye(B))]).tocsr()
rng = np.random.default_rng(4); per = 0.6 * Rmat.sum(1) / B
def inst():
    C = (per[:, None] * rng.uniform(.7, 1.3, (R, B)))
    w = rng.uniform(.5, 1.5, I)
    return np.concatenate([np.ones(I), C.reshape(-1)]), -np.repeat(w, B)

def cuopt_solve(b, c):
    dm = lp.DataModel()
    dm.set_csr_constraint_matrix(A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32))
    dm.set_constraint_lower_bounds(np.full(A.shape[0], -np.inf)); dm.set_constraint_upper_bounds(b)
    dm.set_variable_lower_bounds(np.zeros(I*B)); dm.set_variable_upper_bounds(np.ones(I*B))
    dm.set_objective_coefficients(c); dm.set_maximize(False)
    ss = lp.SolverSettings(); ss.set_optimality_tolerance(1e-4); ss.set_parameter("method", int(lp.SolverMethod.PDLP))
    sol = lp.Solve(dm, ss); return float(sol.get_solve_time())

ch, hh = [], []
for _ in range(10):
    b, c = inst()
    ch.append(cuopt_solve(b, c))
    t = time.perf_counter(); linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)]*(I*B), method="highs"); hh.append(time.perf_counter()-t)
with open(os.path.join(PLOTS, "competitors_binpack.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["solver", "median_solve_s", "throughput_per_s"])
    w.writerow(["cuOpt(PDLP)", np.median(ch), 1/np.median(ch)])
    w.writerow(["HiGHS", np.median(hh), 1/np.median(hh)])
print(f"binpack: cuOpt {np.median(ch)*1e3:.0f}ms ({1/np.median(ch):.0f}/s)  HiGHS {np.median(hh)*1e3:.0f}ms ({1/np.median(hh):.0f}/s)")

# ---------- (C) FFT wedge: materialization wall, WITH A HARD MEMORY GUARD ----------
# A matrix solver must FORM the dense subsampled-DFT operator (2m x N).  We PREDICT
# its size and STOP well before it could exhaust RAM -- never pushing the machine to
# the crash limit.
MEM_LIMIT_GB = 6.0                      # never allocate more than this (host has ~100GB)
rows = []
for N in [2**k for k in range(12, 22)]:
    m = N // 4
    op_gb = (2 * m) * N * 4 / 1e9        # dense real-split operator (2m x N), float32
    peak_gb = m * N * 16 / 1e9 + op_gb   # complex build buffer + operator
    if peak_gb > MEM_LIMIT_GB:
        rows.append([N, round(op_gb, 3), None, f"SKIP (needs ~{peak_gb:.1f}GB > {MEM_LIMIT_GB:.0f}GB guard)"])
        print(f"  N={N:>9,}  dense operator {op_gb:7.2f} GB (peak ~{peak_gb:.1f}GB) > {MEM_LIMIT_GB:.0f}GB guard -> STOP (would OOM)")
        break
    t = time.perf_counter()
    idx = np.sort(np.random.default_rng(0).choice(N, m, replace=False))
    Fr = np.exp(-2j * np.pi * np.outer(idx, np.arange(N)) / N) / np.sqrt(N)   # sampled DFT rows only
    Adense = np.vstack([Fr.real, Fr.imag]).astype(np.float32)                 # (2m x N)
    rows.append([N, round(Adense.nbytes / 1e9, 3), round(time.perf_counter() - t, 2), "built"])
    print(f"  N={N:>9,}  materialize dense operator {Adense.nbytes/1e9:7.2f} GB")
    del Adense, Fr
with open(os.path.join(PLOTS, "competitors_fft.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N", "dense_operator_GB", "build_s", "status"]); w.writerows(rows)
print("-> wrote competitor CSVs to", PLOTS)
