"""Competitor SOLVE-TIME curves for the wedge plots (optenv): SCS on the
materialized SOCP forms of FFT-LASSO and MRI-TV, small sizes (SCS factorizes the
dense operator each iteration -> slow, and OOMs), with a hard memory guard.
Run:  /home/miria/envs/optenv/bin/python examples/competitors_socp.py
"""
import os, csv, time
import numpy as np, cvxpy as cp
PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
MEM_GB = 6.0

def dft(N, m, seed):
    idx = np.sort(np.random.default_rng(seed).choice(N, m, replace=False))
    Fr = np.exp(-2j*np.pi*np.outer(idx, np.arange(N))/N)/np.sqrt(N)
    return np.vstack([Fr.real, Fr.imag]).astype(np.float64)

rows = []
for N in [256, 512, 1024, 2048, 4096]:
    m = N//4; op = (2*m)*N*8/1e9
    if op > MEM_GB: rows.append([N, None, "SKIP"]); print(f"  FFT N={N}: {op:.1f}GB > guard, stop"); break
    A = dft(N, m, 0); y = A @ np.random.default_rng(1).normal(size=N)
    x = cp.Variable(N); t = cp.Variable()
    p = cp.Problem(cp.Minimize(t + 1e-3*cp.norm1(x)), [cp.norm(A@x - y, 2) <= t])
    tt = time.perf_counter(); p.solve(solver=cp.SCS); dt = time.perf_counter()-tt
    rows.append([N, dt, "solved"]); print(f"  FFT N={N:6d}  SCS {dt:8.2f}s", flush=True)
    if dt > 120: break
with open(os.path.join(PLOTS, "competitors_fft_time.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N", "scs_solve_s", "status"]); w.writerows(rows)

rows = []
for L in [16, 24, 32, 48, 64, 96]:
    n = L*L; m = n//3; op = (2*m)*n*8/1e9
    if op > MEM_GB: rows.append([n, None, "SKIP"]); print(f"  MRI L={L}: {op:.1f}GB > guard, stop"); break
    A = dft(n, m, 2); ph = np.zeros((L, L)); ph[L//4:3*L//4, L//4:3*L//4] = 1.0; y = A @ ph.reshape(-1)
    X = cp.Variable((L, L)); t = cp.Variable()
    p = cp.Problem(cp.Minimize(t + 0.05*cp.tv(X)), [cp.norm(A@cp.vec(X, order="C") - y, 2) <= t])
    tt = time.perf_counter(); p.solve(solver=cp.SCS); dt = time.perf_counter()-tt
    rows.append([n, dt, "solved"]); print(f"  MRI L={L:4d} ({n:6d} cones)  SCS {dt:8.2f}s", flush=True)
    if dt > 120: break
with open(os.path.join(PLOTS, "competitors_mri_time.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cones", "scs_solve_s", "status"]); w.writerows(rows)
print("-> wrote SOCP competitor CSVs", flush=True)
