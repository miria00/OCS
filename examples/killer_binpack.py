"""Killer example: fractional bin packing (LP), operator-native + batched.

    maximize  sum_ib w_i X_ib
    s.t.      sum_b X_ib <= 1          (each item placed at most once)
              sum_i R_ri X_ib <= C_rb  (per-resource, per-bin capacity)
              X >= 0,                    X in R^{I x B}.

The constraint operator (assignment + capacity) is applied by BinPack and never
formed.  Same resource structure R (fixed K), many instances with different
capacities C and weights w -> compile once, solve_many on GPU.  This is the family
benchmarked against NVIDIA cuOpt.

Run:  python examples/killer_binpack.py
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog
import ocs

I, B, R = 120, 40, 3
Rmat = np.random.default_rng(1).uniform(0.1, 1.0, (R, I)).astype(np.float32)
K = ocs.BinPack(Rmat, B)
negK = ocs.Weighted(K, -1.0)                      # for  K x <= b  <=>  -Kx + b >= 0
x = ocs.Variable(I * B)
rhs = ocs.Parameter("rhs")                        # [1_I ; vec(C)]
cost = ocs.Parameter("c")                         # -w per assignment

se = jnp.full(I * B, np.sqrt(0.1), jnp.float32)   # small quadratic reg -> fast PDHG
prob = ocs.minimize([ocs.nonneg(x),                          # x >= 0  (g)
                     ocs.nonneg(negK @ x + rhs),              # K x <= b
                     ocs.sum_squares(ocs.Diagonal(se) @ x)], cost=cost)
compiled = prob.compile(max_blocks=2000, tol=1e-5)
print(compiled)

rng = np.random.default_rng(4)
def instance():
    per_bin = 0.6 * Rmat.sum(1) / B
    C = (per_bin[:, None] * rng.uniform(0.7, 1.3, (R, B))).astype(np.float32)
    w = rng.uniform(0.5, 1.5, I).astype(np.float32)
    rhs = np.concatenate([np.ones(I, np.float32), C.reshape(-1)])
    c = -np.repeat(w, B).astype(np.float32)
    return rhs, c, w

rhs0, c0, w0 = instance()
sol = compiled.solve_instance({"rhs": jnp.asarray(rhs0), "c": jnp.asarray(c0)})
X = np.asarray(sol.x).reshape(I, B)
packed = float((w0[:, None] * X).sum())
# scipy reference (same LP)
A_ub = np.asarray(np.vstack([__import__("numpy").kron(np.eye(I), np.ones((1, B))),
                             np.kron(Rmat, np.eye(B))]))     # (I+RB) x IB
res = linprog(c0, A_ub=A_ub, b_ub=rhs0, bounds=[(0, None)] * (I * B), method="highs")
print(f"one instance: packed={packed:.3f}  scipy={-res.fun:.3f}  rel_err={abs(packed+res.fun)/(1+abs(res.fun)):.1e}  "
      f"residual={sol.residual:.1e}")

B_N = 64
insts = [instance() for _ in range(B_N)]
batch = {"rhs": jnp.asarray(np.stack([i[0] for i in insts]), jnp.float32),
         "c": jnp.asarray(np.stack([i[1] for i in insts]), jnp.float32)}
_ = compiled.solve_many(batch); jax.block_until_ready(_[0])
t = time.perf_counter(); out = compiled.solve_many(batch); jax.block_until_ready(out[0]); dt = time.perf_counter() - t
print(f"batch of {B_N} ({I*B:,} vars each): {dt:.2f}s = {B_N/dt:.0f} instances/s  median res {float(np.median(np.asarray(out[3]))):.1e}")
