"""Killer example: optimal transport (Kantorovich LP), operator-native.

    minimize_X  <C, X> + (eps/2)||X||_F^2   s.t.  X 1 = p, X^T 1 = q, X >= 0.

The variable is a MATRIX (N^2 entries); the constraint operator K : vec(X) ->
(row sums, col sums) is applied by MarginalOperator and never formed (the explicit
constraint matrix would be (2N) x N^2).  A small quadratic regularization eps makes
the transport LP strongly convex so first-order PDHG converges fast (pure-LP OT is
slow for every first-order method -- this is the standard quadratically-regularized
OT).  We compile the family once and solve many transport instances (different cost
C and marginals p, q) on the GPU, validating the transport cost against POT.

Run:  python examples/killer_ot.py
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

import ocs

N = 64
EPS = 0.1
K = ocs.MarginalOperator(N, N)                 # vec(X) -> (row sums, col sums)
x = ocs.Variable(N * N)
c = ocs.Parameter("c")                         # vec(C)
mrg = ocs.Parameter("m")                       # [p; q]
se = jnp.full(N * N, np.sqrt(EPS), jnp.float32)

# min c^T x + (eps/2)||x||^2  s.t.  K x = [p; q],  x >= 0
prob = ocs.minimize([ocs.nonneg(x), ocs.equality(K @ x - mrg),
                     ocs.sum_squares(ocs.Diagonal(se) @ x)], cost=c)
compiled = prob.compile(max_blocks=800, tol=1e-5)
print(compiled)

rng = np.random.default_rng(0)
def instance():
    C = rng.uniform(0, 1, (N, N)).astype(np.float32)
    p = rng.uniform(0.2, 1, N).astype(np.float32); p /= p.sum()
    q = rng.uniform(0.2, 1, N).astype(np.float32); q /= q.sum()
    return C, p, q

# ---- one instance + validation against POT ----
C, p, q = instance()
data = {"c": jnp.asarray(C.reshape(-1)), "m": jnp.asarray(np.concatenate([p, q]))}
sol = compiled.solve_instance(data)
X = np.asarray(sol.x).reshape(N, N)
ocs_obj = float((C * X).sum())
print(f"one instance: obj={ocs_obj:.5f}  residual={sol.residual:.1e}  "
      f"marginal err: rows {np.abs(X.sum(1)-p).max():.1e}  cols {np.abs(X.sum(0)-q).max():.1e}  "
      f"min(X)={X.min():.1e}")
try:
    import ot
    ref = float(ot.emd2(p.astype(np.float64), q.astype(np.float64), C.astype(np.float64)))
    print(f"   vs POT ot.emd2: OCS={ocs_obj:.5f}  POT={ref:.5f}  rel_err={abs(ocs_obj-ref)/(1+abs(ref)):.2e}")
except Exception as e:
    print(f"   (POT not available: {e})")

# ---- repeated solve: same K (marginal structure), many (C, p, q) ----
B = 64
Cs, ps, qs = zip(*[instance() for _ in range(B)])
batch = {"c": jnp.asarray(np.stack([c.reshape(-1) for c in Cs]), jnp.float32),
         "m": jnp.asarray(np.stack([np.concatenate([p, q]) for p, q in zip(ps, qs)]), jnp.float32)}
_ = compiled.solve_many(batch); jax.block_until_ready(_[0])
t = time.perf_counter()
xb, yb, it, res = compiled.solve_many(batch); jax.block_until_ready(xb)
dt = time.perf_counter() - t
print(f"batch of {B} transport problems (N={N}, {N*N:,} vars each): {dt:.2f}s  "
      f"= {B/dt:.0f} instances/s  median res {float(np.median(np.asarray(res))):.1e}")
