"""OCS hello world -- the two entry points.

  Entry 2 (trust):  sanity-check a SMALL problem against CVXPY (OCS vs SCS).
  Entry 1 (scale):  solve the real thing -- one fixed matrix-free K, many data,
                    batched on the GPU, at a size where a formed matrix is hopeless.

Run:  python examples/hello_world.py
"""
import numpy as np
import jax.numpy as jnp
import ocs

# ======================================================================
# Entry point 2 -- sanity-check a small problem against CVXPY.
# (Build any small cvxpy problem; the harness solves it with OCS and SCS and
#  compares.  This is separate from the core solver -- just for trust.)
# ======================================================================
try:
    import cvxpy as cp
    from ocs.cvxpy_harness import validate_against_cvxpy
    c0 = np.random.default_rng(0).normal(size=12)
    xcp = cp.Variable(12)
    small = cp.Problem(cp.Minimize(c0 @ xcp), [cp.norm(xcp, 2) <= 1])   # small SOCP
    rep = validate_against_cvxpy(small)
    print(f"[entry 2]  small SOCP:  OCS vs SCS objective rel-err = {rep['rel_err']:.1e}")
except ImportError:
    print("[entry 2]  (install cvxpy to run the sanity-check harness)")

# ======================================================================
# Entry point 1 -- solve at scale:  min  lambda||x||_1 + 1/2||M F x - y||_2^2.
# K = M F (subsampled orthonormal FFT) is never formed.  Compile the family ONCE,
# then solve one instance -- and a whole batch with the same K, different y.
# ======================================================================
n = 4096
K = ocs.FFTMask.random(n, ratio=0.25, seed=0)      # K : R^n -> R^{2m}, matrix-free
x = ocs.Variable(n)
y = ocs.Parameter("y")                             # the per-instance data (varies)

problem = ocs.minimize([
    ocs.l1(x, lam=1e-3),                           # g(x)
    ocs.least_squares(K @ x - y),                  # f(Kx + l),  l = -y
])
compiled = problem.compile(block=100, max_blocks=80, tol=1e-3)
print("[entry 1] ", compiled)

# ground-truth sparse signal and its measurements
rng = np.random.default_rng(1)
def measure():
    z = np.zeros(n); z[rng.choice(n, n // 50, replace=False)] = rng.normal(size=n // 50)
    return np.asarray(K.matvec(jnp.asarray(z, jnp.float32)))

# one instance
sol = compiled.solve_instance({"y": measure()})
print(f"[entry 1]  one instance:  iters~{sol.iters}  residual={sol.residual:.1e}  "
      f"nnz(x)={int((np.abs(np.asarray(sol.x)) > 1e-4).sum())}")

# MANY instances, same K, different y -- one batched GPU solve
Y = jnp.asarray(np.stack([measure() for _ in range(16)]), jnp.float32)
xb, yb, iters, res = compiled.solve_many({"y": Y})
print(f"[entry 1]  batch of 16:   median residual={float(np.median(np.asarray(res))):.1e}  "
      f"x batch shape={tuple(xb.shape)}")
