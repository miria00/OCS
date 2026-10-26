"""Killer example: matrix-variable feature selection (multi-task group lasso).

    minimize_Theta  1/2 ||A Theta - B||_F^2  +  lambda sum_f ||Theta_{f,:}||_2

Theta (F x T) is a MATRIX variable: F features x T tasks.  The row-group-l2 term
zeroes whole rows (features), selecting a common feature set across tasks.  The
operator applies A to the F x T block directly (LeftMatMul) -- the matrix variable
is never flattened into an (mF)-vector, which is exactly where dense/stored-matrix
solvers blow up.

Run:  python examples/killer_feature_selection.py
"""
import numpy as np
import jax.numpy as jnp
import ocs

# Paper scale: F = T = 10^4 -> 10^8 variables in 10^4 row groups.
# Override with OCS_FS_SMALL=1 for the quick smoke size.
import os as _os
if _os.environ.get("OCS_FS_SMALL"):
    m, F, T = 120, 200, 8
else:
    m, F, T = 600, 10000, 10000  # samples, features, tasks
K_ACTIVE, LAM = 12, 0.5
rng = np.random.default_rng(0)
A = rng.normal(size=(m, F)).astype(np.float32)
active = rng.choice(F, K_ACTIVE, replace=False)
Theta_true = np.zeros((F, T), np.float32)
Theta_true[active] = rng.normal(size=(K_ACTIVE, T))
B = (A @ Theta_true + 0.01 * rng.normal(size=(m, T))).astype(np.float32)

K = ocs.LeftMatMul(A, T)         # vec(Theta) -> vec(A Theta), matrix variable
th = ocs.Variable(F * T)
b = ocs.Parameter("b")
# min lam * sum_f ||Theta_{f,:}||_2  +  1/2||A Theta - B||^2
prob = ocs.minimize([ocs.group_l2(th, lam=LAM, group_size=T),      # row-group lasso (g)
                     ocs.least_squares(K @ th - b)])                # data fidelity
compiled = prob.compile(max_blocks=int(_os.environ.get('OCS_FS_BLOCKS', 120)), tol=1e-3)
print(compiled, f"  (variable is {F}x{T} matrix = {F*T} vars)")

sol = compiled.solve_instance({"b": jnp.asarray(B.reshape(-1))})
Th = np.asarray(sol.x).reshape(F, T)
row_norm = np.linalg.norm(Th, axis=1)
sel = set(np.argsort(row_norm)[-K_ACTIVE:])
print(f"residual={sol.residual:.1e}  iters~{sol.iters}")
print(f"selected {K_ACTIVE} strongest rows; true-positives = "
      f"{len(sel & set(active.tolist()))}/{K_ACTIVE}  "
      f"(median inactive-row norm {np.median(row_norm[[i for i in range(F) if i not in active]]):.2e})")
