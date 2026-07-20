"""Killer example: Sylvester least squares -- the pure-LS / CG regime.

    minimize_X  1/2 || A X + X B - C ||_F^2,     X in R^{p x q}.

The Sylvester operator vec(X) -> vec(A X + X B) is applied directly (never forming
the (pq) x (pq) matrix I(x)A + B^T(x)I).  This is a *pure* least-squares problem --
one quadratic term, no other structure -- which is exactly the conjugate-gradient
regime Boyd flagged: the restart-PDHG engine converges but needs a huge iteration
count, while matrix-free CGLS reaches the same accuracy in O(sqrt(kappa)) matvecs.

OCS exposes both: the default composite engine, and `compile(method="cg")` for the
pure-LS case.  Validated against scipy.linalg.solve_sylvester (a direct method).

Run:  python examples/killer_sylvester.py
"""
import numpy as np, jax, jax.numpy as jnp, ocs
from scipy.linalg import solve_sylvester

p, q = 60, 50
rng = np.random.default_rng(0)
A = rng.normal(size=(p, p)).astype(np.float32) * 0.5 + 5 * np.eye(p, dtype=np.float32)
B = rng.normal(size=(q, q)).astype(np.float32) * 0.5 + 5 * np.eye(q, dtype=np.float32)
Xt = rng.normal(size=(p, q)).astype(np.float32)
C = (A @ Xt + Xt @ B).astype(np.float32)
Xr = solve_sylvester(A.astype(np.float64), B.astype(np.float64), C.astype(np.float64))

S = ocs.Sylvester(A, B); x = ocs.Variable(p * q); c = ocs.Parameter("c")
prob = ocs.minimize([ocs.least_squares(S @ x - c)])
data = {"c": jnp.asarray(C.reshape(-1))}
rel = lambda X: np.linalg.norm(X - Xr) / np.linalg.norm(Xr)
print(f"Sylvester LS,  X is {p}x{q} = {p*q} vars")

# restart-PDHG (the composite engine) -- converges but slowly on a pure-LS problem
comp = prob.compile(max_blocks=2000, tol=1e-8, adaptive_omega=False)
sol = comp.solve_instance(data); jax.block_until_ready(sol.x)
Xp = np.asarray(sol.x).reshape(p, q)
print(f"  PDHG : {sol.iters:>7d} iters   residual={sol.residual:.1e}   ||X-X*||/||X*|| = {rel(Xp):.2e}")

# matrix-free CGLS -- the right tool for the pure-LS regime
compc = prob.compile(method="cg", cg_iters=500, tol=1e-12)
solc = compc.solve_instance(data); jax.block_until_ready(solc.x)
Xc = np.asarray(solc.x).reshape(p, q)
print(f"  CGLS : {solc.iters:>7d} matvecs residual={solc.residual:.1e}   ||X-X*||/||X*|| = {rel(Xc):.2e}")
print(f"  -> CGLS reaches the same accuracy in ~{sol.iters // max(int(solc.iters), 1)}x fewer iterations")
