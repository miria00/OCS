"""The matrix scaling tier, consumed by a preconditioned PDHG solve.

Problem (ill-conditioned quadratic + a coupled ball constraint):

    minimize   1/2 x^T P x - c^T x
    subject to ||x||_2 <= r,

with P symmetric PD, condition number ~1e3.  g(x)=1/2 x^T P x - c^T x is a
`Quadratic` block (the MATRIX tier); the ball is a scalar-tier indicator.

Chambolle-Pock with a *matrix* primal metric M uses the block's matrix prox
    prox_g^M(v) = (P + M)^{-1}(M v + c),
so choosing M ~ P absorbs the curvature of P and the iteration count becomes
insensitive to conditioning.  With the usual *scalar* metric (1/tau) I the same
solve crawls.  Both use identical dual updates; only the primal metric differs.

Run:  python examples/matrix_metric_demo.py
"""
import numpy as np, jax.numpy as jnp, ocs

n = 40
rng = np.random.default_rng(0)
U, _ = np.linalg.qr(rng.normal(size=(n, n)))
eig = np.logspace(0, 3, n).astype("f")                 # condition number 1e3
P = (U * eig) @ U.T
c = rng.normal(size=n).astype("f")
r = 1.0
g = ocs.Quadratic(P.astype("f"), (-c).astype("f"))     # 1/2 x^T P x - c^T x

# reference optimum
try:
    import cvxpy as cp
    xv = cp.Variable(n)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, P) - c @ xv), [cp.norm(xv, 2) <= r]).solve(solver=cp.CLARABEL)
    xstar = np.asarray(xv.value, np.float32)
except Exception:
    xstar = np.linalg.solve(P, c).astype("f"); xstar *= min(1.0, r / (np.linalg.norm(xstar) + 1e-30))
rel = lambda x: np.linalg.norm(np.asarray(x) - xstar) / (np.linalg.norm(xstar) + 1e-30)


def shrink(w, lam):                                    # prox of the ball's support r||.||
    nrm = jnp.linalg.norm(w)
    return jnp.maximum(0.0, 1.0 - lam / (nrm + 1e-30)) * w


def solve(metric, maxit=20000, tol=1e-4):
    Pj = jnp.asarray(P); x = jnp.zeros(n); y = jnp.zeros(n); xbar = x
    if metric == "matrix":
        beta = float(eig.min())
        M = Pj + beta * jnp.eye(n)
        sigma = 1.0 / (eig.min() + beta)               # M >= (1/sigma) I  (CP stability)
        Minv = jnp.linalg.inv(M)
    else:
        L = float(eig.max())                           # scalar steps, tau*sigma < 1
        tau = 1.0 / np.sqrt(L); sigma = 1.0 / np.sqrt(L)
    for k in range(1, maxit + 1):
        y = shrink(y + sigma * xbar, sigma * r)
        if metric == "matrix":
            x_new = g.prox_matrix(x - Minv @ y, M)      # (P+M)^{-1}(M x - y + c)
        else:
            x_new = g.prox(x - tau * y, tau)            # (P + (1/tau)I)^{-1}(...)
        xbar = 2.0 * x_new - x; x = x_new
        if rel(x) < tol:
            return k, rel(x)
    return maxit, rel(x)


print(f"ill-conditioned QP, n={n}, cond(P)~{eig.max()/eig.min():.0f}, ball ||x||<= {r}")
for metric in ["scalar", "matrix"]:
    k, e = solve(metric)
    tag = "(1/tau) I" if metric == "scalar" else "M ~ P  (matrix tier)"
    print(f"  {metric:6s} metric  {tag:22s}: {k:6d} iters to rel-err {e:.1e}")
