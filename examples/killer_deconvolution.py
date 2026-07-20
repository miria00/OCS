"""Killer example: sparse deconvolution with a matrix-free fast Gauss transform.

    minimize_x  1/2 || G x - b ||_2^2  +  lambda ||x||_1,

where G is a Gaussian blur (the fast Gauss transform), applied via FFT and never
formed.  Recover a sparse spike train from its blurred, noisy measurement -- the
classic deconvolution problem, operator-native.  Compile once, solve many blurred
signals with the same G.

Run:  python examples/killer_deconvolution.py
"""
import numpy as np, jax.numpy as jnp, ocs

n, NS = 4096, 30
G = ocs.GaussianBlur(n, sigma=3.0)                 # fast Gauss transform, matrix-free
x = ocs.Variable(n); b = ocs.Parameter("b")

prob = ocs.minimize([ocs.l1(x, lam=1e-3), ocs.least_squares(G @ x - b)])
comp = prob.compile(max_blocks=400, tol=1e-4)
print(comp)

# ground-truth spike train -> blur -> add noise
rng = np.random.default_rng(0)
x_true = np.zeros(n, np.float32)
idx = rng.choice(n, NS, replace=False); x_true[idx] = rng.uniform(1, 3, NS) * rng.choice([-1, 1], NS)
b_clean = np.asarray(G.matvec(jnp.asarray(x_true)))
b_noisy = (b_clean + 0.005 * rng.normal(size=n)).astype(np.float32)

sol = comp.solve_instance({"b": jnp.asarray(b_noisy)})
xr = np.asarray(sol.x)
# support recovery: how many of the true spikes are among the recovered peaks
rec = set(np.argsort(-np.abs(xr))[:NS].tolist()); true = set(idx.tolist())
print(f"residual={sol.residual:.1e}  spikes recovered {len(rec & true)}/{NS}  "
      f"reconstruction rel-err={np.linalg.norm(xr - x_true)/np.linalg.norm(x_true):.2e}")

# many blurred signals, same G, one batched GPU solve
def blurred_spikes():
    z = np.zeros(n, np.float32); j = rng.choice(n, NS, replace=False)
    z[j] = rng.uniform(1, 3, NS) * rng.choice([-1, 1], NS)
    return np.asarray(G.matvec(jnp.asarray(z))) + 0.005 * rng.normal(size=n)
B = jnp.asarray(np.stack([blurred_spikes() for _ in range(16)]), jnp.float32)
xb, yb, it, res = comp.solve_many({"b": B})
print(f"batch of 16 deconvolutions: median residual {float(np.median(np.asarray(res))):.1e}  shape {tuple(xb.shape)}")
