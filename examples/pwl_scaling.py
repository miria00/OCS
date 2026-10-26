"""Is the PWL residual floor a float32 resolution effect?

The optimum of  min sum_i phi(x_i - b_i) + (mu/2)||x||^2  sits near
x_i = -s_active/mu, so shrinking mu inflates ||x||.  If the floor is float32
resolution, the residual should track eps_f32 * ||x||_inf and shrink as mu grows.
"""
import numpy as np, jax, jax.numpy as jnp, ocs

SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
_I = [0.0]
for _j, _b in enumerate([0.5, 1.0, 1.5]):
    _I.append(_I[-1] + (SLOPES[_j] - SLOPES[_j + 1]) * _b)
INTER = np.array(_I, np.float32)
EPS32, n = np.finfo(np.float32).eps, 200
rng = np.random.default_rng(0)
b = rng.normal(size=n).astype("f")

print(f"float32 eps = {EPS32:.3e}\n")
print(f"{'mu':>8s} {'|x|_inf':>10s} {'OCS resid':>11s} {'eps*|x|':>10s} "
      f"{'ratio':>7s} {'status':>11s}")
for mu in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    comp = ocs.minimize([
        ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, "f")) @ x),
        ocs.pwl(ocs.Identity(n) @ x - bp, SLOPES, INTER),
    ]).compile(block=100, max_blocks=400, tol=1e-9, equilibrate="sgd")
    s = comp.solve_instance({"b": jnp.asarray(b)}); jax.block_until_ready(s.x)
    xs = np.asarray(s.x, dtype=np.float64)
    xmax = np.abs(xs).max()
    pred = EPS32 * xmax
    print(f"{mu:8.0e} {xmax:10.2f} {s.residual:11.3e} {pred:10.3e} "
          f"{s.residual/max(pred,1e-30):7.2f} {s.status:>11s}")
