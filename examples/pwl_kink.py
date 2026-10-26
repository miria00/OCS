"""Is the returned point actually optimal, and is the residual a kink artifact?

For  min_x sum_i phi(x_i - b_i) + (mu/2)||x||^2,  phi(y) = max_j (s_j y + r_j),
optimality at coordinate i is exactly

    -mu x_i  in  d phi(x_i - b_i),

where the subdifferential is {s_j} away from a breakpoint and the interval
[s_j, s_{j+1}] at one.  We check that membership directly, so the verdict does
not depend on the solver's own residual.
"""
import numpy as np, jax, jax.numpy as jnp, ocs

SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
BREAKS = np.array([0.5, 1.0, 1.5], np.float64)
_I = [0.0]
for _j, _b in enumerate(BREAKS):
    _I.append(_I[-1] + (SLOPES[_j] - SLOPES[_j + 1]) * _b)
INTER = np.array(_I, np.float32)
MU, n = 1e-3, 200

rng = np.random.default_rng(0)
b = rng.normal(size=n).astype("f")
x = ocs.Variable(n); bp = ocs.Parameter("b")
comp = ocs.minimize([
    ocs.sum_squares(ocs.Diagonal(np.sqrt(MU) * np.ones(n, "f")) @ x),
    ocs.pwl(ocs.Identity(n) @ x - bp, SLOPES, INTER),
]).compile(block=100, max_blocks=400, tol=1e-8, equilibrate="sgd")
s = comp.solve_instance({"b": jnp.asarray(b)}); jax.block_until_ready(s.x)
xs = np.asarray(s.x, dtype=np.float64)
y = xs - b.astype(np.float64)

S = SLOPES.astype(np.float64)
vals = S[None, :] * y[:, None] + INTER.astype(np.float64)[None, :]
top = vals.max(axis=1, keepdims=True)
active = vals >= top - 1e-6 * np.maximum(1.0, np.abs(top))     # pieces attaining the max
lo = np.where(active, S[None, :], np.inf).min(axis=1)
hi = np.where(active, S[None, :], -np.inf).max(axis=1)
at_kink = active.sum(axis=1) > 1

g = -MU * xs                                                    # required subgradient
slack = np.maximum(lo - g, g - hi)                              # <=0 means g in [lo,hi]
viol = np.maximum(slack, 0.0)

print(f"OCS reports residual {s.residual:.3e}, status '{s.status}', {s.iters} iterations")
print(f"coordinates sitting at a kink: {int(at_kink.sum())} of {n}")
print(f"exact subdifferential violation  max {viol.max():.3e}   mean {viol.mean():.3e}")
print(f"  among kink coordinates         max {viol[at_kink].max() if at_kink.any() else 0:.3e}")
print(f"  among smooth coordinates       max {viol[~at_kink].max() if (~at_kink).any() else 0:.3e}")
opt = viol.max() < 1e-5
print(f"\nverdict: the returned point IS {'optimal' if opt else 'NOT optimal'} "
      f"to 1e-5 by the exact subdifferential test")
if opt:
    print("so the nonzero residual is a property of the certificate, not of the iterate")
