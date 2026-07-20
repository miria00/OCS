"""Killer example: MRI-style reconstruction -- a real SOCP, operator-native.

    minimize_x  1/2 || M F x - y ||_2^2  +  lambda * sum_pixel ||(grad x)_pixel||_2

M F is a subsampled FFT (never formed); the isotropic total-variation term is a
sum of per-pixel 2-norms of the image gradient -- i.e. H*W little second-order
cones ("millions of little cones"), all evaluated as ONE grouped GPU kernel.  We
reconstruct a piecewise-constant phantom from undersampled Fourier data and
compare to the zero-fill baseline.

Run:  python examples/killer_mri.py
"""
import numpy as np
import jax.numpy as jnp

import ocs

H = W = 64
n = H * W
LAM = 0.03

# ---- phantom: a few constant rectangles ----
ph = np.zeros((H, W), np.float32)
ph[10:30, 12:40] = 1.0
ph[35:55, 30:58] = 0.6
ph[20:45, 8:22] += 0.4
phantom = ph.reshape(-1)

# ---- operators: subsampled FFT (35% of frequencies) + 2-D gradient ----
F = ocs.FFTMask.random(n, ratio=0.35, seed=0)
Gr = ocs.Gradient2D(H, W)
y_meas = np.asarray(F.matvec(jnp.asarray(phantom)))

x = ocs.Variable(n)
y = ocs.Parameter("y")
prob = ocs.minimize([
    ocs.least_squares(F @ x - y),                      # data fidelity
    ocs.group_l2(Gr @ x, lam=LAM, group_size=2),       # isotropic TV: H*W SOC cones
])
compiled = prob.compile(max_blocks=1500, tol=1e-4)
print(compiled, f"  (TV cones = {n:,})")

sol = compiled.solve_instance({"y": jnp.asarray(y_meas)})
rec = np.asarray(sol.x)
zerofill = np.asarray(F.rmatvec(jnp.asarray(y_meas)))   # naive adjoint reconstruction

def relerr(z):
    return np.linalg.norm(z - phantom) / np.linalg.norm(phantom)

print(f"residual={sol.residual:.1e}  iters~{sol.iters}")
print(f"reconstruction rel-error:  TV-SOCP {relerr(rec):.3f}   zero-fill {relerr(zerofill):.3f}   "
      f"(TV is {relerr(zerofill)/relerr(rec):.1f}x better)")
