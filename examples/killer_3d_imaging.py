"""Killer example: 3-D imaging / PET-style reconstruction, operator-native.

    minimize_x  1/2||F x - y||^2 + lambda*sum_voxel ||(grad3 x)_voxel||_2   s.t. x>=0

x is a D x H x W volume; F is a subsampled FFT of the flattened volume (measure the
transform, never form it); the 3-D isotropic-TV term is D*H*W little 3-D cones as
one grouped kernel; x>=0 is a nonnegativity (density) constraint.

Run:  python examples/killer_3d_imaging.py
"""
import numpy as np, jax.numpy as jnp, ocs
D = H = W = 40                                   # 64,000-voxel volume
n = D * H * W
vol = np.zeros((D, H, W), np.float32)
vol[8:28, 10:34, 12:30] = 1.0; vol[20:36, 6:20, 18:38] += 0.5     # two blobs
phantom = vol.reshape(-1)
F = ocs.FFTMask.random(n, ratio=0.4, seed=0)
G3 = ocs.Gradient3D(D, H, W)
y = np.asarray(F.matvec(jnp.asarray(phantom)))
x = ocs.Variable(n); yp = ocs.Parameter("y")
prob = ocs.minimize([ocs.nonneg(x),                                  # density >= 0
                     ocs.least_squares(F @ x - yp),
                     ocs.group_l2(G3 @ x, lam=0.02, group_size=3)])   # 3-D TV cones
comp = prob.compile(max_blocks=1500, tol=1e-4)
print(comp, f"  (volume {D}x{H}x{W} = {n:,} voxels, {n:,} 3-D TV cones)")
sol = comp.solve_instance({"y": jnp.asarray(y)})
rec = np.asarray(sol.x); zf = np.maximum(np.asarray(F.rmatvec(jnp.asarray(y))), 0)
re = lambda z: np.linalg.norm(z - phantom) / np.linalg.norm(phantom)
print(f"residual={sol.residual:.1e}  recon rel-error: TV+nonneg {re(rec):.3f}  zero-fill {re(zf):.3f}  "
      f"min(x)={rec.min():.1e}")
