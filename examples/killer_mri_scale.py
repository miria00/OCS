"""MRI-SOCP at scale: "millions of tiny cones", and the wall matrix solvers hit.

We solve the isotropic-TV MRI SOCP  min 1/2||M F x - y||^2 + lambda*TV(x)  on
increasingly large images L x L.  The TV term is L^2 per-pixel second-order cones
("millions of tiny cones"), all one grouped GPU kernel; M F is a subsampled FFT,
applied in O(N log N) and never formed.  A matrix-forming solver would need the
dense measurement operator (~N^2 entries) -- it exceeds a 24 GB GPU at L~280 while
OCS keeps going.  We report solve time (fixed iteration budget) and cone count.

Run:  python examples/killer_mri_scale.py
"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ocs

Ls = [64, 128, 256, 512, 1024]
GPU_BYTES = 22e9
budget = 40                                   # fixed 4000-iteration budget
cones, times = [], []
for L in Ls:
    n = L * L
    F = ocs.FFTMask.random(n, ratio=0.35, seed=0)
    Gr = ocs.Gradient2D(L, L)
    x = ocs.Variable(n); y = ocs.Parameter("y")
    prob = ocs.minimize([ocs.least_squares(F @ x - y),
                         ocs.group_l2(Gr @ x, lam=0.03, group_size=2)])
    comp = prob.compile(max_blocks=budget, tol=1e-12, equilibrate="scalar")
    ym = jnp.asarray(np.random.default_rng(0).normal(size=F.shape[0]).astype(np.float32))
    out = comp.solve_many({"y": ym[None]}); jax.block_until_ready(out[0])   # warmup
    t = time.perf_counter()
    out = comp.solve_many({"y": ym[None]}); jax.block_until_ready(out[0])
    dt = time.perf_counter() - t
    cones.append(n); times.append(dt)
    print(f"  L={L:5d}  image {n:>9,} px  =  {n:>9,} TV cones   solve {dt:6.3f}s  "
          f"(dense operator would be {n*n*4/1e9:>12,.0f} GB)")

N_wall = np.sqrt(GPU_BYTES / 4)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.loglog(cones, times, "o-", lw=2.6, ms=9, color="#1f77b4", label="OCS  (matrix-free FFT + grouped TV)")
ax.axvline(N_wall, color="#76b900", lw=2.5)
ax.axvspan(N_wall, cones[-1] * 2, color="#76b900", alpha=0.10)
ax.text(N_wall * 1.15, min(times) * 1.5,
        "matrix-forming solvers\nCANNOT ENTER\n(dense operator > 24 GB)", color="#3f6000", fontsize=10)
ax.set_xlabel("image size / TV cones  (pixels)"); ax.set_ylabel(f"solve time, {budget*100} iters (s)")
ax.set_title("MRI-SOCP at scale: millions of TV second-order cones, matrix-free\n"
             "OCS scales O(N log N); a formed measurement operator OOMs at ~N=77k")
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
out = os.path.join(os.path.dirname(__file__), "output", "mri_scale_wedge.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("-> wrote", out, f"| largest: {cones[-1]:,} cones solved matrix-free")
