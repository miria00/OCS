"""Repeated-solve benchmark: one FFT operator, many measurement vectors.

This is the OCS product story -- compile the family once, equilibrate once, then
solve a whole batch of instances (same K, different data y) in a single GPU call.
We sweep the batch size and plot throughput (instances / second); it rises as the
batch amortizes the fixed cost, then plateaus when the GPU saturates.

Run:  python examples/repeated_solve_fft_lasso.py
"""
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ocs

n = 8192
K = ocs.FFTMask.random(n, ratio=0.25, seed=0)
x = ocs.Variable(n)
y = ocs.Parameter("y")

# minimize lambda||x||_1 + 1/2||K x - y||_2^2  -- compiled ONCE
compiled = ocs.minimize([ocs.l1(x, 1e-3), ocs.least_squares(K @ x - y)]).compile(
    max_blocks=100, equilibrate="scalar")
print(compiled)

rng = np.random.default_rng(1)
def measurements(N):
    Y = []
    for _ in range(N):
        z = np.zeros(n); z[rng.choice(n, n // 50, replace=False)] = rng.normal(size=n // 50)
        Y.append(np.asarray(K.matvec(jnp.asarray(z, jnp.float32))))
    return jnp.asarray(np.stack(Y), jnp.float32)

Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
thru = []
for N in Ns:
    Y = measurements(N)
    _ = compiled.solve_many({"y": Y}); jax.block_until_ready(_[0])   # warmup this N
    t = time.perf_counter()
    out = compiled.solve_many({"y": Y}); jax.block_until_ready(out[0])
    dt = time.perf_counter() - t
    thru.append(N / dt)
    print(f"  N={N:4d}   {dt:6.3f}s   {N/dt:7.1f} instances/s   "
          f"median res={float(np.median(np.asarray(out[3]))):.1e}")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.semilogx(Ns, thru, "o-", lw=2.5, ms=7, color="#1f77b4")
ax.set_xlabel("batch size  N  (instances solved together)")
ax.set_ylabel("throughput (instances / second)")
ax.set_title(f"OCS repeated solve: one FFT operator (n={n:,}), many data\n"
             "compile once, equilibrate once, solve_many in one GPU call")
ax.grid(True, which="both", ls=":", alpha=0.4)
out_path = os.path.join(os.path.dirname(__file__), "output", "repeated_solve_throughput.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.tight_layout(); fig.savefig(out_path, dpi=150)
print("-> wrote", out_path)
print(f"   throughput rose {thru[0]:.0f} -> {max(thru):.0f} instances/s "
      f"({max(thru)/thru[0]:.1f}x by batching)")
