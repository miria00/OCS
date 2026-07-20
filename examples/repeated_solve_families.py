"""Repeated-solve families for QP and SOCP -- the "fixed K, many data" regime,
operator-pure (FFT), native OCS (no CVXPY).

QP   : min_x 1/2||K x - b||_2^2  s.t.  x >= 0            (nonneg least squares)
SOCP : min_x ||x||_2             s.t.  K x = y            (min-norm recovery)

K = subsampled FFT, never formed.  Each family is compiled ONCE and solved for a
batch of data vectors in a single GPU call; we sweep the batch size and plot
throughput (instances / second).

Run:  python examples/repeated_solve_families.py
"""
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ocs

n = 4096
K = ocs.FFTMask.random(n, ratio=0.4, seed=0)
Ns = [1, 4, 16, 64, 256]
rng = np.random.default_rng(1)


def bench(compiled, make_data, Ns):
    thru = []
    for N in Ns:
        data = {"d": jnp.asarray(np.stack([make_data() for _ in range(N)]), jnp.float32)}
        _ = compiled.solve_many(data); jax.block_until_ready(_[0])       # warmup
        t = time.perf_counter()
        out = compiled.solve_many(data); jax.block_until_ready(out[0])
        thru.append(N / (time.perf_counter() - t))
    return thru


# ---- QP: nonneg least squares, min 1/2||Kx - b||^2 s.t. x>=0 ----
x = ocs.Variable(n); d = ocs.Parameter("d")
qp = ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - d)]).compile(max_blocks=150)
def qp_data():
    z = np.zeros(n); z[rng.choice(n, n // 20, replace=False)] = np.abs(rng.normal(size=n // 20))
    return np.asarray(K.matvec(jnp.asarray(z, jnp.float32)))
qp_thru = bench(qp, qp_data, Ns)
print("QP  ", qp, "\n   ", [f"N={N}:{t:.0f}/s" for N, t in zip(Ns, qp_thru)])

# ---- SOCP: min ||x||_2 s.t. Kx = y ----
x2 = ocs.Variable(n); d2 = ocs.Parameter("d")
socp = ocs.minimize([ocs.l2(x2, 1.0), ocs.equality(K @ x2 - d2)]).compile(max_blocks=150)
def socp_data():
    z = rng.normal(size=n).astype(np.float32)
    return np.asarray(K.matvec(jnp.asarray(z, jnp.float32)))
socp_thru = bench(socp, socp_data, Ns)
print("SOCP", socp, "\n   ", [f"N={N}:{t:.0f}/s" for N, t in zip(Ns, socp_thru)])

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.semilogx(Ns, qp_thru, "o-", lw=2.5, ms=8, label="QP  (nonneg least squares)")
ax.semilogx(Ns, socp_thru, "s-", lw=2.5, ms=8, label="SOCP  (min-norm, Kx = y)")
ax.set_xlabel("batch size  N  (instances solved together)")
ax.set_ylabel("throughput (instances / second)")
ax.set_title(f"OCS repeated solve, operator-pure FFT (n={n:,})\n"
             "one compiled family, many data instances, one GPU call")
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
out = os.path.join(os.path.dirname(__file__), "output", "repeated_solve_families.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("-> wrote", out)
