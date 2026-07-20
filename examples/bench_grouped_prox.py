"""Grouped prox dispatch -- the evidence for Stephen's direct question:

    "if I give f as separable, a list of many different f_i, can I efficiently
     evaluate prox_f on a GPU?"

Answer: group blocks by type and launch ONE vmapped kernel per group.  Cost then
scales with the number of distinct block TYPES, not the number of blocks.

We time the stacked dual prox for a growing number of second-order cone blocks
(all the same type), two ways:
  * naive  : a Python loop calling each block's prox separately;
  * grouped: build_groups + grouped_moreau_dual (one vmap kernel).

Run:  python examples/bench_grouped_prox.py
"""
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ocs
from ocs.prox.groups import build_groups, grouped_moreau_dual

CONE = 8                                   # each SOC block has dimension 8
# the naive loop UNROLLS one prox per block into the graph, so its compile blows
# up past a few hundred blocks -- that is itself the point.  Grouped scales on.
Bs_naive = [8, 32, 128, 512]
Bs = [8, 32, 128, 512, 2048, 8192]

def naive_apply_factory(blocks, slices, m):
    def apply(v, sw):
        out = jnp.zeros(m, v.dtype)
        for blk, (s, e) in zip(blocks, slices):
            vi, si = v[s:e], sw[s:e]
            out = out.at[s:e].set(vi - si * blk.prox(vi / si, 1.0 / si))
        return out
    return jax.jit(apply)

def bench(fn, v, sw, reps=50):
    o = fn(v, sw); jax.block_until_ready(o)         # compile
    t = time.perf_counter()
    for _ in range(reps):
        o = fn(v, sw)
    jax.block_until_ready(o)
    return (time.perf_counter() - t) / reps * 1e3   # ms

naive_t, group_t = [], []
for B in Bs:
    blocks = [ocs.SecondOrderCone() for _ in range(B)]
    slices = [(i * CONE, (i + 1) * CONE) for i in range(B)]
    m = B * CONE
    v = jnp.asarray(np.random.default_rng(0).normal(size=m), jnp.float32)
    sw = jnp.full((m,), 0.7, jnp.float32)

    grouped = jax.jit(grouped_moreau_dual(build_groups(blocks, slices), m))
    tg = bench(grouped, v, sw)
    group_t.append(tg)
    if B in Bs_naive:
        tn = bench(naive_apply_factory(blocks, slices, m), v, sw)
        naive_t.append(tn)
        print(f"  {B:6d} SOC blocks (m={m:7d}):  naive {tn:8.3f} ms   grouped {tg:7.3f} ms"
              f"   {tn/tg:6.1f}x")
    else:
        print(f"  {B:6d} SOC blocks (m={m:7d}):  naive  (compile blows up)   "
              f"grouped {tg:7.3f} ms")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(Bs_naive, naive_t, "s--", lw=2.4, ms=8, color="#d62728", label="naive per-block loop")
ax.loglog(Bs, group_t, "o-", lw=2.6, ms=8, color="#1f77b4", label="grouped vmap (1 kernel / type)")
ax.set_xlabel("number of prox blocks (all second-order cones)")
ax.set_ylabel("dual prox evaluation time (ms)")
ax.set_title("Type-grouped GPU prox dispatch\n"
             "a list of many f_i evaluates as one batched kernel per type")
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
out = os.path.join(os.path.dirname(__file__), "output", "grouped_prox.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("-> wrote", out)
