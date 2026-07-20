"""Does microbatching recover bin-pack throughput at N=1024?

Solve 1024 instances as one monolithic batch vs. tiles of size 256/128/64,
same compiled family, same tolerance.  Decides whether B gets a second curve.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import ocs

I, B, R = 120, 40, 3
Rmat = np.random.default_rng(1).uniform(.1, 1, (R, I)).astype("f")
K = ocs.BinPack(Rmat, B); negK = ocs.Weighted(K, -1.0)
x = ocs.Variable(I*B); rhs = ocs.Parameter("rhs"); cost = ocs.Parameter("c")
se = jnp.full(I*B, np.sqrt(0.1), jnp.float32)
comp = ocs.minimize([ocs.nonneg(x), ocs.nonneg(negK@x+rhs),
                     ocs.sum_squares(ocs.Diagonal(se)@x)],
                    cost=cost).compile(max_blocks=500, tol=1e-12)
print(f"LP size = {I*B} vars/instance")

rng = np.random.default_rng(4); per = 0.6*Rmat.sum(1)/B
def inst():
    C = (per[:, None]*rng.uniform(.7, 1.3, (R, B))).astype("f"); w = rng.uniform(.5, 1.5, I).astype("f")
    return np.concatenate([np.ones(I, "f"), C.reshape(-1)]), -np.repeat(w, B).astype("f")
def data(N):
    return {"rhs": jnp.asarray(np.stack([inst()[0] for _ in range(N)])),
            "c":   jnp.asarray(np.stack([inst()[1] for _ in range(N)]))}

N = 1024
d = data(N)

def tile_throughput(bs):
    """Throughput solving N instances in tiles of size bs."""
    idx = [(i, min(i+bs, N)) for i in range(0, N, bs)]
    sub = [{"rhs": d["rhs"][a:b], "c": d["c"][a:b]} for a, b in idx]
    for s in sub:                       # warm each distinct tile shape (last may differ)
        jax.block_until_ready(comp.solve_many(s)[0])
    t = time.perf_counter()
    for _ in range(3):
        for s in sub:
            o = comp.solve_many(s)
        jax.block_until_ready(o[0])
    dt = (time.perf_counter()-t)/3
    return N/dt

for bs in [1024, 512, 256, 128, 64]:
    print(f"tile={bs:5d}:  {tile_throughput(bs):7.1f} inst/s")
