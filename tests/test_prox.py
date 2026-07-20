"""Cone projections + type-grouped dispatch.  Run: python tests/test_prox.py"""
import numpy as np
import jax.numpy as jnp

import ocs
from ocs.prox.cones import _in_exp, _in_pow, proj_exp, proj_pow
from ocs.prox.groups import build_groups, grouped_moreau_dual


def test_soc():
    blk = ocs.SecondOrderCone()
    v = jnp.array([0.5, 2.0, 1.0, -1.0], jnp.float32)     # (s, z...)
    p = np.asarray(blk.prox(v, 1.0))
    assert np.linalg.norm(p[1:]) <= p[0] + 1e-3, "SOC infeasible"
    p2 = np.asarray(blk.prox(jnp.asarray(p), 1.0))
    assert np.allclose(p, p2, atol=1e-4), "SOC not idempotent"
    print("  SOC projection OK")


def test_psd():
    k = 5
    rng = np.random.default_rng(0)
    M = rng.normal(size=(k, k)).astype(np.float32); M = M + M.T
    blk = ocs.PSDCone(k)
    P = np.asarray(blk.prox(jnp.asarray(M.reshape(-1)), 1.0)).reshape(k, k)
    w = np.linalg.eigvalsh(P)
    assert w.min() > -1e-3, f"PSD has negative eig {w.min()}"
    P2 = np.asarray(blk.prox(jnp.asarray(P.reshape(-1)), 1.0)).reshape(k, k)
    assert np.allclose(P, P2, atol=1e-4), "PSD not idempotent"
    print(f"  PSD projection OK (min eig {w.min():.2e})")


def test_exp_pow():
    for v in [jnp.array([1.0, 1.0, -1.0]), jnp.array([-2.0, 0.5, 0.1]),
              jnp.array([0.3, -0.4, 0.9])]:
        p = proj_exp(v)
        assert bool(_in_exp(p[0], p[1], p[2])), f"exp proj infeasible for {v}"
        assert np.allclose(np.asarray(p), np.asarray(proj_exp(p)), atol=1e-3)
    for v in [jnp.array([1.0, 1.0, 2.0]), jnp.array([0.2, 3.0, -1.5])]:
        p = proj_pow(v, 0.5)
        assert bool(_in_pow(p[0], p[1], p[2], 0.5)), f"pow proj infeasible for {v}"
    print("  exp / power cone projection OK")


def test_grouping():
    # 5 SOC blocks (size 3) + 3 box blocks (size 2) -> 2 kernels, not 8
    blocks = [ocs.SecondOrderCone() for _ in range(5)] + [ocs.Box(0.0, 1.0) for _ in range(3)]
    slices, a = [], 0
    for blk in blocks:
        d = 3 if isinstance(blk, ocs.SecondOrderCone) else 2
        slices.append((a, a + d)); a += d
    m = a
    groups = build_groups(blocks, slices)
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"

    v = jnp.asarray(np.random.default_rng(1).normal(size=m), jnp.float32)
    sw = jnp.full((m,), 0.7, jnp.float32)                 # per-coordinate step
    grouped = np.asarray(grouped_moreau_dual(groups, m)(v, sw))
    # reference: per-block Moreau loop
    ref = np.zeros(m, np.float32)
    for blk, (s, e) in zip(blocks, slices):
        vi, si = v[s:e], sw[s:e]
        ref[s:e] = np.asarray(vi - si * blk.prox(vi / si, 1.0 / si))
    assert np.allclose(grouped, ref, atol=1e-5), "grouped != per-block reference"
    print(f"  grouping OK: {len(blocks)} blocks -> {len(groups)} vmap kernels, matches loop")


if __name__ == "__main__":
    print("running OCS prox/cone/grouping tests...")
    test_soc()
    test_psd()
    test_exp_pow()
    test_grouping()
    print("all prox tests passed.")
