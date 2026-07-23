"""Type-grouped GPU dispatch for a heterogeneous list of blocks.

Given f separable as a list of many different f_i, can we
evaluate prox_f efficiently on a GPU?  Solution: group blocks by (type, size) and
evaluate each group as a single vmapped kernel.  The number of kernels is the
number of distinct block types/sizes, not the number of blocks — so 10,000
identical second-order cones cost one kernel, fully parallel.  Per-block Python
branching (which causes warp divergence) is eliminated.

Each group carries a precomputed gather index into the stacked dual vector, so
the gather / vmap-prox / scatter is fully vectorized (no runtime Python loop over
blocks; the loop over groups is unrolled at trace time and is tiny).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


class _Group:
    __slots__ = ("idx", "size", "count", "block")

    def __init__(self, idx, size, count, block):
        self.idx = idx          # int32 gather index, shape (count*size,)
        self.size = size        # per-block dimension
        self.count = count      # number of blocks in this group
        self.block = block      # a representative block (same type + params)


def build_groups(blocks, slices):
    """Partition (block, slice) pairs by (block.group_key(), size)."""
    buckets = {}
    for blk, (s, e) in zip(blocks, slices):
        key = (blk.group_key(), e - s)
        buckets.setdefault(key, ([], blk))[0].append((s, e))
    groups = []
    for (key, size), (spans, blk) in buckets.items():
        idx = jnp.concatenate([jnp.arange(s, e, dtype=jnp.int32) for s, e in spans])
        groups.append(_Group(idx, size, len(spans), blk))
    return groups


def grouped_moreau_dual(groups, m):
    """Return apply(v, sw) computing the stacked dual prox  prox_{sw f*}(v)
    block-wise, via Moreau:  v_i - sw_i * prox_{f_i}(v_i / sw_i, 1/sw_i).

    ``sw`` is the (m,) per-coordinate dual step (diagonal equilibration); for a
    coupled (scalar-class) block it is uniform over the block's rows, so the same
    code serves scalar and diagonal blocks.  Each group runs one vmapped prox
    over its (count, size) segment (and matching step segment).
    """
    def apply(v, sw):
        out = jnp.zeros(m, v.dtype)
        for g in groups:
            seg = v[g.idx].reshape(g.count, g.size)                 # gather
            sws = sw[g.idx].reshape(g.count, g.size)                # per-block step
            pr = jax.vmap(lambda row, s: g.block.prox(row / s, 1.0 / s))(seg, sws)
            out = out.at[g.idx].set((seg - sws * pr).reshape(-1))   # scatter
        return out
    return apply
