"""Parameter-preserving, type-grouped dispatch for heterogeneous prox blocks.

Built-in blocks of the same type, dimensions and parameter shapes share a
vmapped evaluation. Their numeric parameters are data, not part of a cache key
and not borrowed from a representative instance. Nested calculus wrappers use
the same recursive packing contract. Dispatch stays independent of the number
of distinct parameter values.

Custom blocks have no implicit equivalence contract: even equal ``group_key``
values cannot establish that their prox functions agree. They are evaluated
separately, except for repeated references to the same object. A new built-in
must explicitly declare its prox-relevant fields in the registry below before
it can participate in parameter batching.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import base, block_metric, cone_metric, cones, library, spatial_soc


# Exact classes only: a subclass may add parameters or override the prox.
# Entries are (dynamic numeric fields, static structural fields).
_BATCH_FIELDS = {
    library.SumSquares: ((), ()),
    library.Quadratic: (("P", "q"), ("dim",)),
    library.L1: (("lam",), ()),
    library.L2: (("lam",), ()),
    library.GroupL2: (("lam",), ("g",)),
    library.PiecewiseLinearConvex: (("s", "r"), ()),
    library.NonNeg: ((), ()),
    library.Box: (("lo", "hi"), ()),
    library.Zero: ((), ()),
    library.Equality: ((), ()),
    library.Pinball: (("tau",), ()),
    library.Logistic: ((), ()),
    cones.SecondOrderCone: ((), ()),
    cones.PSDCone: ((), ("k",)),
    cones.ExpCone: ((), ()),
    cones.PowerCone: (("alpha",), ()),
    cones.PowerConeBatch: (("alpha",), ("q",)),
    cones.SOCBatch: ((), ("q", "d")),
    cones.ExpConeBatch: ((), ("q",)),
    base._Scaled: (("c",), ()),
    base._AffinePre: (("a", "b"), ()),
    base._LinQuad: (("d", "e"), ()),
}
_WRAPPERS = (base._Scaled, base._AffinePre, base._LinQuad)


class _PackedBlock:
    """Trace-time template; constructors never receive batched tracers."""

    def __init__(self, block, names, child):
        self.cls = type(block)
        self.attributes = vars(block).copy()
        self.names = names
        self.child = child

    def bind(self, parameters):
        values, child_parameters = parameters
        block = object.__new__(self.cls)
        block.__dict__.update(self.attributes)
        for name, value in zip(self.names, values):
            setattr(block, name, value)
        if self.child is not None:
            block.base = self.child.bind(child_parameters)
        return block


def _packing(block):
    """Return (structural signature, template, parameter tree), or None.

    Additional scalar metadata (e.g. wrapper branch flags) is static and must
    agree. Unknown instance arrays cannot silently be copied from another
    block: without a declared packing rule, use the safe identity fallback.
    Smooth-gradient closures are not called by proximal dispatch.
    """
    cls = type(block)
    if cls not in _BATCH_FIELDS or "prox" in vars(block):
        return None
    names, static_names = _BATCH_FIELDS[cls]
    child, child_key, child_parameters = None, None, ()
    if cls in _WRAPPERS:
        packed_child = _packing(block.base)
        if packed_child is None:
            return None
        child_key, child, child_parameters = packed_child
    metadata = []
    ignored = {*names, *static_names, "base", "smooth_grad"}
    for name, value in sorted(vars(block).items()):
        if name in ignored:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            metadata.append((name, value))
        else:
            # An unregistered cache or method can depend on numeric fields.
            # Keeping it from a representative would be unsafe.
            return None
    values = tuple(jnp.asarray(getattr(block, name)) for name in names)
    parameter_shapes = tuple((value.shape, str(value.dtype)) for value in values)
    static = tuple((name, getattr(block, name)) for name in static_names)
    key = (cls, block.scaling_class,
           getattr(block, "coordinate_separable", None),
           static, tuple(metadata), parameter_shapes, child_key)
    return key, _PackedBlock(block, names, child), (values, child_parameters)


class _Group:
    __slots__ = ("idx", "span", "size", "count", "block", "parameters", "_packed", "_parameter_axes",
                 "metric")

    def __init__(self, idx, span, size, blocks, packed, parameters, parameter_axes, metric=None):
        self.idx = idx
        self.span = span
        self.size = size
        self.count = len(blocks)
        # Kept for structural metadata consumers; parameters may differ.
        self.block = blocks[0]
        self.parameters = parameters
        self._packed = packed
        self._parameter_axes = parameter_axes
        # Optional declared row-side block metric shared by every member:
        # (k, ((start, end, BlockMetric), ...)). None keeps the original path.
        self.metric = metric

    def dual_prox(self, rows, steps, *, compact_step=False):
        if compact_step:
            # Uniformity of each row was checked against the family's fixed
            # sigma. A runtime scalar omega preserves that equality. Retain
            # one entry PER block, since different blocks can have different
            # steps even when their functions share one dispatch group.
            steps = steps[:, :1]
        if self.metric is not None:
            return self._metric_dual_prox(rows, steps)
        if self._packed is None:
            return jax.vmap(lambda v, s: v - s * self.block.prox(v / s, 1 / s))(
                rows, steps)

        def apply(v, s, parameters):
            block = self._packed.bind(parameters)
            return v - s * block.prox(v / s, 1 / s)

        return jax.vmap(apply, in_axes=(0, 0, self._parameter_axes))(
            rows, steps, self.parameters)

    def _metric_dual_prox(self, rows, steps):
        """Weighted Moreau per mathematical block; members never share a norm.

        Dispatches on the carrier type: a ``ConeAutomorphism`` range takes the
        closed-form cone update, a ``SpatialFullSOCMetric`` range the weighted
        second-order-cone projection of ``ocs.prox.spatial_soc``, and a
        ``BlockMetric`` range the norm-block one.  Keyed by the carrier and not by
        the block, because the two cone classes meet only at their scalar members
        and a carrier that reached the wrong branch would apply a DIFFERENT
        metric without any check noticing.
        """
        k, ranges = self.metric
        if isinstance(ranges[0][2], cone_metric.ConeAutomorphism):
            def moreau(blk, v, s):
                return cone_metric.automorphism_moreau(blk, v, s, ranges)
        elif isinstance(ranges[0][2], spatial_soc.SpatialFullSOCMetric):
            def moreau(blk, v, s):
                return spatial_soc.spatial_full_moreau(blk, v, s, ranges)
        else:
            def moreau(blk, v, s):
                return block_metric.metric_moreau(blk, v, s, ranges, k)
        if self._packed is None:
            return jax.vmap(lambda v, s: moreau(self.block, v, s))(rows, steps)

        def apply(v, s, parameters):
            return moreau(self._packed.bind(parameters), v, s)

        return jax.vmap(apply, in_axes=(0, 0, self._parameter_axes))(
            rows, steps, self.parameters)


def _stack_parameters(parameters):
    """Stack changing leaves; share identical array objects with in_axes=None.

    For example, thousands of scaled wrappers around the same dense quadratic
    need one matrix, plus one scale per wrapper, not thousands of matrices.
    No host equality test or device-to-host copy of numeric arrays is needed.
    """
    first, structure = jax.tree_util.tree_flatten(parameters[0])
    rows = [first, *[jax.tree_util.tree_leaves(p) for p in parameters[1:]]]
    values, axes = [], []
    for leaves in zip(*rows):
        shared = all(value is leaves[0] for value in leaves)
        values.append(leaves[0] if shared else jnp.stack(leaves))
        axes.append(None if shared else 0)
    return (jax.tree_util.tree_unflatten(structure, values),
            jax.tree_util.tree_unflatten(structure, axes))


def build_groups(blocks, slices, metrics=None):
    """Group by registered type/shape schema, preserving every numeric field.

    Repeated instances and parameter-free groups keep a shared prox without
    duplicating parameter arrays. Objects and parameters are static family
    configuration: rebuild groups after changing a block.

    ``metrics`` (compiler-only) is None or one entry per block: None, or a
    validated ``(k, ranges)`` row-side block metric. Blocks are batched only
    with blocks carrying the identical metric declaration.
    """
    blocks, slices = list(blocks), list(slices)
    if len(blocks) != len(slices):
        raise ValueError("one slice is required for every proximal block")
    if metrics is not None and len(metrics) != len(blocks):
        raise ValueError("one metric entry is required for every proximal block")
    buckets = {}
    for position, (block, (start, end)) in enumerate(zip(blocks, slices)):
        if start < 0 or end <= start:
            raise ValueError("proximal block slices must have positive size and nonnegative start")
        packed = _packing(block)
        if packed is None:
            key, template, parameters = ("instance", id(block)), None, ()
        else:
            key, template, parameters = packed
        if metrics is not None and metrics[position] is not None:
            k, ranges = metrics[position]
            key = (key, "block_metric", k, tuple((entry[0], entry[1], id(entry[2])) + tuple(entry[3:])
                                                 for entry in ranges))
            template = (template, metrics[position])
        bucket = buckets.setdefault((key, end-start), ([], [], template, []))
        bucket[0].append((start, end))
        bucket[1].append(block)
        bucket[3].append(parameters)
    groups = []
    for (key, size), (spans, members, template, parameters) in buckets.items():
        metric = None
        if isinstance(key, tuple) and len(key) == 4 and key[1] == "block_metric":
            template, metric = template
        contiguous = all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        span = (spans[0][0], spans[-1][1]) if contiguous else None
        # A contiguous group needs only two bounds, not an index per coordinate.
        # Keep the indexed representation for interleaved or overlapping spans.
        idx = None if contiguous else jnp.asarray(np.concatenate([
            np.arange(s, e, dtype=np.int32) for s, e in spans]))
        if all(member is members[0] for member in members) or not jax.tree_util.tree_leaves(parameters[0]):
            template, packed_parameters, parameter_axes = None, (), ()
        else:
            packed_parameters, parameter_axes = _stack_parameters(parameters)
        groups.append(_Group(idx, span, size, members, template, packed_parameters, parameter_axes,
                             metric))
    return groups


def _verified_scalar_groups(groups, m, static_sigma, *, compact_scalar_sumsquares=False):
    """Check fixed family metrics once, without copying the vector to the host.

    Sigma is an argument to a separate compiled reduction, rather than a
    captured constant in the solver's prox. Only one boolean per group is
    transferred back. Failed checks retain ordinary vector validation.
    """
    disabled = (False,) * len(groups)
    if static_sigma is None or isinstance(static_sigma, jax.core.Tracer):
        return disabled
    sigma = jnp.asarray(static_sigma)
    if sigma.ndim == 0:
        # A scalar is structurally uniform; only finite positivity is checked.
        valid = bool(jnp.isfinite(sigma) & (sigma > 0))
        return tuple(valid and (group.block.scaling_class == base.SCALAR or
                     (compact_scalar_sumsquares and
                      type(group.block) is library.SumSquares and not vars(group.block)))
                     for group in groups)
    if sigma.shape != (m,) or not any(
            group.block.scaling_class == base.SCALAR for group in groups):
        return disabled

    @jax.jit
    def verify(sigma):
        flags = []
        for group in groups:
            if group.block.scaling_class != base.SCALAR:
                flags.append(jnp.asarray(False))
                continue
            if group.span is not None:
                start, end = group.span
                if end > m:
                    flags.append(jnp.asarray(False))
                    continue
                rows = sigma[start:end].reshape(group.count, group.size)
            else:
                rows = sigma[group.idx].reshape(group.count, group.size)
            flags.append(jnp.all(jnp.isfinite(rows) & (rows > 0)
                                  & (rows == rows[:, :1])))
        return jnp.stack(flags)

    return tuple(bool(flag) for flag in np.asarray(verify(sigma)))


def grouped_moreau_dual(groups, m, *, static_sigma=None, compact_scalar_sumsquares=False):
    """Return the stacked dual prox ``prox_{sw f*}(v)`` by weighted Moreau.

    ``sw`` has one step per coordinate. Coupled blocks require a uniform step
    within each block, but different blocks in a group can have different
    steps. Contiguous groups use slices, and an ordered partition is assembled
    by concatenation. Other layouts retain indexed gather/scatter semantics.

    The optional compiler-only ``static_sigma`` specializes scalar-tier groups
    after exact finite, positive, within-block uniformity checks. That path
    requires every subsequent ``sw`` to be this sigma times a scalar; it is
    unsuitable for changing coordinatewise metrics. Without this argument,
    arbitrary runtime vectors retain their full metric validation.

    ``compact_scalar_sumsquares`` is an additional compiler-only specialization
    for an unmodified built-in SumSquares with scalar ``static_sigma``. The
    compiler must disable it when online coordinatewise adaptation is active.
    The scalar passed to the prox still undergoes the ordinary metric checks;
    vectors, wrappers and custom blocks retain their existing dispatch.
    """
    groups = tuple(groups)
    compact = _verified_scalar_groups(groups, m, static_sigma,
                                     compact_scalar_sumsquares=compact_scalar_sumsquares)
    cursor, partition = 0, bool(groups)
    for group in groups:
        if group.span is None or group.span[0] != cursor:
            partition = False
            break
        cursor = group.span[1]
    partition = partition and cursor == m

    def sliced(group, v, sw, compact_step):
        start, end = group.span
        rows = v[start:end].reshape(group.count, group.size)
        steps = sw[start:end].reshape(group.count, group.size)
        return group.dual_prox(rows, steps, compact_step=compact_step).reshape(-1)

    def apply(v, sw):
        if partition and v.shape == (m,) and sw.shape == (m,):
            return jnp.concatenate([sliced(group, v, sw, compact_step)
                                    for group, compact_step in zip(groups, compact)])
        out = jnp.zeros(m, v.dtype)
        for group, compact_step in zip(groups, compact):
            compact_step = compact_step and sw.shape == (m,)
            if (group.span is not None and group.span[1] <= m
                    and group.span[1] <= v.shape[0] and group.span[1] <= sw.shape[0]):
                values = sliced(group, v, sw, compact_step)
                out = jax.lax.dynamic_update_slice(out, values, (group.span[0],))
                continue
            # An out-of-range contiguous span must keep the previous indexed
            # clipping/drop behavior; Python slices would shorten its block.
            idx = (group.idx if group.idx is not None else
                   jnp.arange(group.span[0], group.span[1], dtype=jnp.int32))
            rows = v[idx].reshape(group.count, group.size)
            steps = sw[idx].reshape(group.count, group.size)
            values = group.dual_prox(rows, steps, compact_step=compact_step)
            out = out.at[idx].set(values.reshape(-1))
        return out
    return apply
