"""Conservative iteration-only specialization for a proven fixed metric.

The public prox, certificate and refinement callbacks keep their ordinary
validation. This factory is used only by fixed-step standard PDHG.
"""
import jax
import jax.numpy as jnp

from ..prox.base import _LinQuad, DIAGONAL
from ..prox.library import NonNeg


_LINQUAD_PROX = _LinQuad.prox
_NONNEG_PROX = NonNeg.prox
_FIELDS = {"base", "d", "e", "scaling_class", "coordinate_separable",
           "eval_kind", "_e_mode", "_quadratic", "value_available",
           "conj_value_available"}


@jax.jit
def _safe_metric(t, e, d):
    # Deliberately reject extremes to the generic path. A margin from both
    # float32 range boundaries also proves validity for float64 point state.
    # These arrays are runtime inputs to this one setup reduction, preventing
    # XLA from constant-folding a full metric reduction in every loop trace.
    t, e, d = (jnp.asarray(a, jnp.float32) for a in (t, e, d))
    info = jnp.finfo(jnp.float32)
    lower, upper = info.tiny * 16, info.max / 16
    product = t * e
    scale = 1.0 + product
    transformed = t / scale
    return (jnp.all(jnp.isfinite(t) & (t >= lower) & (t <= upper))
            & jnp.all(jnp.isfinite(e) & (e >= 0))
            & jnp.all(jnp.isfinite(d))
            & jnp.all(jnp.isfinite(product) & (product <= upper))
            & jnp.all(jnp.isfinite(transformed) & (transformed >= lower)))


def fixed_nonnegative_quadratic_prox(block, tau, n):
    """Return a fixed-step prox or None for the fully validated fallback.

    Recognize only the unchanged built-in NonNeg + linear/quadratic calculus
    block. The returned callable accepts a point, not an arbitrary metric;
    its step is the exact family tau. It removes only redundant validity
    reductions, retaining the original subtraction, division and clipping.
    """
    if (type(block) is not _LinQuad or type(block.base) is not NonNeg
            or _LinQuad.prox is not _LINQUAD_PROX or NonNeg.prox is not _NONNEG_PROX
            or set(vars(block)) != _FIELDS or vars(block.base)
            or block.scaling_class != DIAGONAL
            or not block.coordinate_separable or block._quadratic):
        return None
    if any(isinstance(a, jax.core.Tracer)
           for a in jax.tree_util.tree_leaves((tau, block.d, block.e))):
        return None
    tau = jnp.asarray(tau)
    if tau.dtype not in (jnp.float32, jnp.float64):
        return None
    if any(jnp.broadcast_shapes(jnp.shape(a), (n,)) != (n,)
           for a in (tau, block.d, block.e)):
        return None
    if not bool(_safe_metric(tau, block.e, block.d)):
        return None
    d, e = block.d, block.e

    def apply(v):
        if v.shape != (n,) or v.dtype not in (jnp.float32, jnp.float64):
            return block.prox(v, jnp.broadcast_to(tau, (n,)))
        t = jnp.asarray(jnp.broadcast_to(tau, (n,)), dtype=v.dtype)
        scale = 1.0 + t * e
        return jnp.maximum((v - t * d) / scale, 0.0)

    return apply
