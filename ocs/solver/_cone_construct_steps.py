"""Stable float32 arithmetic for opt-in symmetric-cone metric construction.

SOC: evaluate the original projected geodesic step with scaled exponentials
and a log-determinant identity. PSD: explicitly different, per-element bounded
geodesic step. Invalid inputs preserve the old carrier and return an explicit failure flag.
"""
import jax
import jax.numpy as jnp
from ..prox import cone_metric as cm


def soc_step(value, tangent, eta, box=2.):
    """Rows of bounded Jordan elements; eta >= 0; box <= 2."""
    l1, l2, _ = cm.soc_eig(value)
    half = cm.soc_spectral(value, jnp.sqrt)
    h1, h2, direction = cm.soc_eig(tangent)
    # h1 >= h2, hence hi_exp >= lo_exp. No positive exponent is evaluated.
    hi_exp, lo_exp = -eta * h2, -eta * h1
    small = jnp.exp(jnp.minimum(lo_exp - hi_exp, 0.))
    scaled_step = jnp.concatenate([
        (.5 * (1. + small))[:, None],
        (-.5 * (1. - small))[:, None] * direction], axis=1)
    scaled = cm.apply_soc(half, cm.soc_det(half), scaled_step)
    large, _, eigdir = cm.soc_eig(scaled)
    log_large_scaled = jnp.log(large)
    # det(P(sqrt(value)) step) = det(value) det(step).
    # Cancel hi_exp symbolically, not by subtracting two large logarithms.
    log_large = hi_exp + log_large_scaled
    log_small = jnp.log(l1) + jnp.log(l2) + lo_exp - log_large_scaled
    a = jnp.exp(jnp.clip(log_large, -box, box))
    b = jnp.exp(jnp.clip(log_small, -box, box))
    result = jnp.concatenate([(.5 * (a + b))[:, None],
                              (.5 * (a - b))[:, None] * eigdir], axis=1)
    valid = (jnp.isfinite(eta) & (eta >= 0) & (box >= 0) & (box <= 2)
             & jnp.all(l2 >= jnp.exp(-box)*(1-1e-4))
             & jnp.all(l1 <= jnp.exp(box)*(1+1e-4))
             & jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(tangent))
             & jnp.all(jnp.isfinite(hi_exp)) & jnp.all(jnp.isfinite(lo_exp))
             & jnp.all(large > 0) & jnp.all(jnp.isfinite(result)))
    return jnp.where(valid & (eta != 0), result, value), valid


def psd_bounded_step(value, tangent, eta, box=2.):
    """Per-carrier exponent norm <= .5, so preclip spectrum is in e^±2.5.

    A step-size change, not an exact evaluation of the old unlimited step.
    Return the actual per-carrier eta so its use cannot be hidden in records.
    """
    def mm(a, b):
        return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)

    def spectral(a, fn):
        d, q = jnp.linalg.eigh(.5 * (a + jnp.swapaxes(a, -1, -2)))
        return mm(q * fn(d)[..., None, :], jnp.swapaxes(q, -1, -2))

    scale = jnp.maximum(1., jnp.max(jnp.abs(tangent), axis=(-2, -1)))
    scaled = tangent / scale[..., None, None]
    norm_scaled = jnp.sqrt(jnp.sum(scaled * scaled, axis=(-2, -1)))
    max_eta = (.5 / scale) / jnp.maximum(norm_scaled, jnp.finfo(value.dtype).tiny)
    effective_eta = jnp.minimum(eta, max_eta)
    oldvals, oldvecs = jnp.linalg.eigh(.5*(value+jnp.swapaxes(value,-1,-2)))
    half = mm(oldvecs * jnp.sqrt(oldvals)[...,None,:],jnp.swapaxes(oldvecs,-1,-2))
    step = spectral(-effective_eta[..., None, None] * tangent, jnp.exp)
    trial = mm(mm(half, step), half)
    result = spectral(trial, lambda d: jnp.exp(jnp.clip(jnp.log(d), -box, box)))
    valid = (jnp.isfinite(eta) & (eta >= 0) & (box >= 0) & (box <= 2)
             & jnp.all(oldvals >= jnp.exp(-box)*(1-1e-4))
             & jnp.all(oldvals <= jnp.exp(box)*(1+1e-4))
             & jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(tangent))
             & jnp.all(jnp.isfinite(result))
             & jnp.all((effective_eta > 0) | (eta == 0)))
    return jnp.where(valid & (eta != 0), result, value), valid, effective_eta
