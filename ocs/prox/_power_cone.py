"""Euclidean projection onto a three-dimensional power cone.

The scalar KKT reduction is also used by SCS (src/cones.c). Unlike a clipped
multivariate Newton solve, the logarithmic equation below is monotone and
keeps its root bracket. Negative-coordinate quadratic roots are rationalized;
no positive coordinate floor is imposed on the mathematical projection.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _positive_root(coordinate, coefficient):
    """Positive root of x*x-coordinate*x-coefficient=0, coefficient>=0."""
    discriminant = jnp.sqrt(coordinate*coordinate + 4.0*coefficient)
    direct = 0.5*(coordinate + discriminant)
    denominator = discriminant-coordinate
    rationalized = 2.0*coefficient/jnp.where(denominator > 0, denominator, 1.0)
    return jnp.where(coordinate >= 0, direct, rationalized)


def _log_positive(value):
    return jnp.log(jnp.where(value > 0, value, 1.0))


def _project_unit(point, alpha):
    x0, y0, z0 = point
    beta = 1.0-alpha
    radius = jnp.abs(z0)

    # Compare logarithms only in the positive domain. Zero-radius points have
    # their own exact face solution, avoiding log(0) and an unnecessary search.
    log_radius = _log_positive(radius)
    primal = ((x0 > 0) & (y0 > 0)
              & (alpha*_log_positive(x0) + beta*_log_positive(y0) >= log_radius))
    polar = ((x0 < 0) & (y0 < 0)
             & (alpha*(_log_positive(-x0)-jnp.log(alpha))
                + beta*(_log_positive(-y0)-jnp.log(beta)) >= log_radius))

    def curved(_):
        def radius_parts(fraction, near_upper):
            # Store the smaller of r/R and (R-r)/R directly. Near a flat
            # face r/R can round to one while R-r is still essential to the
            # quadratic roots; forming it by subtraction loses the projection.
            projected_fraction = jnp.where(near_upper, 1.0-fraction, fraction)
            remaining_fraction = jnp.where(near_upper, fraction, 1.0-fraction)
            log_fraction = jnp.log(fraction)
            log_other = jnp.log1p(-fraction)
            return (radius*projected_fraction, radius*remaining_fraction,
                    log_radius+jnp.where(near_upper, log_other, log_fraction),
                    log_radius+jnp.where(near_upper, log_fraction, log_other))

        def coordinates(fraction, near_upper):
            projected_radius, remainder, _lr, _ld = radius_parts(fraction, near_upper)
            product = projected_radius*remainder
            return (_positive_root(x0, alpha*product),
                    _positive_root(y0, beta*product), projected_radius)

        def log_ratio(coordinate, weight, fraction, near_upper):
            # For coordinate<0, x/r=2*a*(radius-r)/(sqrt(...)-coordinate).
            # For coordinate>=0, subtract logs rather than forming x/r,
            # which may overflow near the zero endpoint.
            r, remaining, log_r, log_remaining = radius_parts(fraction, near_upper)
            coefficient = weight*r*remaining
            discriminant = jnp.sqrt(coordinate*coordinate + 4.0*coefficient)
            positive = 0.5*(coordinate+discriminant)
            direct = jnp.log(positive) - log_r
            negative = (jnp.log(2.0*weight) + log_remaining
                        - jnp.log(discriminant-coordinate))
            return jnp.where(coordinate >= 0, direct, negative)

        def equation(fraction, near_upper):
            value = (alpha*log_ratio(x0, alpha, fraction, near_upper)
                     + beta*log_ratio(y0, beta, fraction, near_upper))
            # F decreases with r, so reverse its sign when searching R-r.
            return jnp.where(near_upper, -value, value)

        near_upper = equation(jnp.asarray(0.5, point.dtype), False) > 0

        def body(state):
            lower, upper, iteration = state
            middle = lower + 0.5*(upper-lower)
            value = equation(middle, near_upper)
            return (jnp.where(value >= 0, middle, lower),
                    jnp.where(value < 0, middle, upper), iteration+1)

        def unfinished(state):
            lower, upper, iteration = state
            middle = lower + 0.5*(upper-lower)
            return (iteration < 64) & (middle > lower) & (middle < upper)

        lower, upper, _ = lax.while_loop(
            unfinished, body, (jnp.asarray(0.0, point.dtype),
                               jnp.asarray(0.5, point.dtype), 0))
        # Select the side with x**alpha*y**beta >= r. Near an axis, a
        # tiny error in x would otherwise be amplified by the fractional
        # power when clipping z. The two brackets have opposite orientations.
        fraction = jnp.where(near_upper, upper, lower)
        x, y, r = coordinates(fraction, near_upper)
        # The root can lie between adjacent float32 values. Keep the returned
        # point on the feasible side of the boundary despite that rounding.
        bound = x**alpha * y**beta
        return jnp.stack([x, y, jnp.sign(z0)*jnp.minimum(r, bound)])

    return lax.cond(
        radius == 0,
        lambda _: jnp.stack([jnp.maximum(x0, 0.0), jnp.maximum(y0, 0.0), z0]),
        lambda _: lax.cond(primal, lambda _: point,
                           lambda _: lax.cond(polar, lambda _: jnp.zeros_like(point),
                                              curved, None), None),
        None,
    )


def proj_pow(value, alpha=0.5):
    """Project onto {x,y>=0 : x**alpha*y**(1-alpha)>=abs(z)}.

    Each triple is normalized independently. The method is compatible with
    jit/vmap and uses no matrix or solver-state information.
    """
    point = jnp.asarray(value, dtype=jnp.float32)
    exponent = jnp.asarray(alpha, dtype=point.dtype)
    scale = jnp.max(jnp.abs(point))
    normalized = point/jnp.where(scale > 0, scale, 1.0)
    projected = scale*_project_unit(normalized, exponent)
    valid = jnp.all(jnp.isfinite(point)) & (exponent > 0) & (exponent < 1)
    return jnp.where(valid, projected, jnp.full_like(point, jnp.nan))
