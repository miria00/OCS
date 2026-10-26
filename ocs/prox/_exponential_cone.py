"""Exponential-cone projection in normalized input units.

The scalar equation and base interval follow H. A. Friberg,
``Projection onto the exponential cone: a univariate root-finding problem''
(2021), https://docs.mosek.com/whitepapers/expcone-proj.pdf.
Geometric bracketing and safeguarded Newton avoid an imposed angular range.
Only nonpositive exponentials and scaled polynomials are evaluated.
"""
import jax.numpy as jnp
from jax import lax


def _equation(v, rho):
    """Friberg's h and h' multiplied by the same positive exp(-abs(rho))."""
    r, s, t = v
    inverse = 1.0 / jnp.maximum(1.0, jnp.abs(rho))
    z = rho * inverse
    quadratic = z*z - z*inverse + inverse*inverse
    a = inverse * ((z-inverse)*r + inverse*s) / quadratic
    b = inverse * (inverse*r-z*s) / quadratic
    qp = inverse * (2.0*z-inverse) / quadratic
    positive = jnp.exp(jnp.minimum(2.0*rho, 0.0))
    negative = jnp.exp(jnp.minimum(-2.0*rho, 0.0))
    decay = jnp.exp(-jnp.abs(rho))
    value = a*positive - b*negative - t*decay
    derivative = (inverse*(z*r+inverse*s)/quadratic * positive
                  + inverse*(inverse*r-(z-inverse)*s)/quadratic * negative
                  - qp*t*decay - qp*value)
    return value, derivative


def _moreau_pair(v, rho):
    """Feasible orthogonal primal/polar rays, reconstructed without overflow.

    In exact arithmetic, p+d is a point whose cone projection is p. Thus
    ||v-p-d|| bounds ||Pi(v)-p||. A small scaled root value alone would not.
    """
    magnitude = jnp.maximum(1.0, jnp.abs(rho))
    ep = jnp.exp(jnp.minimum(rho, 0.0))
    em = jnp.exp(jnp.minimum(-rho, 0.0))
    primal_ray = jnp.where(rho >= 0,
        jnp.stack([rho*em, em, jnp.ones_like(rho)]),
        jnp.stack([rho/magnitude, 1.0/magnitude, ep/magnitude]))
    polar_ray = jnp.where(rho >= 0,
        jnp.stack([1.0/magnitude, (1.0-rho)/magnitude, -em/magnitude]),
        jnp.stack([ep, (1.0-rho)*ep, -jnp.ones_like(rho)]))
    primal = primal_ray * (jnp.maximum(jnp.dot(v, primal_ray), 0.0)
                          / jnp.dot(primal_ray, primal_ray))
    polar = polar_ray * (jnp.maximum(jnp.dot(v, polar_ray), 0.0)
                        / jnp.dot(polar_ray, polar_ray))
    error = jnp.linalg.norm(v-primal-polar)
    return primal, error


def _curved(v):
    r, s, _t = v
    tolerance = 4.0*jnp.finfo(v.dtype).eps*jnp.linalg.norm(v)
    # These are the positivity bounds for the primal/polar coefficients:
    # (rho-1)*r+s > 0 and r-rho*s > 0. At least one endpoint is finite
    # outside the primal, polar and exact flat-face cases.
    base_low = jnp.where(r > 0, 1.0-s/jnp.where(r > 0, r, 1.0), -jnp.inf)
    base_high = jnp.where(s > 0, r/jnp.where(s > 0, s, 1.0), jnp.inf)
    center = jnp.clip(jnp.asarray(0.0, v.dtype), base_low, base_high)
    low = jnp.maximum(base_low, center-1.0)
    high = jnp.minimum(base_high, center+1.0)
    # Near a flat face the exact root can be less than one float32 ulp
    # from a base endpoint. Evaluating r-rho*s there loses its sign to
    # cancellation; the geometric Moreau pair can already be accurate.
    endpoint_point, endpoint_error = _moreau_pair(v, center)
    for endpoint in (low, high):
        candidate, candidate_error = _moreau_pair(v, endpoint)
        improve = jnp.isfinite(candidate_error) & (candidate_error < endpoint_error)
        endpoint_point = jnp.where(improve, candidate, endpoint_point)
        endpoint_error = jnp.where(improve, candidate_error, endpoint_error)

    def need_bracket(state):
        lo, hi, _radius, iteration = state
        fl, _ = _equation(v, lo)
        fh, _ = _equation(v, hi)
        return (((fl > 0) | (fh < 0)) & (iteration < 64)
                & (endpoint_error > tolerance))

    def expand(state):
        lo, hi, radius, iteration = state
        fl, _ = _equation(v, lo)
        fh, _ = _equation(v, hi)
        radius *= 2.0
        lo = jnp.where(fl > 0, jnp.maximum(base_low, center-radius), lo)
        hi = jnp.where(fh < 0, jnp.minimum(base_high, center+radius), hi)
        return lo, hi, radius, iteration+1

    low, high, _radius, _count = lax.while_loop(
        need_bracket, expand, (low, high, jnp.asarray(1.0, v.dtype), 0))
    fl, _ = _equation(v, low)
    fh, _ = _equation(v, high)
    bracket_valid = (jnp.isfinite(low) & jnp.isfinite(high)
                     & jnp.isfinite(fl) & jnp.isfinite(fh)
                     & (fl <= 0) & (fh >= 0) & (low <= high))
    rho = low + 0.5*(high-low)
    point, error = _moreau_pair(v, rho)
    use_endpoint = jnp.isfinite(endpoint_error) & (
        ~jnp.isfinite(error) | (endpoint_error < error))
    point = jnp.where(use_endpoint, endpoint_point, point)
    error = jnp.where(use_endpoint, endpoint_error, error)

    def continue_search(state):
        lo, hi, _rho, _point, best_error, iteration = state
        midpoint = lo + 0.5*(hi-lo)
        return (bracket_valid & (best_error > tolerance) & (iteration < 96)
                & (midpoint > lo) & (midpoint < hi))

    def refine(state):
        lo, hi, current, best, best_error, iteration = state
        value, derivative = _equation(v, current)
        lo = jnp.where(value <= 0, current, lo)
        hi = jnp.where(value >= 0, current, hi)
        midpoint = lo + 0.5*(hi-lo)
        trial = current - value/jnp.where(derivative > 0, derivative, 1.0)
        # Alternate Newton with bisection: even poor Newton progress cannot
        # leave the bracket large indefinitely. Preserve converged points.
        safe_newton = (jnp.isfinite(trial) & (derivative > 0)
                       & (trial > lo) & (trial < hi) & (iteration % 2 == 0))
        trial = jnp.where(safe_newton, trial, midpoint)
        candidate, candidate_error = _moreau_pair(v, trial)
        improve = jnp.isfinite(candidate_error) & (candidate_error < best_error)
        return (lo, hi, trial, jnp.where(improve, candidate, best),
                jnp.where(improve, candidate_error, best_error), iteration+1)

    _lo, _hi, _rho, point, error, _count = lax.while_loop(
        continue_search, refine, (low, high, rho, point, error, 0))
    # Never substitute a distant flat-face point because an exp(r/s)
    # re-evaluation rejects a constructed boundary point by a few ulps.
    # Failure to bracket without an accurate Moreau pair is a numerical
    # failure, rather than a silently incorrect projection.
    # Adjacent representable rho values and ray reconstruction need a few
    # ulps beyond the iteration target. This remains an input-relative
    # error check; a valid bracket alone is never evidence of accuracy.
    accepted_error = 32.0*jnp.finfo(v.dtype).eps*jnp.linalg.norm(v)
    valid = jnp.isfinite(error) & (error <= accepted_error)
    return jnp.where(valid & jnp.all(jnp.isfinite(point)), point, jnp.nan)


def project_unit(v):
    """Project a finite triple with max(abs(v)) <= 1; output is float32."""
    r, s, t = v
    tiny = jnp.finfo(v.dtype).tiny
    primal = ((s > 0) & (t > 0)
              & (r <= s*(jnp.log(jnp.maximum(t, tiny))
                          - jnp.log(jnp.maximum(s, tiny)))))
    primal |= (s == 0) & (r <= 0) & (t >= 0)
    polar = ((r > 0) & (t < 0)
             & (s <= r*(1.0+jnp.log(jnp.maximum(-t, tiny))
                         - jnp.log(jnp.maximum(r, tiny)))))
    polar |= (r == 0) & (s <= 0) & (t <= 0)
    boundary = jnp.stack([jnp.minimum(r, 0.0), jnp.zeros_like(r), jnp.maximum(t, 0.0)])
    # The flat primal/polar rays are exactly orthogonal. Their reconstruction
    # error is the norm of the positive parts of (r,s), giving a relative
    # projection-error bound for this inexpensive numerical boundary case.
    flat_error = jnp.linalg.norm(jnp.maximum(v[:2], 0.0))
    flat = flat_error <= 2.0*jnp.finfo(v.dtype).eps*jnp.linalg.norm(v)
    simple = primal | polar | flat
    exact = jnp.where(primal, v, jnp.where(polar, jnp.zeros_like(v), boundary))
    # vmap evaluates both cond branches. Feed solved entries an inexpensive
    # exact endpoint so they cannot force a whole batch to expand brackets.
    search_input = jnp.where(simple, jnp.asarray([0.0, 1.0, 1.0], v.dtype), v)
    return lax.cond(simple, lambda _: exact, _curved, search_input)
