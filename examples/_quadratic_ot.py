"""Shared native quadratic-OT family and independent host diagnostics.

This example uses the existing proximal calculus to put the quadratic and
nonnegative domain in one primal block. Marginals are the only dual block.
The fixed primal weight balances steps; it does not change the objective or
termination certificate. Float32 curvature is explicit and shared with POT.
"""
import time
import numpy as np
import jax.numpy as jnp
import ocs
from ocs.ir import Term

EPS = float(np.float32(.1))


def family(n, *, tol=2e-6, max_blocks=2000, primal_weight=30., loop_unroll=4):
    # Repeated OT solves opt into unrolling: faster warm calls, more compile
    # work and workspace. The generic compiler default remains loop_unroll=1.
    x = ocs.Variable(n*n)
    marginal = ocs.Parameter('m')
    cost = ocs.Parameter('c')
    nonnegative_quadratic = ocs.NonNeg().with_linear_quadratic(e=EPS)
    problem = ocs.minimize([
        Term(ocs.Identity(n*n), nonnegative_quadratic),
        ocs.equality(ocs.MarginalOperator(n, n) @ x - marginal),
    ], cost=cost)
    return problem.compile(tol=tol, max_blocks=max_blocks,
                           primal_weight=primal_weight, pdhg_variant='standard',
                           loop_unroll=loop_unroll)


def _dyadic(a):
    """Exactly representable probabilities with exactly equal total masses."""
    z = np.asarray(a, np.float64)
    z = z/z.sum()*2**20
    k = np.floor(z).astype(np.int64)
    k[np.argsort(-(z-k))[:2**20-int(k.sum())]] += 1
    return (k/2**20).astype(np.float32)


def instance(n, rng, *, geometric=False):
    if geometric:
        u = np.sort(rng.uniform(0, 1, n)).astype(np.float32)
        v = np.sort(rng.uniform(0, 1, n)).astype(np.float32)
        c = (u[:, None]-v[None, :])**2
    else:
        c = rng.uniform(0, 1, (n, n)).astype(np.float32)
    p = rng.uniform(.2, 1, n).astype(np.float32); p /= p.sum()
    q = rng.uniform(.2, 1, n).astype(np.float32); q /= q.sum()
    return c, _dyadic(p), _dyadic(q)


def data(c, p, q):
    return {'c': jnp.asarray(c.reshape(-1)), 'm': jnp.asarray(np.r_[p, q])}


def diagnostics(c, p, q, x, y):
    """Float64 checks of the returned point, independent of OCS residuals.

    The primal-dual difference is a numerical diagnostic: this routine does
    not replace the returned point by an exactly feasible transport plan.
    """
    c, p, q = [np.asarray(v, np.float64) for v in (c, p, q)]
    n = len(p); x = np.asarray(x, np.float64).reshape(n, n)
    y = np.asarray(y, np.float64)
    marginal = np.r_[x.sum(1), x.sum(0)]; target = np.r_[p, q]
    linear = c+y[:n, None]+y[None, n:2*n]
    gradient = linear+EPS*x
    objective = float(np.sum(c*x)+.5*EPS*np.sum(x*x))
    dual = float(-target@y[:2*n]-np.sum(np.minimum(linear, 0)**2)/(2*EPS))
    scale = max(np.linalg.norm(c), np.linalg.norm(EPS*x),
                np.linalg.norm(y[:n, None]+y[None, n:2*n]))
    tiny = np.finfo(float).tiny
    return dict(objective=objective, marginal_relative=float(
        np.linalg.norm(marginal-target)/np.linalg.norm(target)),
        min_x=float(x.min()), dual_objective=dual,
        relative_primal_dual_difference=abs(objective-dual)/max(abs(objective),abs(dual),tiny),
        stationarity_relative=float(np.linalg.norm(np.minimum(gradient,0))/max(scale,tiny)),
        complementarity_relative=float(np.linalg.norm(x*gradient)/max(np.linalg.norm(x)*scale,tiny)))


def pot_reference(c, p, q):
    """POT's quadratic L2 semi-dual, not entropic Sinkhorn or unregularized LP."""
    import ot
    c, p, q = [np.asarray(v, np.float64) for v in (c, p, q)]
    t = time.perf_counter()
    x, log = ot.smooth.smooth_ot_semi_dual(
        p, q, c, reg=EPS, reg_type='l2', stopThr=1e-14,
        numItermax=20000, log=True)
    elapsed = time.perf_counter()-t
    value = float(np.sum(c*x)+.5*EPS*np.sum(x*x))
    marginal = np.r_[x.sum(1), x.sum(0)]; target = np.r_[p, q]
    return dict(objective=value, seconds=elapsed,
                success=bool(log['res'].success),
                marginal_relative=float(np.linalg.norm(marginal-target)/np.linalg.norm(target)))
