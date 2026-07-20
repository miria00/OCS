"""The surface modeling API.

    x  = Variable(n)
    K  = FFTMask(...)
    p  = minimize([ l1(x, lam), least_squares(K @ x - Parameter("y")) ])

Everything here lowers to the IR (ir.Term / ir.ProblemFamily).  Per-instance data
enters only through Parameters, which appear in the affine offset l.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..ir import Term, ProblemFamily
from ..operators.common import Identity
from ..prox.library import SumSquares, L1, L2, NonNeg, Box, Equality, Pinball, Logistic, PiecewiseLinearConvex
from ..prox.cones import SecondOrderCone, PSDCone, ExpCone, PowerCone


class Variable:
    def __init__(self, n, name="x"):
        self.n = int(n)
        self.name = name


class Parameter:
    """A named per-instance datum, filled in at solve time from ``data[name]``."""
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class AffineExpr:
    """Represents K x + l.  ``offset_fn(data) -> l``; None means l = 0 (so a
    bare identity term with no offset is recognizable as the primal g)."""
    def __init__(self, op, var, offset_fn=None):
        self.op = op
        self.var = var
        self.offset_fn = offset_fn            # None == zero offset

    @property
    def out_dim(self):
        return self.op.shape[0]

    def _combine(self, other, sign):
        prev = self.offset_fn or (lambda data: 0.0)
        if isinstance(other, Parameter):
            off = lambda data, p=other, s=sign: prev(data) + s * jnp.asarray(
                data[p.name], jnp.float32)
        else:                                             # constant array/scalar
            c = jnp.asarray(other, jnp.float32)
            off = lambda data, c=c, s=sign: prev(data) + s * c
        return AffineExpr(self.op, self.var, off)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __add__(self, other):
        return self._combine(other, +1.0)


def _as_affine(expr, n):
    """Accept a Variable or an AffineExpr; return an AffineExpr."""
    if isinstance(expr, AffineExpr):
        return expr
    if isinstance(expr, Variable):
        return AffineExpr(Identity(n), expr)
    raise TypeError(f"expected Variable or AffineExpr, got {type(expr)}")


# ---- term builders: f(K x + l) --------------------------------------------
def _term(block, expr, n):
    a = _as_affine(expr, n)
    return Term(op=a.op, block=block, offset_fn=a.offset_fn)


def least_squares(expr, n=None):
    """1/2 || K x + l ||_2^2.  Write least_squares(K @ x - y)."""
    return _term(SumSquares(), expr, n or _infer_n(expr))


def sum_squares(expr, n=None):
    return least_squares(expr, n)


def l1(expr, lam=1.0, n=None):
    """lam || K x + l ||_1  (typically l1(x, lam) for a regularizer)."""
    return _term(L1(lam), expr, n or _infer_n(expr))


def l2(expr, lam=1.0, n=None):
    return _term(L2(lam), expr, n or _infer_n(expr))


def nonneg(expr, n=None):
    return _term(NonNeg(), expr, n or _infer_n(expr))


def box(expr, lo=0.0, hi=1.0, n=None):
    return _term(Box(lo, hi), expr, n or _infer_n(expr))


def equality(expr, n=None):
    """Enforce K x + l = 0  (an equality constraint).  Write equality(A @ x - b)."""
    return _term(Equality(), expr, n or _infer_n(expr))


def group_l2(expr, lam=1.0, group_size=2, n=None):
    """lam * sum_g ||(K x + l)_g||_2 over contiguous groups (isotropic TV with a
    2-D gradient operator and group_size=2; row-group lasso otherwise)."""
    from ..prox.library import GroupL2
    return _term(GroupL2(lam, group_size), expr, n or _infer_n(expr))



def pwl(expr, slopes, intercepts, n=None):
    """Separable convex piecewise-linear  sum_i max_j(slopes[j]*(Kx+l)_i + intercepts[j]).
    PWL convex costs / concave-utility epigraphs (e.g. congestion cost on flows)."""
    return _term(PiecewiseLinearConvex(slopes, intercepts), expr, n or _infer_n(expr))


def pinball(expr, tau=0.5, n=None):
    """Quantile / pinball loss sum_i rho_tau((K x + l)_i).  Quantile regression:
    pinball(y - X @ theta, tau).  tau=0.5 is (a scaled) l1 / median regression."""
    return _term(Pinball(tau), expr, n or _infer_n(expr))


quantile = pinball                                    # alias


def logistic(expr, n=None):
    """Logistic loss sum_i log(1 + exp((K x + l)_i)).  Logistic regression with
    labels b in {-1,+1}:  logistic(Diagonal(-b) @ (X @ theta))."""
    return _term(Logistic(), expr, n or _infer_n(expr))


# ---- cone-membership constraints:  (K x + l) in cone ----------------------
def soc(expr, n=None):
    """Second-order cone:  (K x + l) = (s, z) with ||z||_2 <= s (s is entry 0)."""
    return _term(SecondOrderCone(), expr, n or _infer_n(expr))


def psd(expr, k, n=None):
    """PSD cone:  (K x + l) reshaped to a k x k matrix is symmetric PSD."""
    return _term(PSDCone(k), expr, n or _infer_n(expr))


def exp_cone(expr, n=None):
    """Exponential cone:  (K x + l) = (r, s, t) in cl{ s>0, t >= s exp(r/s) }."""
    return _term(ExpCone(), expr, n or _infer_n(expr))


def pow_cone(expr, alpha=0.5, n=None):
    """Power cone:  (K x + l) = (x0, y0, z0) with x0^alpha y0^(1-alpha) >= |z0|."""
    return _term(PowerCone(alpha), expr, n or _infer_n(expr))


def _infer_n(expr):
    if isinstance(expr, Variable):
        return expr.n
    if isinstance(expr, AffineExpr):
        return expr.op.shape[1]
    raise TypeError("cannot infer variable dimension")


def minimize(terms, cost=None):
    """Build a ProblemFamily from a list of terms.

    ``cost`` adds a linear objective term c^T x.  Pass a Parameter (varies per
    instance) or a constant array; None means no linear cost.
    """
    n = None
    for t in terms:
        n = t.op.shape[1]
    cost_fn = None
    if isinstance(cost, Parameter):
        cost_fn = lambda data, p=cost: jnp.asarray(data[p.name], jnp.float32)
    elif cost is not None:
        c = jnp.asarray(cost, jnp.float32)
        cost_fn = lambda data, c=c: c
    return ProblemFamily(n=n, terms=list(terms), cost_fn=cost_fn)
