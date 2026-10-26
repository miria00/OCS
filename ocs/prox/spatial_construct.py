"""Constructing a spatial-full cone metric from OPERATOR PRODUCTS ALONE, float32.

Implements SPEC.md of ``candidates/product_only_metric_20260930``, sha256
``29401a6ab8c30a8801a3d9f151d8d380a5399157757d2ae05ea934e822dbc489``, frozen
before this file.

WHAT THIS CHANGES, AND THE ONE THING IT DOES NOT
------------------------------------------------
The archived construction that produced the measured gain reads its operator
through exactly one object: the cone term's scaled Gram

    G = A T A^T,       A = K restricted to the cone term's rows,

and it consumes ``G`` as ``G[0, 0]`` together with the eigendecomposition of
``G[1:, 1:]``.  Everything after that is a scalar level problem in ``k``
variables and touches no operator at all.  The archive obtains ``G`` by forming
``K.toarray()``, which its own code marks ``REFERENCE ONLY``.

This module obtains the SAME ``G`` from products,

    G e_j = A ( T ( A^T e_j ) ),      j = 1, ..., k,

at a cost of ``k`` adjoint and ``k`` forward applications, and then runs the
archive's selection rule UNCHANGED.  The objective, the box, the condition cap,
the constant ``MU`` and the scalar-weight policy are the archive's, ported and
not redesigned: a different energy minimizer would be a different construction
and would say nothing about the one that won.

WHAT THIS IS, NAMED HONESTLY
----------------------------
A deterministic product-only BLOCK-MOMENT constructor.  It is NOT Gram-free.  It
holds a ``k`` by ``k`` matrix and runs one symmetric eigendecomposition on it, so
it costs ``O(k^2)`` storage and ``O(k^3)`` work in the SPATIAL BLOCK SIZE, not in
the problem dimension.  The metric representation it produces is already
quadratic in ``k``, so this adds no new asymptotic storage class, but it is not a
constructor for a million-dimensional cone and nothing here describes it as one.
A cone above ``MAX_CONSTRUCT_K`` is REFUSED by name, with its cost, and never
silently sketched.

THE SELECTION RULE, PORTED
--------------------------
With ``phi(x, g) = g e^x / 2 - r x / 2 + MU x^2 / 8``, convex in ``x``:

* ``clipped_roots`` minimizes each ``phi`` over the box ``[LO, HI]``, by
  bisection on ``h(x) = g e^x + MU x / 2 - r``, which is increasing;
* ``level_solve`` takes those roots, and when their spread exceeds the condition
  cap ``LCAP`` it bisects the level ``m`` and clips to ``[m, m + LCAP]``;
* ``solve_cone`` for the spatial class reads ``g = (G00, eig(Gzz))``, solves the
  level problem, and returns ``a = exp(x_0)``, ``b = exp(x_1..)``, ``Q`` the
  spatial eigenvectors.

Every constant is the archive's: ``LO = -4``, ``HI = 4``,
``LCAP = log(100)``, ``MU = 1``.

FLOAT32 IS THE PRODUCTION TARGET
--------------------------------
The products, the accumulated moment, the eigendecomposition, the root solves and
the resulting factors are float32.  This module does not ask float32 to reproduce
a float64 trajectory and does not compare eigenvectors.  What is compared, in the
gate, is the metric's ACTION and the construction's objective value.

``T`` is FIXED while the moment is taken, so what this produces is a CONDITIONAL
ROW-METRIC construction and not a solution of the joint row and column problem.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .spatial_soc import SpatialFullSOCMetric

__all__ = ["block_moment", "solve_cone_spatial", "construct_spatial_full",
           "ConstructionRefused", "ProductCount",
           "LO", "HI", "LCAP", "MU", "ROOT_STEPS", "LEVEL_STEPS",
           "MAX_CONSTRUCT_K"]

#: The archive's log-eigenvalue box for H, its condition cap and its constant.
LO, HI = -4.0, 4.0
LCAP = math.log(100.0)
MU = 1.0

#: Bisection counts, the archive's.  Fixed, so the construction is deterministic.
ROOT_STEPS = 120
LEVEL_STEPS = 200

#: The temporary budget for the moment, in float32 ELEMENTS.  The microbatch is
#: chosen so that the adjoint intermediate is at most this many elements, which
#: bounds the construction's peak temporary at O(n q + k^2) and keeps it from
#: materializing A^T.  Two million elements is eight megabytes in float32.
MOMENT_TEMPORARY_ELEMENTS = 2 << 20

#: The supported spatial block size.  Above this the moment's O(k^2) storage and
#: O(k^3) eigendecomposition stop being a sensible price for a metric, and the
#: constructor REFUSES rather than substituting something cheaper without saying
#: so.  Raising it is a measurement, not a default.
MAX_CONSTRUCT_K = 1024


class ConstructionRefused(Exception):
    """Raised when no metric of the declared class can be constructed."""


class ProductCount:
    """Counts what the constructor asked of the operator, so the cost is a fact.

    A constructor is product-only because the only thing it touched was
    ``matvec`` and ``rmatvec``, and the way to say that credibly is to count
    them and to report the count beside the result.
    """

    __slots__ = ("forward", "adjoint", "metric", "vectorized", "probe",
                 "microbatch", "peak_temporary")

    def __init__(self):
        self.forward = 0
        self.adjoint = 0
        self.metric = 0
        self.vectorized = None
        self.probe = 0
        self.microbatch = None
        self.peak_temporary = None

    def record(self):
        return {"forward_applications": int(self.forward),
                "adjoint_applications": int(self.adjoint),
                "primal_metric_applications": int(self.metric),
                "total_operator_applications": int(self.forward + self.adjoint),
                "vectorized": self.vectorized,
                "failed_vectorization_probe": bool(self.probe),
                "microbatch_columns": self.microbatch,
                "peak_temporary_elements": self.peak_temporary}

    def __repr__(self):                                  # pragma: no cover
        return f"ProductCount({self.record()})"


def block_moment(matvec, rmatvec, k, n, tau, *, count=None, dtype=jnp.float32,
                 batch=None):
    """``G = A T A^T`` as a ``k`` by ``k`` matrix, from PRODUCTS ONLY.

    ``matvec`` maps ``R^n -> R^k`` and is the cone term's forward action.
    ``rmatvec`` maps ``R^k -> R^n``.  ``tau`` is the fixed primal scaling, a
    vector of length ``n`` or a scalar.

    Column ``j`` is ``A ( T ( A^T e_j ) )``, so the cost is ``k`` adjoint and
    ``k`` forward applications.  Nothing here reads an entry of ``A`` and nothing
    reconstructs it: the ``n`` by ``k`` adjoint images are consumed one column at
    a time (or one modest batch at a time) and only the ``k`` by ``k`` result is
    retained.

    The moment is symmetrized EXPLICITLY at the end, because ``A T A^T`` is
    symmetric in exact arithmetic and its float32 evaluation is not, and a
    silent asymmetry would be inherited by the eigendecomposition.
    """
    k = int(k)
    if k > MAX_CONSTRUCT_K:
        raise ConstructionRefused(
            f"spatial-full construction: the cone's block size is {k}, above the "
            f"supported {MAX_CONSTRUCT_K}. The block moment costs O(k^2) storage "
            f"and its eigendecomposition O(k^3), so this is refused rather than "
            f"replaced by a cheaper estimate without saying so.")
    tau = jnp.asarray(tau, dtype)
    eye = jnp.eye(k, dtype=dtype)
    # MICROBATCH, and the default is NOT k.  Mapping all k columns at once makes
    # the adjoint's intermediate an (n, k) array, and
    #
    #     [A^T e_1, ..., A^T e_k]  =  A^T,
    #
    # so a full map materializes the operator's transpose as a temporary.  That
    # would still be oracle access, the entries were never read, but it would
    # destroy the storage claim and the memory advantage of holding a k by k
    # metric instead of a k by n operator.  So the batch is chosen to bound the
    # temporary at O(n q + k^2) rather than O(n k + k^2).
    if batch is None:
        batch = int(min(k, max(1, MOMENT_TEMPORARY_ELEMENTS // max(1, int(n)))))
    else:
        batch = max(1, int(batch))

    # vmap over the columns, which is the SAME k adjoint and k forward
    # applications: a vectorized map of matvec is matvec applied to each column,
    # not a different operation and not an entry read.  Column by column in
    # Python costs one dispatch each, which at k = 129 dominated the construction
    # and would have made the constructor unable to pay for itself.
    #
    # A caller's matvec need not be traceable, though.  One that converts to
    # numpy internally cannot be vmapped, and refusing such an operator would be
    # refusing a legitimate one, so the loop is kept as a FALLBACK and the record
    # says which path ran, because the two differ by an order of magnitude in
    # cost and a timing that does not say which one it measured is not a timing.
    def column(e):
        return matvec(rmatvec(e) * tau)

    columns, mapped = [], True
    try:
        for start in range(0, k, batch):
            stop = min(k, start + batch)
            columns.append(jax.vmap(column, in_axes=1, out_axes=1)(
                eye[:, start:stop]))
    except Exception:
        # The attempt enters the callable before it fails, so the fallback costs
        # AT MOST one extra application of each, and where it fails decides
        # which: the column function applies the adjoint first, so a callable
        # that cannot be traced is entered once on the adjoint and never on the
        # forward. The flag records that an attempt was made and paid for.
        columns, mapped = [], False
        for j in range(k):
            columns.append(jnp.reshape(column(eye[:, j]), (-1, 1)))
    if count is not None:
        count.adjoint += k
        count.metric += k
        count.forward += k
        count.vectorized = mapped
        count.probe = 0 if mapped else 1
        count.microbatch = batch if mapped else 1
        # the adjoint intermediate plus the retained moment, in elements
        count.peak_temporary = int(n) * (batch if mapped else 1) + k * k
    G = jnp.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
    return 0.5 * (G + G.T)


def _phi(x, g, r):
    """The archive's per-coordinate energy, ``g e^x / 2 - r x / 2 + MU x^2 / 8``."""
    return 0.5 * g * jnp.exp(x) - 0.5 * r * x + MU * x * x / 8.0


def _clipped_roots(g, r):
    """``argmin`` of ``phi`` over ``[LO, HI]``, by bisection on an increasing h.

    ``h(x) = g e^x + MU x / 2 - r`` is ``2 phi'(x)`` and is increasing, so a
    fixed count of bisections is exact enough and is deterministic, which is what
    the rest of this project asks of a root solve.
    """
    dtype = g.dtype

    def h(x):
        return g * jnp.exp(x) + 0.5 * MU * x - r

    def body(_i, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        positive = h(mid) > 0
        return (jnp.where(positive, lo, mid), jnp.where(positive, mid, hi))

    lo = jnp.full(g.shape, jnp.asarray(LO, dtype))
    hi = jnp.full(g.shape, jnp.asarray(HI, dtype))
    lo, hi = jax.lax.fori_loop(0, ROOT_STEPS, body, (lo, hi))
    x = 0.5 * (lo + hi)
    x = jnp.where(h(jnp.full(g.shape, jnp.asarray(LO, dtype))) >= 0, LO, x)
    x = jnp.where(h(jnp.full(g.shape, jnp.asarray(HI, dtype))) <= 0, HI, x)
    return x


def _level_solve(g, weights, r):
    """The archive's level problem, with the condition cap enforced by clipping.

    ``min sum_i w_i phi_{g_i}(x_i)`` over the box, subject to
    ``max x - min x <= LCAP``.  Unconstrained the answer is the per-coordinate
    roots.  When their spread exceeds the cap the optimal level is found by
    bisecting the derivative of the partial minimum, exactly as the archive does.
    """
    dtype = g.dtype
    roots = _clipped_roots(g, r)
    spread = jnp.max(roots) - jnp.min(roots)

    def dphi(x, gg):
        return 0.5 * gg * jnp.exp(x) - 0.5 * r + MU * x / 4.0

    def dF(m):
        low = roots < m
        high = roots > m + LCAP
        return (jnp.sum(jnp.where(low, weights * dphi(m, g), 0.0))
                + jnp.sum(jnp.where(high, weights * dphi(m + LCAP, g), 0.0)))

    def body(_i, bounds):
        a, b = bounds
        mid = 0.5 * (a + b)
        return jax.lax.cond(dF(mid) > 0, lambda: (a, mid), lambda: (mid, b))

    a0 = jnp.min(roots)
    b0 = jnp.max(roots) - jnp.asarray(LCAP, dtype)
    a, b = jax.lax.fori_loop(0, LEVEL_STEPS, body, (a0, b0))
    level = 0.5 * (a + b)
    capped = jnp.clip(roots, level, level + LCAP)
    x = jnp.where(spread <= LCAP, roots, capped)
    return x, jnp.sum(weights * _phi(x, g, r))


def solve_cone_spatial(G, r):
    """The archive's spatial-class selection rule, on a block moment.

    Returns ``(a, b, Q, energy, g)``.  ``g`` is the vector of energies the level
    problem was solved on, ``(G00, eig(Gzz))`` floored at zero, which is the
    archive's own guard against a moment that float arithmetic made slightly
    indefinite.  The floor is EXPLICIT and the number of coordinates it moved is
    reported by the caller, because clipping a materially indefinite moment
    silently is how a construction stops describing its problem.
    """
    G = jnp.asarray(G)
    k = G.shape[0]
    spatial = 0.5 * (G[1:, 1:] + G[1:, 1:].T)
    evz, Vz = jnp.linalg.eigh(spatial)
    raw = jnp.concatenate([G[0:1, 0], evz])
    g = jnp.maximum(raw, 0.0)
    x, energy = _level_solve(g, jnp.ones((k,), G.dtype), jnp.asarray(r, G.dtype))
    return jnp.exp(x[0]), jnp.exp(x[1:]), Vz, energy, g, raw


def construct_spatial_full(matvec, rmatvec, k, n, tau, r, *, layout=None,
                           dtype=jnp.float32, batch=None, count=None):
    """``(a, b, Q)`` for one second-order cone, from products alone.

    ``r`` is the archive's shape scalar ``sqrt(n / m)``, which is read from the
    problem's DIMENSIONS and not from the operator.

    Returns ``(metric, record)``.  ``record`` carries the product counts, the
    construction energy, the floor's activity and the resulting condition number,
    so the cost and the domain of the construction are facts beside the metric
    rather than claims about it.
    """
    count = ProductCount() if count is None else count
    G = block_moment(matvec, rmatvec, k, n, tau, count=count, dtype=dtype,
                     batch=batch)
    if not bool(jnp.all(jnp.isfinite(G))):
        raise ConstructionRefused(
            "spatial-full construction: the block moment is not finite, so the "
            "operator's products are not usable here. Nothing is repaired.")
    a, b, Q, energy, g, raw = solve_cone_spatial(G, r)
    floored = int(jnp.sum(raw < 0))
    worst = float(jnp.min(raw))
    metric = SpatialFullSOCMetric(a, b, Q, layout=layout)
    record = {
        "products": count.record(),
        "block_size_k": int(k),
        "construction_energy": float(energy),
        "moment_floor_active": floored,
        "moment_min_eigenvalue": worst,
        "metric_condition": float(jnp.max(jnp.concatenate([a[None], b]))
                                  / jnp.max(jnp.asarray(
                                      [float(jnp.min(jnp.concatenate([a[None], b]))),
                                       1e-30]))),
        "conditional": ("T was held FIXED while the moment was taken, so this is "
                        "a conditional ROW-metric construction and not a solution "
                        "of the joint row and column problem"),
        "dtype": str(jnp.asarray(G).dtype),
    }
    return metric, record
