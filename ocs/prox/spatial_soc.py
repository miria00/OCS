"""The weighted second-order-cone projection for a SPATIAL-FULL metric.

Implements SPEC.md of ``candidates/spatial_full_soc_20260930``, sha256
``31405f48f9e1150c593d7cd6a36a768fb3aab6c3881ccddfd695ed9ebc090c48``, frozen
before this file.  Application only: nothing here CONSTRUCTS a metric, nothing
here changes a released block's scaling class, and nothing here is reachable
from the public supplied path yet.

The class
---------

    M = diag(a, Q diag(b) Q^T),   a > 0,   b_j > 0,   Q^T Q = I,

a full spatial block with NO time-space cross terms.  It is the class the
archived n = 2048 comparison used.  Within the block-diagonal family diag(a, B)
the only cone automorphisms are B = a I, so a non-scalar spatial block is
outside the automorphism class and ``ConeAutomorphism`` cannot express it.  The
nontrivial Lorentz automorphisms, the quadratic representations P(w) with wbar
nonzero, have time-space cross terms and are not in this class either: the two
classes meet only at the scalar members.

The primitive
-------------

    P_M(v) = argmin over z in Q of (1/2) (z - v)^T M (z - v),

the manuscript's convention, in which the proximal superscript is the quadratic
METRIC and not a step size.  Three branches, and the polar test is on ``M v``:

    v in Q          ->  z = v
    M v in -Q       ->  z = 0
    otherwise       ->  the reduced root below

With ``ytilde = Q^T y`` for ``v = (t, y)`` and ``bw_j = b_j ytilde_j``,
stationarity of the reduced problem gives

    u_j(theta) = bw_j / (b_j + theta),      r(theta) = a t / (a - theta),

with ``theta = lambda / ||u||`` the scaled cone multiplier and the active
condition ``||u(theta)|| = r(theta)``.  Feasibility of ``r > 0`` puts theta in
(0, a) when t > 0, in (a, infinity) when t < 0, and forces theta = a when
t = 0.

The returned radius is ``||u(theta)||`` rather than ``r(theta)``, which is the
same number at the exact root and is what makes the inexact point honest.  With
``z = (||u||, Q u)``:

* ``z`` is ON the cone boundary by construction, so the primal defect is only
  the roundoff of the rotation;
* ``b_j (ytilde_j - u_j) = theta u_j`` identically, so the residual is exactly
  ``n = M (v - z) = (a(t - ||u||), theta Q u)``, and ``n`` lies in ``-Q``
  exactly when ``G(theta) >= 0``;
* ``z^T n = -||u|| G(theta)``.

So the bisection returns the bracket end where ``G >= 0``, not the midpoint.
Then ``z`` is feasible and ``n`` is polar-feasible EXACTLY, and the whole root
error appears in one reported number, ``eps = |z^T n| = ||u|| G``.  That number
is a certificate: for the exact projection ``z*``,

    (z - z*)^T M (z - z*)
        = (z - z*)^T (-n + n*)
        = eps + z^T n* + z*^T n  <=  eps,

because ``z^T n* <= 0`` and ``z*^T n <= 0`` by self-duality, so

    ||z - z*||_M  <=  sqrt(eps).

An inexact root therefore costs a MEASURED distance and not an unknown one.

The root equation, in the form solved here
------------------------------------------
The specification writes the root equation squared.  This module bisects the
equivalent

    G(theta) = (a - theta) ||u(theta)|| - a t,

which has the SAME root and is strictly decreasing on both brackets, so one
uniform bisection serves both signs of t:

* t > 0, theta in (0, a): each ``(a - theta) / (b_j + theta)`` is positive and
  decreasing, so ``(a - theta)||u||`` falls from ``a ||y||`` to 0.  Hence
  ``G(0) = a(||y|| - t) > 0`` off the cone and ``G(a) = -a t < 0``.
* t < 0, theta in (a, infinity): with ``phi = theta - a``, each
  ``phi / (b_j + a + phi)`` is increasing, so ``phi ||u(a + phi)||`` rises from
  0 to ``||B y|| = ||bw||``, and ``G = a|t| - phi ||u||`` falls from ``a |t|``
  to ``a|t| - ||bw||``, which is negative exactly when the cone is NOT in the
  polar branch.

So on the root branch a sign change exists by the branch test itself, never by
assumption.

Both ends of both brackets are DERIVED, which is what makes the solve accurate
near the branch boundaries.  Bisecting ``(0, a)`` would give absolute
resolution, and the root is tiny when ``v`` is nearly feasible, so the relative
error would blow up exactly where the metric matters least but the arithmetic
matters most.  Squeezing ``||u||`` between its ``bmin`` and ``bmax`` surrogates,

    ||ytilde|| bmin / (bmin + theta)  <=  ||u(theta)||
                                      <=  ||ytilde|| bmax / (bmax + theta),

and solving each surrogate equation exactly gives, for t > 0,

    theta_lo = a bmin (||ytilde|| - t) / (bmin ||ytilde|| + a t)   (G >= 0 below)
    theta_up = a bmax (||ytilde|| - t) / (bmax ||ytilde|| + a t)   (G <= 0 above)

and, for t < 0 in the variable ``phi = theta - a`` with ``cmin = a + bmin``,
``cmax = a + bmax`` and ``kappa = a |t| / ||bw|| < 1``,

    phi_lo = cmin kappa / (1 - kappa),      phi_up = cmax kappa / (1 - kappa).

Either bracket therefore has a WIDTH PROPORTIONAL TO THE ROOT, with ratio at
most the spread of ``b``, so geometric bisection (``mid = lo sqrt(up/lo)``)
resolves it to relative unit roundoff in a fixed count of steps however close
``v`` is to the cone or to the polar cone.

The variable is native to its branch, which removes the one remaining
cancellation: for t < 0 the factor ``a - theta`` is ``-phi`` EXACTLY, and for
t > 0 the denominator ``b_j + theta`` is exact.  In the leftover corner, t > 0
with theta near a, ``a - theta`` does cancel, but the returned point depends on
theta only through ``b_j + theta >= a``, so its error stays at unit roundoff.

Numerical domain
----------------
A fixed count of bisection steps on a maintained bracket, which is
deterministic, jittable and vectorized over cones.  Each cone carries its own
status: one cone's success may not conceal another's failure.  An unestablished
bracket, a non-finite point or a metric that is not positive is a FAILED status
and a NaN point, never an approximate iterate reported as accepted.  The
requested metric is never modified to make the root easier, and the accepted
point is checked in the ORIGINAL coordinates by ``optimality_defects``, outside
the rotated calculation.

Two cases are benign rather than fatal, and both are recorded rather than
hidden.  The branch tests use ``||y||`` and ``||b ytilde||`` in the ORIGINAL
coordinates while the root uses ``||ytilde||``, so for a ``v`` within rotation
roundoff of the cone or of the polar cone the root can fall just outside its
bracket.  The bracket ends are floored at ``a 2^-60`` in their own variable,
which is where the root would be, so the point returned is the boundary answer
to within that same roundoff.  The polar condition then holds only to roundoff
instead of exactly and the reported defect shows it.  Neither is flagged FAILED,
because the point is right.  A bracket whose UPPER end still has ``G > 0`` is a
different matter: the root is then outside the bracket on the side that makes
``u`` too large, so that cone is FAILED.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "SpatialFullSOCMetric",
    "project",
    "project_polar_step",
    "spatial_full_moreau",
    "rows",
    "unrows",
    "normalize_spatial_full_metric",
    "metric_amplitude",
    "apply_spatial_full",
    "describe_spatial_full",
    "project_polar_step_moreau",
    "optimality_defects",
    "distance_estimate",
    "relative_distance_estimate",
    "BISECTION_STEPS",
    "EXPANSION_STEPS",
    "STATUS_INSIDE",
    "STATUS_POLAR",
    "STATUS_ROOT",
    "STATUS_APEX",
    "STATUS_FAILED",
    "STATUS_POLAR_LIMIT",
    "STATUS_NAMES",
    "STATUS_ACCEPTED",
]

#: Bisection steps on the maintained bracket.  Sixty halvings drive a float64
#: bracket to about 1e-18 of its width and a float32 one far below its unit
#: roundoff, where the midpoint stops moving.  The count is FIXED so the kernel
#: is deterministic and jittable, and so an exhausted budget is a status rather
#: than a longer run.
BISECTION_STEPS = 60

#: Doublings allowed when establishing the t < 0 upper bracket.  The starting
#: point is the derived bound above, so this is a guard against its float
#: evaluation, not a search.
EXPANSION_STEPS = 24

STATUS_INSIDE = 0
STATUS_POLAR = 1
STATUS_ROOT = 2
STATUS_APEX = 3
STATUS_FAILED = 4
#: The zero answer taken because the root is past the resolvable range, with the
#: error bound stated where it is decided.  It has its own value so a caller can
#: count it: a fold that cannot be seen is a fold that will be forgotten.
STATUS_POLAR_LIMIT = 5

STATUS_NAMES = {
    STATUS_INSIDE: "inside",
    STATUS_POLAR: "polar",
    STATUS_ROOT: "root",
    STATUS_APEX: "apex",
    STATUS_FAILED: "failed",
    STATUS_POLAR_LIMIT: "polar_limit",
}

#: The statuses that carry an answer.  Only FAILED does not.
STATUS_ACCEPTED = (STATUS_INSIDE, STATUS_POLAR, STATUS_ROOT, STATUS_APEX,
                   STATUS_POLAR_LIMIT)


def _rotate(Q, w):
    """``Q w`` per cone, at the backend's HIGHEST precision.

    The default float32 matmul precision on an NVIDIA GPU is TF32, which carries
    a 19-bit mantissa, about 1e-3 relative.  A rotation computed that way is not
    a rotation: it breaks the orthogonality this class rests on, it moves the
    returned radius off the cone boundary, and it makes a near-boundary cone's
    branch decision depend on which kernel the autotuner happened to pick.  That
    was observed on KEVIN, as a cone that came back FAILED in one run of the gate
    and accepted in the next on identical data.  ``ocs.prox.cones.PSDCone``
    already pins its float32 congruences for the same reason.
    """
    return jnp.einsum("cij,cj->ci", Q, w, precision=jax.lax.Precision.HIGHEST)


def _rotate_t(Q, w):
    """``Q^T w`` per cone, at the same precision and for the same reason."""
    return jnp.einsum("cji,cj->ci", Q, w, precision=jax.lax.Precision.HIGHEST)


def normalize_spatial_full_metric(declared, term_index, block, size):
    """Validate a supplied ``SpatialFullSOCMetric`` for one term.

    Returns ``(k, ranges)`` in the same shape ``cone_metric.normalize_cone_metric``
    returns: the whole term is ONE plumbing block, so ``k`` is the term's row
    count and ``ranges`` is the single entry ``(0, 1, carrier)``.  How many cones
    the term stacks is read from the BLOCK, never from a range bound.

    Accepted on a ``SecondOrderCone`` or ``SOCBatch`` term only.  A ``PSDCone``
    is refused by name: this class is the Lorentz cone's, and there is no
    spatial-full metric for a matrix cone.  Nothing is converted, flattened or
    dropped to make a declaration fit, because the whole point of this class is
    that it is NOT the automorphism class.
    """
    from . import cone_metric, cones

    struct = cone_metric.symmetric_cone_structure(block, size)
    if struct is None or struct[0] != "soc":
        raise ValueError(
            f"block_metrics[{term_index}]: a SpatialFullSOCMetric is a metric for a "
            f"SECOND-ORDER cone block of exact type SecondOrderCone or SOCBatch. "
            f"{type(block).__name__} over {size} rows is not one. The class is "
            "diag(a, Q diag(b) Q^T) on the Lorentz cone, and it is not converted "
            "to a ConeAutomorphism to fit another block: the two classes meet "
            "only at their scalar members.")
    _algebra, k, cones_count = struct
    if declared.k != k:
        raise ValueError(
            f"block_metrics[{term_index}]: SpatialFullSOCMetric is for k = "
            f"{declared.k} and {type(block).__name__} over {size} rows has "
            f"k = {k}")
    if not declared.shared and declared.cones != cones_count:
        raise ValueError(
            f"block_metrics[{term_index}]: SpatialFullSOCMetric carries "
            f"{declared.cones} per-cone factors and the term stacks "
            f"{cones_count} cones. A shared metric is declared by passing ONE "
            "(a, b, Q), never by relying on a broadcast.")
    for name, value in (("a", declared.a), ("b", declared.b)):
        arr = np.asarray(value, np.float64)
        if not np.all(np.isfinite(arr)) or not np.all(arr > 0):
            raise ValueError(
                f"block_metrics[{term_index}]: every entry of {name} must be "
                f"finite and strictly positive, and it is not. The factors are "
                "used as given: they are not clipped, floored or repaired.")
    Q = np.asarray(declared.Q, np.float64)
    if not np.all(np.isfinite(Q)):
        raise ValueError(
            f"block_metrics[{term_index}]: Q carries a non-finite entry")
    # A shared triple on a term with several cones means ONE metric for every
    # cone.  That is legitimate and is what the caller asked for, and the carrier
    # records it in ``shared`` so a record can say which was declared.  Nothing
    # is broadcast to make a per-cone declaration out of a shared one.
    del cones                                      # only the type check wanted it
    carrier = SpatialFullSOCMetric(declared.a, declared.b, declared.Q,
                                   layout=cone_metric.cone_layout(block))
    return int(size), ((0, 1, carrier),)


def apply_spatial_full(carrier, u, power=1):
    """``M^power u`` on the term's flat vector, power 1 or 0.5.

    ``M^{1/2} = diag(sqrt(a), Q diag(sqrt(b)) Q^T)`` is in the SAME class
    exactly, because ``Q`` is orthogonal, so the half power is a closed form and
    not an approximation.  Any other power is refused rather than approximated.
    """
    if power == 1:
        factors = carrier
    elif power == 0.5:
        factors = SpatialFullSOCMetric(jnp.sqrt(carrier.a), jnp.sqrt(carrier.b),
                                       carrier.Q, layout=carrier.layout)
    else:
        raise ValueError(f"spatial-full metric: power must be 1 or 0.5, not {power!r}")
    return unrows(carrier.layout, factors.apply(rows(carrier.layout, u)))


def describe_spatial_full(carrier):
    """The record entry for one supplied carrier, host floats only."""
    low, high = metric_amplitude(carrier)
    b = np.asarray(carrier.b, np.float64)
    return {"class": "spatial_full_soc", "k": int(carrier.k),
            "cones": 1 if carrier.shared else int(carrier.cones),
            "shared": bool(carrier.shared),
            "eigenvalue_min": low, "eigenvalue_max": high,
            "condition": (high / low) if low > 0 else float("inf"),
            "spatial_spread": float(b.max() / b.min()) if b.min() > 0
                              else float("inf"),
            "spatial_is_scalar": bool(np.all(b == b.reshape(-1)[0])),
            "normalization": ("the CALLER's: a supplied metric carries no "
                              "certified ceiling, and the given steps must be "
                              "normalized against the metric actually supplied")}


def metric_amplitude(declared):
    """``(min eigenvalue, max eigenvalue)`` of the supplied metric, exactly.

    The spatial block's eigenvalues ARE ``b``, so the extremes of ``M`` are
    ``min(a, min_j b_j)`` and ``max(a, max_j b_j)`` with no estimate involved.
    A caller normalizing its own steps for ``equilibrate=('given', ...)`` needs
    these, and a supplied metric carries NO certified ceiling: the normalization
    is the caller's, which the record states rather than implies.
    """
    a = np.asarray(declared.a, np.float64)
    b = np.asarray(declared.b, np.float64)
    return (float(min(a.min(), b.min())), float(max(a.max(), b.max())))


def rows(layout, v, xp=jnp):
    """The term's flat dual vector as ``(cones, k)``, in the compiled layout.

    The same tuple convention ``cone_metric.rows`` uses, because these are the
    same compiled terms: ``("socbatch", q, d)`` for the CVXPY-stacked batch,
    anything else for one cone.  Guessing the layout from a caller's declaration
    instead of reading the term's own would apply the right metric to the wrong
    cone, which is a defect no accuracy check would catch.
    """
    if layout is not None and layout[0] == "socbatch":
        q, d = layout[1], layout[2]
        z = xp.reshape(v[q:], (d, q), order="F").T
        return xp.concatenate([v[:q, None], z], axis=1)
    return v[None, :]


def unrows(layout, R, xp=jnp):
    """The inverse of :func:`rows`."""
    if layout is not None and layout[0] == "socbatch":
        return xp.concatenate([R[:, 0], xp.reshape(R[:, 1:].T, (-1,), order="F")])
    return R[0]


def spatial_full_moreau(block, v, s, ranges):
    """The dual update of one cone term under a supplied spatial-full metric.

    ``v - M P_M(M^{-1} v)`` per cone, which is
    ``prox_{f^*}^{Sigma^{-1}}(v)`` for ``Sigma = c M`` with ``c`` the term's one
    dual step.  THE STEP CANCELS, for two reasons that both have to hold and both
    do: ``P_{cM} = P_M``, because the argmin does not move when the metric is
    scaled by a positive number, and ``P_M`` is positively homogeneous of degree
    one, because ``Q`` is a cone.  So

        v - cM P_{cM}((cM)^{-1} v) = v - cM (1/c) P_M(M^{-1} v)
                                   = v - M P_M(M^{-1} v),

    exactly as ``cone_metric.automorphism_moreau`` reads no step, and for the
    same two reasons.  The step is therefore used ONLY as a gate: one that is not
    finite, positive and uniform over the term returns NaN, which is the backstop
    behind the host's own eager refusal.

    Signature-compatible with ``automorphism_moreau`` so the dispatch in
    ``ocs.prox.groups`` is a third branch and not a new mechanism.
    """
    del block                          # a cone indicator carries no parameters
    carrier = ranges[0][2]
    s = jnp.broadcast_to(jnp.asarray(s, v.dtype), v.shape)
    valid = jnp.all(jnp.isfinite(s) & (s > 0)) & jnp.all(s == s[0])
    R = rows(carrier.layout, v)
    Y = R - carrier.apply(project(_inverse_apply(carrier, R), carrier)[0])
    return jnp.where(valid, unrows(carrier.layout, Y), jnp.nan)


def _inverse_apply(carrier, R):
    """``M^{-1} R`` without building a second carrier object per call."""
    return carrier.inverse().apply(R)


class SpatialFullSOCMetric:
    """The represented factors of one term's spatial-full metric.

    Per cone: ``a`` of shape ``(cones,)``, ``b`` of shape ``(cones, k-1)``,
    ``Q`` of shape ``(cones, k-1, k-1)``.  Shared: ``b`` of shape ``(k-1,)``,
    ``Q`` of shape ``(k-1, k-1)`` and one ``a``, meaning ONE metric for every
    cone of the term.  Any other layout is refused here rather than broadcast
    into something the caller did not ask for.

    The factors are held as given.  They are not reorthogonalized, rescaled or
    reordered, because a metric the caller supplied is a checked object and
    silently repairing it would hide the defect the checks exist to find.
    """

    __slots__ = ("a", "b", "Q", "cones", "k", "shared", "layout")

    def __init__(self, a, b, Q, cones=None, layout=None):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        Q = jnp.asarray(Q)
        if b.ndim not in (1, 2) or Q.ndim not in (2, 3):
            raise ValueError(
                "spatial-full metric: b must be (k-1,) or (cones, k-1) and Q "
                f"must be (k-1, k-1) or (cones, k-1, k-1), got b {b.shape} and "
                f"Q {Q.shape}")
        shared = (b.ndim == 1)
        if shared != (Q.ndim == 2):
            raise ValueError(
                "spatial-full metric: b and Q must both be shared or both be "
                f"per cone, got b {b.shape} and Q {Q.shape}")
        d = b.shape[-1]
        if d < 1:
            raise ValueError("spatial-full metric: the spatial part is empty")
        if Q.shape[-1] != d or Q.shape[-2] != d:
            raise ValueError(
                f"spatial-full metric: Q must be {d} by {d} to match b, got "
                f"{Q.shape}")
        if shared:
            if a.ndim > 1 or (a.ndim == 1 and a.shape[0] != 1):
                raise ValueError(
                    "spatial-full metric: a shared metric takes one a, got "
                    f"{a.shape}")
        else:
            if a.ndim == 0:
                a = jnp.broadcast_to(a, (b.shape[0],))
            elif a.shape != (b.shape[0],):
                raise ValueError(
                    f"spatial-full metric: a must be ({b.shape[0]},) to match "
                    f"b, got {a.shape}")
            if Q.shape[0] != b.shape[0]:
                raise ValueError(
                    f"spatial-full metric: Q has {Q.shape[0]} cones and b has "
                    f"{b.shape[0]}")
        if cones is not None and not shared and int(cones) != int(b.shape[0]):
            raise ValueError(
                f"spatial-full metric: the factors carry {b.shape[0]} cones, "
                f"not {int(cones)}")
        self.a, self.b, self.Q = a, b, Q
        self.shared = bool(shared)
        self.cones = None if shared else int(b.shape[0])
        self.k = int(d + 1)
        # The COMPILED TERM's row layout, the same tuple convention
        # ``ocs.prox.cone_metric`` uses.  None means one cone.  It is the term's
        # own, never inferred: see :func:`rows`.
        self.layout = None if layout is None else tuple(layout)

    def __repr__(self):                                  # pragma: no cover
        where = "shared" if self.shared else f"{self.cones} cones"
        return f"SpatialFullSOCMetric(k={self.k}, {where}, dtype={self.b.dtype})"

    def describe(self):
        """The record entry, in the shape every other carrier's ``describe`` uses."""
        return describe_spatial_full(self)

    def factors_for(self, cones):
        """``(a, b, Q)`` broadcast to ``cones`` rows of the represented values."""
        if not self.shared and int(cones) != self.cones:
            raise ValueError(
                f"spatial-full metric carries {self.cones} cones, not {cones}")
        if self.shared:
            a = jnp.broadcast_to(jnp.reshape(self.a, (-1,))[0], (cones,))
            b = jnp.broadcast_to(self.b, (cones, self.k - 1))
            Q = jnp.broadcast_to(self.Q, (cones, self.k - 1, self.k - 1))
        else:
            a, b, Q = self.a, self.b, self.Q
        return a, b, Q

    def inverse(self):
        """``M^{-1}``, which is in the same class: the factors invert entrywise.

        ``diag(a, Q diag(b) Q^T)^{-1} = diag(1/a, Q diag(1/b) Q^T)`` because Q
        is orthogonal.  This is what lets one Moreau form be production and the
        other the test.
        """
        return SpatialFullSOCMetric(1.0 / self.a, 1.0 / self.b, self.Q,
                                    layout=self.layout)

    def apply(self, v):
        """``M v`` from the factors, without forming ``M``."""
        single = (jnp.ndim(v) == 1)
        V = jnp.reshape(v, (1, -1)) if single else v
        a, b, Q = self.factors_for(V.shape[0])
        y = V[:, 1:]
        spatial = _rotate(Q, b * _rotate_t(Q, y))
        out = jnp.concatenate([(a * V[:, 0])[:, None], spatial], axis=1)
        return out[0] if single else out

    def dense(self):
        """The represented matrices, for tests and records only."""
        a, b, Q = self.factors_for(1 if self.shared else self.cones)
        B = jnp.einsum("cij,cj,ckj->cik", Q, b, Q,
                       precision=jax.lax.Precision.HIGHEST)
        k = self.k
        M = jnp.zeros((B.shape[0], k, k), B.dtype)
        M = M.at[:, 0, 0].set(a)
        M = M.at[:, 1:, 1:].set(B)
        return M


def _state(a, t, b, bw, x, neg):
    """``(G, u, theta)`` at the bracket variable ``x``, native to each branch.

    ``x`` is ``theta`` where ``t > 0`` and ``phi = theta - a`` where ``t < 0``,
    so the factor ``a - theta`` is exact on the branch where it would otherwise
    cancel.
    """
    theta = jnp.where(neg, a + x, x)
    a_minus = jnp.where(neg, -x, a - x)
    u = bw / (b + theta[:, None])
    g = a_minus * jnp.sqrt(jnp.sum(u * u, axis=1)) - a * t
    return g, u, theta


def _residual(a, t, b, bw, x, neg):
    return _state(a, t, b, bw, x, neg)[0]


def _brackets(a, t, b, bw, norm_yt, pos, neg, root):
    """The derived bracket in each branch's own variable, then bounded repair.

    Returns ``(lo, up, established)``.  ``lo`` always keeps ``G >= 0`` and ``up``
    keeps ``G <= 0``, up to the two documented roundoff corners.  A cone whose
    upper end still has ``G > 0`` after ``EXPANSION_STEPS`` doublings is not
    established and is FAILED: its root lies outside the bracket on the side
    that makes ``u`` too large.
    """
    bmin = jnp.min(b, axis=1)
    bmax = jnp.max(b, axis=1)
    floor = a * 2.0 ** -60

    # t > 0, in theta.  Both ends solve a surrogate equation exactly.
    gap = jnp.maximum(norm_yt - t, 0.0)
    den_lo = bmin * norm_yt + a * t
    den_up = bmax * norm_yt + a * t
    th_lo = 0.5 * a * bmin * gap / jnp.where(den_lo > 0, den_lo, 1.0)
    th_up = jnp.minimum(a, 2.0 * a * bmax * gap
                        / jnp.where(den_up > 0, den_up, 1.0))

    # t < 0, in phi = theta - a.
    norm_bw = jnp.sqrt(jnp.sum(bw * bw, axis=1))
    kappa = jnp.abs(a * t) / jnp.where(norm_bw > 0, norm_bw, 1.0)
    room = 1.0 - kappa
    open_room = jnp.where(room > 0, room, 1.0)
    ph_lo = jnp.where(room > 0, 0.5 * (a + bmin) * kappa / open_room, jnp.inf)
    ph_up = jnp.where(room > 0, 2.0 * (a + bmax) * kappa / open_room, jnp.inf)

    # A cone decided by a branch test, and the apex, both sit at theta = a with
    # an idle bracket, so one loop serves every cone and nothing divides by zero.
    lo = jnp.where(pos, th_lo, jnp.where(neg, ph_lo, a))
    up = jnp.where(pos, th_up, jnp.where(neg, ph_up, a))
    lo = jnp.where(root & (pos | neg), jnp.maximum(lo, floor), lo)
    up = jnp.maximum(up, lo)

    def repair(_i, bounds):
        lo_, up_ = bounds
        low = jnp.logical_and(root, _residual(a, t, b, bw, lo_, neg) < 0)
        high = jnp.logical_and(root, _residual(a, t, b, bw, up_, neg) > 0)
        lo_ = jnp.where(low, jnp.maximum(0.5 * lo_, floor), lo_)
        # the t > 0 bracket may never leave (0, a], and G(a) = -a t < 0 there
        up_ = jnp.where(high, jnp.where(pos, jnp.minimum(2.0 * up_, a),
                                        2.0 * up_), up_)
        return (lo_, up_)

    lo, up = jax.lax.fori_loop(0, EXPANSION_STEPS, repair, (lo, up))
    # For t > 0 the end theta = a ALWAYS has G(a) = -a t < 0, so taking it
    # settles any cone the derived end and the doublings did not, and the t > 0
    # bracket is then established unconditionally.  Without this the decision
    # depends on roundoff in the derived end, and a backend that reassociates a
    # dot product between processes can then refuse a cone it accepted before.
    up = jnp.where(jnp.logical_and(pos, _residual(a, t, b, bw, up, neg) > 0),
                   a, up)
    # Only a ROOT cone has a bracket to establish.  A cone decided by a branch
    # test sits at theta = a, where G = -a t, which is positive for every polar
    # cone: reading that as an unestablished bracket would fail the whole polar
    # branch.
    established = jnp.logical_or(~root, jnp.logical_and(
        jnp.logical_and(jnp.isfinite(lo), jnp.isfinite(up)),
        _residual(a, t, b, bw, up, neg) <= 0))
    return lo, up, established


def project(v, metric):
    """``P_M(v)`` for the spatial-full metric, with a status per cone.

    ``v`` is one cone of shape ``(k,)`` or a batch of shape ``(cones, k)``.
    Returns ``(z, status)``, ``status`` one of this module's STATUS_ values.  A
    FAILED cone returns NaN for its point: a caller cannot mistake it for an
    answer, and a batch reports it without disturbing its neighbours.
    """
    single = (jnp.ndim(v) == 1)
    V = jnp.reshape(v, (1, -1)) if single else jnp.asarray(v)
    if V.ndim != 2:
        raise ValueError(
            f"spatial-full SOC prox takes (k,) or (cones, k), got {V.shape}")
    n_cones, kk = V.shape
    if metric.k != kk:
        raise ValueError(
            f"spatial-full metric is for k = {metric.k}, not {kk}")
    a, b, Q = metric.factors_for(n_cones)
    a = a.astype(V.dtype)
    b = b.astype(V.dtype)
    Q = Q.astype(V.dtype)

    t = V[:, 0]
    y = V[:, 1:]
    yt = _rotate_t(Q, y)                             # Q^T y, the rotated spatial part
    bw = b * yt                                      # b_j ytilde_j

    norm_y = jnp.sqrt(jnp.sum(y * y, axis=1))
    inside = t >= norm_y
    # The POLAR test is on M v = (a t, Q diag(b) Q^T y), and ||M v spatial||
    # equals ||bw|| because Q is orthogonal, so no second rotation is needed.
    norm_by = jnp.sqrt(jnp.sum(bw * bw, axis=1))
    polar = jnp.logical_and(~inside, (a * t) <= -norm_by)

    root = jnp.logical_and(~inside, ~polar)
    pos = jnp.logical_and(root, t > 0)
    neg = jnp.logical_and(root, t < 0)
    apex = jnp.logical_and(root, t == 0)

    norm_yt = jnp.sqrt(jnp.sum(yt * yt, axis=1))
    lo, up, established = _brackets(a, t, b, bw, norm_yt, pos, neg, root)

    # A t < 0 cone whose bracket cannot be established is NOT a failure.  In
    # exact arithmetic G tends to a|t| - ||bw|| < 0 off the polar branch, so the
    # only way the float G stays positive is that ``1 - kappa`` has reached its
    # own roundoff floor, with ``kappa = a|t| / ||bw||``.  There the root is
    # beyond the resolvable range and, from ``phi* >= cmin kappa / (1 - kappa)``,
    #
    #     ||u(theta*)||  <=  a|t| / phi*  <=  ||bw|| (1 - kappa) / cmin,
    #
    # which at that floor is roundoff times the size of the input.  So the answer
    # is zero to within roundoff, and returning zero is right where refusing
    # would be useless.  It is reported under its OWN status, because a fold is
    # only acceptable if a caller can see it happened.
    # Only for a FINITE input.  A cone carrying NaN also fails to establish a
    # bracket, and folding that to zero would turn a refusal into an answer,
    # which is the one thing this module must never do.
    beyond = jnp.logical_and(jnp.logical_and(neg, ~established),
                             jnp.all(jnp.isfinite(V), axis=1))
    polar = jnp.logical_or(polar, beyond)
    root = jnp.logical_and(root, ~beyond)
    pos = jnp.logical_and(pos, ~beyond)
    neg = jnp.logical_and(neg, ~beyond)
    apex = jnp.logical_and(apex, ~beyond)

    def bisect(_i, bounds):
        lo_, up_ = bounds
        # GEOMETRIC, because the derived bracket's width is proportional to the
        # root: halving the ratio is what gives relative accuracy near either
        # branch boundary.  Written as lo sqrt(up/lo) so a wide bracket cannot
        # overflow through the product.
        ratio = up_ / jnp.where(lo_ > 0, lo_, 1.0)
        mid = jnp.where(lo_ > 0, lo_ * jnp.sqrt(ratio), 0.5 * (lo_ + up_))
        # G is strictly decreasing on either bracket, so G(mid) > 0 puts the
        # root above mid.  One rule serves both signs of t.
        above = _residual(a, t, b, bw, mid, neg) > 0
        return (jnp.where(above, mid, lo_), jnp.where(above, up_, mid))

    lo, _up = jax.lax.fori_loop(0, BISECTION_STEPS, bisect, (lo, up))
    # The LOWER end, where G >= 0.  That choice is what makes z feasible and
    # n polar-feasible exactly, leaving the root error in |z^T n| alone.
    _g, u, _theta = _state(a, t, b, bw, lo, neg)
    # The radius is the norm of the ROTATED-BACK spatial part, by the same
    # expression the branch test uses on y.  So the returned point is on the
    # cone boundary in the ORIGINAL coordinates, not merely in the rotated
    # calculation, and P_M(P_M(v)) = P_M(v) bitwise through the inside branch.
    # The price is that z^T n = -||u|| G now holds only to rotation roundoff,
    # which the reported defects carry.
    spatial = _rotate(Q, u)
    radius = jnp.sqrt(jnp.sum(spatial * spatial, axis=1))
    z_root = jnp.concatenate([radius[:, None], spatial], axis=1)

    zero = jnp.zeros_like(V)
    z = jnp.where(inside[:, None], V, jnp.where(polar[:, None], zero, z_root))

    status = jnp.where(inside, STATUS_INSIDE,
                       jnp.where(beyond, STATUS_POLAR_LIMIT,
                                 jnp.where(polar, STATUS_POLAR,
                                           jnp.where(apex, STATUS_APEX,
                                                     STATUS_ROOT))))
    well_posed = jnp.logical_and(a > 0, jnp.all(b > 0, axis=1))
    # FAILED is now reserved for what it should mean: a metric that is not
    # positive, a non-finite input, a non-finite intermediate.  The remaining
    # ``~established`` term can only be a t > 0 cone, which the unconditional
    # theta = a end rules out, so it is a guard and not a branch.
    bad = jnp.logical_or(
        ~well_posed,
        jnp.logical_or(jnp.logical_and(root, ~established),
                       ~jnp.all(jnp.isfinite(z), axis=1)))
    status = jnp.where(bad, STATUS_FAILED, status)
    z = jnp.where(bad[:, None], jnp.asarray(jnp.nan, V.dtype), z)
    if single:
        return z[0], status[0]
    return z, status


def project_polar_step(w, sigma):
    """The dual update's prox of the conjugate indicator, production form.

    ``w - Sigma P_Sigma(Sigma^{-1} w)``, which is the form a dual PDHG step
    with step matrix ``Sigma`` needs, the conjugate of ``I_Q`` being
    ``I_{-Q}``.  Uses ONE projection in the ``Sigma`` metric.
    """
    inv = sigma.inverse()
    z, status = project(inv.apply(w), sigma)
    return w - sigma.apply(z), status


def project_polar_step_moreau(w, sigma):
    """The same quantity by the other exact form, ``-P_{Sigma^{-1}}(-w)``.

    Kept separate ON PURPOSE.  The two forms agree analytically (the proof is
    the optimality conditions of one read as those of the other), so a
    disagreement is evidence, and supplying ``M`` where ``M^{-1}`` is needed
    breaks exactly this pair.
    """
    z, status = project(-w, sigma.inverse())
    return -z, status


def optimality_defects(v, z, metric):
    """The three defects of the specification, in the ORIGINAL coordinates.

    Returns a dict of arrays: ``primal`` is how far ``z`` is outside ``Q``,
    ``polar`` how far ``n = M (v - z)`` is outside ``-Q``, ``complementarity``
    is ``|z^T n|``, and the three ``*_scale`` entries are the magnitudes those
    are to be read against.  These are quantities to REPORT.  Nothing inside
    the kernel compares against them, so no tolerance here can silently become
    an acceptance rule.

    Two things about the scales, because the wrong one turns a good point into a
    bad number and the reverse.

    ``complementarity_scale`` is ``z^T M z``, NOT ``||z|| ||n||``.  The ratio
    against ``z^T M z`` is the squared relative M-distance to the exact
    projection, by the certificate above, so it says something.  The ratio
    against ``||n||`` says almost nothing near the cone boundary: there ``n`` is
    a difference of nearly equal terms, so its DIRECTION is noise and the
    normalized angle can reach order one while the point itself is correct to
    unit roundoff.

    ``primal`` is zero by construction, since the radius is the norm of the
    returned spatial part.  It is reported so a caller can see that, and it is
    NOT evidence: the two conditions carrying evidence here are the polar one
    and the complementarity.
    """
    single = (jnp.ndim(v) == 1)
    V = jnp.reshape(v, (1, -1)) if single else jnp.asarray(v)
    Z = jnp.reshape(z, (1, -1)) if single else jnp.asarray(z)
    a, b, Q = metric.factors_for(V.shape[0])
    d = V - Z
    n = metric.apply(d)
    if n.ndim == 1:
        n = jnp.reshape(n, (1, -1))
    primal = jnp.maximum(jnp.sqrt(jnp.sum(Z[:, 1:] ** 2, axis=1)) - Z[:, 0], 0.0)
    polar = jnp.maximum(jnp.sqrt(jnp.sum(n[:, 1:] ** 2, axis=1)) + n[:, 0], 0.0)
    comp = jnp.abs(jnp.sum(Z * n, axis=1))
    norm_v = jnp.sqrt(jnp.sum(V * V, axis=1))
    norm_z = jnp.sqrt(jnp.sum(Z * Z, axis=1))
    energy = jnp.sum(Z * metric.apply(Z), axis=1)
    top = jnp.maximum(a, jnp.max(b, axis=1))
    out = {"primal": primal, "polar": polar, "complementarity": comp,
           "primal_scale": norm_z, "polar_scale": top * norm_v,
           "complementarity_scale": energy}
    if single:
        return {key: val[0] for key, val in out.items()}
    return out


def distance_estimate(defects):
    """``sqrt(|z^T n|)``, an ESTIMATE of ``||z - z*||_M``, from the defects.

    The inequality it comes from, derived above, is exact in real arithmetic:
    with ``z`` in ``Q`` and ``n`` in ``-Q``, ``||z - z*||_M^2 <= |z^T n|``.  The
    number returned here is NOT a bound, for two reasons, and both are real
    rather than pedantic:

    * the float ``z^T n`` is a difference of nearly equal products and can round
      to exactly zero at a point that is a measurable distance from the exact
      projection, which ``tests/test_spatial_soc_prox.py`` measures;
    * near either boundary the hypothesis ``n`` in ``-Q`` cannot be established
      from the float residual at all.

    ``validated_distance_bound`` in that test file is the certified version: it
    charges the summation allowance on an upper bound of the exact ``|z^T n|``
    and carries the polar residual through a quadratic instead of assuming it
    away.  Quote that one, and use this only as a diagnostic.
    """
    return jnp.sqrt(defects["complementarity"])


def relative_distance_estimate(defects):
    """The same estimate over ``||z||_M``, which is what makes it readable.

    Its square is the reported complementarity over ``z^T M z``.  A
    complementarity at unit roundoff corresponds to a relative distance near its
    SQUARE ROOT, which is as far as a complementarity residual can ever be
    pushed, so a loose-looking number here is the certificate's nature and not
    the point's error.  The measured error of the point is smaller, and the
    exact rational root in the gate is what establishes by how much.
    """
    energy = defects["complementarity_scale"]
    return jnp.sqrt(defects["complementarity"]
                    / jnp.where(energy > 0, energy, 1.0))
