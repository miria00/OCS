"""Row-side block metrics for rotation-invariant Euclidean-norm blocks.

A dual term ``f(K x + l)`` whose block is a sum of Euclidean norms over
contiguous groups of size ``k`` (``GroupL2``, a single-group ``L2``, or a
positively scaled wrapper of either) can accept a symmetric positive-definite
dual metric per group, not only one scalar for the whole term. Three concepts
are kept separate:

* the mathematical proximal block: the ``k`` coordinates coupled by one norm;
* the execution group: blocks evaluated together by one batched call
  (``ocs.prox.groups._Group``); batching never merges norms;
* the metric-sharing group: a contiguous range of mathematical blocks that use
  one declared shape ``S = Q diag(w) Q^T``, stored once as ``(Q, w)``.

For mathematical block ``i`` the effective dual step is

    Sigma_i = diag(sigma_i) S          (S diagonal: sigma may vary inside i)
    Sigma_i = c_i S                    (S full: sigma_i = c_i is uniform)

where ``sigma`` is the compiler's ordinary per-coordinate dual step (after any
primal weight or Halpern weight). Both forms are symmetric positive definite
and commute with ``S``. The dual update is the weighted Moreau identity

    prox_{f*}^{Sigma^{-1}}(v) = v - Sigma prox_f^{Sigma}(Sigma^{-1} v).

Because ``||Q^T z||_2 = ||z||_2``, ``prox_f^{Q D Q^T}(u) = Q prox_f^{D}(Q^T u)``
for these blocks only. The diagonal-metric norm prox has no closed form; it is
computed by a safeguarded Newton solve on a scalar secular equation per group
with a fixed iteration count (``NEWTON_ITERS``). This is an inexact prox inside
PDHG with a fixed budget, not an adaptive or certified inner solve.

Arbitrary functions are NOT rotation invariant, so every other block type is
rejected with an error, with one further family admitted below. The
certificate and status checks never see a block metric: they keep calling
``block.prox`` with scalar steps.

Quadratic blocks
----------------
A dual term whose block is ``SumSquares``, ``Quadratic``, ``Zero``, or a
positively scaled wrapper of one of those, ``f(z) = 1/2 z^T H z + b^T z`` with
``H`` symmetric positive semidefinite, admits ANY symmetric positive definite
metric through one ``k x k`` linear solve on the block variable:

    prox_f^{Sigma}(u) solves (H + Sigma) z = Sigma u - b,

and with the weighted Moreau identity above (``u = Sigma^{-1} v``) the dual
update is ``(H + Sigma) z = v - b``, ``y = v - Sigma z``. For these blocks
``k`` is not fixed by the function: ``Quadratic`` couples all ``dim``
coordinates and takes ``k = dim``, while the separable ``SumSquares`` and
``Zero`` couple nothing, so the grouping of their rows into metric blocks of
``k`` coordinates is DECLARED, never intrinsic. The carrier is
``QuadraticShape``, one per (term, sharing group), which holds the declared
``(Q, w)`` plus the block-dependent factorization computed once on the host:

* ``SumSquares`` (``H = a I``): ``H`` commutes with ``S``, so in the eigenbasis
  of ``S`` the solve is elementwise, ``y = Q diag(a / (a + c w)) Q^T v``. No
  factorization beyond ``(Q, w)``.
* ``Quadratic`` (``H = a P``, ``b = a q``): ``c`` is a runtime scalar, so the
  PENCIL is factorized once, ``M = L^{-1} H L^{-1}`` with ``L = S^{1/2}``,
  ``M = U diag(lam) U^T``, ``G = L^{-1} U``, giving
  ``(H + c S)^{-1} = G diag(1 / (lam + c)) G^T`` for every ``c > 0``, even
  when ``P`` is singular. Two ``k x k`` products with the cached ``G`` per
  block per iteration, never a fresh ``jnp.linalg.solve``.
* ``Zero`` (``H = 0``): ``f^* = I_{{0}}``, so the dual rows are exactly zero
  under every metric. The shape on those rows never enters a dual prox. It
  DOES enter the joint metric construction and the step normalization, and
  so changes the constructed metric of every other term.

The Quadratic pencil path has a SECOND error axis beyond ``cond(S)``: on a
null direction of ``P`` the solve divides by ``c`` alone, so the float32 KKT
defect grows like ``eps32 * (cond(S) + ||H||_2 / (c w_min))``. The ratio
``||H||_2 / (c w_min)`` is a property of the block's Hessian against the
dual step, not of the shape, and it is checked on the concrete compile-time
steps against ``MAX_QUADRATIC_HESSIAN_RATIO`` (measured below). The native
``Quadratic.prox`` (``jnp.linalg.solve(P + diag(1/t), ...)`` in float32) has
the same hazard and is NOT validated against this bar by anything in this
module.

A symmetric-cone block (``SecondOrderCone``, ``SOCBatch``, ``PSDCone``) takes
a different carrier, ``ocs.prox.cone_metric.ConeAutomorphism``: one Jordan
element per cone drawn from the cone's automorphism group, applied as
``Sigma_i = c P(w_i)`` with ONE step ``c`` per term, and a dual update in
closed form. It shares the ``(k, ranges)`` plumbing below (the term is one
range) and dispatches on the carrier type; it is never a ``(Q, w)`` eigenpair
and never a diagonal shape, which a second-order cone admits only when scalar.

The same ``(Q, w)`` carrier and ``make_shape_apply`` also serve a declared
COLUMN shape, ``compile_family(equilibrate="construct", metric_columns=("full",
k))``: one ``k x k`` shape shared by every block of ``k`` consecutive primal
coordinates, applied to the primal vector (the adjoint and the cost of the
gradient step) on a family with no distinguished primal block. No prox is
involved on that side, so the numerical-domain constant below is inherited
there as the carrier's one documented bound, not as a measurement of the
column application.

Numerical domain
----------------
The weighted prox is evaluated in float32 and returns a float32 point. Its
proximal optimality (KKT) defect

    ||Sigma (z - u) + lam z / ||z||_2 ||_2 / (||Sigma u||_2 + lam)

measured in float64 against an independent reference grows LINEARLY in the
condition number of a rotated shape ``S``: an error of one float32 ulp in the
returned array costs ``w_max`` in the numerator while the normalizing scale is
set by ``w_min``. This is a property of returning a float32 array, not of the
algorithm: the exact float64 minimizer, merely rounded to float32, already has a
defect of the same size, and the arithmetic inside the rotation is already at
``jax.lax.Precision.HIGHEST`` (verified in the compiled HLO:
``operand_precision={highest,highest}``, no tf32).

So a rotated shape is admitted only inside a measured range, ``MAX_FULL_COND``;
outside it the declaration is REJECTED with an error. A shape supplied by the
caller is never clipped: ``restrict_condition`` exists for a caller that builds
a shape automatically and is willing to restrict it explicitly and recompute its
step normalization. A diagonal shape rotates nothing and shows no such growth,
so it carries only a loose guard, ``MAX_DIAGONAL_COND``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

BLOCK = "block"
NEWTON_ITERS = 20
_ORTHOGONALITY_TOL = 1e-5

# ----------------------------------------------------- validated numerical domain
# Declared proximal parity bar for the weighted prox: the float64 KKT defect of the
# returned float32 point, ||Sigma (z-u) + lam z/||z|| ||_2 / (||Sigma u||_2 + lam).
PROX_KKT_BAR = 1e-4

# Measured on KEVIN (RTX 5090, GPU-f5e87fe4), 1152 cases per condition number
# (k in 2,3,4,5,8,16; lam in 0.1,1,10; six radii; 64 rows), full rotated shapes.
# "floor" is the EXACT float64 minimizer merely rounded to float32, so it is what any
# float32-output implementation pays; "shipped" is this module at Precision.HIGHEST.
#     cond(S)      2       5      10      30     100     300    1000    2981    1e4
#     floor     6.3e-8  1.3e-7  3.2e-7  6.9e-7  2.9e-6  8.2e-6  3.3e-5  1.1e-4  2.2e-4
#     shipped   2.3e-7  3.9e-7  9.2e-7  1.9e-6  6.4e-6  2.2e-5  7.2e-5  1.9e-4  7.7e-4
# At cond 2981 the FLOOR alone is 1.12e-4: the bar cannot be met there by any float32
# implementation, so no precision path exists and the range is restricted instead.
# 100 is the largest ladder point whose measured worst stays a full decade under the
# bar (6.36e-6, margin 15.7x); 300 leaves 4.6x and 1000 only 1.4x.
MAX_FULL_COND = 100.0

# A diagonal shape rotates nothing, so no eigendirection is mixed and the defect does
# not grow with the spread: measured flat at 0.8e-7 to 8.7e-7 across effective
# condition numbers 1 to 3.9e4 (same grid). This bound is a loose guard on the
# EFFECTIVE metric diag(sigma_i) S, measured at its own boundary, not a cliff.
MAX_DIAGONAL_COND = 1e5

# Quadratic blocks under a shape (QuadraticShape). MEASURED on MAZEPPA, RTX 5090
# (GPU-78b0d21c-fe02-946b-2e74-6930e848f338), JAX 0.11.1, 2026-09-21T00:25:35Z, script
# quad_kkt_ladder.py sha256 5e97e492154b4a5c..., record candidates/full_metric_extension_20260919/
# work/item1_quadratic_matrix/logs/quad_kkt_ladder.json. Float32 KKT defect
# ||(H + Sigma) z32 - (Sigma u - b)|| / (||Sigma u|| + ||b||), float64 on the float32 point,
# worst over k in 2,3,4,5,8,16, six radii of v, 64 rows, 3 seeds (and a in 0.5,1,2 with c in
# 0.1,1,10 for SumSquares; singular and nonsingular random dyadic PSD P with ||P||_2 in (1/2, 1]
# for Quadratic). "floor" is the same defect of the exact float64 minimizer rounded to float32.
# Criterion, as for the KEVIN norm table above: a full decade under PROX_KKT_BAR (<= 1e-5).
QUADRATIC_LADDER_TABLE = """
SumSquares (H = a I), rotated shape, worst defect32 / floor by cond(S):
  2: 2.86e-07/8.88e-08  5: 4.90e-07/2.13e-07  10: 8.83e-07/4.15e-07  30: 2.70e-06/1.26e-06  100: 9.76e-06/4.30e-06  300: 2.27e-05/1.17e-05  1000: 8.59e-05/4.29e-05  2981: 2.70e-04/1.20e-04  10000: 8.08e-04/3.84e-04
  -> cond(S) <= 100 meets the criterion (9.8e-6 at 100), so MAX_FULL_COND = 100 applies as measured.
Quadratic (H = a P), rotated shape, worst defect32 by cond(S) (rows) and ||H||_2 / (c w_min) (columns):
  cond\ratio        1       10       30      100      300     1000     3000    10000
          1   3.3e-07  8.7e-07  2.7e-06  7.5e-06  2.3e-05  8.6e-05  3.0e-04  7.5e-04
          2   3.4e-07  8.9e-07  1.8e-06  5.6e-06  2.3e-05  6.3e-05  2.4e-04  6.0e-04
          5   5.9e-07  9.1e-07  1.6e-06  5.3e-06  1.6e-05  6.1e-05  2.0e-04  5.0e-04
         10   6.7e-07  1.1e-06  1.3e-06  3.3e-06  1.5e-05  3.2e-05  1.3e-04  6.5e-04
         30   1.4e-06  1.6e-06  2.0e-06  6.6e-06  1.7e-05  6.7e-05  1.3e-04  5.8e-04
        100   4.1e-06  4.4e-06  7.0e-06  8.2e-06  9.7e-06  2.7e-05  6.5e-05  1.4e-04
        300   1.3e-05  1.9e-05  9.3e-06  1.4e-05  3.3e-05  3.0e-05  3.9e-05  6.9e-05
  -> every cond(S) <= 100 meets the criterion at ratio <= 100 (worst 8.2e-6 at cond 100, ratio 100);
     ratio 300 fails it already at cond(S) = 1 (2.3e-5). The domain is two-parameter, as the
     refutation of the design predicted: cond(S) <= MAX_FULL_COND AND ratio <= MAX_QUADRATIC_HESSIAN_RATIO.
"""
MAX_QUADRATIC_HESSIAN_RATIO = 100.0    # the largest ladder ratio meeting the criterion at every cond(S) <= MAX_FULL_COND


def _mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _domain_message(cond, limit, diagonal, what="block metric"):
    kind = "diagonal" if diagonal else "full (rotated)"
    if diagonal:
        why = ("Beyond the measured range the weighted prox has not been validated against the "
               f"declared {PROX_KKT_BAR:g} proximal optimality bar.")
        how = ("Rescale the shape, or split the term so each sharing group spans a narrower "
               "range of row scales.")
    else:
        why = ("Beyond this bound the float32 weighted prox no longer meets the declared "
               f"{PROX_KKT_BAR:g} proximal optimality bar: the KKT defect of the returned "
               "float32 point grows like cond(S) * eps32, and at cond(S) = 2981 even the exact "
               "float64 minimizer rounded to float32 has a defect of 1.1e-4, so no amount of "
               "internal precision would make it pass.")
        how = ("Either declare a diagonal shape (Q=None, no bound of this kind), or restrict the "
               "shape explicitly with ocs.prox.block_metric.restrict_condition(S, max_cond) and "
               "RECOMPUTE the step normalization against the restricted shape. A shape you supply "
               "is never clipped for you.")
    return (f"{what}: a {kind} shape must have condition number at most {limit:g}; this one has "
            f"{cond:.6g}. {why} {how}")


class BlockMetric:
    """One shared dual metric shape ``S = Q diag(w) Q^T`` for a range of groups.

    ``w``: (k,) finite positive values. ``Q``: None (``S = diag(w)``) or a
    (k, k) orthogonal matrix. ``groups``: None for every mathematical block of
    the term, or ``(start, end)`` group indices (half-open) inside the term.
    The shape is validated and stored once on the host (float64) and once on
    the device (float32); it is never broadcast to one copy per group.
    """

    def __init__(self, w, Q=None, groups=None):
        with np.errstate(over="ignore", invalid="ignore"):
            w64 = np.asarray(w, dtype=np.float64)
        if w64.ndim != 1 or not w64.size or not np.all(np.isfinite(w64)) or not np.all(w64 > 0):
            raise ValueError("block metric w must be a nonempty vector of finite positive values")
        k = w64.size
        q64 = None
        if Q is not None:
            q64 = np.asarray(Q, dtype=np.float64)
            if q64.shape != (k, k) or not np.all(np.isfinite(q64)):
                raise ValueError(f"block metric Q must be a finite ({k}, {k}) matrix")
            gap = float(np.max(np.abs(q64.T @ q64 - np.eye(k))))
            if gap > _ORTHOGONALITY_TOL:
                raise ValueError(f"block metric Q must be orthogonal (max |Q^T Q - I| = {gap:.3g})")
        if float(np.max(w64)) > float(np.finfo(np.float32).max) or float(np.min(w64)) < float(np.finfo(np.float32).tiny):
            raise ValueError("block metric w must be representable in float32")
        cond = float(np.max(w64) / np.min(w64))
        limit = MAX_DIAGONAL_COND if q64 is None else MAX_FULL_COND
        if not (cond <= limit):
            raise ValueError(_domain_message(cond, limit, q64 is None))
        if groups is not None:
            start, end = (int(groups[0]), int(groups[1]))
            if start < 0 or end <= start or (start, end) != tuple(groups):
                raise ValueError("block metric groups must be integer (start, end) with 0 <= start < end")
            groups = (start, end)
        self.k = k
        self.groups = groups
        self.cond = cond
        self.w_host = w64
        self.Q_host = q64
        self.w = jnp.asarray(w64, jnp.float32)
        self.Q = None if q64 is None else jnp.asarray(q64, jnp.float32)
        self.w_sqrt = jnp.asarray(np.sqrt(w64), jnp.float32)

    @classmethod
    def from_matrix(cls, S, groups=None):
        """Eigendecompose one small SPD matrix on the host (O(k^3), once)."""
        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.all(np.isfinite(S)):
            raise ValueError("block metric matrix must be finite and square")
        if not np.array_equal(S, S.T):
            raise ValueError("block metric matrix must be exactly symmetric")
        off = S - np.diag(np.diag(S))
        if not np.any(off):
            return cls(np.diag(S), None, groups)
        values, vectors = np.linalg.eigh(S)
        if not np.all(values > 0):
            raise ValueError("block metric matrix must be positive definite")
        return cls(values, vectors, groups)

    @property
    def kind(self):
        return "diagonal" if self.Q is None else "full"

    def matrix(self):
        """Host float64 (k, k) SPD shape, for logs and references only."""
        if self.Q_host is None:
            return np.diag(self.w_host)
        return self.Q_host @ np.diag(self.w_host) @ self.Q_host.T

    def describe(self):
        return {"kind": self.kind, "k": self.k, "groups": self.groups,
                "w": self.w_host.tolist(),
                "Q": None if self.Q_host is None else self.Q_host.tolist(),
                "eig_min": float(self.w_host.min()), "eig_max": float(self.w_host.max()),
                "cond": self.cond,
                "cond_limit": MAX_DIAGONAL_COND if self.Q is None else MAX_FULL_COND,
                "stored_device_arrays": ["w (k,)", "sqrt(w) (k,)"] + ([] if self.Q is None else ["Q (k,k)"])}

    def __repr__(self):
        return f"BlockMetric(kind={self.kind}, k={self.k}, groups={self.groups})"


def restrict_condition(S, max_cond=MAX_FULL_COND):
    """Explicitly restrict a symmetric positive definite shape to the validated range.

    ``S``: a (k, k) symmetric positive definite matrix, or a (k,) vector of
    eigenvalues. Returns ``(restricted, info)`` in the same form as the input.

    The restriction raises the small eigenvalues to ``w_max / max_cond`` and
    leaves ``w_max`` and the eigenvectors alone, so ``restricted >= S`` in the
    positive semidefinite order and the dual step shrinks in exactly the
    restricted directions.

    NOTHING CALLS THIS FOR YOU. ``BlockMetric`` and ``BlockMetric.from_matrix``
    REJECT an out-of-range shape rather than clip it, because a metric that is
    promised to be applied unchanged must be applied unchanged. A caller that
    builds a shape automatically may call this, must record ``info``, and must
    RECOMPUTE its step normalization against the restricted shape: raising
    eigenvalues raises ||Sigma^1/2 K T^1/2||_2, so a bound measured on the
    original shape no longer certifies the PDHG step condition.
    """
    max_cond = float(max_cond)
    if not (np.isfinite(max_cond) and max_cond >= 1.0):
        raise ValueError("max_cond must be finite and at least 1")
    a = np.asarray(S, dtype=np.float64)
    vector = a.ndim == 1
    if vector:
        if not a.size or not np.all(np.isfinite(a)) or not np.all(a > 0):
            raise ValueError("eigenvalues must be finite and positive")
        w, V = a, None
    else:
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
            raise ValueError("shape must be a finite square matrix")
        if not np.array_equal(a, a.T):
            raise ValueError("shape must be exactly symmetric")
        w, V = np.linalg.eigh(a)
        if not np.all(w > 0):
            raise ValueError("shape must be positive definite")
    top = float(np.max(w))
    # A hair inside the bound, so that reconstructing S and eigendecomposing it again (which
    # BlockMetric.from_matrix does) cannot land a few ulp OUTSIDE the bound and be refused.
    floor = (top / max_cond) * (1 + 1e-9)
    w_new = np.maximum(w, floor)
    info = {"restricted": bool(np.any(w_new != w)),
            "max_cond": max_cond,
            "cond_before": float(top / np.min(w)),
            "cond_after": float(top / np.min(w_new)),
            "eigenvalue_floor": floor,
            "eigenvalues_raised": int(np.count_nonzero(w_new != w)),
            "eigenvalues_before": w.tolist(),
            "eigenvalues_after": w_new.tolist(),
            "step_normalization": "must be recomputed against the restricted shape"}
    if vector:
        return w_new, info
    out = V @ np.diag(w_new) @ V.T
    out = 0.5 * (out + out.T)       # exactly symmetric, so from_matrix accepts it
    return out, info


# ----------------------------------------------------------- admissible blocks
def euclidean_norm_structure(block, size):
    """Return the group size k if ``block`` is a supported rotation-invariant norm.

    Exact types only: a subclass or an instance-level prox override may change
    the function. Returns None for every unsupported block.
    """
    from . import base, library
    if "prox" in vars(block):
        return None
    if type(block) is library.GroupL2:
        return block.g if size % block.g == 0 else None
    if type(block) is library.L2:
        return size
    if type(block) is base._Scaled:
        return euclidean_norm_structure(block.base, size)
    return None


QUADRATIC_KINDS = ("sumsquares", "quadratic", "zero")


def quadratic_structure(block, size):
    """Return ``(kind, a, P_host, q_host)`` if ``block`` is a supported quadratic block.

    ``kind`` is ``"sumsquares"`` (``f = a/2 ||z||^2``), ``"quadratic"``
    (``f = a/2 z^T P z + a q^T z``) or ``"zero"`` (``f = 0``); ``a`` is the
    product of ``_Scaled`` coefficients (1 when there is none). Exact types
    only, as ``euclidean_norm_structure``: a subclass, an instance-level prox
    override, and the ``_AffinePre`` / ``_LinQuad`` calculus wrappers return
    None (a retained limitation, refused rather than routed). ``size`` is the
    term's row count and is not checked against ``Quadratic.dim`` here; the
    carrier checks the shape against the concrete Hessian.
    """
    from . import base, library
    if "prox" in vars(block):
        return None
    kind = type(block)
    if kind is library.SumSquares:
        return "sumsquares", 1.0, None, None
    if kind is library.Zero:
        return "zero", 1.0, None, None
    if kind is library.Quadratic:
        return ("quadratic", 1.0, np.asarray(jax.device_get(block.P), np.float64),
                np.asarray(jax.device_get(block.q), np.float64))
    if kind is base._Scaled:
        inner = quadratic_structure(block.base, size)
        if inner is None:
            return None
        inner_kind, a, P, q = inner
        return inner_kind, float(block.c) * a, P, q
    return None


def _ratio_message(ratio, limit, term_index, groups):
    return (f"block_metrics[{term_index}] groups {groups[0]}..{groups[1]}: the Quadratic block's "
            f"Hessian-to-metric ratio ||H||_2 / (c w_min) is {ratio:.6g}, above the validated "
            f"bound {limit:g}. On a null direction of the Hessian the cached pencil solve divides "
            f"by the dual step alone, so the float32 KKT defect of the returned point grows like "
            f"eps32 * ||H|| / (c w_min) regardless of cond(S), and beyond this bound it has not "
            f"been measured under the declared {PROX_KKT_BAR:g} proximal optimality bar. The "
            "domain of the quadratic full class is two-parameter: cond(S) and this ratio. "
            "Nothing is rescaled for you. Declare 'diagonal' per_block for this term (the "
            "block's native prox, which is not validated against this bar either), or scale "
            "the block so its Hessian is smaller against the dual step.")


class QuadraticShape(BlockMetric):
    """A declared shape ``S = Q diag(w) Q^T`` plus one quadratic block's factorization.

    Created only by ``normalize_term_metrics`` when the block is a quadratic
    block, one per (term, sharing group). The shape validation, condition
    bound and ``describe()`` of ``BlockMetric`` are inherited unchanged (the
    constructor re-validates the copied ``(w, Q, groups)``). The factorization
    is computed once on the host in float64 from the concrete block and stored
    once on the device in float32. ``k`` is the metric block size.
    """

    def __init__(self, metric, block, size, term_index=0):
        structure = quadratic_structure(block, size)
        if structure is None:
            raise ValueError(f"block_metrics[{term_index}]: {type(block).__name__} is not a "
                             "quadratic block")
        super().__init__(metric.w_host, metric.Q_host, metric.groups)
        kind, a, P, q = structure
        self.quadratic_kind = kind
        self.a = float(a)
        self.G = self.lam = self.b = None
        self.G_host = self.lam_host = self.b_host = None
        self.hessian_ratio = None            # filled by check_static_steps (host, concrete steps)
        if kind == "sumsquares":
            self.hessian_norm = self.a
        elif kind == "zero":
            self.hessian_norm = 0.0
        else:
            k = self.k
            if P.shape != (k, k):
                raise ValueError(f"block_metrics[{term_index}]: a Quadratic block of dimension "
                                 f"{P.shape[0]} accepts only a {P.shape[0]} x {P.shape[0]} shape, "
                                 f"got k = {k}")
            H = self.a * P
            b = self.a * q
            inv_root = 1.0 / np.sqrt(self.w_host)
            if self.Q_host is None:
                inverse_root = np.diag(inv_root)
            else:
                inverse_root = (self.Q_host * inv_root[None, :]) @ self.Q_host.T
            M = inverse_root @ H @ inverse_root
            M = 0.5 * (M + M.T)
            lam, U = np.linalg.eigh(M)
            floor = -16 * k * np.finfo(np.float64).eps * max(float(np.max(np.abs(M))), 1.0)
            G = inverse_root @ U
            if (not np.all(np.isfinite(lam)) or not np.all(np.isfinite(G))
                    or float(np.min(lam)) < floor):
                raise ValueError(
                    f"block_metrics[{term_index}]: the generalized eigenpair of the quadratic "
                    "block's Hessian against the declared shape is not finite positive "
                    f"semidefinite (min eigenvalue {float(np.min(lam)):.3g}). No factorization "
                    "is cached and nothing is repaired.")
            lam = np.maximum(lam, 0.0)
            self.hessian_norm = float(self.a * max(float(np.max(np.linalg.eigvalsh(P))), 0.0))
            self.G_host, self.lam_host, self.b_host = G, lam, b
            self.G = jnp.asarray(G, jnp.float32)
            self.lam = jnp.asarray(lam, jnp.float32)
            self.b = jnp.asarray(b, jnp.float32)

    @property
    def prox_label(self):
        return {
            "sumsquares": "closed form in the eigenbasis of S: y = Q diag(a / (a + c w)) Q^T v, "
                          "no solve, exact up to the two rotations by Q",
            "quadratic": "cached generalized eigenpair of (H, S), one k x k solve per block: "
                         "z = G diag(1 / (lam + c)) G^T (v - b), y = v - c S z",
            "zero": "identity prox, the dual rows are exactly zero under every metric",
        }[self.quadratic_kind]

    def describe(self):
        record = super().describe()
        hessian = {"sumsquares": "a I", "quadratic": "a P, dense PSD", "zero": "0"}[self.quadratic_kind]
        quad = {"kind": self.quadratic_kind, "hessian": hessian, "scale_a": self.a,
                "hessian_norm": float(self.hessian_norm),
                "factorization": ("none beyond (Q, w): H commutes with S" if self.quadratic_kind == "sumsquares"
                                  else "none: H = 0" if self.quadratic_kind == "zero"
                                  else "pencil eigenpair (lam, G = S^-1/2 U) of S^-1/2 H S^-1/2, "
                                       "computed once on the host in float64, O(k^3) once"),
                "prox": self.prox_label,
                "pencil_eigenvalues_min": None if self.lam_host is None else float(np.min(self.lam_host)),
                "pencil_eigenvalues_max": None if self.lam_host is None else float(np.max(self.lam_host)),
                "hessian_to_metric_ratio": self.hessian_ratio,
                "hessian_to_metric_ratio_limit": (MAX_QUADRATIC_HESSIAN_RATIO
                                                  if self.quadratic_kind == "quadratic" else None),
                "stored_device_arrays": (["G (k,k)", "lam (k,)", "b (k,)"]
                                         if self.quadratic_kind == "quadratic" else [])}
        record["quadratic"] = quad
        return record

    def __repr__(self):
        return (f"QuadraticShape({self.quadratic_kind}, kind={self.kind}, k={self.k}, "
                f"groups={self.groups})")


def _norm_weight(block):
    """lam of f = lam * sum_g ||.||_2, read at evaluation time (packed or traced)."""
    from . import base
    if type(block) is base._Scaled:
        return block.c * _norm_weight(block.base)
    return block.lam


def normalize_term_metrics(declared, term_index, block, size):
    """Validate one term's declaration; return ``(k, ranges)``.

    ``ranges`` is a tuple of ``(start, end, carrier)``, sorted by ``start``,
    contiguous from 0, covering exactly ``[0, size // k)``. ``k`` is a
    positive divisor of ``size`` and is the number of dual rows in ONE
    plumbing block. ``size`` is the term's row count.

    Two carrier families. They are never mixed inside one term, and the
    family is read from ``type(ranges[0][2])``.

    * Shape carriers, ``BlockMetric`` and its ``QuadraticShape`` subclass.
      ``k`` is the mathematical block size in rows, the term's rows are
      contiguous and ``k``-periodic, there are ``size // k`` mathematical
      blocks, and ``(start, end)`` index those blocks. For a norm block ``k``
      is the norm's group size and every metric must match it. For a
      quadratic block ``k`` is read from the declared metric (all sharing
      groups must agree), must divide the row count, must equal the dimension
      of a ``Quadratic``, and every metric is wrapped in a ``QuadraticShape``
      carrying the block's factorization.

    * The symmetric-cone carrier ``cone_metric.ConeAutomorphism``. The whole
      term is ONE plumbing block: ``k == size``, ``size // k == 1``, and
      ``ranges`` is exactly ``((0, 1, carrier),)``. The term's rows are NOT
      assumed periodic (the ``SOCBatch`` layout is not), so no consumer may
      reshape them by ``k`` for any purpose other than a whole-term view.
      Every cone-level fact is read from the carrier and never from ``k`` or
      from the range bounds: ``carrier.algebra``, ``carrier.k`` (the Jordan
      dimension of one cone), ``carrier.cones`` (how many cones the term
      stacks), ``carrier.count`` (how many elements are stored, 1 or
      ``cones``), ``carrier.sharing`` and ``carrier.layout``.
    """
    from . import cone_metric, spatial_soc
    if isinstance(declared, cone_metric.ConeAutomorphism):
        return cone_metric.normalize_cone_metric(declared, term_index, block, size)
    if isinstance(declared, spatial_soc.SpatialFullSOCMetric):
        return spatial_soc.normalize_spatial_full_metric(declared, term_index,
                                                         block, size)
    # The automorphism message is UNCHANGED, word for word. A stage that adds a
    # capability must not alter what an existing caller is told about an existing
    # mistake, so the new carrier gets its own sentence rather than a shared one.
    if isinstance(declared, (list, tuple)) and any(
            isinstance(item, cone_metric.ConeAutomorphism) for item in declared):
        raise ValueError(
            f"block_metrics[{term_index}]: a symmetric-cone term takes exactly ONE "
            "ConeAutomorphism, not a list of carriers. Per-cone variation is carried inside "
            "that one element array.")
    if isinstance(declared, (list, tuple)) and any(
            isinstance(item, spatial_soc.SpatialFullSOCMetric) for item in declared):
        raise ValueError(
            f"block_metrics[{term_index}]: a second-order cone term takes exactly ONE "
            "SpatialFullSOCMetric, not a list of carriers. Per-cone variation is carried "
            "inside that one carrier's (a, b, Q) arrays.")
    k = euclidean_norm_structure(block, size)
    quadratic = None if k is not None else quadratic_structure(block, size)
    if k is None and quadratic is None:
        hint = (" For a symmetric-cone block declare a ConeAutomorphism instead, or"
                " a SpatialFullSOCMetric on a second-order cone."
                if cone_metric.symmetric_cone_structure(block, size) is not None else "")
        raise ValueError(
            f"block_metrics[{term_index}]: {type(block).__name__} is not a supported "
            "rotation-invariant Euclidean-norm block (GroupL2, L2, or a positive scaling of "
            "either) and not a supported quadratic block (SumSquares, Quadratic, Zero, or a "
            "positive scaling of one). A block metric is only a valid proximal metric for those "
            "two families." + hint)
    metrics = [declared] if isinstance(declared, BlockMetric) else list(declared)
    if not metrics or not all(isinstance(item, BlockMetric) for item in metrics):
        raise ValueError(f"block_metrics[{term_index}] must be a BlockMetric or a list of BlockMetric")
    if quadratic is not None:
        sizes = sorted({item.k for item in metrics})
        if len(sizes) != 1:
            raise ValueError(f"block_metrics[{term_index}]: every sharing group of a quadratic "
                             f"block must declare the same metric block size, got {sizes}")
        k = sizes[0]
        if quadratic[0] == "quadratic" and k != size:
            raise ValueError(f"block_metrics[{term_index}]: a Quadratic block of dimension {size} "
                             f"accepts only a {size} x {size} shape, got k = {k}")
        if size % k:
            raise ValueError(f"block_metrics[{term_index}]: {size} rows is not a multiple of the "
                             f"declared metric block size {k}")
        metrics = [item if isinstance(item, QuadraticShape) else
                   QuadraticShape(item, block, size, term_index) for item in metrics]
    count = size // k
    ranges = []
    for item in metrics:
        if item.k != k:
            raise ValueError(f"block_metrics[{term_index}]: metric size {item.k} does not match "
                             f"the block's group size {k}")
        start, end = (0, count) if item.groups is None else item.groups
        ranges.append((start, end, item))
    ranges.sort(key=lambda r: r[0])
    cursor = 0
    for start, end, _ in ranges:
        if start != cursor or end > count:
            raise ValueError(f"block_metrics[{term_index}]: group ranges must partition "
                             f"[0, {count}) exactly, contiguously and without overlap")
        cursor = end
    if cursor != count:
        raise ValueError(f"block_metrics[{term_index}]: group ranges must cover [0, {count})")
    return k, tuple(ranges)


def check_static_steps(sigma, start, end, k, ranges, term_index):
    """Eager host check of the compiler's fixed dual steps on one declared term.

    Returns the ranges with a fourth, static field: True when every block of the
    range has an isotropic effective metric (S = w I and one step inside each
    block). The PDHG iteration only multiplies these steps by a positive scalar,
    so the flag stays true inside the loop, and those blocks use the closed-form
    radial prox instead of the Newton solve (same function, exact).

    A ``ConeAutomorphism`` range is checked by ``cone_metric.check_uniform_step``
    and always carries False in that fourth field: the flag describes a
    ``(Q, w)`` shape and would misdescribe the applied cone metric.
    """
    from . import cone_metric
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError("dual steps must be finite and positive")
        for a, b, metric in ranges:
            _check_hessian_ratio(metric, float(sigma), term_index, (a, b))
        # A cone automorphism has no isotropic closed-form flag: the flag is
        # a property of a (Q, w) shape and would misdescribe the applied metric.
        return tuple((a, b, metric, _isotropic_shape(metric)) for a, b, metric in ranges)
    rows = sigma[start:end].reshape(-1, k)
    if not np.all(np.isfinite(rows) & (rows > 0)):
        raise ValueError(f"block_metrics[{term_index}]: dual steps must be finite and positive")
    checked = []
    for a, b, metric in ranges:
        if isinstance(metric, cone_metric.ConeAutomorphism):
            # One step for the whole term (the SOCBatch layout is not k-periodic,
            # so no reshape by k is meaningful here), refused, never averaged.
            cone_metric.check_uniform_step(rows, metric, term_index, start, end)
            checked.append((a, b, metric, False))
            continue
        if isinstance(metric, _spatial_soc().SpatialFullSOCMetric):
            # The SAME requirement, for the same reason and one more: the step
            # cancels out of this Moreau update only because it is ONE positive
            # scalar over the term, so a term carrying several distinct steps is
            # refused rather than reduced to one of them.
            _check_uniform_spatial_step(rows, term_index, start, end)
            checked.append((a, b, metric, False))
            continue
        uniform_steps = bool(np.all(rows[a:b] == rows[a:b, :1]))
        pencil = isinstance(metric, QuadraticShape) and metric.quadratic_kind == "quadratic"
        if (metric.Q is not None or pencil) and not uniform_steps:
            what = ("a full (rotated) block metric" if metric.Q is not None else
                    "a Quadratic block's cached pencil factorization (Sigma = c S, one scalar c per block)")
            raise ValueError(
                f"block_metrics[{term_index}]: {what} requires the dual "
                f"step to be uniform inside every mathematical block; groups {a}..{b} are not. "
                "It is never averaged or collapsed silently.")
        if pencil:
            _check_hessian_ratio(metric, float(np.min(rows[a:b, 0])), term_index, (a, b))
        if metric.Q is None and not uniform_steps:
            # The EFFECTIVE metric of block i is diag(sigma_i) S, whose condition number the
            # declaration alone does not fix. The loop only rescales these steps by a positive
            # scalar, so this host check settles it for the whole solve.
            eff = rows[a:b] * metric.w_host
            cond = float(np.max(np.max(eff, axis=1) / np.min(eff, axis=1)))
            if not (cond <= MAX_DIAGONAL_COND):
                raise ValueError(_domain_message(
                    cond, MAX_DIAGONAL_COND, True,
                    what=f"block_metrics[{term_index}] groups {a}..{b}: the effective metric "
                         "diag(sigma_i) S"))
        checked.append((a, b, metric, uniform_steps and bool(np.all(metric.w_host == metric.w_host[0]))))
    return tuple(checked)


def _check_hessian_ratio(metric, c_min, term_index, groups):
    """The second axis of the Quadratic pencil domain, on the concrete compile-time steps.

    ``c_min`` is the smallest fixed dual step over the range's blocks (after
    the primal weight, before any runtime rescale). The ratio is recorded on
    the shape and refused above ``MAX_QUADRATIC_HESSIAN_RATIO``. A runtime
    primal-weight rescale (``adaptive_omega``) moves ``c`` after this check and
    is NOT covered by it.
    """
    if not (isinstance(metric, QuadraticShape) and metric.quadratic_kind == "quadratic"):
        return
    ratio = float(metric.hessian_norm / (c_min * float(np.min(metric.w_host))))
    metric.hessian_ratio = ratio if metric.hessian_ratio is None else max(metric.hessian_ratio, ratio)
    if not (ratio <= MAX_QUADRATIC_HESSIAN_RATIO):
        raise ValueError(_ratio_message(ratio, MAX_QUADRATIC_HESSIAN_RATIO, term_index, groups))


def _spatial_soc():
    """The primitive module, imported lazily to keep the import graph acyclic."""
    from . import spatial_soc
    return spatial_soc


def _check_uniform_spatial_step(sigma_rows, term_index, start, end):
    """The term's fixed dual step must be ONE value, refused otherwise.

    ``prox_{f^*}^{Sigma^{-1}}`` for ``Sigma = c M`` reduces to
    ``v - M P_M(M^{-1} v)`` only because ``c`` is one positive scalar over the
    term: ``P_{cM} = P_M`` and ``P_M`` is positively homogeneous.  With several
    distinct steps there is no single ``c`` to cancel, so the update this module
    dispatches would not be the requested prox.  It is refused, never averaged.
    """
    flat = np.asarray(sigma_rows, np.float64).reshape(-1)
    if not np.all(flat == flat[0]):
        raise ValueError(
            f"block_metrics[{term_index}]: a spatial-full cone metric requires ONE "
            f"dual step for the whole term, because the step cancels out of the "
            f"weighted Moreau update only when it is a single positive scalar "
            f"(P_{{cM}} = P_M and P_M is positively homogeneous). Rows "
            f"{start}:{end} carry {np.unique(flat).size} distinct values. The "
            "per-cone variation belongs in the supplied (a, b, Q), and the step "
            "is never averaged or collapsed silently.")


def _isotropic_shape(metric):
    """True for a (Q, w) shape with every w equal, never for a cone carrier.

    A cone carrier is excluded because the flag describes a (Q, w) SHAPE and
    would misdescribe the applied cone metric.  A ``SpatialFullSOCMetric`` also
    carries an attribute named ``Q``, its rotation, which is NOT a shape's Q, so
    it has to be excluded by TYPE here and not by attribute.
    """
    from . import cone_metric, spatial_soc
    if isinstance(metric, (cone_metric.ConeAutomorphism,
                           spatial_soc.SpatialFullSOCMetric)):
        return False
    return bool(np.all(metric.w_host == metric.w_host[0]))


def _identity_shape(metric):
    """Exactly the scalar-tier metric: S = I bitwise (norm shape) or every element e / I.

    For a ``SpatialFullSOCMetric`` this is ``a`` and every ``b_j`` bitwise equal,
    since ``diag(a, Q diag(b) Q^T) = a I`` exactly when ``b = a 1``, whatever
    ``Q`` is.  A declaration that reduces to the scalar tier must be recognised as
    such rather than routed through the weighted projection for no reason.
    """
    from . import cone_metric, spatial_soc
    if isinstance(metric, cone_metric.ConeAutomorphism):
        return metric.is_identity
    if isinstance(metric, spatial_soc.SpatialFullSOCMetric):
        a = np.asarray(metric.a, np.float64).reshape(-1)
        b = np.asarray(metric.b, np.float64).reshape(-1)
        return bool(np.all(a == 1.0)) and bool(np.all(b == 1.0))
    return metric.Q is None and bool(np.all(metric.w_host == 1.0))


def equivalent_to_scalar_tier(sigma, start, end, ranges):
    """True iff the declaration is EXACTLY the shipped scalar-tier metric.

    That is: every sharing group declares S = I (Q None, every w exactly 1) and
    the fixed dual step is exactly uniform over the whole term. The compiler
    then runs the term through the unmodified scalar-tier code, so endpoints are
    byte-identical to an undeclared compile. Nothing is rounded or averaged:
    any other declaration keeps the weighted path.
    """
    if not all(_identity_shape(entry[2]) for entry in ranges):
        return False
    sigma = np.asarray(sigma)
    if sigma.ndim == 0:
        return True
    span = sigma[start:end]
    return bool(span.size and np.all(span == span[0]))


# ------------------------------------------------------------- the weighted prox
def prox_norm_diagonal(u, d, lam, iters=NEWTON_ITERS):
    """argmin_z lam ||z||_2 + 1/2 sum_j d_j (z_j - u_j)^2, batched over rows.

    u, d: (B, k); lam: scalar >= 0. Zero iff ||d u||_2 <= lam. Otherwise
    z_j = d_j u_j r / (d_j r + lam) with r = ||z||_2 the root in (0, ||u||] of
    1/||q(r)||_2 = 1, q_j = d_j u_j / (d_j r + lam). Newton from r = 0 inside a
    bisection bracket, a fixed ``iters`` steps (exact after one step when d is
    uniform). Ported from the validated research routine.
    """
    lam = jnp.asarray(lam, u.dtype)
    positive = lam > 0
    lam = jnp.where(positive, lam, 1.0)                   # lam = 0: f = 0 and prox = u
    du = d * u
    active = jnp.sum(du * du, axis=1, keepdims=True) > lam * lam
    hi = jnp.sqrt(jnp.sum(u * u, axis=1, keepdims=True))
    lo = jnp.zeros_like(hi)
    r = jnp.zeros_like(hi)

    def body(_, state):
        lo, hi, r = state
        den = d * r + lam
        q = du / den
        n2 = jnp.sum(q * q, axis=1, keepdims=True)
        nrm = jnp.sqrt(n2)
        g = 1.0 / jnp.maximum(nrm, 1e-30) - 1.0
        gp = jnp.sum(q * q * d / den, axis=1, keepdims=True) / jnp.maximum(n2 * nrm, 1e-30)
        lo = jnp.where(g < 0, r, lo)
        hi = jnp.where(g < 0, hi, r)
        rn = r - g / jnp.maximum(gp, 1e-30)
        inside = (rn > lo) & (rn < hi)
        return lo, hi, jnp.where(inside, rn, 0.5 * (lo + hi))

    k = u.shape[1]
    if k <= _COMPONENT_MAX_K:
        return _prox_norm_diagonal_components(u, d, lam, positive, iters)
    lo, hi, r = jax.lax.fori_loop(0, iters, body, (lo, hi, r))
    z = du * r / (d * r + lam)
    return jnp.where(positive, jnp.where(active, z, 0.0), u)


_COMPONENT_MAX_K = 4


def _prox_norm_diagonal_components(u, d, lam, positive, iters):
    """The same safeguarded Newton iteration written over k separate (B,) arrays.

    For a small static k the per-row sums are explicit additions, so no
    reduction splits the fused elementwise kernels (the research routine's
    ``components`` form). Same equations, same fixed iteration count.
    """
    k = u.shape[1]
    us = [u[:, j] for j in range(k)]
    ds = [d[:, j] if d.shape[0] != 1 else jnp.broadcast_to(d[0, j], us[0].shape) for j in range(k)]
    dus = [dj * uj for dj, uj in zip(ds, us)]
    norm2 = dus[0] * dus[0]
    for j in range(1, k):
        norm2 = norm2 + dus[j] * dus[j]
    active = norm2 > lam * lam
    hi = us[0] * us[0]
    for j in range(1, k):
        hi = hi + us[j] * us[j]
    hi = jnp.sqrt(hi)
    lo = jnp.zeros_like(hi)
    r = jnp.zeros_like(hi)
    for _ in range(iters):
        dens = [dj * r + lam for dj in ds]
        qs = [duj / den for duj, den in zip(dus, dens)]
        n2 = qs[0] * qs[0]
        deriv = qs[0] * qs[0] * ds[0] / dens[0]
        for j in range(1, k):
            n2 = n2 + qs[j] * qs[j]
            deriv = deriv + qs[j] * qs[j] * ds[j] / dens[j]
        nrm = jnp.sqrt(n2)
        g = 1.0 / jnp.maximum(nrm, 1e-30) - 1.0
        gp = deriv / jnp.maximum(n2 * nrm, 1e-30)
        lo = jnp.where(g < 0, r, lo)
        hi = jnp.where(g < 0, hi, r)
        rn = r - g / jnp.maximum(gp, 1e-30)
        inside = (rn > lo) & (rn < hi)
        r = jnp.where(inside, rn, 0.5 * (lo + hi))
    z = jnp.stack([duj * r / (dj * r + lam) for duj, dj in zip(dus, ds)], axis=1)
    return jnp.where(positive, jnp.where(active[:, None], z, 0.0), u)


def _radial_dual_rows(v, d, lam):
    """Isotropic metric per row (d constant inside each row): the closed form.

    prox_f^{d I}(u) = max(0, 1 - lam / (d ||u||)) u, used inside the same
    weighted Moreau identity v - d prox_f^{d I}(v / d).
    """
    lam = jnp.asarray(lam, v.dtype)
    dr = d[:, :1]
    u = v / dr
    nrm = jnp.linalg.norm(u, axis=1, keepdims=True)
    z = jnp.maximum(0.0, 1.0 - lam / (dr * nrm + 1e-30)) * u
    return v - dr * z


def prox_norm_block(u, c, metric, lam):
    """prox_f^{Sigma}(u) per group, Sigma = diag(c) S; rows (B, k)."""
    if metric.Q is None:
        return prox_norm_diagonal(u, c * metric.w, lam)
    rotated = _mm(u, metric.Q)                            # rows Q^T u
    return _mm(prox_norm_diagonal(rotated, c * metric.w, lam), metric.Q.T)


def moreau_dual_rows(v, c, metric, lam, isotropic=False):
    """prox_{f*}^{Sigma^{-1}}(v) = v - Sigma prox_f^{Sigma}(Sigma^{-1} v), rows (B, k).

    Evaluated in the eigenbasis of S, where Sigma is the diagonal ``c * w``.
    ``isotropic`` (static, from ``check_static_steps``) selects the closed form.
    """
    d = c * metric.w
    if isotropic:
        return _radial_dual_rows(v, d, lam)             # Sigma_i = d_i I commutes with any Q
    if metric.Q is None:
        return v - d * prox_norm_diagonal(v / d, d, lam)
    vt = _mm(v, metric.Q)
    yt = vt - d * prox_norm_diagonal(vt / d, d, lam)
    return _mm(yt, metric.Q.T)


# ---------------------------------------------------------- the quadratic prox
def _shape_rows(rows, shape):
    """rows S, rows (B, k): S = diag(w) or Q diag(w) Q^T."""
    if shape.Q is None:
        return rows * shape.w
    return _mm(_mm(rows, shape.Q) * shape.w, shape.Q.T)


def _sigma_rows(rows, c, shape):
    """rows Sigma with Sigma = diag(c) S (S diagonal, c (B, k) or (B, 1)) or c S (S full, c (B, 1))."""
    if shape.Q is None:
        return rows * (c * shape.w)
    return c * _shape_rows(rows, shape)


def quadratic_solve_rows(r, c, shape):
    """z solving (H + Sigma) z = r per row, rows (B, k), using the cached factorization.

    ``Sigma = diag(c) S`` when ``S`` is diagonal (``c`` may vary inside a
    block for ``SumSquares`` and ``Zero``) and ``c S`` otherwise. For a
    ``Quadratic`` block ``c`` must be one scalar per row, (B, 1).
    """
    kind = shape.quadratic_kind
    if kind == "zero":                                   # H = 0
        if shape.Q is None:
            return r / (c * shape.w)
        return _mm(_mm(r, shape.Q) / (c * shape.w), shape.Q.T)
    if kind == "sumsquares":                             # H = a I commutes with S
        d = c * shape.w
        if shape.Q is None:
            return r / (shape.a + d)
        return _mm(_mm(r, shape.Q) / (shape.a + d), shape.Q.T)
    t = _mm(r, shape.G)                                  # G^T r, rows
    return _mm(t / (shape.lam + c), shape.G.T)           # G diag(1/(lam + c)) t


def prox_quadratic_rows(u, c, shape):
    """prox_f^{Sigma}(u) = argmin_z f(z) + 1/2 (z - u)^T Sigma (z - u), rows (B, k).

    Stationary at (H + Sigma) z = Sigma u - b. Exposed so a test can measure
    the KKT defect of the primal point directly. ``Zero`` returns ``u``.
    """
    if shape.quadratic_kind == "zero":
        return u
    r = _sigma_rows(u, c, shape)
    if shape.b is not None:
        r = r - shape.b
    return quadratic_solve_rows(r, c, shape)


def quadratic_moreau_rows(v, c, shape, return_primal=False):
    """prox_{f*}^{Sigma^{-1}}(v) = v - Sigma prox_f^{Sigma}(Sigma^{-1} v), rows (B, k).

    With u = Sigma^{-1} v the block solve is (H + Sigma) z = v - b and
    y = v - Sigma z. ``Sigma^{-1}`` is never formed on the loop path: the
    ``SumSquares`` and ``Zero`` shortcuts are closed forms. With
    ``return_primal`` the pair ``(y, z)`` is returned (test use only).
    """
    kind = shape.quadratic_kind
    if kind == "zero":
        y = jnp.zeros_like(v)
        return (y, quadratic_solve_rows(v, c, shape)) if return_primal else y
    if kind == "sumsquares":
        d = c * shape.w
        if shape.Q is None:
            y = shape.a * v / (shape.a + d)
        else:
            y = _mm(_mm(v, shape.Q) * (shape.a / (shape.a + d)), shape.Q.T)
        return (y, quadratic_solve_rows(v, c, shape)) if return_primal else y
    z = quadratic_solve_rows(v - shape.b, c, shape)
    y = v - _sigma_rows(z, c, shape)
    return (y, z) if return_primal else y


def metric_moreau(block, v, s, ranges, k):
    """Weighted conjugate prox of one (possibly packed) block over its groups."""
    quadratic = bool(ranges) and isinstance(ranges[0][2], QuadraticShape)
    lam = None if quadratic else _norm_weight(block)
    s = jnp.broadcast_to(s, v.shape)
    rows = v.reshape(-1, k)
    steps = s.reshape(-1, k)
    valid = jnp.all(jnp.isfinite(steps) & (steps > 0))
    parts = []
    for entry in ranges:
        start, end, metric = entry[:3]
        isotropic = len(entry) > 3 and entry[3]
        c = steps[start:end]
        if quadratic:
            if metric.Q is not None or metric.quadratic_kind == "quadratic":
                valid &= jnp.all(c == c[:, :1])
                c = c[:, :1]
            parts.append(quadratic_moreau_rows(rows[start:end], c, metric))
            continue
        if metric.Q is not None or isotropic:
            valid &= jnp.all(c == c[:, :1])
        parts.append(moreau_dual_rows(rows[start:end], c, metric, lam, isotropic=isotropic))
    out = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
    return jnp.where(valid, out.reshape(-1), jnp.nan)


def make_shape_apply(term_metrics, slices, m):
    """Return u -> S^power u on the stacked dual vector (power 1 or 0.5).

    Terms without a declared metric pass through unchanged. Applying S touches
    only the VECTOR, never the operator.

    This is the one function of the contract that is not restricted to dual
    terms: it accepts any mapping whose values satisfy the invariants of
    ``normalize_term_metrics``, for the primal space as well as the dual, with
    ``slices`` and ``m`` read in that space. The declared column shape of
    ``metric_columns=("full", k)`` is applied through exactly this function.
    """
    from . import cone_metric
    declared = sorted(term_metrics.items())

    def apply(u, power=1):
        pieces, cursor = [], 0
        for index, (k, ranges) in declared:
            start, end = slices[index]
            if start > cursor:
                pieces.append(u[cursor:start])
            if isinstance(ranges[0][2], cone_metric.ConeAutomorphism):
                # P(w)^power on the term's rows through the cone's own layout.
                pieces.append(cone_metric.apply_automorphism(ranges[0][2], u[start:end], power))
                cursor = end
                continue
            if isinstance(ranges[0][2], _spatial_soc().SpatialFullSOCMetric):
                # M^power on the term's rows, through the term's own layout.
                # M^{1/2} = diag(sqrt(a), Q diag(sqrt(b)) Q^T) is in the SAME
                # class, exactly, because Q is orthogonal, so the half power is
                # not an approximation of anything.  It has to be wired here or
                # the spectral safety check below would measure the identity for
                # this term and pass a step pair it should refuse.
                pieces.append(_spatial_soc().apply_spatial_full(
                    ranges[0][2], u[start:end], power))
                cursor = end
                continue
            rows = u[start:end].reshape(-1, k)
            out = []
            for a, b, metric, *_ in ranges:
                w = metric.w if power == 1 else metric.w_sqrt
                block_rows = rows[a:b]
                if metric.Q is None:
                    out.append(block_rows * w)
                else:
                    out.append(_mm(_mm(block_rows, metric.Q) * w, metric.Q.T))
            pieces.append((out[0] if len(out) == 1 else jnp.concatenate(out)).reshape(-1))
            cursor = end
        if cursor < m:
            pieces.append(u[cursor:m])
        return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)

    return apply
