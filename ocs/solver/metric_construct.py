"""Build the dual/primal metric the user DECLARES, from operator products only.

``compile_family(..., equilibrate="construct", metric_classes=...)`` lets a
caller declare, per dual term, *which class of metric it wants* -- scalar,
diagonal, full, or (for a symmetric-cone block only) automorphism -- and
*which mathematical blocks share one metric*. This
module builds that metric and hands the compiler steps it can apply. The caller
never supplies numbers.

The primal (column) side has the same vocabulary, declared with
``metric_columns``, and one contract, stated once and enforced everywhere:

    ``metric_columns="scalar"`` means ONE SCALAR STEP PER DECLARED PRIMAL
    PROXIMAL BLOCK, never one scalar for the whole primal space.

A primal block is what a metric is a metric for on that side too: the
coordinates the primal prox couples. Stating it per block is what makes the
three classes nest -- scalar inside diagonal inside full -- because that
containment only holds blockwise: a per-block scalar metric is the per-block
diagonal metric whose entries inside a block are tied. One scalar over the
whole primal space is a different, coarser restriction, and it is spelled as
one span, ``metric_columns=[(0, n)]``. See ``COLUMN_CONTRACT`` and
``normalize_columns``, which resolve a declared ``"scalar"`` against the
primal block structure and REFUSE it (rather than accept and quietly replace
it) when this release cannot apply it.

Three concepts are kept apart everywhere below, because they are different
groupings of the same rows:

* the **mathematical proximal block**: the ``k`` coordinates coupled by one
  norm (or one coordinate, for a separable block and for a MATRIX-class
  quadratic block, whose prox accepts a per-coordinate step). This is what a
  metric is a metric *for*, and it is what the natural declaration and the
  scalar class are resolved against.
* the **metric block**: the rows one stored shape acts on. It is the
  mathematical block for every norm block and for every scalar-class
  declaration. For a quadratic block (``SumSquares``, ``Quadratic``, ``Zero``
  and their positive scalings) under a SHARED shape, ``("full", "term")`` or
  ``("diagonal", "term")``, it is DECLARED: ``Quadratic`` takes its whole
  dimension, and the separable ``SumSquares`` and ``Zero`` take the third
  element of the declaration ``(cls, "term", k)`` or, when it is omitted, the
  whole term. Both records (``report["declared"]`` and
  ``meta["effective_metric"]``) carry the mathematical block and the metric
  block size under separate names.
* the **metric-sharing group**: the contiguous range of metric blocks that
  use ONE stored shape. ``"per_block"`` gives every block its own, ``"term"``
  gives the whole term one.
* the **batched execution group**: which blocks one compiled prox call
  evaluates together (``ocs.prox.groups``). It is chosen by the compiler for
  throughput and never merges norms or metrics.

Two regimes
-----------
REGIME "smfe" -- every term declares ``scalar`` or ``diagonal`` with
``per_block`` sharing and the columns take one of the two shipped forms. The
declaration is then exactly the constraint set of the shipped stochastic
matrix-free equilibration, and this module CALLS THAT SHIPPED PROGRAM
(``ocs.solver.equilibrate_sgd.smfe_equilibration``) unchanged. Nothing is
reimplemented, so the diagonal case is Diamond & Boyd's method not by
resemblance but by being the same code with the same seed; when the declaration
is the default (each term declaring the class its own proximal block natively
supports) the call is bit-for-bit the ``equilibrate="sgd"`` call.

REGIME "geodesic" -- any ``full`` class, any ``automorphism`` class, any
``term`` sharing, or a declared column tie that the shipped program cannot
express. Then the metric is built by
projected stochastic gradient descent in the affine-invariant (geodesic)
geometry of the positive definite cone, which is what lets a *matrix* iterate
stay positive definite under an exact exponential map rather than a retraction.

What the geodesic estimator is, and is not
------------------------------------------
It is projected SGD on the Diamond-Boyd equilibration energy, lifted from the
diagonal (log-scaling) parameterization to block matrices:

    minimize  1/2 E || A^1/2 K B^1/2 xi ||^2 pieces ... - r tr log A - s tr log B
              + (mu/4) ( ||log A||_F^2 + ||log B||_F^2 )
    subject to  exp(-2 box) I <= A, B <= exp(2 box) I   (spectral box)

State: one SPD matrix per row metric-sharing group (or one log-vector for a
diagonal/scalar group, or one Jordan element ``w`` per cone for an
automorphism group, see ``ocs.prox.cone_metric``) and one log-vector for the
columns, or, for a declared ``metric_columns=("full", k)``, one ``k x k`` SPD
matrix shared by every column block (the column analogue of a shared full row
shape, with the block size declared instead of read from a norm block;
accepted only when the family has no distinguished primal block).
For an automorphism group the objective is restricted to the totally
geodesic submanifold ``{P(w)}``: the Riemannian gradient is the Frobenius
projection of the ambient tangent onto that submanifold, computed in closed
form from the per-cone energy and first cross moment of the probe images
(never a ``k x k`` Gram), and the step ``w <- P(w^{1/2}) exp_J(-eta h)`` is the
ambient exponential map of the projected tangent, so every iterate is a
``P(w)`` by construction and nothing is projected from ambient SPD onto the
family. Probes: fresh
independent Rademacher vectors, ``probes`` per side per update, drawn in the
ambient space and never broadcast per tie. Estimator: the pooled sample Gram
``(1/Q) sum_q w_q w_q^T`` of the probe images, contracted directly into ``k by
k`` storage -- for a shared group the per-block Grams are summed, never
materialized. Geometry: affine invariant; the step is the exact exponential map
``A <- A^1/2 exp(-eta T) A^1/2`` with ``T`` the tangent, so no retraction error
and positive definiteness is structural. Schedule: ``eta = 2 / ((mu/4)(t +
shift))``, the strongly convex 1/t schedule with a shift. Projection: the exact
spectral box projection (eigenvalue clip); the exponent of a shared full step
is floored at ``-GEODESIC_EXP_FLOOR`` per eigen-direction so that float32
never produces a denormal step matrix (which the CUDA eigensolver turns into
NaN while the CPU flushes it to zero); overflow anywhere in the step
ABANDONS the step and keeps the previous state, and is reported, so an overflow
can never become a silently clipped answer. Averaging: pre-update geodesic
(weighted matrix-geometric) averaging with weight ``t + shift - 1``. Budget: a
fixed update count; there is NO residual test and NO stopping rule.

It is NOT a certified or rate-bearing estimator. No convergence rate is claimed
for it here: the averaged iterate is returned after a fixed budget, and the
bound that IS enforced is the downstream one -- the returned metric is rescaled
so the measured ``|| Sigma^1/2 K T^1/2 ||_2`` sits at ``safety < 1``, and the
compiler re-measures it and refuses the metric if it does not.

The one-shot Hutchinson estimator in ``ocs.solver.equilibrate`` and the
projected-SGD program in ``ocs.solver.equilibrate_sgd`` keep their own separate
documentation; neither acquires a rate from this module either.

Operator-only contract
----------------------
``K`` is reached exclusively through ``matvec`` and ``rmatvec``. The only dense
linear algebra is on ``k by k`` blocks (``k`` is the norm group size, e.g. 2 or
3): pooled local Gram estimates and symmetric eigendecompositions of those
small matrices. No entry, row, column or factorization of ``K`` is formed.
"""
from __future__ import annotations

import math
import numbers
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..prox import block_metric as _bm
from ..prox import cone_metric as _cm
from .equilibrate import _power_norm
from ._cone_construct_steps import soc_step, psd_bounded_step

CLASSES = ("scalar", "diagonal", "full", "automorphism", "spatial_full")
SHARING = ("per_block", "term")

# Automatic symmetric-cone construction uses stable float32 arithmetic.
# SOC evaluates the same projected step using scaled/log spectral quantities.
# PSD bounds the per-carrier exponent norm at .5; this is an explicit schedule
# change, recorded with the actual step history. Defaults remain scalar/diagonal.

# Defaults of the geodesic regime. They are the settings the research campaigns
# (profiling/shared_shape_20260915, profiling/gate1_ablation_20260915) ran, kept
# so a compiled result is comparable with the recorded measurements.
DEFAULT_UPDATES = 80
DEFAULT_PROBES = 2
DEFAULT_MU = 1.0
DEFAULT_BOX = 2.0
DEFAULT_SHIFT = 20
DEFAULT_SAFETY = 0.9
NORM_ITERS = 15
NORM_SEED = 7
# The largest block size a full column shape may declare, ``metric_columns=
# ("full", k)``. One k x k shape is stored once and eigendecomposed inside
# every constructor update (apply_power, update_side, average_side) and inside
# the step-condition measurement, so the block size is capped. Set from the
# per-iteration cost measurement of the receipt that introduced it.
MAX_COLUMN_FULL_K = 64
# The exponential map of a shared full step, ``A^1/2 exp(-eta T) A^1/2``, is
# evaluated by a float32 spectral exp. Every eigen-direction of ``-eta T`` is
# floored at ``-GEODESIC_EXP_FLOOR`` before the exp. Why: below about -87
# float32 exp is zero, and between about -87 and -103 it is a DENORMAL, which
# the CPU backend flushes to zero and the CUDA backend keeps; a step matrix
# with denormal-scale entries makes cuSOLVER's symmetric eigensolver return
# NaN (measured on the RTX 5090, JAX 0.11.1: eigh of a denormal 2 x 2 gives
# NaN on cuda and (0, I) on cpu), so the same declaration built a metric on
# the CPU and failed with status 2 on the GPU. Every direction below the floor
# ends far under the spectral box floor exp(-2 box) (for box <= 13 the product
# with A^1/2 stays a normal float32) and is projected to exactly that floor
# whatever its value, so the floor changes no projected iterate beyond float32
# rounding: it only keeps every intermediate a normal number on both backends.
# The positive side is NOT capped: an overflow still abandons the step, keeps
# the previous state and is reported as status 2, never as a clipped answer.
GEODESIC_EXP_FLOOR = 60.0


# --------------------------------------------------------------- declarations
class TermDeclaration:
    """One dual term's declared metric class, sharing, and block structure."""

    __slots__ = ("index", "cls", "sharing", "k", "count", "start", "end",
                 "scaling_class", "block_name", "norm_block", "math_k", "k_source",
                 "quadratic", "cone")

    def __init__(self, index, cls, sharing, k, count, start, end,
                 scaling_class, block_name, norm_block, math_k=None,
                 k_source="mathematical block", quadratic=None, cone=None):
        self.index, self.cls, self.sharing = index, cls, sharing
        self.k, self.count = k, count                    # the METRIC block: size and number
        self.start, self.end = start, end
        self.scaling_class = scaling_class
        self.block_name = block_name
        self.norm_block = norm_block
        self.math_k = k if math_k is None else math_k    # the MATHEMATICAL block size
        self.k_source = k_source
        self.quadratic = quadratic                       # None, or the quadratic kind
        # Only for an "automorphism" declaration: {"algebra", "k", "cones",
        # "layout", "jordan_rank", "degrees_of_freedom"}. None otherwise, so the
        # record of every pre-existing declaration is unchanged.
        self.cone = cone

    @property
    def size(self):
        return self.end - self.start

    def describe(self):
        out = {"term": self.index, "block": self.block_name,
               "declared_class": self.cls, "metric_sharing": self.sharing,
               "mathematical_block_size_k": self.math_k,
               "mathematical_blocks": self.size // self.math_k,
               "metric_block_size_k": self.k,
               "metric_blocks": self.count,
               "metric_block_size_source": self.k_source,
               "metric_sharing_groups": 1 if self.sharing == "term" else self.count,
               "rows": [self.start, self.end],
               "block_scaling_class": self.scaling_class,
               "quadratic_block": self.quadratic}
        if self.cone is not None:
            out["cone"] = dict(self.cone)
            out["degrees_of_freedom"] = int(self.cone["degrees_of_freedom"]) * (
                1 if self.sharing == "term" else self.count)
            out["class_chain"] = ("scalar (1 dof) subset spatial-full diag(a, Q diag(b) Q^T) subset SPD; "
                                  "disjoint from the automorphism class except at scalar members"
                                  ) if self.cone.get("class") == "spatial_full" else ("scalar (1 dof) subset automorphism {P(w)} subset SPD (no closed "
                                  "form); diagonal is outside except its scalar members")
        return out

    def __repr__(self):
        return (f"TermDeclaration(term={self.index}, class={self.cls}, "
                f"sharing={self.sharing}, k={self.k}, blocks={self.count})")


def _parse_entry(value, index):
    """``cls``, ``(cls, sharing)`` or ``(cls, sharing, k)``; returns (cls, sharing, k or None)."""
    k_declared = None
    if isinstance(value, str):
        cls, sharing = value, None
    else:
        try:
            items = tuple(value)
        except TypeError:
            items = ()
        if len(items) == 2:
            cls, sharing = items
        elif len(items) == 3:
            cls, sharing, k_declared = items
        else:
            raise ValueError(
                f"metric_classes[{index}] must be a class name, a (class, sharing) pair or a "
                f"(class, sharing, k) triple, got {value!r}")
    if cls not in CLASSES:
        raise ValueError(f"metric_classes[{index}]: class must be one of {CLASSES}, got {cls!r}")
    if sharing is None:
        # A bare "automorphism" is one Jordan element per cone (the carrier is
        # batched by construction and the NT point is per cone).
        sharing = "term" if cls in ("full", "spatial_full") else "per_block"
    if sharing not in SHARING:
        raise ValueError(f"metric_classes[{index}]: sharing must be one of {SHARING}, got {sharing!r}")
    if k_declared is not None:
        if (isinstance(k_declared, bool) or not isinstance(k_declared, numbers.Integral)
                or k_declared <= 0):
            raise ValueError(f"metric_classes[{index}]: the metric block size must be a positive "
                             f"integer, got {k_declared!r}")
        k_declared = int(k_declared)
    return cls, sharing, k_declared


def normalize_declarations(metric_classes, blocks, slices):
    """Validate the declaration and resolve each term's block structure.

    ``metric_classes`` is None (every term declares the class its own proximal
    block natively supports) or ``{term_index: class}`` / ``{term_index:
    (class, sharing)}``; terms left out keep the natural declaration.

    The NATURAL declaration is ``("scalar", "term")`` for a scalar-class block
    -- one step for the whole term, which is exactly what the shipped path
    gives it today -- and ``("diagonal", "per_block")`` for every other block,
    whose mathematical blocks are single coordinates.
    """
    from ..prox.base import SCALAR
    if metric_classes is not None and not isinstance(metric_classes, dict):
        raise ValueError("metric_classes must be None or a dict {dual_term_index: class}")
    given = dict(metric_classes or {})
    for index in given:
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < len(blocks):
            raise ValueError(f"metric_classes key {index!r} is not a dual term index "
                             f"in [0, {len(blocks)})")
    declarations = []
    for index, (block, (start, end)) in enumerate(zip(blocks, slices)):
        size = end - start
        name = type(block).__name__
        scaling_class = block.scaling_class
        k_norm = _bm.euclidean_norm_structure(block, size)
        quadratic = None if k_norm is not None else _bm.quadratic_structure(block, size)
        quadratic_kind = None if quadratic is None else quadratic[0]
        cone = _cm.symmetric_cone_structure(block, size)
        coupled = scaling_class == SCALAR
        natural = ("scalar", "term") if coupled else ("diagonal", "per_block")
        if int(index) in given:
            cls, sharing, k_declared = _parse_entry(given[int(index)], index)
        else:
            (cls, sharing), k_declared = natural, None
        # The mathematical proximal block: the coordinates one prox couples, or
        # one coordinate for a block whose prox accepts a per-coordinate step.
        # This is what the natural declaration and the scalar class resolve
        # against, and it is UNCHANGED by anything declared below.
        if k_norm is not None:
            math_k = k_norm
        elif coupled:
            math_k = size                  # a coupled block is one whole block
        else:
            math_k = 1                     # separable, or any-metric: one coordinate per block
        cone_record = None
        if cls == "automorphism":
            if type(block) in _NONSYMMETRIC_CONES:
                raise ValueError(
                    f"metric_classes[{index}]: {name} is not a symmetric cone. The "
                    "automorphism tier is the closed-form weighted projection W^{-1} Pi_C(W v) for "
                    "W in the automorphism group of C, which exists for the self-scaled (symmetric) "
                    "cones only (Nesterov and Todd 1997, Hauser and Gueler 2002). The exponential "
                    "and power cones have no self-scaled barrier and no such family of positive "
                    "definite scalings, so this release keeps them scalar. Declare 'scalar'.")
            if cone is None:
                raise ValueError(
                    f"metric_classes[{index}]: 'automorphism' is a cone metric, defined only for a "
                    "symmetric-cone block of exact type SecondOrderCone, SOCBatch or PSDCone; "
                    f"{name} over {size} rows is not one. Declare 'scalar', "
                    "'diagonal' or 'full' according to the block's class.")
            if k_declared is not None:
                raise ValueError(
                    f"metric_classes[{index}]: the automorphism tier takes no metric block size. "
                    f"{name} fixes it at the cone's Jordan dimension. Omit the third element.")
            algebra, k_cone, cones = cone
            cone_record = {"algebra": algebra, "k": int(k_cone), "cones": int(cones),
                           "layout": _cm.cone_layout(block),
                           "jordan_rank": _cm.jordan_rank(algebra, k_cone),
                           "degrees_of_freedom": _cm.degrees_of_freedom(algebra, k_cone)}
        if cls == "spatial_full":
            # The Lorentz cone's spatial-full class, M = diag(a, Q diag(b) Q^T),
            # constructed from products by ocs.prox.spatial_construct and applied
            # by the weighted projection of ocs.prox.spatial_soc. This release
            # constructs it for ONE second-order cone per term; a stacked
            # SOCBatch term and the matrix cone are refused by name, and a
            # per-cone construction stays on the supplied block_metrics path.
            from ..prox import cones as _cones
            if cone is None or cone[0] != "soc":
                raise ValueError(
                    f"metric_classes[{index}]: 'spatial_full' is the second-order cone's "
                    f"structured full class and {name} over {size} rows is not a second-order "
                    "cone block. Declare 'scalar', 'diagonal' or 'full' according to the "
                    "block's class.")
            if type(block) is not _cones.SecondOrderCone:
                raise ValueError(
                    f"metric_classes[{index}]: this release constructs 'spatial_full' for one "
                    f"SecondOrderCone per term; {name} stacks {cone[2]} cones. Build the carrier "
                    "with ocs.prox.spatial_construct.construct_spatial_full and supply it through "
                    "block_metrics=, or declare 'scalar'.")
            if k_declared is not None:
                raise ValueError(
                    f"metric_classes[{index}]: 'spatial_full' takes no metric block size; the "
                    "cone fixes it. Omit the third element.")
            if sharing != "term":
                raise ValueError(
                    f"metric_classes[{index}]: 'spatial_full' on one cone is one metric for the "
                    "term. Declare ('spatial_full', 'term') or the bare name.")
            algebra, k_cone, cones_count = cone
            cone_record = {"algebra": algebra, "k": int(k_cone), "cones": int(cones_count),
                           "layout": _cm.cone_layout(block), "class": "spatial_full",
                           "jordan_rank": _cm.jordan_rank(algebra, k_cone),
                           "degrees_of_freedom": int(k_cone) * (int(k_cone) + 1) // 2}
        if cls == "diagonal" and cone is not None:
            if cone[0] == "soc":
                raise ValueError(
                    f"metric_classes[{index}]: {name} admits no diagonal metric in "
                    "closed form except a scalar one: a positive diagonal D is an automorphism of "
                    "the second-order cone only when D J D = mu J, which forces every diagonal "
                    "entry to be equal, so a diagonal weighted projection has no closed form on "
                    "this block. Declare 'scalar' or 'automorphism'.")
            raise ValueError(
                f"metric_classes[{index}]: PSDCone admits no free diagonal metric in closed form: "
                "an arbitrary positive diagonal on the k*k entries is not a congruence of the "
                "cone, so its weighted projection has no closed form. The diagonal congruences "
                "Z -> D Z D with D = diag(s) (entries s_i s_j, k degrees of freedom) are cone "
                "automorphisms and are members of the automorphism class, not of the free "
                "diagonal class. Declare 'scalar' or 'automorphism'.")
        # A shared shape needs a block it is a valid proximal metric for.
        shared_shape = cls == "full" or (cls == "diagonal" and sharing == "term")
        if cls == "full" and k_norm is None and quadratic_kind is None:
            raise ValueError(
                f"metric_classes[{index}]: a full (rotated) metric is only a valid proximal "
                "metric for a rotation-invariant Euclidean-norm block with group size > 1 or "
                "for a quadratic block (SumSquares, Quadratic, Zero, or a positive scaling of "
                f"one). {name} over {size} rows is neither. Declare 'scalar' or 'diagonal' "
                "for this term." + (
                    " For this symmetric-cone block the structured full class is "
                    "'automorphism'. Declare that." if cone is not None else ""))
        if cls == "diagonal" and sharing == "term" and k_norm is None and quadratic_kind is None:
            raise ValueError(
                f"metric_classes[{index}]: a SHARED diagonal shape needs a rotation-invariant "
                "Euclidean-norm block whose group size defines the shape, or a quadratic block "
                "(SumSquares, Quadratic, Zero, or a positive scaling of one) whose metric block "
                f"size is declared. {name} over {size} rows is neither. Use ('diagonal', 'per_block').")
        # The metric block: the rows one stored shape acts on. It is the
        # mathematical block except for a quadratic block under a SHARED shape,
        # where the block's function does not fix it and it is declared.
        k, k_source = math_k, "mathematical block"
        if quadratic_kind is not None and shared_shape:
            if quadratic_kind == "quadratic":
                k, k_source = size, "the quadratic's dimension"
                if k_declared is not None and k_declared != size:
                    raise ValueError(
                        f"metric_classes[{index}]: a metric block size can be declared only for "
                        "a separable quadratic block (SumSquares, Zero, or a positive scaling of "
                        f"one), whose function couples no coordinates. {name} fixes its block "
                        f"size at {size} (from the quadratic's dimension). Omit the third element.")
            elif k_declared is not None:
                k, k_source = k_declared, "declared"
            else:
                k, k_source = size, "whole term (no metric block size declared)"
        elif k_declared is not None:
            if quadratic_kind is not None:
                raise ValueError(
                    f"metric_classes[{index}]: a metric block size applies only to a shared shape, "
                    f"('full', 'term') or ('diagonal', 'term'). Under ({cls!r}, {sharing!r}) the "
                    f"metric blocks of {name} are its mathematical blocks, one coordinate each. "
                    "Omit the third element.")
            if k_norm is not None:
                fixed, source = k_norm, "the norm's group size"
            elif coupled:
                fixed, source = size, "the whole coupled block"
            else:
                fixed, source = 1, "one coordinate per block"
            raise ValueError(
                f"metric_classes[{index}]: a metric block size can be declared only for a "
                "separable quadratic block (SumSquares, Zero, or a positive scaling of one), "
                f"whose function couples no coordinates. {name} fixes its block size at {fixed} "
                f"(from {source}). Omit the third element.")
        if cls == "full":
            if k == 1:
                raise ValueError(
                    f"metric_classes[{index}]: a full (rotated) metric is only a valid proximal "
                    "metric for a rotation-invariant Euclidean-norm block with group size > 1 or "
                    f"for a quadratic block with metric block size > 1. {name} over {size} rows "
                    f"resolves to metric blocks of size 1 ({k_source}), on which a full metric is "
                    "a scalar. Declare 'scalar' or 'diagonal' for this term.")
            if sharing != "term":
                if size // k == 1:
                    raise ValueError(
                        f"metric_classes[{index}]: a full metric must be shared ('term'). This "
                        f"term is exactly one metric block of {k} rows, so one shape per block "
                        "and one shape per term name the same metric, and this release accepts "
                        "only the ('full', 'term') spelling for it. Declare ('full', 'term').")
                raise ValueError(
                    f"metric_classes[{index}]: a full metric must be shared ('term'). One "
                    f"independent {k}x{k} shape for each of {size // k} metric blocks is "
                    "neither built nor applied by this release: the application layer stores one "
                    "(Q, w) pair per metric-sharing group, and a per-block full metric would need "
                    f"{size // k} of them. Declare ('full', 'term').")
        if cls == "diagonal" and coupled and k_norm is None:
            raise ValueError(
                f"metric_classes[{index}]: {name} is a coupled (scalar-class) "
                "block that is not a rotation-invariant Euclidean-norm block, so a diagonal "
                "metric is not a valid proximal metric for it. Declare 'scalar'.")
        if cls == "automorphism":
            # The metric block is ONE cone: its Jordan dimension and how many of
            # them the term stacks. The structure resolver already checked both
            # against the row count. The mathematical block is one cone's rows,
            # so size // math_k is the cone count for SOC and PSD alike.
            k, count = cone_record["k"], cone_record["cones"]
            math_k, k_source = size // count, "the cone's Jordan dimension"
        else:
            count = size // k
            if k * count != size:
                raise ValueError(f"metric_classes[{index}]: {size} rows is not a multiple of the "
                                 f"metric block size {k} ({k_source})")
        # Canonicalize declarations that name the same metric two ways, so the
        # regime test below sees one spelling: a scalar on single-coordinate
        # blocks IS a free diagonal, and a scalar on a term with one block IS
        # that term's shared scalar. (Both rules see the mathematical block:
        # the scalar class never takes a declared metric block.)
        if cls == "scalar" and k == 1 and count > 1:
            cls, sharing = "diagonal", "per_block"
        if cls == "scalar" and count == 1:
            sharing = "term"
        if cls == "automorphism" and count == 1:
            sharing = "per_block"          # one cone: the two sharings coincide
        declarations.append(TermDeclaration(index, cls, sharing, k, count, start, end,
                                            scaling_class, name, k_norm is not None,
                                            math_k=math_k, k_source=k_source,
                                            quadratic=quadratic_kind, cone=cone_record))
    return tuple(declarations)


def _nonsymmetric_cones():
    from ..prox import cones
    return (cones.ExpCone, cones.ExpConeBatch, cones.PowerCone, cones.PowerConeBatch)


class _LazyTuple:
    """``type(block) in _NONSYMMETRIC_CONES`` without importing cones at module load."""

    def __contains__(self, item):
        return item in _nonsymmetric_cones()


_NONSYMMETRIC_CONES = _LazyTuple()


COLUMN_CONTRACT = (
    "metric_columns='scalar' declares ONE SCALAR STEP PER DECLARED PRIMAL "
    "PROXIMAL BLOCK. It does NOT mean one scalar for the whole primal space. "
    "A primal block is the set of coordinates the primal prox couples -- the "
    "same definition the row side uses for a mathematical proximal block -- so "
    "scalar, diagonal and full are nested BLOCKWISE: a per-block scalar metric "
    "is the special case of a per-block diagonal metric whose entries inside a "
    "block are tied, and a per-block diagonal metric is the special case of a "
    "per-block full metric that is diagonal. One scalar over the WHOLE primal "
    "space is a different, coarser restriction and is spelled as one span: "
    "metric_columns=[(0, n)]. A full column shape is declared with a block size, "
    "metric_columns=('full', k), and is one k x k shape shared by every block of k "
    "consecutive coordinates, accepted only when the family has no distinguished "
    "primal block.")

COLUMN_SPELLINGS = (
    "metric_columns must be None, 'scalar', 'diagonal', a sequence of (start, end) "
    "column spans, or ('full', k) for one shared k x k column shape.")


def primal_block_structure(primal_block, n):
    """The primal prox's mathematical blocks: ``(k, count, name)``.

    Exactly the definition ``normalize_declarations`` uses on the row side: a
    rotation-invariant Euclidean-norm block gives its group size; any other
    coupled (scalar-class) block is one whole block; a separable block is one
    coordinate per block. ``None`` (no primal proximal term, ``g = 0``) is
    separable: every coordinate is its own block.
    """
    from ..prox.base import SCALAR
    if primal_block is None:
        return 1, int(n), "none"
    k_norm = _bm.euclidean_norm_structure(primal_block, n)
    if k_norm is not None:
        k = int(k_norm)
    elif primal_block.scaling_class == SCALAR:
        k = int(n)                       # a coupled block is one whole block
    else:
        k = 1                            # separable: one coordinate per block
    k = max(1, k)
    return k, int(n) // k, type(primal_block).__name__


def normalize_columns(metric_columns, n, primal_scalar_metric, primal_block=None):
    """Resolve the declared primal (column) metric structure.

    Returns ``(kind, spans, record)``. ``kind`` is the routing kind --
    ``"diagonal"`` (a free per-coordinate diagonal), ``"scalar"`` (one step
    over the whole primal space) or ``"spans"`` -- and ``record`` is the public
    report of what was declared, what it resolves to, and how many metric
    sharing groups the applied primal metric is allowed to have.

    ``None``   the shipped form: one tied step when the primal block is
               scalar class, otherwise a free per-coordinate diagonal. This is
               NOT a declaration and its bytes are the shipped program's.
    ``"scalar"``   one scalar step per declared primal proximal block
               (``COLUMN_CONTRACT``). With one coordinate per block that class
               IS the free per-coordinate diagonal -- the same feasible set,
               not a reduction of one -- and it routes there. With the whole
               primal space in one block it is one tied step, built with the
               tie in place. With several coupled blocks it is REFUSED: this
               release applies exactly one primal step to a coupled primal
               prox, so that declaration can be neither built nor applied.
    ``"diagonal"``   a free per-coordinate diagonal, named explicitly.
    a sequence of (start, end) spans   each span shares one step; the spans
               must partition [0, n) contiguously. Not expressible by the
               shipped program, so it selects the geodesic regime.
    ``("full", k)`` or ``("full", k, "term")``   ONE shared k x k symmetric
               positive definite shape S over the n / k contiguous column
               blocks [i k, (i+1) k), so the applied primal step operator is
               T = c (I kron S) with c one scalar carrier. Accepted ONLY when
               the family has no distinguished primal proximal block, because
               there the primal update is the gradient step x - T (K^T y + c)
               and a full T asks only to be applied to a vector. Every
               distinguished-block family is REFUSED with the reason for its
               block's class. The caller's variable ordering decides what a
               block is: nothing permutes coordinates. Geodesic regime.
    """
    k, count, name = primal_block_structure(primal_block, n)
    blocks = {"block": name, "k": int(k), "count": int(count)}

    def record(kind, spans, declared, applied, groups, **extra):
        # ``extra`` is present ONLY for the full kind, so the record of every
        # other declaration (the default included) keeps its exact keys.
        return kind, spans, {"kind": kind, "spans": spans,
                             "metric_sharing_groups": int(groups),
                             "declared": declared, "applied": applied,
                             "primal_blocks": blocks, **extra,
                             "contract": COLUMN_CONTRACT}

    if metric_columns is None:
        if primal_scalar_metric:
            return record("scalar", None, None,
                          "the shipped default: one scalar step over the whole primal space, "
                          "which is the only primal metric a coupled (scalar-class) primal "
                          "prox accepts in this release", 1)
        return record("diagonal", None, None,
                      "the shipped default: a free per-coordinate diagonal", n)
    if isinstance(metric_columns, str):
        if metric_columns == "full":
            raise ValueError(
                COLUMN_SPELLINGS + " metric_columns='full' without a block size is refused: "
                "with no primal proximal block every primal coordinate is its own block, so a "
                "'full' metric of block size 1 would be the free per-coordinate diagonal under "
                "a wider name. Declare ('full', k) with the block size k that the coordinates "
                "to be mixed occupy contiguously.")
        if metric_columns not in ("scalar", "diagonal"):
            raise ValueError(COLUMN_SPELLINGS)
        if metric_columns == "diagonal":
            if primal_scalar_metric:
                raise ValueError("metric_columns='diagonal' is not admissible: the primal "
                                 "proximal block is scalar class and accepts exactly one step. "
                                 "Use 'scalar' or a span declaration whose spans the block can "
                                 "honour.")
            return record("diagonal", None, "diagonal",
                          "a free per-coordinate diagonal: one step per primal coordinate", n)
        # "scalar": one scalar per DECLARED PRIMAL BLOCK. See COLUMN_CONTRACT.
        if k == 1:
            # Each primal block is a single coordinate, so the per-block scalar
            # class and the per-coordinate diagonal class are the SAME set of
            # metrics. Routing there is an identity of feasible sets, not a
            # collapse of a free solution onto a coarser class.
            return record("diagonal", None, "scalar",
                          f"one scalar per primal proximal block; the primal block ({name}) "
                          f"couples one coordinate per block, so the declared class is exactly "
                          f"the free per-coordinate diagonal on {count} blocks -- the same "
                          f"feasible set, not a reduction of one", count)
        if count == 1:
            return record("scalar", None, "scalar",
                          f"one scalar per primal proximal block; the primal block ({name}) "
                          f"couples all {n} primal coordinates into ONE block, so the declared "
                          f"class is one scalar over the whole primal space, and it is "
                          f"constructed with that constraint in place", 1)
        raise ValueError(
            f"metric_columns='scalar' declares one scalar primal step per primal proximal "
            f"block, and this family's primal block ({name}) has {count} mathematical blocks "
            f"of {k} coordinates. This release applies EXACTLY ONE primal step to a coupled "
            f"(scalar-class) primal block: the compiled loop calls g.prox(x, tau) once over "
            f"the whole primal vector and that prox requires a uniform step, so {count} "
            f"distinct primal steps are neither built nor applied here. The declaration is "
            f"REFUSED rather than accepted and quietly replaced by one step. Declare "
            f"metric_columns=[(0, {n})] for one scalar over the whole primal space.")
    try:
        first = metric_columns[0] if len(metric_columns) else None
    except TypeError:
        first = None
    if isinstance(first, str):
        # A class declaration, ('full', k[, sharing]), on any sequence type
        # (tuple or list). Resolved here, before any probe is drawn.
        return _normalize_full_columns(metric_columns, n, primal_block, k, count, name,
                                       blocks, record)
    spans = tuple((int(a), int(b)) for a, b in metric_columns)
    if not spans:
        raise ValueError("metric_columns spans must be nonempty")
    cursor = 0
    for a, b in spans:
        if a != cursor or b <= a:
            raise ValueError(f"metric_columns spans must partition [0, {n}) contiguously, "
                             f"in order and without overlap; got {spans}")
        cursor = b
    if cursor != n:
        raise ValueError(f"metric_columns spans must cover [0, {n}); they cover [0, {cursor})")
    if primal_scalar_metric and len(spans) > 1:
        raise ValueError("the primal proximal block is scalar class and accepts exactly one "
                         "step, so the column metric cannot be split into "
                         f"{len(spans)} tied spans")
    return record("spans", spans, "spans",
                  f"{len(spans)} declared tied column spans, one step each", len(spans))


def _full_column_refusal(primal_block, k, name, cls, count, k0):
    """Refusal 3: the class-specific reason a distinguished block cannot take T."""
    from ..prox.base import SCALAR, DIAGONAL, MATRIX
    from ..prox.cones import (SecondOrderCone, SOCBatch, PSDCone, ExpCone, ExpConeBatch,
                              PowerCone, PowerConeBatch)
    from ..prox.library import Zero
    core = (
        f"metric_columns=('full', {k}) is REFUSED on this family: it has a distinguished "
        f"primal proximal block ({name}, scaling class '{cls}', {count} mathematical block(s) "
        f"of {k0} coordinates), and the compiled loop evaluates that block as g.prox(x, tau) "
        "with a scalar or per-coordinate step. No primal proximal block is routed a full "
        "(non-diagonal) metric by this release, so the declaration can be neither built nor "
        "applied. A full column shape is accepted only when no distinguished primal block "
        "exists, where the primal update is the gradient step x - T (K^T y + c). Declare "
        "'scalar', 'diagonal' or (start, end) column spans for this family.")
    if type(primal_block) in (SecondOrderCone, SOCBatch, PSDCone):
        why = (f"{name} is a symmetric cone: the closed-form metric class that applies to it "
               "is the automorphism tier (SPEC Item 2, metric_classes 'automorphism' on the "
               "dual side), and its application to a PRIMAL cone block is NOT built in this "
               "release.")
    elif type(primal_block) in (ExpCone, ExpConeBatch, PowerCone, PowerConeBatch):
        why = (f"{name} is not a symmetric cone and admits no closed-form weighted projection "
               "beyond the scalar class.")
    elif cls == SCALAR:
        why = f"{name} accepts exactly one primal step."
    elif cls == DIAGONAL:
        why = (f"{name} accepts a per-coordinate step and has no weighted prox for a "
               "non-diagonal metric.")
    else:
        why = (f"{name} defines prox_matrix, which the compiler does not call on the primal "
               "side in this release (routing it is SPEC Item 1 territory on the row side and "
               "is not part of this change).")
        if isinstance(primal_block, Zero):
            why += (" Zero is evaluated as a proximal block by the compiled loop "
                    "(ocs/prox/library.py Zero.prox) even though the certificate treats it as "
                    "no block, so it is refused with the others. Route the term through an "
                    "operator to remove the distinguished block.")
    assert cls in (SCALAR, DIAGONAL, MATRIX), cls
    return core + " " + why


def _normalize_full_columns(value, n, primal_block, k0, count0, name, blocks, record):
    """Resolve ``("full", k[, sharing])``: refusals 2 to 7 of the design, in order.

    The structural reason (a distinguished primal block) is checked first, so it
    is the reason a caller reads first; the block size is checked after it.
    """
    malformed = ValueError(
        f"metric_columns={value!r} is not a column declaration. A full column shape is "
        "declared as ('full', k) or ('full', k, 'term') with an integer block size k. Tied "
        "spans are declared as a sequence of (start, end) pairs. See COLUMN_CONTRACT.")
    items = tuple(value)
    if items[0] != "full" or len(items) not in (2, 3):
        raise malformed
    k = items[1]
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
        raise malformed
    k = int(k)
    sharing = items[2] if len(items) == 3 else "term"
    if sharing not in SHARING:
        raise malformed
    if primal_block is not None:
        raise ValueError(_full_column_refusal(primal_block, k, name,
                                              primal_block.scaling_class, count0, k0))
    if sharing != "term":
        raise ValueError(
            f"metric_columns=('full', {k}, 'per_block'): a full column shape must be shared "
            f"('term'). One independent {k}x{k} shape for each of {n // k} column blocks is "
            "neither built nor applied by this release: the application layer stores one "
            "(Q, w) pair per metric-sharing group and the constructor's geodesic state has no "
            f"batched per-block full kind. Declare ('full', {k}) or ('full', {k}, 'term').")
    if k < 2:
        raise ValueError(
            f"metric_columns=('full', {k}) is the free per-coordinate diagonal under a wider "
            "name and is refused. Declare 'diagonal', or a block size of at least 2.")
    if k > MAX_COLUMN_FULL_K:
        raise ValueError(
            f"metric_columns=('full', {k}): the block size exceeds MAX_COLUMN_FULL_K = "
            f"{MAX_COLUMN_FULL_K}. A full column shape is stored once as a {k}x{k} eigenpair "
            "and eigendecomposed inside every constructor update and inside the step-condition "
            "measurement, so the block size is capped. Declare a smaller block, or tied "
            "(start, end) spans.")
    if n % k != 0:
        raise ValueError(
            f"metric_columns=('full', {k}): {n} primal coordinates is not a multiple of the "
            f"block size {k}. The column blocks are the contiguous groups [i k, (i+1) k) and "
            f"must tile [0, {n}).")
    count = n // k
    return record("full", None, ("full", k),
                  "one shared k x k SPD shape over n/k contiguous column blocks, "
                  "T = c (I kron S), c one scalar carrier", 1,
                  column_blocks={"k": k, "count": count, "sharing": "term"})


def shipped_expressible(declarations, column_kind):
    """Is the declaration exactly a constraint set the shipped SMFE program has?

    Returns (bool, reason). The shipped program constrains rows by tying whole
    spans and columns by tying all of them or none; it has no notion of a
    shared shape or of a partial column tie.

    ``column_kind`` is the RESOLVED kind from ``normalize_columns``, never the
    caller's spelling: a declared ``"scalar"`` has already been resolved
    against the primal block structure into ``"diagonal"`` (one coordinate per
    primal block), ``"scalar"`` (the whole primal space in one block) or a
    refusal. Both surviving kinds are constraint sets the shipped program has,
    and ``_construct_via_smfe`` selects between them with ``tie_columns``.
    """
    if column_kind == "spans":
        return False, "a partial column tie is not a constraint the shipped program expresses"
    if column_kind == "full":
        return False, "a full column shape is not a constraint the shipped program expresses"
    for d in declarations:
        if d.cls == "full":
            return False, f"term {d.index} declares a full metric"
        if d.cls == "spatial_full":
            return False, (f"term {d.index} declares 'spatial_full', constructed on the term's rows "
                           "after the geodesic run from the run's primal metric")
        if d.cls == "automorphism":
            return False, (f"term {d.index} declares 'automorphism' (a cone metric the shipped "
                           "diagonal program cannot express)")
        if d.sharing == "term" and d.cls == "diagonal":
            return False, f"term {d.index} declares one diagonal shape shared across its blocks"
        if d.cls == "scalar" and d.sharing == "per_block" and d.count > 1:
            return False, (f"term {d.index} declares one scalar per mathematical block "
                           f"({d.count} of them); the shipped program ties whole spans only")
    return True, "every term is a free diagonal or a whole-span scalar, and the columns are shipped form"


def natural_declaration(declarations):
    """True when every term declares the class its own proximal block supports.

    This is the default declaration, and the only one for which the shipped
    call is bit-for-bit the ``equilibrate="sgd"`` call.
    """
    from ..prox.base import SCALAR
    for d in declarations:
        natural = ("scalar", "term") if d.scaling_class == SCALAR else ("diagonal", "per_block")
        if (d.cls, d.sharing) != natural:
            return False
    return True


# ------------------------------------------------------------- metric groups
# One entry per metric-sharing group of one side. ``count`` sharing groups of
# ``size`` coordinates each for the per-block kinds; one shape shared by
# ``count`` blocks for the shared kinds.
class _Group:
    __slots__ = ("kind", "size", "count", "algebra", "shared", "layout")

    def __init__(self, kind, size, count=1, algebra=None, shared=False, layout=None):
        self.kind, self.size, self.count = kind, int(size), int(count)
        # Automorphism groups only: the Jordan algebra, whether one element is
        # shared by every cone of the term, and the term's row layout.
        self.algebra, self.shared, self.layout = algebra, bool(shared), layout

    @property
    def dimension(self):
        if self.kind == "automorphism" and self.algebra == "psd":
            return self.size * self.size * self.count
        return self.size * self.count

    @property
    def elements(self):
        """How many Jordan elements the state carries (automorphism groups)."""
        return 1 if self.shared else self.count

    def __repr__(self):
        return f"_Group({self.kind}, size={self.size}, count={self.count})"


def _sym(a):
    return (a + jnp.swapaxes(a, -1, -2)) * .5


def _mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _spectral(a, transform):
    values, vectors = jnp.linalg.eigh(_sym(a))
    return _sym(_mm(vectors * transform(values)[..., None, :], jnp.swapaxes(vectors, -1, -2)))


def _spectral_many(a, transforms):
    values, vectors = jnp.linalg.eigh(_sym(a))
    transpose = jnp.swapaxes(vectors, -1, -2)
    return tuple(_sym(_mm(vectors * fn(values)[..., None, :], transpose)) for fn in transforms)


def initial_state(groups, dtype=jnp.float32, seeds=None):
    """The identity metric in every declared representation.

    ``seeds`` (optional) maps a group position to a host array that replaces
    the identity for that group (an automorphism group's ``metric_initial``).
    """
    out = []
    for position, g in enumerate(groups):
        if seeds is not None and position in seeds:
            out.append(jnp.asarray(seeds[position], dtype))
        elif g.kind == "automorphism":
            if g.algebra == "soc":
                e = np.zeros((g.elements, g.size))
                e[:, 0] = 1.0
                out.append(jnp.asarray(e, dtype))
            else:
                out.append(jnp.asarray(np.broadcast_to(np.eye(g.size), (g.elements, g.size, g.size)), dtype))
        elif g.kind == "shared_full":
            out.append(jnp.eye(g.size, dtype=dtype))
        elif g.kind == "shared_diagonal":
            out.append(jnp.zeros((g.size,), dtype))
        elif g.kind == "scalar":
            out.append(jnp.zeros((g.count,), dtype))
        else:
            out.append(jnp.zeros((g.count, g.size), dtype))
    return tuple(out)


def apply_power(groups, state, vector, power=1.):
    """Apply ``M^power`` of the whole side's metric to one ambient vector."""
    pieces, start = [], 0
    for g, value in zip(groups, state):
        v = vector[start:start + g.dimension]
        if g.kind == "automorphism":
            R = _cm.rows(g.layout, v)
            if g.algebra == "soc":
                a = value if power == 1 else _cm.soc_spectral(value, lambda t: t**power)
                result = _cm.unrows(g.layout, _cm.apply_soc(a, _cm.soc_det(a), R))
            else:
                A = value if power == 1 else _spectral(value, lambda d: d**power)
                result = _cm.unrows(g.layout, _cm.apply_psd(A, R))
        elif g.kind == "shared_full":
            matrix = value if power == 1 else _spectral(value, lambda d: d**power)
            result = _mm(v.reshape(g.count, g.size), matrix.T).reshape(-1)
        elif g.kind == "shared_diagonal":
            result = (v.reshape(g.count, g.size) * jnp.exp(power * value)[None, :]).reshape(-1)
        elif g.kind == "scalar":
            result = (jnp.exp(power * value)[:, None] * v.reshape(g.count, g.size)).reshape(-1)
        else:
            result = (jnp.exp(power * value) * v.reshape(g.count, g.size)).reshape(-1)
        pieces.append(result)
        start += g.dimension
    return jnp.concatenate(pieces)


def _pooled_gram(w, size):
    """Direct (Q*count) by k contraction into k by k; no per-block Gram."""
    z = w.reshape(-1, size)
    return _sym(_mm(z.T, z) / z.shape[0])


def update_side(groups, state, samples, target, mu, box, eta, diagnostics=False):
    """One projected geodesic step for one side. Returns (state, finite)."""
    result, finite, start = [], jnp.asarray(True), 0
    step_records = []
    for g, value in zip(groups, state):
        w = samples[:, start:start + g.dimension]
        if g.kind == "automorphism":
            new, valid, used_eta = _automorphism_step(g, value, w, target, mu, box, eta, diagnostics=True)
            step_records.append(jnp.stack([jnp.min(used_eta), jnp.max(used_eta),
                jnp.sum(used_eta < eta).astype(jnp.float32), jnp.asarray(used_eta.size,jnp.float32)]))
        elif g.kind == "shared_full":
            half, log = _spectral_many(value, (jnp.sqrt, jnp.log))
            gram = _pooled_gram(w, g.size)
            tangent = (.5 * _mm(_mm(half, gram), half)
                       - .5 * target * jnp.eye(g.size, dtype=value.dtype)
                       + mu / 4 * log)
            # The exponent is floored (never capped above) at -GEODESIC_EXP_FLOOR
            # in every eigen-direction: see the constant. Directions below it
            # land under the box floor and are projected there regardless.
            exp = _spectral(-eta * tangent,
                            lambda d: jnp.exp(jnp.maximum(d, -GEODESIC_EXP_FLOOR)))
            unconstrained = _sym(_mm(_mm(half, exp), half))
            valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in
                                       (half, log, gram, tangent, exp, unconstrained)]))
            # Overflow never becomes a successfully clipped answer: keep the
            # previous valid metric and propagate finite=False.
            new = jax.lax.cond(valid,
                               lambda a: _spectral(a, lambda d: jnp.clip(
                                   d, jnp.exp(-2 * box), jnp.exp(2 * box))),
                               lambda _: value, unconstrained)
        else:
            if g.kind == "shared_diagonal":
                z = w.reshape(-1, g.size)
                energy = jnp.mean(z * z, axis=0)
            else:
                z = w.reshape(-1, g.count, g.size)
                energy = jnp.mean(z * z, axis=0)
                if g.kind == "scalar":
                    # The intrinsic scalar metric carries the block dimension:
                    # average the coordinate energy, never the unnormalized sum.
                    energy = jnp.mean(energy, axis=1)
            tangent = .5 * jnp.exp(value) * energy - .5 * target + mu / 4 * value
            unconstrained = value - eta * tangent
            valid = (jnp.all(jnp.isfinite(energy)) & jnp.all(jnp.isfinite(tangent))
                     & jnp.all(jnp.isfinite(unconstrained)))
            new = jax.lax.cond(valid, lambda z: jnp.clip(z, -2 * box, 2 * box),
                               lambda _: value, unconstrained)
        result.append(new)
        if g.kind != "automorphism":
            step_records.append(jnp.stack([eta, eta, jnp.asarray(0.,jnp.float32), jnp.asarray(0.,jnp.float32)]))
        finite = finite & valid
        start += g.dimension
    if diagnostics:
        return tuple(result), finite, jnp.stack(step_records)
    return tuple(result), finite


def _automorphism_step(g, value, w, target, mu, box, eta, diagnostics=False):
    """Projected geodesic step, with a declared per-carrier PSD step bound.

    The ambient tangent ``T = 0.5 A^1/2 G A^1/2 - 0.5 r I + (mu/4) log A`` at
    ``A = P(w)`` is projected (Frobenius) onto the tangent space ``{2 L(h)}``
    of the family and the ambient exponential map of the projection is
    ``P(P(w^1/2) exp_J(-eta h))``: the family is totally geodesic. Second-order
    cone, per cone (pooled over the term's cones when shared): with ``u =
    P(w^1/2) z``,

        h0   = E||u||^2 / (4k) - r/4 + (mu/8)(log lam1 + log lam2)
        hbar = E[u0 ubar] / 4 + (mu/4) ((log lam1 - log lam2)/2) dir

    PSD: ``U = S^1/2 Z S^1/2``, ``M = 0.5 E[U U^T + U^T U] - r k I + (mu/4)(2k
    log S + 2 tr(log S) I)``, ``Y = (M - tr(M)/(2k) I)/(2k)``, ``S <- S^1/2
    exp(-eta_eff Y) S^1/2``, eta_eff=min(eta,.5/||Y||_F). The box is the ambient spectral box intersected with
    the family, ``lam in [exp(-box), exp(box)]`` on the Jordan spectrum. An
    overflow keeps the previous element and reports ``finite = False``.
    """
    R = jax.vmap(lambda v: _cm.rows(g.layout, v))(w)          # (probes, cones, k[, k])
    k = g.size
    if g.algebra == "soc":
        half = _cm.soc_spectral(value, jnp.sqrt)                # (elements, k)
        u = _cm.apply_soc(half[None], _cm.soc_det(half)[None], R)
        energy = jnp.mean(jnp.sum(u * u, axis=-1), axis=0)     # (cones,)
        cross = jnp.mean(u[..., :1] * u[..., 1:], axis=0)      # (cones, k-1)
        if g.shared:
            energy = jnp.mean(energy, axis=0, keepdims=True)
            cross = jnp.mean(cross, axis=0, keepdims=True)
        lam1, lam2, direction = _cm.soc_eig(value)
        log1, log2 = jnp.log(lam1), jnp.log(lam2)
        h0 = energy / (4.0 * k) - target / 4.0 + (mu / 8.0) * (log1 + log2)
        hbar = cross / 4.0 + (mu / 4.0) * (0.5 * (log1 - log2))[:, None] * direction
        h = jnp.concatenate([h0[:, None], hbar], axis=1)
        new, arithmetic_valid = soc_step(value, h, eta, box)
        valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in
                                   (half, u, h)])) & arithmetic_valid
        result = (jnp.where(valid, new, value), valid)
        return result + (jnp.full((value.shape[0],),eta),) if diagnostics else result
    half, log = _spectral_many(value, (jnp.sqrt, jnp.log))    # (elements, k, k)
    U = _cm.apply_psd(half[None], R)                           # (probes, cones, k, k)
    UT = jnp.swapaxes(U, -1, -2)
    gram = 0.5 * jnp.mean(_mm(U, UT) + _mm(UT, U), axis=0)    # (cones, k, k)
    if g.shared:
        gram = jnp.mean(gram, axis=0, keepdims=True)
    eye = jnp.eye(k, dtype=value.dtype)
    trace_log = jnp.trace(log, axis1=-2, axis2=-1)[:, None, None]
    M = gram - target * k * eye + (mu / 4.0) * (2.0 * k * log + 2.0 * trace_log * eye)
    Y = (M - (jnp.trace(M, axis1=-2, axis2=-1) / (2.0 * k))[:, None, None] * eye) / (2.0 * k)
    new, arithmetic_valid, effective_eta = psd_bounded_step(value, Y, eta, box)
    valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in
                               (half, log, gram, Y)])) & arithmetic_valid
    result = (jnp.where(valid, new, value), valid)
    return result + (effective_eta,) if diagnostics else result


def average_side(groups, old, new, ratio):
    """Weighted geodesic (matrix-geometric) mean; linear in the logs when
    the state commutes, which is exact for the scalar and diagonal kinds."""
    result = []
    for g, a, b in zip(groups, old, new):
        if g.kind == "automorphism" and g.algebra == "soc":
            # P(a) #_ratio P(b) = P(w_t), w_t = P(a^1/2) (P(a^-1/2) b)^ratio, O(k) per cone.
            half = _cm.soc_spectral(a, jnp.sqrt)
            invhalf = _cm.soc_spectral(a, lambda t: 1 / jnp.sqrt(t))
            inner = _cm.apply_soc(invhalf, _cm.soc_det(invhalf), b)
            # The relative element may have condition exp(4*box). Recover
            # its small spectrum from the bounded operands' determinants.
            a1, a2, _ = _cm.soc_eig(a)
            b1, b2, _ = _cm.soc_eig(b)
            large, _, direction = _cm.soc_eig(inner)
            loglarge = jnp.log(large)
            logsmall = (jnp.log(b1)+jnp.log(b2)-jnp.log(a1)-jnp.log(a2)-loglarge)
            p1,p2 = jnp.exp(ratio*loglarge),jnp.exp(ratio*logsmall)
            powered = jnp.concatenate([(.5*(p1+p2))[:,None],
                                        (.5*(p1-p2))[:,None]*direction],axis=1)
            result.append(_cm.apply_soc(half, _cm.soc_det(half), powered))
        elif g.kind == "shared_full" or g.kind == "automorphism":
            half, invhalf = _spectral_many(a, (jnp.sqrt, lambda d: 1 / jnp.sqrt(d)))
            power = _spectral(_mm(_mm(invhalf, b), invhalf), lambda d: d**ratio)
            result.append(_sym(_mm(_mm(half, power), half)))
        else:
            result.append((1 - ratio) * a + ratio * b)
    return tuple(result)


def _state_valid(groups, state, box):
    valid = jnp.asarray(True)
    slack = 64 * jnp.finfo(jnp.float32).eps
    for g, value in zip(groups, state):
        if g.kind == "automorphism" and g.algebra == "soc":
            lam1, lam2, _ = _cm.soc_eig(value)
            valid = (valid & jnp.all(jnp.isfinite(value))
                     & jnp.all(lam2 >= jnp.exp(-box) * (1 - slack))
                     & jnp.all(lam1 <= jnp.exp(box) * (1 + slack)))
        elif g.kind == "automorphism":
            eigenvalues = jnp.linalg.eigvalsh(_sym(value))
            scale = jnp.maximum(1., jnp.max(jnp.abs(value)))
            valid = (valid & jnp.all(jnp.isfinite(value))
                     & (jnp.max(jnp.abs(value - jnp.swapaxes(value, -1, -2))) <= slack * scale)
                     & jnp.all(eigenvalues >= jnp.exp(-box) * (1 - slack))
                     & jnp.all(eigenvalues <= jnp.exp(box) * (1 + slack)))
        elif g.kind == "shared_full":
            eigenvalues = jnp.linalg.eigvalsh(_sym(value))
            scale = jnp.maximum(1., jnp.max(jnp.abs(value)))
            valid = (valid & jnp.all(jnp.isfinite(value))
                     & (jnp.max(jnp.abs(value - jnp.swapaxes(value, -1, -2))) <= slack * scale)
                     & jnp.all(eigenvalues >= jnp.exp(-2 * box) * (1 - slack))
                     & jnp.all(eigenvalues <= jnp.exp(2 * box) * (1 + slack)))
        else:
            valid = valid & jnp.all(jnp.isfinite(value)) & jnp.all(jnp.abs(value) <= 2 * box + slack)
    return valid


def make_geodesic_constructor(matvec, rmatvec, rows, columns, *, iterations=DEFAULT_UPDATES,
                              probes=DEFAULT_PROBES, mu=DEFAULT_MU, box=DEFAULT_BOX,
                              shift=DEFAULT_SHIFT, target_row=None, target_col=None):
    """Simultaneous fresh-probe projected geodesic SGD with pre-update averaging.

    ``status``: 0 the budget ran to the end with finite arithmetic throughout;
    1 the initial state was not admissible; 2 some step overflowed and was
    abandoned. Status 0 is a statement about arithmetic and budget only. It is
    NOT an optimality claim, NOT a convergence claim, and carries no rate.
    """
    rows, columns = tuple(rows), tuple(columns)
    if not rows or not columns:
        raise ValueError("nonempty row and column declarations required")
    m = sum(g.dimension for g in rows)
    n = sum(g.dimension for g in columns)
    for name, value in (("iterations", iterations), ("probes", probes), ("shift", shift)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(mu) or mu <= 0 or not math.isfinite(box) or box < 0:
        raise ValueError("positive finite mu and nonnegative finite box required")
    automorphic = any(g.kind == "automorphism" for g in rows + columns)
    if automorphic and box > 2:
        raise ValueError("research stable automorphism construction supports metric_box <= 2")
    r = math.sqrt(n / m) if target_row is None else float(target_row)
    s = math.sqrt(m / n) if target_col is None else float(target_col)
    if any(not math.isfinite(v) or v <= 0 for v in (r, s)):
        raise ValueError("positive finite targets required")

    def run(key, a0, b0):
        initial_valid = _state_valid(rows, a0, box) & _state_valid(columns, b0, box)

        def body(index, carry):
            key, a, b, avg_a, avg_b, total_weight, finite, history = carry
            key, row_key, col_key = jax.random.split(key, 3)
            xi = jax.random.rademacher(row_key, (probes, n), dtype=jnp.float32)
            zeta = jax.random.rademacher(col_key, (probes, m), dtype=jnp.float32)
            # Both oracle estimates see exactly the same pre-update (A, B).
            forward = jax.vmap(lambda v: matvec(apply_power(columns, b, v, .5)))(xi)
            reverse = jax.vmap(lambda v: rmatvec(apply_power(rows, a, v, .5)))(zeta)
            t = index + 1
            weight = t + shift - 1
            next_total = total_weight + weight
            ratio = weight / next_total
            avg_a = jax.lax.cond(index == 0, lambda _: a,
                                 lambda _: average_side(rows, avg_a, a, ratio), None)
            avg_b = jax.lax.cond(index == 0, lambda _: b,
                                 lambda _: average_side(columns, avg_b, b, ratio), None)
            eta = 2 / ((mu / 4) * (t + shift))
            next_a, fa, da = update_side(rows, a, forward, r, mu, box, eta, diagnostics=True)
            next_b, fb, db = update_side(columns, b, reverse, s, mu, box, eta, diagnostics=True)
            history = history.at[index].set(jnp.concatenate((da,db),axis=0))
            finite = finite & fa & fb & jnp.isfinite(eta) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(z)) for z in jax.tree_util.tree_leaves(
                    (next_a, next_b, avg_a, avg_b))]))
            return key, next_a, next_b, avg_a, avg_b, next_total, finite, history

        _, a, b, avg_a, avg_b, weight, finite, history = jax.lax.fori_loop(
            0, iterations, body,
            (key, a0, b0, a0, b0, jnp.asarray(0., jnp.float32), initial_valid,
             jnp.zeros((iterations,len(rows)+len(columns),4),jnp.float32)))
        if automorphic:
            finite = (finite & _state_valid(rows, avg_a, box) & _state_valid(columns, avg_b, box)
                      & _state_valid(rows, a, box) & _state_valid(columns, b, box))
        status = jnp.where(~initial_valid, 1, jnp.where(finite, 0, 2))
        return {"A": avg_a, "B": avg_b, "last_A": a, "last_B": b,
                "weight_sum": weight, "finite": finite, "status": status,
                "actual_step_history": history,
                "operator_products": jnp.asarray(2 * iterations * probes, jnp.int32)}
    return jax.jit(run), {"targets": {"row": r, "col": s}, "m": m, "n": n}


def state_bytes(groups, dtype_bytes=4):
    """The metric state itself, excluding probes, copies and workspaces."""
    total = 0
    for g in groups:
        if g.kind == "automorphism":
            total += g.elements * (g.size if g.algebra == "soc" else g.size * g.size)
        elif g.kind == "shared_full":
            total += g.size * g.size
        elif g.kind == "shared_diagonal":
            total += g.size
        elif g.kind == "scalar":
            total += g.count
        else:
            total += g.count * g.size
    return dtype_bytes * total


# ------------------------------------------------------ declaration -> groups
def row_groups(declarations):
    groups, owner = [], []
    for d in declarations:
        if d.cls == "spatial_full":
            # Constructed AFTER the run, on the term's rows, from the run's tau:
            # during the run the term is one coupled scalar group.
            groups.append(_Group("scalar", d.size, 1))
            owner.append(d)
            continue
        if d.cls == "automorphism":
            groups.append(_Group("automorphism", d.k, d.count, algebra=d.cone["algebra"],
                                 shared=(d.sharing == "term"), layout=d.cone["layout"]))
        elif d.cls == "full":
            groups.append(_Group("shared_full", d.k, d.count))
        elif d.cls == "diagonal" and d.sharing == "term":
            groups.append(_Group("shared_diagonal", d.k, d.count))
        elif d.cls == "diagonal":
            groups.append(_Group("diagonal", d.k, d.count))
        elif d.sharing == "term":
            groups.append(_Group("scalar", d.size, 1))
        else:
            groups.append(_Group("scalar", d.k, d.count))
        owner.append(d)
    return tuple(groups), tuple(owner)


def column_groups(kind, spans, n, k=None):
    if kind == "scalar":
        return (_Group("scalar", n, 1),)
    if kind == "diagonal":
        return (_Group("diagonal", n, 1),)
    if kind == "full":
        # One k x k shape shared by the n / k contiguous column blocks: the same
        # group kind the row side emits for ("full", "term"), so initial_state,
        # apply_power, update_side, average_side, _state_valid and state_bytes
        # serve it unchanged.
        if k is None or n % int(k) != 0:
            raise ValueError("a full column group needs a block size k that divides n")
        return (_Group("shared_full", int(k), n // int(k)),)
    lengths = {b - a for a, b in spans}
    if len(lengths) == 1:
        # Equal spans are one group of `len(spans)` tied blocks: the same
        # reduction the research constructor used for a channel tie.
        return (_Group("scalar", spans[0][1] - spans[0][0], len(spans)),)
    return tuple(_Group("scalar", b - a, 1) for a, b in spans)


def _decode_rows(group, value, k):
    """Host float64 squared row metric of one group: per-block or one shared."""
    value = np.asarray(value, np.float64)
    if group.kind == "automorphism":
        if group.algebra == "soc":
            return "automorphism", value.reshape(group.elements, group.size)
        a = value.reshape(group.elements, group.size, group.size)
        return "automorphism", 0.5 * (a + np.swapaxes(a, -1, -2))
    if group.kind == "shared_full":
        a = value.reshape(k, k)
        return "shared_full", 0.5 * (a + a.T)
    if group.kind == "shared_diagonal":
        return "shared_diagonal", np.exp(value.reshape(k))
    if group.kind == "scalar":
        return "scalar", np.exp(value.reshape(group.count))
    return "diagonal", np.exp(value.reshape(group.count, group.size))


def _decode_columns(groups, state):
    """Host float64 column shape per group, in the ``_decode_rows`` form.

    Only the ``shared_full`` kind is decoded here: its state is the SPD matrix
    itself, never a log, so it must not pass through ``_expand_columns``.
    """
    out = []
    for g, value in zip(groups, state):
        value = np.asarray(value, np.float64)
        if g.kind != "shared_full":
            raise ValueError(f"_decode_columns decodes shared_full groups only, got {g!r}")
        a = value.reshape(g.size, g.size)
        out.append(("shared_full", 0.5 * (a + a.T)))
    return out


def _expand_columns(groups, state, n):
    out = np.empty(n, np.float64)
    start = 0
    for g, value in zip(groups, state):
        # A shared_full column group holds an SPD matrix, not logs: exp of it
        # would be a silently wrong tau. It is decoded by _decode_columns.
        assert g.kind != "shared_full", "a shared_full column group is decoded by _decode_columns"
        value = np.asarray(value, np.float64)
        if g.kind == "scalar":
            out[start:start + g.dimension] = np.repeat(np.exp(value.reshape(g.count)), g.size)
        else:
            out[start:start + g.dimension] = np.exp(value).reshape(-1)
        start += g.dimension
    return out


# ------------------------------------------------------------- the entry point
def _shape_note(d, base):
    """The applied-shape note, by the owning block's family."""
    if d.quadratic is None:
        return base
    note = (base + f", on metric blocks of {d.k} rows ({d.k_source}), applied through the "
            "quadratic block's closed form (H + Sigma) z = v - b, y = v - Sigma z")
    if d.quadratic == "zero":
        note += (". prox_{f*} = 0 for every Sigma, so the shape never enters a dual prox on this "
                 "term. It enters the joint construction and the normalization, and so changes "
                 "the constructed metric of every other term")
    return note


def _identity_shape(w, q):
    return q is None and bool(np.all(w == 1.0))


def construct_metric(matvec, rmatvec, n, m, blocks, slices, *,
                     metric_classes=None, metric_columns=None,
                     primal_scalar_metric=False, primal_block=None, updates=DEFAULT_UPDATES,
                     probes=DEFAULT_PROBES, seed=0, safety=DEFAULT_SAFETY,
                     mu=DEFAULT_MU, box=DEFAULT_BOX, shift=DEFAULT_SHIFT,
                     domain="reject", smfe_iters=None, smfe_probes=None,
                     smfe_schedule="sgd", cache_operators=None, well_conditioned=False,
                     metric_initial=None):
    """Build the declared metric. Returns (tau, sigma, scaled_norm, declared, report).

    ``metric_initial`` (optional) is ``{dual_term_index: ConeAutomorphism}``:
    the initial state of the geodesic run for a term declared
    ``"automorphism"`` (for example ``ocs.nt_scaling_point(block, s, z)``).
    Accepted only for such a term, only when it matches the term's algebra,
    ``k`` and sharing and lies inside the box; otherwise ``ValueError``, never
    silently dropped.

    ``declared`` is ``{dual_term_index: BlockMetric or ConeAutomorphism}`` for the terms whose
    proximal block must be told about a shape (or about steps that vary inside
    it); it is passed straight to the compiler's existing ``block_metrics``
    path, so the metric is APPLIED by code that this change does not touch.
    ``tau`` and ``sigma`` are the ordinary per-coordinate steps, already carrying
    the single reported normalization factor.
    """
    from .equilibrate_sgd import smfe_equilibration
    declarations = normalize_declarations(metric_classes, blocks, slices)
    column_kind, column_spans, columns = normalize_columns(
        metric_columns, n, primal_scalar_metric, primal_block)
    expressible, why = shipped_expressible(declarations, column_kind)
    if domain not in ("reject", "restrict"):
        raise ValueError("metric_domain must be 'reject' or 'restrict'")
    seeds = _resolve_initial(metric_initial, declarations, box)
    # The restriction the caller declared, in one hashable value. It is the
    # cache identity of every program built for this declaration: a cached
    # executable can then never be reused under a different restriction or a
    # different block/sharing structure.
    restriction = ("ocs.metric_restriction.v1",
                   tuple((int(d.index), d.cls, d.sharing, int(d.k), int(d.count))
                         for d in declarations),
                   column_kind, column_spans,
                   (columns["primal_blocks"]["k"], columns["primal_blocks"]["count"]))
    if column_kind == "full":
        # Only for the full kind, so every other declaration's identity (and
        # its recorded metadata) is byte for byte what it was.
        restriction += (("column_blocks", columns["column_blocks"]["k"],
                         columns["column_blocks"]["count"], "term"),)
    report = {"declared": [d.describe() for d in declarations],
              "columns": columns,
              "restriction_cache_identity": restriction,
              "natural_declaration": natural_declaration(declarations),
              "operator_only": "K is reached only through matvec/rmatvec; dense work is on "
                               "k-by-k blocks (pooled Gram, symmetric eigendecomposition)"}
    if seeds and (well_conditioned and report["natural_declaration"] and column_kind != "full"
                  or expressible):
        raise ValueError("internal: metric_initial resolved on a path without a geodesic run")
    if well_conditioned and report["natural_declaration"] and column_kind != "full":
        # Every dual operator declares itself well conditioned, so the shipped
        # compiler skips equilibration entirely and takes one scalar step. With
        # the natural declaration there is nothing else to build: take the same
        # skip, so the result is the shipped one, byte for byte. A declared
        # full column shape is NOT built by this skip (it would return one
        # scalar under the name of a k x k class), so it goes to the geodesic
        # regime below whatever the operators certify.
        from .equilibrate import scalar_steps
        start = time.perf_counter()
        t0, s0, nrm = scalar_steps(matvec, rmatvec, n)
        elapsed = time.perf_counter() - start
        report.update(
            regime="scalar",
            regime_note="every dual operator reports well_conditioned, and the declaration is "
                        "the natural one, so the shipped operator-aware skip applies: one scalar "
                        "step from a power estimate of ||K||_2, exactly as equilibrate='sgd' "
                        "resolves on this family",
            estimator="20-iteration power estimate of ||K||_2; a power estimate is a LOWER bound "
                      "on the norm",
            rate_claim="none",
            operator_products={"gradient": 0, "normalization": 40, "total": 40,
                               "note": "the 20-iteration power estimate, 2 products per iteration"},
            seconds={"total_public_call": elapsed},
            measured_scaled_norm=float(nrm), bit_identical_to_sgd=True)
        return (jnp.asarray(t0, jnp.float32), jnp.asarray(s0, jnp.float32),
                float(nrm), {}, report)
    if expressible:
        built = _construct_via_smfe(matvec, rmatvec, n, m, declarations, column_kind,
                                    primal_scalar_metric, report, smfe_equilibration,
                                    updates if smfe_iters is None else smfe_iters,
                                    probes if smfe_probes is None else smfe_probes,
                                    smfe_schedule, cache_operators, restriction)
    else:
        report["regime_reason"] = why
        built = _construct_via_geodesic(matvec, rmatvec, n, m, declarations, column_kind,
                                        column_spans, report, updates=updates, probes=probes,
                                        seed=seed, safety=safety, mu=mu, box=box, shift=shift,
                                        domain=domain, seeds=seeds)
    return _attach_spatial_full(matvec, rmatvec, n, m, blocks, slices, declarations, built,
                                safety)


def _attach_spatial_full(matvec, rmatvec, n, m, blocks, slices, declarations, built, safety):
    """Construct every declared 'spatial_full' carrier from products, after the run.

    The run (SMFE or geodesic) has already fixed the primal metric T and one
    uniform dual step on each spatial term.  For each such term the product-only
    constructor of ``ocs.prox.spatial_construct`` builds ``(a, b, Q)`` on the
    term's rows from ``k`` adjoint and ``k`` forward applications with T in place,
    the carrier joins ``declared`` exactly as a supplied one would, and the
    spectral safety is RE-MEASURED with every carrier applied: one common factor
    then brings the power estimate of ||Sigma^1/2 K T^1/2||_2 back to ``safety``
    on both sides, as the geodesic run does for its own shapes.  The estimate is
    a lower bound; the compiler re-measures and refuses at 1.  Nothing about the
    other terms' shapes changes except that common factor, which is recorded.
    """
    spatial = [d for d in declarations if d.cls == "spatial_full"]
    if not spatial:
        return built
    from ..prox import spatial_construct as _sc
    from ..prox import spatial_soc as _ss
    tau, sigma, nrm, declared, report = built
    tau = jnp.broadcast_to(jnp.asarray(tau, jnp.float32).reshape(-1), (n,))
    sigma = np.broadcast_to(np.asarray(sigma, np.float64).reshape(-1), (m,)).copy()
    declared = dict(declared)
    records = {}
    r = float(math.sqrt(n / m))                     # the archive's shape scalar, from dimensions
    start_time = time.perf_counter()
    for d in spatial:
        s, e = d.start, d.end
        k = e - s

        def term_matvec(x, s=s, e=e):
            return matvec(x)[s:e]

        def term_rmatvec(v, s=s, e=e):
            return rmatvec(jnp.zeros((m,), jnp.float32).at[s:e].set(jnp.asarray(v, jnp.float32)))

        count = _sc.ProductCount()
        metric, crec = _sc.construct_spatial_full(term_matvec, term_rmatvec, k, n, tau, r,
                                                  layout=("soc",), count=count)
        declared[d.index] = metric
        low, high = _ss.metric_amplitude(metric)
        records[int(d.index)] = {"construction": crec, "products": count.record(),
                                 "metric_amplitude": {"min_eigenvalue": low, "max_eigenvalue": high},
                                 "shape_scalar_r": r, "cone": dict(d.cone)}
    construct_seconds = time.perf_counter() - start_time
    # Re-measure with every carrier in place, through the same applier the
    # compiler uses, then one common factor on both sides.
    term_metrics = {}
    for index, metric in declared.items():
        s0, e0 = slices[index]
        k0, ranges = _bm.normalize_term_metrics(metric, index, blocks[index], e0 - s0)
        term_metrics[int(index)] = (k0, _bm.check_static_steps(sigma, s0, e0, k0, ranges, index))
    shape_apply = _bm.make_shape_apply(term_metrics, slices, m)
    root_tau = jnp.sqrt(tau)
    root_sigma = jnp.sqrt(jnp.asarray(sigma, jnp.float32))

    def scaled_mv(x):
        return root_sigma * shape_apply(matvec(root_tau * x), power=0.5)

    def scaled_rmv(y):
        return root_tau * rmatvec(shape_apply(root_sigma * y, power=0.5))

    mark = time.perf_counter()
    measured = float(_power_norm(scaled_mv, scaled_rmv, n, iters=15, seed=7))
    norm_seconds = time.perf_counter() - mark
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError(f"the spatial-full carrier gives a non-positive or nonfinite measured "
                         f"operator norm ({measured!r}); no metric is returned")
    factor = float(safety) / measured
    tau = tau * jnp.asarray(factor, jnp.float32)
    sigma = sigma * factor
    report["spatial_full"] = {
        "terms": records,
        "measured_scaled_norm_before_carrier": report.get("measured_scaled_norm"),
        "measured_scaled_norm_with_carrier": measured,
        "common_factor": factor,
        "seconds": {"construction": construct_seconds, "normalization": norm_seconds},
        "normalization": ("one common factor on tau and sigma so the 15-iteration power estimate of "
                          "||Sigma^1/2 K T^1/2||_2 with every carrier applied equals safety; a power "
                          "estimate is a LOWER bound and the compiler re-measures and refuses at 1"),
        "operator_only": "the carrier is built from k adjoint and k forward applications of the "
                         "term's rows with T in place; no entry of K is read"}
    report["measured_scaled_norm"] = measured * factor
    products = report.get("operator_products")
    if isinstance(products, dict):
        extra = sum(rec["products"]["total_operator_applications"] for rec in records.values())
        products = dict(products)
        products["spatial_full_construction"] = int(extra)
        products["normalization_with_carrier"] = 30
        products["total"] = int(products.get("total", 0)) + int(extra) + 30
        report["operator_products"] = products
    return (tau, jnp.asarray(sigma, jnp.float32), float(measured * factor), declared, report)


def _resolve_initial(metric_initial, declarations, box):
    """Validate ``metric_initial`` against the declaration; return {term: host state}."""
    if metric_initial is None:
        return {}
    if not isinstance(metric_initial, dict):
        raise ValueError("metric_initial must be None or a dict {dual_term_index: ConeAutomorphism}")
    by_index = {d.index: d for d in declarations}
    seeds = {}
    for index, carrier in metric_initial.items():
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) or int(index) not in by_index:
            raise ValueError(f"metric_initial key {index!r} is not a dual term index")
        d = by_index[int(index)]
        if d.cls != "automorphism":
            raise ValueError(
                f"metric_initial[{index}]: term {index} declares {d.cls!r}, not 'automorphism'; an "
                "initial element is accepted only for an automorphism term. It is never dropped.")
        if not isinstance(carrier, _cm.ConeAutomorphism):
            raise ValueError(f"metric_initial[{index}] must be a ConeAutomorphism, got "
                             f"{type(carrier).__name__}")
        elements = 1 if d.sharing == "term" else d.count
        if carrier.algebra != d.cone["algebra"] or carrier.k != d.k or carrier.count != elements:
            raise ValueError(
                f"metric_initial[{index}]: ConeAutomorphism(algebra={carrier.algebra}, k={carrier.k}, "
                f"count={carrier.count}) does not match term {index} ({d.block_name}: algebra="
                f"{d.cone['algebra']}, k={d.k}, {elements} element(s) for sharing {d.sharing!r})")
        value = carrier.elements
        if carrier.algebra == "soc":
            lam1, lam2, _ = _cm._soc_eig64(value)
            lo, hi = float(np.min(lam2)), float(np.max(lam1))
        else:
            values = np.linalg.eigvalsh(value)
            lo, hi = float(values.min()), float(values.max())
        if lo < math.exp(-box) or hi > math.exp(box):
            raise ValueError(
                f"metric_initial[{index}]: the element's Jordan eigenvalues span [{lo:.6g}, {hi:.6g}], "
                f"outside the spectral box [exp(-box), exp(box)] = [{math.exp(-box):.6g}, "
                f"{math.exp(box):.6g}] with box={float(box)}. Nothing is clipped for you: rescale "
                "the element (only its shape is used; the step normalization sets the scale) or "
                "widen metric_box.")
        seeds[int(index)] = (value, carrier)
    return seeds


def _construct_via_smfe(matvec, rmatvec, n, m, declarations, column_kind,
                        primal_scalar_metric, report, smfe_equilibration,
                        iters, probes, schedule, cache_operators, restriction):
    """The declaration IS the shipped program's constraint set: run that program.

    Not a reimplementation and not an approximation of Diamond & Boyd's method:
    the shipped ``smfe_equilibration`` is called, with the scalar spans the
    declaration asks for. When the declaration is the natural one those spans
    are the compiler's own, so the call is the ``equilibrate="sgd"`` call and
    every returned array is bit-for-bit identical to it.
    """
    from ..prox.base import SCALAR
    spans = tuple((d.start, d.end) for d in declarations if d.cls == "scalar" and d.sharing == "term")
    # The COLUMN restriction is the resolved declaration, not a property of the
    # primal block: `tie_columns` ties the column log-scalings inside the
    # projected-SGD loop (the Euclidean projection onto the constant subspace
    # at every step), so the program solves the constrained problem. It is
    # never a mean, geometric mean or minimum taken over a free diagonal
    # solution afterwards, which would not in general be the constrained
    # minimizer.
    tie_columns = column_kind == "scalar"
    start = time.perf_counter()
    tau, sigma, nrm, diag = smfe_equilibration(
        matvec, rmatvec, n, m, spans, diagnostics=True, probes=probes, iters=iters,
        schedule=schedule, tie_columns=tie_columns,
        _cache_operators=cache_operators, _cache_tag=restriction)
    jax.block_until_ready((tau, sigma))
    elapsed = time.perf_counter() - start
    declared = {}
    for d in declarations:
        # A coupled block accepts ONE step. When the declaration asks for steps
        # that vary inside it, the block has to be told, with the identity
        # shape: the numbers are the shipped program's, the application is the
        # weighted prox.
        if d.scaling_class == SCALAR and not (d.cls == "scalar" and d.sharing == "term"):
            declared[d.index] = _bm.BlockMetric(np.ones(d.k), None)
    report.update(
        regime="smfe",
        regime_note=("the shipped stochastic matrix-free equilibration (Diamond & Boyd, JOTA "
                     "2017) called unchanged; this module adds no arithmetic of its own here"),
        estimator="one Rademacher row/column energy probe pair per update, projected SGD on "
                  "the log-scalings, iterate averaged; the common factor comes from "
                  "ocs.solver.equilibrate._power_norm, 15 iterations, and a power estimate is a "
                  "LOWER bound on the norm",
        rate_claim="none is made here; see ocs.solver.equilibrate_sgd for that method's own "
                   "documentation",
        scalar_spans=spans, smfe_iters=int(iters), smfe_probes=int(probes),
        smfe_schedule=schedule, tie_columns=tie_columns,
        diagnostics={k: float(v) for k, v in diag.items()},
        operator_products={"gradient": 2 * int(iters) * int(probes), "normalization": 30,
                           "diagnostics": 32,
                           "total": 2 * int(iters) * int(probes) + 30 + 32,
                           "note": "2 products per update per side, plus the 15-iteration power "
                                   "estimate (2 products per iteration), plus the shipped "
                                   "program's own before/after equilibration diagnostic "
                                   "(2 x 8 probes x 2 products), which it always computes here"},
        seconds={"total_public_call": elapsed,
                 "note": "one uninstrumented public call; the shipped program lowers, compiles "
                         "and executes inside it and caches the executable"},
        measured_scaled_norm=float(nrm),
        bit_identical_to_sgd=bool(report["natural_declaration"] and column_kind ==
                                  ("scalar" if primal_scalar_metric else "diagonal")))
    return tau, sigma, float(nrm), declared, report


def _construct_via_geodesic(matvec, rmatvec, n, m, declarations, column_kind, column_spans,
                            report, *, updates, probes, seed, safety, mu, box, shift, domain,
                            seeds=None):
    rows, owners = row_groups(declarations)
    column_record = report["columns"]
    column_full = column_kind == "full"
    column_k = int(column_record["column_blocks"]["k"]) if column_full else None
    columns = column_groups(column_kind, column_spans, n, column_k)
    program, info = make_geodesic_constructor(
        matvec, rmatvec, rows, columns, iterations=updates, probes=probes,
        mu=mu, box=box, shift=shift)
    key = jax.random.PRNGKey(seed)
    seeds = seeds or {}
    initializer = []
    by_position = {}
    for position, d in enumerate(owners):
        if d.cls != "automorphism":
            continue
        entry = {"term": d.index, "algebra": d.cone["algebra"], "sharing": d.sharing,
                 "returned_metric": ("the weighted geodesic mean of the pre-update states "
                                     "a_0 .. a_{T-1}; with metric_updates=1 the returned element "
                                     "is the initial element itself, unchanged")}
        if d.index in seeds:
            value, carrier = seeds[d.index]
            by_position[position] = value
            entry.update(source="metric_initial", element=carrier.describe(),
                         origin=carrier.origin)
        else:
            entry.update(source="identity", origin=None)
        initializer.append(entry)
    if initializer:
        report["initializer"] = initializer
    a0, b0 = initial_state(rows, seeds=by_position), initial_state(columns)
    mark = time.perf_counter(); lowered = program.lower(key, a0, b0)
    lower_seconds = time.perf_counter() - mark
    mark = time.perf_counter(); executable = lowered.compile()
    compile_seconds = time.perf_counter() - mark
    mark = time.perf_counter(); answer = executable(key, a0, b0); jax.block_until_ready(answer)
    execute_seconds = time.perf_counter() - mark
    status = int(answer["status"])
    if status != 0:
        why = ("the initial state was not admissible" if status == 1 else
               "nonfinite arithmetic: at least one geodesic step overflowed and was abandoned")
        raise ValueError(
            f"metric construction failed numerically (status {status}: {why}). No metric is "
            "returned and nothing is clipped into a plausible answer. Reduce metric_box or "
            "metric_updates, or compile without equilibrate='construct'.")
    row_state = tuple(answer["A"])
    col_state = tuple(answer["B"])
    decoded = [_decode_rows(g, v, d.k) for g, v, d in zip(rows, row_state, owners)]
    column_decoded = _decode_columns(columns, col_state) if column_full else None

    def measure(state_rows, state_cols):
        # ||A^1/2 K B^1/2||_2 on the float32 state of BOTH sides, so that a
        # restriction on either side is re-measured on the state it changed.
        def mv(x):
            return apply_power(rows, state_rows, matvec(apply_power(columns, state_cols, x, .5)), .5)

        def rmv(y):
            return apply_power(columns, state_cols, rmatvec(apply_power(rows, state_rows, y, .5)), .5)
        value = _power_norm(jax.jit(mv), jax.jit(rmv), n, iters=NORM_ITERS, seed=NORM_SEED)
        jax.block_until_ready(value)
        return float(value)

    mark = time.perf_counter()
    measured = measure(row_state, col_state)
    norm_seconds = time.perf_counter() - mark
    norm_calls = 1

    # ------------------------------------------------ enforced numerical domain
    restrictions = []
    for position, (d, (kind, value)) in enumerate(zip(owners, decoded)):
        if kind == "automorphism":
            cond = float(np.max(_cm.cond_of(d.cone["algebra"], value)))
            limit = _cm.MAX_AUTOMORPHISM_COND
        elif kind == "shared_full":
            cond = float(np.linalg.cond(value))
            limit = _bm.MAX_FULL_COND
        elif kind == "shared_diagonal":
            cond = float(np.max(value) / np.min(value))
            limit = _bm.MAX_DIAGONAL_COND
        else:
            continue
        if cond <= limit:
            continue
        if domain == "reject":
            if kind == "automorphism":
                message = _cm._domain_message(cond, limit, d.cone["algebra"],
                                              what="constructed cone automorphism metric")
            else:
                message = _bm._domain_message(cond, limit, kind == "shared_diagonal",
                                              what="constructed block metric")
            raise ValueError(
                f"metric_classes[{d.index}]: the constructed "
                f"{ {'shared_full': 'full (rotated)', 'automorphism': 'automorphism'}.get(kind, 'diagonal')} "
                f"shape has condition number {cond:.6g}, outside the validated numerical domain "
                f"(at most {limit:g}). " + message +
                " Nothing is clipped for you. Pass metric_domain='restrict' to restrict the "
                "constructed shape explicitly, which recomputes and reports the normalization, "
                "or declare a narrower class for this term.")
        if kind == "automorphism":
            restricted, restriction_info = _cm.restrict_cone_condition(d.cone["algebra"], value, limit)
        else:
            restricted, restriction_info = _bm.restrict_condition(value, limit)
        restriction_info.update(term=d.index, kind=kind)
        restrictions.append(restriction_info)
        decoded[position] = (kind, restricted)
        row_state = tuple(
            jnp.asarray(restricted if kind in ("shared_full", "automorphism") else np.log(restricted),
                        jnp.float32)
            if i == position else v for i, v in enumerate(row_state))
    if column_full:
        # The same domain and the same policy on the column shape. The constant
        # MAX_FULL_COND was measured for the float32 weighted norm PROX on the
        # row side; here the shape is only applied to a vector, and the bound
        # is inherited as the one documented domain constant of the carrier,
        # not as a measurement of the column application.
        _kind, value = column_decoded[0]
        cond = float(np.linalg.cond(value))
        limit = _bm.MAX_FULL_COND
        if cond > limit:
            if domain == "reject":
                raise ValueError(
                    f"metric_columns=('full', {column_k}): the constructed full (rotated) column "
                    f"shape has condition number {cond:.6g}, outside the validated numerical "
                    f"domain (at most {limit:g}). "
                    + _bm._domain_message(cond, limit, False, what="constructed column metric")
                    + " Nothing is clipped for you. Pass metric_domain='restrict' to restrict "
                    "the constructed shape explicitly, which recomputes and reports the "
                    "normalization, or declare 'scalar', 'diagonal' or (start, end) spans.")
            restricted, restriction_info = _bm.restrict_condition(value, limit)
            restriction_info.update(side="columns", kind="shared_full",
                                    declared=("full", column_k))
            restrictions.append(restriction_info)
            column_decoded[0] = ("shared_full", restricted)
            col_state = (jnp.asarray(restricted, jnp.float32),)
    if restrictions:
        # Raising eigenvalues raises || Sigma^1/2 K T^1/2 ||, so the bound measured
        # on the unrestricted shape no longer certifies the step condition.
        mark = time.perf_counter()
        measured_after = measure(row_state, col_state)
        norm_seconds += time.perf_counter() - mark
        norm_calls += 1
        report["restriction"] = {"applied": restrictions,
                                 "measured_before_restriction": measured,
                                 "measured_after_restriction": measured_after,
                                 "policy": "metric_domain='restrict': the shape was restricted "
                                           "explicitly and the normalization was recomputed "
                                           "against the restricted shape"}
        measured = measured_after
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError(f"the constructed metric has a non-positive or nonfinite measured "
                         f"operator norm ({measured!r}); no metric is returned")
    factor = float(safety) / measured

    # ------------------------------------------------------------ steps + shapes
    sigma = np.empty(m, np.float64)
    declared = {}
    effective = []
    for d, (kind, value) in zip(owners, decoded):
        rows_slice = slice(d.start, d.end)
        if kind == "automorphism":
            # The scalar factor multiplies sigma only; the recorded element is
            # the constructed w and the effective metric is factor * P(w).
            sigma[rows_slice] = factor
            declared[d.index] = _cm.ConeAutomorphism.from_state(
                d.cone["algebra"], value, d.cone["layout"], d.sharing)
            shape_note = ("Sigma_i = c P(w_i): one Jordan element per cone (per_block) or one "
                          "shared element (term), applied as W = P(w^{1/2}); c is the term's "
                          "uniform dual step, so the effective metric is c P(w) = P(sqrt(c) w)")
        elif kind == "shared_full":
            sigma[rows_slice] = factor
            metric = _bm.BlockMetric.from_matrix(np.ascontiguousarray(value))
            declared[d.index] = metric
            shape_note = _shape_note(d, "Sigma_i = c S with one shared full SPD S = Q diag(w) Q^T")
        elif kind == "shared_diagonal":
            sigma[rows_slice] = factor
            metric = _bm.BlockMetric(value, None)
            declared[d.index] = metric
            shape_note = _shape_note(d, "Sigma_i = c S with one shared diagonal S")
        elif kind == "scalar":
            sigma[rows_slice] = np.repeat(factor * value, d.size // value.size)
            shape_note = "Sigma_i = sigma_i I, one step per metric-sharing group"
        else:
            sigma[rows_slice] = (factor * value).reshape(-1)
            shape_note = "Sigma_i = diag(sigma_i), one step per coordinate"
        if d.index not in declared and d.scaling_class == "scalar":
            uniform = bool(np.all(sigma[rows_slice] == sigma[d.start]))
            if not uniform:
                declared[d.index] = _bm.BlockMetric(np.ones(d.k), None)
        entry = d.describe()
        entry.update(effective=shape_note,
                     dual_step_min=float(np.min(sigma[rows_slice])),
                     dual_step_max=float(np.max(sigma[rows_slice])))
        if d.index in declared:
            entry["shape"] = declared[d.index].describe()
        effective.append(entry)
    if column_full:
        # T = c (I kron S): the carrier c is ONE scalar (the representation the
        # well-conditioned skip and a uniform sigma already use) and the shape
        # S travels once, as the row side's (Q, w) carrier, decoded on the host
        # in float64 (one k x k eigh). from_matrix re-checks the domain.
        tau = np.asarray(factor, np.float64)
        column_shape = _bm.BlockMetric.from_matrix(np.ascontiguousarray(column_decoded[0][1]))
        column_record["shape_carrier"] = column_shape
        column_record["effective"] = (
            f"T_i = c S, one shared full SPD S = Q diag(w) Q^T over "
            f"{column_record['column_blocks']['count']} blocks of {column_k} coordinates, "
            "c one scalar carrier")
    else:
        tau = factor * _expand_columns(columns, col_state, n)
    for name, value in (("tau", tau), ("sigma", sigma)):
        if not (np.all(np.isfinite(value)) & np.all(value > 0)):
            raise ValueError(f"the constructed {name} is not finite and positive; no metric is "
                             "returned and nothing is repaired silently")
    for d in owners:
        # A coupled block accepts exactly one step unless it was given a shape.
        if d.scaling_class == "scalar" and d.index not in declared:
            piece = sigma[d.start:d.end]
            if piece.size and not np.all(piece == piece[0]):
                raise ValueError(
                    f"internal: the constructed dual step is not uniform on coupled term "
                    f"{d.index} and no shape was declared for it")
    automorphic = any(g.kind == "automorphism" for g in rows)
    if automorphic:
        report["automorphism_arithmetic"] = {
            "identity": "stable_soc_psd_bounded_v1",
            "soc": "same ideal projected step, scaled exponentials and log determinant recovery",
            "soc_average": "relative Jordan eigenvalues from bounded operand determinants",
            "psd": "changed per-carrier schedule: eta_eff=min(eta,.5/||Y||_F)",
            "scope": "box <= 2; opt-in automatic symmetric-cone construction; measured norm safeguard",
            "eta_history": {
                "axes": "update, group (rows then columns), statistic",
                "statistics": ["eta_min", "eta_max", "limited_carriers", "carrier_count"],
                "groups": [{"side":side,"index":i,"kind":g.kind,"algebra":g.algebra}
                           for side,groups in (("row",rows),("column",columns)) for i,g in enumerate(groups)],
                "values": np.asarray(answer["actual_step_history"]).tolist()},
            "final_state": "averaged and last states checked against the spectral box"}
    report.update(
        regime="geodesic",
        regime_note="projected stochastic gradient descent in the affine-invariant geometry of "
                    "the positive definite cone, on the Diamond-Boyd equilibration energy lifted "
                    "to block matrices",
        state="one SPD matrix (full), one log-vector (diagonal/shared diagonal) or one log "
              "(scalar) per metric-sharing group, on each side" + (
                  "; one Jordan element w per cone (automorphism)" if automorphic else ""),
        probe_distribution="fresh independent ambient Rademacher vectors, "
                           f"{probes} per side per update, never broadcast per tie",
        estimator="pooled sample Gram of the probe images contracted directly into k-by-k "
                  "storage (the k coordinate energies, for a diagonal group); the per-block "
                  "Grams are summed, never materialized" + (
                      "; for an automorphism group the probe images are mapped by P(w^1/2) and "
                      "only their per-cone energy and first cross moment are accumulated, never a "
                      "k-by-k Gram (PSD: the two partial traces of the Gram)" if automorphic else ""),
        geometry="affine invariant; the step is the exact exponential map "
                 "A <- A^1/2 exp(-eta T) A^1/2, not a retraction" + (
                     "; for an automorphism group the exact geodesic step w <- P(w^1/2) "
                     "exp_J(-eta h) in the Jordan algebra, so every iterate is a P(w) by "
                     "construction" if automorphic else ""),
        step_schedule=f"eta = 2 / ((mu/4)(t + shift)) with mu={float(mu)}, shift={int(shift)}" + (
            "; PSD automorphism only: per-carrier eta_eff=min(eta,.5/||Y||_F)" if automorphic else ""),
        projection=f"exact spectral box projection to [exp(-2*box), exp(2*box)] with box={float(box)}; "
                   "an overflow abandons the step, keeps the previous state and is reported as a "
                   "failure, never as a clipped answer" + (
                       "; for an automorphism group the box is on the Jordan eigenvalues "
                       "[exp(-box), exp(box)], which is the ambient box intersected with {P(w)}"
                       if automorphic else ""),
        exponent_floor={"value": -GEODESIC_EXP_FLOOR,
                        "applies_to": "every eigen-direction of -eta T of a shared full step before "
                                      "the spectral exp; the positive side is never capped",
                        "why": "float32 exp below about -87 is zero on cpu and a denormal on cuda, "
                               "and cuSOLVER's eigh returns NaN on a denormal-scale matrix; every "
                               "floored direction is projected to the box floor exp(-2 box) "
                               "regardless of its value, so the projected iterate is unchanged "
                               "beyond float32 rounding while every intermediate stays a normal "
                               "float32 on both backends"},
        averaging="pre-update geodesic (weighted matrix-geometric) mean with weight t + shift - 1",
        stopping=f"none: a fixed budget of {int(updates)} updates, no residual test",
        rate_claim="NO convergence rate is claimed for this estimator. The averaged iterate is "
                   "returned after the fixed budget. The only bound enforced is the downstream "
                   "step condition, measured and re-measured by the compiler",
        targets=info["targets"],
        constructor_status=status,
        operator_products={"gradient": int(answer["operator_products"]),
                           "normalization": 30 * norm_calls,
                           "total": int(answer["operator_products"]) + 30 * norm_calls,
                           "note": f"2 products per update per side, plus {norm_calls} x 15-iteration "
                                   "power estimate (2 products per iteration)"},
        seconds={"lower": lower_seconds, "compile": compile_seconds, "execute": execute_seconds,
                 "normalization": norm_seconds,
                 "cold_total": lower_seconds + compile_seconds + execute_seconds + norm_seconds,
                 "note": "cold: lowering and compilation are separated from execution; a warm "
                         "replay of the compiled executable costs `execute` alone"},
        state_bytes={"rows": state_bytes(rows), "columns": state_bytes(columns)},
        normalization={"policy": f"factor = safety / measured, safety = {float(safety)}",
                       "measured_before": measured, "factor": factor,
                       "estimator": f"ocs.solver.equilibrate._power_norm, {NORM_ITERS} iterations, "
                                    f"seed {NORM_SEED}, on A^1/2 K B^1/2 (operator-only); a power "
                                    "estimate is a LOWER bound on the norm"},
        effective_metric=effective)
    if column_full:
        report["columns_state"] = (f"one {column_k} x {column_k} SPD shape (full column kind), "
                                   "shared by every column block")
    # A dual step that is one value everywhere is handed over as one value, not
    # as an m-vector of copies: the iteration then broadcasts a scalar instead
    # of reading an m-sized array every update. This is the representation the
    # shipped scalar tier and a hand-supplied scalar sigma both use.
    uniform_sigma = bool(sigma.size and np.all(sigma == sigma[0]))
    report["dual_step_representation"] = ("one scalar (uniform over every term)" if uniform_sigma
                                          else f"one value per row ({m} of them)")
    sigma_out = (jnp.asarray(np.float32(sigma[0])) if uniform_sigma
                 else jnp.asarray(sigma, jnp.float32))
    return (jnp.asarray(tau, jnp.float32), sigma_out, float(measured), declared, report)
