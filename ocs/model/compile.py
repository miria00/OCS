"""compile(): lower a ProblemFamily to a CompiledFamily.

Once per family: route the primal term g (its operator is the identity), stack
the dual operators K_i, run matrix-free DIAGONAL equilibration (respecting each
block's scaling class), group blocks by type for GPU dispatch, and assemble the
engine.  Then solve many instances with changing data (offsets l and cost c) on
the GPU via solve_instance / solve_many, with optional warm starts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
import functools
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from ..operators.common import Identity, Diagonal, Dense, Weighted
from ..operators.structured import BinPack, Gradient3D
from ..operators.fft import FFTMask
from ..prox.base import ProxBlock, _has_oracle
from ..prox.library import SumSquares, Zero, NonNeg, GroupL2, L1
from ..prox.cones import (ExpCone, ExpConeBatch, PowerCone, PowerConeBatch,
                          PSDCone, SecondOrderCone, SOCBatch,
                          PSD_RESOLVED_SIGN_C as _PSD_RESOLVED_SIGN_C)
from ..prox.groups import build_groups, grouped_moreau_dual
from ..prox import block_metric as _block_metric
from ..prox import cone_metric as _cone_metric
from ..prox import spatial_soc as _spatial_soc
from ..solver.composite_pdhg import make_solver, make_online_solver, make_osgm_solver
from ..solver.equilibrate import diagonal_equilibration, scalar_steps, _power_norm
from ..solver.equilibrate_sgd import smfe_equilibration
from ..solver.equilibrate_adv import diagpp_equilibration
from ..solver.stopping import (make_residual_components, make_relative_components, make_dual_repair, NORMS,
                               stable_norm, relative_ratio)
from ..solver.certificates import make_status_checks
from ..solver.domain_checks import make_domain_checks, _indicator, _rows
from ..solver.cone_scales import make_cone_reference_scales
from ..solver.contract import ScaledContract, make_contract_components
from ..solver.refine import gauss_newton_refine
from ..solver.diff import make_diff_solve
from ..solver.fixed_prox import fixed_nonnegative_quadratic_prox
from ..solver import nodc_witness as _nodc_witness
from .._operands import operand_jit, set_window
from ..solver import nodc_tiled as _nodc_tiled       # patch D: host residency helpers


# ---- objective lower bound: reported status codes -----------------------
# A bound is a claim about the OPTIMAL value.  When one cannot be made, the
# report says so with a code, never with a weaker number under a stronger name.
OBJECTIVE_BOUND_AVAILABLE = 0
OBJECTIVE_BOUND_NO_CONTRACT = 1
OBJECTIVE_BOUND_DUAL_NONFINITE = 2
OBJECTIVE_BOUND_COUPLING_VIOLATED = 3
OBJECTIVE_BOUND_BLOCK_DOMAIN_VIOLATED = 4
OBJECTIVE_BOUND_VALUE_NONFINITE = 5
OBJECTIVE_BOUND_ABOVE_PRIMAL = 6

OBJECTIVE_BOUND_STATUS = {
    OBJECTIVE_BOUND_AVAILABLE:
        "available: the supplied dual is exactly dual feasible and its dual "
        "objective is finite",
    OBJECTIVE_BOUND_NO_CONTRACT:
        "unavailable: some block (or g) declares no exact conjugate-domain "
        "contract, so no bound can be established",
    OBJECTIVE_BOUND_DUAL_NONFINITE:
        "unavailable: the supplied dual (or K^T y + c) is not finite",
    OBJECTIVE_BOUND_COUPLING_VIOLATED:
        "unavailable: the coupling condition is violated, i.e. -K^T y - c is "
        "not in dom g^*; for g = 0 that domain is {0}, so exact equality "
        "K^T y + c = 0 is required and no tolerance is admissible",
    OBJECTIVE_BOUND_BLOCK_DOMAIN_VIOLATED:
        "unavailable: some y_i is outside dom f_i^* (exact, tolerance-free test)",
    OBJECTIVE_BOUND_VALUE_NONFINITE:
        "unavailable: the dual objective is not finite",
    OBJECTIVE_BOUND_ABOVE_PRIMAL:
        "unavailable: the computed dual objective exceeds the primal value at "
        "the supplied point, which a valid bound cannot do; the evaluation is "
        "numerically inconsistent and is refused rather than clipped",
}

# What a reported bound is, and is NOT.  Repeated wherever the field is
# documented, because the distinction decides whether a number may be quoted.
#
#   * The predicate that gates the bound is tolerance-free ON THE REPRESENTED
#     FLOATING-POINT DATA.  That is NOT a rigorous real-arithmetic enclosure:
#     the conjugate values and the inner products are themselves rounded, and
#     nothing here proves every computed quantity rounds in the safe direction.
#     Agreement with an independent float64 evaluation is EVIDENCE, not proof.
#   * An objective LOWER BOUND needs no primal point.  A finite GAP needs, in
#     addition, a valid primal UPPER bound, which an infeasible iterate cannot
#     supply: discarding an indicator's +infinity does not make the point
#     feasible.  ``objective_primal_value`` is therefore extended-valued and
#     is an upper bound on the optimum ONLY when it is finite.
OBJECTIVE_BOUND_EPISTEMIC_LIMIT = (
    "tolerance-free on represented float32 data; NOT a rigorous real-arithmetic "
    "enclosure. A finite gap additionally requires a finite (hence feasible) "
    "primal value at the same point.")


@dataclass
class Solution:
    """One solved instance.

    Under ``accuracy="objective_gap"``, ``y`` is the certifying dual selected
    by the final bound check whenever one is available, including unsuccessful
    accuracy outcomes. ``y_iterate`` retains the repaired iteration dual;
    only that iterate can seed the next segment. ``dual_source`` distinguishes
    the two. Certificate statuses (``primal_infeasible``, ``unbounded`` and
    ``numerical_error``) keep their original ``y`` and certificate direction.
    The opt-in residual path is unchanged and returns its iterate. CGLS is a
    documented exception: both dual fields are ``None`` and ``dual_source``
    is ``"none"``. For ``rho > 0``, its objective-gap acceptance uses a
    certified primal suboptimality bound from strong convexity, not the
    dual-witness criterion of the primal-dual paths.
    ``objective_bound_source="strong_convexity"`` identifies that bound.

    ``residual`` and the per-pair facts below are evaluated at the returned
    ``(x, y)``. The relative proximal-graph/KKT residual is not an
    objective-error guarantee. For the legacy Fenchel surrogate (NOT a bound)
    see :meth:`CompiledFamily.certificate`.

    Three facts about the returned point are reported SEPARATELY, because they
    answer three different questions and one status cannot carry all three:

    ``strict_certificate``
        (i) did this pair meet the relative proximal-graph criterion,
        residual <= tol? This diagnostic does not award objective-gap success.
    ``contract_certificate``
        (ii) did this pair meet the explicitly supplied absolute-relative
        contract (``compile_family(certificate_contract=...)``)?  ``None``
        when no contract was declared -- never ``False``, which would claim a
        test was run.
    ``termination``
        (iii) why did the ITERATION stop: ``"criterion"`` (the loop's own
        stopping test passed at the returned point), ``"iteration_limit"``
        (the iteration budget ran out first), ``"time_limit"`` (reserved: this
        engine has no wall-clock budget, so it is never produced) or
        ``"numerical_failure"``, or ``"stalled"`` (the objective-gap
        controller observed insufficient progress; this is not success or
        a proof that future progress is impossible), or ``"unsupported"``
        (the requested bound contract is unavailable, with zero iterations).

    ``status`` is NOT widened by the contract: a point that misses the
    accepted criterion is ``inaccurate`` whatever the contract says.  See
    ``ocs.cvxpy_harness.backend._cvxpy_status`` for the documented mapping to
    CVXPY statuses.

    WHICH CRITERION AWARDED ``status`` is reported in ``accuracy_criterion``
    and, for the primal-dual paths, is the approximate-KKT objective-gap policy named by
    ``ACCEPTANCE_POLICY_ID``: a valid dual bound, relative primal feasibility
    and a relative objective discrepancy within ``tol``. An approximately
    feasible primal point does not itself bound the optimum.
    That test mentions only the returned point and the problem data, so two
    different algorithms can be held to it and compared on it.  The relative
    prox-graph residual is still reported, in ``residual``, but it no longer
    awards ``status``: it is a property of this algorithm's operator splitting
    and not an objective-accuracy claim.  ``"unverified"`` means no bound was
    available, so no accuracy was certified; it is never reported as a solve.
    """
    x: object
    y: object
    iters: int
    residual: float
    status: str = "solved"          # solved | primal_infeasible | unbounded | inaccurate | numerical_error
    certificate: object = None      # Farkas/unboundedness direction when infeasible
    # (i) strict relative criterion, (ii) declared-budget contract, (iii) why
    # the iteration stopped.  Three fields, three questions.
    strict_certificate: object = None
    contract_certificate: object = None
    termination: str = "criterion"
    # The opt-in contract's own ratios (pass <= 1), reported BESIDE the strict
    # ``residual``, which is never replaced.  None when none was declared.
    contract: object = None
    # ---- WHICH CRITERION ACCEPTED THIS POINT ------------------------------
    # ``status`` above is awarded by exactly one criterion, and this field
    # names it.  No reader has to guess, and no comparison can quietly be made
    # between two points accepted on different tests.
    #   "objective_gap"  the default: a strict lower bound on the optimal
    #                    value was certified AT THIS POINT and the relative gap
    #                    to the primal value here met ``tol``.
    #   "fixed_point"    the former default, now reachable only by explicit
    #                    request: the relative prox-graph residual met ``tol``.
    #                    A property of this algorithm's splitting, NOT of the
    #                    point's objective accuracy.
    #   "none"           no accuracy was certified; ``status`` is "unverified".
    #: Which criterion awarded ``status``.  ``"objective_gap"`` means the
    #: acceptance policy named by ``ACCEPTANCE_POLICY_ID`` passed; under the
    #: default that is an approximate-KKT test at the returned point and NOT a
    #: certified bound on the optimal value.  The report's ``claim`` field
    #: states this exactly, and ``target_met_bracket`` carries the stronger
    #: guarantee where one is available. CGLS keeps ``"objective_gap"`` for
    #: its primal strong-convexity bound, identified by ``objective_bound_source``.
    accuracy_criterion: str = "objective_gap"
    # The signed acceptance discrepancy (J(x) - D(y)) / max(|J(x)|, |D(y)|),
    # or None when unavailable. For CGLS, D(y) is replaced by its
    # strong-convexity lower bound. Two zero values have zero discrepancy.
    # This is not an optimal-value enclosure for an approximately feasible x.
    objective_gap: object = None
    objective_lower_bound: object = None
    objective_primal_value: object = None
    # "AX really close to B": the relative primal infeasibility at the
    # returned point, in the same pure relative convention as `residual`'s
    # scaling.  Accepting requires BOTH this and ``objective_gap`` to meet
    # ``tol``, because a duality gap at an infeasible point bounds nothing.
    relative_primal_infeasibility: object = None
    # Why an accuracy claim was withheld or missed, in words.  Empty on a
    # clean accept.
    accuracy_detail: str = ""
    #: Check history used by the unchanged controller. ``legacy_residual``
    #: is evaluated at the repaired ITERATE dual, not the certifying dual.
    #: Its progress governs stalls; it need not equal the returned residual.
    gap_history: object = None
    #: Repaired iteration dual retained for continuation, even when ``y`` is
    #: a recovered certifying dual. None for the dual-free CGLS path.
    y_iterate: object = None
    #: ``"certifying"`` when ``y`` certified the final bound; ``"iterate"``
    #: for the primal-dual iterate, including certificate/residual results;
    #: ``"none"`` for every dual-free CGLS result.
    dual_source: str = "iterate"
    #: Bound provenance when recorded. CGLS records ``"strong_convexity"``
    #: for an available primal suboptimality bound, or ``"unavailable"``.
    #: None when no source is recorded (including residual-only CGLS).
    objective_bound_source: object = None
    #: The bounded refinement policy's record (``compile(refine="auto")``),
    #: or None when the policy was off.  When it is a dict, ``outcome`` says
    #: what happened (``"accepted"``: the returned pair IS the refined pair,
    #: assessed by the unchanged criterion; ``"not_needed"``, ``"not_triggered"``,
    #: ``"unavailable"``, ``"unsupported"``, ``"skipped: ..."`` or ``"refused: ..."``:
    #: the returned pair is the ordinary path's packet, unchanged), ``route``
    #: names the operation (``"primal_dual_refinement"`` or
    #: ``"active_support_solver"``; witness recovery is a different operation
    #: and is never performed here), ``started_from`` is the assessed state
    #: the stage began at and ``work`` carries its operator applications, CG
    #: iterations, assessments and wall seconds so a row can be charged end
    #: to end.  ``iters`` counts PDHG iterations only; a refinement runs none.
    refinement: object = None

    def __post_init__(self):
        if self.y_iterate is None:
            self.y_iterate = self.y


ACCURACY_CRITERIA = ("objective_gap", "residual")

# How the dual-correction rounds of the objective-bound evaluator apply each
# block's interior safety adjustment (the second-order cones' inside-pull of
# (4 k + 16) eps32 on the spatial part):
#   "compound": every round applies the block's ``conj_domain_project``, so a
#               cone block is contracted once per round (32 rounds by default)
#               whether or not the round corrected anything.  The behaviour
#               through 0.3.3; selectable per compile.
#   "once":     the rounds use ``conj_domain_project_exact`` (the projection
#               with no adjustment) and the adjustment is applied one time, by
#               the final round, on the materialized candidate.  The default
#               since 0.3.4.
# Either way the candidate is validated as a whole (every block's exact
# conjugate-domain test and the coupling condition) by the unchanged gates
# before any bound is reported.  ``witness_contraction=None`` selects the
# module default below.
WITNESS_CONTRACTION_POLICIES = ("compound", "once")
WITNESS_CONTRACTION_DEFAULT = "once"

# The dual-bound routes of the objective-gap controller (0.3.6), selected per
# compile by ``dual_routes``.  Each one builds ONE more dual candidate from the
# current iterate, for one structure decided on exact host facts at compile
# time; each candidate is validated by the evaluator's strict-bound gates, and a
# route is consulted only at a check where the evaluator's own packet does not
# meet the criterion.  None of them changes criterion (7), the acceptance
# predicate or any validity gate.
#   "shift":   the recession shift.  The block-wise exact projection of the
#              iterate moved along w (+1 on unrestricted dual blocks, -1 on
#              NonNeg dual blocks, 0 elsewhere), a recession direction of
#              every block domain, by the smallest step that clears the
#              coupling with a rounding allowance.  Eligible when the primal
#              block is NonNeg and some dual block is NonNeg or unrestricted.
#   "packing": the packing-LP assignment dual.  The iterate's resource dual
#              (projected onto >= 0) with the smallest admissible assignment
#              dual for it, formed in scoped float64 and rounded up.  Eligible
#              when the primal block is NonNeg and the only dual block is
#              NonNeg with operator Weighted(BinPack, -1).
#   "ridge":   the compound ridge.  The multiplier of the one SumSquares dual
#              block with operator Diagonal(d > 0) on all n columns,
#              reconstructed from the other blocks (projected into their own
#              domains), formed in scoped float64 and rounded up.  Eligible
#              when the primal block is NonNeg and there is exactly one such
#              block among at least two dual blocks.
DUAL_ROUTES = ("shift", "packing", "ridge")
DUAL_ROUTES_DEFAULT = ("shift", "packing", "ridge")
#: ``Solution.objective_bound_source`` when a route's candidate certified.
DUAL_ROUTE_SOURCES = {"shift": "recession_shift", "packing": "packing_assignment_dual",
                      "ridge": "ridge_nonneg_compound"}
#: Operator applications charged per consultation, counted at the call sites
#: of each route's traced body as the evaluator counts its own (an upper
#: bound: XLA may fuse identical subexpressions).  shift: the forward
#: application for the primal value, the adjoint of the projected iterate, the
#: adjoint of w and the adjoint of the shifted point.  packing: the forward
#: application, the float64 resource-row adjoint of the projected resource
#: dual and the adjoint of the witness.  ridge: the forward application, the
#: adjoint of the non-ridge blocks and the adjoint of the witness.  As in the
#: evaluator's count, the float64 envelope of the strict-bound gate on the
#: guarded Dense and BinPack families is not counted separately.
DUAL_ROUTE_OPERATOR_APPLICATIONS = {"shift": 4, "packing": 3, "ridge": 3}

#: Names the exact normalization in force, so a stored decision can be
#: re-derived later and two reports can be told apart.
#: Three conditions at the RETURNED POINT -- objective gap, primal
#: infeasibility, and exact dual feasibility of the certifying witness --
#: matching Diamond and Boyd arXiv:1506.00760, which ran ECOS, matrix-free SCS
#: and POGS at "absolute and relative tolerance of 10^-3 for the duality gap,
#: l2 norm of the primal residual, and l2 norm of the dual residual".  The
#: stricter rigorous bracket is computed and reported beside it.
ACCEPTANCE_POLICY_ID = "returned-point-three-condition-scaleinv-v2"


def gap_budget(upper, lower, tol_rel, *, maximum=max, absolute=abs):
    """The allowance a gap must meet:  eps_rel * max(|P|, |D|).

    SCALE INVARIANT, which VOICE_NOTES.md l.869 to l.871 require: rescaling
    the problem data must not change the verdict.  Both the objective value
    and the bound carry the objective's units, so numerator and denominator
    scale together and the ratio does not move.

    The previous form, ``max(eps_abs, eps_rel * |P|)``, failed that test.  Its
    absolute floor is an anchor in objective units, so scaling the data down
    drove the gap toward zero while the floor stayed fixed, and eventually
    anything passed.  That is precisely the failure he names: "by multiply A&B
    by 10 to -3, the next equals 0 is a solution. And then I reject that."

    The denominator vanishes only when the objective value and the bound are
    both zero, and then the gap is zero as well and the point is exactly
    optimal, so that case is accepted instead of divided.

    ONE expression, bound twice: with the builtins for host decisions and with
    ``jnp`` for the traced batch.  Writing it out separately in each consumer
    is what let the scalar path, the batch path and the comparator drift
    apart.
    """
    return tol_rel * maximum(absolute(upper), absolute(lower))

#: Unit roundoff of the production dtype.  The bound is a float32 evaluation,
#: so a gap far below this says nothing about the point.
_FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _gap_resolution(primal):
    """The finest acceptance gap float32 can resolve at this objective scale.

    The primal value and the certified bound are float32 numbers of magnitude
    ``|primal|``; subtracting them leaves an absolute error of a few ulps of
    ``|primal|``.  Normalized the way the acceptance gap is normalized, that is
    the floor below which a reported gap carries no information.

    Measured: on a CGLS instance with an exact float64 optimum of
    1.9120513632708755, the float32 bound came out 7.6e-08 ABOVE it while the
    subtraction reported a gap of 1.37e-12.
    """
    scale = abs(float(primal))
    return 4.0 * _FLOAT32_EPS * scale / max(1.0, scale)


def _gap_accepts(available, certified, gap, infeasibility, primal, tol):
    """THE success predicate of the objective-gap criterion.

    True exactly when :meth:`CompiledFamily._award` returns ``"solved"`` for
    these fields, and it is the test both drive loops stop on.  One function
    serves the award, the scalar loop and the batched loop so that the loop
    never halts at a point the award then refuses (the former loop stopped
    on ``gap <= tol`` while the award refused a gap within the float32
    resolution of ``tol`` as ``unverified``) and never runs past a point the
    award would accept.

    The three conditions, in this order, are the documented criterion: the
    witness is valid (``available``) and the cost at the returned point is
    finite (``certified``); the returned point is primal feasible to ``tol``
    in the relative sense; and the objective discrepancy is within ``tol``
    in MAGNITUDE, with a nonnegative discrepancy within the float32
    resolution of ``tol`` counted as neither met nor missed.  Feasibility is
    tested before the sign of the discrepancy: a negative discrepancy at an
    infeasible point bounds nothing, exactly as a positive one does.

    Works elementwise on JAX arrays (the batched loop, under tracing) and on
    Python scalars (the scalar loop and the award); the arithmetic is the
    same expression in both cases.  Returns a bool for scalar inputs.
    """
    traced = any(isinstance(a, jax.Array)
                 for a in (available, certified, gap, infeasibility, primal))
    xp = jnp if traced else np
    with np.errstate(invalid="ignore", divide="ignore"):
        # A non-finite objective value gives a non-finite floor, and every
        # comparison against it is False, exactly as `_gap_resolution` did.
        scale = xp.abs(primal)
        floor = 4.0 * _FLOAT32_EPS * scale / xp.maximum(1.0, scale)
        finite = xp.isfinite(gap)
        feasible = infeasibility <= tol
        negative_ok = (gap < 0.0) & (-gap <= tol)
        band = xp.abs(gap - tol) < floor
        nonneg_ok = (gap >= 0.0) & ~band & (gap <= tol)
        ok = (xp.asarray(available, bool) & xp.asarray(certified, bool) & finite
              & feasible & (negative_ok | nonneg_ok))
    return ok if traced else bool(ok)


def _bound_row(report, i):
    """One row of a batched objective-bound report, in the scalar form.

    The scalar and batch paths then make their decision from the SAME dict
    shape through the same ``_award``, which is what keeps them in step.
    """
    row = {}
    for key, value in report.items():
        v = value[i]
        row[key] = bool(v) if getattr(v, "dtype", None) == bool else float(v)
    code = int(row.pop("supplied_dual_status_code", OBJECTIVE_BOUND_NO_CONTRACT))
    source = int(row.get("source_code", 0))
    row["source"] = {0: "none", 1: "violation_corrected",
                     2: "residual_scaled", 3: "feasible_anchor",
                     4: "supplied_witness", 5: "zero_feasibility",
                     6: "retained_witness", 7: "equality_projected"}[source]
    row["bound_kind"] = "strict_dual" if row.get("objective_lower_bound_available") else "none"
    row["arithmetic"] = "mixed" if row.get("assessment_uses_float64", False) else "float32"
    if not row.get("objective_lower_bound_available", False):
        row["unavailable_reason"] = (
            "no candidate direction passed the dual domain/value and arithmetic "
            "checks; the supplied dual failed because: "
            + OBJECTIVE_BOUND_STATUS.get(code, "unknown"))
    elif not row.get("relative_gap_certified", False):
        row["gap_unavailable_reason"] = (
            "the bound is valid, but the primal value at the supplied x is not "
            "finite, so there is no valid primal upper bound and no finite "
            "optimality gap follows")
    return row


class _HostHeldDual:
    """Patch L: a device certifying dual held on the host between gap checks.

    Its bytes are copied to the host when the check that produced it ends and
    copied back, unchanged, when the gap loop ends.  Nothing reads it in between.
    The device array is not deleted here, only no longer referenced: jit can
    forward an unchanged input, so the witness may BE the loop's own y."""
    __slots__ = ("host",)

    def __init__(self, value):
        self.host = np.asarray(jax.device_get(value))

    def restore(self):
        return jnp.asarray(self.host)


def _retain_stronger_lower(current, previous, *, batched=False):
    """Keep a previously checked lower for the SAME fixed problem data.

    Only the bound and its provenance are retained; cost and feasibility are
    always those of the current x. Caller retains the matching dual bytes.
    Assessment counts still describe the newly paid evaluation. This is not a
    cache across separate calls, parameter changes, or model bindings.
    """
    xp = jnp if batched else np
    out = dict(current)
    if batched:
        out.setdefault('source_code', xp.zeros_like(current['objective_lower_bound']))
    if previous is None:
        take = xp.zeros_like(current['objective_lower_bound_available'], dtype=bool)
    else:
        take = (xp.asarray(previous['objective_lower_bound_available']) &
                (~xp.asarray(current['objective_lower_bound_available']) |
                 (xp.asarray(previous['objective_lower_bound']) >
                  xp.asarray(current['objective_lower_bound']))))
        for key in ('objective_lower_bound', 'certifying_dual_coupling_defect'):
            if key in current and key in previous:
                out[key] = xp.where(take, previous[key], current[key])
        out['objective_lower_bound_available'] = (
            xp.asarray(current['objective_lower_bound_available']) | take)
        if batched:
            out['source_code'] = xp.where(take, 6, out['source_code'])
        elif bool(take):
            out['source'] = 'retained_witness'
            out['bound_kind'] = 'strict_dual'
            out['objective_bound_status_code'] = OBJECTIVE_BOUND_AVAILABLE
            out['objective_bound_status'] = OBJECTIVE_BOUND_STATUS[OBJECTIVE_BOUND_AVAILABLE]
            out.pop('unavailable_reason', None)
        lower = out['objective_lower_bound']
        raw = out.get('objective_at_returned_point', out['objective_primal_value'])
        finite_raw = xp.isfinite(raw) & xp.isfinite(lower)
        safe_raw = xp.where(finite_raw, raw, 0.)
        safe_lower = xp.where(finite_raw, lower, 0.)
        scale = xp.maximum(xp.abs(safe_raw), xp.abs(safe_lower))
        out['gap_at_returned_point'] = xp.where(
            out['objective_lower_bound_available'] & finite_raw,
            (safe_raw-safe_lower)/xp.where(scale>0, scale, 1.), xp.nan)
        upper = out.get('objective_primal_upper_bound', out['objective_primal_value'])
        finite = xp.isfinite(upper) & xp.isfinite(lower)
        safe_upper = xp.where(finite, upper, 0.)
        safe_lower = xp.where(finite, lower, 0.)
        out['relative_gap_certified'] = out['objective_lower_bound_available'] & finite
        out['relative_gap'] = xp.where(finite & (xp.abs(upper)>0),
            (safe_upper-safe_lower)/xp.where(xp.abs(safe_upper)>0, xp.abs(safe_upper), 1.), xp.nan)
    out['retained_previous_lower'] = take
    return out, take


def _check_schedule(max_blocks, first=1, growth=2.0, max_checks=64):
    """Block counts between successive gap checks: 1, 2, 4, 8, ...

    Geometric rather than uniform so that a problem which reaches the
    requested accuracy early is not carried to the end of a coarse segment,
    and so that a problem which does not is checked only a logarithmic number
    of times.  The counts sum to exactly ``max_blocks``: the TOTAL ITERATION
    BUDGET IS UNCHANGED by the criterion.
    """
    out, total, seg = [], 0, max(1, int(first))
    cap = max(0, int(max_blocks))          # a zero budget yields NO segments
    while total < cap and len(out) < max_checks:
        seg = min(seg, cap - total)
        out.append(int(seg))
        total += int(seg)
        seg = max(1, int(seg * growth))
    if total < cap and out:                     # max_checks reached first
        out[-1] += cap - total
    return out


def _check_accuracy(value):
    if value not in ACCURACY_CRITERIA:
        raise ValueError("accuracy must be one of %r, not %r"
                         % (ACCURACY_CRITERIA, value))
    return value


def _check_witness_contraction(value):
    if value is None:
        value = WITNESS_CONTRACTION_DEFAULT
    if value not in WITNESS_CONTRACTION_POLICIES:
        raise ValueError("witness_contraction must be one of %r, not %r"
                         % (WITNESS_CONTRACTION_POLICIES, value))
    return value


def _check_dual_routes(value):
    """The requested dual-bound routes as a tuple in ``DUAL_ROUTES`` order.
    ``None`` takes ``DUAL_ROUTES_DEFAULT``; ``()`` turns every route off."""
    if value is None:
        return tuple(DUAL_ROUTES_DEFAULT)
    if isinstance(value, (str, bytes)) or isinstance(value, bool):
        raise ValueError("dual_routes must be a tuple of route names from %r (() turns "
                         "every route off), not %r" % (DUAL_ROUTES, value))
    try:
        names = tuple(value)
    except TypeError:
        raise ValueError("dual_routes must be a tuple of route names from %r (() turns "
                         "every route off), not %r" % (DUAL_ROUTES, value)) from None
    unknown = [name for name in names if name not in DUAL_ROUTES]
    if unknown:
        raise ValueError("dual_routes: unknown route(s) %r; the routes are %r"
                         % (unknown, DUAL_ROUTES))
    if len(set(names)) != len(names):
        raise ValueError("dual_routes names a route more than once: %r" % (names,))
    return tuple(name for name in DUAL_ROUTES if name in names)


#: The refinement policies: off, or ONE bounded stage on the stall guard.
REFINE_POLICIES = (False, "auto")


def _check_refine_policy(value):
    if value is None or value is False or value == "off":
        return False
    if value is True:
        raise ValueError("refine=True is ambiguous; use refine='auto' (one bounded stage "
                         "on the stall guard, assessed by the unchanged criterion) or False")
    if value not in REFINE_POLICIES:
        raise ValueError("refine must be False or 'auto', not %r" % (value,))
    return value


def _overrides(block, name):
    return _has_oracle(block, name)


@operand_jit
def _unstack_batch_points(x, y):
    """Extract fixed batch rows in one compiled call, keeping results on device."""
    return tuple((x[i], y[i]) for i in range(x.shape[0]))


def _resolve_certificate_batch(resolve, x, y, data_batch, n, m=None):
    """Validate public certificate batch axes before tracing the evaluator."""
    x = jnp.asarray(x)
    if x.ndim != 2 or x.shape[1] != n:
        raise ValueError(f"x must have shape (batch, {n})")
    if m is not None:
        y = jnp.asarray(y)
        if y.ndim != 2 or y.shape != (x.shape[0], m):
            raise ValueError(f"y must have shape (batch, {m}) with the same batch as x")
    raw = {} if data_batch is None else data_batch
    for value in jax.tree_util.tree_leaves(raw):
        shape = jnp.shape(value)
        if not shape or shape[0] != x.shape[0]:
            raise ValueError("every data_batch leaf must have the same leading batch axis as x")
    data = jax.vmap(resolve, axis_size=x.shape[0])(raw)
    return x, y, data


def _make_batch_status_checks(infeas, *, batched=True):
    """Classify rows on device without retaining batch-sized ray outputs.

    Codes are 0=ordinary, 1=invalid, 2=primal infeasible, 3=unbounded.
    Ordinary rows retain the host's Python-float tolerance comparison.  In
    particular, casting that tolerance to float32 can change its boundary.
    Certificate rows are checked again by the scalar path when their actual
    direction is requested, rather than returning an unchecked reconstruction.
    """
    def classify(x, y, data, residual):
        valid = ~jnp.isnan(residual) & (residual >= 0)
        for a in (x, y, *jax.tree_util.tree_leaves(data)):
            valid = valid & jnp.all(jnp.isfinite(a))
        code = jnp.asarray(0, jnp.int32)
        if infeas is not None and getattr(infeas, "status_may_certify", True):
            checks = infeas(x, y, data)
            code = jnp.where(checks["unbounded"], 3, code)
            code = jnp.where(checks["primal_infeasible"], 2, code)
        return jnp.where(valid, code, 1)

    return operand_jit(jax.vmap(classify) if batched else classify)


class CompiledFamily:
    #: The enabled dual-bound routes (``compile(dual_routes=...)``):
    #: {route name: (jitted route, operator applications)}, set by
    #: compile_family after construction and empty when no requested route is
    #: eligible.  Consulted only at checks where the evaluator's own packet
    #: does not meet the criterion; see ``_extend_with_routes``.
    _dual_routes = None

    def _extend_with_routes(self, ob, x, y, dense):
        """The evaluator's packet ``ob`` (after its own retention), with each
        enabled dual-bound route applied to the SAME iterate offered as one
        more candidate under the same rule: valid AND strictly better, or the
        incumbent stands.  Each route's work is charged: wall seconds
        (synchronized) and its operator applications.  Gap fields are
        recomputed by the same formulas as ``_retain_stronger_lower``."""
        import time as _time
        out = dict(ob)
        report, best = {}, None
        for name in sorted(self._dual_routes):
            fn, n_ops = self._dual_routes[name]
            t0 = _time.perf_counter()
            res = fn(x, y, dense)
            jax.block_until_ready(res["bound"])
            seconds = _time.perf_counter() - t0
            valid = bool(res["valid"])
            bound = float(res["bound"])
            report[name] = {"valid": valid, "bound": bound, "scale": float(res["s"]),
                            "status_code": int(res["status"]),
                            "status": OBJECTIVE_BOUND_STATUS.get(int(res["status"]), "unknown"),
                            "certifying_dual_coupling_defect": float(res["defect"]),
                            "allowance_max": float(res["margin_max"]), "need_max": float(res["need_max"]),
                            "seconds": seconds, "operator_applications": int(n_ops)}
            if valid and (best is None or bound > best[1]):
                best = (name, bound, res)
        incumbent_ok = bool(ob.get("objective_lower_bound_available", False))
        take = best is not None and (not incumbent_ok
                                     or best[1] > float(ob.get("objective_lower_bound", float("nan"))))
        out["dual_routes"] = report
        out["dual_route_taken"] = best[0] if take else None
        out["dual_route_seconds"] = sum(r["seconds"] for r in report.values())
        out["dual_route_operator_applications"] = sum(r["operator_applications"] for r in report.values())
        if not take:
            return out
        name, bound, res = best
        out["objective_lower_bound"] = bound
        out["objective_lower_bound_available"] = True
        out["certifying_dual"] = res["dual"]
        out["certifying_dual_coupling_defect"] = float(res["defect"])
        out["source"] = DUAL_ROUTE_SOURCES[name]
        out["bound_kind"] = "strict_dual"
        out["objective_bound_status_code"] = OBJECTIVE_BOUND_AVAILABLE
        out["objective_bound_status"] = OBJECTIVE_BOUND_STATUS[OBJECTIVE_BOUND_AVAILABLE]
        out.pop("unavailable_reason", None)
        raw = float(out.get("objective_at_returned_point", out.get("objective_primal_value", float("nan"))))
        if math.isfinite(raw):
            scale = max(abs(raw), abs(bound))
            out["gap_at_returned_point"] = (raw - bound) / (scale if scale > 0 else 1.0)
        upper = float(out.get("objective_primal_upper_bound", out.get("objective_primal_value", float("nan"))))
        finite = math.isfinite(upper)
        out["relative_gap_certified"] = finite
        out["relative_gap"] = ((upper - bound) / abs(upper)) if (finite and abs(upper) > 0) else float("nan")
        if finite:
            out.pop("gap_unavailable_reason", None)
        return out

    def __init__(self, solver, resolve_data, report, n, m, meta, refine=None,
                 infeas=None, tol=1e-3, diff_solve=None, report_device=None, resolve_dense=None,
                 single_solver=None, objective_bound=None, contract=None,
                 contract_stops_loop=False, max_iters=None, contract_factory=None,
                 accuracy="objective_gap", bound_capable=False, drive_solver=None,
                 single_drive_solver=None, gap_checks=64, bound_rounds=32,
                 gap_check_growth=2.0, gap_check_first=1,
                 max_blocks=None, objective_bound_many=None,
                 returned_components=None, refine_policy=False, refine_budget=None,
                 refine_memory_bytes=None, refine_structure=None):
        self._solver = solver
        # ---- the bounded refinement policy (compile(refine=...)) ----------
        # Off by default.  "auto": after the ordinary objective-gap run ends
        # on the stall guard with the request unmet, ONE bounded refinement
        # stage on a supported structure, assessed by the unchanged criterion;
        # see ocs.solver.refine_auto.  The structure record is what the route
        # detection reads (original-coordinate operators and blocks).
        self._refine_policy = _check_refine_policy(refine_policy)
        self._refine_budget = refine_budget
        self._refine_memory_bytes = refine_memory_bytes
        self._refine_structure = refine_structure
        self._refine_cache = {}
        # Test-only override of the stage's proposals (see refine_auto).
        self._refine_test_hook = None
        self._resolve = resolve_data
        self._resolve_dense = resolve_data if resolve_dense is None else resolve_dense
        self._report = report
        # ``report_device`` is the UN-JITTED report body.  vmap must trace
        # the body itself: its guarded objective-bound assessment runs in a
        # scoped float64 context, and batching an already jitted evaluator
        # re-traces that block outside the scope (float64 Dense products
        # truncated to float32, then a mixed-dtype add).
        self._report_many = (operand_jit(jax.vmap(report_device))
                             if report_device is not None else None)
        self._refine = refine
        self._objective_bound = objective_bound
        self._infeas = infeas
        self._tol = tol
        self._diff_solve = diff_solve
        self._batch_status_checks = _make_batch_status_checks(infeas)
        self._scalar_status_checks = _make_batch_status_checks(infeas, batched=False)
        self.n, self.m = n, m
        self.meta = meta
        # Opt-in scaled contract evaluator (x, y, data) -> ratios; None by
        # default.  It never stops the loop unless the caller declared
        # stopping="strict_or_contract" (experimental).
        self._contract = contract
        self._contract_many = (operand_jit(jax.vmap(contract)) if contract is not None else None)
        self._contract_stops_loop = bool(contract_stops_loop)
        # Builds a contract evaluator for ANY declared budgets, so that a
        # caller can score an already-returned point under budgets it did not
        # compile with.  It never re-solves and never touches the loop.
        self._contract_factory = contract_factory
        self._contract_cache = {}
        # Iteration budget, used only to answer "why did the iteration stop".
        self._max_iters = max_iters
        # ---- the ACCEPTANCE criterion ---------------------------------
        # Default: a certified objective gap at the returned point.  The
        # prox-graph residual is still computed and still reported, but it no
        # longer awards ``status``.
        self._accuracy = _check_accuracy(accuracy)
        self._bound_capable = bool(bound_capable)
        # Executables that run a FIXED segment of blocks and never stop on the
        # residual, so that only the gap decides when to stop.  None when the
        # family cannot certify a bound, in which case the gap mode reports
        # "unverified" instead of inventing one.
        self._drive = drive_solver
        self._single_drive = single_drive_solver
        # Patch L (lean live layout): the single-instance gap loop drives the batched
        # executable through ONE program that adds and removes the batch axis itself,
        # so no batch-of-one copies of the data, of x or of y are made outside it, and
        # whose starts are DONATED (the loop passes only package-owned starts), so the
        # segment's outputs reuse the start buffers.  The batched drive inside is the
        # same executable body; nothing in the iteration changes.
        self._drive_flat = None
        if drive_solver is not None and single_drive_solver is None:
            batched_drive = drive_solver

            def flat_drive(data, x0, y0, cap):
                xb, yb, nb, rb = batched_drive(
                    jax.tree_util.tree_map(lambda a: a[None], data), x0[None], y0[None], cap)
                return xb[0], yb[0], nb[0], rb[0]

            self._drive_flat = operand_jit(flat_drive, donate_argnums=(1, 2))
        self._objective_bound_many = objective_bound_many
        # Reporting only, after the controller has stopped. Never repairs y,
        # changes acceptance, or feeds the returned witness into the iteration.
        self._returned_components = (operand_jit(returned_components)
                                    if returned_components is not None else None)
        self._returned_components_many = (operand_jit(jax.vmap(returned_components))
                                         if returned_components is not None else None)
        self._gap_checks = max(1, int(gap_checks))
        self._gap_check_growth = float(gap_check_growth)
        self._gap_check_first = max(1, int(gap_check_first))
        self._bound_rounds = int(bound_rounds)
        self._max_blocks = (int(max_blocks) if max_blocks is not None else
                            (int(max_iters) // int(meta["block"])
                             if max_iters is not None else 1))

        @operand_jit
        def cold_solver(data):
            # These zero starts are private to this invocation. Constructing
            # them inside the executable avoids retaining separate input
            # buffers throughout the solve; caller-owned warm starts still
            # use the ordinary, non-donating solver.
            count = jax.tree_util.tree_leaves(data)[0].shape[0]
            return solver(data, jnp.zeros((count, n), jnp.float32),
                          jnp.zeros((count, m), jnp.float32))

        self._cold_solver = cold_solver
        self._single_solver = single_solver
        self._single_cold_solver = None
        if single_solver is not None:
            @operand_jit
            def single_cold_solver(data):
                # Private starts, as in the batched path; no caller buffer is
                # donated. Omitting vmap avoids inactive-row state preservation
                # when the public call contains exactly one instance.
                return single_solver(data, jnp.zeros(n, jnp.float32),
                                     jnp.zeros(m, jnp.float32))
            self._single_cold_solver = single_cold_solver

    def solve_diff(self, params=None):
        """Differentiable solve: returns x*(params), differentiable w.r.t. the
        parameter values via implicit differentiation of the optimality map
        (jax.grad / jax.jacobian compose through it). Operator-native; not for
        exp/power blocks."""
        if self._diff_solve is None:
            raise NotImplementedError("differentiable solve not available")
        l, c = self._resolve_dense(params or {})
        x, _y = self._diff_solve(l, c)
        return x

    def _status(self, x, y, data, blocks, res):
        """Accept finite points or checked domain-aware certificate directions."""
        # Keep host residual semantics, including negative values too small
        # to survive a float32 conversion. Positive infinity can still carry
        # a valid infeasibility or unboundedness certificate.
        if math.isnan(res) or res < 0:
            return "numerical_error", None
        code = int(self._scalar_status_checks(x, y, data, res))
        if code == 1:
            return "numerical_error", None
        if code in (2, 3):
            # The ordinary path returns only a code: XLA can discard large
            # diagnostic arrays. Recheck actual directions through the full
            # scalar checker before returning either kind of certificate.
            checks = self._infeas(x, y, data)
            if bool(checks["primal_infeasible"]):
                return "primal_infeasible", checks["dual_direction"]
            if bool(checks["unbounded"]):
                return "unbounded", checks["primal_direction"]
        return ("solved" if res <= self._tol else "inaccurate"), None

    # ---- the three separately reportable facts --------------------------
    # (i) strict, (ii) declared-budget contract, (iii) termination reason.
    # Each is computed from its own evidence; none is derived from another,
    # and none of them changes ``status``.
    def _contract_values(self, x, y, data):
        """(ii) the declared-budget contract's ratios, or None if undeclared."""
        if self._contract is None:
            return None
        return {k: float(v) for k, v in self._contract(x, y, data).items()}

    @staticmethod
    def _contract_passed(values):
        return None if values is None else bool(values["contract"] <= 1.0)

    def _strict_passed(self, status, res):
        """(i) did this pair meet the EXISTING relative criterion?"""
        if status == "numerical_error":
            return False
        return bool(not math.isnan(res) and res >= 0 and res <= self._tol)

    def _termination(self, status, iters, res, contract_pass):
        """(iii) why the ITERATION stopped.

        ``"criterion"`` means the loop's own stopping test passed at the
        returned point.  ``"iteration_limit"`` means the budget ran out first;
        such a run is NEVER relabelled a solve, whatever the contract says.
        ``"time_limit"`` is reserved: this engine carries no wall-clock budget,
        so it is never produced here.
        """
        if status == "numerical_error":
            return "numerical_failure"
        if self._max_iters is not None and int(iters) < int(self._max_iters):
            # The loop exited before its budget, so its own test passed.
            return "criterion"
        if not math.isnan(res) and res >= 0 and res <= self._tol:
            return "criterion"
        if self._contract_stops_loop and contract_pass:
            return "criterion"
        return "iteration_limit" if self._max_iters is not None else "criterion"

    def _facts(self, x, y, data, status, iters, res):
        values = self._contract_values(x, y, data)
        passed = self._contract_passed(values)
        return {"strict_certificate": self._strict_passed(status, res),
                "contract_certificate": passed,
                "termination": self._termination(status, iters, res, passed),
                "contract": values}

    def refine(self, x, y, data=None, steps=3, cg_iters=80):
        """Refine an approximate primal-dual pair (x, y) toward high accuracy by
        matrix-free Gauss-Newton on the fixed-point residual (see refinement).
        Returns the refined (x, y).  Operator-pure; not for exp/power blocks."""
        if self._refine is None:
            raise NotImplementedError("refinement not available for this family")
        xr, yr = self._refine(x, y, self._resolve_dense(data or {}), steps, cg_iters)
        return xr, yr

    def fixed_point_residual(self, x, y, data=None):
        """The residual map R(x, y) whose zero is an optimal pair (see refine).

        Read-only: two operator applications, no solve, no state. Used by the
        safeguarded refinement wrapper; ``refine`` itself is unchanged."""
        residual = getattr(self._refine, "residual", None)
        if residual is None:
            raise NotImplementedError("refinement residual not available for this family")
        return residual(x, y, self._resolve_dense(data or {}))

    def _run(self, data_b, x0, y0):
        return self._solver(data_b, x0, y0)

    # ---- the bounded refinement policy ------------------------------------
    def _refine_policy_for(self, refine):
        return self._refine_policy if refine is None else _check_refine_policy(refine)

    def _maybe_refine(self, result, d, dense, *, stalled, x_iterate, y_iterate,
                      policy, batch_lane=None):
        """Apply the ``refine="auto"`` table to ONE assessed packet.

        Returns ``result`` unchanged when the policy is off.  Otherwise the
        packet comes back with ``refinement`` set: either the ordinary packet
        with the outcome recorded (nothing else about it changes), or, when
        one bounded stage produced a pair that passes the UNCHANGED criterion,
        the refined assessed pair.  See ``ocs.solver.refine_auto``.
        """
        if not policy:
            return result
        from ..solver import refine_auto
        return refine_auto.apply_policy(self, result, d, dense, stalled=bool(stalled),
                                        x_iterate=x_iterate, y_iterate=y_iterate,
                                        batch_lane=batch_lane)

    def solve_with_refinement(self, data=None):
        """Fresh ordinary solve with one explicitly requested bounded correction.

        Automatic refinement is disabled during the ordinary solve. An unmet,
        assessable iteration-limit or stall return can receive one bounded
        correction consuming that actual returned pair. The same assessment
        validates the corrected pair. Refusal preserves the ordinary result;
        Solution.refinement records both phases and their work and costs.
        This opt-in entry does not change solve_instance's default policy.
        """
        from ..solver.refine_auto import solve_with_refinement
        return solve_with_refinement(self, data)

    def refine_assessed(self, x, y, data=None):
        """Run the ``refine="auto"`` stage once on a supplied pair, without
        any PDHG iteration: assess (x, y) by the unchanged criterion and, if
        the request is unmet and the structure is supported, attempt ONE
        bounded refinement.  Returns a :class:`Solution` whose ``refinement``
        record states the outcome; ``iters`` is 0 (zero optimizer updates).
        The stall trigger does not apply: the caller asked for the stage."""
        from ..solver import refine_auto
        return refine_auto.refine_assessed(self, x, y, data)

    def solve_instance(self, data=None, warm_start=None, *, accuracy=None, refine=None):
        """Solve one instance and accept it on a CERTIFIED OBJECTIVE GAP.

        The default criterion is ``accuracy="objective_gap"``: ``status`` is
        ``"solved"`` only when a strict lower bound on the optimal objective
        value is certified at the returned point AND the relative gap to the
        primal value there meets ``tol``.  The test names only the returned
        pair and the problem data.  It does not mention this algorithm, its
        metric, its step sizes or its iterate sequence, which is what makes it
        a criterion another solver can be held to and compared on.

        ``accuracy="residual"`` restores the former default, in which
        ``status`` came from the relative prox-graph (fixed-point) residual.
        That quantity measures how nearly the returned pair is a fixed point of
        THIS splitting; it is not an objective-error bound and not a duality
        gap, which is why it no longer awards ``status``.  It stays available,
        explicitly, for reproducing earlier runs.

        When the family declares no exact conjugate-domain contract, no
        accuracy can be certified and ``status`` is ``"unverified"``: never
        ``"solved"``, and never a surrogate in place of the bound.

        The total iteration budget is unchanged.  The existing block budget is
        divided into segments, the gap is evaluated at each segment boundary,
        and the run stops at the first point that meets it.

        ``refine`` overrides the compiled refinement policy for this call
        (``False`` or ``"auto"``; None keeps ``compile(refine=...)``).  Under
        ``"auto"``, a run that ends on the stall guard with the request unmet
        gets ONE bounded refinement stage on a supported structure, assessed
        by the same criterion; the outcome is in ``Solution.refinement``.
        The residual path never refines.
        """
        criterion = self._accuracy if accuracy is None else _check_accuracy(accuracy)
        policy = self._refine_policy_for(refine)
        if criterion == "residual":
            return self._solve_on_residual(data, warm_start)
        return self._solve_on_objective_gap(data, warm_start, policy=policy)

    def acceptance(self, report, tol=None, *, tol_abs=None, tol_rel=None):
        """THE acceptance decision, for any report, from any producer.

        Every interface must route through this rather than re-deriving a
        decision from a field named ``relative_gap``: that field keeps its own
        published pure-relative meaning and is NOT the acceptance quantity.

        Returns the full packet the decision was made on, so the caller can
        record what was asked and what was measured, not just the verdict.
        """
        tol = self._tol if tol is None else float(tol)
        tol_abs = tol if tol_abs is None else float(tol_abs)
        tol_rel = tol if tol_rel is None else float(tol_rel)
        available = bool(report.get("objective_lower_bound_available", False))
        certified = bool(report.get("relative_gap_certified", False))
        lower = float(report.get("objective_lower_bound", float("nan")))
        # The DECIDING quantity (his protocol) and the stricter bracket,
        # reported together so the contract in force is visible.
        upper = float(report.get("objective_at_returned_point",
                                 report.get("objective_primal_upper_bound",
                                            report.get("objective_primal_value",
                                                       float("nan")))))
        bracket_upper = float(report.get("objective_primal_upper_bound",
                                         report.get("objective_primal_value",
                                                    float("nan"))))
        infeas = float(report.get("relative_primal_infeasibility", 0.0))
        out = {"policy_id": ACCEPTANCE_POLICY_ID,
               # WHAT `target_met` DOES AND DOES NOT ASSERT.  The deciding
               # quantity is evaluated at the RETURNED point, whose objective
               # bounds nothing when that point is not exactly feasible, so
               # this is an approximate-KKT pass of the kind SCS and OSQP
               # accept on -- NOT a certified objective bound.  It is
               # meaningful only together with
               # `relative_primal_infeasibility`.  `target_met_bracket` is the
               # one that carries weak duality, and it is reported beside it.
               "claim": ("approximate-KKT: the original cost at the returned "
                         "point, the RELATIVE primal infeasibility, and an "
                         "exactly dual-feasible witness, each within its "
                         "allowance. This is the SCS and OSQP shape and the "
                         "one Diamond and Boyd used. It is NOT a bound on the "
                         "optimal value: an infeasible point's cost bounds "
                         "nothing. See objective_bracket_valid and "
                         "target_met_bracket, which are answered separately "
                         "and are claimed only where an upper bound is "
                         "actually justified."),
               "tol_abs": tol_abs, "tol_rel": tol_rel, "feasibility_tol": tol,
               # NAMED FOR WHAT IT REPORTS: the quantity the SELECTED
               # criterion needs is usable.  This is NOT the bracket's
               # validity, which is answered separately by
               # `objective_bracket_valid`.
               "acceptance_quantity_valid": available and certified,
               # Compatibility alias, same value, older name.
               "bound_valid": available and certified,
               "objective_lower_bound": lower,
               "objective_primal_upper_bound": upper,
               "primal_feasible": bool(infeas <= tol),
               "relative_primal_infeasibility": infeas,
               "absolute_gap": None, "gap_budget": None, "target_met": False,
               "reason": ""}
        # The APPROXIMATE-KKT quantity: the original cost at the returned
        # point.  Valid when a dual bound exists and both values are finite.
        out["acceptance_quantity_valid"] = bool(
            available and math.isfinite(lower) and math.isfinite(upper))
        out["bound_valid"] = out["acceptance_quantity_valid"]
        # The BRACKET needs a JUSTIFIED upper bound.  A finite
        # `objective_primal_value` does not supply one: the indicators award
        # a finite value whenever the residual sits under an ABSOLUTE
        # threshold, which is not feasibility.  It is justified only when the
        # point is exactly feasible, or when the upper came from a point
        # CONSTRUCTED to be feasible.
        from_restored = bool(report.get("primal_upper_from_restored_point", False))
        out["objective_bracket_valid"] = bool(
            available and math.isfinite(bracket_upper) and math.isfinite(lower)
            and (from_restored or infeas == 0.0))
        out["bracket_upper_justified_by"] = (
            "restored feasible point" if from_restored else
            "exactly feasible returned point" if infeas == 0.0 else
            "NOT JUSTIFIED: a finite extended-value objective here only means "
            "the residual is under an absolute indicator threshold, which is "
            "not feasibility")
        if not out["acceptance_quantity_valid"]:
            out["reason"] = ("no valid bound, or the original cost at the "
                             "returned point is not finite")
            return out
        delta = upper - lower
        budget = gap_budget(upper, lower, tol_rel)
        out["absolute_gap"] = delta
        out["gap_budget"] = budget
        # Feasibility first: no discrepancy, of any sign or size, can accept a
        # point that is not feasible within its allowance.
        if not out["primal_feasible"]:
            out["reason"] = "primal infeasibility exceeds its allowance"
            return out
        # MAGNITUDE, matching `_award`.  The approximate-KKT quantity is the
        # cost at a point that may be only approximately feasible, so it is
        # not an upper bound and may sit below a valid lower bound with
        # nothing wrong.  Testing the signed value would reject an ordinary
        # small negative discrepancy; testing `delta <= budget` or
        # `max(0, delta) <= budget` would ACCEPT delta = -0.2 at a budget of
        # 1e-3.  Neither is right, so the test is |delta| <= budget.  The
        # signed-inconsistency reading belongs to the bracket below.
        scale = max(abs(upper), abs(lower))
        discrepancy = delta / scale if scale > 0.0 else 0.0
        out["target_met"] = _gap_accepts(
            available, out["acceptance_quantity_valid"], discrepancy,
            0.0, upper, tol_rel)
        out["arithmetic_boundary_unresolved"] = bool(
            discrepancy >= 0.0 and abs(discrepancy - tol_rel) < _gap_resolution(upper))
        out["reason"] = ("within budget" if out["target_met"] else
                         "arithmetic boundary unresolved" if out["arithmetic_boundary_unresolved"] else
                         "objective discrepancy exceeds budget in magnitude")
        # The stricter rigorous verdict, retained and reported.  It demands an
        # EXACTLY feasible upper bound, which an approximately feasible point
        # only reaches by paying objective value for a restoration, so it can
        # fail where his three-condition protocol passes.  Reported, never
        # discarded, and never the decider.
        if out["objective_bracket_valid"]:
            bdelta = bracket_upper - lower
            bbudget = gap_budget(bracket_upper, lower, tol_rel)
            out["bracket_gap"] = bdelta
            out["bracket_budget"] = bbudget
            out["target_met_bracket"] = bool(0.0 <= bdelta <= bbudget)
        else:
            # An unavailable bracket NEVER falls through to the
            # approximate-KKT verdict.  The two requests are answered
            # separately and neither inherits the other's guarantee.
            out["bracket_gap"] = None
            out["bracket_budget"] = None
            out["target_met_bracket"] = False
        return out

    def _solve_on_residual(self, data=None, warm_start=None):
        """The former default, kept verbatim and reachable only on request.

        Labelled ``accuracy_criterion="fixed_point"`` so that no reader and no
        downstream table can mistake it for an objective-accuracy claim.
        """
        d = self._resolve(data or {})
        if self._single_solver is not None:
            if warm_start is None:
                x, y, blocks, res = self._single_cold_solver(d)
            else:
                x, y, blocks, res = self._single_solver(d, warm_start[0], warm_start[1])
            status, cert = self._status(x, y, d, blocks, float(res))
            iters = int(blocks) * self.meta["block"]
            return Solution(x, y, iters, float(res), status=status, certificate=cert,
                            accuracy_criterion="fixed_point",
                            **self._facts(x, y, d, status, iters, float(res)))
        db = jax.tree_util.tree_map(lambda a: a[None], d)
        if warm_start is None:
            x, y, blocks, res = self._cold_solver(db)
        else:
            x, y, blocks, res = self._run(db, warm_start[0][None], warm_start[1][None])
        status, cert = self._status(x[0], y[0], d, blocks[0], float(res[0]))
        iters = int(blocks[0]) * self.meta["block"]
        return Solution(x[0], y[0], iters, float(res[0]), status=status, certificate=cert,
                        accuracy_criterion="fixed_point",
                        **self._facts(x[0], y[0], d, status, iters, float(res[0])))

    # ---- the default acceptance criterion ---------------------------------
    _NO_CONTRACT_DETAIL = (
        "no objective accuracy was certified at the returned point: this family "
        "declares no exact conjugate-domain contract, so no strict lower bound on "
        "the optimal value is available. `residual` is this algorithm's relative "
        "prox-graph quantity and is NOT an objective-accuracy claim.")

    def _unverified(self, s, detail=None):
        """Report a point that carries no certificate, without calling it solved."""
        if s.status in ("solved", "inaccurate"):
            return _dc_replace(s, status="unverified", accuracy_criterion="none",
                               accuracy_detail=detail or self._NO_CONTRACT_DETAIL)
        return _dc_replace(s, accuracy_criterion="none")

    @staticmethod
    def _gap_fields(ob):
        """(available, a finite gap exists, THE ACCEPTANCE GAP).

        The signed discrepancy uses ``max(|J(x)|, |lower|)``. Two zero values
        have zero discrepancy. The separately reported restored-primal bracket
        retains its own meaning; it is not the quantity accepted here.
        """
        available = bool(ob.get("objective_lower_bound_available", False))
        # NOT `relative_gap_certified`: that field belongs to the BRACKET
        # request and goes False whenever the extended-value objective is
        # +inf, which an absolute indicator threshold can cause at one data
        # scale and not another.  The approximate-KKT request needs its own
        # quantity to be valid, and the feasibility it requires is tested
        # separately, relatively, by the caller.
        # Resolved through the SAME fallback chain as the value below.  A
        # bare lookup of `objective_at_returned_point` refuses every provider
        # that does not build that key, such as the hand-built CGLS report.
        certified = available and math.isfinite(
            float(ob.get("objective_at_returned_point",
                         ob.get("objective_primal_upper_bound",
                                ob.get("objective_primal_value", float("nan"))))))
        lower = float(ob.get("objective_lower_bound", float("nan")))
        # THE OBJECTIVE AT THE RETURNED POINT, which is what Diamond and
        # Boyd's protocol tests ("duality gap, l2 norm of the primal residual,
        # and l2 norm of the dual residual", each at the returned point) and
        # what l.817 asks for.  The restored point's value is still computed
        # and still reported as the stricter bracket; it is simply not the
        # quantity that decides.  Where the point is feasible the two are the
        # same number, so only indicator-constrained families are affected.
        primal = float(ob.get("objective_at_returned_point",
                              ob.get("objective_primal_upper_bound",
                                     ob.get("objective_primal_value", float("nan")))))
        if not (certified and math.isfinite(lower) and math.isfinite(primal)):
            return available, False, float("nan")
        # Scale invariant: see `gap_budget`.  An exactly optimal point with a
        # zero objective and a zero bound has a zero gap and is accepted
        # rather than divided.
        scale = max(abs(primal), abs(lower))
        if scale <= 0.0:
            return available, True, 0.0
        return available, True, (primal - lower) / scale

    @staticmethod
    def _infeasibility(ob):
        """Relative primal infeasibility at the returned point, or 0.0.

        Reported by the evaluator for families with indicator blocks.  It is
        0.0 when there are none, which makes the feasibility half of the
        acceptance vacuous exactly when there is nothing to be feasible to.
        """
        value = ob.get("relative_primal_infeasibility", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _stalled(history, window=3, improvement=0.01):
        """Have gap, residual and feasibility stopped improving at checks?

        This is a relative-progress heuristic, not a stationarity certificate.
        The caller tests acceptance first. An unmet request remains unmet;
        neither its tolerance nor the acceptance predicate changes. In
        particular a residual plateau above 1e-7 no longer prevents an exit.
        """
        if len(history) < window + 1:
            return False
        tail = history[-(window + 1):]
        gaps = [h["relative_gap"] for h in tail]
        if any(g is None or not math.isfinite(g) for g in gaps):
            return False
        first, last = gaps[0], gaps[-1]
        if not first > 0.0:
            return False
        if (first - last) / first >= improvement:
            return False
        for key in ("legacy_residual", "relative_primal_infeasibility"):
            values = [h.get(key, 0.0) for h in tail]
            if any(not math.isfinite(v) or v < 0.0 for v in values):
                return False
            if values[0] > 0.0 and (values[0] - values[-1]) / values[0] >= improvement:
                return False
        return True

    def _solve_on_objective_gap(self, data=None, warm_start=None, policy=False):
        if not self._bound_capable or self._objective_bound is None:
            self._resolve(data or {})  # Validate the declared data without a solve.
            unsupported = self._unsupported_gap(warm_start)
            if policy:
                # Assessment unavailable: the policy reports the missing
                # capability and does no refinement work; it never
                # manufactures an assessment.
                from ..solver import refine_auto
                unsupported = refine_auto.record_unavailable(self, unsupported)
            return unsupported
        drive = self._single_drive if self._single_drive is not None else self._drive
        if drive is None:
            # No driving executable exists for this configuration, so the gap
            # cannot steer the iteration.  Score the ordinary solve on the gap
            # anyway: weaker than driving to it, but still the right test.
            s = self._solve_on_residual(data, warm_start)
            if s.status in ("solved", "inaccurate"):
                ob = self.objective_bound(s.x, s.y, data, rounds=self._bound_rounds,
                                          return_dual=True)
                awarded = self._award(s, ob, None)
                d = self._resolve(data or {})
                result = self._return_certifying_pair(awarded, ob, d)
                # No stall guard runs on this path, so the fixed trigger
                # cannot fire; the policy records that and does no work.
                return self._maybe_refine(result, d, self._resolve_dense(data or {}),
                                          stalled=False, x_iterate=s.x, y_iterate=s.y,
                                          policy=policy)
            return _dc_replace(s, accuracy_criterion="objective_gap")
        return self._drive_to_gap(data, warm_start, drive, policy=policy)

    def _unsupported_gap(self, warm_start=None):
        """Refuse a structurally unavailable request before optimizer work."""
        x = jnp.zeros(self.n, jnp.float32) if warm_start is None else warm_start[0]
        y = jnp.zeros(self.m, jnp.float32) if warm_start is None else warm_start[1]
        if x.shape != (self.n,) or y.shape != (self.m,):
            raise ValueError("warm_start has incompatible primal or dual shape")
        missing = self.meta.get("objective_bound_missing_contracts", [])
        detail = "objective_gap is unsupported: missing conjugate/value contract for "
        detail += ", ".join(missing) if missing else "the configured evaluator route"
        return Solution(x, y, 0, float("nan"), status="unverified",
                        termination="unsupported", accuracy_criterion="objective_gap",
                        accuracy_detail=detail, objective_bound_source="unavailable")

    def _drive_to_gap(self, data, warm_start, drive, policy=False):
        # patch D: phase separation (residency only; same executables, same
        # schedule, same bytes).  The continuation pair and the data live on
        # the host between phases: each segment uploads only the drive's
        # operands, each check evaluates on host inputs (the traced evaluator
        # uploads its own, the tiled route streams), and everything returns
        # to the device for the return path.
        separated = (bool(getattr(self._objective_bound, "phase_separated", False))
                     and self._single_drive is None and self._resolve_dense is self._resolve)
        d = self._resolve(data or {})
        # Patch L: ONE resolve when the dense resolver is the resolver itself (the dense
        # layout); a second call made a second identical copy of every offset and cost.
        # (Patch D makes the same change under `separated`; merged: L's condition, which
        # covers both.)
        dense = d if self._resolve_dense is self._resolve else self._resolve_dense(data or {})
        batched = self._single_drive is None
        # Patch L: the batched drive is reached through `_drive_flat` (no batch-of-one copies
        # of d, x or y outside the program; the starts are donated).  Only package-owned
        # buffers are donated: a cold start's zeros, the drive's own outputs, and a private
        # copy of a caller's warm start (under `separated`, the private uploads).
        flat = batched and self._drive_flat is not None
        x = (jnp.zeros(self.n, jnp.float32) if warm_start is None else warm_start[0])
        y = (jnp.zeros(self.m, jnp.float32) if warm_start is None else warm_start[1])
        if separated:
            x, y, d = _nodc_tiled.to_host((x, y, d))
            dense = d
            db = None
        else:
            if flat and warm_start is not None:
                x, y = jnp.array(x, copy=True), jnp.array(y, copy=True)
            db = jax.tree_util.tree_map(lambda a: a[None], d) if batched and not flat else None
        total_blocks, history, ob, res, stalled = 0, [], None, float("inf"), False
        strongest = None
        strongest_ext = None
        # A leading zero-block segment evaluates the criterion AT THE START
        # POINT: an instance that already meets it must not be charged an
        # iteration for being asked.
        schedule = [0] + _check_schedule(self._max_blocks, first=self._gap_check_first,
                                         growth=self._gap_check_growth,
                                         max_checks=self._gap_checks)
        for check, segment in enumerate(schedule):
            # A TRACED budget, so one executable serves every segment length.
            cap = jnp.asarray(segment, jnp.int32)
            if separated and flat:
                # patch D iteration phase, merged with patch L: only the drive's operands
                # go up (private, unbatched uploads), the flat drive adds the batch axis
                # inside and writes its outputs into the donated x, y uploads (this is what
                # patch D's donation add-on does, through operand_jit), and the outputs come
                # straight back to the host.
                dd, xin, yin = _nodc_tiled.to_device((d, x, y))
                # p2: the uploads complete before the drive is launched, so no transfer or
                # pending release overlaps the drive's own allocations (allocator hygiene in the
                # binding phase; the bytes and the computation are unchanged).
                jax.block_until_ready((dd, xin, yin))
                xo, yo, nb, rb = self._drive_flat(dd, xin, yin, cap)
                del dd, xin, yin
                nb, res = int(nb), float(rb)
                x, y = _nodc_tiled.to_host((xo, yo))
                del xo, yo
            elif separated:
                # patch D iteration phase as delivered (reached only without a flat drive).
                db, xin, yin = _nodc_tiled.to_device(
                    (jax.tree_util.tree_map(lambda a: a[None], d), x[None], y[None]))
                xb, yb, nb, rb = drive(db, xin, yin, cap)
                del db, xin, yin
                xb, yb = _nodc_tiled.to_host((xb, yb))
                x, y, nb, res = xb[0], yb[0], int(nb[0]), float(rb[0])
                del xb, yb
            elif flat:
                x, y, nb, rb = self._drive_flat(d, x, y, cap)
                nb, res = int(nb), float(rb)
            elif batched:
                xb, yb, nb, rb = drive(db, x[None], y[None], cap)
                x, y, nb, res = xb[0], yb[0], int(nb[0]), float(rb[0])
            else:
                x, y, nb, rb = drive(d, x, y, cap)
                nb, res = int(nb), float(rb)
            total_blocks += nb
            if math.isnan(res) or res < 0:
                break
            ob = self._objective_bound(x, y, dense, self._bound_rounds,
                                       return_dual=True)
            ob, retained = _retain_stronger_lower(ob, strongest)
            if bool(retained):
                ob['certifying_dual'] = strongest['certifying_dual']
            if separated:
                # patch D: a device-resident certifying dual waits on the host.
                ob['certifying_dual'] = _nodc_tiled.to_host(ob.get('certifying_dual'))
            # Patch L: a DEVICE certifying dual is held on the host until the loop ends,
            # so it occupies no device memory while the next segment runs (same bytes).
            # (Under `separated` patch D has already moved it; this then does nothing.)
            if isinstance(ob.get('certifying_dual'), jax.Array):
                ob['certifying_dual'] = _HostHeldDual(ob['certifying_dual'])
            strongest = ob
            available, certified, gap = self._gap_fields(ob)
            infeasibility = self._infeasibility(ob)
            history.append({"check": check,
                            "segment_blocks": int(segment),
                            "iterations": total_blocks * self.meta["block"],
                            "legacy_residual": res,
                            "bound_available": available,
                            "objective_at_returned_point": float(ob.get('objective_at_returned_point', float('nan'))),
                            "objective_lower_bound": float(ob.get('objective_lower_bound', float('nan'))),
                            "retained_previous_lower": bool(retained),
                            "relative_gap": gap if certified else None,
                            "relative_primal_infeasibility": infeasibility})
            # THE acceptance in `_award`, through the one shared predicate.
            # A loop that stopped on the gap alone would halt at a point the
            # award then refuses, for infeasibility or because the gap lies
            # within the float32 resolution of the tolerance, discarding the
            # iterations that would have reached a better answer.
            if _gap_accepts(available, certified, gap, infeasibility,
                            float(ob.get("objective_at_returned_point",
                                         ob.get("objective_primal_value", float("nan")))),
                            self._tol):
                break
            # ---- the dual-bound routes (compile(dual_routes=...)), only where
            # the evaluator's packet above does not meet the criterion.  Their
            # own retention chain (strongest_ext) never feeds back into the
            # evaluator's chain, so a check that packet accepts is decided and
            # returned exactly as without the routes, and the stall rule below
            # reads the unchanged history fields.
            if self._dual_routes:
                ext = self._extend_with_routes(ob, x, y, dense)
                ext, ext_retained = _retain_stronger_lower(ext, strongest_ext)
                if bool(ext_retained):
                    ext['certifying_dual'] = strongest_ext['certifying_dual']
                strongest_ext = ext
                e_available, e_certified, e_gap = self._gap_fields(ext)
                history[-1].update({"extended_lower": float(ext.get('objective_lower_bound', float('nan'))),
                                    "extended_source": ext.get('source'),
                                    "extended_relative_gap": e_gap if e_certified else None,
                                    "dual_routes": ext.get("dual_routes"),
                                    "dual_route_taken": ext.get("dual_route_taken"),
                                    "dual_route_seconds": ext.get("dual_route_seconds"),
                                    "dual_route_operator_applications": ext.get("dual_route_operator_applications")})
                if _gap_accepts(e_available, e_certified, e_gap, infeasibility,
                                float(ext.get("objective_at_returned_point",
                                              ext.get("objective_primal_value", float("nan")))),
                                self._tol):
                    ob = ext
                    break
            stalled = self._stalled(history)
            if stalled:
                break
        if (strongest_ext is not None and ob is not None and ob is not strongest_ext
                and strongest_ext.get("objective_lower_bound_available", False)
                and (not ob.get("objective_lower_bound_available", False)
                     or float(strongest_ext["objective_lower_bound"]) > float(ob["objective_lower_bound"]))
                and not _gap_accepts(*self._gap_fields(ob), self._infeasibility(ob),
                                     float(ob.get("objective_at_returned_point",
                                                  ob.get("objective_primal_value", float("nan")))),
                                     self._tol)):
            # Budget or stall exit with the request unmet: report the strongest
            # valid bound either chain holds (the award decides as always).
            ob = strongest_ext
        if separated:
            x, y, d = _nodc_tiled.to_device((x, y, d))
            dense = d
            if ob is not None:
                dual = ob.get('certifying_dual')
                if isinstance(dual, np.ndarray) and dual.dtype == np.float32:
                    ob['certifying_dual'] = jnp.asarray(dual)
        if ob is not None and isinstance(ob.get('certifying_dual'), _HostHeldDual):
            ob['certifying_dual'] = ob['certifying_dual'].restore()
        iters = total_blocks * self.meta["block"]
        status, cert = self._status(x, y, d, total_blocks, res)
        s = Solution(x, y, iters, res, status=status, certificate=cert,
                     accuracy_criterion="objective_gap", gap_history=history,
                     **self._facts(x, y, d, status, iters, res))
        if status in ("primal_infeasible", "unbounded", "numerical_error"):
            # These come from certificates, not from the accuracy criterion,
            # and the accuracy criterion does not overrule them.
            return s
        awarded = self._award(s, ob, history, stalled=stalled)
        awarded = _dc_replace(awarded, termination=(
            "criterion" if awarded.status == "solved" else
            "stalled" if stalled else "iteration_limit"))
        result = self._return_certifying_pair(awarded, ob, d)
        # The ordinary packet is complete here.  The refinement policy, when
        # on, reads it and either records why nothing was done or replaces it
        # by a refined pair that passes the same criterion (refine_auto).
        return self._maybe_refine(result, d, dense, stalled=stalled,
                                  x_iterate=x, y_iterate=y, policy=policy)

    def _return_certifying_pair(self, s, ob, data, *, checked=None,
                               contract_values=None):
        """Bind terminal reporting to the dual that certified this bound.

        Called only after stopping/award. The continuation dual, check history,
        certificate statuses and reason for ending the iteration are preserved.
        One additional original-problem residual evaluation (and the optional
        declared contract) is paid when a certifying dual is returned.
        """
        if s.status in ("primal_infeasible", "unbounded", "numerical_error"):
            return s
        if ob is None or not ob.get("objective_lower_bound_available", False):
            return s
        witness = ob.get("certifying_dual")
        if witness is None:
            raise RuntimeError("available objective bound omitted its certifying dual")
        if checked is None:
            check = self._returned_components or self._report
            if check is None:
                raise RuntimeError("certifying dual requires a returned-pair residual evaluator")
            checked = check(s.x, witness, data)
        residual = float(jnp.maximum(checked["r_primal"], checked["r_dual"]))
        if contract_values is None:
            contract_values = self._contract_values(s.x, witness, data)
        facts = {
            "strict_certificate": self._strict_passed(s.status, residual),
            "contract_certificate": self._contract_passed(contract_values),
            "contract": contract_values,
            # Preserve the controller's decision, not a new stopping decision
            # from the residual at the recovered witness.
            "termination": s.termination,
        }
        return _dc_replace(s, y=witness, residual=residual,
                           y_iterate=s.y_iterate, dual_source="certifying", **facts)

    def _award(self, s, ob, history, stalled=False):
        """Set ``status`` from the certified objective gap, and only from it.

        ``"solved"`` is awarded exactly when :func:`_gap_accepts` is true for
        the same fields; the branches below only explain a refusal.
        """
        if ob is None:
            return self._unverified(s)
        available, certified, gap = self._gap_fields(ob)
        common = {"accuracy_criterion": "objective_gap",
                  "objective_lower_bound": float(ob.get("objective_lower_bound",
                                                        float("nan"))),
                  "objective_primal_value": float(ob.get("objective_primal_value",
                                                         float("nan"))),
                  "objective_bound_source": ob.get("source", "unavailable"),
                  "gap_history": history}
        infeasibility = self._infeasibility(ob)
        assessment_primal = float(ob.get("objective_at_returned_point",
                                        common["objective_primal_value"]))
        accepts = _gap_accepts(available, certified, gap, infeasibility,
                               assessment_primal, self._tol)
        if accepts:
            if stalled:
                common["termination"] = "criterion"
            common["relative_primal_infeasibility"] = infeasibility
            if gap < 0.0:
                # NOT an inconsistency for this criterion.  The approximate-KKT
                # quantity is the finite cost at a point that may be only
                # approximately feasible, so it is not an upper bound and may
                # sit below a valid lower bound.  The conventional LP
                # discrepancy test is on the MAGNITUDE |J(x) - D(y)|.  The
                # signed reading belongs to the bracket, where the upper
                # really is an upper bound, and is kept there.
                return _dc_replace(
                    s, status="solved", objective_gap=gap,
                    accuracy_detail="the objective discrepancy is within allowance "
                                    "in magnitude; it is negative because the cost at "
                                    "an approximately feasible point is not an upper "
                                    "bound on the constrained optimum",
                    **common)
            return _dc_replace(s, status="solved", objective_gap=gap,
                               accuracy_detail="", **common)
        if stalled:
            common["termination"] = "stalled"
        if not available:
            return _dc_replace(
                s, status="unverified", objective_gap=None,
                accuracy_detail="no objective accuracy was certified at the returned "
                                "point: " + str(ob.get("unavailable_reason", ""))[:300],
                **common)
        if not certified:
            # The criterion WAS evaluated and the point fails it: x lies
            # outside a block domain, so its extended-valued objective is
            # infinite and no finite gap can exist.  That is a definite
            # negative, so it is "inaccurate" and not "unverified", which is
            # reserved for the case where nothing could be checked at all.
            return _dc_replace(
                s, status="inaccurate", objective_gap=None,
                accuracy_detail="a valid lower bound was certified, but the primal "
                                "value at the returned x is not finite: x is outside "
                                "a block domain, so no finite optimality gap follows. "
                                + str(ob.get("gap_unavailable_reason", ""))[:160],
                **common)
        # A gap at an INFEASIBLE point bounds nothing, whatever its sign.  The
        # acceptance has two halves and both must hold: x is feasible to `tol`
        # in the pure relative sense, and its objective is within `tol` of
        # optimal.  Feasibility is tested first, as `acceptance` does; a
        # release before 0.3.0rc3 tested a negative discrepancy before it.
        common["relative_primal_infeasibility"] = infeasibility
        if not (infeasibility <= self._tol):
            return _dc_replace(
                s, status="inaccurate", objective_gap=gap,
                accuracy_detail=(
                    "the returned point has relative primal infeasibility %.6e, "
                    "above the requested tol %.6e, so the certified gap %.6e "
                    "bounds nothing: a duality gap at an infeasible point is not "
                    "a bound on the optimal value"
                    % (infeasibility, self._tol, gap)),
                **common)
        if gap < 0.0:
            # A large negative difference fails because its magnitude exceeds
            # the budget; a small one was accepted above.
            return _dc_replace(
                s, status="inaccurate", objective_gap=gap,
                accuracy_detail="the objective discrepancy %.6e exceeds the requested "
                                "tol %.6e in magnitude" % (-gap, self._tol),
                **common)
        floor = _gap_resolution(assessment_primal)
        if abs(gap - self._tol) < floor:
            # The margin that decides this is finer than the arithmetic that
            # produced it, so NEITHER the accept nor the miss is established.
            # Refuse the comparison instead of picking a side of it.
            return _dc_replace(
                s, status="unverified", objective_gap=gap,
                accuracy_detail=(
                    "the computed gap %.3e and the requested tol %.3e differ by "
                    "less than %.3e, the finest gap float32 can resolve at an "
                    "objective value of %.6g, so neither meeting nor missing the "
                    "tolerance is established by this arithmetic."
                    % (gap, self._tol, floor, common["objective_primal_value"])),
                **common)
        # The normalization named here must match `gap_budget`, which is
        # scale invariant.  The old text still said max(1, |primal value|),
        # describing a rule the code no longer applies.
        detail = ("the objective discrepancy %.6e, normalized by "
                  "max(|objective value|, |lower bound|), exceeds the "
                  "requested tol %.6e" % (gap, self._tol))
        if stalled:
            detail += ("; stopped after insufficient relative improvement in "
                       "the gap, residual and feasibility over three check intervals. "
                       "This heuristic does not prove that further progress is impossible")
        return _dc_replace(s, status="inaccurate", objective_gap=gap,
                           accuracy_detail=detail, **common)

    def solve_many(self, data_batch, warm_start=None, *, accuracy=None):
        """Return batched ``(x, y, blocks, residuals)`` device arrays.

        ``blocks`` counts iteration blocks; multiply by ``meta['block']``
        for iteration counts. Use ``solve_many_results`` for host-side
        per-instance statuses and certificate directions. The family's selected
        accuracy criterion controls continuation here too. Objective-gap mode
        returns each lane's certifying pair and performs host-side checks;
        ``accuracy="residual"`` explicitly selects the legacy device-only loop.
        Call objective-gap mode outside an enclosing ``jax.jit``; its numerical
        kernels are already compiled internally.
        A returned tuple alone does not assert that every lane was successful.
        The refinement policy is never applied here: these arrays carry no
        record to charge a refinement to; use ``solve_many_results``.
        """
        criterion = self._accuracy if accuracy is None else _check_accuracy(accuracy)
        if criterion == "objective_gap":
            if any(isinstance(v, jax.core.Tracer)
                   for v in jax.tree_util.tree_leaves((data_batch, warm_start))):
                raise ValueError("objective-gap solve_many uses host-side assessment; "
                                 "call it outside jax.jit. The explicitly selected "
                                 "accuracy='residual' path is device-only and has "
                                 "a different stopping criterion.")
            rows = self._solve_many_on_objective_gap(data_batch, warm_start)
            return (jnp.stack([s.x for s in rows]), jnp.stack([s.y for s in rows]),
                    jnp.asarray([s.iters // self.meta["block"] for s in rows], jnp.int32),
                    jnp.asarray([s.residual for s in rows], jnp.float32))
        d = jax.vmap(self._resolve)(data_batch)
        if warm_start is None:
            return self._cold_solver(d)
        return self._run(d, warm_start[0], warm_start[1])  # (x, y, blocks, res)

    def solve_many_results(self, data_batch, warm_start=None, *, accuracy=None, refine=None):
        """Solve one batch and return a list of :class:`Solution` objects.

        Held to the SAME acceptance criterion as :meth:`solve_instance`, and
        through the same ``_award`` function, so a row cannot be accepted here
        on a test the scalar path would refuse.  See :meth:`solve_instance`
        for what the criteria mean.  ``refine`` is as in :meth:`solve_instance`:
        under ``"auto"`` each lane that ended on the stall guard with its
        request unmet gets its own bounded stage, after the batched iteration
        has finished; a lane's stage reads and replaces only that lane's row.
        """
        criterion = self._accuracy if accuracy is None else _check_accuracy(accuracy)
        if criterion == "residual":
            return self._solve_many_on_residual(data_batch, warm_start)
        return self._solve_many_on_objective_gap(data_batch, warm_start,
                                                 policy=self._refine_policy_for(refine))

    def _solve_many_on_residual(self, data_batch, warm_start=None):
        """The former default on a batch: status from the prox-graph residual.

        The numerical solve and status checks are batched. Only scalar result
        conversion and the final tolerance comparison run on the host.
        Certificate directions receive the same scalar checks as solve_instance.
        """
        data = jax.vmap(self._resolve)(data_batch)
        if warm_start is None:
            x, y, blocks, res = self._cold_solver(data)
        else:
            x, y, blocks, res = self._run(data, warm_start[0], warm_start[1])
        return self._many_solutions(data, x, y, blocks, res,
                                    criterion="fixed_point")

    def _many_solutions(self, data, x, y, blocks, res, *, criterion, bound=None,
                        certifying_dual=None):
        """Build per-row Solutions from batched arrays.

        ``bound`` is the batched objective-bound report, already frozen at
        each row's own accepted point; when it is None no accuracy claim is
        attached beyond the legacy labelling.
        """
        codes, host_blocks, host_res = jax.device_get((
            self._batch_status_checks(x, y, data, res), blocks, res,
        ))
        points = _unstack_batch_points(x, y)
        contracts = None
        if self._contract_many is not None:
            contracts = jax.device_get(self._contract_many(x, y, data))
        host_bound = None if bound is None else jax.device_get(bound)
        returned_checks = returned_contracts = None
        if (criterion == "objective_gap" and host_bound is not None
                and certifying_dual is not None
                and self._returned_components_many is not None):
            use = np.asarray(host_bound["objective_lower_bound_available"]) & (codes == 0)
            if np.any(use):
                returned_y = jnp.where(jnp.asarray(use)[:, None], certifying_dual, y)
                returned_checks = jax.device_get(
                    self._returned_components_many(x, returned_y, data))
                if self._contract_many is not None:
                    returned_contracts = jax.device_get(
                        self._contract_many(x, returned_y, data))
        results = []
        for i, (xx, yy) in enumerate(points):
            value = float(host_res[i])
            cert = None
            if codes[i] == 1:
                status = "numerical_error"
            elif codes[i] in (2, 3):
                d = jax.tree_util.tree_map(lambda a: a[i], data)
                status, cert = self._status(xx, yy, d, host_blocks[i], value)
            else:
                status = "solved" if value <= self._tol else "inaccurate"
            iters = int(host_blocks[i]) * self.meta["block"]
            values = (None if contracts is None else
                      {k: float(v[i]) for k, v in contracts.items()})
            passed = self._contract_passed(values)
            s = Solution(
                xx, yy, iters, value, status=status, certificate=cert,
                accuracy_criterion=criterion,
                strict_certificate=self._strict_passed(status, value),
                contract_certificate=passed,
                termination=self._termination(status, iters, value, passed),
                contract=values,
            )
            if criterion != "objective_gap":
                results.append(s)
                continue
            if status in ("primal_infeasible", "unbounded", "numerical_error"):
                results.append(s)
                continue
            if host_bound is None:
                results.append(self._unverified(s))
                continue
            ob = _bound_row(host_bound, i)
            if certifying_dual is not None:
                ob["certifying_dual"] = certifying_dual[i]
            awarded = self._award(s, ob, None)
            d = jax.tree_util.tree_map(lambda a: a[i], data)
            checked = (None if returned_checks is None else
                       {k: v[i] for k, v in returned_checks.items()})
            values = (None if returned_contracts is None else
                      {k: float(v[i]) for k, v in returned_contracts.items()})
            results.append(self._return_certifying_pair(
                awarded, ob, d, checked=checked, contract_values=values))
        return results

    def _solve_many_on_objective_gap(self, data_batch, warm_start=None, policy=False):
        if not self._bound_capable or self._objective_bound_many is None:
            data = jax.vmap(self._resolve)(data_batch)
            count = jax.tree_util.tree_leaves(data)[0].shape[0]
            rows = [self._unsupported_gap(None if warm_start is None else
                                         (warm_start[0][i], warm_start[1][i]))
                    for i in range(count)]
            if policy:
                from ..solver import refine_auto
                rows = [refine_auto.record_unavailable(self, r) for r in rows]
            return rows
        if (not self._bound_capable or self._objective_bound_many is None
                or self._drive is None):
            # Preserve the existing fallback: no final bound is evaluated,
            # so there is no certifying dual to return and no accuracy award.
            rows = self._solve_many_on_residual(data_batch, warm_start)
            return [self._unverified(r) for r in rows]
        data = jax.vmap(self._resolve)(data_batch)
        dense = (data if self._resolve_dense is self._resolve
                 else jax.vmap(self._resolve_dense)(data_batch))
        count = jax.tree_util.tree_leaves(data)[0].shape[0]
        x = (jnp.zeros((count, self.n), jnp.float32) if warm_start is None
             else warm_start[0])
        y = (jnp.zeros((count, self.m), jnp.float32) if warm_start is None
             else warm_start[1])
        schedule = [0] + _check_schedule(self._max_blocks, first=self._gap_check_first,
                                         growth=self._gap_check_growth,
                                         max_checks=self._gap_checks)
        stopped = np.zeros(count, dtype=bool)
        stalled_rows = np.zeros(count, dtype=bool)
        histories = [[] for _ in range(count)]
        done_x, done_y = x, y
        done_blocks = jnp.zeros(count, jnp.int32)
        done_res = jnp.full(count, jnp.inf, jnp.float32)
        running = jnp.zeros(count, jnp.int32)
        frozen = None
        frozen_dual = None
        strongest = strongest_dual = None
        for check, segment in enumerate(schedule):
            cap = jnp.asarray(segment, jnp.int32)
            x, y, nb, res = self._drive(data, x, y, cap)
            running = running + jnp.asarray(nb, jnp.int32)
            report, certifying_dual = self._objective_bound_many(
                x, y, dense, self._bound_rounds, return_dual=True)
            report, retained = _retain_stronger_lower(report, strongest, batched=True)
            if strongest_dual is not None:
                certifying_dual = jnp.where(retained[:, None], strongest_dual, certifying_dual)
            strongest, strongest_dual = report, certifying_dual
            lower = report["objective_lower_bound"]
            # The same quantity the scalar path accepts on: the objective at
            # the returned point.  Keeping these two expressions identical is
            # what makes one request mean one thing on both interfaces.
            primal = report.get("objective_at_returned_point",
                                report.get("objective_primal_upper_bound",
                                           report["objective_primal_value"]))
            # The same scale-invariant normalization the scalar path uses.
            _scale = jnp.maximum(jnp.abs(primal), jnp.abs(lower))
            norm_gap = jnp.where(_scale > 0.0,
                                 (primal - lower) / jnp.where(_scale > 0.0, _scale, 1.0),
                                 0.0)
            infeasibility = (report["relative_primal_infeasibility"]
                             if "relative_primal_infeasibility" in report
                             else jnp.zeros_like(norm_gap))
            # THE acceptance of `_award`, per lane, through the one shared
            # predicate: the same `certified` as `_gap_fields` (a finite cost
            # at the returned point, NOT `relative_gap_certified`, which
            # belongs to the bracket), the same feasibility, the same float32
            # resolution floor at the same objective value.  A lane stops
            # exactly when its row would be awarded "solved" at the end.
            certified = (report["objective_lower_bound_available"]
                         & jnp.isfinite(primal) & jnp.isfinite(lower))
            ok = _gap_accepts(report["objective_lower_bound_available"], certified,
                              norm_gap, infeasibility,
                              primal, self._tol)
            # Preserve each terminal lane's own point, report and work count.
            # The vectorized driver may still evaluate inactive lanes while
            # other lanes progress, as it did for accepted lanes before 0.3.1;
            # those extra values cannot replace a terminal result.
            keep = jnp.asarray(stopped)
            done_x = jnp.where(keep[:, None], done_x, x)
            done_y = jnp.where(keep[:, None], done_y, y)
            done_blocks = jnp.where(keep, done_blocks, running)
            done_res = jnp.where(keep, done_res, res)
            if frozen is None:
                frozen = dict(report)
                frozen_dual = certifying_dual
            else:
                frozen = {k: jnp.where(keep, frozen[k], report[k])
                          for k in frozen}
                frozen_dual = jnp.where(keep[:, None], frozen_dual, certifying_dual)
            host_gap, host_res, host_cert, host_infeas, host_run, host_ok, host_available, host_lower, host_primal, host_retained = jax.device_get(
                (norm_gap, res, certified, infeasibility, running, ok,
                 report["objective_lower_bound_available"], lower, primal, retained))
            for i in range(count):
                if stopped[i]:
                    continue
                histories[i].append({
                    "check": check, "segment_blocks": int(segment),
                    "iterations": int(host_run[i]) * self.meta["block"],
                    "legacy_residual": float(host_res[i]),
                    "bound_available": bool(host_available[i]),
                    "objective_at_returned_point": float(host_primal[i]),
                    "objective_lower_bound": float(host_lower[i]),
                    "retained_previous_lower": bool(host_retained[i]),
                    "relative_gap": float(host_gap[i]) if host_cert[i] else None,
                    "relative_primal_infeasibility": float(host_infeas[i]),
                })
                if host_ok[i]:
                    stopped[i] = True
                elif math.isnan(float(host_res[i])) or host_res[i] < 0:
                    stopped[i] = True
                elif self._stalled(histories[i]):
                    stopped[i] = stalled_rows[i] = True
            if stopped.all():
                break
        results = self._many_solutions(data, done_x, done_y, done_blocks, done_res,
                                       criterion="objective_gap", bound=frozen,
                                       certifying_dual=frozen_dual)
        for i, result in enumerate(results):
            if result.status in ("primal_infeasible", "unbounded", "numerical_error"):
                continue
            if stalled_rows[i]:
                result = self._award(result, _bound_row(jax.device_get(frozen), i),
                                     histories[i], stalled=True)
            results[i] = _dc_replace(result, gap_history=histories[i], termination=(
                "criterion" if result.status == "solved" else
                "stalled" if stalled_rows[i] else "iteration_limit"))
        if policy:
            # One lane at a time, after the batched iteration has finished:
            # a lane's stage reads only its own point, data and row, and can
            # replace only its own row.  A failing or slow lane therefore
            # cannot touch another lane's packet; what it does cost is wall
            # time on the whole call's return, which the record states.
            for i, result in enumerate(results):
                if result.status in ("primal_infeasible", "unbounded", "numerical_error"):
                    continue
                d_i = jax.tree_util.tree_map(lambda a: a[i], data)
                dense_i = jax.tree_util.tree_map(lambda a: a[i], dense)
                results[i] = self._maybe_refine(
                    result, d_i, dense_i, stalled=bool(stalled_rows[i]),
                    x_iterate=done_x[i], y_iterate=done_y[i], policy=policy,
                    batch_lane=i)
        return results

    def certificate(self, x, y, data=None):
        """Report the matrix-free certificate for a single (x, y).

        ``r_primal`` / ``r_dual`` are the fixed-point/KKT residuals.

        ``gap`` is a LEGACY FENCHEL SURROGATE and is NOT a duality gap: it is
        built with a finite-part replacement that turns an out-of-domain
        ``+inf`` conjugate into ``0``, so its dual value can sit ABOVE the primal
        value at the same point.  It is kept, unchanged, for existing scripts.
        ``gap_legacy_fenchel_surrogate`` carries the same number under a name
        that cannot be misread, and ``gap_surrogate_above_primal`` says outright
        when the surrogate is above the primal.

        ``objective_lower_bound`` (with ``objective_lower_bound_available`` and
        ``objective_bound_status``, a code in
        :data:`OBJECTIVE_BOUND_STATUS`) is the strict bound: no finite-part
        replacement, no conjugate tolerance, and only for a dual that is exactly
        in every conjugate domain AND satisfies the coupling condition.  When it
        cannot be established the value is ``NaN`` and the status says why; a
        surrogate is never substituted, a negative gap is never clamped to zero,
        and the dual is never clipped to the primal.

        Epistemic limit: the gates are tolerance-free on the represented float32
        data, which is NOT a rigorous real-arithmetic enclosure."""
        return self._report(x, y, self._resolve(data or {}))

    def contract_report(self, x, y, data=None, contract=None):
        """Score one (x, y) under DECLARED budgets, without re-solving.

        ``contract`` is a :class:`ocs.solver.contract.ScaledContract` or a dict
        of declared budgets; omit it to use the family's own compiled contract.
        Returns the contract ratios (pass <= 1) as floats.  This evaluates a
        point that already exists: it changes no status, no residual and no
        iterate, and it is the intended way to see what a point would score
        under other budgets before declaring one.
        """
        if contract is None:
            if self._contract is None:
                raise ValueError("this family compiled no certificate contract; "
                                 "pass one to score a point under declared budgets")
            return self._contract_values(x, y, self._resolve(data or {}))
        if self._contract_factory is None:
            raise ValueError("a scaled contract requires residual_norm='relative' and PDHG")
        declared = ScaledContract.coerce(contract)
        key = (declared.primal_budget if isinstance(declared.primal_budget, (int, float))
               else tuple(declared.primal_budget),
               declared.dual_budget if isinstance(declared.dual_budget, (int, float))
               else tuple(declared.dual_budget),
               declared.variable_budget, declared.stationarity_budget)
        evaluator = self._contract_cache.get(key)
        if evaluator is None:
            evaluator = operand_jit(self._contract_factory(declared))
            self._contract_cache[key] = evaluator
        return {k: float(v) for k, v in evaluator(x, y, self._resolve(data or {})).items()}

    def report_text(self, x, y, data=None, bound=None):
        """The ordinary printed report, as text.

        Shows the residuals, the objective lower bound and its availability (or
        the reason it is unavailable), and the legacy surrogate explicitly
        labelled as a surrogate rather than as a gap.  Pass ``bound`` (the dict
        from :meth:`objective_bound`) to report a searched bound instead of the
        one the supplied dual establishes by itself; nothing here applies an
        operator beyond what :meth:`certificate` already does."""
        cert = self.certificate(x, y, data)
        get = lambda k, default=float("nan"): float(cert.get(k, default))
        lines = ["OCS certificate",
                 f"  r_primal                {get('r_primal'):.6e}",
                 f"  r_dual                  {get('r_dual'):.6e}"]
        if bound is None:
            available = bool(get("objective_lower_bound_available", 0.0))
            status = int(get("objective_bound_status", OBJECTIVE_BOUND_NO_CONTRACT))
            value = get("objective_lower_bound")
            primal = get("objective_primal_value")
            defect = get("dual_coupling_defect")
        else:
            available = bool(bound["objective_lower_bound_available"])
            status = int(bound.get("objective_bound_status_code",
                                   OBJECTIVE_BOUND_NO_CONTRACT))
            value = float(bound["objective_lower_bound"])
            primal = float(bound.get("objective_primal_value", float("nan")))
            defect = float(bound.get("certifying_dual_coupling_defect", float("nan")))
        if available:
            lines.append(f"  objective lower bound   {value:.9g}   (AVAILABLE)")
            if math.isfinite(primal):
                lines.append(f"  primal value at x       {primal:.9g}")
                scale = max(abs(primal), 1e-30)
                lines.append(f"  relative gap            {(primal - value) / scale:.6e}")
            else:
                lines.append("  primal value at x       not finite: x is outside a "
                             "block domain, so no finite gap follows from this bound")
            lines.append(f"  coupling defect         {defect:.3e}   (diagnostic only)")
        else:
            lines.append("  objective lower bound   UNAVAILABLE")
            lines.append(f"    reason: {OBJECTIVE_BOUND_STATUS.get(status, 'unknown')}")
            if status == OBJECTIVE_BOUND_COUPLING_VIOLATED and math.isfinite(defect):
                lines.append(f"    measured coupling defect ||z - P_dom(z)||_inf = {defect:.3e}"
                             "  (diagnostic; it does NOT license a bound)")
        surrogate = get("gap")
        if math.isnan(surrogate):
            lines.append("  legacy Fenchel surrogate  unavailable")
        else:
            flag = ("  -- ITS DUAL VALUE IS ABOVE THE PRIMAL VALUE, so it is not an "
                    "optimality gap" if get("gap_surrogate_above_primal", 0.0) else "")
            lines.append(f"  legacy Fenchel surrogate  gap={surrogate:.6e}"
                         "   NOT A BOUND" + flag)
        lines.append("  " + OBJECTIVE_BOUND_EPISTEMIC_LIMIT)
        return "\n".join(lines)

    def objective_bound(self, x, y, data=None, rounds=32, return_dual=False,
                        *, recover=True):
        """Strict lower bound on the optimal objective value, from a supplied
        primal-dual point.  This is what the DEFAULT acceptance criterion
        calls: `solve_instance` evaluates it along the run and accepts on it
        (see :meth:`solve_instance`).  It remains callable directly to score
        any point, including one another solver returned.

        Returns a dict with ``objective_lower_bound``,
        ``objective_lower_bound_available``, ``objective_primal_value``,
        ``relative_gap``, ``source`` (which candidate direction certified:
        ``supplied_witness``, ``violation_corrected``, ``residual_scaled``
        or ``feasible_anchor``), and
        ``operator_applications``.  When no candidate is exactly dual feasible,
        or a block does not declare the exact conjugate-domain contract, the
        bound is reported UNAVAILABLE with a reason; a surrogate is never
        substituted.

        With ``return_dual=True``, ``certifying_dual`` contains the exact
        selected candidate, or an all-NaN array when no candidate is available.
        Retaining it adds no operator action to the bound evaluator.

        ``recover=False`` evaluates only the supplied witness and raw point:
        no ray search, dual correction, anchor or primal restoration runs.
        This permits comparison of identical packets across evaluators.

        The default direction search corrects the conjugate-domain violation of
        the supplied dual (``rounds`` corrections, two operator applications
        each) and keeps the supplied dual, scaled along its own ray, as the
        fallback.  ``x`` is used only for the primal value it is compared
        against.  No solve is performed and the supplied point is not changed.

        This is a float32 evaluation with exact validity gates: not an interval
        enclosure and not a proof."""
        if self._objective_bound is None:
            return {"objective_lower_bound": float("nan"),
                    "objective_lower_bound_available": False,
                    "objective_bound_capable": False,
                    "unavailable_reason": "this family does not support an objective bound",
                    "operator_applications": 0, "rounds": int(rounds),
                    "source": "unavailable"}
        return self._objective_bound(x, y, self._resolve_dense(data or {}), rounds,
                                     return_dual=return_dual, recover=recover)

    def validate(self, x, y, data=None, rounds=32):
        """Evaluate the SUPPLIED pair, with no recovery of any kind.

        Read-only: no corrected dual, no ray search, no anchor construction,
        no primal restoration, no solver state touched.  Those routes are not
        staged into the traced program, and the returned
        ``n_*`` counters say so rather than asserting it.

        It is NOT zero numerical work.  The operator is still applied for the
        objective, the adjoint of the supplied dual and the feasibility
        slack, and those applications are counted.

        This shares `_strict_bound` and the acceptance code with the ordinary
        recovery path on purpose: a separate implementation that happened to
        agree would say nothing about the path that ships.
        """
        if self._objective_bound is None:
            return {"objective_lower_bound": float("nan"),
                    "objective_lower_bound_available": False,
                    "objective_bound_capable": False,
                    "unavailable_reason": "this family does not support an "
                                          "objective bound",
                    "recovery_enabled": False,
                    "operator_applications": 0, "rounds": int(rounds),
                    "source": "unavailable"}
        return self._objective_bound(x, y, self._resolve_dense(data or {}),
                                     rounds, return_dual=False, recover=False)

    def certificate_many(self, x, y, data_batch=None):
        """Report numerical diagnostics for batched supplied primal-dual points.

        Returns the same fields as :meth:`certificate`, each as a device array
        with leading batch axis. The evaluator and residual scales are shared
        with scalar reporting; no solve, dual repair or status assignment is
        performed. Use ``jax.device_get`` once to retrieve the whole report.
        Boolean flags remain boolean device arrays (the scalar wrapper returns
        them as floats). Batching may change floating-point reduction order.
        The ``gap`` fields retain their scalar availability and their LEGACY
        finite-part-surrogate meaning: they are not bounds.  The
        ``objective_lower_bound`` fields retain their strict meaning and their
        per-row availability and status.

        ``x`` and ``y`` have shape ``(batch, n)`` and ``(batch, m)``. Every data
        leaf has the same leading batch axis; omit data for a constant family.
        Different batch sizes compile distinct executables.
        """
        if self._report_many is None:
            raise NotImplementedError("batched certificate evaluator not available")
        x, y, data = _resolve_certificate_batch(
            self._resolve, x, y, data_batch, self.n, self.m)
        return self._report_many(x, y, data)

    def __repr__(self):
        declared = ""
        if "block_metrics" in self.meta:
            terms = self.meta["block_metrics"]["terms"]
            declared = "block_metrics=" + str({t["term"]: [g["kind"] for g in t["sharing_groups"]]
                                               for t in terms}) + ", "
        return (f"CompiledFamily(n={self.n}, m={self.m}, "
                f"dual_blocks={self.meta['n_dual']}, "
                f"prox_groups={self.meta['n_groups']}, "
                f"equilibration={self.meta['equilibrate']}, "
                + declared +
                f"||K_scaled||~{self.meta['scaled_norm']:.3g})")


class CGCompiledFamily:
    """Compiled pure-least-squares family solved by matrix-free CGLS.

    Explicit dual-free exception to PDHG's returned-witness contract:
    ``Solution.y`` and ``y_iterate`` stay None, with ``dual_source="none"``.
    For ``rho > 0``, objective-gap acceptance uses the certified primal
    suboptimality bound ``f(x) - f* <= ||grad f(x)||^2 / (2 rho)`` and records
    ``objective_bound_source="strong_convexity"``. This is not the dual-witness
    criterion of the primal-dual paths. The bound is evaluated in float32,
    not with rigorous interval arithmetic. With ``rho = 0`` the bound remains
    unavailable and objective-gap results remain ``"unverified"``.
    """
    def __init__(self, solver, resolve_b, matvec, rmatvec, n, m, meta,
                 accuracy="objective_gap"):
        self._accuracy = _check_accuracy(accuracy)
        # The raw parameter mapping of the solve in progress, so that the
        # strong-convexity bound can re-form the right-hand side.
        self._last_raw_data = {}
        self._tol = float(meta["tol"])
        self._solver = solver
        self._resolve = resolve_b
        self._matvec, self._rmatvec = matvec, rmatvec
        self.n, self.m, self.meta = n, m, meta
        self._report_many = operand_jit(jax.vmap(self._certificate_arrays))

    def _solution(self, x, b, iterations, residual):
        # CGLS has no dual witness. Its separate strong-convexity bound can
        # certify primal suboptimality when rho > 0; rho = 0 stays unverified.
        # These labels do not steer or change the CG iteration.
        value = float(residual)
        finite = not math.isnan(value) and value >= 0 and bool(jnp.all(jnp.isfinite(x))) \
            and bool(jnp.all(jnp.isfinite(b)))
        status = ("numerical_error" if not finite else
                  "solved" if value <= self.meta["tol"] else "inaccurate")
        # The same three separate facts.  CGLS takes no declared-budget
        # contract, so (ii) is None: no such test was run.
        termination = ("numerical_failure" if status == "numerical_error" else
                       "criterion" if (int(iterations) < int(self.meta["cg_iters"])
                                       or status == "solved") else "iteration_limit")
        strict = (status != "numerical_error" and value <= self.meta["tol"])
        base = Solution(x, None, int(iterations), value, status=status,
                        accuracy_criterion="fixed_point",
                        strict_certificate=strict,
                        contract_certificate=None, termination=termination,
                        dual_source="none")
        if self._accuracy == "residual":
            return base
        if status == "numerical_error":
            return _dc_replace(base, accuracy_criterion="objective_gap")
        # Strong convexity supplies the bound when rho > 0; otherwise nothing
        # can be certified and the point is reported unverified.
        ob = self.objective_bound(x, None, self._last_raw_data)
        return _dc_replace(self._award(base, ob, None),
                           objective_bound_source=ob["source"])

    def solve_instance(self, data=None, warm_start=None, *, accuracy=None):
        """Score the returned CG point using its primal strong-convexity bound.

        See :meth:`objective_bound`: with ``rho > 0`` strong convexity gives a
        strict lower bound and ``status`` is awarded by certified primal
        suboptimality, without the primal-dual paths' dual-witness requirement.
        The result records ``objective_bound_source="strong_convexity"``.
        With ``rho = 0`` nothing can be certified and ``status`` is
        ``"unverified"``.  The CG iteration is NOT steered by the gap: the
        point it returns is scored on it.
        """
        previous, self._accuracy = self._accuracy, (
            self._accuracy if accuracy is None else _check_accuracy(accuracy))
        try:
            self._last_raw_data = data or {}
            b = self._resolve(data or {})
            x0 = (warm_start[0][None] if warm_start
                  else jnp.zeros((1, self.n), jnp.float32))
            x, k, rel = self._solver(b[None], x0)
            return self._solution(x[0], b, k[0], rel[0])
        finally:
            self._accuracy = previous

    def solve_many(self, data_batch, warm_start=None):
        """Return ``(x, iterations, residuals)`` device arrays (CG iterations).

        Use ``solve_many_results`` for the same Solution interface as PDHG.
        """
        b = jax.vmap(self._resolve)(data_batch)
        N = b.shape[0]
        x0 = warm_start[0] if warm_start else jnp.zeros((N, self.n), jnp.float32)
        return self._solver(b, x0)                 # (x, iters, rel_resid)

    def solve_many_results(self, data_batch, warm_start=None, *, accuracy=None):
        """Run batched CG and return per-instance Solution objects.

        Held to the same criterion as :meth:`solve_instance`.
        """
        previous, self._accuracy = self._accuracy, (
            self._accuracy if accuracy is None else _check_accuracy(accuracy))
        try:
            x, iterations, residuals = self.solve_many(data_batch, warm_start)
            b = jax.vmap(self._resolve)(data_batch)
            rows = []
            for i in range(x.shape[0]):
                self._last_raw_row = {k: v[i] for k, v in (data_batch or {}).items()}
                self._last_raw_data = self._last_raw_row
                rows.append(self._solution(x[i], b[i], iterations[i], residuals[i]))
            return rows
        finally:
            self._accuracy = previous

    def objective_bound(self, x, y=None, data=None, rounds=32, return_dual=False):
        """Strict lower bound on the optimal value, from STRONG CONVEXITY.

        CGLS returns no dual point, so no Fenchel bound is available here.  A
        bound is available anyway when ``rho > 0``, because then

            f(x) = (1/2)||A x - b||^2 + (rho/2)||x||^2

        is rho-strongly convex and, for every x,

            f^*  >=  f(x) - ||grad f(x)||^2 / (2 rho),
            grad f(x) = A^T(A x - b) + rho x.

        This costs one matvec and one rmatvec and uses no entry of A.  ``y``
        is accepted and ignored: this route needs no dual point.

        With ``rho = 0`` the objective is not strongly convex, the argument
        gives nothing, and the bound is reported UNAVAILABLE rather than
        replaced by a surrogate.  A float32 evaluation, not a proof.
        """
        rho = float(self.meta.get("rho", 0.0))
        if not rho > 0.0:
            return {"objective_lower_bound": float("nan"),
                    "objective_lower_bound_available": False,
                    "objective_bound_capable": False,
                    "objective_bound_status_code": OBJECTIVE_BOUND_NO_CONTRACT,
                    "unavailable_reason":
                        "the CGLS path returns no dual point, and with rho = 0 the "
                        "objective is not strongly convex, so no lower bound follows "
                        "from the gradient either; one would need the smallest "
                        "singular value of A, which products do not give",
                    "epistemic_limit": OBJECTIVE_BOUND_EPISTEMIC_LIMIT,
                    "operator_applications": 0, "rounds": int(rounds),
                    "source": "unavailable"}
        b = self._resolve(data or {})
        r = self._matvec(x) - b
        grad = self._rmatvec(r) + rho * x
        primal = float(0.5 * jnp.vdot(r, r).real + 0.5 * rho * jnp.vdot(x, x).real)
        gnorm2 = float(jnp.vdot(grad, grad).real)
        lower = primal - gnorm2 / (2.0 * rho)
        finite = math.isfinite(primal) and math.isfinite(lower)
        return {"objective_lower_bound": lower if finite else float("nan"),
                "objective_lower_bound_available": bool(finite),
                "objective_bound_capable": True,
                "objective_primal_value": primal,
                "objective_primal_value_finite": bool(math.isfinite(primal)),
                "relative_gap": ((primal - lower) / abs(primal)
                                 if finite and primal != 0.0 else float("nan")),
                "relative_gap_certified": bool(finite),
                "objective_bound_status_code": (OBJECTIVE_BOUND_AVAILABLE if finite
                                                else OBJECTIVE_BOUND_NO_CONTRACT),
                "epistemic_limit": OBJECTIVE_BOUND_EPISTEMIC_LIMIT,
                "operator_applications": 2, "rounds": int(rounds),
                "source": "strong_convexity" if finite else "unavailable"}

    # One award function for both families: see CompiledFamily._award.  Taken
    # from __dict__ where a staticmethod wrapper must survive the assignment.
    _NO_CONTRACT_DETAIL = CompiledFamily._NO_CONTRACT_DETAIL
    _gap_fields = CompiledFamily.__dict__["_gap_fields"]
    _infeasibility = CompiledFamily.__dict__["_infeasibility"]
    _unverified = CompiledFamily._unverified
    _award = CompiledFamily._award
    report_text = CompiledFamily.report_text

    def _certificate_arrays(self, x, b):
        rho = self.meta["rho"]
        gradient = self._rmatvec(self._matvec(x) - b)
        s = gradient + rho * x
        scale = jnp.maximum(stable_norm(self._rmatvec(b)), jnp.maximum(
            stable_norm(gradient), stable_norm(rho * x)))
        valid = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(b))
        rel = jnp.where(valid, relative_ratio(stable_norm(s), scale), jnp.inf)
        return {"r_primal": rel, "r_dual": jnp.asarray(0.0), "gap": jnp.asarray(jnp.nan)}

    def certificate(self, x, y=None, data=None):
        values = self._certificate_arrays(x, self._resolve(data or {}))
        return {"r_primal": float(values["r_primal"]), "r_dual": 0.0,
                "gap": float("nan")}

    def certificate_many(self, x, y=None, data_batch=None):
        """Return batched CG certificate fields as device arrays.

        Uses the scalar CG residual and retains its unavailable (NaN) gap.
        ``x`` and every data leaf have the same leading batch axis. Omit data
        for a constant family. As in :meth:`certificate`, ``y`` is unused.
        """
        x, _y, data = _resolve_certificate_batch(
            self._resolve, x, None, data_batch, self.n)
        return self._report_many(x, data)

    def __repr__(self):
        return (f"CGCompiledFamily(n={self.n}, m={self.m}, method=cgls, "
                f"cg_iters={self.meta['cg_iters']}, rho={self.meta['rho']:g})")


FOLD_POLICY = (
    "a compile without an explicit metric mapping (equilibrate a string other than "
    "'construct'; no block_metrics, metric_classes or metric_columns) folds the first "
    "explicit nonsingular diagonal least-squares term without an offset into the primal "
    "step when the family declares no identity-operator primal term and has at least two "
    "non-identity terms. Under an explicit metric mapping (equilibrate=('given', tau, "
    "sigma), block_metrics, metric_classes, metric_columns or equilibrate='construct') "
    "every non-identity term is a dual term in declared order and nothing is folded, so "
    "supplied steps and metrics bind to the terms as declared")


def _foldable_ridge(term):
    """The plain routing policy's fold candidate: an explicit nonsingular diagonal
    least-squares term without an offset. An opaque operator is never inspected."""
    return (type(term.op) is Diagonal and type(term.block) is SumSquares
            and term.offset_fn is None and bool(jnp.all(term.op.d != 0)))


def _dual_term_lines(dual, dual_positions, blocks, slices):
    return [f"dual term {j} = declared term {p} ({type(b).__name__} over "
            f"{type(t.op).__name__}, rows {s}:{e})"
            for j, (t, p, b, (s, e)) in enumerate(zip(dual, dual_positions, blocks, slices))]


def _given_layout_note(dual, dual_positions, blocks, slices, m, fold_candidate_position):
    """One sentence naming the compiled dual layout, for refusals of supplied steps."""
    note = (f"the compiled model has {m} dual rows: "
            + "; ".join(_dual_term_lines(dual, dual_positions, blocks, slices)) + ". ")
    if fold_candidate_position is not None:
        j = dual_positions.index(fold_candidate_position)
        s, e = slices[j]
        note += (f"Under an explicit metric mapping nothing is folded into the primal step, so "
                 f"declared term {fold_candidate_position} ({type(blocks[j]).__name__} over "
                 f"Diagonal) is dual term {j} here (rows {s}:{e}); a plain compile of this "
                 f"family (no supplied steps or metrics) folds it into the primal step and has "
                 f"{m - (e - s)} dual rows. ")
    note += ("Supply one step per dual row of THIS layout, or a dict {dual_term_index: scalar "
             "or (rows,) array} bound to the dual terms by position; the compiled family "
             "records the mapping in meta['routing']")
    return note


def _bind_given_sigma(mapping, dual, dual_positions, blocks, slices, m):
    """Bind ``("given", tau, {dual_term_index: steps})`` to the compiled dual rows.

    Every dual term must have exactly one entry, a scalar or an array of that
    term's row count; keys are dual term indices (the same indices as
    ``block_metrics``). Anything else is refused by name before the flat
    vector exists. The result is the flat ``(m,)`` float32 step vector in
    compiled row order, which then passes the ordinary given-metric checks.
    """
    prefix = "equilibrate=('given', tau, sigma): "
    layout = (f"the compiled model has {m} dual rows: "
              + "; ".join(_dual_term_lines(dual, dual_positions, blocks, slices)))
    if not mapping:
        raise ValueError(f"{prefix}sigma is an empty dict; {layout}")
    for key in mapping:
        if isinstance(key, bool) or not isinstance(key, numbers.Integral) or not 0 <= key < len(dual):
            raise ValueError(f"{prefix}sigma key {key!r} is not a dual term index in "
                             f"[0, {len(dual)}); {layout}")
    parts = []
    for j, (t, p, b, (s, e)) in enumerate(zip(dual, dual_positions, blocks, slices)):
        name = f"dual term {j} = declared term {p} ({type(b).__name__} over {type(t.op).__name__}, rows {s}:{e})"
        if j not in mapping:
            raise ValueError(f"{prefix}sigma has no entry for {name}; it needs a scalar or a "
                             f"({e - s},) array; {layout}")
        value = jnp.asarray(mapping[j], jnp.float32)
        if value.ndim != 0 and value.shape != (e - s,):
            raise ValueError(f"{prefix}sigma[{j}] has shape {tuple(value.shape)}; {name} takes a "
                             f"scalar or a ({e - s},) array, one step per dual row of that term")
        parts.append(jnp.broadcast_to(value, (e - s,)))
    return jnp.concatenate(parts)


def _routing_table(family, primal_term_index, folded_term_position, fold_candidate_position,
                   dual, dual_positions, blocks, slices, m, explicit_metric_mapping,
                   step_source, metric_source):
    """Original term -> compiled role -> dual rows -> step and metric source -> witness.

    Recorded in ``meta["routing"]`` so that supplied steps and metrics can be
    bound to the declared terms by position and the returned dual read back by
    rows without re-deriving the compiler's routing.
    """
    rows = []
    for position, term in enumerate(family.terms):
        entry = {"declared_term": position, "op": type(term.op).__name__,
                 "block": type(term.block).__name__, "offset": term.offset_fn is not None}
        if position == primal_term_index:
            entry.update(role="primal", dual_term=None, rows=None, step_source="tau",
                         metric_source=None,
                         witness="none: no dual rows; the block enters the primal step and the "
                                 "coupling condition -K^T y - c in dom g^* is checked on it")
        elif position == folded_term_position:
            entry.update(role="primal_folded", dual_term=None, rows=None, step_source="tau",
                         metric_source=None,
                         witness="none: no dual rows; the ridge (1/2)||d x||^2 enters the primal "
                                 "step as its prox and its would-be dual d (d x) is never formed")
        else:
            j = dual_positions.index(position)
            s, e = slices[j]
            entry.update(role="dual", dual_term=j, rows=[s, e], step_source=step_source(j),
                         metric_source=metric_source(j), witness=f"sol.y[{s}:{e}]")
        rows.append(entry)
    return {"dual_rows": m, "terms": rows, "explicit_metric_mapping": explicit_metric_mapping,
            "fold_candidate": fold_candidate_position,
            "folded": folded_term_position is not None, "fold_policy": FOLD_POLICY}


def _validate_given_metric(tau, sigma, n, m, blocks, slices, block_metric_terms=(),
                           layout_note=""):
    """Eagerly reject a user-supplied metric the proximal blocks cannot honour.

    ``ocs.prox.base._validate_step`` raises for an invalid CONCRETE step, but a
    metric supplied through ``equilibrate=("given", ...)`` reaches the prox as a
    traced value inside the compiled loop, where the same check can only
    propagate NaN (a ``numerical_error`` status with an infinite residual and
    no explanation). This check runs once, at compile time, on host copies of
    the concrete arrays. It neither converts nor replaces ``tau``/``sigma``, so
    an accepted metric reaches the engine unchanged. The traced check remains
    as a backstop, e.g. for a runtime step multiplier or a calculus wrapper
    whose internal step products overflow.

    Contract checked, matching what the iteration hands the blocks:
      * ``tau`` broadcasts to ``(n,)`` and ``sigma`` to ``(m,)``;
      * every entry is finite and positive; every ``1/sigma`` is finite, since
        each dual block is evaluated with the step ``1/sigma``;
      * ``sigma`` is exactly uniform on each dual block of scaling class
        ``"scalar"`` (a coupled prox accepts one step per block).
    A non-uniform ``tau`` on a scalar-class PRIMAL block is not an error here:
    the compiler honours it by reducing it to one value, as before.
    Traced inputs (compilation inside a transformation) are left to the backstop.
    """
    if isinstance(tau, jax.core.Tracer) or isinstance(sigma, jax.core.Tracer):
        return
    prefix = "equilibrate=('given', tau, sigma): "
    host = {}
    for name, value, size in (("tau", tau, n), ("sigma", sigma, m)):
        array = np.asarray(value)
        try:
            shape = np.broadcast_shapes(array.shape, (size,))
        except ValueError:
            shape = None
        if shape != (size,):
            raise ValueError(f"{prefix}{name} has shape {array.shape}; it must be a scalar "
                             f"or broadcast to ({size},), one step per "
                             f"{'primal column' if name == 'tau' else 'dual row'}"
                             + (f". {layout_note}" if name == "sigma" and layout_note else ""))
        array = np.broadcast_to(array, (size,))
        with np.errstate(all="ignore"):
            bad = ~(np.isfinite(array) & (array > 0))
            if name == "sigma":
                bad |= ~np.isfinite(np.float32(1) / array)
        if bad.any():
            index = int(np.argmax(bad))
            raise ValueError(
                f"{prefix}{name}[{index}] = {array[index]!r} is invalid: every step must be "
                f"finite and positive{' with a finite reciprocal' if name == 'sigma' else ''} "
                f"({int(bad.sum())} of {size} entries fail)")
        host[name] = array
    for index, (block, (start, end)) in enumerate(zip(blocks, slices)):
        if block.scaling_class != "scalar" or index in block_metric_terms:
            # A declared row-side block metric has its own per-mathematical-block
            # step contract, checked by ocs.prox.block_metric.check_static_steps.
            continue
        rows = host["sigma"][start:end]
        if rows.size and not np.all(rows == rows[0]):
            raise ValueError(
                f"{prefix}sigma is not uniform on dual block {index} "
                f"({type(block).__name__}, scaling class 'scalar', rows {start}:{end}): "
                f"a scalar-class proximal block accepts exactly one step, but these rows carry "
                f"{np.unique(rows).size} distinct values in [{rows.min()!r}, {rows.max()!r}]. "
                "Supply one value per scalar-class block (or use a block with a "
                "diagonal or matrix scaling class).")


def _compile_cg(family, n, *, cg_iters, rho, tol, accuracy="objective_gap"):
    from ..prox.library import SumSquares, Zero
    from ..solver.cg import make_cgls_solver
    terms = family.terms
    if len(terms) != 1 or not isinstance(terms[0].block, SumSquares) or family.cost_fn is not None:
        raise ValueError("method='cg' needs exactly one least-squares term "
                         "1/2||K x - b||^2 and no linear cost")
    t = terms[0]; op = t.op; m = op.shape[0]
    solver = make_cgls_solver(op.matvec, op.rmatvec, n, m, iters=cg_iters, rho=rho, tol=tol)

    def resolve_b(raw):
        l = t.offset_fn(raw) if t.offset_fn is not None else 0.0
        return -jnp.broadcast_to(jnp.asarray(l, jnp.float32), (m,))   # min 1/2||Kx+l||^2

    meta = {"method": "cg", "cg_iters": cg_iters, "rho": rho, "n_dual": 1, "n_groups": 1,
            "scaled_norm": float("nan"), "equilibrate": "none", "block": 1,
            "tol": tol, "residual_norm": "relative", "gap_available": False, "scaling_classes": ["matrix"], "eval_kinds": ["closed_form"]}
    return CGCompiledFamily(solver, resolve_b, op.matvec, op.rmatvec, n, m, meta,
                            accuracy=accuracy)


def _shaped_eval_kind(ranges, block_eval_kind):
    """How the loop evaluates a shaped term.

    The cone automorphism keeps the block's OWN evaluation kind, because its
    weighted update wraps the cone's own Euclidean projection and runs no
    Newton solve. A quadratic shape takes one of the two quadratic closed
    forms. Everything else is the Newton norm prox.
    """
    shape = ranges[0][2]
    if isinstance(shape, _cone_metric.ConeAutomorphism):
        return block_eval_kind
    if isinstance(shape, _block_metric.QuadraticShape):
        return "linear_solve" if shape.quadratic_kind == "quadratic" else "closed_form"
    return "newton"


def _describe_block_metrics(term_metrics, slices, blocks, matvec, rmatvec, tau, sigma,
                            shape_apply, n, m, primal_scalar_metric, primal_weight,
                            column_shape_apply=None, column_shape=None):
    """Record the EFFECTIVE block metric and its measured spectral safety.

    Host float64 summaries of the declared shapes and of the dual steps after
    coercion, plus a 15-iteration operator-only power estimate of
    ||diag(sigma)^1/2 S^1/2 K diag(tau)^1/2 S_col^1/2||_2 on the steps the loop
    uses, with S the row shape (identity when no row term carries one) and
    S_col the declared column shape (identity when none was declared). A
    power estimate is a lower bound on the norm: an estimate above 1 is a
    definite violation of the PDHG step condition and is rejected.

    Three carrier families reach this record and each keeps its own keys: the
    norm/quadratic shape carriers of ``block_metric``, the symmetric-cone
    carrier of ``cone_metric``, and the declared column shape of the primal
    side. None of them overwrites another's entry.
    """
    root_tau = jnp.sqrt(tau)
    root_sigma = jnp.sqrt(sigma)

    def row_half(u):
        return u if shape_apply is None else shape_apply(u, power=0.5)

    def col_half(v):
        return v if column_shape_apply is None else column_shape_apply(v, power=0.5)

    def scaled_mv(x):
        return root_sigma * row_half(matvec(col_half(root_tau * x)))

    def scaled_rmv(y):
        return col_half(root_tau * rmatvec(row_half(root_sigma * y)))

    measured = float(_power_norm(scaled_mv, scaled_rmv, n, iters=15, seed=7))
    if not math.isfinite(measured) or measured >= 1.0:
        raise ValueError(f"block metric steps violate the PDHG step bound: measured "
                         f"||Sigma^1/2 K T^1/2||_2 ~ {measured:.6g} >= 1")
    norm_carrier = "(Q, w) eigenpair per sharing group, closure constant"
    cone_prox = ("closed form W^{-1} Pi_C(W v), W = P(w^{1/2}), on the cone's own Euclidean "
                 "projection, no Newton solve")
    spatial_prox = ("the weighted second-order-cone projection of ocs.prox.spatial_soc: three "
                    f"branches with the polar test on M v, and a bounded deterministic root "
                    f"solve of {_spatial_soc.BISECTION_STEPS} geometric bisection steps on a "
                    "derived bracket, with a status per cone")
    terms = []
    cone_terms = []
    newton_prox = (f"fixed {_block_metric.NEWTON_ITERS}-step safeguarded Newton, float32 (component "
                   f"form for k <= {_block_metric._COMPONENT_MAX_K}); closed-form radial prox on "
                   "sharing groups whose effective metric is isotropic in every block")
    for index, (k, ranges) in sorted(term_metrics.items()):
        start, end = slices[index]
        steps = jnp.broadcast_to(sigma, (m,))[start:end].reshape(-1, k)
        smin, smax = (float(v) for v in jax.device_get((jnp.min(steps), jnp.max(steps))))
        shape = ranges[0][2]
        quadratic = isinstance(shape, _block_metric.QuadraticShape)
        entry = {"term": index, "block": type(blocks[index]).__name__, "k": k,
                 "metric_blocks": (end - start) // k,
                 "mathematical_blocks": (end - start) // k,
                 "sharing_groups": [{**metric.describe(), "groups": [a, b],
                                     "isotropic_closed_form": bool(iso)}
                                    for a, b, metric, iso in ranges],
                 "dual_step_min": smin, "dual_step_max": smax,
                 "effective": ("Sigma_i = diag(sigma_i) S (S diagonal) or c_i S (S full), "
                               "applied through the quadratic block's closed form "
                               "(H + Sigma) z = v - b, y = v - Sigma z, on metric blocks of "
                               f"{k} rows" if quadratic else
                               "Sigma_i = diag(sigma_i) S (S diagonal) or c_i S (S full)"),
                 "prox": shape.prox_label if quadratic else newton_prox,
                 "block_family": "quadratic" if quadratic else "norm",
                 "carrier": norm_carrier}
        if isinstance(shape, _cone_metric.ConeAutomorphism):
            # The whole term is ONE plumbing block (k == the term's rows), so every
            # cone-level number is read from the carrier and never from k or from a
            # range bound, which would report 1 for a batch of several cones.
            cone_terms.append(index)
            entry["block_family"] = "cone"
            entry["k"] = int(shape.k)
            entry["rows"] = int(end - start)
            entry["plumbing_blocks"] = 1
            entry["metric_blocks"] = int(shape.cones)
            entry["mathematical_blocks"] = int(shape.cones)
            entry["prox"] = cone_prox
            # Sharing-group bounds are reported in units of the term's
            # mathematical blocks, which for a cone term are its cones, exactly
            # as item 2 disclosed them. The plumbing range (0, 1) is internal.
            entry["sharing_groups"] = [{**metric.describe(), "groups": [0, int(metric.cones)],
                                        "isotropic_closed_form": bool(iso)}
                                       for _a, _b, metric, iso in ranges]
            entry["effective"] = ("Sigma_i = c P(w_i) with ONE step c for the term; the dual update "
                                  "is y <- v - W Pi_C(W^{-1} v), W = P(w_i^{1/2}), in closed form")
            entry["carrier"] = ("Jordan element per cone: w (count, k) with cached w^1/2, w^-1/2 and "
                                "their determinants" if shape.algebra == "soc" else
                                "S (count, k, k) with cached S^1/2, S^-1/2")
            entry["cone_structure"] = {"algebra": shape.algebra, "cones": int(shape.cones),
                                       "k": int(shape.k), "elements": int(shape.count),
                                       "sharing": shape.sharing}
        if isinstance(shape, _spatial_soc.SpatialFullSOCMetric):
            # The same term-level reading as the automorphism carrier, for the
            # same reason: the whole term is ONE plumbing block, so every
            # cone-level number comes from the carrier and never from k or from a
            # range bound, which would report 1 for a batch of several cones.
            cone_terms.append(index)
            entry["block_family"] = "cone"
            entry["k"] = int(shape.k)
            entry["rows"] = int(end - start)
            entry["plumbing_blocks"] = 1
            cone_count = 1 if shape.shared else int(shape.cones)
            entry["metric_blocks"] = cone_count
            entry["mathematical_blocks"] = cone_count
            entry["prox"] = spatial_prox
            entry["sharing_groups"] = [{**metric.describe(),
                                        "groups": [0, cone_count],
                                        "isotropic_closed_form": bool(iso)}
                                       for _a, _b, metric, iso in ranges]
            entry["effective"] = (
                "Sigma_i = c M_i with M_i = diag(a_i, Q_i diag(b_i) Q_i^T) and ONE step c "
                "for the term; the dual update is y <- v - M P_M(M^{-1} v), in which c "
                "cancels because P_{cM} = P_M and P_M is positively homogeneous")
            entry["carrier"] = ("spatial-full factors per cone: a, b of shape (cones, k-1) "
                               "and an orthogonal Q of shape (cones, k-1, k-1), used AS "
                               "GIVEN and never reorthogonalized")
            entry["cone_structure"] = {"algebra": "soc", "cones": cone_count,
                                       "k": int(shape.k),
                                       "class": "spatial_full",
                                       "shared": bool(shape.shared)}
        terms.append(entry)
    prox = ("per term, see terms[i].prox. Norm blocks: " + newton_prox +
            ". Quadratic blocks: the closed form or the cached pencil solve "
            "(H + Sigma) z = v - b, y = v - Sigma z")
    domain = {"prox_kkt_bar": _block_metric.PROX_KKT_BAR,
              "max_full_cond": _block_metric.MAX_FULL_COND,
              "max_diagonal_cond": _block_metric.MAX_DIAGONAL_COND,
              "max_quadratic_hessian_ratio": _block_metric.MAX_QUADRATIC_HESSIAN_RATIO,
              "enforced": "rejected at declaration; a supplied shape is never clipped",
              "restrict_helper": "ocs.restrict_condition(S, max_cond)"}
    carrier_text = norm_carrier
    if cone_terms:
        # Only when a cone term is present, so the record of every existing
        # norm-block declaration keeps its bytes.
        prox += (". Automorphism cone terms: closed form W^{-1} Pi_C(W v) on the cone's own "
                 "Euclidean projection, no Newton")
        domain.update(max_automorphism_cond=_cone_metric.MAX_AUTOMORPHISM_COND,
                      automorphism_domain_status=_cone_metric.AUTOMORPHISM_DOMAIN_STATUS,
                      automorphism_restrict_helper="ocs.prox.cone_metric.restrict_cone_condition")
        carrier_text = ("per term, see terms[i]['carrier']: a (Q, w) eigenpair for a norm block, a "
                        "Jordan element per cone for an automorphism cone block, or spatial-full "
                        "(a, b, Q) factors per cone for a second-order cone block")
    out = {"present": bool(term_metrics),
           "carrier": carrier_text,
           "terms": terms, "measured_scaled_norm": measured,
           "numerical_domain": domain,
           "measured_scaled_norm_iterations": 15,
           "primal_scalar_metric_applied": bool(primal_scalar_metric),
           "primal_weight": primal_weight,
           "prox": prox}
    if column_shape is not None:
        # Present only with a declared column shape, so the row-only record is
        # byte for byte what it was.
        out["columns"] = {"present": True, "k": int(column_shape.k),
                          "count": int(n // column_shape.k), "cond": float(column_shape.cond),
                          "kind": column_shape.kind, "carrier": "one scalar",
                          "effective": "T_i = c S, one shared full SPD S = Q diag(w) Q^T, "
                                       "applied to the adjoint and to the cost"}
        out["carrier"] = (carrier_text + "; the column shape is one (Q, w) pair applied to the "
                          "primal vector")
        out["measured_scaled_norm_includes"] = "Sigma^1/2 (rows) and T^1/2 (columns)"
        out["measured_scaled_norm_method"] = (
            "re-measured AFTER construction by ocs.solver.equilibrate._power_norm: a 15-iteration "
            "operator-only power estimate (seed 7) of ||Sigma^1/2 K T^1/2||_2 on the decoded "
            "float32 column eigenpair and the steps the loop uses, after the primal weight; "
            "a power estimate is a LOWER bound on the norm, and a value at or above 1 is refused")
    return out



def _enforce_column_restriction(tau, n, columns):
    """The declared primal restriction, checked on the steps the loop will use.

    A declaration is never merely recorded. Whatever the construction did, the
    primal steps that reach the compiled loop must have the declared sharing
    structure: constant inside every declared primal metric-sharing group, and
    no more distinct values than the declaration allows. A construction that
    dropped the restriction fails the compile here instead of returning a
    metric from a class the caller did not ask for.
    """
    if isinstance(tau, jax.core.Tracer):
        # Compiled inside a transformation: the steps are traced, so there is
        # nothing concrete to check. Recorded as unchecked, never as checked.
        columns["enforced_on_applied_steps"] = "not checked: the primal steps are traced"
        return
    kind, spans = columns["kind"], columns["spans"]
    allowed = int(columns["metric_sharing_groups"])
    if kind == "full":
        # T = c (I kron S): the carrier c must be one value, uniform inside
        # every column block of k coordinates, and there is one sharing group.
        # `allowed` is 1 by design: no column path in this release produces a
        # per-block carrier c_i, so a per-block carrier would be a metric no
        # declaration asked for. Never vacuous.
        k, count = int(columns["column_blocks"]["k"]), int(columns["column_blocks"]["count"])
        flat = jnp.broadcast_to(tau, (n,))
        segments = flat.reshape(count, k)
        uniform = bool(jnp.all(segments == segments[:, :1]))
        distinct = int(jnp.unique(segments[:, 0]).size) if uniform else None
        if not uniform or distinct > allowed:
            raise ValueError(
                "the applied primal metric does not honour the declared column restriction: a "
                f"full column shape T = c (I kron S) requires one carrier c, uniform inside every "
                f"column block of {k} coordinates ({count} blocks declared), and the applied "
                f"steps carry {'a value that varies inside a block' if not uniform else str(distinct) + ' distinct values'}. "
                "The compile fails rather than apply a metric from a class the caller did not "
                "declare.")
        shape = columns.get("shape")
        columns["enforced_on_applied_steps"] = {
            "declared_sharing_groups": allowed, "column_blocks": count, "block_size": k,
            "distinct_applied_carriers": distinct,
            "shape_cond": None if shape is None else shape["cond"], "shape_k": k}
        return
    if kind == "diagonal":
        # Every primal coordinate is its own sharing group: nothing to tie, and
        # `allowed` is n, so no count can violate it either.
        columns["enforced_on_applied_steps"] = "vacuous: one sharing group per coordinate"
        return
    groups = ((0, n),) if kind == "scalar" else tuple((int(a), int(b)) for a, b in spans)
    flat = jnp.broadcast_to(tau, (n,))
    lengths = {b - a for a, b in groups}
    if len(lengths) == 1 and len(groups) > 1:
        segments = flat.reshape(len(groups), -1)
        uniform = bool(jnp.all(segments == segments[:, :1]))
        firsts = segments[:, 0]
    else:
        uniform = all(bool(jnp.all(flat[a:b] == flat[a])) for a, b in groups)
        firsts = jnp.stack([flat[a] for a, _b in groups])
    if not uniform:
        raise ValueError(
            f"the applied primal metric does not honour the declared column restriction: it "
            f"varies inside a declared metric-sharing group ({len(groups)} groups declared). "
            "The compile fails rather than apply a metric from a class the caller did not "
            "declare.")
    # Every group is uniform, so the distinct count is read from one value per
    # group: no sort over the whole primal space.
    distinct = int(jnp.unique(firsts).size)
    if distinct > allowed:
        raise ValueError(
            f"the applied primal metric has {distinct} distinct steps but the declaration "
            f"allows at most {allowed} metric-sharing groups")
    columns["enforced_on_applied_steps"] = {"declared_sharing_groups": allowed,
                                            "distinct_applied_steps": distinct}


def _describe_effective_metric(blocks, slices, groups, term_metrics, tau, sigma, n, m,
                               primal_scalar_metric, primal_metric_tie, primal_weight,
                               scaled_norm, column_shape=None):
    """Record the metric the compiled solver will actually apply.

    Written after every compile, so a silent collapse of a declared or
    constructed metric cannot recur unnoticed: whatever the caller asked for,
    this is what the loop got. Three groupings of the same rows are kept apart
    and never merged in this record:

    * ``mathematical_block``: the coordinates one proximal operator couples.
    * ``metric_sharing_groups``: how many distinct metrics those blocks use.
    * ``execution_groups``: which batched prox call evaluates them. Batching is
      a throughput choice and never merges norms or metrics.
    """
    if isinstance(tau, jax.core.Tracer) or isinstance(sigma, jax.core.Tracer):
        # Compiled inside a transformation: the steps are traced, so there are no
        # concrete numbers to record. Say that, rather than guess or fail.
        return {"terms": None, "execution_groups": None, "n_execution_groups": len(groups),
                "primal": None, "scaled_norm_at_construction": scaled_norm,
                "unavailable": "the metric is traced (compiled inside a transformation)"}
    host_tau = np.asarray(jax.device_get(tau))
    host_sigma = np.asarray(jax.device_get(sigma))

    def span(values, start, end, size):
        if values.ndim == 0:
            return float(values), float(values), 1
        piece = np.broadcast_to(values, (size,))[start:end]
        return float(piece.min()), float(piece.max()), int(np.unique(piece).size)

    terms = []
    for index, (blk, (start, end)) in enumerate(zip(blocks, slices)):
        size = end - start
        k = _block_metric.euclidean_norm_structure(blk, size)
        if k is None:
            k = size if blk.scaling_class == "scalar" else 1
        low, high, distinct = span(host_sigma, start, end, m)
        entry = {"term": index, "block": type(blk).__name__, "rows": [start, end],
                 "scaling_class": blk.scaling_class,
                 "mathematical_block": {"size_k": k, "count": size // k if k else 0,
                                        "coupled": blk.scaling_class == "scalar"},
                 "dual_step_min": low, "dual_step_max": high,
                 "distinct_dual_steps": distinct}
        cone = _cone_metric.symmetric_cone_structure(blk, size)
        if cone is not None:
            # The shipped scalar tier treats a batched cone as ONE coupled block
            # (mathematical_block above, the release's limit); the cone structure
            # sits beside it so neither count overwrites the other.
            entry["cone_structure"] = {"algebra": cone[0], "k": cone[1], "cones": cone[2],
                                       "scalar_tier_treats_term_as": "one coupled block"}
        if index in term_metrics:
            k_declared, ranges = term_metrics[index]
            carrier = ranges[0][2]
            quadratic = isinstance(carrier, _block_metric.QuadraticShape)
            if isinstance(carrier, _cone_metric.ConeAutomorphism):
                # ``blocks`` is in units of the term's mathematical blocks, its
                # cones, read from the carrier. The plumbing range bound is (0, 1)
                # because the whole term is one plumbing block.
                entry["metric_sharing_groups"] = [
                    {"blocks": [0, int(metric.cones)], "cones": int(metric.cones), "kind": metric.kind,
                     "algebra": metric.algebra, "k": metric.k, "count": metric.count,
                     "sharing": metric.sharing,
                     "cond": metric.cond, "cond_min": float(np.min(metric.cond_per_block)),
                     "degrees_of_freedom": metric.degrees_of_freedom * metric.count,
                     "jordan_rank": metric.jordan_rank,
                     "isotropic_closed_form": False}
                    for _a, _b, metric, _iso in ranges]
                entry["shape_applied"] = ("Sigma_i = c P(w_i), W = P(w_i^{1/2}), dual update "
                                          "y <- v - W Pi_C(W^{-1} v) in closed form; c is the "
                                          "term's one step")
                entry["k_declared"] = carrier.k
                entry["metric_block_size_k"] = carrier.k
            elif isinstance(carrier, _spatial_soc.SpatialFullSOCMetric):
                # Read in the same units as the automorphism carrier above, and
                # for the same reason: the whole term is one plumbing block, so
                # the cone count comes from the carrier.
                cone_count = 1 if carrier.shared else int(carrier.cones)
                entry["metric_sharing_groups"] = [
                    {"blocks": [0, cone_count], "cones": cone_count,
                     "kind": "spatial_full_soc", "algebra": "soc",
                     "k": int(metric.k), "shared": bool(metric.shared),
                     "isotropic_closed_form": False, **metric.describe()}
                    for _a, _b, metric, _iso in ranges]
                entry["shape_applied"] = (
                    "Sigma_i = c M_i with M_i = diag(a_i, Q_i diag(b_i) Q_i^T), dual update "
                    "y <- v - M P_M(M^{-1} v) through the weighted second-order-cone "
                    "projection; c is the term's one step and it cancels")
                entry["k_declared"] = int(carrier.k)
                entry["metric_block_size_k"] = int(carrier.k)
            else:
                entry["metric_sharing_groups"] = [
                    {"blocks": [a, b], "kind": metric.kind, "k": metric.k, "cond": metric.cond,
                     "isotropic_closed_form": bool(iso),
                     **({"quadratic": metric.describe()["quadratic"]} if quadratic else {})}
                    for a, b, metric, iso in ranges]
                entry["shape_applied"] = (
                    ("Sigma_i = diag(sigma_i) S (S diagonal) or c_i S (S full), applied through the "
                     "quadratic block's closed form (H + Sigma) z = v - b, y = v - Sigma z, on "
                     f"metric blocks of {k_declared} rows") if quadratic else
                    "Sigma_i = diag(sigma_i) S (S diagonal) or c_i S (S full)")
                entry["k_declared"] = k_declared
                entry["metric_block_size_k"] = k_declared
        else:
            entry["metric_sharing_groups"] = None
            entry["shape_applied"] = ("Sigma_i = diag(sigma_i): no shape; the scalar/diagonal "
                                      "tier code path, unmodified")
        terms.append(entry)

    execution = []
    for position, group in enumerate(groups):
        execution.append({"execution_group": position, "block": type(group.block).__name__,
                          "members": int(group.count), "rows_per_member": int(group.size),
                          "contiguous_span": list(group.span) if group.span is not None else None,
                          "indexed": group.span is None,
                          "carries_block_metric": group.metric is not None})
    low, high, distinct = span(host_tau, 0, n, n)
    primal = {"steps_min": low, "steps_max": high, "distinct_steps": distinct,
              "scalar_class_block": bool(primal_scalar_metric),
              "tie": primal_metric_tie, "primal_weight": primal_weight}
    if column_shape is not None:
        # Only with a declared column shape, in the form of a row term's entry.
        count = n // column_shape.k
        primal["metric_sharing_groups"] = [{"blocks": [0, count], "kind": column_shape.kind,
                                            "k": column_shape.k, "cond": column_shape.cond}]
        primal["column_blocks"] = {"k": column_shape.k, "count": count}
        primal["shape_applied"] = "T_i = c S, one shared full SPD S = Q diag(w) Q^T"
    return {"terms": terms, "execution_groups": execution,
            "n_execution_groups": len(groups),
            "primal": primal,
            "scaled_norm_at_construction": scaled_norm,
            "concepts": ("mathematical_block: coordinates coupled by one prox; "
                         "metric_sharing_groups: blocks sharing one stored shape; "
                         "execution_groups: blocks evaluated by one batched call")}


def compile_family(family, *, block=100, max_blocks=200, tol=1e-3,
                   probes=32, equilibrate="sgd", adaptive_omega=False,
                   method="pdhg", cg_iters=300, rho=0.0,
                   smfe_probes=2, smfe_iters=80, smfe_schedule="sgd",
                   online_warmup=0, online_method="jacobi",
                   residual_norm="relative", pdhg_variant="halpern", primal_weight=1.0,
                   dual_layout="flat", loop_unroll=1, data_layout="dense",
                   block_metrics=None, metric_classes=None, metric_columns=None,
                   metric_updates=None, metric_probes=None, metric_seed=0,
                   metric_safety=None, metric_mu=None, metric_box=None,
                   metric_shift=None, metric_domain="reject", metric_initial=None,
                   certificate_contract=None, accuracy="objective_gap",
                   gap_checks=64, gap_check_growth=2.0, gap_check_first=1,
                   witness_contraction=None,
                   refine=False, refine_budget=None, refine_memory_bytes=None,
                   dual_routes=("shift", "packing", "ridge")):
    """Compile a problem family.

    ``dual_routes`` (default ``("shift", "packing", "ridge")``, all on; ``()``
    turns them off; ``None`` takes ``DUAL_ROUTES_DEFAULT``) selects the
    dual-bound routes of the objective-gap controller (``DUAL_ROUTES``).  Each
    route builds one more dual candidate from the current iterate, for one
    structure decided on exact host facts at compile time: ``"shift"``, the
    recession shift (primal block NonNeg, some dual block NonNeg or
    unrestricted); ``"packing"``, the packing-LP assignment dual (primal block
    NonNeg, the only dual block NonNeg with operator ``Weighted(BinPack,
    -1)``); ``"ridge"``, the compound ridge (primal block NonNeg, exactly one
    SumSquares dual block with operator ``Diagonal(d > 0)`` on all columns,
    among at least two dual blocks).  A requested route that is not eligible
    is not built.  At a check where the evaluator's own packet meets the
    criterion no route runs, and the pair, status and iteration count returned
    there are those of ``dual_routes=()``; a run can therefore differ from
    ``dual_routes=()`` only by stopping at an earlier check on a route's
    candidate, or by reporting a stronger valid bound at a budget or stall
    exit with the request unmet.  At any other check each enabled route
    is applied to the same iterate, and its candidate counts only if the
    evaluator's strict-bound gates validate it and it is strictly better than
    the bound held; it is kept on its own retention chain, which never feeds
    the evaluator's chain or the stall rule.  The routes change neither
    criterion (7), the acceptance predicate nor any gate.  The routes run on the
    single-instance controller (``solve_instance``); the batched path
    (``solve_many``) does not consult them.  Each consultation is charged in
    ``Solution.gap_history`` (``dual_routes``, ``dual_route_seconds``,
    ``dual_route_operator_applications``; ``DUAL_ROUTE_OPERATOR_APPLICATIONS``
    per route), and a certifying route is named in
    ``Solution.objective_bound_source`` (``DUAL_ROUTE_SOURCES``).  Requested,
    eligible and enabled routes are recorded in ``meta["dual_routes"]``.

    ``refine`` (default ``False``) selects the bounded refinement policy of
    ``ocs.solver.refine_auto``.  With ``"auto"``, a run under the default
    ``objective_gap`` criterion that ends on the stall guard with the request
    unmet gets AT MOST ONE bounded refinement stage, on the two supported
    structures only (a separable-penalty-plus-least-squares family with a
    fixed support, the FFT lasso shape, by the primal-dual refinement route;
    a nonnegative-quadratic-plus-equality family, the quadratic OT shape, by
    the active-support solver route).  The refined pair is assessed by the
    UNCHANGED criterion (7) and returned only when it passes; otherwise the
    ordinary packet is returned unchanged with the outcome recorded.  A pair
    that already passes gets no refinement work; a family without an exact
    conjugate-domain contract gets the missing capability reported, never a
    manufactured assessment.  ``refine_budget`` is a
    ``refine_auto.RefineBudget`` (CG iterations, trials, assessments,
    operator applications, seconds); ``refine_memory_bytes`` is the admitted
    workspace budget in bytes (None: derived from the device, or the host,
    at the time of the stage).  ``Solution.refinement`` carries the record.

    ``accuracy`` selects WHICH CRITERION AWARDS ``status``.  The default,
    ``"objective_gap"``, accepts a point only when a strict lower bound on the
    optimal objective value is certified there and the relative gap to the
    primal value meets ``tol``.  ``"residual"`` restores the former behaviour,
    in which the relative prox-graph residual awarded ``status``; that quantity
    describes this algorithm's splitting rather than the point's objective
    accuracy, so it is no longer the default.  A family that declares no exact
    conjugate-domain contract cannot certify anything and reports
    ``"unverified"`` under the default, never ``"solved"``.

    ``gap_checks``, ``gap_check_growth`` and ``gap_check_first`` set the
    SCHEDULE on which the objective-gap criterion is evaluated along the
    run: at most ``gap_checks`` checks, the first after ``gap_check_first``
    blocks, each segment ``gap_check_growth`` times the previous one, the
    segments summing to ``max_blocks``.  The defaults (64, 2.0, 1) are the
    shipped geometric schedule 1, 2, 4, 8, ... blocks.  These change WHEN the
    criterion is asked, never what it accepts; a finer schedule resolves the
    stopping point more finely and pays for more checks inside the solve.
    The schedule in force is recorded in ``meta["gap_check_schedule"]``.

    ``equilibrate=("given", tau, sigma[, scaled_norm])`` supplies the steps.
    ``sigma`` is a scalar, a vector with one step per COMPILED dual row, or a
    dict ``{dual_term_index: scalar or (rows,) array}`` bound to the dual terms
    by position (the same indices as ``block_metrics``); every dual term needs
    exactly one entry and a wrong key, a missing term or a wrong length is
    refused by name. The compiled dual layout is a routing decision:
    ``FOLD_POLICY`` says when a diagonal least-squares term is folded into the
    primal step, which happens only WITHOUT an explicit metric mapping, so a
    plain compile and a ``("given", ...)`` compile of the same family can have
    different dual row counts. Every compiled family records the mapping in
    ``meta["routing"]`` (declared term -> role -> dual term -> rows -> step and
    metric source -> witness rows of ``sol.y``), and a refusal of a wrong-length
    ``sigma`` names that layout.

    ``block_metrics`` (optional, default None) declares row-side dual block
    metrics: ``{dual_term_index: BlockMetric or ConeAutomorphism or
    [BlockMetric, ...]}``, indices counting dual terms in order. A
    ``ConeAutomorphism`` (``ocs.prox.cone_metric``) is accepted only on a
    symmetric-cone block and only alone, never inside a list. Only supported with
    ``equilibrate=("given", tau, sigma, ...)``, where the caller has normalized
    ``Sigma = diag(sigma) S`` and ``tau`` against the spectral bound. See
    ``ocs.prox.block_metric``. With None every compiled path is unchanged.

    ``equilibrate="construct"`` BUILDS the metric instead of taking one: the
    caller declares a class per dual term in ``metric_classes``
    (``{dual_term_index: "scalar" | "diagonal" | "full" | "automorphism"}``, or
    ``(class, "per_block" | "term")`` to say which mathematical blocks share
    one metric, or ``(class, "term", k)`` on a separable quadratic block,
    ``SumSquares`` or ``Zero``, to declare the metric block size ``k`` of a
    shared shape; a ``Quadratic`` block takes its whole dimension and needs no
    ``k``) and the compiler constructs it, normalizes it once, records what it
    built in ``meta["metric_construction"]`` and applies it.
    ``"automorphism"`` is the cone tier of ``ocs.prox.cone_metric``: for a
    symmetric-cone block (``SecondOrderCone``, ``SOCBatch``, ``PSDCone``) one
    Jordan element per cone from the cone's automorphism group, with the
    weighted projection in closed form; the exponential and power cones
    refuse it with the reason. In this release the construct path REFUSES
    ``"automorphism"`` on every symmetric cone as well: the constructor of
    energy-selected SOC-automorphism metrics (and its PSD counterpart) is
    research-only because of a known float32 defect in its Jordan-eigenvalue
    reconstruction, and the ``ValueError`` names it. A ``ConeAutomorphism``
    supplied through ``block_metrics`` does not use that constructor and is
    accepted. ``metric_initial`` (construct path only) seeds the geodesic run
    of such a term with a ``ConeAutomorphism``, for example
    ``ocs.nt_scaling_point(block, s, z)``, and is therefore refused with it.

    ``metric_columns`` declares the primal (column) side. ``None`` is the
    shipped form and is not a declaration. ``"scalar"`` declares ONE SCALAR
    STEP PER DECLARED PRIMAL PROXIMAL BLOCK -- not one scalar for the whole
    primal space -- because that is what makes the three classes nest, scalar
    inside diagonal inside full, blockwise. A primal block is the set of
    coordinates the primal prox couples. So: with one coordinate per primal
    block (a separable primal prox, or none) the declared class IS the free
    per-coordinate diagonal, the same set of metrics and not a reduction of
    one, and it is reported with that many sharing groups; with the whole
    primal space in one coupled block it is one tied step, built with the tie
    in place; with several coupled primal blocks it is REFUSED, because this
    release applies exactly one primal step to a coupled primal prox and
    cannot honour it. ``"diagonal"`` declares a free per-coordinate diagonal.
    A sequence of tied ``(start, end)`` column spans declares one step per
    span; ``[(0, n)]`` is how one scalar over the WHOLE primal space is
    declared. ``("full", k)`` declares ONE shared ``k x k`` symmetric positive
    definite shape ``S`` over the ``n / k`` contiguous column blocks, so the
    primal step operator is ``T = c (I kron S)`` with ``c`` one scalar. It is
    accepted only when the family has no distinguished primal proximal block
    (there the primal update is the gradient step ``x - T (K^T y + c)``, and
    ``S`` is applied to the adjoint and to the cost inside the loop); every
    distinguished-block family refuses it with the reason for its block's
    class. The step condition ``||Sigma^1/2 K T^1/2||_2 < 1`` is measured
    inside the constructor and measured again here on the decoded shape and
    the steps the loop uses (``meta["block_metrics"]["measured_scaled_norm"]``).
    The certificate never sees ``T``. The caller's variable ordering decides
    what a column block is; nothing permutes coordinates. Whatever is
    declared, the restriction is enforced on the steps
    the compiled loop actually uses, and ``meta["effective_metric"]["primal"]``
    records them. ``metric_domain`` is ``"reject"`` (default) or ``"restrict"`` for a
    constructed shape outside the validated numerical domain; nothing is ever
    clipped silently. With ``metric_classes=None`` the declaration is the one
    each proximal block natively supports, and the construction is the shipped
    ``equilibrate="sgd"`` program called unchanged. See
    ``ocs.solver.metric_construct``.

    ``certificate_contract`` (optional, default None) declares an EXPLICITLY
    SCALED certificate contract beside the strict relative one: per-block
    absolute budgets the caller states in the units of the problem (see
    ``ocs.solver.contract.ScaledContract``). With the shipped default
    ``stopping="strict_only"`` the iteration loop is EXACTLY the default loop
    and the contract only classifies the point that loop already reached; the
    strict residual is still computed, still reported, and still the only
    thing that sets ``status``. ``stopping="strict_or_contract"`` also lets
    the contract end the iteration and is EXPERIMENTAL. With None every
    compiled path is unchanged.

    ``witness_contraction`` (``None``, ``"compound"`` or ``"once"``) selects
    how the objective-bound evaluator's dual-correction rounds apply each
    block's interior safety adjustment when they build a recovery candidate
    (``WITNESS_CONTRACTION_POLICIES``).  ``"compound"`` applies it in every
    round, so a second-order cone block is pulled ``rounds`` times
    ``(4 k + 16) eps32`` inside its polar cone whether or not a round corrected
    a coupling violation; ``"once"`` builds the candidate with the exact
    projections and applies the adjustment one time, on the final candidate.
    Neither changes what is ACCEPTED: the candidate is still validated as a
    whole by the unchanged exact conjugate-domain and coupling gates, the
    acceptance decision and criterion (7) are untouched, and ``recover=False``
    evaluations are identical under both.  ``None`` takes the module default
    ``WITNESS_CONTRACTION_DEFAULT``.  Recorded in ``meta["witness_contraction"]``.
    """
    n = family.n
    witness_contraction = _check_witness_contraction(witness_contraction)
    dual_routes = _check_dual_routes(dual_routes)
    if data_layout not in ("dense", "compact"):
        raise ValueError("data_layout must be 'dense' or 'compact'")
    if data_layout == "compact" and method != "pdhg":
        raise ValueError("compact data layout requires PDHG")
    if dual_layout not in ("flat", "blocks"):
        raise ValueError("dual_layout must be 'flat' or 'blocks'")
    if dual_layout == "blocks" and (method != "pdhg" or pdhg_variant != "standard"):
        raise ValueError("block dual layout requires standard PDHG")
    adaptive_omega = bool(adaptive_omega)
    # Fixed steps by default. The adaptive weight (opt-in) carries bounded
    # updates; measured on the scheduling LP family it wins on the degenerate
    # per-server program and fails on the single-shape program at 1e6 tasks,
    # so no automatic choice is shipped. adaptive_blocks freezes it after that
    # many blocks when set.
    adaptive_blocks = None
    if isinstance(loop_unroll, bool) or not isinstance(loop_unroll, numbers.Integral) or loop_unroll <= 0:
        raise ValueError("loop_unroll must be a positive integer")
    loop_unroll = int(loop_unroll)
    if loop_unroll != 1 and (method != "pdhg" or pdhg_variant != "standard"
                             or adaptive_omega or online_warmup > 0):
        raise ValueError("nondefault loop_unroll requires standard PDHG with fixed steps")
    primal_weight = float(primal_weight)
    if (not math.isfinite(primal_weight) or primal_weight <= 0
            or primal_weight > jnp.finfo(jnp.float32).max):
        raise ValueError("primal_weight must be finite and positive")
    if primal_weight != 1.0 and (method != "pdhg" or online_warmup > 0):
        raise ValueError("nondefault primal_weight requires PDHG without online equilibration")
    if pdhg_variant not in ("halpern", "standard"):
        raise ValueError("pdhg_variant must be 'halpern' or 'standard'")
    if pdhg_variant == "standard" and (method != "pdhg" or adaptive_omega or online_warmup > 0):
        raise ValueError("standard PDHG requires fixed steps without adaptive or online updates")
    if residual_norm not in NORMS:
        raise ValueError(f"residual_norm must be one of {NORMS}, got {residual_norm!r}")
    if tol <= 0 or not math.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    refine_policy = _check_refine_policy(refine)
    if refine_policy and method != "pdhg":
        raise ValueError("refine='auto' is a policy of the primal-dual (PDHG) path; "
                         "the CGLS path has no stall guard and no refinement route")
    if certificate_contract is not None:
        # Opt-in only. The declared budgets are validated here and never
        # inferred; the strict relative residual is still computed and reported.
        certificate_contract = ScaledContract.coerce(certificate_contract)
        if residual_norm != "relative" or method != "pdhg":
            raise ValueError("certificate_contract requires residual_norm='relative' and PDHG")

    # ---- pure least-squares: conjugate-gradient path --------------------
    # min 1/2||K x - b||^2 (one SumSquares term, no g, no cost) is the CG/LSQR
    # regime, where restart-PDHG plateaus.  Route it to matrix-free CGLS.
    if method == "cg":
        return _compile_cg(family, n, cg_iters=cg_iters, rho=rho, tol=tol,
                           accuracy=accuracy)

    # ---- route primal g vs dual blocks ----------------------------------
    g_block, dual, dual_positions, primal_term_index = None, [], [], None
    for position, t in enumerate(family.terms):
        if g_block is None and isinstance(t.op, Identity) and t.offset_fn is None:
            g_block, primal_term_index = t.block, position
        else:
            dual.append(t)
            dual_positions.append(position)
    # An explicitly declared diagonal least-squares term has a coordinatewise
    # primal prox. Folding it avoids imposing a gratuitous exact adjoint
    # cancellation on its auxiliary dual. The represented coefficients and
    # objective are unchanged. Preserve existing primal choices and indexed
    # metric declarations; never inspect an opaque operator to find a matrix.
    # The policy is FOLD_POLICY, recorded with the routing in meta["routing"]:
    # an explicit metric mapping (supplied steps or metrics, or a constructed
    # metric) binds to the terms as declared, so nothing is folded under one.
    folded_diagonal_quadratic = False
    folded_term_position = fold_candidate_position = None
    explicit_metric_mapping = (block_metrics is not None or metric_classes is not None
                               or metric_columns is not None
                               or not isinstance(equilibrate, str) or equilibrate == "construct")
    if g_block is None and len(dual) > 1:
        fold_candidate = next((i for i, t in enumerate(dual) if _foldable_ridge(t)), None)
        if fold_candidate is not None:
            fold_candidate_position = dual_positions[fold_candidate]
            if not explicit_metric_mapping:
                t = dual[fold_candidate]
                g_block = t.block.precomposed(a=t.op.d)
                dual.pop(fold_candidate)
                folded_term_position = dual_positions.pop(fold_candidate)
                folded_diagonal_quadratic = True
    if not dual:
        raise ValueError("need at least one term through a non-identity operator")

    ops = [t.op for t in dual]
    blocks = [t.block for t in dual]
    dims = [op.shape[0] for op in ops]
    m = sum(dims)
    slices, a = [], 0
    for d in dims:
        slices.append((a, a + d)); a += d
    scalar_spans = [sl for blk, sl in zip(blocks, slices)
                    if blk.scaling_class == "scalar"]

    if online_warmup > 0 and scalar_spans:
        raise ValueError("online equilibration does not preserve scalar dual metrics")

    # ---- constructed metric: declaration checked before anything is built --
    construct = equilibrate == "construct"
    metric_settings = {"updates": metric_updates, "probes": metric_probes,
                       "seed": metric_seed, "safety": metric_safety, "mu": metric_mu,
                       "box": metric_box, "shift": metric_shift, "domain": metric_domain}
    if not construct:
        offered = [name for name, value in
                   (("metric_classes", metric_classes), ("metric_columns", metric_columns),
                    ("metric_updates", metric_updates), ("metric_probes", metric_probes),
                    ("metric_safety", metric_safety), ("metric_mu", metric_mu),
                    ("metric_box", metric_box), ("metric_shift", metric_shift),
                    ("metric_initial", metric_initial))
                   if value is not None]
        offered += [name for name, value, default in
                    (("metric_seed", metric_seed, 0),
                     ("metric_domain", metric_domain, "reject"))
                    if value != default]
        if offered:
            raise ValueError(f"{', '.join(offered)} require equilibrate='construct': the metric "
                             "is only built for you on that path")
    else:
        if block_metrics is not None:
            raise ValueError("block_metrics supplies a metric and equilibrate='construct' builds "
                             "one; choose one of them")
        if online_warmup > 0:
            raise ValueError("online equilibration does not preserve a constructed block metric")
        if method != "pdhg":
            raise ValueError("equilibrate='construct' requires PDHG")

    # ---- declared row-side block metrics (validated eagerly, host only) --
    term_metrics = {}
    if block_metrics is not None:
        if not isinstance(block_metrics, dict) or not block_metrics:
            raise ValueError("block_metrics must be a nonempty dict {dual_term_index: BlockMetric}")
        if not (isinstance(equilibrate, tuple) and equilibrate and equilibrate[0] == "given"):
            raise ValueError("block_metrics require equilibrate=('given', tau, sigma, ...): the "
                             "caller must normalize the block metric against the spectral bound")
        if online_warmup > 0:
            raise ValueError("online equilibration does not preserve block dual metrics")
        if residual_norm != "relative":
            raise ValueError("block_metrics require residual_norm='relative' (metric-free stopping)")
        for index, declared in block_metrics.items():
            if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < len(dual):
                raise ValueError(f"block_metrics key {index!r} is not a dual term index in [0, {len(dual)})")
            s0, e0 = slices[index]
            term_metrics[int(index)] = _block_metric.normalize_term_metrics(
                declared, index, blocks[index], e0 - s0)

    # ---- stacked operator (matrix-free) ---------------------------------
    def matvec(x):
        return jnp.concatenate([op.matvec(x) for op in ops])

    def rmatvec(y):
        return sum(op.rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))

    # ---- matrix-free equilibration (once) -------------------------------
    # operator-aware skip: if every dual operator is already well conditioned
    # (e.g. an orthonormal transform), equilibration has nothing to do.
    well_conditioned_ops = all(getattr(op, "well_conditioned", False) for op in ops)
    if equilibrate == "sgd" and well_conditioned_ops:
        equilibrate = "scalar"

    # A scalar-class primal block (every cone, GroupL2, L2, a default ProxBlock)
    # accepts one primal step. Constructors that can solve their problem with
    # that constraint in place (SMFE) are told so; any other metric is reduced
    # to its minimum below, as before.
    primal_scalar_metric = g_block is not None and g_block.scaling_class == "scalar"
    primal_metric_tie = None

    equil_diag = {}
    constructed_metrics, construction_report = {}, None
    given_sigma_form = None      # "scalar" | "vector" | "dict" on the given path
    column_shape = None          # a constructed ("full", k) column shape, else None
    if isinstance(equilibrate, tuple) and equilibrate[0] == "given":
        # externally supplied per-coordinate steps (e.g. residual-targeted
        # re-equilibration at a restart boundary); caller has already applied
        # the spectral safety rescale
        if len(equilibrate) < 3:
            raise ValueError("equilibrate=('given', tau, sigma[, scaled_norm]) "
                             f"needs a tau and a sigma; got {len(equilibrate) - 1} entries")
        tau = jnp.asarray(equilibrate[1], jnp.float32)
        if isinstance(equilibrate[2], dict):
            # Steps bound to the dual terms by position (the block_metrics
            # indices), assembled into the flat vector in compiled row order.
            given_sigma_form = "dict"
            sigma = _bind_given_sigma(equilibrate[2], dual, dual_positions, blocks, slices, m)
        else:
            sigma = jnp.asarray(equilibrate[2], jnp.float32)
            given_sigma_form = "scalar" if sigma.ndim == 0 else "vector"
        scaled_norm = float(equilibrate[3]) if len(equilibrate) > 3 else float("nan")
        # The prox blocks re-check their steps inside the compiled loop, but
        # there the metric is traced and an unhonourable one can only become
        # NaN. Refuse it here, on concrete values, before anything is traced.
        _validate_given_metric(tau, sigma, n, m, blocks, slices, tuple(term_metrics),
                               layout_note=_given_layout_note(dual, dual_positions, blocks, slices,
                                                              m, fold_candidate_position))
    elif equilibrate == "exact":                 # deterministic RAS via the squared-operator oracle
        if all(hasattr(op, "sq_matvec") for op in ops):
            from ..solver.equilibrate_exact import exact_equilibration

            def sq_mv(x):
                return jnp.concatenate([op.sq_matvec(x) for op in ops])

            def sq_rmv(y):
                return sum(op.sq_rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))

            tau, sigma, scaled_norm = exact_equilibration(
                sq_mv, sq_rmv, matvec, rmatvec, n, m, scalar_spans)
        else:                                     # structure absent: stochastic fallback
            tau, sigma, scaled_norm, equil_diag = smfe_equilibration(
                matvec, rmatvec, n, m, scalar_spans, diagnostics=True,
                probes=smfe_probes, iters=smfe_iters, schedule=smfe_schedule,
                tie_columns=primal_scalar_metric, _cache_operators=tuple(ops))
            primal_metric_tie = "construction" if primal_scalar_metric else None
    elif construct:                              # DECLARED class: build the metric
        # The caller declared what it wants; ocs.solver.metric_construct builds
        # it from operator products only and reports exactly what it built. A
        # declaration the shipped SMFE program already expresses IS run by that
        # program, unchanged, so the diagonal case stays Diamond-Boyd's method.
        from ..solver.metric_construct import construct_metric, DEFAULT_UPDATES, DEFAULT_PROBES, \
            DEFAULT_SAFETY, DEFAULT_MU, DEFAULT_BOX, DEFAULT_SHIFT
        tau, sigma, scaled_norm, constructed_metrics, construction_report = construct_metric(
            matvec, rmatvec, n, m, blocks, slices,
            metric_classes=metric_classes, metric_columns=metric_columns,
            primal_scalar_metric=primal_scalar_metric, primal_block=g_block,
            updates=DEFAULT_UPDATES if metric_updates is None else metric_updates,
            probes=DEFAULT_PROBES if metric_probes is None else metric_probes,
            seed=metric_seed,
            safety=DEFAULT_SAFETY if metric_safety is None else metric_safety,
            mu=DEFAULT_MU if metric_mu is None else metric_mu,
            box=DEFAULT_BOX if metric_box is None else metric_box,
            shift=DEFAULT_SHIFT if metric_shift is None else metric_shift,
            domain=metric_domain, smfe_iters=smfe_iters, smfe_probes=smfe_probes,
            smfe_schedule=smfe_schedule, cache_operators=tuple(ops),
            well_conditioned=well_conditioned_ops, metric_initial=metric_initial)
        equil_diag = construction_report.get("diagnostics", {})
        # A declared full column shape travels as a BlockMetric carrier on the
        # columns record; its public form is the describe() dict. None on every
        # other path, so an undeclared compile's record is what it was.
        column_shape = construction_report["columns"].pop("shape_carrier", None)
        if column_shape is not None:
            construction_report["columns"]["shape"] = column_shape.describe()
        if construction_report["columns"].get("kind") == "full" and column_shape is None:
            # A declared k x k class must arrive with its shape or be refused.
            # Whatever path returned a "full" record without a carrier (the
            # well-conditioned scalar skip would, if it ever accepted the full
            # kind again) would otherwise hand the loop one scalar step under
            # the name of a k x k class. Refused, with the regime named.
            raise ValueError(
                f"metric_columns={construction_report['columns'].get('declared')!r} was declared "
                f"and the constructor (regime {construction_report.get('regime')!r}) returned no "
                "column shape: the compile refuses rather than apply one scalar step under the "
                "name of a k x k column class. Nothing was built for this declaration.")
        if primal_scalar_metric:
            # The construction solved its problem with the one-primal-step
            # constraint in place, so the reduction below is the identity.
            primal_metric_tie = "construction"
    elif equilibrate == "sgd":                   # Diamond-Boyd matrix-free SMFE
        tau, sigma, scaled_norm, equil_diag = smfe_equilibration(
            matvec, rmatvec, n, m, scalar_spans, diagnostics=True,
            probes=smfe_probes, iters=smfe_iters, schedule=smfe_schedule,
            tie_columns=primal_scalar_metric, _cache_operators=tuple(ops))
        primal_metric_tie = "construction" if primal_scalar_metric else None
    elif equilibrate == "diagpp":                # Remedy A: Hutch++ Gram-diagonal + binorm
        tau, sigma, scaled_norm, equil_diag = diagpp_equilibration(
            matvec, rmatvec, n, m, scalar_spans, diagnostics=True)
    elif equilibrate == "diagonal":              # one-shot Hutchinson (experimental)
        tau, sigma, scaled_norm = diagonal_equilibration(
            matvec, rmatvec, n, m, scalar_spans, probes=probes)
    elif equilibrate == "scalar":
        t0, s0, scaled_norm = scalar_steps(matvec, rmatvec, n)
        # Keep uniform family metrics scalar; prox boundaries preserve
        # the existing vector-step contract through traced broadcasting.
        if online_warmup > 0:
            # Online engines subsequently form coordinatewise metrics; retain
            # their original initial representation and rounding behavior.
            tau = jnp.full((n,), t0, jnp.float32)
            sigma = jnp.full((m,), s0, jnp.float32)
        else:
            tau = jnp.asarray(t0, jnp.float32)
            sigma = jnp.asarray(s0, jnp.float32)
    else:
        raise ValueError("equilibrate must be 'scalar', 'diagonal', 'sgd', 'diagpp', "
                         "'construct', or ('given', tau, sigma[, scaled_norm])")

    if primal_weight != 1.0:
        tau = tau / primal_weight
        sigma = sigma * primal_weight
        if not bool(jnp.all(jnp.isfinite(tau) & (tau > 0))) or not bool(
                jnp.all(jnp.isfinite(sigma) & (sigma > 0))):
            raise ValueError("primal_weight must preserve finite positive steps")

    if primal_scalar_metric:
        if online_warmup > 0:
            raise ValueError("online equilibration does not preserve a scalar primal metric")
        # A coupled primal prox (e.g. PSD) cannot consume coordinatewise steps.
        # SMFE already solved its construction with one tied primal step and
        # measured the norm of that metric, so the reduction below is the
        # identity there. For other metrics (given, exact RAS, diagonal,
        # diagpp) lowering every step to their minimum preserves the spectral
        # safety bound without another operator pass. It is not a good choice:
        # columns the operator never touches drag the minimum down while
        # contributing nothing to that bound.
        if primal_metric_tie is None and tau.ndim != 0:
            primal_metric_tie = "minimum"
            tau = jnp.full((n,), jnp.min(tau), dtype=tau.dtype)
        elif tau.ndim != 0:
            # Built with the one-step constraint in place, so the reduction
            # above is the identity here. Verify that on the concrete steps
            # rather than assume it: a metric that reached this point varying
            # inside a coupled primal block must fail the compile, not be
            # quietly replaced by its minimum.
            flat = jnp.broadcast_to(tau, (n,))
            if not bool(jnp.all(flat == flat[0])):
                raise ValueError(
                    "the primal proximal block is scalar class and accepts exactly one step, "
                    f"but the metric built under the '{primal_metric_tie}' tie varies across "
                    "the primal space. The compile fails rather than reduce it.")
            tau = flat

    if construction_report is not None:
        _enforce_column_restriction(tau, n, construction_report["columns"])
    if column_shape is not None and residual_norm != "relative":
        raise ValueError("a constructed column shape requires residual_norm='relative' "
                         "(metric-free stopping): the historical 'iterate' and 'data' residuals "
                         "multiply tau elementwise and cannot take an operator step.")

    if constructed_metrics:
        # The construction decided which terms need a shape (or need steps that
        # vary inside a coupled block). From here the constructed metric takes
        # exactly the path a hand-supplied one takes: it is validated, checked
        # against the enforced numerical domain and applied by code this change
        # does not touch.
        if residual_norm != "relative":
            raise ValueError("a constructed block metric requires residual_norm='relative' "
                             "(metric-free stopping)")
        for index, metric in constructed_metrics.items():
            s0, e0 = slices[index]
            term_metrics[int(index)] = _block_metric.normalize_term_metrics(
                metric, index, blocks[index], e0 - s0)

    block_metric_meta = None
    shape_apply = None
    canonical_terms = []
    if term_metrics:
        for index, (k, ranges) in list(term_metrics.items()):
            s0, e0 = slices[index]
            term_metrics[index] = (k, _block_metric.check_static_steps(sigma, s0, e0, k, ranges, index))
        # A declaration that is exactly the scalar-tier metric (S = I and one
        # uniform step on the term) runs the unmodified scalar-tier code.
        canonical_terms = sorted(index for index, (k, ranges) in term_metrics.items()
                                 if _block_metric.equivalent_to_scalar_tier(sigma, *slices[index], ranges))
        for index in canonical_terms:
            del term_metrics[index]
    if term_metrics:
        if not bool(jnp.all(jnp.isfinite(tau) & (tau > 0))):
            raise ValueError("primal steps must be finite and positive")
        shape_apply = _block_metric.make_shape_apply(term_metrics, slices, m)

    # A declared full column shape S is applied to the PRIMAL vector by the
    # row side's applier called with one pseudo-term covering [0, n) and one
    # range covering every column block: v.reshape(count, k) @ Q * w @ Q^T
    # (power 1) and the same with sqrt(w) (power 0.5). No new numerics.
    column_shape_apply = None
    if column_shape is not None:
        if not bool(jnp.all(jnp.isfinite(tau) & (tau > 0))):
            raise ValueError("primal steps must be finite and positive")
        column_count = n // column_shape.k
        column_shape_apply = _block_metric.make_shape_apply(
            {0: (column_shape.k, ((0, column_count, column_shape),))}, ((0, n),), n)

    # ---- proxes (dual prox uses type-grouped GPU dispatch) --------------
    if term_metrics:
        groups = build_groups(blocks, slices,
                              metrics=[term_metrics.get(i) for i in range(len(blocks))])
    else:
        groups = build_groups(blocks, slices)
    # Scalar blocks cannot use online coordinatewise metric updates (checked
    # above). Their fixed sigma rows can be compacted after exact validation;
    # the ordinary iteration only multiplies sigma by a scalar primal weight.
    dual_apply = grouped_moreau_dual(
        groups, m, static_sigma=sigma,
        compact_scalar_sumsquares=(pdhg_variant == "standard" and online_warmup == 0))

    def prox_primal(xt, tw, data):
        tw = jnp.broadcast_to(tw, (n,))
        xt = xt - tw * data[1]                       # fold in cost c
        return g_block.prox(xt, tw) if g_block is not None else xt

    fixed_primal = (fixed_nonnegative_quadratic_prox(g_block, tau, n)
                     if pdhg_variant == "standard" and online_warmup == 0
                     and not adaptive_omega else None)

    def iteration_prox_primal(xt, tw, data):
        if fixed_primal is None:
            return prox_primal(xt, tw, data)
        tw = jnp.broadcast_to(tw, (n,))
        return fixed_primal(xt - tw * data[1])

    def prox_dual(yt, sw, data):
        sw = jnp.broadcast_to(sw, (m,))
        return dual_apply(yt + sw * data[0], sw)     # fold in offset l

    # With a declared block metric the dual step is Sigma = diag(sw) S. The
    # iteration forms y + Sigma (K xbar + l): S rides on the forward product
    # handed to the PDHG loop (and folds into the offset here), so solver
    # signatures and loop carries are unchanged. The certificate and status
    # checks keep the plain operator and never see S.
    iteration_matvec = matvec
    if shape_apply is not None:
        def iteration_matvec(x):
            return shape_apply(matvec(x))

        def prox_dual(yt, sw, data):
            sw = jnp.broadcast_to(sw, (m,))
            return dual_apply(yt + sw * shape_apply(data[0]), sw)

    # With a declared full column shape the primal step operator is
    # T = c (I kron S). The update x - T (K^T y + c) puts S on BOTH the adjoint
    # (it rides on the adjoint handed to the loop) and the cost (folded here):
    # pushing only the adjoint would leave x - c S K^T y - c c, which is not a
    # gradient step in any metric. There is no primal prox on this path
    # (normalize_columns refused every distinguished block), which is asserted.
    # The default path binds the same objects as before.
    iteration_rmatvec = rmatvec
    if column_shape_apply is not None:
        assert g_block is None and not primal_scalar_metric and fixed_primal is None

        def iteration_rmatvec(y):
            return column_shape_apply(rmatvec(y))

        def prox_primal(xt, tw, data):
            tw = jnp.broadcast_to(tw, (n,))
            return xt - tw * column_shape_apply(data[1])      # x - T (K^T y + c)

    # smooth dual blocks: (slice, grad f_i) -- used for exact termination
    smooth_terms = [(s, e, blk.smooth_grad) for blk, (s, e) in zip(blocks, slices)
                    if getattr(blk, "smooth_grad", None) is not None]
    cone_reference_scales = None
    structural_zeros = ({"zero_offsets": tuple(t.offset_fn is None for t in dual),
                         "zero_cost": family.cost_fn is None}
                        if data_layout == "compact" else {})
    if residual_norm == "relative":
        cone_reference_scales = make_cone_reference_scales(g_block, ops, blocks, slices)
        def adjoint_with_scale(y):
            total = jnp.zeros((n,), y.dtype)
            scale = jnp.asarray(0.0, y.dtype)
            for op, (a, b) in zip(ops, slices):
                value = op.rmatvec(y[a:b])
                total += value
                scale = jnp.maximum(scale, stable_norm(value))
            return total, scale
        raw_components = make_relative_components(
            matvec, rmatvec, g_block, blocks, slices,
            adjoint_with_scale=adjoint_with_scale,
            cone_reference_scales=cone_reference_scales, **structural_zeros)
    else:
        raw_components = make_residual_components(
            matvec, rmatvec, prox_primal, prox_dual, tau, sigma,
            norm=residual_norm, slices=slices, smooth_terms=smooth_terms)
    repair_dual = make_dual_repair(matvec, smooth_terms)
    recoverable_cones = tuple((blk, start, end)
        for blk, (start, end) in zip(blocks, slices)
        if type(blk) in (ExpCone, ExpConeBatch, PowerCone, PowerConeBatch,
                         PSDCone, SecondOrderCone, SOCBatch))
    if recoverable_cones:
        smooth_repair = repair_dual

        def repair_dual(x, y, data):
            original = smooth_repair(x, y, data)
            canonical = original
            kx = matvec(x)
            current = kx + data[0]
            for blk, start, end in recoverable_cones:
                value = original[start:end]
                margin = 64 * jnp.finfo(value.dtype).eps
                if type(blk) is PSDCone:
                    matrix = current[start:end].reshape((blk.k, blk.k))
                    # A small positive slack after Kx+l cancellation can still
                    # need a nonzero near-boundary normal. Judge interior at
                    # the homogeneous data/iterate scale, not slack alone.
                    scale = jnp.maximum(jnp.max(jnp.abs(matrix), initial=0.0),
                        jnp.maximum(jnp.max(jnp.abs(kx[start:end]), initial=0.0),
                                    jnp.max(jnp.abs(data[0][start:end]), initial=0.0)))
                    unit = matrix / jnp.where(scale > 0, scale, 1.0)
                    symmetric = (unit + unit.T) * 0.5
                    skew = (unit - unit.T) * 0.5
                    interior = (jnp.max(jnp.abs(skew), initial=0.0) <= margin) & (
                        jnp.min(jnp.linalg.eigvalsh(symmetric)) > margin)
                    multiplier = value.reshape((blk.k, blk.k))
                    # PSD has relative interior in the symmetric subspace.
                    # Its normal contains skew matrices, even at a PD point.
                    # Keep that valid normal instead of zeroing all entries.
                    proposed = jnp.where(interior,
                        ((multiplier - multiplier.T) * 0.5).reshape(-1), value)
                else:
                    rows = _rows(blk, current[start:end])
                    scale = jnp.maximum(jnp.max(jnp.abs(rows), axis=1, initial=0.0),
                        jnp.maximum(jnp.max(jnp.abs(_rows(blk, kx[start:end])), axis=1, initial=0.0),
                                    jnp.max(jnp.abs(_rows(blk, data[0][start:end])), axis=1, initial=0.0)))
                    unit = rows / jnp.where(scale > 0, scale, 1.0)[:, None]
                    if type(blk) in (SecondOrderCone, SOCBatch):
                        # At a strict SOC interior point the only normal is
                        # zero. Scale each cone by its own slack/data/iterate
                        # magnitudes, preserving near-boundary multipliers.
                        interior = (unit[:, 0] - jnp.linalg.norm(unit[:, 1:], axis=1)) > margin
                    elif type(blk) in (ExpCone, ExpConeBatch):
                        first, second, third = unit[:, 0], unit[:, 1], unit[:, 2]
                        safe_second = jnp.maximum(second, jnp.finfo(value.dtype).tiny)
                        safe_third = jnp.maximum(third, jnp.finfo(value.dtype).tiny)
                        slack = second * (jnp.log(safe_third) - jnp.log(safe_second)) - first
                        interior = (second > margin) & (third > margin) & (slack > margin)
                    else:
                        first, second, third = unit[:, 0], unit[:, 1], unit[:, 2]
                        slack = (jnp.maximum(first, 0.0)**blk.alpha *
                                 jnp.maximum(second, 0.0)**(1-blk.alpha) - jnp.abs(third))
                        interior = (first > margin) & (second > margin) & (slack > margin)
                    proposed = jnp.where(interior[:, None], 0.0, _rows(blk, value))
                    if type(blk) in (ExpConeBatch, PowerConeBatch):
                        proposed = jnp.concatenate([proposed[:, i] for i in range(3)])
                    elif type(blk) is SOCBatch:
                        proposed = _cone_metric.unrows(_cone_metric.cone_layout(blk), proposed)
                    else:
                        proposed = proposed[0]
                canonical = set_window(canonical, start, end, proposed)

            # This recovers an actual dual candidate; it does not relax the
            # certificate. Check the original and proposed pair with the same
            # full coupled KKT evaluator used to report and stop. In particular
            # stationarity protects active multipliers near a cone boundary.
            def score(dual):
                values = components(x, dual, data)
                return jnp.maximum(values["r_primal"], values["r_dual"])

            # Trace one coupled evaluator over the two actual candidates.
            # The selected pair is still checked again before it is returned.
            scores = jax.vmap(score)(jnp.stack((original, canonical)))
            improve = scores[1] < scores[0]
            best = jnp.where(improve, canonical, original)
            return jnp.where(jnp.all(jnp.isfinite(y)), best, y)
    domain_checks = make_domain_checks(matvec, rmatvec, g_block, blocks, slices,
                                      **structural_zeros)
    feasibility_family = ((g_block is None or type(g_block) is Zero
                           or _indicator(g_block) is not None) and all(
        type(b) is Zero or _indicator(b) is not None for b in blocks))
    if feasibility_family:
        ordinary_repair = repair_dual

        def repair_dual(x, y, data):
            checked = ordinary_repair(x, y, data)
            # A zero dual is canonical for a pure feasibility objective. Only
            # recover it after primal feasibility passes, so infeasible cases
            # retain their proposed Farkas direction for status checking.
            feasible = domain_checks(x, jnp.zeros_like(y), data)["r_domain"] <= tol
            recover = (jnp.all(data[1] == 0) & jnp.all(jnp.isfinite(y)) & feasible)
            return jnp.where(recover, jnp.zeros_like(y), checked)

    guards_enabled = (domain_checks.has_experimental_cones or
                      (residual_norm == "relative" and domain_checks.has_indicators))

    def components(x, y, data):
        checked = raw_components(x, y, data)
        stationarity_scale = checked.pop("_stationarity_scale", None)
        cone_scales = checked.pop("_cone_reference_scales", None)
        objective_reference = checked.pop("_objective_reference", None)
        if guards_enabled:
            guards = domain_checks(x, y, data, stationarity_scale,
                                   cone_scales, objective_reference)
            checked["r_fixedpoint_primal"] = checked["r_primal"]
            checked["r_fixedpoint_dual"] = checked["r_dual"]
            checked["r_primal"] = jnp.maximum(checked["r_primal"], guards["r_domain"])
            checked["r_dual"] = jnp.maximum(checked["r_dual"], jnp.maximum(
                guards["r_polar"], guards["r_complementarity"]))
            checked.update(guards)
        return checked

    def residual(x, y, data):
        checked = components(x, repair_dual(x, y, data), data)
        return jnp.maximum(checked["r_primal"], checked["r_dual"])

    def checked_residual(x, y, data):
        # The same residual, together with the repaired dual it was measured
        # at.  The residual path stops on this one evaluation and returns this
        # pair and this value: the candidate that is checked is the candidate
        # that is returned, with the residual that was tested.
        repaired = repair_dual(x, y, data)
        checked = components(x, repaired, data)
        return jnp.maximum(checked["r_primal"], checked["r_dual"]), repaired

    # ---- opt-in scaled contract (declared budgets) ----------------------
    # In the shipped "strict_only" mode nothing below touches the loop: the
    # measure and the tolerance handed to the engine stay the default objects,
    # so the compiled iteration is the default iteration, bit for bit.
    contract_components = None
    contract_factory = None
    loop_measure, loop_tol = residual, tol
    if residual_norm == "relative" and method == "pdhg":
        def contract_factory(declared):
            return make_contract_components(
                matvec, rmatvec, g_block, blocks, slices, declared, tol,
                guards_enabled=guards_enabled, adjoint_with_scale=adjoint_with_scale,
                cone_reference_scales=cone_reference_scales, **structural_zeros)
    if certificate_contract is not None:
        contract_components = contract_factory(certificate_contract)
        if certificate_contract.stops_the_loop:      # EXPERIMENTAL, explicit opt-in
            def loop_measure(x, y, data):
                # Stop when the strict residual meets tol OR the declared
                # scaled contract holds. The reported residual stays strict.
                repaired = repair_dual(x, y, data)
                checked = components(x, repaired, data)
                strict = jnp.maximum(checked["r_primal"], checked["r_dual"])
                ratio = contract_components(x, repaired, data)["contract"]
                return jnp.where(strict <= tol, jnp.zeros_like(ratio), ratio)
            loop_tol = 1.0

    # The residual path evaluates its stopping residual once per check and
    # reports that evaluation.  The experimental online engines and a contract
    # that stops the loop keep the former stop-then-recheck wrapper: the first
    # take no check, and the second stops on a measure that is not the
    # reported residual.
    single_evaluation = (online_warmup == 0 and not (
        certificate_contract is not None and certificate_contract.stops_the_loop))

    def build_iteration_solver(primal, dual_prox, measure, *, batched=True,
                               tol_override=None, max_blocks_override=None,
                               budget_arg=False, check=None):
        # ``tol_override=0.0`` builds a DRIVING executable: one whose own
        # residual test can never end a segment, so that the certified
        # objective gap alone decides when to stop.
        loop_tol_x = loop_tol if tol_override is None else tol_override
        max_blocks_x = max_blocks if max_blocks_override is None else max_blocks_override
        if budget_arg and online_warmup > 0:
            # The online/OSGM warmup engines take no traced budget.  Rather
            # than drive them on a schedule they do not support, report no
            # driving executable; the gap mode then scores the ordinary solve
            # on the gap instead of steering it.
            return None
        if online_warmup > 0:                         # online equilibration (experimental)
            # Online updates change the algorithm metric, not the public accuracy
            # contract. Check against the fixed family metric also used by report().
            def residual_p(x, y, data, _tau, _sigma):
                return measure(x, y, data)
            if online_method == "osgm":               # SMFE cold start + OSGM residual-driven refine
                solver = make_osgm_solver(matvec, rmatvec, tau, sigma, primal, dual_prox,
                                          residual_p, n, m, block=block,
                                          max_blocks=max_blocks_x,
                                          tol=loop_tol_x, adaptive_omega=adaptive_omega,
                                          warmup_blocks=online_warmup)
            else:                                     # legacy one-shot Jacobi (kept for the record)
                solver = make_online_solver(matvec, rmatvec, tau, sigma, primal, dual_prox,
                                            residual_p, n, m, block=block,
                                            max_blocks=max_blocks_x,
                                            tol=loop_tol_x, adaptive_omega=adaptive_omega,
                                            warmup_blocks=online_warmup)
        else:
            solver = make_solver(iteration_matvec, iteration_rmatvec, tau, sigma, primal, dual_prox,
                                 measure, n, m, block=block, max_blocks=max_blocks_x,
                                 tol=loop_tol_x, budget_arg=budget_arg,
                                 adaptive_omega=adaptive_omega, variant=pdhg_variant,
                                 dual_slices=slices if dual_layout == "blocks" else None,
                                 loop_unroll=loop_unroll, batched=batched,
                                 adaptive_blocks=adaptive_blocks, check=check)
        return solver

    solver = build_iteration_solver(
        iteration_prox_primal, prox_dual, loop_measure,
        check=checked_residual if single_evaluation else None)
    iteration_solver = solver

    @operand_jit
    def solver(data, x0, y0):
        if single_evaluation:
            # The loop returns the checked pair and the residual it tested.
            return iteration_solver(data, x0, y0)
        xx, yy, count, _res = iteration_solver(data, x0, y0)
        yy = jax.vmap(repair_dual)(xx, yy, data)
        checked = jax.vmap(components)(xx, yy, data)
        return xx, yy, count, jnp.maximum(checked["r_primal"], checked["r_dual"])

    # ---- driving executables for the default objective-gap criterion -----
    # The existing block budget is divided into ``gap_checks`` segments, so the
    # TOTAL iteration budget is unchanged; what changes is the test that
    # accepts the result and the fact that it is now evaluated along the way.
    gap_checks = max(1, int(gap_checks))
    gap_check_growth = float(gap_check_growth)
    gap_check_first = max(1, int(gap_check_first))
    if not np.isfinite(gap_check_growth) or gap_check_growth < 1.0:
        raise ValueError("gap_check_growth must be a finite number >= 1.0")

    def build_drive_solvers(primal, dual_prox, measure, repair, check):
        """Run a traced number of blocks, never stopping on the rejected residual.

        ``tol_override=0.0`` is what makes this a DRIVING executable: its own
        residual test can never end a segment, so only the certified objective
        gap decides when the run stops.
        """
        if not bound_capable_all:
            return None, None
        iteration = build_iteration_solver(primal, dual_prox, measure,
                                           tol_override=0.0, budget_arg=True)
        if iteration is None:
            return None, None

        @operand_jit
        def batched_drive(data, x0, y0, cap):
            xx, yy, count, _res = iteration(data, x0, y0, cap)
            yy = jax.vmap(repair)(xx, yy, data)
            checked = jax.vmap(check)(xx, yy, data)
            return xx, yy, count, jnp.maximum(checked["r_primal"], checked["r_dual"])

        single_drive = None
        if pdhg_variant == "standard":
            iteration1 = build_iteration_solver(primal, dual_prox, measure,
                                                batched=False, tol_override=0.0,
                                                budget_arg=True)
            if iteration1 is not None:
                @operand_jit
                def single_drive(data, x0, y0, cap):
                    xx, yy, count, _res = iteration1(data, x0, y0, cap)
                    yy = repair(xx, yy, data)
                    checked = check(xx, yy, data)
                    return xx, yy, count, jnp.maximum(checked["r_primal"],
                                                      checked["r_dual"])
        return batched_drive, single_drive

    def build_single_solver(primal, dual_prox, measure, repair, check, checked=None):
        if pdhg_variant != "standard":
            return None
        use_check = single_evaluation and checked is not None
        iteration = build_iteration_solver(primal, dual_prox, measure, batched=False,
                                           check=checked if use_check else None)

        @operand_jit
        def single(data, x0, y0):
            if use_check:
                # The loop returns the checked pair and the residual it tested.
                return iteration(data, x0, y0)
            xx, yy, count, _res = iteration(data, x0, y0)
            yy = repair(xx, yy, data)
            checked = check(xx, yy, data)
            return xx, yy, count, jnp.maximum(checked["r_primal"], checked["r_dual"])
        return single

    # ---- per-instance data:  (l, c) -------------------------------------
    def resolve_data(raw):
        parts = []
        for t, (s, e) in zip(dual, slices):
            off = t.offset_fn(raw) if t.offset_fn is not None else 0.0
            parts.append(jnp.broadcast_to(jnp.asarray(off, jnp.float32), (e - s,)))
        l = jnp.concatenate(parts)
        c = (family.cost_fn(raw) if family.cost_fn is not None
             else jnp.zeros(n, jnp.float32))
        return (l, c)

    def resolve_compact_data(raw):
        # Resolve callbacks eagerly exactly as the dense resolver does. The
        # absent marker is structural: never infer zero from supplied values.
        # A scalar token gives all-zero families a batch leaf without N zeros.
        parts = tuple(None if term.offset_fn is None else
                      jnp.asarray(term.offset_fn(raw), jnp.float32)
                      for term in dual)
        # Preserve dense resolver shape validation without materializing a
        # broadcast here; malformed callbacks must fail even with zero blocks.
        for off, dim in zip(parts, dims):
            if off is not None:
                shape = jnp.broadcast_shapes(off.shape, (dim,))
                if shape != (dim,):
                    raise ValueError(f"offset shape {off.shape} does not broadcast to {(dim,)}")
        cost = family.cost_fn(raw) if family.cost_fn is not None else None
        if family.cost_fn is not None and cost is None:
            raise ValueError("cost callback must return an array, not None")
        return parts, cost, jnp.asarray(0, jnp.int32)

    def expand_compact_data(data):
        parts, cost, _token = data
        l = jnp.concatenate([
            jnp.broadcast_to(jnp.asarray(0.0, jnp.float32) if off is None else off,
                             (dim,))
            for off, dim in zip(parts, dims)])
        c = jnp.zeros(n, jnp.float32) if cost is None else cost
        return l, c

    def compact_prox_primal(xt, tw, data):
        # Keep the same arithmetic even for an absent cost (tw*0 can be NaN
        # for an invalid metric). Only the zero storage becomes scalar.
        cost = jnp.asarray(0.0, jnp.float32) if data[1] is None else data[1]
        # The existing proximal contract accepts a full metric vector even
        # when a future engine stores the uniform metric as a scalar.
        tw = jnp.broadcast_to(tw, (n,))
        xt = xt - tw * cost
        return g_block.prox(xt, tw) if g_block is not None else xt

    if column_shape_apply is not None:
        def compact_prox_primal(xt, tw, data):
            # The same fold as the dense layout: the cost passes through S.
            # An absent cost is the zero vector through S, finite for a finite
            # metric (tw * S 0 is 0, never NaN).
            cost = jnp.asarray(0.0, jnp.float32) if data[1] is None else data[1]
            tw = jnp.broadcast_to(tw, (n,))
            return xt - tw * column_shape_apply(jnp.broadcast_to(cost, (n,)))

    def compact_iteration_prox_primal(xt, tw, data):
        if fixed_primal is None:
            return compact_prox_primal(xt, tw, data)
        cost = jnp.asarray(0.0, jnp.float32) if data[1] is None else data[1]
        tw = jnp.broadcast_to(tw, (n,))
        return fixed_primal(xt - tw * cost)

    def compact_prox_dual(yt, sw, data):
        sw = jnp.broadcast_to(sw, (m,))
        adjusted = []
        for off, (a, b) in zip(data[0], slices):
            value = jnp.asarray(0.0, jnp.float32) if off is None else off
            step = sw[a:b]
            adjusted.append(yt[a:b] + step * value)
        return dual_apply(jnp.concatenate(adjusted), sw)

    if shape_apply is not None:
        def compact_prox_dual(yt, sw, data):
            sw = jnp.broadcast_to(sw, (m,))
            offsets = jnp.concatenate([
                jnp.broadcast_to(jnp.asarray(0.0, jnp.float32) if off is None else off,
                                 (b - a,))
                for off, (a, b) in zip(data[0], slices)])
            return dual_apply(yt + sw * shape_apply(offsets), sw)

    def compact_residual(x, y, data):
        # Expand only at the certificate consumer. Compact pieces remain the
        # loop's input pytree; there is no dense expansion outside the engine.
        return residual(x, y, expand_compact_data(data))

    compact_loop_measure = compact_residual
    if contract_components is not None and certificate_contract.stops_the_loop:
        def compact_loop_measure(x, y, data):
            return loop_measure(x, y, expand_compact_data(data))

    # ---- certificate reporting (primal/dual residual + optional gap) ----
    gap_ok = all(_overrides(b, "conj_value") for b in blocks) and \
        all(_overrides(b, "value") for b in blocks) and \
        (g_block is None or (_overrides(g_block, "conj_value") and
                             _overrides(g_block, "value")))

    # Indicator oracles currently admit feasibility tolerances. Their finite
    # values alone cannot establish an exact Fenchel gap. Keep this flag
    # conservative until each value/conjugate domain has a strict contract.
    exact_gap_contract = (all(type(b) is SumSquares for b in blocks) and
                          (g_block is None or type(g_block) is SumSquares))

    # ---- strict objective lower bound (opt-in, per block) ---------------
    # Every block, and g, must declare the exact conjugate-domain contract
    # (ProxBlock.bound_capable) before ANY objective lower bound is reported.
    # One block without it makes the certificate unavailable: a surrogate is
    # never reported under the stronger name.
    bound_capable_all = bool(gap_ok and
                             all(getattr(b, "bound_capable", False) for b in blocks) and
                             (g_block is None or getattr(g_block, "bound_capable", False)))

    def _conj_argument(kty, c):
        """The argument of g^*: z = -K^T y - c."""
        return -kty - c

    # A demonstrated LP failure occurs when a dense float32 adjoint rounds a
    # negative slack to zero. For this explicit operator class only, bound the
    # product/accumulation and subsequent ray-scaling error by an absolute
    # adjoint action. No matrix is extracted from an opaque operator. The
    # conservative envelope also covers rounding the proposed scaled witness.
    _dense_orthant_guard = (type(g_block) is NonNeg and
                           all(type(op) is Dense for op in ops))
    _allocation_orthant_guard = (type(g_block) is NonNeg and
        any(type(op) is Weighted and type(op.base) is BinPack for op in ops) and
        all(type(op) is Diagonal or
            (type(op) is Weighted and type(op.base) is BinPack) for op in ops))
    _guarded_orthant = _dense_orthant_guard or _allocation_orthant_guard

    # Marginal sums are structurally explicit, but no constraint matrix is
    # formed. Decide the LP domain using the represented multipliers and costs
    # in scoped f64, including an outward roundoff allowance. The ordinary f32
    # gate remains necessary. This is not a relaxation of the orthant domain.
    from ..operators.structured import MarginalOperator
    _marginal_orthant_guard = (type(g_block) is NonNeg and len(ops) == 1
                               and type(ops[0]) is MarginalOperator)

    def _marginal_domain(y, c):
        if not _marginal_orthant_guard:
            return jnp.asarray(True)
        op = ops[0]
        with jax.enable_x64():
            yy = y.astype(jnp.float64)
            cc = c.astype(jnp.float64).reshape(op.N, op.Mc)
            u, v = yy[:op.N, None], yy[None, op.N:]
            z = -u - v - cc
            envelope = (8 * np.finfo(np.float64).eps) * (jnp.abs(u) + jnp.abs(v) + jnp.abs(cc))
            ok = jnp.all(jnp.isfinite(z)) & jnp.all(z + envelope <= 0)
        return ok

    # ---- l1 coupling through a subsampled Fourier adjoint ---------------
    # With g = lam ||x||_1 the coupling condition is ||-K^T y - c||_inf <= lam.
    # L1.conj_in_domain is exact on its argument, but here its argument is the
    # float32 FFT adjoint, whose componentwise rounding is of order
    # eps32 ||y||_1 / sqrt(n). With lam << lam_max that can exceed 1e-4 lam,
    # while the ray search sets its top only 64 ulps inside the float32
    # boundary. A demonstrated false pass (Gaussian right-hand side, N = 1024)
    # accepted a witness 4.4e-6 lam outside the domain in float64. For this
    # structure only, the coupling of the REPRESENTED float32 witness is decided
    # on a scoped float64 adjoint of the same operator (_fftmask_adjoint64),
    # charged a componentwise envelope for the float64 transform. The domain is
    # never relaxed; float32 search trials are unchanged proposals.
    def _unique_frequency_subset(op):
        host = np.asarray(op.idx)
        return bool(host.size and np.all((host >= 0) & (host < op.n))
                    and np.unique(host).size == host.size)

    _fft_l1_guard = (type(g_block) is L1 and bool(ops)
                     and all(type(op) is FFTMask and _unique_frequency_subset(op)
                             for op in ops))
    _eps64 = float(np.finfo(np.float64).eps)
    _eps32 = float(np.finfo(np.float32).eps)

    def _fftmask_adjoint64(op, u):
        """K^T u for one FFTMask in float64: the operator ``op.rmatvec``
        applies (both of its implementations represent this adjoint). The kept
        real and imaginary parts are scattered as two float64 arrays at the
        unique indices; a complex128 scatter is serialized by XLA on the GPU
        (no 16-byte atomic), measured at about 3 s per gap check at
        N = 262,144. One complex128 inverse FFT follows. Assessment only."""
        zr = jnp.zeros(op.n, jnp.float64).at[op.idx].set(u[:op.m], unique_indices=True)
        zi = jnp.zeros(op.n, jnp.float64).at[op.idx].set(u[op.m:], unique_indices=True)
        return jnp.real(jnp.fft.ifft(jax.lax.complex(zr, zi), norm="ortho"))

    def _fft_gamma(n_op, eps):
        # Componentwise envelope constant of one length-n transform: each output
        # is a sum of the n inputs over ceil(log2 n) butterfly levels; 16 ulps
        # per level covers complex products, twiddles and additions, and 64 more
        # cover the ortho scaling, the scatter(-add) and non-radix-2 paths.
        levels = max(1, int(math.ceil(math.log2(max(2, int(n_op))))))
        return (16 * levels + 64) * eps

    def _fft_l1_assess(y, c):
        """(ok, ratio, allowance) for the represented float32 witness ``y``.

        ``ok``: max|-K^T y - c| plus the float64 envelope is <= lam, decided in
        float64.  ``ratio``: that total over lam, rounded up to float32.
        ``allowance``: the a priori float32 envelope of the same adjoint over
        lam (rounded up), i.e. how far the float32 test could be from the true
        value. ``allowance`` is a band for the scaled fallback and a
        diagnostic; it is never a slack on the domain."""
        # The point assessed is the materialized witness, never a fused
        # higher-precision expression of it.
        y = jax.lax.optimization_barrier(y)
        with jax.enable_x64():
            y64 = y.astype(jnp.float64)
            c64 = jnp.asarray(c).astype(jnp.float64)
            kty = sum(_fftmask_adjoint64(op, y64[s:e]) for op, (s, e) in zip(ops, slices))
            assert kty.dtype == jnp.float64
            peak = jnp.max(jnp.abs(-kty - c64))
            env64 = jnp.zeros((), jnp.float64)
            env32 = jnp.zeros((), jnp.float64)
            for op, (s, e) in zip(ops, slices):
                mass = jnp.sum(jnp.abs(y64[s:e])) / math.sqrt(op.n)
                env64 = env64 + _fft_gamma(op.n, _eps64) * mass
                env32 = env32 + _fft_gamma(op.n, _eps32) * mass
            # The sum over operators and the subtraction of c.
            scale = jnp.max(jnp.abs(kty)) + jnp.max(jnp.abs(c64))
            env64 = env64 + (len(ops) + 2) * _eps64 * scale
            env32 = env32 + (len(ops) + 2) * _eps32 * scale
            total = (peak + env64) * (1.0 + 4 * _eps64)
            lam = float(g_block.lam)
            finite = (jnp.all(jnp.isfinite(y64)) & jnp.all(jnp.isfinite(c64))
                      & jnp.isfinite(total))
            ok = finite & (total <= lam)
            if lam > 0:
                ratio, allow = total / lam, env32 / lam
            else:
                ratio = jnp.where(total == 0, 0.0, jnp.inf)
                allow = jnp.where(env32 == 0, 0.0, jnp.inf)
            ratio = jnp.where(finite, ratio, jnp.inf)
            # Unconditional upward steps: no round-trip comparison for XLA to fold.
            ratio32 = jnp.nextafter(ratio.astype(jnp.float32), jnp.float32(jnp.inf))
            allow32 = jnp.nextafter(allow.astype(jnp.float32), jnp.float32(jnp.inf))
        return ok, ratio32, allow32

    def _dense_abs_adjoint(y):
        if not _guarded_orthant:
            return jnp.zeros((), jnp.float32)
        def subnormal(v):
            bits = jax.lax.bitcast_convert_type(v, jnp.uint32) & 0x7fffffff
            return jnp.any((bits != 0) & (bits < 0x00800000))
        unsafe = subnormal(y)
        for op in ops:
            if type(op) is Dense:
                unsafe |= subnormal(op.A)
            elif type(op) is Diagonal:
                unsafe |= subnormal(op.d)
            else:
                unsafe |= subnormal(op.w) | subnormal(op.base.Rmat)
        rounded = rmatvec(y)
        # Recompute the products, not an already rounded scalar, in a scoped
        # X64 assessment. Opaque operators and iteration buffers are unchanged.
        # Dense is the explicit small/validation operator; this can promote its
        # stored matrix temporarily and is not a billion-vector memory promise.
        with jax.enable_x64():
            refs, absolutes = [], []
            for op, (a, b) in zip(ops, slices):
                yy = y[a:b].astype(jnp.float64)
                if type(op) is Dense:
                    aa = op.A.astype(jnp.float64)
                    refs.append(jnp.matmul(aa.T, yy, precision=jax.lax.Precision.HIGHEST))
                    absolutes.append(jnp.matmul(jnp.abs(aa).T, jnp.abs(yy),
                                                precision=jax.lax.Precision.HIGHEST))
                elif type(op) is Diagonal:
                    product = op.d.astype(jnp.float64)*yy
                    refs.append(product)
                    absolutes.append(jnp.abs(product))
                else:
                    # Apply the declared BinPack structure in f64. Only its
                    # stored R-by-I coefficients are promoted; the enormous
                    # (I+R*B)-by-(I*B) operator is never formed.
                    base = op.base
                    yy = op.w.astype(jnp.float64)*yy
                    uu, ww = yy[:base.I], yy[base.I:].reshape(base.R, base.B)
                    aa = base.Rmat.astype(jnp.float64)
                    refs.append((uu[:, None]+jnp.matmul(aa.T, ww,
                                 precision=jax.lax.Precision.HIGHEST)).reshape(-1))
                    absolutes.append((jnp.abs(uu)[:, None]+jnp.matmul(jnp.abs(aa).T,
                                      jnp.abs(ww), precision=jax.lax.Precision.HIGHEST)).reshape(-1))
            ref, absolute = sum(refs), sum(absolutes)
            assert ref.dtype == jnp.float64 and absolute.dtype == jnp.float64
            q = (2*m+8)*np.finfo(np.float64).eps
            gamma = 2*q/(1-q) if q < .5 else float('inf')
            error = jnp.abs(ref-rounded.astype(jnp.float64))+gamma*absolute
            absolute32 = absolute.astype(jnp.float32)
            error32 = error.astype(jnp.float32)
        # Round the two nonnegative enclosures outward. Include flush-to-zero
        # uncertainty for the original f32 action and refuse subnormal inputs.
        zero = jnp.all(y == 0)
        floor = jnp.where(zero, 0., 2*m*np.finfo(np.float32).tiny)
        absolute32 = jnp.where(zero, 0., jnp.nextafter(absolute32, jnp.inf))
        error32 = jnp.where(zero, 0., jnp.nextafter(error32, jnp.inf))+floor
        return jnp.where(unsafe, jnp.inf, jnp.stack([absolute32, error32]))

    def _dense_margin(absolute_product, z, *, materialized_ray=False):
        if not _guarded_orthant:
            return jnp.zeros((), jnp.float32)
        # Measured f32 adjoint error + a roundoff envelope for the materialized
        # scaled witness, cached product scaling and addition of the cost.
        eps = float(np.finfo(np.float32).eps)
        absolute, error = absolute_product[0], absolute_product[1]
        # A supplied witness has already been materialized: its fresh adjoint
        # error was measured above. Charge extra witness/cached-adjoint scaling
        # only while proposing a ray point. Charging it again at recover=False
        # could reject the very witness that just certified the solve.
        return error + (4*eps*absolute if materialized_ray else 0.) + 2*eps*jnp.abs(z)

    # PSD-type blocks (``resolved_sign_domain``) are decided with a RESOLVED
    # sign: ``conj_in_domain_resolved`` admits only what ``conj_in_domain``
    # admits AND clears a float32-scale envelope, evaluated in scoped float64
    # (stricter, never looser).  Every other block is decided exactly as before.
    _psd_resolved = (any(getattr(b, "resolved_sign_domain", False) for b in blocks)
                     or bool(getattr(g_block, "resolved_sign_domain", False)))
    def _block_in_domain(blk, v):
        if getattr(blk, "resolved_sign_domain", False):
            return blk.conj_in_domain_resolved(v)
        return blk.conj_in_domain(v)

    def _dual_domain_parts(y, z, coupling_margin=0.0):
        """The three separable halves of EXACT dual feasibility, kept apart so the
        report can say WHICH one failed.

        ``coupling`` is the part a per-block test cannot see: the dual is
        feasible only if ``-K^T y - c`` lies in ``dom g^*``.  For ``g = 0`` that
        domain is ``{0}``, i.e. exact equality ``K^T y + c = 0``, and NO
        tolerance is admissible: a dual that satisfies every block domain while
        violating this is not dual feasible and its Fenchel value is not a bound
        on the optimum.  (Counterexample, with min 1/2 (x-1)^2 written as
        K = 1, c = 0, f(s) = 1/2 (s-1)^2: at y = -1 every block domain passes and
        the finite-part expression reads +1/2 > 0 = p*.)"""
        finite = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(z))
        if g_block is not None:
            coupling = _block_in_domain(g_block, z)
            if _guarded_orthant:
                coupling &= jnp.all(z <= -coupling_margin)
                bits = jax.lax.bitcast_convert_type(y, jnp.uint32) & 0x7fffffff
                coupling &= ~jnp.any((bits != 0) & (bits < 0x00800000))
        else:
            # The conjugate of g = 0 is the indicator of {0}: no tolerance.
            coupling = jnp.all(z == 0)
        per_block = jnp.asarray(True)
        for blk, (a, b) in zip(blocks, slices):
            per_block &= _block_in_domain(blk, y[a:b])
        return finite, coupling, per_block

    def _dual_in_domain(y, z, coupling_margin=0.0):
        """EXACT, tolerance-free dual feasibility: y_i in dom f_i^* AND the
        coupling condition -K^T y - c in dom g^*.  Local conjugate-domain
        membership alone does NOT establish this."""
        finite, coupling, per_block = _dual_domain_parts(y, z, coupling_margin)
        return finite & coupling & per_block

    def _coupling_defect(z):
        """DIAGNOSTIC ONLY: ||z - P_{dom g^*}(z)||_inf, the measured distance by
        which the coupling condition is missed (for g = 0 this is ||z||_inf).

        It is reported so the size of the failure is visible; it NEVER relaxes
        the acceptance test above, and no bound is ever returned on the strength
        of a small defect."""
        if g_block is not None:
            residue = z - g_block.conj_domain_project(z)
        else:
            residue = z
        return jnp.max(jnp.abs(residue))

    def _dual_objective(y, z, l):
        """d(y) = <l, y> - g^*(-K^T y - c) - sum_i f_i^*(y_i), with NO finite-part
        replacement and NO conjugate tolerance.  Valid only where
        ``_dual_in_domain`` holds."""
        value = jnp.dot(l, y)
        if g_block is not None:
            value -= g_block.conj_value(z)
        for blk, (a, b) in zip(blocks, slices):
            value -= blk.conj_value(y[a:b])
        return value

    # Blocks whose value can be +inf at an approximately feasible point.
    # Excluding them gives the objective AT THE RETURNED POINT, which is what
    # Diamond and Boyd, SCS and OSQP evaluate their gap against.  It is not a
    # bound: an infeasible point's objective bounds nothing, and it is
    # reported under a name that says so.
    def _primal_objective_at_point(x, l, c):
        s = matvec(x) + l
        value = jnp.dot(c, x)
        if g_block is not None and not getattr(g_block, "is_indicator", False):
            value += g_block.value(x)
        for blk, (a, b) in zip(blocks, slices):
            if not getattr(blk, "is_indicator", False):
                value += blk.value(s[a:b])
        return value

    def _primal_objective(x, l, c):
        """p(x) = c^T x + g(x) + sum_i f_i(K_i x + l_i), extended-valued: a +inf
        from any block propagates instead of being replaced by zero."""
        s = matvec(x) + l
        value = jnp.dot(c, x)
        if g_block is not None:
            value += g_block.value(x)
        for blk, (a, b) in zip(blocks, slices):
            value += blk.value(s[a:b])
        return value

    def _strict_bound(y, z, l, primal, c=None):
        """(bound, available, status).  A number is returned ONLY when the dual is
        exactly in every conjugate domain AND satisfies the coupling condition,
        the value is finite, and (in an unconstrained family) it does not exceed
        the finite primal objective. Otherwise NaN, available = False, and a
        status code saying which of those failed.

        The last gate REFUSES; it never clips. In a constrained family the raw
        cost of an approximately feasible point is not an upper bound: a valid
        dual can exceed that cost. Criterion (7) checks the absolute discrepancy
        and feasibility separately; it must not discard such a dual.

        Epistemic limit: the arithmetic is reported by the evaluator, with
        tolerance-free domain gates on represented data. It is not a general
        interval enclosure or a proof. """
        value = _dual_objective(y, z, l)
        margin = _dense_margin(_dense_abs_adjoint(y), z)
        finite_dual, coupling, per_block = _dual_domain_parts(y, z, margin)
        if _marginal_orthant_guard:
            if c is None:
                raise ValueError("marginal coupling assessment requires represented costs")
            coupling &= _marginal_domain(y, c)
        if _fft_l1_guard:
            # The float32 test above stays; the float64 decision on the same
            # represented witness must agree before any bound is reported.
            if c is None:
                raise ValueError("the FFT/l1 coupling assessment needs the cost c")
            coupling = coupling & _fft_l1_assess(y, c)[0]
        finite_value = jnp.isfinite(value)
        # A toleranced indicator value is not a feasible primal upper bound.
        # In a constrained family a valid lower may exceed J(x) at an
        # approximately feasible x; criterion (7) then checks the discrepancy
        # in magnitude. Only an unconstrained finite primal objective supplies
        # the independent upper bound needed for this consistency rejection.
        constrained = (any(getattr(b, "is_indicator", False) for b in blocks)
                       or getattr(g_block, "is_indicator", False))
        below = jnp.asarray(True) if constrained else ~(value > primal)
        ok = finite_dual & coupling & per_block & finite_value & below
        status = jnp.where(
            ok, OBJECTIVE_BOUND_AVAILABLE,
            jnp.where(~finite_dual, OBJECTIVE_BOUND_DUAL_NONFINITE,
                      jnp.where(~coupling, OBJECTIVE_BOUND_COUPLING_VIOLATED,
                                jnp.where(~per_block, OBJECTIVE_BOUND_BLOCK_DOMAIN_VIOLATED,
                                          jnp.where(~finite_value,
                                                    OBJECTIVE_BOUND_VALUE_NONFINITE,
                                                    OBJECTIVE_BOUND_ABOVE_PRIMAL)))))
        return jnp.where(ok, value, jnp.nan), ok, jnp.asarray(status, jnp.int32)

    def _report_body(x, y, data):
        # Traced by the scalar evaluator ``report`` (jit of this body) AND
        # by the batched evaluator (jit of vmap of this body).  The guarded
        # assessment below opens a scoped float64 context; a transform
        # applied to an already traced jaxpr would re-bind that block
        # OUTSIDE the scope, where JAX truncates the f64 Dense/BinPack
        # products to float32 and the f64 sum then fails.  Batching the
        # body, not the jitted function, keeps every trace inside the scope.
        l, c = data
        checked = components(x, y, data)
        gap = jnp.asarray(jnp.nan, jnp.float32)
        if gap_ok:
            # Fenchel dual of  min c^T x + g(x) + sum_i f_i(K_i x + l_i):
            #   D(y) = <l, y> - g^*(-K^T y - c) - sum_i f_i^*(y_i).
            # Indicators contribute 0 (their +inf is feasibility, certified by the
            # residuals), so we take the finite part of every value / conjugate.
            # Only domain +infinity is omitted from the finite-part surrogate.
            # NaN and -infinity must remain invalid, never silently become zero.
            fin = lambda z: jnp.where(jnp.isposinf(z), 0.0, z)
            s = matvec(x) + l
            primal = jnp.dot(c, x)
            primal += fin(g_block.value(x)) if g_block is not None else 0.0
            for blk, (a, b) in zip(blocks, slices):
                primal += fin(blk.value(s[a:b]))
            dual = jnp.dot(l, y)
            if g_block is not None:
                dual -= fin(g_block.conj_value(-rmatvec(y) - c))
            for blk, (a, b) in zip(blocks, slices):
                dual -= fin(blk.conj_value(y[a:b]))
            gap = jnp.abs(primal - dual) / (1.0 + jnp.abs(primal) + jnp.abs(dual))
            # The surrogate dual value can sit ABOVE the primal value at the
            # same point; |primal - dual| hides that.  Say so, so that no reader
            # can mistake this number for a certified optimality gap.
            surrogate_above_primal = dual > primal
        # Block-oracle finite acceptance may include an internal tolerance.
        # Expose that weaker diagnostic separately from the conservative flag.
        feas_p = jnp.asarray(True)
        feas_d = jnp.asarray(True)
        if gap_ok:
            sfp = matvec(x) + l
            if g_block is None:
                # The conjugate of g=0 is indicator{0}, not the zero function.
                feas_d &= jnp.all(rmatvec(y) + c == 0)
            if g_block is not None:
                feas_p &= jnp.isfinite(g_block.value(x))
                feas_d &= jnp.isfinite(g_block.conj_value(-rmatvec(y) - c))
                # A finite toleranced indicator is not membership.  Where the
                # block declares the exact conjugate domain, require it: an
                # absolute slack must never admit an infeasible dual here.
                if getattr(g_block, "bound_capable", False):
                    feas_d &= g_block.conj_in_domain(-rmatvec(y) - c)
            for blk, (a, b) in zip(blocks, slices):
                feas_p &= jnp.isfinite(blk.value(sfp[a:b]))
                feas_d &= jnp.isfinite(blk.conj_value(y[a:b]))
                if getattr(blk, "bound_capable", False):
                    feas_d &= blk.conj_in_domain(y[a:b])
        # ---- strict objective lower bound (opt-in; never a surrogate) ----
        # These fields are ALWAYS present, whatever the family, so that no script
        # can read a missing key as an absent problem.  When the family or the
        # point cannot support a bound the value is NaN, ``_available`` is False
        # and ``objective_bound_status`` says why (see OBJECTIVE_BOUND_STATUS).
        nan = jnp.asarray(jnp.nan, jnp.float32)
        if bound_capable_all:
            primal_value = _primal_objective(x, l, c)
            z = _conj_argument(rmatvec(y), c)
            bound, bound_ok, bound_status = _strict_bound(y, z, l, primal_value, c)
            defect = _coupling_defect(z)
        else:
            primal_value, bound, defect = nan, nan, nan
            bound_ok = jnp.asarray(False)
            bound_status = jnp.asarray(OBJECTIVE_BOUND_NO_CONTRACT, jnp.int32)
        out = {**checked,
               # LEGACY, NOT A BOUND.  ``gap`` is the finite-part Fenchel
               # surrogate kept for existing scripts; its dual value can sit
               # ABOVE the primal value at the same point, in which case it is
               # not an optimality gap at all.  The alias below carries the same
               # number under a name that cannot be misread.
               "gap": gap,
               "gap_legacy_fenchel_surrogate": gap,
               "gap_domains_finite": (feas_p & feas_d) if gap_ok else jnp.asarray(False),
               "gap_exact": (feas_p & feas_d & exact_gap_contract)
               if gap_ok else jnp.asarray(False),
               "gap_surrogate_above_primal": (surrogate_above_primal if gap_ok
                                              else jnp.asarray(False)),
               # The bound, its availability, and the reason when it is absent.
               "objective_bound_capable": jnp.asarray(bound_capable_all),
               "objective_lower_bound": bound,
               "objective_lower_bound_available": bound_ok,
               "objective_bound_status": bound_status,
               "objective_primal_value": primal_value,
               # Diagnostic only: how far the coupling condition is missed.
               # It never relaxes acceptance.
               "dual_coupling_defect": defect}
        if contract_components is not None:
            # Reported BESIDE the strict residuals, never in place of them.
            out = {**out, **contract_components(x, y, data)}
        return out

    report = operand_jit(_report_body)

    def report_py(x, y, data):
        out = report(x, y, data)
        return {k: float(v) for k, v in out.items()}

    # ---- objective lower bound: the direction search.  This IS the default
    # acceptance criterion's evaluator; `accuracy="residual"` is what keeps
    # it off the solve path.
    # Nothing below runs unless the caller asks for a bound.  The solve loop, the
    # residuals, the returned endpoint and ``report`` are untouched by it.
    #
    # The default direction CORRECTS the conjugate-domain violation instead of
    # shrinking the whole dual for it:  z = -K^T y - c leaves dom g^* on a small
    # set; push back along K u with u the violation itself, then project each
    # block's own dual slice into dom f_i^*.  Two operator applications per
    # round, operator-only, no matrix, no re-solve.  The residual-scaled
    # candidate (the supplied dual shrunk along its own ray) is kept as the
    # fallback, and the report says which one certified.
    _bound_uncapable = ([type(b).__name__ for b in blocks
                         if not getattr(b, "bound_capable", False)]
                        + ([type(g_block).__name__]
                           if g_block is not None
                           and not getattr(g_block, "bound_capable", False) else []))

    def _corrected_dual(y, data, rounds):
        _l, c = data

        def _round(yc, project):
            z = _conj_argument(rmatvec(yc), c)
            if g_block is not None:
                violation = z - g_block.conj_domain_project(z)
            else:
                violation = z                     # dom g^* = {0}: the whole of z
            yc = yc + matvec(violation)
            pieces = [project(blk)(yc[a:b])
                      for blk, (a, b) in zip(blocks, slices)]
            return jnp.concatenate(pieces) if len(pieces) > 1 else pieces[0]

        def round_(_, yc):
            return _round(yc, lambda blk: blk.conj_domain_project)

        if witness_contraction == "once" and rounds >= 1:
            # Candidate generation with the EXACT block projections (no
            # interior adjustment), then the adjustment once, by the final
            # round, on the materialized candidate.  The caller validates the
            # whole result through the unchanged gates; nothing here awards.
            def exact_round(_, yc):
                return _round(yc, lambda blk: blk.conj_domain_project_exact)
            yc = jax.lax.fori_loop(0, rounds - 1, exact_round, y)
            return _round(yc, lambda blk: blk.conj_domain_project)

        return jax.lax.fori_loop(0, rounds, round_, y)

    def _fft_l1_settle(y_dir, c, at, value, ok, a):
        """Decide the ray search's chosen point on its represented witness.

        The float32 trials are proposals. Their winner ``a * y_dir`` is the
        point ``evaluate`` returns, so its coupling is decided by
        ``_fft_l1_assess``. When float32 accepted it and float64 places it
        outside by no more than the float32 adjoint allowance (a rounding-level
        disagreement), the witness is scaled into the domain,
        a' = a (1 - 64 eps32) / ratio, and a' * y_dir is evaluated by the same
        float32 gates and assessed in float64; that scaled point is what is
        returned and certified. Otherwise the candidate is refused. Returns
        (value, ok, a, engaged)."""
        ok64, ratio, allow = _fft_l1_assess(a * y_dir, c)
        engage = (ok & ~ok64 & jnp.isfinite(ratio) & (ratio > 1.0)
                  & (ratio <= 1.0 + allow))

        def scaled(_):
            a2 = jax.lax.optimization_barrier(
                a * jnp.float32(1.0 - 64.0 * _eps32) / ratio)
            v2, ok2 = at(a2)
            return v2, ok2 & _fft_l1_assess(a2 * y_dir, c)[0], a2

        def kept(_):
            return value, ok & ok64, a

        value, ok, a = jax.lax.cond(engage, scaled, kept, None)
        return value, ok, a, engage

    def _ray_search(y_dir, kty, data, primal, return_engaged=False):
        """Best checked bound along {a * y_dir : a >= 0}.

        K^T y_dir is supplied once and reused by trials. Guarded explicit
        structures additionally prepare one counted reference/absolute adjoint
        envelope per direction; trials do not repeat those products.
        """
        l, c = data
        eps = jnp.finfo(jnp.float32).eps
        absolute_product = _dense_abs_adjoint(y_dir)

        def at(a):
            yv = a * y_dir
            z = _conj_argument(a * kty, c)
            value = _dual_objective(yv, z, l)
            constrained = (any(getattr(b, "is_indicator", False) for b in blocks)
                           or getattr(g_block, "is_indicator", False))
            below = jnp.asarray(True) if constrained else ~(value > primal)
            margin = _dense_margin(jnp.abs(a)*absolute_product, z, materialized_ray=True)
            ok = _dual_in_domain(yv, z, margin) & jnp.isfinite(value) & below
            return jnp.where(ok, value, -jnp.inf), ok

        # Feasible top of the ray.  The blocks' declared radius is a hint (it is
        # exact only when the cost c is zero); the point itself is verified.
        hint = jnp.asarray(jnp.inf, jnp.float32)
        for blk, (a, b) in zip(blocks, slices):
            hint = jnp.minimum(hint, blk.conj_domain_radius(y_dir[a:b]))
        if g_block is not None:
            hint = jnp.minimum(hint, jnp.where(jnp.all(c == 0),
                                               g_block.conj_domain_radius(-kty), jnp.inf))
        unbounded = jnp.isinf(hint)
        top = jnp.where(unbounded, 1.0, hint * (1.0 - 64.0 * eps))

        def shrink(_, state):
            t, done = state
            _value, ok = at(t)
            return (jnp.where(done | ok, t, 0.5 * t), done | ok)

        top, reached = jax.lax.fori_loop(0, 20, shrink, (top, jnp.asarray(False)))
        value_top, _ok = at(top)

        def expand(_, state):
            t, v, growing = state
            nv, ok = at(2.0 * t)
            better = growing & ok & (nv > v)
            return (jnp.where(better, 2.0 * t, t), jnp.where(better, nv, v), better)

        top, value_top, still_growing = jax.lax.fori_loop(
            0, 16, expand, (top, value_top, unbounded & reached))

        # Golden-section maximisation of the concave d(a) on [0, top].  An
        # invalid trial scores -inf, which moves the bracket toward the valid
        # part of the ray.  The best VERIFIED trial is what is returned.
        phi = jnp.asarray(0.6180339887498949, jnp.float32)
        # Once an expansion fails, `top` is the LAST GOOD point, not the
        # right end of the search interval. In particular, after shrinking
        # an almost-feasible LP multiplier from 1 to 1/2, searching only
        # [0,1/2] permanently loses nearly half the available lower bound.
        # Include the first rejected/non-improving doubled trial. Invalid
        # points still score -inf, and the best verified point is retained.
        # Do not extend an expansion that used all sixteen growing steps.
        hi = jnp.where(unbounded & reached & ~still_growing, 2.0 * top, top)
        lo = jnp.asarray(0.0, jnp.float32)
        x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        f1, _ = at(x1)
        f2, _ = at(x2)
        best_a = jnp.where(value_top >= jnp.maximum(f1, f2), top,
                           jnp.where(f1 >= f2, x1, x2))
        best_v = jnp.maximum(value_top, jnp.maximum(f1, f2))

        def golden(_, state):
            lo, hi, x1, f1, x2, f2, best_a, best_v = state
            left = f1 >= f2                       # keep the left sub-interval
            lo = jnp.where(left, lo, x1)
            hi = jnp.where(left, x2, hi)
            keep_x = jnp.where(left, x1, x2)      # the interior point that survives
            keep_f = jnp.where(left, f1, f2)
            nx = jnp.where(left, hi - phi * (hi - lo), lo + phi * (hi - lo))
            nf, _ok = at(nx)
            x1 = jnp.where(left, nx, keep_x)
            f1 = jnp.where(left, nf, keep_f)
            x2 = jnp.where(left, keep_x, nx)
            f2 = jnp.where(left, keep_f, nf)
            better = nf > best_v
            return (lo, hi, x1, f1, x2, f2,
                    jnp.where(better, nx, best_a), jnp.where(better, nf, best_v))

        state = jax.lax.fori_loop(0, 28, golden, (lo, hi, x1, f1, x2, f2, best_a, best_v))
        best_a = state[6]
        # Returns (value, ok, a, witness, z), plus the FFT/l1 settle flag when
        # return_engaged; witness is the point a caller returns as the
        # certifying dual and z its coupling argument.
        if not _psd_resolved:
            # Re-evaluate at the chosen scaling so the REPORTED number is exactly a
            # verified evaluation, never a value carried through the search.
            value, ok = at(best_a)
            engaged = jnp.asarray(False)
            if _fft_l1_guard:
                value, ok, best_a, engaged = _fft_l1_settle(y_dir, c, at, value, ok, best_a)
        else:
            engaged = jnp.asarray(False)
        # Every family: the chosen point is re-validated AS
        # MATERIALIZED.  The float32 witness a * y_dir is formed once (the
        # barrier keeps it the represented float32 array rather than a fused
        # re-derivation), its coupling argument is formed from ITS OWN
        # adjoint (not the trials' cached a * K^T y_dir), and every block and
        # the coupling are decided by the final resolved-sign gates.  This is
        # the computation a recovery-off recheck of the returned witness
        # performs, so the two cannot disagree on formation; a point whose
        # sign is unresolved is refused, never certified.
        witness = jax.lax.optimization_barrier((best_a * y_dir).astype(jnp.float32))
        z = _conj_argument(rmatvec(witness), c)
        value, ok, _status = _strict_bound(witness, z, l, primal, c)
        out = (jnp.where(ok, value, jnp.nan), ok, best_a, witness, z)
        return out + (engaged,) if return_engaged else out

    # Indicator blocks, resolved once: their feasibility is measured
    # separately from the objective gap (see `evaluate`).
    _indicator_blocks = [(blk, sl) for blk, sl in zip(blocks, slices)
                         if getattr(blk, "is_indicator", False)]
    _indicator_primal = (g_block is not None
                         and getattr(g_block, "is_indicator", False))

    # ---- cone-aware dual recovery: a feasible anchor ---------------------
    # Eligible when the PRIMAL block g is a nonnegativity indicator, so that
    # dom g^* is the nonpositive orthant and the binding condition is the
    # coupling  z = -K^T y - c <= 0,  and when every dual block's own
    # conjugate domain is either the whole space (SumSquares, Quadratic,
    # Equality, Zero) or the nonpositive orthant (NonNeg).  A BOUNDED
    # conjugate domain, such as an l1 or l2 ball, would cap how far the anchor
    # may be scaled, so it is refused rather than silently clipped.
    #
    # With `base` = +1 on unrestricted dual blocks and -1 on NonNeg dual
    # blocks, the anchor is v = t*base and  z(v) = -t*h - c  where h = K^T base.
    # It is strictly feasible exactly when  t h + c > 0,  which needs h > 0;
    # t then clears c with a margin.  Both come from adjoint ACTIONS, so no
    # matrix is formed, read or factorized.
    _UNRESTRICTED = ("SumSquares", "Quadratic", "Equality", "Zero")
    _anchor_ok = False
    _anchor_base = None
    _anchor_cone_mask = None
    _anchor_project_input = False
    _anchor_probe_actions = 0
    _psd_anchor = (bound_capable_all and type(g_block) is PSDCone and
                   all(type(b).__name__ in ('SumSquares', 'Quadratic', 'Equality') for b in blocks))
    if bound_capable_all and isinstance(g_block, NonNeg):
        if all(isinstance(b, NonNeg) or type(b).__name__ in _UNRESTRICTED
               for b in blocks):
            _base = np.zeros(m, np.float32)
            _mask = np.zeros(m, bool)
            for b, (aa, bb) in zip(blocks, slices):
                if isinstance(b, NonNeg):
                    _base[aa:bb] = -1.0
                    _mask[aa:bb] = True
                else:
                    _base[aa:bb] = 1.0
            # Patch L: the anchors stay on the HOST; the evaluator uploads them for
            # its own calls only (operand_jit host operands), so they hold no device
            # memory while the drive runs.  Same values.
            _anchor_base = _base
            _anchor_cone_mask = _mask
            _anchor_ok = True
    if (not _anchor_ok and bound_capable_all and type(g_block) is NonNeg
            and any(type(b) is SumSquares for b in blocks)
            and any(type(b) is GroupL2 for b in blocks)
            and all(type(b) in (SumSquares, GroupL2) for b in blocks)):
        # A bounded TV domain cannot be scaled arbitrarily. Keep the anchor
        # zero on those blocks and use K_free * 1 on the unrestricted rows.
        # If K_free^T K_free 1 is strictly positive, it supplies an interior
        # orthant witness. Subsampled Fourier with DC has this property;
        # without DC it does not. No mask or matrix is inspected here.
        probe = matvec(jnp.ones(n, jnp.float32))
        _anchor_base = jnp.concatenate([
            probe[aa:bb] if type(b) is SumSquares else jnp.zeros(bb-aa, jnp.float32)
            for b, (aa, bb) in zip(blocks, slices)])
        # Patch L: computed on the device as before, then held on the HOST (same bytes).
        _anchor_base = np.asarray(jax.device_get(_anchor_base))
        del probe
        _anchor_cone_mask = np.zeros(m, bool)
        _anchor_ok = _anchor_project_input = True
        _anchor_probe_actions = 1
    if _psd_anchor:
        _anchor_base = np.ones(m, np.float32)        # patch L: host-held
        _anchor_cone_mask = np.zeros(m, bool)
        _anchor_ok = True

    # ---- equality-projected witness: Fourier/TV imaging WITHOUT DC ---------
    # nonneg(x) + least_squares(FFTMask x - b) + group_l2(Gradient3D x, 3).
    # When the mask excludes DC, F1 = D1 = 0 makes any nonnegative coupling
    # vector exactly zero, so no interior anchor exists; the named route in
    # ocs.solver.nodc_witness certifies an equality-eliminated float64 witness
    # instead. Eligibility is decided from exact host facts (operator types,
    # the integer index set, the grid shape); no matrix is formed or read.
    _nodc_route = None
    if (bound_capable_all and type(g_block) is NonNeg and len(blocks) == 2
            and family.cost_fn is None
            and sorted(type(b).__name__ for b in blocks) == ["GroupL2", "SumSquares"]):
        _i_data = 0 if type(blocks[0]) is SumSquares else 1
        _i_tv = 1 - _i_data
        if (type(ops[_i_data]) is FFTMask and type(ops[_i_tv]) is Gradient3D
                and int(blocks[_i_tv].g) == 3):
            _nodc_route = _nodc_witness.EqualityProjectedRoute(
                ops[_i_data], ops[_i_tv], float(blocks[_i_tv].lam),
                slices[_i_data], slices[_i_tv], n, m)

    # ---- primal restoration: a point that is ACTUALLY feasible -----------
    # Eligible when scaling toward the origin stays primal-feasible and the
    # origin satisfies every dual indicator block, both checked here rather
    # than assumed.  `l >= 0` is a DATA condition, so it is re-tested at
    # evaluation time; only the block structure is settled now.
    _restore_ok = bool(bound_capable_all and isinstance(g_block, NonNeg)
                       and _indicator_blocks
                       and all(isinstance(b, NonNeg) for b, _ in _indicator_blocks))

    def _restored_primal(x, data):
        """sigma in (0,1] with K(sigma x) + l feasible, and the value there."""
        l, c = data
        kx = matvec(x)
        ratios = jnp.ones_like(kx)
        offsets_nonneg = jnp.asarray(True)
        for _blk, (aa, bb) in _indicator_blocks:
            kxi, li = kx[aa:bb], l[aa:bb]
            offsets_nonneg = offsets_nonneg & jnp.all(li >= 0.0)
            binding = kxi < 0.0
            ratios = set_window(ratios, aa, bb,
                jnp.where(binding, li / jnp.where(binding, -kxi, 1.0), 1.0))
        sigma = jnp.minimum(1.0, jnp.min(ratios))
        # A strictly interior step, then the value is evaluated at the point
        # actually used -- never at the unscaled point.
        sigma = jnp.where(sigma < 1.0, sigma * (1.0 - 1e-6), sigma)
        sigma = jnp.clip(sigma, 0.0, 1.0)
        upper = _primal_objective(sigma * x, l, c)
        ok = offsets_nonneg & (sigma > 0.0) & jnp.isfinite(upper)
        return sigma, upper, ok

    def _anchor_dual(y, data):
        """(1-theta) y + theta v, with v strictly inside the dual conditions.

        Returns the candidate, whether a strictly feasible anchor exists for
        this data, and theta.  The candidate still goes through the ordinary
        exact gates; the flag only says whether there was an anchor to move
        toward."""
        l, c = data
        # Patch L: _anchor_base / _anchor_cone_mask are host arrays; jnp.asarray at each use
        # makes them traced constants exactly where the device arrays were (hoisted operands).
        if _guarded_orthant:
            # A proposal may replace subnormal multiplier coordinates by zero;
            # every affected domain here is an orthant or the whole space.
            # This changes only the proposed witness, then checks it normally.
            bits = jax.lax.bitcast_convert_type(y, jnp.uint32) & 0x7fffffff
            y = jnp.where((bits != 0) & (bits < 0x00800000), jnp.zeros_like(y), y)
        if _psd_anchor:
            # If H=sym(K^T 1) is positive definite, y+t*1 changes the coupling
            # matrix Z into Z-tH. Weyl's inequality supplies a sufficient t.
            # This is an operator-only proposal, rechecked at the actual
            # represented witness by the ordinary complete domain/value gate.
            k = g_block.k
            h = rmatvec(jnp.asarray(_anchor_base)).reshape(k, k)
            h = .5*(h+h.T)
            z = _conj_argument(rmatvec(y), c).reshape(k, k)
            z = .5*(z+z.T)
            eps = jnp.finfo(y.dtype).eps
            hmargin = 8*k*eps*jnp.max(jnp.sum(jnp.abs(h), axis=1))
            zmargin = 8*k*eps*jnp.max(jnp.sum(jnp.abs(z), axis=1))
            hfloor = jnp.linalg.eigvalsh(h)[0]-hmargin
            ceiling = jnp.linalg.eigvalsh(z)[-1]+zmargin
            good = jnp.isfinite(hfloor) & (hfloor>0)
            t = jnp.maximum(0., ceiling/jnp.where(good, hfloor, 1.))
            t = t*(1+8*eps)
            return y+t*jnp.asarray(_anchor_base), good, t
        if _anchor_project_input:
            # Start in every bounded domain before blending with its zero
            # anchor. The small interior margin addresses boundary rounding;
            # the complete represented witness is checked again below.
            y = jnp.concatenate([
                (1.0 - 1e-6) * b.conj_domain_project(y[aa:bb])
                if type(b) is GroupL2 else y[aa:bb]
                for b, (aa, bb) in zip(blocks, slices)])
        h = rmatvec(jnp.asarray(_anchor_base))
        h_ok = jnp.all(h > 0.0)
        safe_h = jnp.where(h > 0.0, h, 1.0)
        need = jnp.max(jnp.where(h > 0.0, -c / safe_h, 0.0))
        t = jnp.maximum(1.0, need * (1.0 + 1e-4) + 1e-4)
        v = t * jnp.asarray(_anchor_base)
        # z is AFFINE in y, so it blends linearly:
        #   z((1-th) y + th v) = (1-th) z(y) + th z(v).
        zy = _conj_argument(rmatvec(y), c)
        zv = _conj_argument(rmatvec(v), c)
        dz = zy - zv
        # Clear the assessment envelope, not merely the rounded-zero boundary.
        # The proposal still has to pass the complete fresh-witness check.
        need_zy = zy + _dense_margin(_dense_abs_adjoint(y), zy, materialized_ray=True)
        ratio_z = jnp.where((need_zy > 0.0) & (dz > 0.0),
                            need_zy / jnp.where(dz > 0.0, dz, 1.0), 0.0)
        dy = y - v
        ratio_y = jnp.where(jnp.asarray(_anchor_cone_mask) & (y > 0.0) & (dy > 0.0),
                            y / jnp.where(dy > 0.0, dy, 1.0), 0.0)
        theta = jnp.maximum(jnp.max(ratio_z), jnp.max(ratio_y))
        # The exact-arithmetic theta can land ON the boundary, where float32
        # rounding leaves a negative component.  Step inside, then let the
        # ordinary gates recheck the REPRESENTED witness.
        theta = jnp.where(theta > 0.0,
                          jnp.minimum(1.0, theta * (1.0 + 1e-5) + 1e-7), 0.0)
        return (1.0 - theta) * y + theta * v, h_ok, theta

    # ---- dual-bound route "shift": the recession shift (DUAL_ROUTES) -------
    # The segment candidate of the paper moves the iterate toward a fixed
    # strictly feasible v by the ratio test, and pays theta*(d(y) - d(v)).
    # With v = t*w for a recession direction w of every block domain the
    # cost tends, as t -> inf with theta*t -> s, to the cost of the SHIFT
    # y0 + s*w, which is what this candidate forms.  y0 is the block-wise
    # exact projection of the iterate (it removes the float32 Moreau residues
    # that put a NonNeg dual at +1e-7 or a PWL dual a few ulps outside its
    # interval).  w is +1 on unrestricted blocks, -1 on NonNeg blocks and 0 on
    # every other block, so y0 + s*w stays in every block domain for s >= 0.
    # With h = K^T w the coupling moves as z(y0 + s w) = z(y0) - s*h, and the
    # smallest s that clears z <= -margin wherever h > 0 is a closed-form
    # ratio.  The margin is a rounding allowance for the float32 coupling
    # (4 eps32 times the summed block-adjoint magnitudes and |c|; twice the
    # float64 envelope on the guarded Dense/BinPack families), so that the
    # point is interior under both the float32 gate and a float64 recheck.
    # NOTHING here awards: the shifted point is validated by the unchanged
    # _strict_bound, exactly as a supplied witness is.
    _shift_ok = False
    _shift_base = None
    if bound_capable_all and isinstance(g_block, NonNeg):
        _shift_ok = any(isinstance(b, NonNeg) or type(b).__name__ in _UNRESTRICTED
                        for b in blocks)
        if _shift_ok and 'shift' in dual_routes:
            _wb = np.zeros(m, np.float32)
            for b, (aa, bb) in zip(blocks, slices):
                if isinstance(b, NonNeg):
                    _wb[aa:bb] = -1.0
                elif type(b).__name__ in _UNRESTRICTED:
                    _wb[aa:bb] = 1.0
            # Same represented direction, held on the host and uploaded only
            # for calls that actually request this route.
            _shift_base = _wb

    def _shift_route(x, y, data):
        l, c = data
        primal = _primal_objective(x, l, c)
        if _guarded_orthant:
            bits = jax.lax.bitcast_convert_type(y, jnp.uint32) & 0x7fffffff
            y = jnp.where((bits != 0) & (bits < 0x00800000), jnp.zeros_like(y), y)
        pieces = [blk.conj_domain_project_exact(y[a:b]) for blk, (a, b) in zip(blocks, slices)]
        y0 = jnp.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        terms = [op.rmatvec(y0[a:b]) for op, (a, b) in zip(ops, slices)]
        kty0 = sum(terms)
        magnitude = sum(jnp.abs(t) for t in terms) + jnp.abs(c)
        z0 = _conj_argument(kty0, c)
        h = rmatvec(jnp.asarray(_shift_base))
        eps = float(np.finfo(np.float32).eps)
        guard = _dense_margin(_dense_abs_adjoint(y0), z0, materialized_ray=True)
        margin = 2.0 * guard + 4.0 * eps * (magnitude + jnp.abs(z0))
        need = z0 + margin
        reachable = jnp.all((h > 0.0) | (need <= 0.0))
        ratio = jnp.where((need > 0.0) & (h > 0.0), need / jnp.where(h > 0.0, h, 1.0), 0.0)
        s = jnp.max(ratio) * (1.0 + 16.0 * eps)
        ys = y0 + s * jnp.asarray(_shift_base)
        zs = _conj_argument(rmatvec(ys), c)
        bound, valid, status = _strict_bound(ys, zs, l, primal, c)
        valid = valid & reachable & jnp.isfinite(s)
        return {"bound": jnp.where(valid, bound, jnp.nan), "valid": valid, "dual": ys,
                "status": status, "s": s, "reachable": reachable,
                "defect": _coupling_defect(zs),
                "margin_max": jnp.max(margin), "need_max": jnp.max(need)}

    # ---- dual-bound route "packing": packing_assignment_dual (DUAL_ROUTES) --
    # The packing LP  max p^T X 1  s.t.  X 1 <= a,  Rmat X <= C,  X >= 0,  written
    # as  nonneg(x) + nonneg(-BinPack x + l) + c^T x.  With beta >= 0 pricing the
    # resource rows and alpha >= 0 the assignment rows (y = -(alpha; beta)), the
    # coupling of column (i, b) reads  alpha_i >= -c_ib - sum_r R_ri beta_rb.  For
    # ANY beta >= 0 the smallest admissible alpha is
    #     alpha_i = max{0, max_b (-c_ib - sum_r R_ri beta_rb)},
    # so (alpha, beta) is dual feasible by construction and l^T y is a lower bound.
    # beta is the iterate's resource dual projected onto >= 0 (exact in float32).
    # alpha is formed in scoped float64 with a rounding allowance and rounded UP
    # to float32; the materialized witness is validated by the unchanged
    # _strict_bound (guarded BinPack float64 envelope).  Eligibility is decided on
    # exact host facts: one NonNeg dual block, operator Weighted(BinPack, -1).
    _packing_ok = False
    _packing_op = None
    if (bound_capable_all and isinstance(g_block, NonNeg) and len(blocks) == 1
            and isinstance(blocks[0], NonNeg) and type(ops[0]) is Weighted
            and type(ops[0].base) is BinPack and tuple(slices[0]) == (0, m)):
        _pw = np.asarray(ops[0].w)
        if _pw.size and np.all(_pw == -1.0):
            _packing_ok = True
            _packing_op = ops[0].base

    def _packing_route(x, y, data):
        l, c = data
        primal = _primal_objective(x, l, c)
        I_, R_, B_ = _packing_op.I, _packing_op.R, _packing_op.B
        bits = jax.lax.bitcast_convert_type(y, jnp.uint32) & 0x7fffffff
        y = jnp.where((bits != 0) & (bits < 0x00800000), jnp.zeros_like(y), y)
        beta32 = jnp.maximum(-y[I_:], 0.0)
        eps = float(np.finfo(np.float32).eps)
        tiny = float(np.finfo(np.float32).tiny)
        with jax.enable_x64():
            f64 = jnp.float64
            beta = beta32.astype(f64).reshape(R_, B_)
            A = jnp.matmul(_packing_op.Rmat.astype(f64).T, beta,
                           precision=jax.lax.Precision.HIGHEST)          # (I, B)
            cc = c.astype(f64).reshape(I_, B_)
            need = -cc - A
            allowance = 4.0 * eps * (jnp.abs(cc) + A + jnp.abs(need)) + 4.0 * tiny
            alpha = jnp.maximum(0.0, jnp.max(need + allowance, axis=1))
            a32 = alpha.astype(jnp.float32)
            back = jax.lax.optimization_barrier(a32).astype(f64)
            a32 = jnp.where(back < alpha, jnp.nextafter(a32, jnp.asarray(jnp.inf, jnp.float32)), a32)
        yp = jnp.concatenate([-a32, -beta32])
        zs = _conj_argument(rmatvec(yp), c)
        bound, valid, status = _strict_bound(yp, zs, l, primal, c)
        return {"bound": jnp.where(valid, bound, jnp.nan), "valid": valid, "dual": yp,
                "status": status, "s": jnp.max(a32), "reachable": jnp.asarray(True),
                "defect": _coupling_defect(zs), "margin_max": jnp.max(allowance.astype(jnp.float32)),
                "need_max": jnp.max(need.astype(jnp.float32))}

    # ---- dual-bound route "ridge": ridge_nonneg_compound (DUAL_ROUTES) ----
    # g(F) = (rho/2)||F||^2 + I_+(F) read as one compound block: with the ridge
    # compiled as its own dual block SumSquares(Diagonal(d) x), its multiplier is
    # reconstructed from the others.  With Q = sum_{i != s} K_i^T y_i + c (the
    # other blocks projected into their own domains), the coupling of column j
    # is  d_j y_s,j >= -Q_j,  and the best y_s for d(y) = l_s^T y_s - |y_s|^2/2
    # under it is  y_s = max(l_s, (-Q + allowance)/d)  -- [-Q]_+/d for l_s = 0,
    # i.e. V = [-Q]_+ in the paper's notation.  The only remaining domain
    # requirement is each other block's own (the PWL dual in [s_min, s_max],
    # by the exact projection).  Formed in scoped float64, rounded up to
    # float32, validated by the unchanged _strict_bound.  Eligibility: g NonNeg,
    # exactly one SumSquares block whose operator is Diagonal(d > 0) on all n
    # columns (exact host facts; nothing is read from an opaque operator).
    _ridge_ok = False
    _ridge_index = None
    if bound_capable_all and isinstance(g_block, NonNeg) and len(blocks) >= 2:
        _cand = [i for i, (b, op) in enumerate(zip(blocks, ops))
                 if type(b) is SumSquares and type(op) is Diagonal and tuple(op.shape) == (n, n)]
        if len(_cand) == 1 and np.all(np.asarray(ops[_cand[0]].d) > 0):
            _ridge_ok = True
            _ridge_index = _cand[0]

    def _ridge_route(x, y, data):
        l, c = data
        primal = _primal_objective(x, l, c)
        j = _ridge_index
        a0, b0 = slices[j]
        pieces = [y[a:b] if i == j else blk.conj_domain_project_exact(y[a:b])
                  for i, (blk, (a, b)) in enumerate(zip(blocks, slices))]
        yq = jnp.concatenate(pieces)
        terms = [op.rmatvec(yq[a:b]) for i, (op, (a, b)) in enumerate(zip(ops, slices)) if i != j]
        eps = float(np.finfo(np.float32).eps)
        tiny = float(np.finfo(np.float32).tiny)
        with jax.enable_x64():
            f64 = jnp.float64
            Q = sum(t.astype(f64) for t in terms) + c.astype(f64)
            mag = sum(jnp.abs(t.astype(f64)) for t in terms) + jnp.abs(c.astype(f64))
            allowance = 3.0 * eps * (mag + jnp.abs(Q)) + 4.0 * tiny
            dd = ops[j].d.astype(f64)
            lower = (-Q + allowance) / dd
            ys = jnp.maximum(l[a0:b0].astype(f64), lower)
            ys32 = ys.astype(jnp.float32)
            back = jax.lax.optimization_barrier(ys32).astype(f64)
            ys32 = jnp.where(back < ys, jnp.nextafter(ys32, jnp.asarray(jnp.inf, jnp.float32)), ys32)
        yr = jnp.concatenate([ys32 if i == j else pieces[i] for i in range(len(pieces))])
        zs = _conj_argument(rmatvec(yr), c)
        bound, valid, status = _strict_bound(yr, zs, l, primal, c)
        return {"bound": jnp.where(valid, bound, jnp.nan), "valid": valid, "dual": yr,
                "status": status, "s": jnp.max(ys32), "reachable": jnp.asarray(True),
                "defect": _coupling_defect(zs), "margin_max": jnp.max(allowance.astype(jnp.float32)),
                "need_max": jnp.max((-Q).astype(jnp.float32))}

    # ---- the enabled routes: compile(dual_routes=...) -------------------------
    # A route is built only when it is both requested and eligible.  Each is
    # jitted on its own, called only from the scalar controller, and charged
    # DUAL_ROUTE_OPERATOR_APPLICATIONS[name] per consultation.
    _route_table = {"shift": (_shift_ok, _shift_route),
                    "packing": (_packing_ok, _packing_route),
                    "ridge": (_ridge_ok, _ridge_route)}
    _dual_routes_jit = {name: (operand_jit(fn), DUAL_ROUTE_OPERATOR_APPLICATIONS[name])
                        for name, (ok, fn) in _route_table.items()
                        if ok and name in dual_routes}

    @functools.lru_cache(maxsize=None)
    def _bound_evaluator(rounds, with_dual=False, recover=True):
        @operand_jit
        def evaluate(x, y, data):
            l, c = data
            primal = _primal_objective(x, l, c)
            # ---- relative primal feasibility, separately from the gap -----
            # An indicator's prox IS the projection onto its domain, so the
            # violation is the distance the projection moves the point.  The
            # scale is the block's own offset under `relative_ratio`, the same
            # pure relative convention used everywhere else in OCS: no
            # additive unit floor, so a right-hand side of 1e-6 does not make
            # a 1e-6 violation look small.
            infeasibility = jnp.asarray(0.0, jnp.float32)
            if _indicator_blocks or _indicator_primal:
                s_feas = matvec(x) + l
                for blk, (a, bb) in _indicator_blocks:
                    si = s_feas[a:bb]
                    moved = blk.prox(si, jnp.ones_like(si)) - si
                    infeasibility = jnp.maximum(
                        infeasibility,
                        relative_ratio(stable_norm(moved), stable_norm(l[a:bb])))
                if _indicator_primal:
                    moved = g_block.prox(x, jnp.ones_like(x)) - x
                    infeasibility = jnp.maximum(
                        infeasibility,
                        relative_ratio(stable_norm(moved), stable_norm(x)))
            kty_supplied = rmatvec(y)
            # What the SUPPLIED dual establishes on its own, and why not, before
            # any repair direction is tried.  A caller must be able to see that
            # the dual OCS returned was refused, and on which condition.
            _supplied, supplied_ok, supplied_status = _strict_bound(
                y, _conj_argument(kty_supplied, c), l, primal, c)
            supplied_defect = _coupling_defect(_conj_argument(kty_supplied, c))
            scaled_engaged = corrected_engaged = jnp.asarray(False)
            if recover:
                scaled, scaled_ok, scaled_a, scaled_w, scaled_z, scaled_engaged = _ray_search(
                    y, kty_supplied, data, primal, return_engaged=True)
                corrected = _corrected_dual(y, data, rounds)
                kty = rmatvec(corrected)
                best, best_ok, best_a, best_w, best_z, corrected_engaged = _ray_search(
                    corrected, kty, data, primal, return_engaged=True)
            else:
                # Not staged into the graph at all, so the counters below are
                # a fact about the traced program rather than a claim.
                scaled = jnp.asarray(jnp.nan, jnp.float32)
                scaled_ok = jnp.asarray(False)
                scaled_a = jnp.asarray(jnp.nan, jnp.float32)
                kty = kty_supplied
                best = jnp.asarray(jnp.nan, jnp.float32)
                best_ok = jnp.asarray(False)
                best_a = jnp.asarray(jnp.nan, jnp.float32)
                scaled_w = best_w = jnp.full_like(y, jnp.nan)
                scaled_z = best_z = jnp.full_like(kty_supplied, jnp.nan)

            # ---- candidate selection: max over VALID candidates ----------
            # The supplied witness is a candidate in its own right.  It was
            # previously validated and then dropped, so a valid certificate
            # could be discarded when every recovery route failed.  Each
            # candidate replaces the incumbent only when it is valid AND
            # strictly better, and its defect and source code move with it.
            # 0 = none, 1 = violation-corrected, 2 = residual-scaled,
            # 3 = feasible anchor, 4 = the supplied witness unmodified.
            bound = jnp.where(supplied_ok, _supplied, jnp.nan)
            if with_dual:
                chosen = y
            available = supplied_ok
            source = jnp.where(supplied_ok, 4, 0)
            certifying_defect = jnp.where(supplied_ok, supplied_defect, jnp.nan)

            take_scaled = scaled_ok & (~available | (scaled > bound))
            if with_dual and recover:
                chosen = jnp.where(take_scaled, scaled_w, chosen)
            bound = jnp.where(take_scaled, scaled, bound)
            source = jnp.where(take_scaled, 2, source)
            certifying_defect = jnp.where(
                take_scaled,
                _coupling_defect(scaled_z),
                certifying_defect)
            available = available | scaled_ok

            use_corrected = best_ok & (~available | (best > bound))
            if with_dual and recover:
                chosen = jnp.where(use_corrected, best_w, chosen)
            bound = jnp.where(use_corrected, best, bound)
            source = jnp.where(use_corrected, 1, source)
            certifying_defect = jnp.where(
                use_corrected, _coupling_defect(best_z),
                certifying_defect)
            available = available | best_ok
            # ---- third candidate: the feasible anchor (cone-aware) --------
            # NOT ray-searched: with an affine coupling a positive rescaling
            # would break the exact equality the construction was built to
            # satisfy.  Evaluated as constructed, and accepted only by the
            # same gates as every other candidate.
            anchor_bound = jnp.asarray(jnp.nan, jnp.float32)
            anchor_ok = jnp.asarray(False)
            anchor_theta = jnp.asarray(jnp.nan, jnp.float32)
            if _anchor_ok and recover:
                anchor_y, anchor_h_ok, anchor_theta = _anchor_dual(y, data)
                kty_anchor = rmatvec(anchor_y)
                z_anchor = _conj_argument(kty_anchor, c)
                anchor_bound, anchor_valid, _astat = _strict_bound(
                    anchor_y, z_anchor, l, primal, c)
                anchor_ok = anchor_valid & anchor_h_ok
                # With a HOMOGENEOUS coupling (c = 0) every gamma > 0 keeps
                # z = -K^T y <= 0, so the anchor sits on a feasible cone ray
                # and the bound can only improve by optimizing along it.  With
                # c != 0 the coupling is affine and rescaling would break the
                # exact equality, so the ray result is used only where the
                # data say it is admissible -- and even then only if it
                # revalidates through the same gates.
                homogeneous = jnp.all(c == 0)
                ray_bound, ray_ok, _ray_a, _ray_w, _ray_z = _ray_search(
                    anchor_y, kty_anchor, data, primal)
                take_ray = homogeneous & anchor_h_ok & ray_ok & (
                    ~anchor_ok | (ray_bound > anchor_bound))
                anchor_bound = jnp.where(take_ray, ray_bound, anchor_bound)
                anchor_ok = anchor_ok | take_ray
                # Same rule as every other candidate: valid AND strictly
                # better, or the incumbent stands.
                use_anchor = anchor_ok & (~available | (anchor_bound > bound))
                if with_dual:
                    anchor_chosen = jnp.where(take_ray, _ray_w, anchor_y)
                    chosen = jnp.where(use_anchor, anchor_chosen, chosen)
                bound = jnp.where(use_anchor, anchor_bound, bound)
                certifying_defect = jnp.where(
                    use_anchor, _coupling_defect(z_anchor), certifying_defect)
                source = jnp.where(use_anchor, 3, source)
                available = available | anchor_ok
            # For a genuinely zero-cost feasibility problem, zero is a valid
            # global dual regardless of the supplied multipliers. It is a
            # recovery candidate only, not a change to recover=False.
            if feasibility_family and recover:
                use_zero = jnp.all(c == 0) & (~available | (bound < 0))
                if with_dual:
                    chosen = jnp.where(use_zero, jnp.zeros_like(y), chosen)
                bound = jnp.where(use_zero, 0.0, bound)
                source = jnp.where(use_zero, 5, source)
                certifying_defect = jnp.where(use_zero, 0.0, certifying_defect)
                available = available | use_zero
            # An objective LOWER bound needs no primal point.  A finite GAP needs
            # a valid primal UPPER bound too, and an infeasible x does not supply
            # one: its extended-valued objective is +inf, and discarding that
            # infinity would be exactly the surrogate's error.
            # ---- a finite UPPER bound, from a point that is really feasible
            # The returned point is unchanged; when its own extended-value
            # objective is +inf a restored feasible point supplies the upper
            # bound instead, and the scale used is reported.
            at_point = _primal_objective_at_point(x, l, c)
            restore_sigma = jnp.asarray(jnp.nan, jnp.float32)
            upper = primal
            restored = jnp.asarray(False)
            if _restore_ok and recover:
                restore_sigma, restore_upper, restore_valid = _restored_primal(x, data)
                use_restore = restore_valid & (~jnp.isfinite(primal))
                upper = jnp.where(use_restore, restore_upper, primal)
                restored = use_restore
            primal_finite = jnp.isfinite(upper)
            # The denominator convention is UNCHANGED by this patch: it is the
            # same |value| the scalar path already used, now taken at the
            # upper bound the gap is actually against.  (Note the batch path
            # uses max(1, |value|) and its comment claims the scalar path does
            # too; they genuinely differ.  That discrepancy predates this
            # change and is reported rather than silently resolved here,
            # because altering a denominator alters acceptance.)
            gap = jnp.where(primal_finite & (jnp.abs(upper) > 0),
                            (upper - bound) / jnp.abs(upper), jnp.nan)
            gap_certified = available & primal_finite
            out = {"objective_lower_bound": jnp.where(available, bound, jnp.nan),
                    "objective_lower_bound_available": available,
                    "objective_primal_value": primal,
                    # NB: this says the CERTIFIED UPPER BOUND is finite, which
                    # is what the gap needs; the returned point's own
                    # extended-value objective may still be +inf and is
                    # reported separately above.
                    "objective_primal_upper_bound_finite": primal_finite,
                    "objective_primal_value_finite": jnp.isfinite(primal),
                    # The upper bound the gap was actually taken against, and
                    # whether it came from a RESTORED point rather than the
                    # returned one.  Equal to the returned point's value
                    # whenever that value is finite.
                    "objective_primal_upper_bound": upper,
                    "primal_upper_from_restored_point": restored,
                    "primal_restoration_scale": restore_sigma,
                    # The objective AT THE RETURNED POINT, with indicator
                    # terms excluded, and the gap against it.  This is the
                    # quantity Diamond and Boyd's protocol tests, beside the
                    # primal residual.  NOT A BOUND: valid only when read
                    # together with `relative_primal_infeasibility`.
                    "objective_at_returned_point": at_point,
                    "gap_at_returned_point": jnp.where(
                        available & jnp.isfinite(at_point),
                        (at_point - bound) / jnp.where(
                            jnp.maximum(jnp.abs(at_point), jnp.abs(bound)) > 0.0,
                            jnp.maximum(jnp.abs(at_point), jnp.abs(bound)), 1.0),
                        jnp.nan),
                    "relative_gap": jnp.where(gap_certified, gap, jnp.nan),
                    "relative_gap_certified": gap_certified,
                    # "AX really close to B", relative and exact.  Reported
                    # beside the gap, and required alongside it for acceptance.
                    "relative_primal_infeasibility": infeasibility,
                    "source_code": source,
                    "supplied_dual_available": supplied_ok,
                    "supplied_dual_status_code": supplied_status,
                    "supplied_dual_coupling_defect": supplied_defect,
                    "certifying_dual_coupling_defect":
                        jnp.where(available, certifying_defect, jnp.nan),
                    "corrected_bound": jnp.where(best_ok, best, jnp.nan),
                    "corrected_available": best_ok,
                    "corrected_scaling": best_a,
                    "scaled_bound": jnp.where(scaled_ok, scaled, jnp.nan),
                    "scaled_available": scaled_ok,
                    "scaled_scaling": scaled_a,
                    "anchor_bound": anchor_bound,
                    "anchor_available": anchor_ok,
                    "anchor_theta": (jnp.asarray(jnp.nan, jnp.float32) if _psd_anchor else anchor_theta),
                    "anchor_psd_shift": (anchor_theta if _psd_anchor else jnp.asarray(jnp.nan, jnp.float32)),
                    # Staged-route counters.  Zero means the route was never
                    # traced, not that its result was discarded.
                    "recovery_enabled": jnp.asarray(bool(recover)),
                    "n_corrected_dual_rounds": jnp.asarray(
                        float(rounds if recover else 0)),
                    "n_ray_searches": jnp.asarray(float(2 if recover else 0)),
                    "n_dense_absolute_adjoints": jnp.asarray(float(
                        (1 + (4 + 4*int(_anchor_ok) if recover else 0))
                        if _dense_orthant_guard else 0)),
                    "n_structured_absolute_adjoints": jnp.asarray(float(
                        (1 + (4 + 4*int(_anchor_ok) if recover else 0))
                        if _allocation_orthant_guard else 0)),
                    "assessment_uses_float64": jnp.asarray(_guarded_orthant or _fft_l1_guard or _marginal_orthant_guard),
                    "n_anchor_constructions": jnp.asarray(
                        float(1 if (_anchor_ok and recover) else 0)),
                    "n_primal_restorations": jnp.asarray(
                        float(1 if (_restore_ok and recover) else 0))}
            if _fft_l1_guard:
                # The supplied witness's float64 coupling ratio and the a priori
                # float32 allowance of its adjoint (both over lam, rounded up),
                # and whether a ray search scaled its winner into the domain.
                _ok64, sup_ratio, sup_allow = _fft_l1_assess(y, c)
                out["fft_l1_supplied_ratio_float64"] = sup_ratio
                out["fft_l1_supplied_allowance_float32"] = sup_allow
                out["fft_l1_scaled_engaged"] = scaled_engaged
                out["fft_l1_corrected_engaged"] = corrected_engaged
                out["n_fft_float64_adjoints"] = jnp.asarray(float(
                    2 + (6 if recover else 0) + (3 if (_anchor_ok and recover) else 0)))
            out['n_dense_reference_adjoints'] = out['n_dense_absolute_adjoints']
            out['n_dense_rounded_rechecks'] = out['n_dense_absolute_adjoints']
            out['n_structured_reference_adjoints'] = out['n_structured_absolute_adjoints']
            out['n_structured_rounded_rechecks'] = out['n_structured_absolute_adjoints']
            if not with_dual:
                return out
            # The dual point the reported bound was evaluated at, so an
            # independent checker can re-derive the same number.
            return out, jnp.where(available, chosen, jnp.nan)
        return evaluate

    def _assessment_work(rounds, recover, equality_projected=0):
        # One definition is used both to reserve work before a refinement and
        # to report it after evaluation. A previous observation can be cheaper
        # than a later assessment, so it cannot replace this structural bound.
        guarded = int(bool(_dense_orthant_guard)) + int(bool(_allocation_orthant_guard))
        guard_actions = 3 * guarded * (1 + (4 + 4*int(_anchor_ok) if recover else 0))
        if _fft_l1_guard:
            guard_actions += 2 + (6 if recover else 0) + (3 if (_anchor_ok and recover) else 0)
        return {
            "corrected_dual": 2 * int(rounds) if recover else 0,
            "base_objective_and_duals": 4 if recover else 3,
            "feasibility_slack": int(bool(_indicator_blocks or _indicator_primal)),
            "primal_restoration": 2 if _restore_ok and recover else 0,
            "feasible_anchor": 4 if _anchor_ok and recover else 0,
            "equality_projected": int(equality_projected),
            "ray_search": 0,
            "ray_search_materialized_revalidation": (2 + int(_anchor_ok)) if recover else 0,
            "arithmetic_guard_actions": guard_actions}

    def _assessment_work_bound(rounds=32, recover=True):
        if not bound_capable_all:
            return 0
        extra = 0
        if (_nodc_route is not None and _nodc_route.eligible
                and _nodc_witness.settings()["enabled"]):
            p = _nodc_route.params
            admm = int(p["admm_iters"]) if recover else 0
            cg = (admm + 1) * int(p["cg_maxit"]) if recover else 0
            extra = 4*cg + 4*admm + 8 + (4 if recover else 2)
        return sum(_assessment_work(rounds, bool(recover), extra).values())

    def objective_bound_fn(x, y, data, rounds=32, return_dual=False,
                           recover=True):
        """Strict objective lower bound from a supplied dual point.

        Assessment arithmetic and logical action bounds are reported separately.
        This is not a general interval enclosure or proof and never runs the
        optimizer. Explicit guarded structures use scoped float64 products.
        """
        rounds = int(rounds)
        if rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if not bound_capable_all:
            return {"objective_lower_bound": float("nan"),
                    "objective_lower_bound_available": False,
                    "objective_bound_capable": False,
                    "objective_bound_status_code": OBJECTIVE_BOUND_NO_CONTRACT,
                    "unavailable_reason":
                        "no exact conjugate-domain contract for: "
                        + ", ".join(sorted(set(_bound_uncapable)) or ["(missing value/conjugate oracle)"]),
                    "epistemic_limit": OBJECTIVE_BOUND_EPISTEMIC_LIMIT,
                    "operator_applications": 0, "rounds": rounds, "source": "unavailable"}
        raw = _bound_evaluator(rounds, bool(return_dual), bool(recover))(x, y, data)
        certifying_dual = None
        if return_dual:
            raw, certifying_dual = raw
        out = {k: (bool(v) if isinstance(v, (bool, np.bool_))
                   or getattr(v, 'dtype', None) == jnp.bool_ else float(v))
               for k, v in raw.items()}
        if return_dual:
            out["certifying_dual"] = certifying_dual
        code = int(out.pop("source_code"))
        # ---- equality-projected candidate (no-DC imaging), Python level ----
        # Same rule as every traced candidate: valid AND strictly better, or
        # the incumbent stands.  With recover=False only the SUPPLIED pair is
        # certified under the route's declared semantics; nothing is moved.
        route_report = None
        if (_nodc_route is not None and _nodc_route.eligible
                and _nodc_witness.settings()["enabled"]):
            l_data, c_data = data
            upper = out.get("objective_primal_upper_bound", out["objective_primal_value"])
            # patch D: with the memory-bounded engine, the traced candidate's
            # dual waits on the host while the route runs; restored below,
            # same bytes, when the route does not replace it.
            _parked = (return_dual and isinstance(certifying_dual, jax.Array)
                       and _nodc_route.params.get("engine") == "tiled")
            if _parked:
                certifying_dual = np.asarray(jax.device_get(certifying_dual))
                out["certifying_dual"] = None
            route_report = _nodc_route.run(x, y, l_data, c_data, recover=bool(recover),
                                           upper=upper if math.isfinite(upper) else None)
            take = (route_report["valid"]
                    and (not out["objective_lower_bound_available"]
                         or route_report["d_lower"] > out["objective_lower_bound"]))
            if take:
                lower = float(route_report["d_lower"])
                out["objective_lower_bound"] = lower
                out["objective_lower_bound_available"] = True
                out["certifying_dual_coupling_defect"] = float(route_report["delta"])
                out["assessment_uses_float64"] = True
                finite = bool(out.get("objective_primal_upper_bound_finite", math.isfinite(upper)))
                out["relative_gap"] = ((upper - lower) / abs(upper)
                                       if finite and abs(upper) > 0 else float("nan"))
                out["relative_gap_certified"] = finite
                at_point = out.get("objective_at_returned_point", upper)
                scale = max(abs(at_point), abs(lower))
                out["gap_at_returned_point"] = ((at_point - lower) / (scale if scale > 0 else 1.0)
                                                if math.isfinite(at_point) else float("nan"))
                if return_dual:
                    out["certifying_dual"] = route_report["dual_float64"]
                out["witness_representation"] = _nodc_witness.REPRESENTATION
                out["certifying_dual_dtype"] = "float64"
                code = 7
            elif _parked:
                out["certifying_dual"] = jnp.asarray(certifying_dual)
            out["equality_projected"] = {k: v for k, v in route_report.items() if k != "dual_float64"}
        out["source"] = {0: "none", 1: "violation_corrected",
                         2: "residual_scaled", 3: "feasible_anchor",
                         4: "supplied_witness", 5: "zero_feasibility",
                         7: _nodc_witness.SOURCE}[code]
        out["bound_kind"] = "strict_dual" if out["objective_lower_bound_available"] else "none"
        out["arithmetic"] = "mixed" if (_guarded_orthant or _fft_l1_guard or _marginal_orthant_guard or code == 7) else "float32"
        out["coupling_arithmetic_guard"] = (
            "marginal_float64_adjoint_envelope" if _marginal_orthant_guard else
            "dense_float64_adjoint_envelope" if _dense_orthant_guard else
            "binpack_float64_adjoint_envelope" if _allocation_orthant_guard else
            "fftmask_l1_float64_adjoint_envelope" if _fft_l1_guard else
            "operator_callback_arithmetic")
        out["dense_assessment_matrix_dtype"] = "float64" if _dense_orthant_guard else None
        out["structured_assessment_dtype"] = "float64" if _allocation_orthant_guard else None
        out["objective_bound_capable"] = True
        # Counted at the call sites of the traced path.  The old fixed
        # `2*rounds + 3` predated the anchor, restoration and returned-point
        # routes and reported their operator work as free; see this module's
        # recovery notes.  Identical subexpressions (several routes apply
        # `matvec(x)`) may be fused by XLA, so this BOUNDS the applications
        # executed rather than measuring them, and is named accordingly.
        _work = _assessment_work(rounds, bool(recover),
            int(route_report.get("operator_applications", 0)) if route_report is not None else 0)
        out["operator_applications"] = sum(_work.values())
        out["operator_applications_is_upper_bound"] = True
        out["operator_applications_breakdown"] = _work
        out["rounds"] = rounds
        out["witness_contraction"] = witness_contraction
        # Why the dual OCS actually returned did or did not establish a bound on
        # its own, stated separately from whatever a repair direction achieved.
        supplied_code = int(out.pop("supplied_dual_status_code"))
        out["supplied_dual_status"] = OBJECTIVE_BOUND_STATUS[supplied_code]
        out["supplied_dual_status_code"] = supplied_code
        out["epistemic_limit"] = OBJECTIVE_BOUND_EPISTEMIC_LIMIT
        if out["objective_lower_bound_available"]:
            out["objective_bound_status_code"] = OBJECTIVE_BOUND_AVAILABLE
            if not out["relative_gap_certified"]:
                out["gap_unavailable_reason"] = (
                    "the bound is valid, but the primal value at the supplied x is "
                    "not finite, so there is no valid primal upper bound and no "
                    "finite optimality gap follows")
        else:
            out["objective_bound_status_code"] = supplied_code
            out["unavailable_reason"] = (
                "no candidate direction passed the dual domain and finite-value "
                "checks (including unconstrained primal consistency where applicable); the supplied dual failed "
                "because: " + OBJECTIVE_BOUND_STATUS[supplied_code])
        return out

    # patch D: the objective-gap drive holds its inactive state on the host
    # during checks when this family's no-DC route runs the tiled engine.
    objective_bound_fn.operator_applications_bound = _assessment_work_bound
    objective_bound_fn.phase_separated = bool(
        _nodc_route is not None and _nodc_route.eligible
        and _nodc_route.params.get("engine") == "tiled")

    def objective_bound_many_fn(x, y, data, rounds=32, return_dual=False):
        """Per-row objective bound for a batch.

        The SAME evaluator as the scalar path, under ``jax.vmap``: the batch
        cannot certify anything the scalar path would refuse, because it is
        not a second implementation.
        """
        if not bound_capable_all:
            return None
        if _guarded_orthant or _fft_l1_guard or _psd_resolved or _marginal_orthant_guard:
            # The batching transform must preserve the scoped f64 literals
            # already present in this evaluator's jaxpr. Otherwise JAX can
            # canonicalize an addend to f32 while batching its f64 sum.
            with jax.enable_x64():
                raw = jax.vmap(_bound_evaluator(int(rounds), bool(return_dual)))(x, y, data)
                # Leave the scoped context with the declared f32 report/dual
                # interface. Otherwise later batched selection can implicitly
                # downcast weak f64 report constants outside this context.
                return jax.tree_util.tree_map(
                    lambda v: (v.astype(jnp.float32) if jnp.issubdtype(v.dtype, jnp.floating)
                               else v.astype(jnp.int32) if jnp.issubdtype(v.dtype, jnp.integer) else v),
                    raw)
        return jax.vmap(_bound_evaluator(int(rounds), bool(return_dual)))(x, y, data)

    # ---- solution refinement: matrix-free Gauss-Newton on the fixed-point residual
    def refine_residual_fn(x, y, data):
        # The fixed-point residual map R(x, y) of refine_fn, exposed unchanged
        # so that a safeguarded wrapper (ocs.solver.refine_safeguarded) can
        # line-search and assess its own steps.  Two operator applications.
        rp = x - prox_primal(x - tau * iteration_rmatvec(y), tau, data)
        rd = y - prox_dual(y + sigma * iteration_matvec(x), sigma, data)
        return jnp.concatenate([rp, rd])

    def refine_fn(x, y, data, steps=3, cg_iters=80):
        def R(z):
            return refine_residual_fn(z[:n], z[n:], data)
        z = gauss_newton_refine(R, jnp.concatenate([x, y]), steps=steps, cg_iters=cg_iters)
        return z[:n], z[n:]
    refine_fn.residual = refine_residual_fn

    # What the bounded refinement policy (refine="auto") reads to detect a
    # supported structure and to run its route in ORIGINAL coordinates: the
    # declared blocks and the raw stacked operator, never the equilibrated
    # steps.  Recording it costs nothing when the policy is off.
    refine_structure = {"g_block": g_block, "ops": list(ops), "blocks": list(blocks),
                        "slices": list(slices), "matvec": matvec, "rmatvec": rmatvec,
                        "n": n, "m": m, "has_cost": family.cost_fn is not None,
                        "folded_diagonal_quadratic": folded_diagonal_quadratic}
    if refine_budget is not None:
        from ..solver.refine_auto import RefineBudget as _RefineBudget
        if not isinstance(refine_budget, _RefineBudget):
            raise TypeError("refine_budget must be an ocs.solver.refine_auto.RefineBudget")
    if refine_memory_bytes is not None:
        if (isinstance(refine_memory_bytes, bool) or not isinstance(refine_memory_bytes, numbers.Integral)
                or refine_memory_bytes < 0):
            raise ValueError("refine_memory_bytes must be a nonnegative integer byte count")
        refine_memory_bytes = int(refine_memory_bytes)

    # Numerical separating/recession directions must satisfy the actual block
    # domains. Divergence magnitudes and data-direction cosines are insufficient.
    nrm_K_raw = _power_norm(matvec, rmatvec, n, iters=15, seed=0)
    infeas_stats = make_status_checks(
        matvec, rmatvec, g_block, blocks, slices, n, m,
        tol=min(tol, 1e-6), operator_norm=nrm_K_raw)

    meta = {"block": block, "primal_weight": primal_weight, "adaptive_omega": adaptive_omega,
            "adaptive_blocks": adaptive_blocks,
            "n_dual": len(dual), "n_groups": len(groups),
            "pdhg_variant": pdhg_variant, "dual_layout": dual_layout, "loop_unroll": loop_unroll,
            "data_layout": data_layout,
            "scaled_norm": scaled_norm, "gap_available": gap_ok,
            "objective_bound_capable": bound_capable_all,
            "objective_bound_missing_contracts": sorted(set(_bound_uncapable)),
            "equilibrate": equilibrate, "equil_diag": equil_diag,
            "online_warmup": online_warmup, "residual_norm": residual_norm, "tol": tol,
            "domain_guards": guards_enabled,
            "experimental_cones": domain_checks.has_experimental_cones,
            "cone_data_scales": cone_reference_scales is not None,
            "primal_scalar_metric": primal_scalar_metric,
            "primal_metric_tie": primal_metric_tie,
            "fixed_primal_validation": fixed_primal is not None,
            "feasibility_dual_recovery": feasibility_family,
            "folded_diagonal_quadratic": folded_diagonal_quadratic,
            "witness_contraction": witness_contraction,
            "psd_resolved_sign": ({"active": True, "envelope": "lambda_max(sym) <= -c k eps32 ||sym||_2, scoped float64",
                                   "c": float(_PSD_RESOLVED_SIGN_C),
                                   "ray_search_point": "re-validated as materialized, own adjoint"}
                                  if _psd_resolved else {"active": False}),
            "bound_anchor_setup_forward_actions": _anchor_probe_actions,
            "equality_projected_route": (_nodc_route.describe() if _nodc_route is not None else None),
            "dual_routes": {"requested": list(dual_routes),
                            "eligible": {"shift": bool(_shift_ok), "packing": bool(_packing_ok),
                                         "ridge": bool(_ridge_ok)},
                            "enabled": sorted(_dual_routes_jit),
                            "operator_applications": dict(DUAL_ROUTE_OPERATOR_APPLICATIONS),
                            "sources": dict(DUAL_ROUTE_SOURCES)},
            "scaling_classes": [b.scaling_class for b in blocks],
            "eval_kinds": [b.eval_kind for b in blocks]}

    def _step_source(j):
        s, e = slices[j]
        if given_sigma_form == "dict":
            return f"given sigma[{j}] (dict entry bound to rows {s}:{e})"
        if given_sigma_form == "vector":
            return f"given sigma[{s}:{e}]"
        if given_sigma_form == "scalar":
            return "given sigma (one scalar for every dual row)"
        if construct:
            return "constructed (equilibrate='construct')"
        return f"equilibrate={equilibrate!r}"

    def _metric_source(j):
        if j in term_metrics:
            return (f"block_metrics[{j}]" if block_metrics is not None
                    else f"constructed metric_classes[{j}]")
        if j in canonical_terms:
            return (f"block_metrics[{j}], canonicalized to the scalar tier (S = I, one step)"
                    if block_metrics is not None else
                    f"constructed metric_classes[{j}], canonicalized to the scalar tier")
        return "scalar tier: diag(sigma) on these rows"

    meta["routing"] = _routing_table(
        family, primal_term_index, folded_term_position, fold_candidate_position,
        dual, dual_positions, blocks, slices, m, explicit_metric_mapping,
        _step_source, _metric_source)
    meta["gap_check_schedule"] = {"max_checks": gap_checks, "growth": gap_check_growth,
                                  "first_blocks": gap_check_first,
                                  "segments_blocks": _check_schedule(max_blocks, first=gap_check_first,
                                                                     growth=gap_check_growth,
                                                                     max_checks=gap_checks)}
    if certificate_contract is not None:
        # Present only when one was declared, so that an undeclared family's
        # metadata is byte-for-byte what it was before this option existed.
        meta["certificate_contract"] = certificate_contract.describe(len(blocks))
    if construction_report is not None:
        construction_report["canonicalized_to_scalar_tier"] = list(canonical_terms)
        meta["metric_construction"] = construction_report
    meta["effective_metric"] = _describe_effective_metric(
        blocks, slices, groups, term_metrics, tau, sigma, n, m,
        primal_scalar_metric, primal_metric_tie, primal_weight, scaled_norm,
        column_shape=column_shape)
    if term_metrics or column_shape is not None:
        # The step condition is RE-MEASURED on the decoded shapes and the steps
        # the loop uses whenever any shape exists on either side.
        block_metric_meta = _describe_block_metrics(
            term_metrics, slices, blocks, matvec, rmatvec, tau, sigma, shape_apply, n, m,
            primal_scalar_metric, primal_weight,
            column_shape_apply=column_shape_apply, column_shape=column_shape)
        meta["block_metrics"] = block_metric_meta
    if term_metrics:
        # A cone term under an automorphism metric is labelled by the precise
        # class name rather than the generic "block", and keeps its own
        # evaluation kind through _shaped_eval_kind.
        automorphic = {i for i, (k, ranges) in term_metrics.items()
                       if isinstance(ranges[0][2], _cone_metric.ConeAutomorphism)}
        meta["scaling_classes"] = [_cone_metric.KIND if i in automorphic else
                                   _block_metric.BLOCK if i in term_metrics else b.scaling_class
                                   for i, b in enumerate(blocks)]
        meta["eval_kinds"] = [_shaped_eval_kind(term_metrics[i][1], b.eval_kind)
                              if i in term_metrics else b.eval_kind
                              for i, b in enumerate(blocks)]
    if canonical_terms:
        meta.setdefault("block_metrics", {"present": False, "terms": []})
        meta["block_metrics"]["canonicalized_to_scalar_tier"] = canonical_terms
    diff_solve = make_diff_solve(solver, iteration_matvec, iteration_rmatvec, prox_primal,
                                 prox_dual, tau, sigma, n, m)
    contract_stops = (certificate_contract is not None
                      and certificate_contract.stops_the_loop)
    if data_layout == "dense":
        single = build_single_solver(iteration_prox_primal, prox_dual, loop_measure,
                                     repair_dual, components, checked=checked_residual)
        drive_b, drive_s = build_drive_solvers(iteration_prox_primal, prox_dual,
                                               loop_measure, repair_dual, components)
        _fam = CompiledFamily(solver, resolve_data, report_py, n, m, meta,
                              refine=refine_fn, infeas=infeas_stats, tol=tol,
                              diff_solve=diff_solve, report_device=_report_body, single_solver=single,
                              objective_bound=objective_bound_fn,
                              contract=(None if contract_components is None
                                        else operand_jit(contract_components)),
                              contract_stops_loop=contract_stops,
                              max_iters=block * max_blocks,
                              contract_factory=contract_factory,
                              accuracy=accuracy, bound_capable=bound_capable_all,
                              drive_solver=drive_b, single_drive_solver=drive_s,
                              gap_checks=gap_checks, gap_check_growth=gap_check_growth,
                              gap_check_first=gap_check_first, max_blocks=max_blocks,
                              objective_bound_many=objective_bound_many_fn,
                              returned_components=components,
                              refine_policy=refine_policy, refine_budget=refine_budget,
                              refine_memory_bytes=refine_memory_bytes,
                              refine_structure=refine_structure)
        _fam._dual_routes = _dual_routes_jit
        return _fam

    def compact_checked(x, y, data):
        return checked_residual(x, y, expand_compact_data(data))

    compact_iteration_solver = build_iteration_solver(
        compact_iteration_prox_primal, compact_prox_dual, compact_loop_measure,
        check=compact_checked if single_evaluation else None)

    def compact_repair(x, y, data):
        return repair_dual(x, y, expand_compact_data(data))

    def compact_components(x, y, data):
        return components(x, y, expand_compact_data(data))

    @operand_jit
    def compact_solver(data, x0, y0):
        if single_evaluation:
            # The loop returns the checked pair and the residual it tested.
            return compact_iteration_solver(data, x0, y0)
        xx, yy, count, _res = compact_iteration_solver(data, x0, y0)
        yy = jax.vmap(compact_repair)(xx, yy, data)
        checked = jax.vmap(compact_components)(xx, yy, data)
        return xx, yy, count, jnp.maximum(checked["r_primal"], checked["r_dual"])

    @operand_jit
    def compact_infeas(x, y, data):
        return infeas_stats(x, y, expand_compact_data(data))

    # Preserve the static eligibility optimization in status classification.
    compact_infeas.status_may_certify = getattr(infeas_stats, "status_may_certify", True)

    def _compact_report_body(x, y, data):
        return _report_body(x, y, expand_compact_data(data))

    compact_report = operand_jit(_compact_report_body)

    def compact_report_py(x, y, data):
        return {key: float(value) for key, value in compact_report(x, y, data).items()}

    single = build_single_solver(compact_iteration_prox_primal, compact_prox_dual,
                                 compact_loop_measure, compact_repair, compact_components,
                                 checked=compact_checked)
    compact_contract = None
    if contract_components is not None:
        @operand_jit
        def compact_contract(x, y, data):
            return contract_components(x, y, expand_compact_data(data))
    drive_b, drive_s = build_drive_solvers(compact_iteration_prox_primal,
                                           compact_prox_dual, compact_loop_measure,
                                           compact_repair, compact_components)
    _fam = CompiledFamily(compact_solver, resolve_compact_data, compact_report_py,
                          n, m, meta, refine=refine_fn, infeas=compact_infeas,
                          tol=tol, diff_solve=diff_solve, report_device=_compact_report_body,
                          resolve_dense=resolve_data, single_solver=single,
                          objective_bound=objective_bound_fn,
                          contract=compact_contract,
                          contract_stops_loop=contract_stops,
                          max_iters=block * max_blocks,
                          accuracy=accuracy, bound_capable=bound_capable_all,
                          drive_solver=drive_b, single_drive_solver=drive_s,
                          gap_checks=gap_checks, gap_check_growth=gap_check_growth,
                              gap_check_first=gap_check_first, max_blocks=max_blocks,
                          objective_bound_many=objective_bound_many_fn,
                          returned_components=compact_components,
                          refine_policy=refine_policy, refine_budget=refine_budget,
                          refine_memory_bytes=refine_memory_bytes,
                          refine_structure=refine_structure,
                          contract_factory=(None if contract_factory is None else
                                            (lambda declared: (lambda x, y, data:
                                             contract_factory(declared)(
                                                 x, y, expand_compact_data(data))))))
    _fam._dual_routes = _dual_routes_jit
    return _fam
