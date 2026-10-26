"""The bounded automatic refinement policy, ``compile(refine="auto")``.

The policy is a TABLE, applied once to the assessed packet the ordinary
objective-gap path produced (``CompiledFamily._maybe_refine``):

====================================================  ==================================
situation                                             action
====================================================  ==================================
the assessed pair passes criterion (7)                return it; no refinement work
assessment unavailable (missing contract)             report the missing capability;
                                                      refinement never manufactures it
a supported structure, request unmet, stall guard     attempt ONE bounded refinement
fired                                                 stage
the stage is not admitted (support, workspace or      skip with the estimate and the
work estimate over its limit)                         limit named; packet unchanged
the refined pair passes the UNCHANGED criterion (7)   return the assessed refined pair
the correction is invalid, nonfinite, unsuccessful,   preserve the previous assessed
over budget or past its deadline                      packet; report the outcome
====================================================  ==================================

The trigger is fixed (not per benchmark): the existing stall event of the
objective-gap controller (``CompiledFamily._stalled``) on a supported
structure.  At most one stage runs per solve (per lane in a batch).

Three operations are kept distinct by name, in code and in every record:

* ``witness_recovery`` -- the objective bound's existing dual recovery
  (``_corrected_dual``, ray search, anchor; the SOC and no-DC routes).  It
  belongs to the assessment, is unchanged, and is NEVER performed here; the
  record only names the witness source the assessment reported.
* ``primal_dual_refinement`` -- the fixed-support primal-dual correction for
  a separable-penalty-plus-least-squares family (the FFT lasso shape):
  support S = {x_i != 0} of the stalled iterate, s = sign(x_S), solve
  K_S^T K_S d = -(K_S^T (K x + l) + c_S + lam s) by Jacobi-preconditioned CG
  on operator products only, propose (x + P_S d, K (x + P_S d) + l).
* ``active_support_solver`` -- the active-support Newton correction for a
  nonnegative-quadratic-plus-equality family (the quadratic OT shape):
  support S = {z > 0}, z = -K^T y - c; solve K M_S K^T dy = mu (K X_pred + l),
  X_pred = (z)_S / mu, by Jacobi-preconditioned CG on operator products (the
  constraint matrix is never formed; for a ``MarginalOperator`` the known
  per-component gauge is projected out); propose the plan candidates
  X = [z(y + t dy)]_+ / mu (full and restricted to S) with the dual y + t dy,
  for a backtracking t with the Armijo test on the concave dual value.

Every proposal is a float32 pair in original coordinates and is assessed
through the family's UNCHANGED evaluator exactly as the controller assesses
a check point (``_objective_bound`` with recovery, ``_gap_fields``,
``_infeasibility``, ``_gap_accepts`` at the compiled ``tol``).  Nothing in
this module touches criterion (7), its normalizations, the conjugate-domain
checks or the acceptance decision.  The route arithmetic runs under a scoped
``jax.enable_x64`` and the record states the dtype the operator actually
produced.  The record charges the stage end to end: operator applications
(correction products exactly counted, assessment products as the evaluator's
own upper bound), CG iterations, assessments, trials, wall seconds, and the
assessed state it started from.

Bounded execution
-----------------
A stage is a sequence of SEGMENTS, each one compiled call (or one bounded
host step) followed by ``jax.block_until_ready`` on its outputs, so that the
clock read at a segment boundary measures finished device work and not
asynchronous dispatch.  Before a segment starts, its operator applications
are checked against ``RefineBudget.max_operator_applications`` and, when its
duration can be projected from measured product times, ``elapsed +
projected`` is checked against ``RefineBudget.max_seconds``; after it ends,
the device memory the stage holds is checked against the admitted bytes.
The Krylov solve is split into compiled chunks of at most
``RefineBudget.cg_chunk_iters`` iterations; the first chunk runs at most
8 iterations and measures the iteration time, and every later chunk is
sized from it to about ``RefineBudget.segment_seconds`` (never fewer than
one iteration, i.e. two operator products, which is the smallest segment
this route can have).  The deadline covers everything inside the stage,
including tracing and compilation of its executables: the clock and the
admission estimate are checked BEFORE anything is compiled, and for
``refine_assessed`` the clock starts at the call's entry, so the assessment
of the supplied pair counts against it.  Compiled executables are cached on
the compiled family, so a later stage (another lane, another call) neither
recompiles nor runs without a projection.

A stage ends through exactly one of these exits, written to
``refinement["exit"]``:

``admission_refusal``
    Not attempted: a support-size/density, workspace (device or host),
    corrections/trials/assessments or operator-work check failed BEFORE the
    correction was constructed.  ``outcome`` is ``"skipped: ..."``,
    ``refusal`` names the check with its estimate and its limit, and the
    previous assessed packet is returned byte for byte.  (The
    active-support route needs one adjoint product to read its support;
    ``admission["products_before_admission"]`` says so.)
``bounded_stop``
    A limit bound the execution at a segment boundary: the deadline, the
    operator-application budget, the memory budget, the assessment or trial
    count, or the Krylov iteration allowance.  ``bounded_stop`` names the
    limit and the segment.  If no proposal was assessed, the previous packet
    is returned byte for byte; if the truncated work produced a proposal, it
    was assessed by the unchanged criterion and either returned (accepted)
    or refused with the previous packet kept.
``completed``
    The stage ran to its natural end; the outcome is ``accepted`` or a
    refusal by validity or by the criterion.
``failed``
    The correction raised; refused, previous packet kept.

Rows that never start a stage carry ``exit == "no_stage"``.

What the deadline cannot do, and the external watchdog
------------------------------------------------------
A segment that has started runs to its end: neither one compiled call nor
an XLA compilation can be interrupted from Python, and device work cannot
be cancelled.  The overshoot past ``max_seconds`` is therefore at most the
error of one segment's projection (or one compilation, which is measured
but not projected on the first stage of a family).  The library does NOT
rely on anything stronger.  A caller that needs a hard process-level wall
limit runs the solve in a process it can kill, for example::

    # shell: SIGKILL after 900 s
    timeout --signal=KILL 900 python row.py ...
    # Python parent: the child is killed when the timeout expires
    subprocess.run([sys.executable, "row.py", ...], timeout=900)

or arms :func:`process_watchdog` around the call inside the process::

    with refine_auto.process_watchdog(900, marker_path="row.watchdog.json"):
        sol = comp.solve_instance(data)

The watchdog is a separate child process that SIGKILLs the caller when the
limit expires and writes the marker ``{"exit": "external_watchdog",
"verdict": null, ...}``; disarming it is leaving the ``with`` block.  A
killed process returns no ``Solution``: the caller must record the row as
interrupted with no numerical verdict, never as accepted or refused.
"""
from __future__ import annotations

import contextlib
import math
import numbers
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict, replace as _dc_replace

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ..operators.structured import MarginalOperator
from ..prox.library import L1, SumSquares, NonNeg, Equality
from ..prox.base import _LinQuad

__all__ = ["RefineBudget", "ROUTE_PRIMAL_DUAL_REFINEMENT", "ROUTE_ACTIVE_SUPPORT_SOLVER",
           "WITNESS_RECOVERY", "detect_route", "workspace_bytes", "host_workspace_bytes",
           "admitted_memory_bytes", "admitted_host_bytes", "apply_policy", "record_unavailable",
           "refine_assessed", "process_watchdog", "EXIT_NO_STAGE", "EXIT_ADMISSION_REFUSAL",
           "EXIT_BOUNDED_STOP", "EXIT_COMPLETED", "EXIT_FAILED", "EXIT_EXTERNAL_WATCHDOG"]

#: The two refinement operations this policy can perform.
ROUTE_PRIMAL_DUAL_REFINEMENT = "primal_dual_refinement"
ROUTE_ACTIVE_SUPPORT_SOLVER = "active_support_solver"
#: The assessment's own dual recovery (SOC / no-DC witness routes).  Named so
#: that no record can confuse it with a refinement; never performed here.
WITNESS_RECOVERY = "witness_recovery"

#: ``refinement["exit"]`` values (see the module docstring).
EXIT_NO_STAGE = "no_stage"
EXIT_ADMISSION_REFUSAL = "admission_refusal"
EXIT_BOUNDED_STOP = "bounded_stop"
EXIT_COMPLETED = "completed"
EXIT_FAILED = "failed"
#: Written only by :func:`process_watchdog` into its marker file; the stage
#: never produces it (a killed process returns nothing).
EXIT_EXTERNAL_WATCHDOG = "external_watchdog"

POLICY = "auto"

#: Iterations of the first Krylov chunk: short, so that its measured
#: duration sizes the later chunks before a long one is dispatched.
_FIRST_CHUNK_ITERS = 8


@dataclass(frozen=True)
class RefineBudget:
    """Hard bounds of the one stage.  Each is checked before the work it bounds.

    ``max_cg_iters``, ``max_trials``, ``max_assessments``, ``max_corrections``
    and ``max_operator_applications`` are counts; ``max_seconds`` is the
    wall-clock deadline of the stage (see the module docstring for what it
    covers); ``max_support_fraction`` and ``support_within_rows`` are the
    admission limits on the support; ``max_extra_bytes`` caps the admitted
    device workspace below what ``compile(refine_memory_bytes=...)`` or the
    device reports; ``cg_chunk_iters`` and ``segment_seconds`` size the
    compiled Krylov chunks.  Every route performs at most ONE correction, so
    ``max_corrections`` admits it (>= 1) or refuses it (0).
    """
    max_cg_iters: int = 4000              # Krylov iterations of the one linear solve
    cg_rtol: float = 1e-12                # its relative residual target
    max_trials: int = 3                   # backtracking trials (active-support route)
    max_assessments: int = 8              # criterion-(7) evaluations of proposals
    max_operator_applications: int = 50_000   # correction + assessment, upper bound
    max_seconds: float = 600.0            # wall clock of the whole stage
    max_corrections: int = 1              # outer corrections (linear solves)
    max_support_fraction: float = 0.5     # admitted |S| / n
    support_within_rows: bool = True      # primal-dual route: admit only |S| <= rows of K
    max_extra_bytes: int | None = None    # cap on the admitted extra device bytes
    cg_chunk_iters: int = 1024            # Krylov iterations per compiled chunk, at most
    segment_seconds: float = 2.0          # target duration of one Krylov chunk

    def __post_init__(self):
        for name in ("max_cg_iters", "max_trials", "max_assessments",
                     "max_operator_applications", "max_corrections"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
                raise ValueError("RefineBudget.%s must be a nonnegative integer, not %r" % (name, v))
        v = self.cg_chunk_iters
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError("RefineBudget.cg_chunk_iters must be a positive integer, not %r" % (v,))
        for name in ("cg_rtol", "max_seconds"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or math.isnan(v) or v < 0:
                raise ValueError("RefineBudget.%s must be a nonnegative number, not %r" % (name, v))
        v = self.segment_seconds
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not (0 < v < math.inf):
            raise ValueError("RefineBudget.segment_seconds must be a positive finite number, not %r" % (v,))
        v = self.max_support_fraction
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not (0.0 <= v <= 1.0):
            raise ValueError("RefineBudget.max_support_fraction must lie in [0, 1], not %r" % (v,))
        if not isinstance(self.support_within_rows, bool):
            raise ValueError("RefineBudget.support_within_rows must be a bool")
        v = self.max_extra_bytes
        if v is not None and (isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0):
            raise ValueError("RefineBudget.max_extra_bytes must be None or a nonnegative integer, not %r" % (v,))


# ---------------------------------------------------------------------------
# Route detection: a structural fact about the compiled family.
# ---------------------------------------------------------------------------
def _block_name(block):
    if block is None:
        return "none"
    base = getattr(block, "base", None)
    name = type(block).__name__
    return f"{name}({_block_name(base)})" if base is not None else name


def detect_route(structure):
    """Return ``(route, reason)``: the supported refinement route for this
    compiled structure, or ``(None, why)``.  Reads blocks and operator
    counts only; never the operator's entries."""
    if structure is None:
        return None, "no structure record (not a primal-dual compiled family)"
    g = structure["g_block"]
    blocks = structure["blocks"]
    if structure.get("folded_diagonal_quadratic"):
        return None, "a folded diagonal least-squares term has no refinement route"
    if len(blocks) != 1:
        return None, ("refinement routes exist for exactly one dual term; this family has %d"
                      % len(blocks))
    blk = blocks[0]
    if type(g) is L1 and type(blk) is SumSquares:
        return ROUTE_PRIMAL_DUAL_REFINEMENT, None
    if (type(g) is _LinQuad and type(g.base) is NonNeg and getattr(g, "_e_mode", 0) == 1
            and bool(np.all(np.asarray(g.d) == 0.0)) and type(blk) is Equality):
        return ROUTE_ACTIVE_SUPPORT_SOLVER, None
    return None, ("no refinement route for primal block %s with dual block %s"
                  % (_block_name(g), _block_name(blk)))


def workspace_bytes(route, n, m):
    """An estimate of the device arrays the stage holds, in bytes (the
    admission check).

    Primal-dual route: 28 float64 n-vectors (the iterate, cost, support
    copies, gradients, right-hand side, preconditioner, the Krylov state in
    and out of a chunk with its two temporaries, and three float64-sized
    slots per complex128 transform buffer) and 8 float64 m-vectors, plus the
    support mask.  Active-support route: 8 float64 n-vectors (the reduced
    cost, prediction, two operator temporaries, a trial's dual slack and its
    plan candidates) and 8 m-vectors, plus the mask.  It does not count
    compilation or library scratch (autotuning, FFT plans), which on the
    first stage of a compiled family can exceed it.  The runtime check at
    every segment boundary counts every byte the device allocator reports
    beyond the stage's starting level, scratch included, against the
    admitted budget, so an estimate that is too low cannot let the stage
    hold more than it was admitted."""
    if route == ROUTE_PRIMAL_DUAL_REFINEMENT:
        return 8 * (28 * n + 8 * m) + n
    if route == ROUTE_ACTIVE_SUPPORT_SOLVER:
        return 8 * (8 * n + 8 * m) + n
    return 0


def host_workspace_bytes(route, n, m, support, rows=None, cols=None):
    """An upper estimate of the stage's HOST workspace in bytes.  Only the
    active-support route on a ``MarginalOperator`` uses host memory: the
    support mask copied to the host (n bytes), its coordinates (two int64
    per entry), the support graph in COO and CSR form and its conversion
    copy (about 24 bytes per entry), and per-vertex labels and counts."""
    if route == ROUTE_ACTIVE_SUPPORT_SOLVER and rows is not None and cols is not None:
        return int(n + 40 * int(support) + 16 * (int(rows) + int(cols) + 1))
    return 0


def admitted_memory_bytes(configured=None):
    """The admitted workspace budget: the configured byte count, else 80 % of
    what the default device reports free, else 80 % of host MemAvailable.
    Returns ``(bytes or None, how)``."""
    if configured is not None:
        return int(configured), "configured: compile(refine_memory_bytes=...)"
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:                                   # pragma: no cover - backend specific
        stats = None
    if stats and "bytes_limit" in stats:
        free = int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))
        return int(0.8 * max(free, 0)), "device: 0.8 (bytes_limit - bytes_in_use)"
    host, how = admitted_host_bytes()
    if host is not None:
        return host, how
    return None, "no memory accounting available"


def admitted_host_bytes():
    """80 % of host MemAvailable, or ``(None, why)``."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return int(0.8 * kb * 1024), "host: 0.8 MemAvailable"
    except OSError:
        pass
    return None, "no host memory accounting available"


# ---------------------------------------------------------------------------
# Bounded execution: segments, their boundaries and the three exits.
# ---------------------------------------------------------------------------
class _Admission(Exception):
    """Exit 1: the stage is not attempted (a named admission check failed)."""

    def __init__(self, name, detail, estimate=None, limit=None, also=()):
        super().__init__(name)
        self.name, self.detail, self.estimate, self.limit, self.also = (
            name, detail, estimate, limit, list(also))


class _Stop(Exception):
    """Exit 2: a limit bound the execution at a segment boundary."""

    def __init__(self, limit, segment, detail, projected=None):
        super().__init__(limit)
        self.limit, self.segment, self.detail, self.projected = limit, segment, detail, projected

    def as_dict(self, elapsed):
        return {"limit": self.limit, "segment": self.segment, "detail": self.detail,
                "projected_seconds": self.projected, "elapsed_seconds": elapsed,
                "then": "no proposal assessed; the previous assessed packet is returned"}


def _device_memory():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:                                   # pragma: no cover - backend specific
        return None
    if not stats or "bytes_in_use" not in stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return int(stats["bytes_in_use"]), (int(peak) if peak is not None else None)


class _MemoryWatch:
    """The device bytes the stage holds beyond what was in use when it began,
    read at every segment boundary (and from the allocator's peak when the
    peak moved during the stage)."""

    def __init__(self, admitted):
        self.admitted = admitted
        s = _device_memory()
        self.available = s is not None
        self.base_in_use, self.base_peak = s if s is not None else (None, None)
        self.extra_max = 0
        self.checks = 0

    def extra(self):
        if not self.available:
            return None
        s = _device_memory()
        if s is None:
            return None
        in_use, peak = s
        e = in_use - self.base_in_use
        if peak is not None and self.base_peak is not None and peak > self.base_peak:
            e = max(e, peak - self.base_in_use)
        return e

    def check(self, segment):
        e = self.extra()
        if e is None:
            return
        self.checks += 1
        self.extra_max = max(self.extra_max, e)
        if self.admitted is not None and e > self.admitted:
            raise _Stop("max_extra_bytes", segment,
                        "the stage holds %d extra device bytes after this segment; admitted %d"
                        % (e, self.admitted))

    def record(self):
        return {"runtime_check": ("device memory_stats at every segment boundary" if self.available
                                  else "unavailable (the backend reports no memory_stats)"),
                "observed_extra_bytes_max": (self.extra_max if self.available else None),
                "boundary_checks": self.checks}


def _cache(comp):
    c = getattr(comp, "_refine_cache", None)
    if c is None:
        c = {}
        try:
            comp._refine_cache = c
        except Exception:                               # pragma: no cover - defensive
            pass
    return c


class _Control:
    """Segment boundaries of one stage: deadline, operator budget, memory."""

    def __init__(self, comp, route, budget, t0, work, admitted, rec):
        self.comp, self.route, self.budget, self.t0, self.work, self.rec = (
            comp, route, budget, t0, work, rec)
        self.memory = _MemoryWatch(admitted)
        self.cache = _cache(comp)
        self.timing = self.cache.setdefault(("timing", route, int(comp.n), int(comp.m)), {})
        self.segments = 0
        self.synchronizations = 0
        self.longest = 0.0
        self.longest_name = None
        self.compile_seconds = 0.0
        self.compiles = 0
        self.reused = 0
        self.by_name = {}
        self.unreported_products = 0
        self.unreported_cg = 0
        self.bound = None
        self.corrections = 0
        self._open = None

    def elapsed(self):
        return time.perf_counter() - self.t0

    def operator_applications(self):
        return int(self.work["operator_applications"]) + int(self.unreported_products)

    def begin(self, segment, *, products=0, projected=None, check=True):
        """A segment boundary BEFORE ``segment`` starts."""
        if check:
            el = self.elapsed()
            cap = self.budget.max_seconds
            if el > cap:
                raise _Stop("max_seconds", segment,
                            "%.3f s elapsed exceed max_seconds %.3f s" % (el, cap))
            if projected is not None and el + projected > cap:
                raise _Stop("max_seconds", segment,
                            "%.3f s elapsed + %.3f s projected for this segment exceed max_seconds %.3f s"
                            % (el, projected, cap), projected=projected)
            if products and self.operator_applications() + products > self.budget.max_operator_applications:
                raise _Stop("max_operator_applications", segment,
                            "%d operator applications used + %d for this segment exceed %d"
                            % (self.operator_applications(), products,
                               self.budget.max_operator_applications))
        self._open = (segment, time.perf_counter())

    def end(self, *outputs, products=0, cg=0):
        """The boundary AFTER the open segment: synchronize, time, charge, check memory."""
        if outputs:
            jax.block_until_ready(outputs)
            self.synchronizations += 1
        name, t = self._open
        self._open = None
        dt = time.perf_counter() - t
        self.segments += 1
        slot = self.by_name.setdefault(name, [0, 0.0, 0.0])
        slot[0] += 1
        slot[1] += dt
        slot[2] = max(slot[2], dt)
        if dt > self.longest:
            self.longest, self.longest_name = dt, name
        self.charge(products, cg)
        self.memory.check(name)
        return dt

    def charge(self, products=0, cg=0):
        self.unreported_products += int(products)
        self.unreported_cg += int(cg)

    def take(self):
        p, c = self.unreported_products, self.unreported_cg
        self.unreported_products = self.unreported_cg = 0
        return p, c

    def compiled(self, key, name, fn, *args):
        """An AOT-compiled executable, cached on the compiled family.  The clock
        and the operator budget are checked BEFORE compiling."""
        full = ("exe", self.route) + tuple(key)
        exe = self.cache.get(full)
        if exe is not None:
            self.reused += 1
            return exe
        self.begin("compile: " + name)
        t = time.perf_counter()
        exe = jax.jit(fn).lower(*args).compile()
        self.compile_seconds += time.perf_counter() - t
        self.compiles += 1
        self.end()
        self.cache[full] = exe
        return exe

    def assessment_seconds(self):
        return float(self.timing.get("assessment_seconds", 0.0))

    def record(self):
        return {"segments": self.segments, "synchronizations": self.synchronizations,
                "longest_segment_seconds": self.longest, "longest_segment": self.longest_name,
                "compile_seconds": self.compile_seconds, "compiled_executables": self.compiles,
                "compiled_executables_reused": self.reused,
                "by_segment": {k: {"count": v[0], "seconds": v[1], "max_seconds": v[2]}
                               for k, v in self.by_name.items()},
                "measured_times": {k: v for k, v in self.timing.items()
                                   if isinstance(v, (int, float))}}


def _assessment_ops_estimate(comp, ctl):
    """Reserve the evaluator's structural work bound, including domain guards.

    The previous assessment's observed cost is diagnostic only: data-dependent
    witness work can cost more on the next proposal.
    """
    bound = getattr(comp._objective_bound, "operator_applications_bound", None)
    if bound is None:
        raise _Stop("max_operator_applications", "assessment reservation",
                    "the assessment does not declare an operator-work bound")
    return int(bound(comp._bound_rounds, recover=True))


def _admit(ctl, *, support, n, rows, fixed_products, rows_rule, extra_checks=(),
           products_before=0, per_iteration_products=2):
    """Admission BEFORE the correction is constructed and before anything is
    compiled.  Records every check; raises ``_Admission`` on the first that
    fails.  Returns the Krylov iteration allowance."""
    b = ctl.budget
    frac = (support / n) if n else 0.0
    checks = [
        {"name": "max_corrections", "estimate": 1, "limit": b.max_corrections,
         "ok": b.max_corrections >= 1,
         "detail": "the route performs one correction; RefineBudget.max_corrections = %d"
                   % b.max_corrections},
        {"name": "max_assessments", "estimate": 1, "limit": b.max_assessments,
         "ok": b.max_assessments >= 1,
         "detail": "a proposal needs one assessment; RefineBudget.max_assessments = %d"
                   % b.max_assessments},
        {"name": "support_density", "estimate": frac, "limit": b.max_support_fraction,
         "ok": frac <= b.max_support_fraction,
         "detail": "the support has %d of %d entries (fraction %.6f); RefineBudget.max_support_fraction = %.6f"
                   % (support, n, frac, b.max_support_fraction)},
    ]
    if rows_rule:
        checks.append(
            {"name": "support_exceeds_rows", "estimate": int(support), "limit": int(rows),
             "ok": (not b.support_within_rows) or support <= rows,
             "detail": ("the support has %d entries and K has %d rows, so K_S^T K_S has rank at most %d%s; "
                        "RefineBudget.support_within_rows = %s"
                        % (support, rows, min(support, rows),
                           " and the fixed-support system is rank deficient" if support > rows else "",
                           b.support_within_rows))})
    checks.extend(extra_checks)
    assess = _assessment_ops_estimate(ctl.comp, ctl)
    used = ctl.operator_applications()
    base = used + fixed_products + assess + 2
    minimum = base + (per_iteration_products if b.max_cg_iters >= 1 else 0)
    checks.append(
        {"name": "operator_applications", "estimate": int(minimum), "limit": b.max_operator_applications,
         "ok": minimum <= b.max_operator_applications,
         "detail": ("the smallest useful stage needs %d operator applications (%d used, %d fixed products, "
                    "one Krylov iteration, one assessment bounded by %d, 2 for the returned pair); "
                    "RefineBudget.max_operator_applications = %d"
                    % (minimum, used, fixed_products, assess, b.max_operator_applications))})
    allowance = int(min(b.max_cg_iters, max(0, (b.max_operator_applications - base) // per_iteration_products)))
    t = ctl.timing
    planned = {"operator_applications_minimum": int(minimum),
               "operator_applications_maximum": int(base + per_iteration_products * allowance),
               "krylov_iteration_allowance": allowance,
               "krylov_allowance_bound_by": ("max_cg_iters" if allowance >= b.max_cg_iters
                                             else "max_operator_applications"),
               "assessment_operator_applications_estimate": int(assess)}
    if "forward_seconds" in t and "adjoint_seconds" in t:
        tp = float(t["forward_seconds"]) + float(t["adjoint_seconds"])
        ti = float(t.get("iteration_seconds", tp))
        ta = float(t.get("assessment_seconds", 0.0))
        fixed_s = fixed_products * max(float(t["forward_seconds"]), float(t["adjoint_seconds"]))
        smin = fixed_s + (ti if b.max_cg_iters >= 1 else 0.0) + ta
        smax = fixed_s + allowance * ti + ta
        planned.update(projected_seconds_minimum=smin, projected_seconds_maximum=smax,
                       projection_from="product times measured by an earlier stage on this compiled family")
        el = ctl.elapsed()
        checks.append(
            {"name": "projected_seconds", "estimate": el + smin, "limit": b.max_seconds,
             "ok": el + smin <= b.max_seconds,
             "detail": ("%.3f s elapsed + %.3f s projected for the smallest useful stage exceed "
                        "max_seconds %.3f s" % (el, smin, b.max_seconds))})
    else:
        planned.update(projected_seconds_minimum=None, projected_seconds_maximum=None,
                       projection_from=("none yet: the first stage on this compiled family measures its "
                                        "product times in the setup segments, and every later segment "
                                        "is admitted against them"))
    adm = ctl.rec.setdefault("admission", {"checks": [], "admitted": None})
    adm["checks"].extend(checks)
    adm["planned"] = planned
    adm["products_before_admission"] = int(products_before)
    bad = [c for c in checks if not c["ok"]]
    if bad:
        adm["admitted"] = False
        raise _Admission(bad[0]["name"], bad[0]["detail"], bad[0].get("estimate"), bad[0].get("limit"),
                         also=[c["name"] for c in bad[1:]])
    adm["admitted"] = True
    return allowance


# ---------------------------------------------------------------------------
# Jacobi-preconditioned CG on a symmetric positive semidefinite operator, on
# device, with breakdown detection, in compiled chunks of bounded length.
# The body is the same arithmetic as one lax.while_loop over all iterations;
# the chunk only adds the ``it < stop`` bound of its own segment.
# ---------------------------------------------------------------------------
def _cg_init(rhs, minv):
    bnorm = jnp.linalg.norm(rhs)
    z0 = minv * rhs
    return (jnp.zeros_like(rhs), rhs, z0, rhs @ z0, jnp.asarray(0, jnp.int32),
            jnp.asarray(jnp.inf, rhs.dtype), jnp.asarray(False), bnorm == 0.0, bnorm, bnorm)


def _cg_chunk(apply, st, stop, minv, rtol, aux):
    def cond(s):
        return (~s[7]) & (~s[6]) & (s[4] < stop)

    def body(s):
        x, r, p, rz, it, minrq, brk, done, rn, bnorm = s
        Ap = apply(p, aux)
        pAp = p @ Ap
        pp = p @ p
        rq = jnp.where(pp > 0, pAp / jnp.where(pp > 0, pp, 1.0), 0.0)
        minrq = jnp.minimum(minrq, rq)
        brk = ~(pAp > 1e-300 * jnp.maximum(1.0, pp))        # rank deficient / indefinite
        a = jnp.where(brk, 0.0, rz / jnp.where(brk, 1.0, pAp))
        x = x + a * p
        r = r - a * Ap
        it = it + jnp.where(brk, 0, 1)
        rn = jnp.linalg.norm(r)
        done = rn <= rtol * bnorm
        z = minv * r
        rz2 = r @ z
        beta = jnp.where(rz > 0, rz2 / jnp.where(rz > 0, rz, 1.0), 0.0)
        p = jnp.where(brk | done, p, z + beta * p)
        rz = jnp.where(brk | done, rz, rz2)
        return (x, r, p, rz, it, minrq, brk, done, rn, bnorm)

    return lax.while_loop(cond, body, st)


def _pcg_bounded(ctl, key, apply, rhs, minv, aux, *, allowance, max_iters, rtol, t_iter, reserve):
    """The one linear solve, segment by segment.  ``allowance`` is the
    iteration count admitted (``max_cg_iters`` or less, when the operator
    budget binds); each chunk is admitted against the deadline with the
    projected chunk time plus ``reserve`` (what must still run after it)."""
    b = ctl.budget
    if ctl.corrections >= b.max_corrections:
        raise _Stop("max_corrections", "linear solve",
                    "RefineBudget.max_corrections = %d corrections already ran" % b.max_corrections)
    ctl.corrections += 1
    rtol = float(rtol)
    init_exe = ctl.compiled(("cg_init",) + tuple(key), "Krylov initialization", _cg_init, rhs, minv)
    ctl.begin("Krylov initialization")
    st = init_exe(rhs, minv)
    ctl.end(st)

    def chunk_fn(s, stop, w, a):
        return _cg_chunk(apply, s, stop, w, rtol, a)
    chunk_exe = ctl.compiled(("cg_chunk", rtol) + tuple(key), "Krylov chunk", chunk_fn,
                             st, jnp.asarray(0, jnp.int32), minv, aux)
    chunks, largest = 0, 0
    t_iter = max(float(t_iter), 1e-12)
    while True:
        it = int(st[4])
        if bool(st[7]) or bool(st[6]) or it >= allowance:
            break
        k = int(min(b.cg_chunk_iters, allowance - it, max(1, int(b.segment_seconds // t_iter))))
        if chunks == 0:
            k = min(k, _FIRST_CHUNK_ITERS)               # a short first chunk measures the iteration
        ctl.begin("Krylov chunk", products=2 * k, projected=k * t_iter + reserve)
        st = chunk_exe(st, jnp.asarray(it + k, jnp.int32), minv, aux)
        dt = ctl.end(st)
        ran = int(st[4]) - it
        ctl.charge(products=2 * ran, cg=ran)
        chunks += 1
        largest = max(largest, k)
        if ran > 0:
            t_iter = max(dt / ran, 1e-12)
    ctl.timing["iteration_seconds"] = t_iter
    x, r, p, rz, it, minrq, brk, done, rn, bnorm = st
    it, done, brk = int(it), bool(done), bool(brk)
    bnorm = float(bnorm)
    minrq = float(minrq)
    if done:
        stopped_by = "converged"
    elif brk:
        stopped_by = "breakdown"
    else:
        stopped_by = "max_cg_iters" if allowance >= max_iters else "max_operator_applications"
        ctl.bound = {"limit": stopped_by, "segment": "linear solve",
                     "detail": ("the Krylov solve stopped at its iteration allowance %d (max_cg_iters %d) "
                                "before reaching rtol; its proposal is assessed by the unchanged criterion"
                                % (allowance, max_iters)),
                     "then": "proposal assessed"}
    return x, {"iterations": it, "converged": done, "breakdown": brk,
               "final_relative_residual": (float(rn) / bnorm if bnorm > 0 else 0.0),
               "min_rayleigh": (minrq if math.isfinite(minrq) else None),
               "rhs_norm": bnorm, "max_iters": int(max_iters), "rtol": rtol,
               "iteration_allowance": int(allowance), "stopped_by": stopped_by,
               "chunks": chunks, "largest_chunk_iterations": largest,
               "seconds_per_iteration": t_iter}


# ---------------------------------------------------------------------------
# Route 1: primal-dual refinement (separable penalty + least squares).
# ---------------------------------------------------------------------------
def _propose_primal_dual(structure, x32, dense, budget, hook, ctl):
    """One fixed-support correction; yields at most one proposal."""
    K, KT = structure["matvec"], structure["rmatvec"]
    op = structure["ops"][0]
    lam = float(np.asarray(structure["g_block"].lam))
    l32, c32 = dense
    n, m = int(ctl.comp.n), int(ctl.comp.m)
    info = {"kind": ROUTE_PRIMAL_DUAL_REFINEMENT, "operation": ROUTE_PRIMAL_DUAL_REFINEMENT,
            "linear_solver": "jacobi_pcg on K_S^T K_S (operator products only)",
            "products": 0, "cg_iterations": 0}
    t0 = time.perf_counter()
    # ---- the support, read from the iterate: no operator work, nothing compiled
    override = hook.get("support") if hook else None
    if override is None:
        mask32 = jnp.asarray(x32) != 0.0
        info["support_source"] = "stalled iterate: {i : x_i != 0}"
    else:
        host = np.asarray(override, bool).reshape(-1)
        if host.shape != (n,):
            raise ValueError("support override has the wrong shape")
        mask32 = jnp.asarray(host)
        info["support_source"] = "override (test hook)"
    n_s = int(jnp.count_nonzero(mask32))
    info["support_size"] = n_s
    if n_s == 0:
        info.update(withdrawn="empty support", seconds=time.perf_counter() - t0)
        return
    well = bool(getattr(op, "well_conditioned", False))
    # ---- admission, before the correction is constructed or compiled
    allowance = _admit(ctl, support=n_s, n=n, rows=m, fixed_products=4 + (1 if well else 0),
                       rows_rule=True)
    with jax.enable_x64():
        x = jnp.asarray(x32, jnp.float64)
        l = jnp.asarray(l32, jnp.float64)
        c = jnp.asarray(c32, jnp.float64)
        mask = mask32
        kfwd = ctl.compiled(("fwd", n), "forward product (float64)", lambda v: jnp.asarray(K(v)), x)
        kadj = ctl.compiled(("adj", m), "adjoint product (float64)", lambda u: jnp.asarray(KT(u)), l)
        xS = jnp.where(mask, x, 0.0)
        s = jnp.sign(xS)
        tf, ta = ctl.timing.get("forward_seconds"), ctl.timing.get("adjoint_seconds")
        ctl.begin("forward product", products=1, projected=tf)
        r0 = kfwd(xS) + l
        t_fwd = ctl.end(r0, products=1)
        ctl.begin("adjoint product", products=1, projected=ta)
        g = kadj(r0) + c
        t_adj = ctl.end(g, products=1)
        ctl.timing["forward_seconds"], ctl.timing["adjoint_seconds"] = t_fwd, t_adj
        info["arithmetic"] = str(np.dtype(g.dtype))
        inactive0 = jnp.where(mask, 0.0, jnp.abs(g)) / lam
        info["active_deviation_before"] = float(jnp.max(jnp.where(mask, jnp.abs(g + lam * s), 0.0)) / lam)
        info["inactive_max_before"] = float(jnp.max(inactive0))
        info["inactive_violations_before"] = int(jnp.count_nonzero(inactive0 > 1.0))
        rhs = jnp.where(mask, -(g + lam * s), 0.0)
        # Jacobi diagonal: one probed column when the operator certifies a
        # uniform column scale; otherwise the identity (recorded).
        if well:
            first = int(jnp.argmax(mask))
            e = jnp.zeros_like(x).at[first].set(1.0)
            ctl.begin("forward product", products=1, projected=t_fwd)
            col = kfwd(e)
            ctl.end(col, products=1)
            colnorm2 = float(jnp.sum(col ** 2))
            info["jacobi_diagonal"] = colnorm2
            minv = jnp.full_like(x, 1.0 / colnorm2 if colnorm2 > 0 else 1.0)
        else:
            info["jacobi_diagonal"] = None
            minv = jnp.ones_like(x)

        def apply(v, aux):
            msk = aux[0]
            return jnp.where(msk, jnp.asarray(KT(jnp.asarray(K(jnp.where(msk, v, 0.0))))), 0.0)

        d, lin = _pcg_bounded(ctl, ("pd", n, m), apply, rhs, minv, (mask,), allowance=allowance,
                              max_iters=budget.max_cg_iters, rtol=budget.cg_rtol,
                              t_iter=t_fwd + t_adj,
                              reserve=t_fwd + t_adj + ctl.assessment_seconds())
        info["linear_solve"] = lin
        x_new = xS + jnp.where(mask, d, 0.0)
        ctl.begin("forward product", products=1, projected=t_fwd)
        r_new = kfwd(x_new) + l
        ctl.end(r_new, products=1)
        ctl.begin("adjoint product", products=1, projected=t_adj)
        g_new = kadj(r_new) + c
        ctl.end(g_new, products=1)
        inactive1 = jnp.where(mask, 0.0, jnp.abs(g_new)) / lam
        info["sign_flips"] = int(jnp.count_nonzero(mask & (jnp.sign(x_new) != s)))
        info["active_deviation_after"] = float(jnp.max(jnp.where(mask, jnp.abs(g_new + lam * s), 0.0)) / lam)
        info["inactive_max_after"] = float(jnp.max(inactive1))
        info["inactive_violations_after"] = int(jnp.count_nonzero(inactive1 > 1.0))
        info["correction_norm"] = float(jnp.linalg.norm(d))
        info["support_valid_f64"] = bool(lin["converged"] and not lin["breakdown"]
                                         and info["sign_flips"] == 0
                                         and info["inactive_violations_after"] == 0)
        J = 0.5 * float(r_new @ r_new) + lam * float(jnp.sum(jnp.abs(x_new))) + float(c @ x_new)
        dval = float(l @ r_new) - 0.5 * float(r_new @ r_new)
        info["f64_primal_value"] = J
        info["f64_dual_value_same_witness"] = dval
        info["f64_discrepancy"] = abs(J - dval) / max(abs(J), abs(dval), 1e-300)
        info["f64_witness_in_domain"] = bool(float(jnp.max(jnp.abs(g_new))) <= lam)
        xp = jnp.asarray(x_new, jnp.float32)
        yp = jnp.asarray(r_new, jnp.float32)
    info["seconds"] = time.perf_counter() - t0
    info["products"], info["cg_iterations"] = ctl.take()
    yield xp, yp, info


# ---------------------------------------------------------------------------
# Route 2: active-support solver (nonnegative quadratic + equality).
# ---------------------------------------------------------------------------
def _marginal_gauge(op, mask, info):
    """For a MarginalOperator: the known nullspace of K M_S K^T, one
    (1 on rows, -1 on columns) vector per connected component of the support
    graph, found on the host from the support's coordinates.  Returns
    ``(k, lab_u, lab_v, size)`` for :func:`_gauge_project`."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    N, Mc = op.N, op.Mc
    host = np.asarray(mask).reshape(N, Mc)
    ii, jj = np.nonzero(host)
    g = sp.coo_matrix((np.ones(len(ii), np.int8), (ii, N + jj)), shape=(N + Mc, N + Mc)).tocsr()
    k, lab = csg.connected_components(g, directed=False)
    lab_u = jnp.asarray(lab[:N]); lab_v = jnp.asarray(lab[N:])
    size = jnp.asarray(np.bincount(lab, minlength=k).astype(np.float64))
    info["nullspace"] = {"handling": "MarginalOperator gauge projected per connected component",
                         "n_components": int(k), "support_edges": int(len(ii)),
                         "empty_rows": int(np.sum(host.sum(1) == 0)),
                         "empty_cols": int(np.sum(host.sum(0) == 0))}
    return int(k), lab_u, lab_v, size


def _gauge_project(v, lab_u, lab_v, size, k, N):
    ru, rv = v[:N], v[N:]
    alpha = (jax.ops.segment_sum(ru, lab_u, num_segments=k)
             - jax.ops.segment_sum(rv, lab_v, num_segments=k)) / size
    return jnp.concatenate([ru - alpha[lab_u], rv + alpha[lab_v]])


def _propose_active_support(structure, x32, y32, dense, budget, hook, ctl):
    """One Newton correction of the equality multipliers on the active
    support, then backtracking trials; yields (restricted, full) plan
    candidates per trial until an Armijo-accepted trial has been yielded."""
    K, KT = structure["matvec"], structure["rmatvec"]
    op = structure["ops"][0]
    g_block = structure["g_block"]
    mu = float(np.asarray(g_block.e).reshape(-1)[0])
    l32, c32 = dense
    n, m = int(ctl.comp.n), int(ctl.comp.m)
    marginal = isinstance(op, MarginalOperator)
    base = {"kind": ROUTE_ACTIVE_SUPPORT_SOLVER, "operation": ROUTE_ACTIVE_SUPPORT_SOLVER,
            "linear_solver": "jacobi_pcg on K M_S K^T (operator products only)", "mu": mu}
    t0 = time.perf_counter()
    b = budget
    if b.max_trials < 1:
        adm = ctl.rec.setdefault("admission", {"checks": [], "admitted": False})
        adm["checks"].append({"name": "max_trials", "estimate": 1, "limit": b.max_trials, "ok": False,
                              "detail": "a proposal needs one line-search trial; RefineBudget.max_trials = 0"})
        adm["admitted"] = False
        adm["products_before_admission"] = 0
        raise _Admission("max_trials", "a proposal needs one line-search trial; RefineBudget.max_trials = 0",
                         1, 0)
    with jax.enable_x64():
        y = jnp.asarray(y32, jnp.float64)
        l = jnp.asarray(l32, jnp.float64)
        c = jnp.asarray(c32, jnp.float64)
        # The support needs one adjoint product: z = -K^T y - c.  Its compile
        # and its product are admitted against the clock and the budget first.
        kadj = ctl.compiled(("adj", m), "adjoint product (float64)", lambda u: jnp.asarray(KT(u)), y)
        ctl.begin("adjoint product", products=1, projected=ctl.timing.get("adjoint_seconds"))
        z = -kadj(y) - c
        t_adj = ctl.end(z, products=1)
        ctl.timing["adjoint_seconds"] = t_adj
        base["arithmetic"] = str(np.dtype(z.dtype))
        override = hook.get("support") if hook else None
        if override is None:
            mask = z > 0.0
            base["support_source"] = "returned dual: {z > 0}, z = -K^T y - c"
        else:
            host = np.asarray(override, bool).reshape(-1)
            if host.shape != tuple(z.shape):
                raise ValueError("support override has the wrong shape")
            mask = jnp.asarray(host)
            base["support_source"] = "override (test hook)"
        n_s = int(jnp.count_nonzero(mask))
        base["support_size"] = n_s
        if n_s == 0:
            base.update(withdrawn="empty support", cg_iterations=0, seconds=time.perf_counter() - t0)
            return
        # ---- admission, before the correction is constructed
        host_need = host_workspace_bytes(ROUTE_ACTIVE_SUPPORT_SOLVER, n, m, n_s,
                                         *((op.N, op.Mc) if marginal else (None, None)))
        host_ok, host_how = admitted_host_bytes()
        extra = [{"name": "workspace_host", "estimate": int(host_need), "limit": host_ok,
                  "ok": host_need == 0 or host_ok is None or host_need <= host_ok,
                  "detail": ("the support graph needs about %d host bytes; admitted %s (%s)"
                             % (host_need, host_ok, host_how))}]
        has_sq = hasattr(op, "sq_matvec")
        allowance = _admit(ctl, support=n_s, n=n, rows=m,
                           fixed_products=1 + (1 if has_sq else 0) + 1, rows_rule=False,
                           extra_checks=extra, products_before=1)
        X_pred = jnp.where(mask, z, 0.0) / mu
        kfwd = ctl.compiled(("fwd", n), "forward product (float64)", lambda v: jnp.asarray(K(v)), X_pred)
        ctl.begin("forward product", products=1, projected=ctl.timing.get("forward_seconds"))
        r = kfwd(X_pred) + l                              # primal residual = grad of d(y)
        t_fwd = ctl.end(r, products=1)
        ctl.timing["forward_seconds"] = t_fwd
        if marginal:
            ctl.begin("host: support graph components")
            k, lab_u, lab_v, size = _marginal_gauge(op, mask, base)
            ctl.end(lab_u, lab_v, size)
            N = op.N
            aux = (mask, lab_u, lab_v, size)

            def proj(v):
                return _gauge_project(v, lab_u, lab_v, size, k, N)

            def apply(w, a):
                v = jnp.asarray(K(jnp.where(a[0], jnp.asarray(KT(w)), 0.0)))
                return _gauge_project(v, a[1], a[2], a[3], k, N)
            key = ("as", n, m, "gauge", k)
        else:
            def proj(v):
                return v
            base["nullspace"] = {"handling": "none (unknown operator nullspace; unprojected CG)"}
            aux = (mask,)

            def apply(w, a):
                return jnp.asarray(K(jnp.where(a[0], jnp.asarray(KT(w)), 0.0)))
            key = ("as", n, m, "plain")
        rhs = proj(mu * r)
        base["rhs_norm"] = float(jnp.linalg.norm(rhs))
        if has_sq:
            ctl.begin("forward product (squared entries)", products=1, projected=t_fwd)
            diag = jnp.asarray(op.sq_matvec(mask.astype(jnp.float64)))
            ctl.end(diag, products=1)
            minv = jnp.where(diag > 0, 1.0 / jnp.where(diag > 0, diag, 1.0), 1.0)
            base["jacobi"] = "diag(K^{o2} 1_S) through sq_matvec"
        else:
            minv = jnp.ones_like(rhs)
            base["jacobi"] = None
        dy, lin = _pcg_bounded(ctl, key, apply, rhs, minv, aux, allowance=allowance,
                               max_iters=b.max_cg_iters, rtol=b.cg_rtol, t_iter=t_fwd + t_adj,
                               reserve=t_adj + ctl.assessment_seconds())
        dy = proj(dy)
        base["linear_solve"] = lin
        base["direction_norm"] = float(jnp.linalg.norm(dy))
        zp = jnp.maximum(z, 0.0)
        d0 = float(l @ y) - float(zp @ zp) / (2.0 * mu)
        slope = float(r @ dy)
        base["dual_value_before"] = d0
        base["directional_derivative"] = slope
        del zp, X_pred
        t = 1.0
        for trial in range(int(b.max_trials)):
            yt = y + t * dy
            ctl.begin("line-search trial: adjoint product", products=1,
                      projected=t_adj + ctl.assessment_seconds())
            zt = -kadj(yt) - c
            ctl.end(zt, products=1)
            ztp = jnp.maximum(zt, 0.0)
            dt = float(l @ yt) - float(ztp @ ztp) / (2.0 * mu)
            armijo = bool(dt >= d0 + 1e-4 * t * slope)
            signs = {"n_z_nonpositive_on_S": int(jnp.count_nonzero(mask & (zt <= 0.0))),
                     "n_z_positive_off_S": int(jnp.count_nonzero((~mask) & (zt > 0.0)))}
            yp = jnp.asarray(yt, jnp.float32)
            for kind in ("restricted", "full"):
                X = jnp.where(mask, ztp, 0.0) / mu if kind == "restricted" else ztp / mu
                info = dict(base)
                # Everything executed since the previous proposal is charged on
                # this one: the setup and the linear solve on the first
                # candidate, each trial's dual evaluation on its first.
                products, cg = ctl.take()
                info.update(trial=trial, step_length=t, candidate=kind, dual_value_after=dt,
                            dual_increase=dt - d0, armijo=armijo, signs_after=signs,
                            products=products, cg_iterations=cg,
                            seconds=time.perf_counter() - t0)
                yield jnp.asarray(X, jnp.float32), yp, info
            if armijo:
                return
            t *= 0.5
        if ctl.bound is None:
            ctl.bound = {"limit": "max_trials", "segment": "line search",
                         "detail": "no Armijo-accepted step within RefineBudget.max_trials = %d" % b.max_trials,
                         "then": "every trial's proposals were assessed"}


# ---------------------------------------------------------------------------
# Assessment through the UNCHANGED criterion, exactly as the controller does.
# ---------------------------------------------------------------------------
def _assess(comp, x, y, dense):
    from ..model.compile import _gap_accepts
    ob = comp._objective_bound(x, y, dense, comp._bound_rounds, return_dual=True)
    available, certified, gap = comp._gap_fields(ob)
    infeasibility = comp._infeasibility(ob)
    primal = float(ob.get("objective_at_returned_point",
                          ob.get("objective_primal_value", float("nan"))))
    accepts = bool(_gap_accepts(available, certified, gap, infeasibility, primal, comp._tol))
    if available and certified and math.isfinite(gap) and math.isfinite(infeasibility):
        score = max(abs(gap), infeasibility)
    else:
        score = math.inf
    if accepts:
        why = "passes criterion (7) at tol %.3e" % comp._tol
    elif not available:
        why = "no valid lower bound at the proposal: " + str(ob.get("unavailable_reason", ""))[:200]
    elif not certified:
        why = "the proposal's primal value is not finite (outside a block domain)"
    elif not (infeasibility <= comp._tol):
        why = "relative primal infeasibility %.6e exceeds tol %.3e" % (infeasibility, comp._tol)
    else:
        why = "objective discrepancy %.6e does not meet tol %.3e" % (gap, comp._tol)
    return {"ob": ob, "available": available, "certified": certified, "gap": gap,
            "infeasibility": infeasibility, "primal": primal, "accepts": accepts,
            "score": score, "why": why,
            "summary": {"objective_lower_bound": float(ob.get("objective_lower_bound", float("nan"))),
                        "objective_at_returned_point": primal,
                        "objective_gap": (gap if certified else None),
                        "relative_primal_infeasibility": infeasibility,
                        "witness_source": ob.get("source", "unavailable"),
                        "operator_applications_upper_bound": int(ob.get("operator_applications", 0)),
                        "target_met": accepts}}


def _refined_solution(comp, previous, xr, yr, ob, d):
    """The Solution for an accepted refined pair, built exactly as the
    controller builds its own: residual at the pair, status classification,
    the shared award, then the certifying-pair binding."""
    from ..model.compile import Solution
    check = comp._returned_components or comp._report
    if check is None:
        return None, "no returned-pair residual evaluator"
    checked = check(xr, yr, d)
    res = float(jnp.maximum(checked["r_primal"], checked["r_dual"]))
    status, cert = comp._status(xr, yr, d, previous.iters // max(1, int(comp.meta["block"])), res)
    if status in ("primal_infeasible", "unbounded", "numerical_error"):
        return None, "the refined pair classified as %s" % status
    s = Solution(xr, yr, previous.iters, res, status=status, certificate=cert,
                 accuracy_criterion="objective_gap", gap_history=previous.gap_history,
                 **comp._facts(xr, yr, d, status, previous.iters, res))
    awarded = comp._award(s, ob, previous.gap_history, stalled=False)
    if awarded.status != "solved":
        return None, "the shared award refused the pair: " + awarded.accuracy_detail[:200]
    awarded = _dc_replace(awarded, termination="criterion")
    return comp._return_certifying_pair(awarded, ob, d), None


def _started_from(result, stalled, trigger):
    return {"iterations": int(result.iters), "status": result.status,
            "termination": result.termination, "stall_guard_fired": bool(stalled),
            "trigger": trigger,
            "objective_gap": result.objective_gap,
            "relative_primal_infeasibility": result.relative_primal_infeasibility,
            "residual": result.residual,
            "objective_lower_bound": result.objective_lower_bound,
            "objective_primal_value": result.objective_primal_value,
            "dual_source": result.dual_source,
            "witness_source": result.objective_bound_source,
            "checks": len(result.gap_history or [])}


def _zero_work():
    return {"operator_applications": 0, "correction_operator_applications": 0,
            "assessment_operator_applications_upper_bound": 0, "cg_iterations": 0,
            "assessments": 0, "proposals": 0, "trials": 0, "pdhg_iterations": 0,
            "seconds": 0.0, "arithmetic": None}


def _scalar_info(info):
    out = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = _scalar_info(v)
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif hasattr(v, "shape") and np.asarray(v).shape == ():
            out[k] = float(v)
    return out


def _record(result, stalled, trigger, batch_lane, outcome, route=None, reason=None):
    rec = {"policy": POLICY, "trigger": trigger, "route": route, "operation": route,
           "outcome": outcome, "reason": reason, "batch_lane": batch_lane,
           "exit": EXIT_NO_STAGE,
           "started_from": _started_from(result, stalled, trigger),
           "work": _zero_work(), "distinct_operations": {
               WITNESS_RECOVERY: "the assessment's dual recovery; not performed here",
               ROUTE_PRIMAL_DUAL_REFINEMENT: "fixed-support primal-dual correction (FFT lasso shape)",
               ROUTE_ACTIVE_SUPPORT_SOLVER: "active-support Newton correction (quadratic OT shape)"}}
    return rec


def _run_stage(comp, route, result, d, dense, x_iterate, y_iterate, *, stalled, trigger,
               batch_lane, clock_start=None):
    budget = comp._refine_budget if comp._refine_budget is not None else RefineBudget()
    rec = _record(result, stalled, trigger, batch_lane, outcome=None, route=route)
    rec["budget"] = asdict(budget)
    entered = time.perf_counter()
    t0 = entered if clock_start is None else float(clock_start)
    rec["deadline"] = {"max_seconds": budget.max_seconds,
                       "clock_started": ("refine_assessed entry (the assessment of the supplied pair counts)"
                                         if clock_start is not None else "stage entry"),
                       "seconds_before_stage_entry": entered - t0}
    work = rec["work"]
    n, m = comp.n, comp.m
    need = workspace_bytes(route, n, m)
    admitted, how = admitted_memory_bytes(comp._refine_memory_bytes)
    if budget.max_extra_bytes is not None and (admitted is None or budget.max_extra_bytes < admitted):
        admitted, how = int(budget.max_extra_bytes), ("RefineBudget.max_extra_bytes"
                                                      + ("" if admitted is None else " (below %s)" % how))
    rec["memory"] = {"workspace_bytes_estimate": int(need), "admitted_bytes": admitted,
                     "admitted_from": how, "admitted": bool(admitted is not None and need <= admitted)}
    if not rec["memory"]["admitted"]:
        rec["outcome"] = "skipped: workspace does not fit the admitted memory budget"
        rec["reason"] = ("the stage needs about %d bytes of extra workspace; admitted %s (%s)"
                         % (need, admitted, how))
        rec["exit"] = EXIT_ADMISSION_REFUSAL
        rec["admission"] = {"admitted": False, "products_before_admission": 0, "checks": [
            {"name": "workspace_device", "estimate": int(need), "limit": admitted, "ok": False,
             "detail": rec["reason"]}]}
        rec["refusal"] = {"exit": EXIT_ADMISSION_REFUSAL, "name": "workspace_device",
                          "estimate": int(need), "limit": admitted, "detail": rec["reason"]}
        work["seconds"] = time.perf_counter() - entered
        return _dc_replace(result, refinement=rec)
    rec.setdefault("admission", {"checks": [], "admitted": None})["checks"].append(
        {"name": "workspace_device", "estimate": int(need), "limit": admitted, "ok": True,
         "detail": "the stage needs about %d bytes of extra device workspace; admitted %s (%s)"
                   % (need, admitted, how)})
    hook = comp._refine_test_hook or {}
    structure = comp._refine_structure
    ctl = _Control(comp, route, budget, t0, work, admitted, rec)
    rows = []
    accepted = None
    refusal = None
    stop = None
    admission = None
    failed = False
    best_score = math.inf
    gen = None

    def over_budget():
        if time.perf_counter() - t0 > budget.max_seconds:
            return "max_seconds"
        if work["operator_applications"] >= budget.max_operator_applications:
            return "max_operator_applications"
        if work["assessments"] >= budget.max_assessments:
            return "max_assessments"
        return None

    def stop_here(limit, segment):
        return {"limit": limit, "segment": segment, "detail": "checked at the boundary %s" % segment,
                "projected_seconds": None, "elapsed_seconds": ctl.elapsed(),
                "then": "no further proposal assessed"}

    try:
        # The correction is built inside the transaction: a route that raises
        # while setting up (or a hooked proposal that raises) is a refused
        # correction, never a failure of the solve.
        if "proposals" in hook:
            gen = hook["proposals"](x_iterate, y_iterate)
            rec["proposals_from"] = "test hook"
        elif route == ROUTE_PRIMAL_DUAL_REFINEMENT:
            gen = _propose_primal_dual(structure, x_iterate, dense, budget, hook, ctl)
        else:
            gen = _propose_active_support(structure, x_iterate, y_iterate, dense, budget, hook, ctl)
        for cand in gen:
            reason = over_budget()
            if reason:
                refusal = "over budget (%s) before the next proposal" % reason
                stop = stop_here(reason, "before the next proposal")
                break
            xc, yc, info = cand
            info = dict(info or {})
            work["proposals"] += 1
            work["correction_operator_applications"] += int(info.get("products", 0))
            work["operator_applications"] += int(info.get("products", 0))
            work["cg_iterations"] += int(info.get("cg_iterations", 0))
            if "trial" in info:
                work["trials"] = max(work["trials"], int(info["trial"]) + 1)
            if info.get("arithmetic"):
                work["arithmetic"] = info["arithmetic"]
            row = {"proposal": _scalar_info(info)}
            valid = (xc is not None and yc is not None
                     and tuple(np.asarray(xc).shape) == (n,) and tuple(np.asarray(yc).shape) == (m,)
                     and bool(jnp.all(jnp.isfinite(xc))) and bool(jnp.all(jnp.isfinite(yc))))
            if not valid:
                row["decision"] = "refused: nonfinite or invalid proposal (not assessed)"
                rows.append(row)
                continue
            reason = over_budget()
            if reason:
                row["decision"] = "not assessed: over budget (%s)" % reason
                rows.append(row)
                refusal = "over budget (%s)" % reason
                stop = stop_here(reason, "before the assessment")
                break
            # The assessment is one bounded segment (one call of the evaluator),
            # admitted against the deadline and the operator budget.
            try:
                ctl.begin("assessment", products=_assessment_ops_estimate(comp, ctl),
                          projected=ctl.timing.get("assessment_seconds"))
            except _Stop as e:
                row["decision"] = "not assessed: over budget (%s)" % e.limit
                rows.append(row)
                refusal = "over budget (%s)" % e.limit
                stop = e.as_dict(ctl.elapsed())
                break
            a = _assess(comp, xc, yc, dense)
            ops = int(a["ob"].get("operator_applications", 0))
            work["assessments"] += 1
            work["assessment_operator_applications_upper_bound"] += ops
            work["operator_applications"] += ops
            row["assessment"] = a["summary"]
            row["score"] = a["score"]
            best_score = min(best_score, a["score"])
            try:
                ctl.timing["assessment_seconds"] = ctl.end()
            except _Stop as e:                            # the memory check after the segment
                row["decision"] = "assessed, then stopped at the boundary (%s)" % e.limit
                rows.append(row)
                raise
            ctl.timing["assessment_operator_applications"] = ops
            if a["accepts"]:
                ctl.begin("returned pair: residual, status and award", check=False)
                sol, why = _refined_solution(comp, result, xc, yc, a["ob"], d)
                work["operator_applications"] += 2        # the returned-pair residual
                try:
                    ctl.end(*(() if sol is None else (sol.x, sol.y)))
                except _Stop as e:
                    row["decision"] = "passed, then stopped at the boundary (%s)" % e.limit
                    rows.append(row)
                    raise
                if sol is None:
                    row["decision"] = "refused: " + why
                    rows.append(row)
                    continue
                row["decision"] = "accepted: " + a["why"]
                rows.append(row)
                accepted = sol
                rec["assessment"] = a["summary"]
                break
            row["decision"] = "refused: " + a["why"]
            rows.append(row)
    except _Admission as e:
        admission = e
    except _Stop as e:
        refusal = "over budget (%s) before the next proposal" % e.limit
        stop = e.as_dict(ctl.elapsed())
    except Exception as e:                                # a failing correction is refused, not fatal
        failed = True
        refusal = "the correction raised %s" % repr(e)[:300]
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1][:300]
    finally:
        if gen is not None and hasattr(gen, "close"):
            try:
                gen.close()                               # leaves the route's scoped x64 now
            except Exception:                             # pragma: no cover - defensive
                pass
    # Work executed but not yet charged on a proposal (a stop, a refusal or a
    # withdrawal inside the route) is charged here.
    products, cg = ctl.take()
    work["correction_operator_applications"] += products
    work["operator_applications"] += products
    work["cg_iterations"] += cg
    rec["candidates"] = rows
    rec["best_score_seen"] = (None if best_score is math.inf else best_score)
    rec["execution"] = ctl.record()
    rec["memory"].update(ctl.memory.record())
    now = time.perf_counter()
    work["seconds"] = now - entered
    rec["deadline"]["elapsed_seconds"] = now - t0
    rec["deadline"]["within_deadline"] = bool(now - t0 <= budget.max_seconds)
    if admission is not None:
        rec["outcome"] = "skipped: ineligible (%s)" % admission.name
        rec["reason"] = admission.detail + "; the previous assessed packet is returned unchanged"
        rec["exit"] = EXIT_ADMISSION_REFUSAL
        rec["refusal"] = {"exit": EXIT_ADMISSION_REFUSAL, "name": admission.name,
                          "estimate": admission.estimate, "limit": admission.limit,
                          "detail": admission.detail, "also_failed": admission.also}
        return _dc_replace(result, refinement=rec)
    if accepted is not None:
        rec["outcome"] = "accepted"
        rec["reason"] = "the refined pair passes the unchanged criterion (7); it is the returned pair"
        rec["y_iterate_source"] = "refined dual proposal"
        if ctl.bound is not None:
            rec["exit"] = EXIT_BOUNDED_STOP
            rec["bounded_stop"] = ctl.bound
        else:
            rec["exit"] = EXIT_COMPLETED
        return _dc_replace(accepted, refinement=rec)
    if refusal is None:
        refusal = ("the correction was withdrawn (no proposal)" if not rows else
                   "no proposal passed criterion (7)")
    rec["outcome"] = "refused: " + refusal
    rec["reason"] = "the previous assessed packet is returned unchanged"
    if failed:
        rec["exit"] = EXIT_FAILED
    elif stop is not None:
        rec["exit"] = EXIT_BOUNDED_STOP
        rec["bounded_stop"] = stop
        rec["refusal"] = {"exit": EXIT_BOUNDED_STOP, "name": stop["limit"], "segment": stop["segment"],
                          "detail": stop["detail"]}
    elif ctl.bound is not None:
        rec["exit"] = EXIT_BOUNDED_STOP
        rec["bounded_stop"] = ctl.bound
    else:
        rec["exit"] = EXIT_COMPLETED
    return _dc_replace(result, refinement=rec)


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------
def apply_policy(comp, result, d, dense, *, stalled, x_iterate, y_iterate, batch_lane=None,
                 trigger="stall_guard", clock_start=None):
    """Apply the ``refine="auto"`` table to one assessed packet."""
    if result.status == "solved":
        rec = _record(result, stalled, trigger, batch_lane,
                      outcome="not_needed",
                      reason="the assessed pair passes criterion (7); no refinement work")
        return _dc_replace(result, refinement=rec)
    bound_available = (result.objective_lower_bound is not None
                       and math.isfinite(float(result.objective_lower_bound))
                       and result.accuracy_criterion == "objective_gap")
    if not bound_available:
        missing = comp.meta.get("objective_bound_missing_contracts", [])
        rec = _record(result, stalled, trigger, batch_lane, outcome="unavailable",
                      reason=("no criterion-(7) assessment at the returned point; refinement "
                              "never manufactures one: " + (result.accuracy_detail or "")[:300]))
        rec["capability_missing"] = list(missing)
        return _dc_replace(result, refinement=rec)
    if trigger == "stall_guard" and not stalled:
        rec = _record(result, stalled, trigger, batch_lane, outcome="not_triggered",
                      reason=("the iteration ended by %r, not by the stall guard; the fixed "
                              "trigger did not fire" % result.termination))
        return _dc_replace(result, refinement=rec)
    route, why = detect_route(comp._refine_structure)
    if route is None:
        rec = _record(result, stalled, trigger, batch_lane, outcome="unsupported", reason=why)
        return _dc_replace(result, refinement=rec)
    return _run_stage(comp, route, result, d, dense, x_iterate, y_iterate,
                      stalled=stalled, trigger=trigger, batch_lane=batch_lane,
                      clock_start=clock_start)


def record_unavailable(comp, result):
    """The missing-contract row: report the capability, do no work."""
    missing = comp.meta.get("objective_bound_missing_contracts", [])
    rec = _record(result, False, "stall_guard", None, outcome="unavailable",
                  reason=("objective_gap is unsupported for this family (no exact conjugate/value "
                          "contract); refinement never manufactures an assessment"))
    rec["capability_missing"] = list(missing)
    return _dc_replace(result, refinement=rec)


def refine_assessed(comp, x, y, data=None):
    """Assess a supplied pair and run the one stage on request (no PDHG).
    The stage's deadline is counted from this call's entry."""
    from ..model.compile import Solution
    t_call = time.perf_counter()
    d = comp._resolve(data or {})
    dense = comp._resolve_dense(data or {})
    x = jnp.asarray(x, jnp.float32)
    y = jnp.asarray(y, jnp.float32)
    if x.shape != (comp.n,) or y.shape != (comp.m,):
        raise ValueError("pair has incompatible primal or dual shape")
    if not comp._bound_capable or comp._objective_bound is None:
        return record_unavailable(comp, comp._unsupported_gap((x, y)))
    check = comp._returned_components or comp._report
    checked = check(x, y, d)
    res = float(jnp.maximum(checked["r_primal"], checked["r_dual"]))
    status, cert = comp._status(x, y, d, 0, res)
    s = Solution(x, y, 0, res, status=status, certificate=cert,
                 accuracy_criterion="objective_gap", gap_history=[],
                 **comp._facts(x, y, d, status, 0, res))
    if status in ("primal_infeasible", "unbounded", "numerical_error"):
        return s
    ob = comp._objective_bound(x, y, dense, comp._bound_rounds, return_dual=True)
    awarded = comp._award(s, ob, [])
    # Zero PDHG budget: "iteration_limit" unless the pair already passes.
    awarded = _dc_replace(awarded, termination=("criterion" if awarded.status == "solved"
                                                else "iteration_limit"))
    result = comp._return_certifying_pair(awarded, ob, d)
    return apply_policy(comp, result, d, dense, stalled=False, x_iterate=x, y_iterate=y,
                        trigger="requested", clock_start=t_call)


# ---------------------------------------------------------------------------
# The external watchdog (exit 3): a hard process-level wall limit that the
# library offers but never relies on.
# ---------------------------------------------------------------------------
def solve_with_refinement(comp, data=None):
    """Explicit fresh first-order budget plus one bounded requested stage.

    The original termination is never relabelled as a stall. Unavailable
    assessments and solver faults cannot invoke correction. The selected
    compiled RefineBudget controls the stage; callers must impose a process
    deadline for compilation/segments which Python cannot interrupt.
    """
    started = time.perf_counter()
    first = comp.solve_instance(data, accuracy="objective_gap", refine=False)
    jax.block_until_ready((first.x, first.y))
    first_seconds = time.perf_counter() - started
    eligible = (first.status == "inaccurate"
                and first.termination in ("iteration_limit", "stalled")
                and first.accuracy_criterion == "objective_gap"
                and first.objective_lower_bound is not None
                and math.isfinite(float(first.objective_lower_bound)))
    stage = None
    stage_seconds = 0.0
    if eligible:
        stage_start = time.perf_counter()
        # Refine the exact selected public pair. The unreturned continuation
        # dual remains separate; it must not silently replace that witness.
        stage = comp.refine_assessed(first.x, first.y, data)
        jax.block_until_ready((stage.x, stage.y))
        stage_seconds = time.perf_counter() - stage_start
    accepted = stage is not None and stage.status == "solved"
    selected = stage if accepted else first
    record = dict(stage.refinement or {}) if stage is not None else {}
    record.update({
        "combined_policy": "fresh_then_one_bounded_refinement_v1",
        "combined_trigger": first.termination if eligible else "ineligible_or_already_solved",
        "first_order": {"status": first.status, "termination": first.termination,
                        "iterations": int(first.iters), "seconds": first_seconds,
                        "objective_gap": first.objective_gap,
                        "relative_primal_infeasibility": first.relative_primal_infeasibility,
                        "objective_lower_bound": first.objective_lower_bound},
        "stage_invoked": eligible, "stage_call_seconds": stage_seconds,
        "stage_input": "exact first-order returned x and certifying y",
        "selected_phase": "refinement" if accepted else "first_order",
        "total_call_seconds": time.perf_counter() - started,
    })
    return _dc_replace(selected, iters=int(first.iters) + (int(stage.iters) if accepted else 0),
                       gap_history=first.gap_history, refinement=record)


_WATCHDOG_CHILD = r'''
import json, os, select, signal, sys, time
pid, seconds, marker = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
armed = time.time()
ready, _, _ = select.select([sys.stdin], [], [], seconds)
if ready:
    sys.exit(0)        # disarmed: the caller left the block (or exited) and the pipe closed
record = {"exit": "external_watchdog", "verdict": None, "limit_seconds": seconds, "pid": pid,
          "armed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(armed)),
          "fired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "note": "the process was killed by the external watchdog; no numerical verdict was produced"}
if marker:
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, marker)
sys.stderr.write("ocs refine watchdog: " + json.dumps(record) + "\n")
sys.stderr.flush()
os.kill(pid, signal.SIGKILL)
'''


@contextlib.contextmanager
def process_watchdog(seconds, *, marker_path=None):
    """Kill THIS process with SIGKILL if the block has not exited after
    ``seconds`` (POSIX).

    A separate child process waits on a pipe; leaving the block (or the
    caller exiting) closes the pipe and disarms it.  When it fires it writes
    ``marker_path`` (if given) with ``{"exit": "external_watchdog",
    "verdict": null, ...}`` and kills the caller, which therefore returns no
    ``Solution``: record such a row as interrupted with no numerical verdict.
    The stage's own deadline (``RefineBudget.max_seconds``) is the bound the
    library relies on; this is the backstop for a segment it cannot
    interrupt (one compiled call or one compilation)."""
    seconds = float(seconds)
    if not (seconds > 0) or math.isinf(seconds):
        raise ValueError("process_watchdog needs a positive finite number of seconds")
    proc = subprocess.Popen([sys.executable, "-I", "-c", _WATCHDOG_CHILD, str(os.getpid()),
                             repr(seconds), marker_path or ""],
                            stdin=subprocess.PIPE, close_fds=True)
    try:
        yield proc
    finally:
        try:
            proc.stdin.close()
        except Exception:                                 # pragma: no cover - defensive
            pass
        try:
            proc.wait(timeout=10)
        except Exception:                                 # pragma: no cover - defensive
            proc.kill()
