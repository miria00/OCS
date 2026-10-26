"""Small LP/QP/SOCP validation through CVXPY's explicit SCS data.

This harness materializes matrices for validation only. It preserves the
quadratic objective, constant offset and objective direction. It is separate
from the experimental matrix-free CVXPY bridge and from OCS's native API.
"""
from __future__ import annotations

import numpy as np

from ..ir import Term, ProblemFamily
from ..operators.common import Dense, Identity
from ..prox.library import Equality, NonNeg, Quadratic, Zero
from ..prox.cones import SecondOrderCone


def _const_offset(b_rows):
    b = np.asarray(b_rows, np.float32)
    return lambda data, b=b: b


#: The documented mapping from OCS's own statuses to CVXPY's.  It is a
#: mapping, not an effort to maximize the number of ``optimal`` labels.
OCS_TO_CVXPY_STATUS = {
    "solved": "optimal",                 # the accepted criterion passed
    "inaccurate": "user_limit",          # it did not
    # No accuracy was certified at all, so no accuracy is claimed.  Not
    # `optimal`, and deliberately not `optimal_inaccurate`, which would assert
    # a solution at a known-but-lower accuracy.
    "unverified": "user_limit",
    "primal_infeasible": "infeasible",
    "unbounded": "unbounded",
    "numerical_error": "solver_error",
}

#: The ONE place a declared-budget contract may change the CVXPY status, and
#: the conditions under which it may.  See ``_cvxpy_status``.
CONTRACT_STATUS_RULE = (
    "A declared-budget contract NEVER produces `optimal`: `optimal` means the "
    "strict relative criterion passed, and nothing else sets it. A point that "
    "missed the strict criterion, met an explicitly declared contract, AND "
    "stopped because a stopping criterion passed (not because a budget ran "
    "out) maps to `optimal_inaccurate`, which is CVXPY's documented code for "
    "attained accuracy lower than requested. A budget-exhausted run keeps "
    "`user_limit` whatever the contract says: a run that ran out of iterations "
    "is never relabelled a solve. With no contract declared the mapping is "
    "exactly OCS_TO_CVXPY_STATUS, unchanged.")


def _cvxpy_status(status, *, contract_certificate=None, termination=None):
    """Map OCS termination without calling a failed solve optimal.

    ``status`` is OCS's own status, the only thing that can yield ``optimal``.
    ``contract_certificate`` is the declared-budget contract's verdict: ``None``
    when no contract was declared (no test was run), otherwise a bool.
    ``termination`` says why the ITERATION stopped: ``"criterion"``,
    ``"iteration_limit"``, ``"time_limit"`` or ``"numerical_failure"``.

    The complete table, with S = strict criterion, C = declared contract,
    T = termination reason:

    A status of ``unverified`` -- no objective accuracy was certified -- maps
    to ``user_limit`` and is NEVER upgraded by the contract, because there is
    no accuracy claim for a contract to qualify.

    ====== ============ ================= ====================
    S      C            T                 CVXPY status
    ====== ============ ================= ====================
    pass   any or none  any               optimal
    fail   none         any               user_limit
    fail   False        any               user_limit
    fail   True         criterion         optimal_inaccurate
    fail   True         iteration_limit   user_limit
    fail   True         time_limit        user_limit
    any    any          numerical_failure solver_error (via status)
    ====== ============ ================= ====================

    The last two rows of the ``C = True`` block are the point: a run whose
    budget ran out is never relabelled, and a strict failure is never called a
    strict success because an extra contract was evaluated.
    """
    import cvxpy.settings as settings
    mapping = {
        "solved": settings.OPTIMAL,
        "inaccurate": settings.USER_LIMIT,
        "unverified": settings.USER_LIMIT,
        "primal_infeasible": settings.INFEASIBLE,
        "unbounded": settings.UNBOUNDED,
        "numerical_error": settings.SOLVER_ERROR,
    }
    if status not in mapping:
        raise ValueError(f"unknown OCS status: {status!r}")
    mapped = mapping[status]
    # A declared-budget contract speaks about the legacy residual.  It cannot
    # upgrade a run in which NOTHING was certified: `unverified` means no
    # accuracy claim exists for the contract to qualify.
    if (mapped == settings.USER_LIMIT and status != "unverified"
            and contract_certificate is True and termination == "criterion"):
        return settings.OPTIMAL_INACCURATE
    return mapped


def _constraint_violation(A, b, x, dims):
    """Maximum raw equality, orthant, or SOC violation in canonical units."""
    slack = b - A @ x
    if not np.all(np.isfinite(slack)):
        return float("inf")
    violation, row = 0.0, int(dims.zero)
    if row:
        violation = float(np.max(np.abs(slack[:row])))
    end = row + int(dims.nonneg)
    if end > row:
        violation = max(violation, float(np.max(-slack[row:end])))
    row = end
    for size in dims.soc:
        cone = slack[row:row + size]
        violation = max(violation, float(np.linalg.norm(cone[1:]) - cone[0]))
        row += size
    return violation


def validate_against_cvxpy(problem, solver=None, max_blocks=2000,
                           equilibrate="diagonal", *, tol=1e-3,
                           reference_options=None, objective_tolerance=1e-3):
    """Validate a small real LP, convex QP or SOCP against CVXPY.

    ``tol`` is OCS's KKT tolerance, not an objective-error guarantee.
    ``rel_err`` retains the historical normalization
    ``abs(ocs_obj-ref_obj)/(1+abs(ref_obj))``. It is NaN unless OCS is
    solved and the finite reference is optimal. Pure-relative error is
    reported separately. ``candidate_rel_err`` records the independent
    comparison even when OCS does not certify the candidate.
    Infeasible, unbounded and failed solves retain
    their statuses and cannot count as objective agreement.

    CVXPY variables contain the reference solution after this call.
    ``canonical_x`` is OCS's point in the SCS canonical variable order.
    PSD, exponential, power, complex and variable-free problems are outside
    this harness's scope. The native solver has its own broader block API.
    """
    import cvxpy as cp
    import cvxpy.settings as settings

    if not np.isfinite(objective_tolerance) or objective_tolerance < 0:
        raise ValueError("objective_tolerance must be finite and nonnegative")
    if not problem.variables() or any(v.is_complex() for v in problem.variables()):
        raise NotImplementedError("validation requires a real problem with variables")
    try:
        data, _chain, inverse_data = problem.get_problem_data(cp.SCS)
    except AttributeError as error:
        import scipy
        raise RuntimeError(
            f"CVXPY {cp.__version__} / SciPy {scipy.__version__} could not "
            "canonicalize this problem. Install compatible validation "
            "dependencies (OCS requires CVXPY >= 1.6)."
        ) from error
    if not isinstance(data, dict) or settings.OFFSET not in inverse_data[-1]:
        raise RuntimeError("unsupported CVXPY canonical objective metadata")
    A = np.asarray(data["A"].toarray(), np.float64)
    b = np.asarray(data["b"], np.float64)
    c = np.asarray(data["c"], np.float64)
    P = data.get("P")
    P = None if P is None else np.asarray(P.toarray(), np.float64)
    offset = float(np.asarray(inverse_data[-1][settings.OFFSET]).reshape(()))
    direction = -1.0 if isinstance(problem.objective, cp.Maximize) else 1.0
    dims, n = data["dims"], c.size
    for attr in ("psd", "exp", "p3d"):
        if getattr(dims, attr, None):
            raise NotImplementedError(f"cone '{attr}' not wired in the validation harness")
    if int(dims.zero) + int(dims.nonneg) + sum(dims.soc) != A.shape[0]:
        raise NotImplementedError("unrecognized canonical cone rows")

    terms, row = [], 0
    if P is not None:
        terms.append(Term(op=Identity(n), block=Quadratic(P)))

    def add(nrows, block):
        nonlocal row
        terms.append(Term(op=Dense(-A[row:row + nrows]), block=block,
                          offset_fn=_const_offset(b[row:row + nrows])))
        row += nrows

    if dims.zero:
        add(dims.zero, Equality())
    if dims.nonneg:
        add(dims.nonneg, NonNeg())
    for size in dims.soc:
        add(size, SecondOrderCone())
    if row == 0:
        # Give the compiler a full-domain dual block without changing an
        # unconstrained quadratic objective.
        terms.append(Term(op=Dense(np.eye(n)), block=Zero()))

    family = ProblemFamily(n=n, terms=terms, cost_fn=lambda d, c=c: c)
    compiled = family.compile(max_blocks=max_blocks, equilibrate=equilibrate, tol=tol)
    sol = compiled.solve_instance({})
    x = np.asarray(sol.x, np.float64)
    canonical_obj = float(c @ x + (0.0 if P is None else 0.5 * x @ P @ x))
    candidate_obj = direction * (canonical_obj + offset)
    ocs_obj = candidate_obj
    if sol.status == "primal_infeasible":
        ocs_obj = direction * float("inf")
    elif sol.status == "unbounded":
        ocs_obj = -direction * float("inf")
    elif sol.status == "numerical_error":
        ocs_obj = float("nan")

    reference_error = None
    try:
        problem.solve(solver=solver or cp.SCS, **dict(reference_options or {}))
        ref_status = problem.status
        ref_obj = float(problem.value) if problem.value is not None else float("nan")
    except cp.error.SolverError as error:
        ref_status, ref_obj = settings.SOLVER_ERROR, float("nan")
        reference_error = str(error)

    comparable = (sol.status == "solved" and ref_status == settings.OPTIMAL
                  and np.isfinite(ocs_obj) and np.isfinite(ref_obj))
    candidate_error = (abs(candidate_obj - ref_obj) / (1.0 + abs(ref_obj))
                       if ref_status == settings.OPTIMAL and
                       np.isfinite(candidate_obj) and np.isfinite(ref_obj)
                       else float("nan"))
    absolute_error = abs(ocs_obj - ref_obj) if comparable else float("nan")
    relative_error = absolute_error / (1.0 + abs(ref_obj)) if comparable else float("nan")
    pure_relative_error = float("nan")
    if comparable:
        pure_relative_error = (absolute_error / abs(ref_obj) if ref_obj else
                               0.0 if absolute_error == 0 else float("inf"))
    return {
        "ocs_obj": ocs_obj, "candidate_obj": candidate_obj, "ref_obj": ref_obj,
        "candidate_rel_err": candidate_error,
        "rel_err": relative_error, "absolute_error": absolute_error,
        "pure_relative_error": pure_relative_error,
        "objective_comparable": bool(comparable),
        "objective_agreement": bool(comparable and relative_error <= objective_tolerance),
        "ocs_status": sol.status, "reference_status": ref_status,
        "status_agreement": _cvxpy_status(sol.status) == ref_status,
        "reference_error": reference_error,
        "residual": sol.residual, "iterations": sol.iters,
        "constraint_violation": _constraint_violation(A, b, x, dims),
        # The whole certificate dict.  Its ``gap`` is the LEGACY Fenchel
        # surrogate and is not a bound; ``objective_lower_bound`` with
        # ``objective_lower_bound_available`` / ``objective_bound_status`` is
        # the strict one.  See docs/stopping_contract.md.
        "cert": compiled.certificate(sol.x, sol.y, {}),
        "cert_gap_is_legacy_surrogate": True,
        "canonical_x": x,
    }
