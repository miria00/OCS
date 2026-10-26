"""Product-only NUMERICAL stopping for the quadratic-ball family, float32.

WHAT THIS IS, AND WHAT IT IS EXPLICITLY NOT
-------------------------------------------
For

    minimize  (1/2) x^T H x + c^T x    subject to   ||A x||_2 <= r,

with ``H`` a positive diagonal, and for any ``v``, writing ``q = c + A^T v``,

    F(x) - l(v) = (1/2) || H^{1/2} x + H^{-1/2} q ||^2 + r ||v|| - v^T A x,

both terms nonnegative at a feasible ``x``.  In EXACT arithmetic that is a true
primal-dual gap.  Evaluated in float32 it is a NUMERICAL GAP EVALUATION, and this
module never calls it anything else.

It is NOT the rigorous bound.  ``ocs.solver.bound_authorization`` remains the only
authority for a certified lower bound, it keeps its own access requirements and
its own cost, and nothing here writes into its fields or reclassifies it.  The
two are separate modes:

    T_production        = T_construct + T_solve_and_numerical_stopping
    T_with_audit        = T_production + T_audit

and an audit's cost is never subtracted from a verified-return measurement.

WHY IT EXISTS
-------------
The rigorous route rebuilds the operator's action from archived ``Dense``
parameters and refuses an operator that is only a pair of callables.  That is the
honest consequence of its own contract, and it means the installed certified
bound reads entries.  This evaluation does not: it needs

    one adjoint product     A^T v,
    one forward product     A x,

and the objective's own diagonal curvature ``H``, which the model declares.  It
needs no eigensystem of ``A^T A``, no exact-zero stationarity test and no entry
of ``A``.

WHAT IS REPORTED, SEPARATELY
----------------------------
The raw point and the restored point are reported distinctly, and the restoration
is the radial one the family admits, ``min(1, r / M) x`` with ``M`` an upper
bound on ``||A x||``, which is valid because the ball is homogeneous.  A negative
or inconsistent evaluation is REPORTED as such and never clamped to zero: an
apparent zero gap obtained by clipping is the failure this whole line of work
exists to avoid.
"""
from __future__ import annotations

import numpy as np

__all__ = ["numerical_gap", "radial_restore", "BallNumericalStop"]


def radial_restore(matvec, x, r, *, count=None):
    """``x_cert = min(1, r / M) x`` with ``M`` an upper bound on ``||A x||``.

    One forward product.  The ball is homogeneous, so scaling toward the origin
    can only help feasibility, and the bound on ``M`` is charged upward so the
    scale is not an underestimate.
    """
    Ax = np.asarray(matvec(x), np.float64)
    if count is not None:
        count["forward"] = count.get("forward", 0) + 1
    m = float(np.linalg.norm(Ax))
    terms = int(Ax.size)
    # charge the norm's own summation, so M is an upper bound and not a value
    unit = float(np.finfo(np.float32).eps)
    m_up = m * (1.0 + unit * (terms + 2))
    scale = 1.0 if m_up <= r else r / m_up
    return np.asarray(x, np.float64) * scale, {"norm_Ax": m, "norm_Ax_upper": m_up,
                                               "scale": scale}


def numerical_gap(matvec, rmatvec, c, h_diag, v, x, r=1.0, *, count=None):
    """The gap expression above, evaluated through PRODUCTS ONLY.

    ``h_diag`` is the objective's declared diagonal curvature, strictly positive.
    ``v`` is any dual point, ``x`` the point the gap is reported at.

    Returns a dict.  ``gap`` is the numerical evaluation, ``stationarity`` and
    ``support`` its two parts, and ``parts_nonnegative`` says whether both came
    out nonnegative, because in exact arithmetic both must be and a negative one
    is information rather than something to clip away.
    """
    c = np.asarray(c, np.float64)
    h = np.asarray(h_diag, np.float64)
    v = np.asarray(v, np.float64)
    x = np.asarray(x, np.float64)
    if not np.all(h > 0):
        raise ValueError("numerical_gap needs a strictly positive diagonal H")

    q = c + np.asarray(rmatvec(v), np.float64)
    if count is not None:
        count["adjoint"] = count.get("adjoint", 0) + 1
    Ax = np.asarray(matvec(x), np.float64)
    if count is not None:
        count["forward"] = count.get("forward", 0) + 1

    root_h = np.sqrt(h)
    stationarity = 0.5 * float(np.sum((root_h * x + q / root_h) ** 2))
    support = r * float(np.linalg.norm(v)) - float(v @ Ax)
    objective = 0.5 * float(np.sum(h * x * x)) + float(c @ x)
    lower = objective - (stationarity + support)

    # ---- the acceptance control ---------------------------------------------
    # Reporting a negative support term is necessary and is NOT sufficient. In
    # exact arithmetic at a feasible point both parts are nonnegative, so a
    # negative one means the hypothesis is not met. If it then CANCELS against
    # the positive squared-defect term the total can look small, and a policy
    # reading only the total would accept on an inconsistency.
    #
    # So the inconsistency is measured against the terms' own scale rather than
    # against a fixed number, and the scale is the largest magnitude that entered
    # the sum. A discrepancy at the rounding level of that scale is rounding, and
    # anything larger is a real inconsistency. This rule is stated here, before
    # any case was observed, and is not adjusted afterwards.
    scale = max(abs(stationarity), abs(support), 1e-300)
    unit = float(np.finfo(np.float32).eps)
    rounding = unit * scale * (int(x.size) + int(v.size) + 2)
    negative = min(stationarity, support)
    inconsistent = bool(negative < -rounding)
    finite = bool(np.isfinite(stationarity) and np.isfinite(support)
                  and np.isfinite(objective))
    return {
        "gap": stationarity + support,
        "stationarity": stationarity,
        "support": support,
        "objective": objective,
        "implied_lower_value": lower,
        "parts_nonnegative": bool(stationarity >= 0.0 and support >= 0.0),
        # the two fields a stopping policy must read, and why they are separate
        # from the gap: a small gap obtained by cancellation is not a small gap
        "acceptable": bool(finite and not inconsistent),
        "inconsistent": inconsistent,
        "finite": finite,
        "negative_part": negative,
        "rounding_allowance": rounding,
        "cancellation_ratio": float(abs(negative) / max(abs(stationarity + support),
                                                        1e-300)),
        "evaluation": ("NUMERICAL, float32 solver state evaluated in host "
                       "float64. This is not a rigorous enclosure and it is not "
                       "the certified bound of ocs.solver.bound_authorization. "
                       "A policy must read 'acceptable', not 'gap' alone: a total "
                       "made small by a negative part cancelling a positive one "
                       "is not a small gap."),
    }


class BallNumericalStop:
    """The production stopping evaluator for the quadratic-ball family.

    Products only: one adjoint for ``q``, one forward for ``A x``, one more
    forward for the restoration.  Three operator applications per evaluation, and
    they are counted.

    The raw point and the restored point are BOTH reported.  Nothing here decides
    to stop: it reports, and the caller's policy decides, which keeps a numerical
    quantity out of a field that means something stronger.
    """

    def __init__(self, matvec, rmatvec, c, h_diag, r=1.0):
        self.matvec = matvec
        self.rmatvec = rmatvec
        self.c = np.asarray(c, np.float64)
        self.h = np.asarray(h_diag, np.float64)
        self.r = float(r)

    def evaluate(self, x, v):
        count = {}
        raw = numerical_gap(self.matvec, self.rmatvec, self.c, self.h, v, x,
                            self.r, count=count)
        x_cert, restoration = radial_restore(self.matvec, x, self.r, count=count)
        restored = numerical_gap(self.matvec, self.rmatvec, self.c, self.h, v,
                                 x_cert, self.r, count=count)
        feasible = bool(restoration["norm_Ax_upper"] <= self.r
                        or restoration["scale"] < 1.0)
        return {
            "raw": raw,
            "restored": restored,
            "restoration": restoration,
            "restored_point_feasible_by_construction": feasible,
            "products": {"forward": count.get("forward", 0),
                         "adjoint": count.get("adjoint", 0),
                         "total": count.get("forward", 0) + count.get("adjoint", 0)},
            "mode": "numerical stopping, product only, float32 production",
        }
