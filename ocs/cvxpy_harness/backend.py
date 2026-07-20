"""CVXPY validation harness -- NOT part of OCS.

A thin, separate bridge used ONLY to validate OCS on small LP / QP / SOCP
instances against a mature solver (SCS).  It imports an unmodified CVXPY,
canonicalizes the problem to SCS cone form

    minimize   c^T x
    subject to A x + s = b,   s in K = (zero, nonneg, second-order),

wraps that explicit cone data as an OCS problem family (A is materialized -- fine,
these are small validation instances), solves with OCS, and compares objectives.
It never touches the OCS core and does not define the public API.

Supported cones: zero (equality), nonnegative, second-order.  PSD / exp / power
raise NotImplementedError here (they exist as OCS blocks; wiring them into the
translation is future work).
"""
from __future__ import annotations

import numpy as np

from ..ir import Term, ProblemFamily
from ..operators.common import Dense
from ..prox.library import Equality, NonNeg
from ..prox.cones import SecondOrderCone


def _const_offset(b_rows):
    b = np.asarray(b_rows, np.float32)
    return lambda data, b=b: b


def validate_against_cvxpy(problem, solver=None, max_blocks=2000,
                           equilibrate="diagonal"):
    """Solve ``problem`` (a cvxpy Problem) with OCS and with SCS; return a report
    dict with both objectives and their relative difference."""
    import cvxpy as cp

    data, _chain, _inv = problem.get_problem_data(cp.SCS)
    A = np.asarray(data["A"].todense(), np.float32)     # (m, n): A x + s = b
    b = np.asarray(data["b"], np.float32)
    c = np.asarray(data["c"], np.float32)
    dims = data["dims"]
    n = A.shape[1]

    for attr in ("psd", "exp", "p"):
        if getattr(dims, attr, None):
            raise NotImplementedError(f"cone '{attr}' not yet wired in the harness")

    # s = b - A x in K  <=>  indicator_K( (-A) x + b )
    terms, row = [], 0

    def add(nrows, block):
        nonlocal row
        op = Dense(-A[row:row + nrows])
        terms.append(Term(op=op, block=block, offset_fn=_const_offset(b[row:row + nrows])))
        row += nrows

    if dims.zero:
        add(dims.zero, Equality())
    if dims.nonneg:
        add(dims.nonneg, NonNeg())
    for q in dims.soc:
        add(q, SecondOrderCone())

    family = ProblemFamily(n=n, terms=terms, cost_fn=lambda d, c=c: c)
    compiled = family.compile(max_blocks=max_blocks, equilibrate=equilibrate)
    sol = compiled.solve_instance({})
    x = np.asarray(sol.x)
    ocs_obj = float(c @ x)

    problem.solve(solver=solver or cp.SCS)
    ref_obj = float(problem.value)

    return {
        "ocs_obj": ocs_obj,
        "ref_obj": ref_obj,
        "rel_err": abs(ocs_obj - ref_obj) / (1.0 + abs(ref_obj)),
        "residual": sol.residual,
        "cert": compiled.certificate(sol.x, sol.y, {}),
    }
