"""Matrix-free CVXPY bridge: canonicalize with CVXPY, solve with OCS,
never form the constraint matrix.

The seam is CVXPY's LinOp tree (built before matrix stuffing); see
``linop_walker`` for the operator emission.
"""
from .linop_walker import LinOpWalker, make_affine_forward, make_affine_operator

__all__ = ["LinOpWalker", "make_affine_forward", "make_affine_operator",
           "solve_mf", "compile_mf"]


def __getattr__(name):
    # solve/compile import cvxpy reduction machinery; load lazily so the
    # walker stays importable without cvxpy installed.
    if name in ("solve_mf", "compile_mf"):
        from . import solve as _solve
        return getattr(_solve, name)
    raise AttributeError(name)
