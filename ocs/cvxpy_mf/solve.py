"""Solve a CVXPY problem with OCS, matrix-free end to end.

CVXPY does the modeling and cone lowering (Dcp2Cone, attribute lowering); we
stop the chain BEFORE ConeMatrixStuffing, so the constraint matrix is never
stuffed.  Each ordered cone group's LinOp trees are walked into matvec/rmatvec
closures (see ``linop_walker``), wrapped as OCS operators, and solved by the
OCS restart-PDHG engine.  The variable values are then inverted back through
the applied reductions onto the original problem's variables.

Cones wired: zero, nonnegative, second-order (single and batched multi-cone),
PSD, exponential (single and batched).

Parameters (the same-operator regime): when the problem is DPP and no
parameter multiplies the variable inside a constraint (checked structurally),
CVXPY ``Parameter``s become OCS per-instance offsets and a per-instance linear
cost.  The family compiles once; each instance is a data change, and
``CompiledFamily.solve_many`` batches instances on the GPU.  A parameter that
multiplies the variable in a constraint is a parameterized operator: the
compile-once family (``compile_mf``) rejects it, while the one-shot paths
(``solve_mf`` and CVXPY's ``problem.solve(solver="OCS")``, which
re-canonicalize on every call) substitute the parameters' current values
into the tree at solve time, exactly as CVXPY's ``EvalParams`` does for
non-DPP problems, so the operator closure is built from the current values
and rebuilt on the next solve.  Non-DPP parameters are evaluated up front
(EvalParams).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from cvxpy.constraints import Equality, Inequality, NonNeg, Zero, PSD, ExpCone, PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.problems.objective import Maximize
from cvxpy.reductions import InverseData
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
try:
    from cvxpy.reductions.eliminate_zero_sized import EliminateZeroSized
except ImportError as error:
    raise ImportError(
        "The experimental matrix-free CVXPY bridge needs newer private "
        "CVXPY reduction APIs. Use ocs.cvxpy_harness.validate_against_cvxpy "
        "for the supported small-problem validation entry point."
    ) from error
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import Solution as CvxSolution, failure_solution
from cvxpy.reductions.utilities import (
    group_constraints,
    lower_equality,
    lower_ineq_to_nonneg,
)
import cvxpy.settings as cvx_s

from ..ir import Term, ProblemFamily
from ..cvxpy_harness.backend import _cvxpy_status
from ..operators.base import LinearOperator
from ..prox.library import Equality as EqBlock, NonNeg as NonNegBlock
from ..prox.cones import (
    SecondOrderCone, SOCBatch, PSDCone, ExpCone as ExpBlock, ExpConeBatch,
    PowerCone, PowerConeBatch,
)
from .linop_walker import check_param_operator, linearize_affine, make_param_forward


class _ClosureOp(LinearOperator):
    """An OCS operator from walker closures: forward/adjoint only, no matrix."""

    def __init__(self, shape, mv, rmv):
        super().__init__(shape)
        self._mv, self._rmv = mv, rmv

    def matvec(self, x):
        return self._mv(x)

    def rmatvec(self, y):
        return self._rmv(y)


_SUPPORTED_CONES = (Zero, NonNeg, SOC, PSD, ExpCone, PowCone3D)


def _solver_context():
    """The CVXPY solver context of the OCS interface, so that canonicalizers
    which consult it (the perspective atom builds an auxiliary solving chain
    for its solver) take the matrix-free route.  ``None`` on a CVXPY without
    ``SolverInfo``."""
    try:
        from cvxpy.utilities.solver_context import SolverInfo
    except ImportError:                     # pragma: no cover
        return None
    return SolverInfo(solver="OCS", supported_constraints=frozenset(_SUPPORTED_CONES),
                      supports_bounds=False, psd_triangle_kind=None,
                      psd_sqrt2_scaling=None)


def _supports_symbolic_perspective():
    """Recognize the tested CVXPY clone's symbolic perspective capability.

    SolverInfo alone is insufficient: stock CVXPY's perspective canonicalizer
    constructs an auxiliary stuffed problem. The known clone needs both its
    registered matrix-free OCS interface and the symbolic canonicalizer hook.
    Unknown private-API layouts are rejected rather than silently stuffed.
    """
    import importlib
    import cvxpy as cp
    try:
        from cvxpy.atoms.perspective import perspective
        from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
        from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC
        mod = importlib.import_module(
            "cvxpy.reductions.dcp2cone.canonicalizers.perspective_canon")
        interface = SOLVER_MAP_CONIC.get("OCS")
        canon = CANON_METHODS.get(perspective)
        return bool(
            "OCS" in cp.installed_solvers()
            and getattr(interface, "MATRIX_FREE", False)
            and canon is mod.perspective_canon
            and callable(getattr(mod, "_perspective_canon_matrix_free", None))
            and "_perspective_canon_matrix_free" in canon.__code__.co_names
        )
    except (ImportError, AttributeError):
        return False


def has_parameterized_operator(prob):
    """True when a parameter multiplies the variable inside a constraint of
    the cone-lowered ``prob`` (checked structurally on the LinOp trees): the
    linear operator itself then depends on the parameter.  Parameters in the
    objective are not consulted; a per-instance linear cost is native."""
    for con in prob.constraints:
        if not con.parameters():
            continue
        for arg in con.args:
            if check_param_operator(arg.canonical_form[0])[2]:
                return True
    return False


def evaluate_operator_parameters(prob):
    """The one-shot regime's answer to a parameterized operator.

    When a parameter multiplies the variable in a constraint of the
    cone-lowered ``prob``, every parameter is replaced by a constant holding
    its current value (CVXPY's ``EvalParams`` reduction, the same substitution
    CVXPY applies to non-DPP problems), so the operator closures are built
    from the current values; a caller that re-canonicalizes on each solve
    rebuilds them with the next values.  Returns ``(problem, evaluated)``;
    the problem is returned unchanged when the operator does not depend on
    any parameter, so the same-operator (offset and cost) regime is kept.
    A sparse parameter value stays a sparse constant.
    """
    if not prob.parameters() or not has_parameterized_operator(prob):
        return prob, False
    evaluated, _ = EvalParams().apply(prob)
    return evaluated, True


def _cone_lower(problem, operator_params="reject"):
    """Apply the CVXPY reductions up to (excluding) matrix stuffing.

    ``operator_params="evaluate"`` (the one-shot regime) appends an
    ``EvalParams`` step when a parameter multiplies the variable inside a
    constraint; ``"reject"`` (the compile-once family) leaves the parameter
    in the tree for ``stuff_mf`` to reject.
    """
    if operator_params not in ("reject", "evaluate"):
        raise ValueError("operator_params must be 'reject' or 'evaluate'")
    from cvxpy.atoms.perspective import perspective
    if perspective in problem.atoms() and not _supports_symbolic_perspective():
        raise NotImplementedError(
            "Matrix-free perspective canonicalization requires the tested "
            "CVXPY clone with its registered MATRIX_FREE OCS interface and "
            "symbolic perspective canonicalizer. Stock CVXPY canonicalizes "
            "perspective through an auxiliary stuffed problem, so this bridge "
            "cannot guarantee its no-matrix contract on that path."
        )
    reductions = []
    if isinstance(problem.objective, Maximize):
        reductions.append(FlipObjective())
    if problem.parameters() and not problem.is_dpp():
        reductions.append(EvalParams())
    try:
        reductions += [
            Dcp2Cone(quad_obj=False, solver_context=_solver_context()),
            CvxAttr2Constr(reduce_bounds=True),
            EliminateZeroSized(),
        ]
    except TypeError as error:
        raise NotImplementedError(
            "The experimental matrix-free CVXPY bridge is incompatible with "
            "these private reduction APIs. Use the separate cvxpy_harness "
            "validation entry point."
        ) from error
    prob, stack = problem, []
    for red in reductions:
        prob, inv = red.apply(prob)
        stack.append((red, inv))
    if operator_params == "evaluate" and prob.parameters() \
            and has_parameterized_operator(prob):
        red = EvalParams()          # its invert is the identity
        prob, inv = red.apply(prob)
        stack.append((red, inv))
    return prob, stack


def _size(shape):
    return int(np.prod(shape)) if shape else 1


class UnconstrainedProblem(Exception):
    """The cone-lowered problem has no constraints: a linear objective over
    all of R^n.  It is unbounded unless the cost vector is zero, in which case
    every point is optimal.  The interface answers without a solve."""

    def __init__(self, x_len, cost_fn):
        super().__init__("no constraints")
        self.x_len = x_len
        self.cost_fn = cost_fn


class Stuffed:
    """The matrix-free analogue of a stuffed cone program: the compiled OCS
    family plus everything needed to map solutions back."""

    def __init__(self, compiled, prob, params, pname, id_to_col, x_len,
                 obj_fwd, cost_fn, penv0, dual_layout=(), terms=()):
        self.compiled = compiled
        self.terms = tuple(terms)
        self.prob = prob
        self.params = params
        self.pname = pname
        self.id_to_col = id_to_col
        self.x_len = x_len
        self.obj_fwd = obj_fwd
        self.cost_fn = cost_fn
        self.penv0 = penv0
        # (constraint, kind, size) in the order the terms stack the dual.
        self.dual_layout = tuple(dual_layout)

    def dual_values(self, y):
        """Map OCS's stacked dual (one segment per term, in term order) to
        CVXPY constraint duals for the cone-lowered problem.

        OCS's dual for a term f_i(K_i x + l_i) is a subgradient y_i of the
        indicator, so it lies in the normal cone; CVXPY's dual for a cone
        constraint lies in the dual cone, hence the sign change there.  Layouts:
        Zero and NonNeg segments are the constraint expression in Fortran
        order (reshaped to the constraint shape, as ConeMatrixStuffing does);
        a PSD segment is the full k x k matrix in Fortran order; an SOC term
        is stacked [t (q); vec_F(X) (n q)] and CVXPY expects one row
        (t_j, X[:, j]) per cone; an exponential-cone term is stacked
        [x (q); y (q); z (q)] and CVXPY expects one row (x_i, y_i, z_i) per
        cone.
        """
        y = np.asarray(y, np.float64).reshape(-1)
        total = sum(size for _, _, size in self.dual_layout)
        if y.size != total:
            raise ValueError(f"dual layout mismatch: y has {y.size} entries, "
                             f"the terms stack {total}")
        out, off = {}, 0
        for con, kind, size in self.dual_layout:
            # CVXPY's sign convention: the multiplier of an equality keeps
            # the solver's sign, the multiplier of a cone membership is the
            # dual-cone element, the negative of the normal-cone subgradient.
            seg = y[off:off + size] if kind == "zero" else -y[off:off + size]
            off += size
            if isinstance(kind, tuple) and kind[0] == "pow":
                _, idx, q = kind
                acc = out.setdefault(con.id, np.zeros((3, q)))
                acc[:, idx] = seg.reshape(3, -1)
                continue
            if isinstance(kind, tuple) and kind[0] == "psd_batch":
                _, b, B, k = kind
                acc = out.setdefault(con.id, np.zeros((B, k, k)))
                M = seg.reshape(k, k, order="F")
                acc[b] = 0.5 * (M + M.T)
                continue
            if kind == "soc":
                q = _size(con.args[0].shape)
                t, Z = seg[:q], seg[q:].reshape(-1, q, order="F")
                val = np.hstack([t[:, None], Z.T]).reshape(-1)
            elif kind == "exp":
                val = seg.reshape(3, -1).T.reshape(-1)
            elif kind == "psd":
                k = int(con.args[0].shape[0])
                M = seg.reshape(k, k, order="F")
                val = 0.5 * (M + M.T)
            else:
                shape = tuple(con.args[0].shape)
                val = float(seg[0]) if shape == () else seg.reshape(shape, order="F")
            out[con.id] = val
        for con, kind, _ in self.dual_layout:
            if isinstance(kind, tuple) and kind[0] == "pow" and out[con.id].ndim == 2:
                v = out[con.id]
                xs = tuple(con.args[0].shape)
                out[con.id] = np.stack([np.reshape(v[i], xs, order="F") for i in range(3)])
        return out

    def param_data(self):
        data = {}
        for p in self.params:
            if p.value is None:
                raise ValueError(f"parameter '{p.name()}' has no value")
            data[self.pname[p.id]] = p.value
        return data

    def penv_from(self, data):
        return {p.id: jnp.asarray(data[self.pname[p.id]], jnp.float32)
                for p in self.params}

    LOOSE_RAY = 1e-3        # evidence thresholds for the inaccurate verdicts

    def primal_violation(self, x, data):
        """Largest relative distance of a term's ``K_i x + l_i`` from its
        cone, by the operators and the cone projections only (no matrix)."""
        x = jnp.asarray(x, jnp.float32)
        worst = 0.0
        for term in self.terms:
            s = term.op.matvec(x)
            if term.offset_fn is not None:
                s = s + jnp.asarray(term.offset_fn(data), jnp.float32)
            proj = term.block.prox(s, jnp.asarray(1.0, s.dtype))
            dist = float(jnp.linalg.norm(s - proj))
            worst = max(worst, dist / max(1.0, float(jnp.linalg.norm(s))))
        return worst

    def _inaccurate_certificate(self, sol, data, attributes):
        infeas = getattr(self.compiled, "_infeas", None)
        if infeas is None:
            return None
        try:
            checks = infeas(sol.x, sol.y, self.compiled._resolve(data))
        except Exception:
            return None
        # The loose ray thresholds below are evidence, not proof: the ray
        # residual is normalized by the operator norm alone, so on a problem
        # whose solution is large a ray with residual 1e-5 can still be
        # consistent with a feasible point (|<K^T d, x>| <= residual * |K| * |x|).
        # A returned point that is itself nearly feasible refutes the ray, and
        # the honest answer is the inaccurate optimum (never a relabeling to
        # optimal): a perspective problem with objective scale 99 stalled with
        # such a point and was reported infeasible_inaccurate before this.
        try:
            violation = self.primal_violation(sol.x, data)
        except Exception:
            violation = float("inf")
        attributes["ocs_primal_violation"] = violation
        if violation <= self.LOOSE_RAY:
            return None

        def f(name):
            try:
                return float(checks[name])
            except (KeyError, TypeError, ValueError):
                return float("nan")
        attributes = dict(attributes)
        attributes["ocs_ray_residual"] = f("primal_infeasibility_residual")
        attributes["ocs_ray_margin"] = f("primal_infeasibility_margin")
        if (f("primal_infeasibility_supported") > 0 and f("primal_infeasibility_residual") <= self.LOOSE_RAY
                and f("primal_infeasibility_margin") > self.LOOSE_RAY):
            duals = None
            try:
                duals = self.dual_values(checks["dual_direction"])
            except (ValueError, TypeError, KeyError):
                duals = None
            return failure_solution(cvx_s.INFEASIBLE_INACCURATE, attributes, duals)
        if (f("recession_supported") > 0 and f("recession_residual") <= self.LOOSE_RAY
                and f("recession_margin") > self.LOOSE_RAY and f("base_feasibility_residual") <= self.LOOSE_RAY):
            return failure_solution(cvx_s.UNBOUNDED_INACCURATE, attributes)
        return None

    def solution(self, sol, data=None):
        """Raw OCS solution -> cvxpy Solution for the cone-lowered problem."""
        from cvxpy.error import SolverError
        # The base mapping first, WITHOUT the declared-budget contract: the
        # infeasible/unbounded ray probe below must see a budget-limited run as
        # a budget-limited run, so that an infeasible problem is never reported
        # as an inaccurate optimum because an extra contract was evaluated.
        status = _cvxpy_status(sol.status)
        # ocs_residual is the relative KKT residual.  It is NOT an
        # objective-error bound and NOT a duality gap, and it no longer awards
        # the status: `ocs_accuracy_criterion` names what did.  Under the
        # default criterion a strict objective lower bound IS established on
        # the solve path, and `ocs_objective_gap` is the certified relative
        # gap at the returned point; it is None when nothing was certified, in
        # which case ocs_status is `unverified` and no accuracy is claimed.
        # The legacy Fenchel surrogate is still not a bound and is still not
        # reported here.
        criterion = getattr(sol, "accuracy_criterion", "fixed_point")
        gap = getattr(sol, "objective_gap", None)
        attributes = {"ocs_status": sol.status, "ocs_residual": sol.residual,
                      "ocs_accuracy_criterion": criterion,
                      "ocs_objective_gap": (None if gap is None else float(gap)),
                      "ocs_objective_lower_bound":
                          getattr(sol, "objective_lower_bound", None),
                      "ocs_objective_primal_value":
                          getattr(sol, "objective_primal_value", None),
                      "ocs_accuracy_detail": getattr(sol, "accuracy_detail", ""),
                      "ocs_objective_bound_capable":
                          bool(self.compiled.meta.get("objective_bound_capable", False)),
                      "ocs_gap_reported": bool(gap is not None),
                      # (i) strict criterion, (ii) declared-budget contract
                      # (None when none was declared), (iii) why the iteration
                      # stopped.  Separately readable, never merged.
                      "ocs_strict_certificate": getattr(sol, "strict_certificate", None),
                      "ocs_contract_certificate": getattr(sol, "contract_certificate", None),
                      "ocs_termination": getattr(sol, "termination", None),
                      "ocs_contract": getattr(sol, "contract", None),
                      cvx_s.NUM_ITERS: sol.iters}
        if status == cvx_s.USER_LIMIT:
            # The budget ran out before the certificate reached its acceptance
            # threshold.  If the returned point already carries a supported
            # Farkas ray (or a descent recession direction from a feasible
            # base) with a small residual and a positive separation margin,
            # say so: CVXPY's inaccurate infeasible and unbounded statuses
            # exist for exactly this, and a finite objective value for a
            # problem that has none would be the wrong kind of answer.
            verdict = self._inaccurate_certificate(sol, data or {}, attributes)
            if verdict is not None:
                return verdict
        # Only now may the declared-budget contract change the label, by the
        # documented rule in ocs.cvxpy_harness.backend._cvxpy_status: never to
        # `optimal`, and never for a run whose iteration budget ran out.
        status = _cvxpy_status(sol.status,
                               contract_certificate=getattr(sol, "contract_certificate", None),
                               termination=getattr(sol, "termination", None))
        if status in (cvx_s.INFEASIBLE, cvx_s.UNBOUNDED):
            duals = None
            if status == cvx_s.INFEASIBLE and sol.certificate is not None:
                try:            # the Farkas direction lives in the dual space
                    duals = self.dual_values(sol.certificate)
                except (ValueError, TypeError):
                    duals = None
            return failure_solution(status, attributes, duals)
        if status == cvx_s.SOLVER_ERROR:
            raise SolverError("OCS returned numerical_error; no optimal solution is available")
        data = data or {}
        x = np.asarray(sol.x, np.float64)
        if not np.all(np.isfinite(x)):
            raise SolverError("OCS returned a nonfinite primal point")
        penv = self.penv_from(data) if self.params else self.penv0
        zero_x = jnp.zeros(self.x_len)
        r_off = float(np.asarray(self.obj_fwd(zero_x, penv)).ravel()[0])
        c_vec = np.asarray(self.cost_fn(data), np.float64)
        primal = {}
        for v in self.prob.variables():
            col = self.id_to_col[v.id]
            size = _size(v.shape)
            val = x[col:col + size]
            primal[v.id] = val.reshape(v.shape, order="F") if v.shape else val[0]
        obj_val = float(c_vec @ x) + r_off
        duals = self.dual_values(sol.y) if sol.y is not None else {}
        return CvxSolution(status, obj_val, primal, duals, attributes)


def stuff_mf(prob, plan=None, **compile_kwargs):
    """Stuff an already cone-lowered problem into an OCS family, matrix-free.

    This is the operator-native replacement for ConeMatrixStuffing: the
    constraint matrix is never formed.  Returns a ``Stuffed``.
    """
    params = prob.parameters()
    pname = {p.id: (p.name() or f"param{p.id}") for p in params}
    param_names = sorted(pname.values())
    penv0 = {p.id: jnp.zeros(p.shape) if p.shape else jnp.zeros(())
             for p in params}

    def penv_from(data):
        return {p.id: jnp.asarray(data[pname[p.id]], jnp.float32)
                for p in params}

    # Lower leftover Equality/Inequality; normalize SOC axis; group cones.
    cons = []
    for con in prob.constraints:
        if isinstance(con, Equality):
            con = lower_equality(con)
        elif isinstance(con, Inequality):
            con = lower_ineq_to_nonneg(con)
        elif isinstance(con, SOC) and con.axis == 1:
            con = SOC(con.args[0], con.args[1].T, axis=0, constr_id=con.constr_id)
        cons.append(con)
    constr_map = group_constraints(cons)
    supported = {Zero, NonNeg, SOC, PSD, ExpCone, PowCone3D}
    leftover = [c for c in cons if type(c) not in supported]
    if leftover:
        raise NotImplementedError(
            f"cones not yet wired matrix-free: {sorted({type(c).__name__ for c in leftover})}")

    inverse_data = InverseData(prob)
    id_to_col, x_len = inverse_data.var_offsets, inverse_data.x_length
    zero_x = jnp.zeros(x_len)

    def build_group(ops, where):
        for op in ops:
            _, _, bad = check_param_operator(op)
            if bad:
                raise NotImplementedError(
                    f"a parameter multiplies the variable in {where}: the "
                    "operator depends on the parameter (parameterized-K family)"
                    "; not supported by the same-operator bridge")
        fwd = make_param_forward(ops, id_to_col, x_len)
        f0 = lambda xx, f=fwd: f(xx, penv0)
        # The linear part A x is the exact linearization of the affine map
        # (no offset is added and subtracted, which would lose low bits of
        # A x in float32 when the offset is large); see linearize_affine.
        b0, mv = linearize_affine(f0, zero_x)
        _, vjp = jax.vjp(f0, zero_x)
        rmv = lambda y, v=vjp: v(y)[0]
        if params:
            off = lambda data, f=fwd: f(zero_x, penv_from(data))
        else:
            bc = np.asarray(b0, np.float32)
            off = lambda data, b=bc: b
        return _ClosureOp((int(b0.shape[0]), x_len), mv, rmv), off, int(b0.shape[0])

    terms = []
    layout = []
    for group, block, kind in ((constr_map[Zero], EqBlock(), "zero"),
                               (constr_map[NonNeg], NonNegBlock(), "nonneg")):
        if not group:
            continue
        ops = [arg.canonical_form[0] for con in group for arg in con.args]
        op_, off, _ = build_group(ops, "an affine constraint")
        terms.append(Term(op=op_, block=block, offset_fn=off))
        layout += [(con, kind, _size(con.args[0].shape)) for con in group]

    for con in constr_map[SOC]:
        t_arg, x_arg = con.args
        q = _size(t_arg.shape)
        ops = [arg.canonical_form[0] for arg in con.args]
        op_, off, m = build_group(ops, "an SOC constraint")
        if q == 1:
            block = SecondOrderCone()
        else:
            d = (m - q) // q
            block = SOCBatch(q, d)
        terms.append(Term(op=op_, block=block, offset_fn=off))
        layout.append((con, "soc", m))

    for con in constr_map.get(PSD, []):
        arg = con.args[0]
        ops = [arg.canonical_form[0]]
        if arg.ndim == 2:
            k = int(arg.shape[0])
            op_, off, m = build_group(ops, "a PSD constraint")
            terms.append(Term(op=op_, block=PSDCone(k), offset_fn=off))
            layout.append((con, "psd", m))
            continue
        # A batch of matrices (batch axis first, CVXPY convention): the
        # forward map flattens the array in Fortran order, so slice b of the
        # batch occupies the rows b + B (i + k j).  Each slice is its own
        # semidefinite cone, selected from the shared forward map.
        B, k = int(arg.shape[0]), int(arg.shape[1])
        full_op, full_off, m = build_group(ops, "a batched PSD constraint")
        if m != B * k * k:
            raise NotImplementedError("unexpected batched PSD stacking")
        for b in range(B):
            sel = b + B * np.arange(k * k)           # F-order positions of slice b
            sel_j = jnp.asarray(sel)
            def mv_sub(x, o=full_op, sj=sel_j):
                return o.matvec(x)[sj]
            def rmv_sub(y, o=full_op, sj=sel_j, mm=m):
                return o.rmatvec(jnp.zeros(mm, dtype=y.dtype).at[sj].set(y))
            def off_sub(data, f=full_off, sn=sel):
                return np.asarray(f(data))[sn]
            terms.append(Term(op=_ClosureOp((int(sel.size), x_len), mv_sub, rmv_sub),
                              block=PSDCone(k), offset_fn=off_sub))
            layout.append((con, ("psd_batch", b, B, k), k * k))

    for con in constr_map[ExpCone]:
        q = _size(con.args[0].shape)
        ops = [arg.canonical_form[0] for arg in con.args]
        op_, off, m = build_group(ops, "an exponential-cone constraint")
        block = ExpBlock() if q == 1 else ExpConeBatch(q)
        terms.append(Term(op=op_, block=block, offset_fn=off))
        layout.append((con, "exp", m))

    # Three-dimensional power cones x^a y^(1-a) >= |z|, elementwise over the
    # cones of one constraint.  OCS's block takes one exponent, so the cones
    # of a constraint are grouped by exponent; each group is its own term
    # that selects its rows from the constraint's [x; y; z] forward map.
    for con in constr_map.get(PowCone3D, []):
        q = _size(con.args[0].shape)
        alpha = np.asarray(con.alpha.value, np.float64).reshape(-1)
        if alpha.size == 1:
            alpha = np.repeat(alpha, q)
        if alpha.size != q:
            raise NotImplementedError("PowCone3D exponent shape does not match its cones")
        ops = [arg.canonical_form[0] for arg in con.args]
        full_op, full_off, m = build_group(ops, "a power-cone constraint")
        if m != 3 * q:
            raise NotImplementedError("unexpected power-cone stacking")
        for a in np.unique(alpha):
            idx = np.flatnonzero(alpha == a)
            sel = np.concatenate([idx, q + idx, 2 * q + idx])
            sel_j = jnp.asarray(sel)
            if idx.size == q:
                op_, off = full_op, full_off
            else:
                def mv_sub(x, o=full_op, sj=sel_j):
                    return o.matvec(x)[sj]
                def rmv_sub(y, o=full_op, sj=sel_j, mm=m):
                    return o.rmatvec(jnp.zeros(mm, dtype=y.dtype).at[sj].set(y))
                def off_sub(data, f=full_off, sn=sel):
                    return np.asarray(f(data))[sn]
                op_ = _ClosureOp((int(sel.size), x_len), mv_sub, rmv_sub)
                off = off_sub
            block = PowerCone(float(a)) if idx.size == 1 else PowerConeBatch(int(idx.size), float(a))
            terms.append(Term(op=op_, block=block, offset_fn=off))
            layout.append((con, ("pow", idx, q), 3 * int(idx.size)))

    # Objective  c(theta)^T x + r(theta): parameters MAY multiply x here
    # (that is a per-instance linear cost, which OCS supports natively).
    obj_fwd = make_param_forward([prob.objective.expr.canonical_form[0]],
                                 id_to_col, x_len)
    if params:
        def cost_fn(data):
            penv = penv_from(data)
            _, ovjp = jax.vjp(lambda xx: obj_fwd(xx, penv), zero_x)
            return ovjp(jnp.ones(1))[0]
    else:
        _, ovjp0 = jax.vjp(lambda xx: obj_fwd(xx, penv0), zero_x)
        c_const = np.asarray(ovjp0(jnp.ones(1))[0], np.float32)
        cost_fn = lambda data, c=c_const: c

    if not terms:
        raise UnconstrainedProblem(x_len, cost_fn)
    family = ProblemFamily(n=x_len, terms=terms, cost_fn=cost_fn)
    if plan:
        # Optional hardware-aware planning above compile: estimate, probe,
        # cache, and let the plan fill the hardware-facing compile keywords
        # the caller left unset.  Parameters need values for the probe.
        from ..experimental.plan import plan_family
        probe_data = None
        if not params or all(p.value is not None for p in params):
            probe_data = {pname[p.id]: p.value for p in params}
        chosen = plan_family(family, probe_data, mode=str(plan), **compile_kwargs)
        compile_kwargs = dict(chosen.compile_kwargs)
    compiled = family.compile(**compile_kwargs)
    return Stuffed(compiled, prob, params, pname, id_to_col, x_len,
                   obj_fwd, cost_fn, penv0, dual_layout=layout, terms=terms)


def compile_mf(problem, operator_params="reject", **compile_kwargs):
    """Cone-lower ``problem`` with CVXPY and compile an OCS family, matrix-free.

    Returns ``(compiled, recover, param_names)``.  ``compiled`` is the OCS
    ``CompiledFamily``: call ``compiled.solve_instance(data)`` (or
    ``solve_many``) with ``data = {param_name: value}``.  ``recover(sol, data)``
    inverts a solution back onto the original problem's variables and returns
    the objective value.

    A parameter that multiplies the variable inside a constraint changes the
    operator, which a compiled family cannot express: the default
    ``operator_params="reject"`` raises ``NotImplementedError`` for it.
    ``"evaluate"`` is the one-shot regime of ``solve_mf``: such parameters
    are frozen at their current values (and are then absent from
    ``param_names``), so the family is good for these values only.
    """
    prob, stack = _cone_lower(problem, operator_params=operator_params)
    st = stuff_mf(prob, **compile_kwargs)

    def recover(sol, data=None):
        cvx_sol = st.solution(sol, data)
        for red, inv in reversed(stack):
            cvx_sol = red.invert(cvx_sol, inv)
        problem.unpack(cvx_sol)
        return cvx_sol.opt_val

    return st.compiled, recover, sorted(st.pname.values())


def solve_mf(problem, **compile_kwargs):
    """One-shot: compile matrix-free, solve with OCS, write values back.

    Parameters take their current ``.value``.  A parameter that multiplies
    the variable inside a constraint is substituted into the tree at its
    current value (the operator closure is rebuilt on every call); the
    others stay per-instance data.  Returns ``(objective, sol)``; the
    problem's variables receive their values, as after ``problem.solve()``.
    """
    compiled, recover, names = compile_mf(problem, operator_params="evaluate",
                                          **compile_kwargs)
    data = {}
    for p in problem.parameters():
        name = p.name() or f"param{p.id}"
        if name not in names:           # frozen into the operator by EvalParams
            continue
        if p.value is None:
            raise ValueError(f"parameter '{p.name()}' has no value")
        data[name] = p.value
    sol = compiled.solve_instance(data)
    val = recover(sol, data)
    return val, sol
