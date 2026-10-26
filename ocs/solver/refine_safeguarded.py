"""Safeguarded refinement: bounded proposals, each assessed by the UNCHANGED
criterion (7), and the previous assessed pair kept whenever a proposal is
invalid or worse.

This wraps, and does not alter, the existing Gauss-Newton helper in
``refine.py`` (whose ``CompiledFamily.refine`` applies ``z + delta`` with no
line search and no acceptance).  Nothing here is imported by default and no
solver default changes.  Two proposal kinds are provided:

* ``gauss_newton_proposal`` -- the matrix-free Gauss-Newton/Krylov correction
  on the fixed-point residual map R(x, y) (jvp/vjp products only), with a
  damped step chosen by backtracking on ||R||.  Damping regularizes the
  correction, never the original objective.
* ``LassoSupportCorrection`` -- the LASSO fixed-support / fixed-sign
  correction: with S the support guessed from the current point and
  s = sign(x_S), solve F_S^T F_S d = -(F_S^T (F x - b) + lam s) and check sign
  stability, rank deficiency (Krylov breakdown / Rayleigh quotient) and the
  inactive conditions |F^T (F x - b)|_{S^c} <= lam before trusting the
  support.  LSQR on the augmented system [[I, -F_S], [F_S^T, 0]] is the
  baseline; PCG on the SPD operator F_S^T F_S (Jacobi preconditioner) is the
  alternative.  A guessed support is never a qualification: whatever the
  correction residual, only criterion (7) through the family's public
  ``objective_bound``/``acceptance`` awards success.

Every proposal is reconstructed as an actual (x, y) pair in original
coordinates, assessed through the public evaluator with recovery exactly as
the solver would, and adopted only if its assessed discrepancy is strictly
smaller and its witness valid.  Nonlinear steps, Krylov iterations,
line-search trials, logical operator products, assessment calls and wall time
are all bounded; the ledger records every number.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp

from .refine import _cg

__all__ = ["RefineBudget", "criterion7_assessor", "gauss_newton_proposal",
           "LassoSupportCorrection", "safeguarded_refine"]


@dataclass(frozen=True)
class RefineBudget:
    """Hard bounds.  Each is checked before the work it bounds is started."""
    max_steps: int = 3            # nonlinear (proposal) steps
    cg_iters: int = 80            # Krylov iterations per Gauss-Newton direction
    max_line_search: int = 8      # backtracking trials per step
    max_products: int = 20_000    # logical operator applications (upper bound)
    max_assessments: int = 16     # criterion-(7) evaluations, start included
    max_seconds: float = 600.0    # wall clock for the whole refinement


def _finite(a):
    return a is not None and bool(np.all(np.isfinite(np.asarray(a))))


def _scalars(report):
    out = {}
    for k, v in report.items():
        if hasattr(v, "shape"):
            if np.asarray(v).shape == ():
                out[k] = float(v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
    return out


def criterion7_assessor(comp, data, tol=None):
    """Assess a pair through the family's public evaluator, recovery on, as the
    solver would at a check point, and decide with ``acceptance`` -- the one
    shared predicate.  The score is the magnitude of the acceptance
    discrepancy |J(x) - d(y)| / max(|J|, |d|), or +inf when the acceptance
    quantity is invalid or the point is not primal feasible to ``tol``."""
    def assess(x, y):
        report = comp.objective_bound(x, y, data, return_dual=True)
        witness = report.pop("certifying_dual", None)
        acc = comp.acceptance(report, tol)
        valid = bool(acc["acceptance_quantity_valid"] and acc["primal_feasible"]
                     and _finite(witness))
        if valid:
            upper, lower = acc["objective_primal_upper_bound"], acc["objective_lower_bound"]
            scale = max(abs(upper), abs(lower))
            score = abs(upper - lower) / scale if scale > 0.0 else 0.0
        else:
            score = math.inf
        return {"score": score, "valid": valid, "target_met": bool(acc["target_met"]),
                "products": int(report.get("operator_applications", 0)),
                "witness": witness, "acceptance": acc, "production": _scalars(report)}
    return assess


# ---------------------------------------------------------------------------
# Proposal 1: damped matrix-free Gauss-Newton on the fixed-point residual.
# ---------------------------------------------------------------------------
def gauss_newton_proposal(comp, data, *, reg=1e-8, cg_iters=80, max_line_search=8,
                          x64=False):
    """Return ``propose(x, y, ledger) -> (x_new, y_new, info)``.

    ``info["products"]`` is an UPPER BOUND on logical operator applications:
    R costs 2 (one forward, one adjoint); a jvp of R is charged 4 (primal and
    tangent), a vjp forward pass 2 and each vjp application 2.  Line search:
    the step t = 1, 1/2, ... is accepted at the first trial with
    ||R(z + t d)|| < (1 - 1e-4 t) ||R(z)||; with none accepted the proposal
    is withdrawn (returns ``None``) and the caller keeps its pair."""
    n = comp.n
    dense = comp._resolve_dense(data or {})
    residual = getattr(comp._refine, "residual", None)
    if residual is None:
        raise NotImplementedError("refinement residual not available for this family")

    def R(z):
        return residual(z[:n], z[n:], dense)

    def _direction(z):
        Rz = R(z)
        Jmv = lambda v: jax.jvp(R, (z,), (v,))[1]
        _, vjp = jax.vjp(R, z)
        JT = lambda w: vjp(w)[0]
        Aop = lambda v: JT(Jmv(v)) + reg * v
        delta = _cg(Aop, -JT(Rz), cg_iters)
        return delta, jnp.linalg.norm(Rz)

    def _rnorm(z):
        return jnp.linalg.norm(R(z))

    direction = jax.jit(_direction)
    rnorm = jax.jit(_rnorm)
    cast = (lambda a: jnp.asarray(a, jnp.float64)) if x64 else (lambda a: jnp.asarray(a))

    def propose(x, y, ledger):
        t0 = time.perf_counter()
        ctx = jax.enable_x64() if x64 else _nullcontext()
        with ctx:
            z = jnp.concatenate([cast(x), cast(y)])
            delta, r0 = direction(z)
            products = 6 + 6 * cg_iters
            r0 = float(r0)
            t, trials, accepted, rt = 1.0, 0, False, float("nan")
            while trials < max_line_search:
                trials += 1
                zt = z + t * delta
                rt = float(rnorm(zt)); products += 2
                if math.isfinite(rt) and rt < (1.0 - 1e-4 * t) * r0:
                    accepted = True
                    break
                t *= 0.5
            info = {"kind": "gauss_newton", "x64": bool(x64), "reg": reg, "cg_iters": cg_iters,
                    "residual_norm_before": r0, "residual_norm_after": rt, "step_length": t,
                    "direction_norm": float(jnp.linalg.norm(delta)),
                    "line_search_trials": trials, "line_search_accepted": accepted,
                    "products": products, "seconds": time.perf_counter() - t0}
            if not accepted:
                return None, None, info
            xn, yn = zt[:n], zt[n:]
        return jnp.asarray(xn, jnp.float32), jnp.asarray(yn, jnp.float32), info
    return propose


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Proposal 2: LASSO fixed-support / fixed-sign correction (operator-only).
# ---------------------------------------------------------------------------
class LassoSupportCorrection:
    """Fixed-support/sign Newton correction for  min 1/2||F x - b||^2 + lam||x||_1.

    ``forward(x)`` and ``adjoint(u)`` are the operator's products in the
    arithmetic of ``dtype`` (NumPy float64 by default); the operator is never
    formed.  One call = one correction round from the given x:

      1. S = {i : |x_i| > threshold} adjusted by the previous round's
         findings (flipped signs removed, inactive violators added);
         s = sign(x_S).
      2. g = F^T (F x - b); record the active deviation max|g_S + lam s|/lam,
         the inactive maximum max|g_{S^c}|/lam and the violation count.
      3. Solve F_S^T F_S d = -(g_S + lam s):  ``solver="lsqr"`` runs LSQR on
         the augmented square system [[I, -F_S], [F_S^T, 0]] [dr; d] =
         [0; -(g_S + lam s)]; ``solver="pcg"`` runs Jacobi-preconditioned CG on
         the SPD operator F_S^T F_S, with breakdown detection (a nonpositive
         Rayleigh quotient is rank deficiency; the smallest quotient seen is
         recorded).
      4. x_new = x + P_S d; recheck sign stability (flips), the inactive
         conditions and the active deviation at x_new in float64.
      5. Return the represented pair (x_new, y_new = F x_new - b), both cast to
         float32 for the family's evaluator, with every number in ``info``.
         ``info["support_valid"]`` is the float64 KKT verdict of the guess; it
         is diagnostic only -- the caller's criterion-(7) assessment decides.
    """

    stateful = True       # successive calls adjust the support; see __call__

    def __init__(self, forward, adjoint, b, lam, *, solver="pcg", max_iters=200,
                 rtol=1e-12, threshold=0.0, dtype=np.float64, max_support=None):
        if solver not in ("pcg", "lsqr"):
            raise ValueError("solver must be 'pcg' or 'lsqr'")
        self.forward, self.adjoint = forward, adjoint
        self.b = np.asarray(b, dtype)
        self.lam = float(lam)
        self.solver, self.max_iters, self.rtol = solver, int(max_iters), float(rtol)
        self.threshold, self.dtype = float(threshold), dtype
        self.max_support = max_support
        self.rounds = []          # per-call diagnostics, in order
        self._remove = set()      # indices whose sign flipped in an earlier round
        self._add = set()         # inactive violators found in an earlier round

    # -- linear solvers ----------------------------------------------------
    def _pcg(self, S, rhs, products):
        n_s = S.size
        colnorm2 = float(np.sum(self.forward(_unit(S[0], self.b, self._n, self.dtype)) ** 2))
        products[0] += 1
        if not colnorm2 > 0.0:
            return np.zeros(n_s, self.dtype), {"converged": False, "iterations": 0,
                                               "breakdown": "zero column", "min_rayleigh": 0.0}
        minv = 1.0 / colnorm2

        def A(v):
            full = np.zeros(self._n, self.dtype); full[S] = v
            products[0] += 2
            return self.adjoint(self.forward(full))[S]

        x = np.zeros(n_s, self.dtype); r = rhs.copy(); zv = minv * r; p = zv.copy()
        rz = float(r @ zv); bnorm = float(np.linalg.norm(rhs)); min_rq = math.inf
        it, converged, breakdown = 0, bnorm == 0.0, None
        while not converged and it < self.max_iters:
            Ap = A(p); pAp = float(p @ Ap); pp = float(p @ p)
            rq = pAp / pp if pp > 0 else 0.0
            min_rq = min(min_rq, rq)
            if not pAp > 1e-300 * max(1.0, pp):
                breakdown = "nonpositive curvature: rank deficient or indefinite"
                break
            a = rz / pAp
            x = x + a * p; r = r - a * Ap; it += 1
            if np.linalg.norm(r) <= self.rtol * bnorm:
                converged = True
                break
            zv = minv * r; rz2 = float(r @ zv); p = zv + (rz2 / rz) * p; rz = rz2
        return x, {"converged": bool(converged), "iterations": it, "breakdown": breakdown,
                   "min_rayleigh": (None if min_rq is math.inf else min_rq),
                   "final_relative_residual": float(np.linalg.norm(r) / bnorm) if bnorm else 0.0,
                   "jacobi_diagonal": colnorm2}

    def _lsqr(self, S, rhs, products):
        from scipy.sparse.linalg import LinearOperator, lsqr
        m2, n_s = self.b.size, S.size

        def mv(v):          # [[I, -F_S], [F_S^T, 0]] [dr; d]
            dr, d = v[:m2], v[m2:]
            full = np.zeros(self._n, self.dtype); full[S] = d
            products[0] += 2
            return np.concatenate([dr - self.forward(full), self.adjoint(dr)[S]])

        def rmv(w):         # transpose: [[I, F_S], [-F_S^T, 0]] [w1; w2]
            w1, w2 = w[:m2], w[m2:]
            full = np.zeros(self._n, self.dtype); full[S] = w2
            products[0] += 2
            return np.concatenate([w1 + self.forward(full), -self.adjoint(w1)[S]])

        op = LinearOperator((m2 + n_s, m2 + n_s), matvec=mv, rmatvec=rmv, dtype=self.dtype)
        full_rhs = np.concatenate([np.zeros(m2, self.dtype), rhs])
        sol = lsqr(op, full_rhs, atol=self.rtol, btol=self.rtol, iter_lim=self.max_iters)
        d = sol[0][m2:]
        res = float(np.linalg.norm(mv(sol[0]) - full_rhs) / max(np.linalg.norm(full_rhs), 1e-300))
        return d, {"converged": bool(sol[1] in (1, 2) and res <= 1e-8), "istop": int(sol[1]),
                   "iterations": int(sol[2]), "lsqr_r1norm": float(sol[3]),
                   "condition_estimate": float(sol[6]), "final_relative_residual": res,
                   "breakdown": None}

    # -- one correction round --------------------------------------------
    def __call__(self, x, y, ledger=None):
        t0 = time.perf_counter(); products = [0]
        x0 = np.asarray(x, self.dtype); self._n = x0.size
        lam = self.lam
        S_mask = np.abs(x0) > self.threshold
        for i in self._remove:
            S_mask[i] = False
        for i in self._add:
            S_mask[i] = True
        S = np.flatnonzero(S_mask)
        info = {"kind": "lasso_support_" + self.solver, "round": len(self.rounds) + 1,
                "support_size": int(S.size), "removed_from_support": len(self._remove),
                "added_to_support": len(self._add), "dtype": str(np.dtype(self.dtype))}
        if S.size == 0 or (self.max_support is not None and S.size > self.max_support):
            info.update(products=products[0], seconds=time.perf_counter() - t0,
                        support_valid=False, reason="empty or oversized support")
            self.rounds.append(info)
            return None, None, info
        xS = x0.copy(); xS[~S_mask] = 0.0
        s = np.sign(xS[S])
        r0 = self.forward(xS) - self.b; g = self.adjoint(r0); products[0] += 2
        inactive = np.abs(g[~S_mask]) / lam
        info.update(active_deviation_before=float(np.max(np.abs(g[S] + lam * s)) / lam),
                    inactive_max_before=float(inactive.max()) if inactive.size else 0.0,
                    inactive_violations_before=int(np.count_nonzero(inactive > 1.0)),
                    zeroed_below_threshold=int(np.count_nonzero((x0 != 0) & ~S_mask)))
        rhs = -(g[S] + lam * s)
        d, lin = self._pcg(S, rhs, products) if self.solver == "pcg" else self._lsqr(S, rhs, products)
        info["linear_solve"] = lin
        x_new = xS.copy(); x_new[S] += d
        r_new = self.forward(x_new) - self.b; g_new = self.adjoint(r_new); products[0] += 2
        flips = np.flatnonzero(np.sign(x_new[S]) != s)
        inactive_new = np.abs(g_new[~S_mask]) / lam
        violators = np.flatnonzero(~S_mask)[inactive_new > 1.0] if inactive_new.size else np.zeros(0, int)
        info.update(sign_flips=int(flips.size),
                    active_deviation_after=float(np.max(np.abs(g_new[S] + lam * s)) / lam),
                    inactive_max_after=float(inactive_new.max()) if inactive_new.size else 0.0,
                    inactive_violations_after=int(violators.size),
                    correction_norm=float(np.linalg.norm(d)),
                    rank_deficient=bool(lin.get("breakdown")),
                    support_valid=bool(lin["converged"] and flips.size == 0 and violators.size == 0
                                       and not lin.get("breakdown")))
        # Active-set safeguard for the NEXT round only; this round's proposal is
        # returned as computed and judged by the caller's criterion.
        self._remove |= set(int(i) for i in S[flips])
        self._add |= set(int(i) for i in violators[:1000])
        J = 0.5 * float(r_new @ r_new) + lam * float(np.abs(x_new).sum())
        dval = -float(self.b @ r_new) - 0.5 * float(r_new @ r_new)
        info.update(f64_J=J, f64_dual_same_witness=dval,
                    f64_discrepancy=abs(J - dval) / max(abs(J), abs(dval)),
                    f64_witness_in_domain=bool(np.max(np.abs(g_new)) <= lam),
                    products=products[0], seconds=time.perf_counter() - t0)
        self.rounds.append(info)
        return (jnp.asarray(x_new, jnp.float32), jnp.asarray(r_new, jnp.float32), info)


def _unit(i, b, n, dtype):
    e = np.zeros(n, dtype); e[i] = 1.0
    return e


# ---------------------------------------------------------------------------
# The safeguard.
# ---------------------------------------------------------------------------
def safeguarded_refine(comp, x, y, data=None, *, proposals, tol=None,
                       budget=RefineBudget(), assess=None):
    """Refine (x, y) by bounded, assessed proposals; never return a worse pair.

    ``proposals`` is one callable or a sequence of callables
    ``propose(x, y, ledger) -> (x_new, y_new, info)`` (``None`` withdraws);
    they are used in round-robin order for at most ``budget.max_steps`` steps.
    Each proposal is assessed by ``assess`` (default: criterion (7) through the
    family's public ``objective_bound``/``acceptance`` with recovery, as the
    solver would).  A proposal replaces the current pair only if it is valid
    and its assessed discrepancy is strictly smaller; otherwise the previous
    assessed pair is kept and the refusal is recorded.  Stops at the first
    assessed pair that meets the criterion, or on any exhausted bound.

    Returns ``(x_best, witness_best, ledger)`` where ``witness_best`` is the
    certifying dual the assessment selected for ``x_best`` (the pair the
    criterion was evaluated on, in original coordinates)."""
    t0 = time.perf_counter()
    if callable(proposals):
        proposals = [proposals]
    proposals = list(proposals)
    if not proposals:
        raise ValueError("at least one proposal is required")
    assess = criterion7_assessor(comp, data, tol) if assess is None else assess
    ledger = {"budget": asdict(budget), "steps": [], "products": 0, "assessments": 0,
              "stop_reason": None, "improved": False}

    def over_budget():
        if time.perf_counter() - t0 > budget.max_seconds:
            return "max_seconds"
        if ledger["products"] >= budget.max_products:
            return "max_products"
        if ledger["assessments"] >= budget.max_assessments:
            return "max_assessments"
        return None

    best = assess(x, y)
    ledger["assessments"] += 1; ledger["products"] += best["products"]
    x_best, y_best, w_best = x, y, best["witness"]
    ledger["start"] = {k: best[k] for k in ("score", "valid", "target_met", "products")}
    ledger["start"]["acceptance"] = best["acceptance"]
    if best["target_met"]:
        ledger["stop_reason"] = "criterion met at the start"
    for step in range(budget.max_steps):
        if ledger["stop_reason"]:
            break
        reason = over_budget()
        if reason:
            ledger["stop_reason"] = reason
            break
        propose = proposals[step % len(proposals)]
        row = {"step": step, "score_before": best["score"]}
        try:
            xp, yp, info = propose(x_best, y_best, ledger)
        except Exception as e:                       # a failing proposal is refused, not fatal
            xp, yp, info = None, None, {"kind": "error", "error": repr(e)[:300], "products": 0}
        ledger["products"] += int(info.get("products", 0))
        row["proposal"] = info
        valid_shape = (xp is not None and yp is not None and np.asarray(xp).shape == (comp.n,)
                       and np.asarray(yp).shape == (comp.m,))
        if not valid_shape or not _finite(xp) or not _finite(yp):
            row["decision"] = "refused: proposal withdrawn or invalid (shape/finiteness)"
            ledger["steps"].append(row)
            if len(proposals) == 1 and not getattr(propose, "stateful", False):
                ledger["stop_reason"] = "proposal refused"
                break
            continue
        reason = over_budget()
        if reason:
            row["decision"] = "not assessed: " + reason
            ledger["steps"].append(row); ledger["stop_reason"] = reason
            break
        cand = assess(xp, yp)
        ledger["assessments"] += 1; ledger["products"] += cand["products"]
        row.update(score_after=cand["score"], valid=cand["valid"], target_met=cand["target_met"],
                   acceptance=cand["acceptance"], production=cand["production"])
        if cand["valid"] and cand["score"] < best["score"]:
            best, x_best, y_best, w_best = cand, xp, yp, cand["witness"]
            row["decision"] = "accepted: assessed discrepancy decreased"
            ledger["improved"] = True
        else:
            row["decision"] = ("refused: invalid witness or infeasible point" if not cand["valid"]
                               else "refused: assessed discrepancy not smaller")
        ledger["steps"].append(row)
        if best["target_met"]:
            ledger["stop_reason"] = "criterion (7) met"
            break
        if (row["decision"].startswith("refused") and len(proposals) == 1
                and not getattr(propose, "stateful", False)):
            # A deterministic proposal refused from the same pair would only
            # repeat itself; stop instead of spending the budget on it.
            ledger["stop_reason"] = "proposal refused"
            break
    if ledger["stop_reason"] is None:
        ledger["stop_reason"] = "max_steps"
    ledger["final"] = {k: best[k] for k in ("score", "valid", "target_met")}
    ledger["final"]["acceptance"] = best["acceptance"]
    ledger["final"]["production"] = best["production"]
    ledger["seconds"] = time.perf_counter() - t0
    ledger["solved"] = bool(best["target_met"])
    return x_best, w_best, ledger
