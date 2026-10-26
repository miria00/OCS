"""Experimental fixed-metric PDHG with auditable adaptive ball-prox errors.

This is a host-scheduled correctness prototype with JIT GPU updates, not a
public OCS API or a performance implementation. It uses the reviewed numerical
error estimates from inexact_full_ball_prox; it does not turn them into interval
certificates or a floating-point convergence theorem.

Problem: mu/2 * ||x||^2 + sum_g,b radius[g,b] * ||(K_g x + l_g)[b]||.
T is fixed positive diagonal. Each fixed Sigma_g is the ACTUAL rounded matrix
Q diag(sigma) Q.T, including its small basis orthogonality defect. The caller
provides an independently justified strict coupling-norm upper bound and
freezes every operator/data/metric during the solve.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from inexact_full_ball_prox import (
    BallMetric, STATUS_NAMES, adaptive_ball_prox, prepare_budget, summable_budget,
)


@dataclass(frozen=True)
class Group:
    name: str
    matvec: Callable
    rmatvec: Callable
    metric: BallMetric
    radius: jax.Array
    offset: jax.Array
    operator_digest: str


@dataclass(frozen=True)
class State:
    x: jax.Array
    xprev: jax.Array
    ys: tuple
    iteration: int = 0  # completed, committed updates; never a chunk-local count
    status: str = "running"
    reason: str = ""
    budget_sum: float = 0.
    estimate_sum: float = 0.
    last_check: dict | None = None


@dataclass(frozen=True)
class AcceptancePolicy:
    """Caller-declared source identity; arbitrary callback semantics are not inferred.

    The caller must freeze the referenced evaluator implementation and closed-
    over problem data. The loop checks this identity against its contract.
    """
    evaluate: Callable
    declared_source_identity: str

    def __call__(self, x, ys):
        return self.evaluate(x, ys)


def _bytes_digest(array):
    a = np.asarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


class InexactLoop:
    def __init__(self, groups, tau, *, mu, budget_c, budget_p,
                 coupling_upper, coupling_provenance, max_inner=80,
                 tolerance=1e-3, acceptance_policy_identity,
                 check_every=5, implementation_version="shared-shape-inexact-v2"):
        self.groups = tuple(groups)
        self.tau = jnp.asarray(tau)
        if self.tau.ndim != 1 or not len(self.tau) or self.tau.dtype != jnp.float32:
            raise ValueError("This prototype requires a nonempty float32 diagonal T")
        if not np.isfinite(np.asarray(self.tau)).all() or not (np.asarray(self.tau) > 0).all():
            raise ValueError("T must be finite and strictly positive")
        self.mu, self.c, self.p = map(float, (mu, budget_c, budget_p))
        if not math.isfinite(self.mu) or self.mu <= 0:
            raise ValueError("A finite positive ridge is required")
        summable_budget(0, self.c, self.p)
        if not math.isfinite(coupling_upper) or not 0 <= coupling_upper < 1:
            raise ValueError("A justified strict coupling norm bound <1 is required")
        if not coupling_provenance:
            raise ValueError("The norm-bound provenance must be recorded")
        if not isinstance(max_inner, int) or max_inner < 0:
            raise ValueError("A finite nonnegative inner iteration cap is required")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("A finite positive outer tolerance is required")
        if not isinstance(check_every, int) or check_every < 1:
            raise ValueError("A fixed positive integer check interval is required")
        if not isinstance(acceptance_policy_identity, str) or not acceptance_policy_identity:
            raise ValueError("A declared acceptance-policy source identity is required")
        if not isinstance(implementation_version, str) or not implementation_version:
            raise ValueError("An explicit implementation version is required")
        self.check_every = check_every
        self.acceptance_policy_identity = acceptance_policy_identity
        self.max_inner, self.tolerance = max_inner, float(tolerance)
        if not self.groups or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Nonempty uniquely named groups required")
        dimensions = []
        group_contracts = []
        for g in self.groups:
            B, d = g.metric.sigma.shape
            if not g.operator_digest or g.metric.Q.dtype != self.tau.dtype:
                raise ValueError("Frozen operator identity and matching metric dtype required")
            if g.offset.shape != (B, d) or g.radius.shape != (B,):
                raise ValueError("Group offset/radius shapes must match the prepared metric")
            if (not np.isfinite(np.asarray(g.offset)).all()
                    or not np.isfinite(np.asarray(g.radius)).all()
                    or not (np.asarray(g.radius) >= 0).all()):
                raise ValueError("Finite offsets and nonnegative finite radii required")
            dimensions.append(B * d)
            group_contracts.append(dict(name=g.name, shape=[B, d], operator=g.operator_digest,
                arrays={name: _bytes_digest(a) for name, a in zip(
                    ("Q", "sigma", "inverse_discrepancy", "orthogonality_estimate", "radius", "offset"),
                    (*g.metric, g.radius, g.offset))}))
        self.shares = tuple(d / sum(dimensions) for d in dimensions)
        helper_path = Path(sys.modules[adaptive_ball_prox.__module__].__file__)
        implementation = dict(version=implementation_version,
            loop_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest())
        self.contract = dict(schema=2, model="ridge plus product group norms",
            ordering="dual first; xbar=2*x-xprev; ordinary fixed-metric PDHG",
            mu=self.mu, budget_c=self.c, budget_p=self.p, tolerance=self.tolerance,
            tau=_bytes_digest(self.tau), groups=group_contracts,
            group_shares=self.shares, max_inner=max_inner,
            implementation=implementation,
            acceptance_policy_identity=acceptance_policy_identity, check_every=check_every,
            coupling_upper=float(coupling_upper), coupling_provenance=coupling_provenance,
            error_scope="absolute numerical type2 estimate; not interval certified")
        self.contract_hash = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()

        def propose(x, xprev, ys, group_totals, block_budgets):
            xbar = 2 * x - xprev
            outs, candidates, inputs = [], [], []
            for g, y, total, budgets in zip(self.groups, ys, group_totals, block_budgets):
                z = g.matvec(xbar).reshape(g.offset.shape) + g.offset
                # Sigma is Q diag(sigma) Q.T, not a basis/inverse shortcut.
                qt_z = jnp.einsum("bji,bj->bi", g.metric.Q, z, precision=lax.Precision.HIGHEST)
                sz = jnp.einsum("bij,bj->bi", g.metric.Q, g.metric.sigma * qt_z,
                                precision=lax.Precision.HIGHEST)
                v = y + sz
                out = adaptive_ball_prox(g.metric, v, g.radius, total, budgets,
                                         max_iterations=self.max_inner)
                inputs.append(v); candidates.append(out["y"]); outs.append(out)
            pullback = sum((g.rmatvec(y.reshape(-1)) for g, y in zip(self.groups, candidates)),
                           jnp.zeros_like(x))
            next_x = (x - self.tau * pullback) / (1 + self.mu * self.tau)
            return next_x, tuple(candidates), tuple(inputs), tuple(outs)
        self._propose = jax.jit(propose)

    def initial_state(self, x=None):
        x = jnp.zeros_like(self.tau) if x is None else jnp.asarray(x, self.tau.dtype)
        if x.shape != self.tau.shape or not np.isfinite(np.asarray(x)).all():
            raise ValueError("Initial x must be finite and match T")
        return State(x=x, xprev=x, ys=tuple(jnp.zeros_like(g.offset) for g in self.groups))

    def point_digest(self, state):
        return hashlib.sha256("|".join(_bytes_digest(a) for a in (state.x, *state.ys)).encode()).hexdigest()

    def step(self, state):
        if state.status != "running":
            return state, None  # terminal failure/solved states cannot be resumed silently
        k = state.iteration
        requested = summable_budget(k, self.c, self.p)
        # One budget spans all heterogeneous groups. Host preparation rounds
        # allocations down; no positive floor is added after underflow.
        allocated = [requested * a for a in self.shares]
        # Resolve the possible last-ulp excess of multiplying shares in host
        # arithmetic; this only tightens allocations and cannot raise a budget.
        while math.fsum(allocated) > requested:
            largest = max(range(len(allocated)), key=allocated.__getitem__)
            allocated[largest] = math.nextafter(allocated[largest], 0.)
        group_allocation = prepare_budget(requested, len(self.groups), per_block=allocated)
        splits = [prepare_budget(float(limit), len(g.radius))
                  for g, limit in zip(self.groups, np.asarray(group_allocation.per_block))]
        newx, newys, inputs, outs = self._propose(state.x, state.xprev, state.ys,
            tuple(a.total for a in splits), tuple(a.per_block for a in splits))
        jax.block_until_ready((newx, newys, inputs, outs))
        host_outs = [{key: np.asarray(value) for key, value in out.items()} for out in outs]
        estimates = [float(out["total_error_upper_estimate"]) for out in host_outs]
        total_estimate = math.fsum(estimates)
        if total_estimate > 0 and math.isfinite(total_estimate):
            total_estimate = math.nextafter(total_estimate, math.inf)
        finite_x = np.isfinite(np.asarray(newx)).all()
        accepted = (all(bool(out["success"]) for out in host_outs) and finite_x
                    and math.isfinite(total_estimate)
                    and total_estimate <= float(group_allocation.total))
        record = dict(iteration=k, requested_total=requested,
            effective_global_total=float(group_allocation.total),
            group_totals=[float(a.total) for a in splits],
            group_status=[STATUS_NAMES[int(out["global_status"])] for out in host_outs],
            estimated_total=total_estimate, committed=bool(accepted),
            # Preserve failed candidates too; they are not the committed state.
            inputs=tuple(np.asarray(v) for v in inputs),
            candidates=tuple(np.asarray(y) for y in newys),
            block_estimates=tuple(out["error_upper_estimate"] for out in host_outs),
            block_budgets=tuple(np.asarray(a.per_block) for a in splits),
            inner_iterations=tuple(out["iterations"] for out in host_outs))
        if not accepted:
            reason = "nonfinite_primal" if not finite_x else ",".join(record["group_status"])
            if all(bool(out["success"]) for out in host_outs) and finite_x:
                reason = "heterogeneous_total_budget_exceeded"
            return replace(state, status="inner_failure", reason=reason), record
        return State(newx, state.x, newys, k + 1, "running", "",
                     math.fsum((state.budget_sum, requested)),
                     math.fsum((state.estimate_sum, total_estimate)), state.last_check), record

    def run_chunk(self, state, updates, *, evaluate=None):
        """Pause after updates; checks follow GLOBAL k, independent of chunk cuts.

        evaluate is an AcceptancePolicy whose declared identity must match.
        It returns public same-point r_primal/r_dual plus an
        independent numerical objective bracket and raw-dual feasibility.
        The loop does not accept a callback's bare 'solved' boolean.
        """
        if not isinstance(updates, int) or updates < 0:
            raise ValueError("A nonnegative update cap is required")
        if evaluate is not None and (not isinstance(evaluate, AcceptancePolicy)
                or evaluate.declared_source_identity != self.acceptance_policy_identity):
            raise ValueError("Runtime acceptance policy does not match the frozen contract")
        records = []
        for _ in range(updates):
            state, row = self.step(state)
            if row is None:
                break
            records.append(row)
            if state.status != "running":
                break
            if evaluate is not None and state.iteration % self.check_every == 0:
                check = dict(evaluate(state.x, state.ys))
                check["point_digest"] = self.point_digest(state)
                values = [check[k] for k in ("r_primal", "r_dual", "relative_bracket")]
                accepted = (all(math.isfinite(float(v)) and v >= 0 for v in values)
                    and max(values[:2]) <= self.tolerance
                    and 0 <= values[2] <= self.tolerance
                    and check.get("raw_dual_feasible") is True)
                state = replace(state, last_check=check,
                    status="solved" if accepted else "running")
                if accepted:
                    break
        return state, records

    def save_state(self, prefix, state):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays_path, meta_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
        if arrays_path.exists() or meta_path.exists():
            raise FileExistsError("Preserve checkpoints; choose a fresh prefix")
        np.savez_compressed(arrays_path, x=np.asarray(state.x), xprev=np.asarray(state.xprev),
                             **{f"y_{i}": np.asarray(y) for i, y in enumerate(state.ys)})
        meta = dict(schema=2, contract_hash=self.contract_hash,
            arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            iteration=state.iteration, status=state.status, reason=state.reason,
            budget_sum=state.budget_sum, estimate_sum=state.estimate_sum,
            last_check=state.last_check)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    def load_state(self, prefix):
        prefix = Path(prefix)
        arrays_path, meta_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        if meta["schema"] != 2 or meta["contract_hash"] != self.contract_hash:
            raise ValueError("Checkpoint problem/metric/budget/implementation/acceptance contract mismatch")
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != meta["arrays_sha256"]:
            raise ValueError("Checkpoint array checksum mismatch")
        if (type(meta["iteration"]) is not int or meta["iteration"] < 0
                or meta["status"] not in ("running", "solved", "inner_failure")
                or any(not math.isfinite(meta[k]) or meta[k] < 0
                       for k in ("budget_sum", "estimate_sum"))):
            raise ValueError("Invalid checkpoint status/counter/error totals")
        with np.load(arrays_path, allow_pickle=False) as data:
            arrays = [data["x"], data["xprev"], *(data[f"y_{i}"] for i in range(len(self.groups)))]
            shapes = [self.tau.shape, self.tau.shape, *(g.offset.shape for g in self.groups)]
            if any(a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all()
                   for a, shape in zip(arrays, shapes)):
                raise ValueError("Invalid checkpoint array shape/dtype/value")
            x, prev, *ys = [jnp.asarray(a) for a in arrays]
        state = State(x, prev, tuple(ys), meta["iteration"], meta["status"], meta["reason"],
                      meta["budget_sum"], meta["estimate_sum"], meta["last_check"])
        if state.status == "solved":
            check = state.last_check
            if (not isinstance(check, dict) or check.get("point_digest") != self.point_digest(state)
                    or check.get("raw_dual_feasible") is not True
                    or any(not math.isfinite(check.get(k, math.inf))
                           or not 0 <= check[k] <= self.tolerance
                           for k in ("r_primal", "r_dual", "relative_bracket"))):
                raise ValueError("Solved checkpoint lacks a valid same-point acceptance record")
        return state
