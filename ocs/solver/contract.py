"""Opt-in explicitly scaled certificate contract (declared absolute budgets).

The default relative certificate (``stopping.make_relative_components`` plus
``domain_checks.make_domain_checks``) divides each graph displacement by its
own point scale. On a homogeneous block that is being satisfied the numerator
and the denominator are the same number, so the ratio is exactly one at every
nonzero point and zero only at the exact solution. That is a property of the
normalization in exact arithmetic, not of floating point.

This module adds a SEPARATE contract that the caller must request and whose
absolute parts the caller must declare. Nothing here is inferred from data,
from an additive objective constant, or from a unit floor. It NEVER replaces
the strict relative residual, and it never changes a status on its own: the
caller reports three independent facts (strict pass, contract pass,
termination reason), see ``ocs.model.compile.Solution``.

Per nonsmooth dual block i, with the default scales
``P_i = max(||K_i x||, ||l_i||)`` and ``Y_i = ||y_i||`` (and the same
quadratic cone reference scales as the default), and the requested tolerance
``eps``::

    A_i = eps * P_i + a_i          (primal / offset units)
    B_i = eps * Y_i + b_i          (dual / multiplier units)
    t_i = A_i / B_i                certificate step: it CHANGES with the budgets
    z_i = prox_{f_i}^{(1/t_i) I}(s_i + t_i y_i)
    p_i = y_i + (s_i - z_i) / t_i  in  subdifferential f_i(z_i), exactly

and the block's contract ratio is ``||z_i - s_i|| / A_i``. A ratio at most one
gives ``||z_i - s_i|| <= A_i`` and, because the step is A_i/B_i,
``||p_i - y_i|| = ||z_i - s_i|| / t_i <= B_i``. Widening only the primal
denominator while keeping the default step P_i/Y_i would silently drop that
second bound; this evaluator never does so.

POSITIVE BRANCH. The witness construction above, and every bound derived from
it, requires ``A_i > 0`` AND ``B_i > 0``. That is the positive-branch
assumption; it is recorded per block and reported as
``contract_witness_active`` (1.0 only when it held on every block that was
tested). When ``A_i = 0`` or ``B_i = 0`` the evaluator does NOT form the
witness: it falls back to the default zero-scale rule, unit-step exact
membership, which is 0 when the prox leaves the point fixed and +inf
otherwise. That branch is a stronger statement than the witness bound, not a
weaker one, but it is not the theorem's clause, so it is flagged rather than
silently folded in. Every ratio uses 0/0 = 0 and nonzero/0 = +inf; a nonzero
numerator is never divided into zero.

Smooth blocks use ``||y_i - grad f_i(s_i)|| <= eps * Y_i^sm + b_i``. The primal
side uses the variable budget ``a_x`` (units of x) and the stationarity budget
``b_c`` (cost / gradient units): a routed nonsmooth g uses the same budgeted
graph construction at ``(x, -q)`` with ``A_g = eps ||x|| + a_x`` and
``B_g = eps G + b_c``; otherwise ``||q + grad g(x)|| <= eps * scale + b_c``.

Domain and polar guards use the same budgets per cone row, with the violation
measured in that row's own units (the cone violations are positively
homogeneous of degree one).

COMPLEMENTARITY (corrected 2026-09-19). Let (z_i, p_i) be the witness, so that
for a cone ``<z_i, p_i> = 0`` and for a box ``sigma_C(p_i) = <z_i, p_i>``.
Writing the defect as

    <s_i, y_i> - <z_i, p_i> = <s_i - z_i, y_i> + <z_i, y_i - p_i>

and using ``||s_i - z_i|| <= A_i``, ``||y_i - p_i|| <= B_i`` and
``||z_i|| <= ||s_i|| + A_i`` gives the allowance actually implied by the
witness bound,

    |<s_i, y_i>|  <=  A_i ||y_i|| + B_i ||s_i|| + A_i B_i ,

with an extra ``B_i ||bounds_i||`` for a box, since its support function is
Lipschitz with that constant. The FULL budgets A_i and B_i appear here, not
their absolute parts a_i and b_i. An earlier draft compared the defect with
``eps`` times an objective-unit scale plus ``sum_i (a_i||y_i|| + b_i||s_i||
+ a_i b_i)``, and asserted that this followed from the witness bound. It does
not. Verified counterexample, pinned in
``tests/test_certificate_contract.py::test_complementarity_counterexample``:
one scalar block with ``A = 0.125``, ``B = 1``, ``s = 1``, ``y = 1``, witness
``z = 0.875``, ``p = 0`` -- admissible, since ``|s - z| = 0.125 <= A`` and
``|y - p| = 1 <= B`` -- has defect ``|<s,y>| = 1.0`` while the old expression
at ``eps = 0.1``, ``P = Y = 1`` (so ``a = 0.025``, ``b = 0.9``) allows only
``0.9475``. The corrected allowance is ``0.125 + 1 + 0.125 = 1.25``. The old
expression is therefore not implied by the witness bound: it is STRICTER than
the theorem permits and can reject a pair the graph test accepted. It was
never unsound, and the correction does not weaken any accepted claim; it makes
the complementarity term the consistency check it is stated to be.

Unit covariance. Under Proposition p-scaleinv (K_i' = alpha_i K_i,
l_i' = alpha_i l_i, f_i' = gamma f_i(./alpha_i), y_i' = (gamma/alpha_i) y_i,
c' = gamma c, g' = gamma g, x' = x) every ratio here is unchanged provided the
declared budgets transform as a_i' = alpha_i a_i, b_i' = (gamma/alpha_i) b_i,
a_x' = a_x and b_c' = gamma b_c. A scalar budget applied to every block is a
declaration that those blocks share units; it is the caller's declaration.

The strict relative residual is still computed and reported by the caller
beside this contract; this module never replaces it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import jax.numpy as jnp

from ..prox.library import Box, Zero
from .domain_checks import _cone_violation, _indicator, _norm, _rows, _support
from .stopping import stable_norm

#: Stopping modes. ``strict_only`` is the shipped, compatibility-preserving
#: mode and the DEFAULT: the iteration loop is exactly the default loop and the
#: contract only classifies the point that loop already reached.
#: ``strict_or_contract`` also stops the loop on the contract. It is
#: EXPERIMENTAL and must be named explicitly; it is never a default, and it is
#: not a speedup at the original strict criterion.
STOPPING_MODES = ("strict_only", "strict_or_contract")
EXPERIMENTAL_STOPPING = ("strict_or_contract",)


def _budget_vector(value, count, name):
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        values = (float(value),) * count
    else:
        try:
            values = tuple(float(v) for v in value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a real number or a sequence of reals") from exc
        if len(values) != count:
            raise ValueError(f"{name} needs one value per dual block ({count}), got {len(values)}")
    for v in values:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} entries must be finite and nonnegative, got {v!r}")
    return values


def _budget_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return value


@dataclass(frozen=True)
class ScaledContract:
    """Declared absolute budgets for the opt-in scaled certificate contract.

    ``primal_budget``: a_i, per dual block in term order (or one real for all),
    in the units of that block's offset K_i x + l_i.
    ``dual_budget``: b_i, per dual block (or one real), in multiplier units.
    ``variable_budget``: a_x, in the units of x (routed primal term g).
    ``stationarity_budget``: b_c, in cost / gradient units.
    ``stopping``: ``"strict_only"`` (the DEFAULT) keeps the default stopping
    test and iterates exactly as the default loop does, using the contract only
    to classify the returned point; ``"strict_or_contract"`` also stops the
    iteration once the contract holds and is EXPERIMENTAL.

    All budgets are declared by the caller; none is inferred. Declaring a
    budget is a statement about the units of the problem, so the caller must
    also state the selection rule it used; see ``docs/stopping_contract.md``.
    """
    primal_budget: object
    dual_budget: object
    variable_budget: float = 0.0
    stationarity_budget: float = 0.0
    stopping: str = "strict_only"

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            unknown = set(value) - {"primal_budget", "dual_budget",
                                    "variable_budget", "stationarity_budget", "stopping"}
            if unknown:
                raise ValueError(f"unknown scaled-contract fields: {sorted(unknown)}")
            if "primal_budget" not in value or "dual_budget" not in value:
                raise ValueError("a scaled contract must declare primal_budget and dual_budget")
            return cls(**value)
        raise TypeError("certificate_contract must be a ScaledContract or a dict of declared budgets")

    def __post_init__(self):
        if self.stopping not in STOPPING_MODES:
            raise ValueError(f"stopping must be one of {STOPPING_MODES}")

    @property
    def stops_the_loop(self):
        """True when this contract is allowed to end the iteration itself."""
        return self.stopping == "strict_or_contract"

    def resolve(self, count):
        return (_budget_vector(self.primal_budget, count, "primal_budget"),
                _budget_vector(self.dual_budget, count, "dual_budget"),
                _budget_scalar(self.variable_budget, "variable_budget"),
                _budget_scalar(self.stationarity_budget, "stationarity_budget"))

    def describe(self, count):
        a, b, ax, bc = self.resolve(count)
        return {"primal_budget": list(a), "dual_budget": list(b),
                "variable_budget": ax, "stationarity_budget": bc,
                "stopping": self.stopping, "declared_by_caller": True,
                "experimental_stopping": self.stopping in EXPERIMENTAL_STOPPING,
                "stops_the_loop": self.stops_the_loop}


def budget_ratio(numerator, budget):
    """numerator / budget with 0/0 = 0 and nonzero/0 = +inf; nonfinite fails."""
    valid = jnp.isfinite(numerator) & jnp.isfinite(budget)
    ratio = numerator / jnp.where(budget > 0, budget, 1.0)
    ratio = jnp.where(budget > 0, ratio, jnp.where(numerator == 0, 0.0, jnp.inf))
    return jnp.where(valid, ratio, jnp.inf)


def complementarity_allowance(primal_allow, dual_allow, s_norm, y_norm, bound_norm=None):
    """The defect allowance actually implied by the witness bound.

    ``A ||y|| + B ||s|| + A B`` (plus ``B ||bounds||`` for a box).  ``A`` and
    ``B`` are the FULL budgets ``eps * scale + absolute`` used by the block's
    graph test, not their absolute parts.  See the module docstring for the
    derivation and for the counterexample that refutes the absolute-part-only
    expression this replaces.
    """
    allowance = primal_allow * y_norm + dual_allow * s_norm + primal_allow * dual_allow
    if bound_norm is not None:
        allowance = allowance + dual_allow * bound_norm
    return allowance


def _budget_graph(block, point, dual, primal_allow, dual_allow):
    """Budgeted prox graph test. Returns (ratio, step, active).

    ``active`` is the positive branch A > 0 and B > 0, the assumption under
    which the witness (z, p) exists and both witness bounds hold. When it is
    false the default zero-scale rule is used instead and ``active`` says so.
    """
    active = (primal_allow > 0) & (dual_allow > 0)
    step = jnp.where(active, primal_allow / jnp.where(dual_allow > 0, dual_allow, 1.0), 1.0)
    projected = block.prox(point + step * dual, step)
    distance = stable_norm(point - projected)
    ratio = jnp.where(active, distance / jnp.where(primal_allow > 0, primal_allow, 1.0),
                      jnp.where(distance == 0, 0.0, jnp.inf))
    valid = (jnp.isfinite(primal_allow) & jnp.isfinite(dual_allow)
             & jnp.isfinite(step) & (step > 0) & jnp.isfinite(distance))
    return jnp.where(valid, ratio, jnp.inf), step, active


def _block_contract(block, s, y, kx, l, eps, a, b, primal_allow, dual_allow,
                    dual_scale=None, reference_scales=None):
    """Budgeted domain, polar, complementarity and its derived allowance.

    ``primal_allow`` / ``dual_allow`` are the block's A_i and B_i as used by
    its graph test; they carry the complementarity allowance.
    """
    if type(block) is Box:
        lo, hi = jnp.asarray(block.lo), jnp.asarray(block.hi)
        bounds = jnp.maximum(jnp.abs(jnp.where(jnp.isfinite(lo), lo, 0.0)),
                             jnp.abs(jnp.where(jnp.isfinite(hi), hi, 0.0)))
        bound_norm = _norm(jnp.broadcast_to(bounds, s.shape))
        offset_norm = jnp.asarray(0.0, s.dtype) if l is None else _norm(l)
        scale = jnp.maximum(jnp.maximum(_norm(kx), offset_norm), bound_norm)
        domain = budget_ratio(_norm(s - jnp.clip(s, lo, hi)), eps * scale + a)
        prohibited = jnp.where(~jnp.isfinite(hi), jnp.maximum(y, 0.0), 0.0)
        prohibited += jnp.where(~jnp.isfinite(lo), jnp.minimum(y, 0.0), 0.0)
        ys = _norm(y) if dual_scale is None else dual_scale
        polar = budget_ratio(_norm(prohibited), eps * ys + b)
        support = _support(block, y)
        complementarity = jnp.abs(support - jnp.vdot(s, y))
        allowance = complementarity_allowance(primal_allow, dual_allow,
                                              _norm(s), _norm(y), bound_norm)
        return domain, polar, complementarity, support, allowance

    sr, yr = _rows(block, s), _rows(block, y)
    offset_norm = (jnp.asarray(0.0, s.dtype) if l is None
                   else _norm(_rows(block, l), axis=1))
    scale = jnp.maximum(_norm(_rows(block, kx), axis=1), offset_norm)
    if reference_scales is not None:
        scale = jnp.maximum(scale, reference_scales[2])
    normalized = sr / jnp.where(scale > 0, scale, 1.0)[:, None]
    raw = _cone_violation(block, normalized) * scale
    raw = jnp.where(scale > 0, raw, jnp.where(_norm(sr, axis=1) == 0, 0.0, jnp.inf))
    raw = jnp.where(jnp.isfinite(scale), raw, jnp.inf)
    domain = budget_ratio(raw, eps * scale + a)
    ys = _norm(yr, axis=1)
    if reference_scales is not None:
        ys = jnp.maximum(ys, reference_scales[3])
    if dual_scale is not None:
        ys = jnp.maximum(ys, dual_scale)
    yn = yr / jnp.where(ys > 0, ys, 1.0)[:, None]
    polar_raw = _cone_violation(block, yn, polar=True) * ys
    polar_raw = jnp.where(jnp.isfinite(ys), polar_raw, jnp.inf)
    polar = budget_ratio(polar_raw, eps * ys + b)
    # Absolute values before summing prevent cancellation between cones; the
    # allowance is derived rowwise and summed by Cauchy-Schwarz, which is why
    # the block Euclidean norms of s and y appear.
    complementarity = jnp.sum(jnp.abs(jnp.sum(sr * yr, axis=1)))
    allowance = complementarity_allowance(primal_allow, dual_allow, _norm(s), _norm(y))
    return (jnp.max(domain, initial=0.0), jnp.max(polar, initial=0.0),
            complementarity, jnp.asarray(0.0, s.dtype), allowance)


def make_contract_components(matvec, rmatvec, g_block, blocks, slices, contract, tol, *,
                             guards_enabled, adjoint_with_scale=None,
                             cone_reference_scales=None, zero_offsets=None, zero_cost=False):
    """Return evaluate(x, y, data) -> dict of scaled-contract ratios (pass <= 1).

    Mirrors the default relative evaluator term by term, replacing each
    relative denominator s by the declared budget eps*s + absolute budget and
    each certificate step by A/B. It never alters or replaces the default
    residual, which the caller evaluates and reports separately.
    """
    eps = float(tol)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("the contract tolerance must be finite and positive")
    spans = tuple((int(a), int(b)) for a, b in slices)
    a_blocks, b_blocks, a_x, b_c = ScaledContract.coerce(contract).resolve(len(spans))
    has_structural_facts = zero_offsets is not None or zero_cost
    zero_offsets = ((False,) * len(spans) if zero_offsets is None else tuple(zero_offsets))
    if len(zero_offsets) != len(spans):
        raise ValueError("one structural-zero flag is required per dual block")
    if type(g_block) is Zero:
        g_block = None
    gi = _indicator(g_block)
    indicators = [_indicator(block) for block in blocks]

    def pullback(y):
        if adjoint_with_scale is not None:
            return adjoint_with_scale(y)
        value = rmatvec(y)
        return value, stable_norm(value)

    def evaluate(x, y, data):
        l, c = data
        dt = x.dtype
        cone_scales, _objective_reference = (cone_reference_scales(data)
            if cone_reference_scales is not None else (None, None))
        kx = matvec(x)
        kty, pullback_scale = pullback(y)
        reference = jnp.zeros_like(y)
        graph = jnp.asarray(0.0, dt)
        witness_active = jnp.asarray(True)
        # A_i and B_i per block, exactly as used by the graph test, so that the
        # complementarity allowance below is the one the witness bound implies.
        block_allows = []
        for i, (blk, (a, b)) in enumerate(zip(blocks, spans)):
            s, yi, li = kx[a:b] + l[a:b], y[a:b], l[a:b]
            ai, bi = jnp.asarray(a_blocks[i], dt), jnp.asarray(b_blocks[i], dt)
            li_norm = jnp.asarray(0.0, li.dtype) if zero_offsets[i] else stable_norm(li)
            grad = getattr(blk, "smooth_grad", None)
            if grad is not None:
                current, anchor = grad(s), grad(li)
                scale = jnp.maximum(stable_norm(yi), jnp.maximum(
                    stable_norm(current), stable_norm(anchor)))
                dual_allow = eps * scale + bi
                ratio = budget_ratio(stable_norm(yi - current), dual_allow)
                block_allows.append((jnp.asarray(0.0, dt), dual_allow))
            else:
                scale = jnp.maximum(stable_norm(kx[a:b]), li_norm)
                dual_scale = stable_norm(yi)
                if cone_scales is not None:
                    scale = jnp.maximum(scale, cone_scales[i][0])
                    dual_scale = jnp.maximum(dual_scale, cone_scales[i][1])
                primal_allow, dual_allow = eps * scale + ai, eps * dual_scale + bi
                ratio, step, active = _budget_graph(blk, s, yi, primal_allow, dual_allow)
                anchor = jnp.where(active, (li - blk.prox(li, step)) / step, 0.0)
                witness_active = witness_active & active
                block_allows.append((primal_allow, dual_allow))
            reference = reference.at[a:b].set(anchor)
            graph = jnp.maximum(graph, ratio)

        q = kty + c
        _ref, reference_scale = pullback(reference)
        cost_norm = jnp.asarray(0.0, c.dtype) if zero_cost else stable_norm(c)
        stationarity_scale = jnp.maximum(jnp.maximum(pullback_scale, reference_scale), cost_norm)
        grad_g = getattr(g_block, "smooth_grad", None)
        ax, bc = jnp.asarray(a_x, dt), jnp.asarray(b_c, dt)
        g_primal_allow = eps * stable_norm(x) + ax
        g_dual_allow = eps * stationarity_scale + bc
        if g_block is not None and grad_g is None:
            stationarity, _step, g_active = _budget_graph(
                g_block, x, -q, g_primal_allow, g_dual_allow)
            witness_active = witness_active & g_active
        else:
            gx = jnp.zeros_like(x) if grad_g is None else grad_g(x)
            g0 = jnp.zeros_like(x) if grad_g is None else grad_g(jnp.zeros_like(x))
            scale = jnp.maximum(stationarity_scale, jnp.maximum(stable_norm(gx), stable_norm(g0)))
            stationarity = budget_ratio(stable_norm(q + gx), eps * scale + bc)

        if has_structural_facts:
            offsets_finite = jnp.asarray(True)
            for absent, (a, b) in zip(zero_offsets, spans):
                if not absent:
                    offsets_finite &= jnp.all(jnp.isfinite(l[a:b]))
        else:
            offsets_finite = jnp.all(jnp.isfinite(l))
        cost_finite = jnp.asarray(True) if zero_cost else jnp.all(jnp.isfinite(c))
        finite = (jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(y))
                  & offsets_finite & cost_finite & jnp.isfinite(stationarity_scale))

        domain = polar = comp_ratio = jnp.asarray(0.0, dt)
        guard_finite = jnp.asarray(True)
        if guards_enabled and (gi is not None or any(ind is not None for ind in indicators)):
            v = -(kty + c)
            complementarity = jnp.asarray(0.0, dt)
            allowance = jnp.asarray(0.0, dt)
            if gi is not None:
                dr, pr, cr, _support_value, al = _block_contract(
                    gi, x, v, x, None, eps, ax, bc, g_primal_allow, g_dual_allow,
                    dual_scale=stationarity_scale)
                domain, polar, complementarity, allowance = dr, pr, cr, al
            for i, (block, indicator, (a, b)) in enumerate(zip(blocks, indicators, spans)):
                s, yi = kx[a:b] + l[a:b], y[a:b]
                if indicator is not None:
                    refs = None if cone_scales is None else cone_scales[i]
                    pa, da = block_allows[i]
                    dr, pr, cr, _support_value, al = _block_contract(
                        indicator, s, yi, kx[a:b], None if zero_offsets[i] else l[a:b], eps,
                        jnp.asarray(a_blocks[i], dt), jnp.asarray(b_blocks[i], dt), pa, da,
                        reference_scales=refs)
                    domain, polar = jnp.maximum(domain, dr), jnp.maximum(polar, pr)
                    complementarity += cr
                    allowance += al
            # The allowance is the bound the witness implies; there is no
            # additional eps * objective-scale term, and none is needed.
            comp_ratio = budget_ratio(complementarity, allowance)
            guard_finite = jnp.isfinite(complementarity) & jnp.isfinite(allowance)

        def gate(value, extra=True):
            return jnp.where(finite & extra & ~jnp.isnan(value), value, jnp.inf)

        out = {
            "contract_graph": gate(graph),
            "contract_stationarity": gate(stationarity),
            "contract_domain": gate(domain, guard_finite),
            "contract_polar": gate(polar, guard_finite),
            "contract_complementarity": gate(comp_ratio, guard_finite),
        }
        out["contract_primal"] = jnp.maximum(out["contract_stationarity"], out["contract_domain"])
        out["contract_dual"] = jnp.maximum(out["contract_graph"], jnp.maximum(
            out["contract_polar"], out["contract_complementarity"]))
        out["contract"] = jnp.maximum(out["contract_primal"], out["contract_dual"])
        # 1.0 only when the positive branch A_i > 0, B_i > 0 held on every
        # block whose witness was formed.  A contract ratio at most one with
        # this flag at 0 rests on the zero-scale rule, not on the witness.
        out["contract_witness_active"] = jnp.where(
            finite & witness_active, jnp.asarray(1.0, dt), jnp.asarray(0.0, dt))
        return out

    return evaluate
