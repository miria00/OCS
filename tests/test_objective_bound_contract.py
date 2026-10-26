"""Adversarial acceptance suite for the strict objective lower bound.

Every case here is a way the bound could be WRONG, or a way a surrogate could be
presented as a bound.  The contract under test:

  * the bound is refused unless the dual is EXACTLY in every conjugate domain
    AND satisfies the coupling condition  -K^T y - c in dom g^*  (for g = 0,
    exact equality K^T y + c = 0);
  * no tolerance is admissible anywhere in that test, in particular none below
    the 1e-6 absolute slack the L1 conjugate indicator used to carry;
  * a block without the exact contract makes the bound unavailable for the
    whole problem;
  * nothing is clamped and nothing is clipped: an unavailable bound is NaN with
    a reason, never a weaker number under a stronger name.

Checking that a bound lies below the returned primal objective is NECESSARY and
NOT SUFFICIENT, so the cases below with a closed-form optimum check the bound
against p* computed analytically in float64.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.model.compile import (
    OBJECTIVE_BOUND_AVAILABLE, OBJECTIVE_BOUND_NO_CONTRACT,
    OBJECTIVE_BOUND_DUAL_NONFINITE, OBJECTIVE_BOUND_COUPLING_VIOLATED,
    OBJECTIVE_BOUND_BLOCK_DOMAIN_VIOLATED, OBJECTIVE_BOUND_ABOVE_PRIMAL,
    OBJECTIVE_BOUND_STATUS,
)

F32 = np.float32


class UndeclaredLogistic(ocs.Logistic):
    """An explicit missing-contract control; Logistic itself is supported."""
    bound_capable = False


def undeclared_logistic_term(op):
    from ocs.ir import Term
    return Term(op, UndeclaredLogistic())


def cert(comp, x, y, data=None):
    return comp.certificate(x, y, data)


def status_of(c):
    return int(c["objective_bound_status"])


def unavailable(c):
    return (not bool(c["objective_lower_bound_available"])
            and math.isnan(float(c["objective_lower_bound"])))


# ===========================================================================
# 1. THE COUPLING CONDITION.  Miria's counterexample, with a known optimum.
#    min 1/2 (x-1)^2  written as  K = 1, l = -1, c = 0, f = 1/2||.||^2, g = 0.
#    f^*(y) = 1/2 y^2, so D(y) = l.y - f^*(y) = -y - 1/2 y^2 on {K^T y = 0}.
#    p* = 0.  At y = -1 every block domain passes (dom f^* = R), the coupling
#    condition fails, and the finite-part surrogate reads +1/2 > 0 = p*.
# ===========================================================================
def _counterexample():
    v = ocs.Variable(1)
    fam = ocs.minimize([ocs.least_squares(ocs.Diagonal(jnp.ones(1)) @ v
                                          - ocs.Parameter("b"))])
    return fam.compile(equilibrate="scalar", max_blocks=0), {"b": jnp.ones(1)}


def test_coupling_counterexample_is_refused_and_the_surrogate_is_not():
    comp, data = _counterexample()
    x = jnp.ones(1)                       # x* = 1, p* = 0
    y = -jnp.ones(1)                      # every block domain passes; K^T y != 0
    c = cert(comp, x, y, data)
    assert bool(c["objective_bound_capable"])
    # the legacy surrogate produces a "dual value" ABOVE the optimum
    surrogate_dual = float(-1.0 * -1.0 - 0.5 * 1.0)     # <l,y> - f^*(y) = 0.5
    assert surrogate_dual > 0.0
    # and the report says so, instead of presenting it as a gap
    assert bool(c["gap_surrogate_above_primal"])
    # the strict bound refuses, naming the coupling condition
    assert unavailable(c)
    assert status_of(c) == OBJECTIVE_BOUND_COUPLING_VIOLATED
    assert float(c["dual_coupling_defect"]) == pytest.approx(1.0, abs=1e-6)
    # and the search cannot rescue it into anything above p* = 0
    b = comp.objective_bound(x, y, data)
    assert b["supplied_dual_status_code"] == OBJECTIVE_BOUND_COUPLING_VIOLATED
    if b["objective_lower_bound_available"]:
        assert b["objective_lower_bound"] <= 0.0 + 1e-6


def test_coupling_counterexample_exact_dual_reaches_the_analytic_optimum():
    comp, data = _counterexample()
    x, y = jnp.ones(1), jnp.zeros(1)      # K^T y + c = 0 exactly
    c = cert(comp, x, y, data)
    assert bool(c["objective_lower_bound_available"])
    assert status_of(c) == OBJECTIVE_BOUND_AVAILABLE
    assert float(c["dual_coupling_defect"]) == 0.0
    assert float(c["objective_lower_bound"]) == pytest.approx(0.0, abs=1e-6)
    assert float(c["objective_lower_bound"]) <= 0.0 + 1e-6      # <= p* = 0


@pytest.mark.parametrize("defect", [1.0, 1e-3, 1e-7, 1e-12])
def test_any_nonzero_coupling_defect_is_refused_for_g_equals_zero(defect):
    comp, data = _counterexample()
    y = jnp.asarray([-defect], jnp.float32)
    c = cert(comp, jnp.ones(1), y, data)
    if float(c["dual_coupling_defect"]) == 0.0:
        pytest.skip("the perturbation is below float32 resolution here")
    assert unavailable(c)
    assert status_of(c) == OBJECTIVE_BOUND_COUPLING_VIOLATED


# ===========================================================================
# 2. THE L1 CONJUGATE DOMAIN, including violations far below the old 1e-6.
# ===========================================================================
def _l1_family(lam):
    u = ocs.Variable(4)
    fam = ocs.minimize([ocs.l1(u, lam),
                        ocs.least_squares(ocs.Diagonal(jnp.ones(4)) @ u
                                          - ocs.Parameter("b"))])
    return fam.compile(equilibrate="scalar", max_blocks=0), {
        "b": jnp.asarray([1., -1., .5, 0.], F32)}


@pytest.mark.parametrize("excess", [5e-7, 1e-7, 1e-8, 0.0, -1e-7])
def test_l1_conjugate_domain_carries_no_absolute_slack(excess):
    lam = 1e-3
    comp, data = _l1_family(lam)
    peak = np.float32(lam) + np.float32(excess)
    y = jnp.asarray([-float(peak), 0., 0., 0.], F32)
    c = cert(comp, jnp.zeros(4), y, data)
    # The comparison the engine makes is in float32, against the float32 lam.
    outside = bool(np.float32(np.abs(np.float32(-peak))) > np.float32(lam))
    if outside:
        assert unavailable(c), f"excess {excess} was accepted"
        assert status_of(c) == OBJECTIVE_BOUND_COUPLING_VIOLATED
    else:
        assert bool(c["objective_lower_bound_available"]), f"excess {excess} was refused"
        assert status_of(c) == OBJECTIVE_BOUND_AVAILABLE


def test_l1_conjugate_value_itself_no_longer_admits_the_1e_6_slack():
    block = ocs.L1(1e-3)
    just_outside = jnp.asarray([1e-3 + 5e-7, 0., 0.], F32)
    assert not bool(block.conj_in_domain(just_outside))
    assert not bool(jnp.isfinite(block.conj_value(just_outside)))
    on_boundary = jnp.asarray([1e-3, 0., 0.], F32)
    assert bool(block.conj_in_domain(on_boundary))
    assert float(block.conj_value(on_boundary)) == 0.0


def test_block_side_l1_violation_is_reported_as_a_block_domain_failure():
    # L1 through a NON-identity operator becomes a DUAL block, so the same
    # violation must be attributed to the block domain, not to the coupling.
    lam = 1e-3
    u = ocs.Variable(4)
    fam = ocs.minimize([ocs.least_squares(ocs.Diagonal(jnp.ones(4)) @ u
                                          - ocs.Parameter("b")),
                        ocs.l1(ocs.Diagonal(jnp.ones(4)) @ u, lam)])
    comp = fam.compile(equilibrate="scalar", max_blocks=0)
    data = {"b": jnp.asarray([1., -1., .5, 0.], F32)}
    y = jnp.concatenate([jnp.zeros(4, F32),
                         jnp.asarray([lam + 5e-7, 0., 0., 0.], F32)])
    c = cert(comp, jnp.zeros(4), y, data)
    assert unavailable(c)
    assert status_of(c) in (OBJECTIVE_BOUND_BLOCK_DOMAIN_VIOLATED,
                            OBJECTIVE_BOUND_COUPLING_VIOLATED)


# ===========================================================================
# 3. UNSUPPORTED CONJUGATE CAPABILITIES.
# ===========================================================================
def test_block_without_the_contract_makes_the_bound_unavailable():
    v = ocs.Variable(3)
    comp = ocs.minimize([undeclared_logistic_term(ocs.Diagonal(jnp.ones(3)))]).compile(
        equilibrate="scalar", max_blocks=0)
    assert not comp.meta["objective_bound_capable"]
    c = cert(comp, jnp.zeros(3), jnp.zeros(3))
    assert unavailable(c)
    assert status_of(c) == OBJECTIVE_BOUND_NO_CONTRACT
    b = comp.objective_bound(jnp.zeros(3), jnp.zeros(3))
    assert not b["objective_lower_bound_available"]
    assert "no exact conjugate-domain contract" in b["unavailable_reason"]
    assert "Logistic" in b["unavailable_reason"]


def test_one_uncapable_block_among_capable_ones_disables_the_problem():
    v = ocs.Variable(3)
    comp = ocs.minimize([ocs.l1(v, 0.1),
                         ocs.least_squares(ocs.Diagonal(jnp.ones(3)) @ v),
                         undeclared_logistic_term(ocs.Diagonal(jnp.ones(3)))]).compile(
        equilibrate="scalar", max_blocks=0)
    assert not comp.meta["objective_bound_capable"]
    assert unavailable(cert(comp, jnp.zeros(3), jnp.zeros(6)))


@pytest.mark.parametrize("transform", ["precomposed", "linquad"])
def test_transformed_quadratics_supply_their_compound_contract(transform):
    """Supported affine/linear/quadratic calculus carries the actual conjugate."""
    base = ocs.SumSquares()
    if transform == "precomposed":
        block = base.precomposed(a=2.0, b=0.5)
    else:
        block = base.with_linear_quadratic(d=jnp.asarray([0.3, 0.3, 0.3]), e=0.5)
    assert getattr(block, "bound_capable", False)
    v = ocs.Variable(3)
    from ocs.ir import Term
    comp = ocs.minimize([Term(ocs.Diagonal(jnp.ones(3)), block)],
                        cost=jnp.asarray([0.1, 0.2, 0.3])).compile(
        equilibrate="scalar", max_blocks=0)
    assert comp.meta["objective_bound_capable"]
    c = cert(comp, jnp.zeros(3), jnp.zeros(3))
    assert unavailable(c)
    # g=0 still imposes exact coupling; having contracts is not a witness.
    assert status_of(c) == OBJECTIVE_BOUND_COUPLING_VIOLATED


def test_a_block_that_declares_the_contract_without_implementing_it_raises():
    """``bound_capable = True`` is a promise.  A block that makes it without
    the methods must fail loudly at compile time, never quietly produce a
    number."""
    class Liar(ocs.SumSquares):
        bound_capable = True

        def conj_in_domain(self, y):
            raise NotImplementedError("deliberate")

    v = ocs.Variable(3)
    from ocs.ir import Term
    with pytest.raises(NotImplementedError):
        comp = ocs.minimize([Term(ocs.Diagonal(jnp.ones(3)), Liar())]).compile(
            equilibrate="scalar", max_blocks=0)
        comp.certificate(jnp.zeros(3), jnp.zeros(3))


# ===========================================================================
# 4. NONFINITE INPUTS.
# ===========================================================================
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_dual_is_refused(bad):
    comp, data = _l1_family(1e-3)
    y = jnp.asarray([bad, 0., 0., 0.], F32)
    c = cert(comp, jnp.zeros(4), y, data)
    assert unavailable(c)
    assert status_of(c) in (OBJECTIVE_BOUND_DUAL_NONFINITE,
                            OBJECTIVE_BOUND_COUPLING_VIOLATED)
    b = comp.objective_bound(jnp.zeros(4), y, data)
    assert not b["objective_lower_bound_available"]
    assert math.isnan(b["objective_lower_bound"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_primal_never_yields_a_finite_gap(bad):
    """A LOWER bound needs no primal point, so a nonfinite x may not invalidate
    a dual-feasible bound; but no finite GAP may be claimed from it."""
    comp, data = _l1_family(1e-3)
    x = jnp.asarray([bad, 0., 0., 0.], F32)
    y = jnp.zeros(4, F32)                      # exactly dual feasible
    c = cert(comp, x, y, data)
    assert bool(c["objective_lower_bound_available"])
    b = comp.objective_bound(x, y, data)
    assert not b["relative_gap_certified"]
    assert math.isnan(b["relative_gap"])
    assert "primal upper bound" in b.get("gap_unavailable_reason", "")


def test_nonfinite_data_is_refused():
    comp, _ = _l1_family(1e-3)
    data = {"b": jnp.asarray([np.nan, 1., 1., 1.], F32)}
    c = cert(comp, jnp.zeros(4), jnp.zeros(4), data)
    assert unavailable(c) or math.isnan(float(c["objective_lower_bound"]))


# ===========================================================================
# 5. OFFSETS, LINEAR TERMS AND OBJECTIVE SCALING IN THE DUAL VALUE.
#    The dual must carry <l, y> and the cost c exactly.  Independent float64.
# ===========================================================================
def test_offsets_and_cost_enter_the_dual_value_exactly():
    rng = np.random.default_rng(3)
    n = 6
    d = np.asarray(0.5 + rng.random(n), np.float64)
    off = np.asarray(rng.normal(size=n), np.float64)
    # Build the cost FROM a chosen dual so that K^T y + c cancels exactly in
    # float32: K = diag(d), so c = -(d * y) evaluated in float32 makes
    # d*y + c exactly zero at that y.
    y32 = np.asarray(0.1 * rng.normal(size=n), np.float32)
    d32 = np.asarray(d, np.float32)
    cst32 = (-(d32 * y32)).astype(np.float32)
    y_exact = jnp.asarray(y32)
    cst = np.asarray(cst32, np.float64)
    v = ocs.Variable(n)
    fam = ocs.minimize(
        [ocs.least_squares(ocs.Diagonal(jnp.asarray(d32)) @ v
                           - ocs.Parameter("b"))],
        cost=jnp.asarray(cst32))
    comp = fam.compile(equilibrate="scalar", max_blocks=0)
    data = {"b": jnp.asarray(off, F32)}
    c = cert(comp, jnp.zeros(n), y_exact, data)
    assert bool(c["objective_lower_bound_available"]), (
        "the cost was built so that K^T y + c cancels exactly in float32")
    # l = -off (the offset enters as K x + l), f^*(y) = 1/2||y||^2.
    y64 = np.asarray(y_exact, np.float64)
    expected = float(-off @ y64 - 0.5 * y64 @ y64)
    assert float(c["objective_lower_bound"]) == pytest.approx(expected, rel=2e-5, abs=2e-6)


# ===========================================================================
# 6. ANALYTICALLY KNOWN OPTIMA.  bound <= p*, not merely bound <= P(x).
# ===========================================================================
def _analytic_cases():
    rng = np.random.default_rng(11)
    n = 8
    d = np.asarray(0.5 + rng.random(n), np.float64)
    b = np.asarray(rng.normal(size=n), np.float64)
    lam = 0.15
    lo, hi = -0.4, 0.6
    v = ocs.Variable(n)
    D = ocs.Diagonal(jnp.asarray(d, F32))
    ls = lambda: ocs.least_squares(D @ v - ocs.Parameter("b"))
    data = {"b": jnp.asarray(b, F32)}

    def p_of(x):
        return float(0.5 * np.sum((d * x - b) ** 2))

    # unconstrained separable least squares: x_i = b_i / d_i, p* = 0
    yield ("unconstrained", ocs.minimize([ls()]), data, 0.0)
    # nonneg-constrained: x_i = max(0, b_i/d_i)
    xs = np.maximum(b / d, 0.0)
    yield ("nonneg", ocs.minimize([ocs.nonneg(v), ls()]), data, p_of(xs))
    # box-constrained: x_i = clip(b_i/d_i, lo, hi)
    xb = np.clip(b / d, lo, hi)
    yield ("box", ocs.minimize([ocs.box(v, lo, hi), ls()]), data, p_of(xb))
    # separable lasso with D = I: soft threshold
    xl = np.sign(b) * np.maximum(np.abs(b) - lam, 0.0)
    fam = ocs.minimize([ocs.l1(v, lam),
                        ocs.least_squares(ocs.Diagonal(jnp.ones(n)) @ v
                                          - ocs.Parameter("b"))])
    pl = float(0.5 * np.sum((xl - b) ** 2) + lam * np.sum(np.abs(xl)))
    yield ("lasso", fam, data, pl)


@pytest.mark.parametrize("name", ["unconstrained", "nonneg", "box", "lasso"])
def test_bound_never_exceeds_the_analytic_optimum(name):
    for case, fam, data, p_star in _analytic_cases():
        if case != name:
            continue
        comp = fam.compile(tol=1e-6)
        sol = comp.solve_instance(data)
        b = comp.objective_bound(sol.x, sol.y, data)
        if not b["objective_lower_bound_available"]:
            # An honest refusal is allowed and is the recorded outcome for a
            # family with no routed primal term: dom g^* = {0} and the returned
            # dual never satisfies K^T y + c = 0 exactly.  See RECEIPT.md.
            pytest.skip(f"{case}: bound UNAVAILABLE, reason: "
                        f"{b.get('unavailable_reason', '')}")
        bound = b["objective_lower_bound"]
        slack = 1e-5 * max(abs(p_star), 1.0)
        assert bound <= p_star + slack, (
            f"{case}: bound {bound!r} exceeds the ANALYTIC optimum {p_star!r}")


# ===========================================================================
# 7. THE FORBIDDEN BEHAVIOURS.
# ===========================================================================
def test_nothing_is_clamped_or_clipped():
    """A refused bound is NaN with a reason, never the primal value; and the
    legacy surrogate is not clipped to the primal either, so the failure stays
    visible instead of reading as a small gap."""
    comp, data = _counterexample()
    x, y = jnp.ones(1), -jnp.ones(1)       # p* = 0, surrogate dual = +1/2
    c = cert(comp, x, y, data)
    # the strict bound refuses, with NaN, not with the primal value
    assert math.isnan(float(c["objective_lower_bound"]))
    assert not bool(c["objective_lower_bound_available"])
    # the surrogate is NOT clipped to the primal: if it were, gap would read 0
    # and gap_surrogate_above_primal would read False.
    assert bool(c["gap_surrogate_above_primal"])
    expected = abs(0.0 - 0.5) / (1.0 + abs(0.0) + abs(0.5))
    assert float(c["gap"]) == pytest.approx(expected, rel=1e-4)
    assert float(c["gap"]) > 0.0
    # and whatever the search returns is a real bound on p* = 0, not the primal
    b = comp.objective_bound(x, y, data)
    if b["objective_lower_bound_available"]:
        assert b["objective_lower_bound"] <= 0.0 + 1e-6
        assert b["source"] in ("violation_corrected", "residual_scaled")


def test_report_always_carries_the_bound_fields_and_the_legacy_label():
    v = ocs.Variable(3)
    uncapable = ocs.minimize([ocs.logistic(ocs.Diagonal(jnp.ones(3)) @ v)]).compile(
        equilibrate="scalar", max_blocks=0)
    capable, data = _l1_family(1e-3)
    for comp, x, y, d in ((uncapable, jnp.zeros(3), jnp.zeros(3), None),
                          (capable, jnp.zeros(4), jnp.zeros(4), data)):
        c = cert(comp, x, y, d)
        for key in ("objective_lower_bound", "objective_lower_bound_available",
                    "objective_bound_status", "objective_bound_capable",
                    "objective_primal_value", "dual_coupling_defect",
                    "gap", "gap_legacy_fenchel_surrogate",
                    "gap_surrogate_above_primal"):
            assert key in c, key
        assert float(c["gap_legacy_fenchel_surrogate"]) == float(c["gap"]) or (
            math.isnan(float(c["gap"])) and math.isnan(float(c["gap_legacy_fenchel_surrogate"])))
        assert int(c["objective_bound_status"]) in OBJECTIVE_BOUND_STATUS


def test_report_text_labels_the_surrogate_and_states_the_epistemic_limit():
    comp, data = _l1_family(1e-3)
    text = comp.report_text(jnp.zeros(4), jnp.zeros(4), data)
    assert "legacy Fenchel surrogate" in text
    assert "NOT A BOUND" in text
    assert "NOT a rigorous real-arithmetic enclosure" in text
    assert "objective lower bound" in text


def test_scaled_blocks_inherit_the_exact_contract_with_the_scaling_algebra():
    """h = c f with c > 0 has h^*(y) = c f^*(y / c), so dom h^* = c dom f^* is
    decided exactly by the base's own test on y / c.  The wrapper therefore
    inherits ``bound_capable`` and a family built with ``.scaled()`` is
    bound-capable; before this, every such family (the ridge primal of the
    deblurring families among them) reported ``unverified``."""
    block = ocs.SumSquares().scaled(2.0)
    assert block.bound_capable is True
    v = ocs.Variable(3)
    from ocs.ir import Term
    comp = ocs.minimize([Term(ocs.Diagonal(jnp.ones(3)), block)],
                        cost=jnp.asarray([0.1, 0.2, 0.3])).compile(
        equilibrate="scalar", max_blocks=0)
    assert comp.meta["objective_bound_capable"]
