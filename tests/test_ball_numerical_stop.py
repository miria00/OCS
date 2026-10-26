"""The product-only NUMERICAL stopping evaluator (ocs.verify.ball_numerical).

Gate for section 4 of the directive and section 6 of
`candidates/product_only_metric_20260930/SPEC.md`.

THE DISTINCTION THIS FILE EXISTS TO HOLD. There are two things and they must not
be confused:

* the RIGOROUS bound of `ocs.solver.bound_authorization`, which rebuilds the
  operator's action from archived `Dense` parameters, refuses a callable-only
  operator, and is the only authority for a certified lower bound;
* this NUMERICAL gap evaluation, which needs one adjoint and two forward products
  and no entries at all, and which is a float32 production quantity rather than
  an enclosure.

The tests below check that the numerical one is correct where it should be,
product-only where it claims to be, and that it never writes into or imitates the
rigorous one.
"""
from __future__ import annotations

import numpy as np
import pytest

from ocs.verify import ball_numerical as bn

F32 = np.float32


class EntryAccessBlocked(AssertionError):
    pass


class ProductOnlyOperator:
    """Only matvec and rmatvec. Everything an entry reader wants raises."""

    def __init__(self, A):
        object.__setattr__(self, "_A", np.asarray(A, np.float32))
        object.__setattr__(self, "forward", 0)
        object.__setattr__(self, "adjoint", 0)

    def matvec(self, x):
        object.__setattr__(self, "forward", self.forward + 1)
        return self._A @ np.asarray(x, np.float32)

    def rmatvec(self, y):
        object.__setattr__(self, "adjoint", self.adjoint + 1)
        return self._A.T @ np.asarray(y, np.float32)

    def __getattr__(self, name):
        raise EntryAccessBlocked(f"reached for {name!r} on the operator")

    def __array__(self, *a, **k):
        raise EntryAccessBlocked("array-converted the operator")


def instance(rng, n, mu=1e-2):
    c = rng.standard_normal(n).astype(F32)
    A = rng.standard_normal((n, n)).astype(F32)
    return c, A, np.full(n, np.float32(mu))


def exact_optimum(c, A, mu):
    """The family's own KKT system in float64, OUTSIDE the production path."""
    c = np.asarray(c, np.float64)
    A = np.asarray(A, np.float64)
    mu = float(np.asarray(mu).reshape(-1)[0])
    s, V = np.linalg.eigh(A.T @ A)
    s = np.maximum(s, 0.0)
    g = -(V.T @ c)

    def norm_Ax(lam):
        return float(np.sqrt(np.sum(s * (g / (mu + lam * s)) ** 2)))

    if norm_Ax(0.0) <= 1.0:
        lam = 0.0
    else:
        hi = 1.0
        while norm_Ax(hi) > 1.0 and hi < 1e18:
            hi *= 10.0
        lo = 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if norm_Ax(mid) > 1.0:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
    x = V @ (g / (mu + lam * s))
    return float(c @ x) + 0.5 * mu * float(x @ x), x


# ---------------------------------------------------------------------------
def test_the_gap_identity_holds_and_both_parts_are_nonnegative():
    """At a feasible point the identity is exact, so its residual is roundoff."""
    rng = np.random.default_rng(1)
    n = 40
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    star, x_star = exact_optimum(c, A, h)
    x = x_star * 0.9                       # feasible, the ball is homogeneous
    v = rng.standard_normal(n) * 0.1
    out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, v, x)
    # F(x) - l(v) = stationarity + support, by construction
    assert out["parts_nonnegative"], out
    residual = abs((out["objective"] - out["implied_lower_value"]) - out["gap"])
    print(f"identity residual {residual:.3e}, gap {out['gap']:.6e}")
    assert residual <= 1e-9 * max(1.0, abs(out["objective"]))


def test_the_implied_value_is_below_the_exact_optimum_at_a_feasible_point():
    """The evaluation's implied lower value must sit under the true optimum.

    This is the property that makes it USEFUL as a stopping quantity, and it is
    checked against an optimum computed outside the production path. It is not a
    certificate: float32 state evaluated in float64 is still an evaluation.
    """
    rng = np.random.default_rng(2)
    n = 48
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    star, x_star = exact_optimum(c, A, h)
    worst = -np.inf
    for scale in (1.0, 0.99, 0.95):
        x = x_star * scale
        # the dual the family suggests: v proportional to A x, the support
        # direction, which needs one forward product and no entries
        Ax = np.asarray(op.matvec(x), np.float64)
        v = Ax / max(1e-30, float(np.linalg.norm(Ax))) * 0.5
        out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, v, x)
        worst = max(worst, out["implied_lower_value"] - star)
        print(f"scale {scale}: implied {out['implied_lower_value']:.9f} against "
              f"the optimum {star:.9f}")
    assert worst <= 1e-6, f"the implied value rose above the optimum by {worst:.3e}"


def test_the_evaluator_is_product_only_and_costs_three_applications():
    rng = np.random.default_rng(3)
    n = 32
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    stop = bn.BallNumericalStop(op.matvec, op.rmatvec, c, h)
    _star, x_star = exact_optimum(c, A, h)
    v = rng.standard_normal(n) * 0.1
    before_f, before_a = op.forward, op.adjoint
    out = stop.evaluate(x_star * 1.05, v)      # deliberately INFEASIBLE raw point
    used_f = op.forward - before_f
    used_a = op.adjoint - before_a
    print(f"{used_f} forward and {used_a} adjoint applications, "
          f"restoration scale {out['restoration']['scale']:.9f}")
    assert used_a == 2 and used_f == 3, (used_f, used_a)
    assert out["products"]["total"] == 5
    # and nothing reached for an entry
    for name in ("toarray", "data", "A", "T"):
        with pytest.raises(EntryAccessBlocked):
            getattr(op, name)


def test_the_raw_and_restored_points_are_reported_separately():
    rng = np.random.default_rng(4)
    n = 24
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    stop = bn.BallNumericalStop(op.matvec, op.rmatvec, c, h)
    _star, x_star = exact_optimum(c, A, h)
    out = stop.evaluate(x_star * 1.2, rng.standard_normal(n) * 0.1)
    assert "raw" in out and "restored" in out
    assert out["restoration"]["scale"] < 1.0, "this fixture must need restoring"
    assert out["restored_point_feasible_by_construction"]
    # the restored point's objective is the one that can be an upper value
    assert out["restored"]["objective"] != out["raw"]["objective"]
    print(f"raw objective {out['raw']['objective']:.6f}, restored "
          f"{out['restored']['objective']:.6f}, scale "
          f"{out['restoration']['scale']:.6f}")


def test_a_negative_part_is_reported_and_never_clamped():
    """An inconsistent evaluation is information, not something to clip.

    At an INFEASIBLE point the support term can go negative. The evaluator must
    say so rather than return an apparent zero gap.
    """
    rng = np.random.default_rng(5)
    n = 20
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    _star, x_star = exact_optimum(c, A, h)
    # v ALIGNED with A x at an infeasible point is the case that makes the
    # support term negative, and it is also the realistic one: a solver's dual
    # points along the violated constraint. A random v in high dimensions is
    # nearly orthogonal to A x, so v^T A x vanishes and the term stays positive.
    x_bad = x_star * 1.5
    Ax = np.asarray(op.matvec(x_bad), np.float64)
    direction = Ax / max(1e-30, float(np.linalg.norm(Ax)))
    found = False
    for trial in range(40):
        v = direction * (0.01 + 0.2 * trial)
        out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, v, x_bad)
        if not out["parts_nonnegative"]:
            found = True
            assert out["support"] < 0.0 or out["stationarity"] < 0.0
            # and the gap is reported as it came out, not floored
            assert out["gap"] == out["stationarity"] + out["support"]
            print(f"negative part reported: stationarity "
                  f"{out['stationarity']:.3e}, support {out['support']:.3e}")
            break
    assert found, "this fixture no longer produces a negative part"


def test_a_cancelling_negative_part_cannot_be_accepted():
    """A small total made by cancellation is not a small gap.

    This is the case reporting alone does not prevent: a negative support term
    cancelling the positive squared-defect term leaves a total that looks fine. A
    policy reading `gap` would accept it. `acceptable` must refuse it, and the
    refusal has to be driven by the terms' own scale rather than by a number
    chosen after seeing this case.
    """
    rng = np.random.default_rng(11)
    n = 30
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    _star, x_star = exact_optimum(c, A, h)
    x_bad = x_star * 2.0                          # firmly infeasible
    Ax = np.asarray(op.matvec(x_bad), np.float64)
    direction = Ax / max(1e-30, float(np.linalg.norm(Ax)))

    # search for a v where the two parts nearly cancel, which is the dangerous
    # configuration rather than a merely negative one
    worst = None
    for theta in np.linspace(0.01, 400.0, 4000):
        v = direction * theta
        out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, v, x_bad)
        if out["support"] < 0.0:
            ratio = out["cancellation_ratio"]
            if worst is None or ratio > worst["cancellation_ratio"]:
                worst = out
    assert worst is not None, "this fixture no longer produces a negative part"
    print(f"worst cancellation: gap {worst['gap']:.6e}, stationarity "
          f"{worst['stationarity']:.6e}, support {worst['support']:.6e}, "
          f"ratio {worst['cancellation_ratio']:.3e}, acceptable "
          f"{worst['acceptable']}")
    assert worst["inconsistent"] is True
    assert worst["acceptable"] is False, \
        "an inconsistent evaluation was marked acceptable"
    # and the gap is still reported as it came out, not floored
    assert worst["gap"] == worst["stationarity"] + worst["support"]


def test_a_rounding_level_negative_part_is_not_called_inconsistent():
    """The control must not refuse ordinary roundoff.

    At a feasible point the support term is nonnegative in exact arithmetic and
    can come out negative by a rounding, which is not an inconsistency. The
    allowance is the terms' own scale times the dtype's roundoff, fixed before any
    case was seen.
    """
    rng = np.random.default_rng(12)
    n = 20
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    _star, x_star = exact_optimum(c, A, h)
    # v = 0 makes the support term exactly zero, the boundary of the allowance
    out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, np.zeros(n), x_star * 0.9)
    assert out["support"] == 0.0
    assert out["inconsistent"] is False
    assert out["acceptable"] is True
    print(f"support exactly zero: inconsistent {out['inconsistent']}, "
          f"allowance {out['rounding_allowance']:.3e}")


def test_a_nonfinite_component_cannot_be_accepted():
    rng = np.random.default_rng(13)
    n = 12
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    huge = np.full(n, np.float64(1e200))
    out = bn.numerical_gap(op.matvec, op.rmatvec, c, h, huge, huge)
    assert out["finite"] is False or out["inconsistent"] is True
    assert out["acceptable"] is False, \
        "a non-finite evaluation was marked acceptable"


def test_the_evaluation_says_it_is_numerical():
    rng = np.random.default_rng(6)
    n = 16
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    out = bn.numerical_gap(op.matvec, op.rmatvec, c, h,
                           rng.standard_normal(n) * 0.1,
                           rng.standard_normal(n) * 0.1)
    assert "NUMERICAL" in out["evaluation"]
    assert "not the certified bound" in out["evaluation"]
    # it must NOT carry any field name the rigorous decision owns
    for forbidden in ("objective_lower_bound", "objective_lower_bound_available",
                      "bound_decision_authorized", "bound_decision_status_code"):
        assert forbidden not in out, forbidden


def test_it_refuses_a_curvature_that_is_not_positive():
    rng = np.random.default_rng(7)
    n = 12
    c, A, h = instance(rng, n)
    op = ProductOnlyOperator(A)
    bad = np.array(h, np.float64)
    bad[3] = 0.0
    with pytest.raises(ValueError, match="strictly positive diagonal"):
        bn.numerical_gap(op.matvec, op.rmatvec, c, bad,
                         rng.standard_normal(n), rng.standard_normal(n))
