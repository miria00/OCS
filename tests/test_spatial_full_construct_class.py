"""'spatial_full' is a declared class of the ONE construct call.

Stephen's requirement is one equilibration that handles every declared class
at once.  Here a second-order cone term declares 'spatial_full' beside the other
terms' classes in a single ``equilibrate="construct"`` call; the product-only
constructor builds the carrier on the term's rows from k adjoint and k forward
applications, the compiler applies it through the same path a supplied carrier
takes, and the returned point is scored under the default policy.  Unsupported
requests refuse by name.  No class is ever chosen for the caller.

What the acceptance assertions of this file do and do not claim (2026-10-22).
The 0.3.2 release returned ``solved`` on this instance at tol 1e-4 (3100
iterations) with a dual witness that lies inside the polar cone by about 3e-5
in float64 (relative to ||y[1:]|| about 14, that is about 2e-6).  The current
exact conjugate-domain contract of ``SecondOrderCone`` refuses that margin as
unresolved float32 roundoff (``_conservative_norm_leq``, envelope
(2 k + 8) eps), and the recovery routes certify witnesses pulled further inside,
whose bound sits about 4e-4 relative below the optimum on this family because
the ridge weight mu = 1e-2 multiplies the dual's sensitivity by 1/mu.  The
primal itself is optimal to about 1e-6 and feasible.  So the older ``solved``
assertion was evidence under the older policy and is not evidence under the
current one; it is replaced below by independent facts (the primal against a
float64 reference, the validity of the certified bound, the agreement of the
public same-pair check with an independent one) and by a regression test that
the current policy rejects the preserved 0.3.2 packet.  Nothing in the policy
was changed to make any of this pass.

Update (0.3.4): the dual-correction rounds no longer compound the cone
inside-pull (``witness_contraction="once"`` is the default), so the recovered
witness sits two to three envelopes inside the polar cone and the regression
test below asserts that policy: the supplied 0.3.2 pair is still refused with
recovery off, the recovered witness is a valid bound two to three envelopes
inside, and the public and float64 same-pair verdicts agree.  The final
conservative membership test is unchanged.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.operators.base import LinearOperator
from ocs.prox import cones
from ocs.prox import spatial_construct as sc
from ocs.prox import spatial_soc as ss
from ocs.solver.equilibrate import scalar_steps

F32 = np.float32
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


class EntryAccessBlocked(AssertionError):
    pass


class BlockedDense(LinearOperator):
    well_conditioned = False

    def __init__(self, A):
        super().__init__((int(A.shape[0]), int(A.shape[1])))
        object.__setattr__(self, "_hidden", jnp.asarray(np.asarray(A, F32)))

    def matvec(self, x):
        return jnp.matmul(self._hidden, jnp.asarray(x), precision=jax.lax.Precision.HIGHEST)

    def rmatvec(self, y):
        return jnp.matmul(self._hidden.T, jnp.asarray(y), precision=jax.lax.Precision.HIGHEST)

    def __getattr__(self, name):
        if name in ("_hidden", "shape", "well_conditioned"):
            raise AttributeError(name)
        raise EntryAccessBlocked(name)

    def __array__(self, *a, **k):
        raise EntryAccessBlocked("__array__")

    def __getitem__(self, i):
        raise EntryAccessBlocked(f"__getitem__ {i!r}")


def ball(n=24, seed=3, blocked=True):
    """minimize c^T x + (mu/2)||x||^2  s.t. ||A x|| <= 1: quadratic primal, one SOC term."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n).astype(F32)
    A = rng.normal(size=(n, n)).astype(F32)
    mu = F32(1e-2)
    P = np.zeros((n, n), F32)
    np.fill_diagonal(P, mu)
    cone = np.vstack([np.zeros((1, n), F32), A])
    op = BlockedDense(cone) if blocked else ocs.Dense(cone)
    term = Term(op, cones.SecondOrderCone(),
                lambda _d: jnp.asarray(np.concatenate([[1.0], np.zeros(n)]).astype(F32)))
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.Quadratic(P, c)), term])
    return prob, op, c, A, float(mu)


def objective(c, mu, x):
    x = np.asarray(x, np.float64)
    return float(np.asarray(c, np.float64) @ x) + 0.5 * mu * float(x @ x)


def exact_optimum(c, A, mu):
    """The optimal value, in float64, from the KKT secular equation in the
    eigenbasis of A^T A (bisection on the multiplier).  Independent of OCS."""
    c = np.asarray(c, np.float64); A = np.asarray(A, np.float64)
    s, V = np.linalg.eigh(A.T @ A); s = np.maximum(s, 0.0); g = -(V.T @ c)

    def norm_Ax(lam):
        return float(np.sqrt(np.sum(s * (g / (mu + lam * s)) ** 2)))
    lam = 0.0
    if norm_Ax(0.0) > 1.0:
        lo, hi = 0.0, 1.0
        while norm_Ax(hi) > 1.0:
            hi *= 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if norm_Ax(mid) > 1.0 else (lo, mid)
        lam = 0.5 * (lo + hi)
    x = V @ (g / (mu + lam * s))
    return float(c @ x) + 0.5 * mu * float(x @ x)


def dual_value(c, A, mu, y):
    """d(y) = y_0 - ||A^T y[1:] + c||^2 / (2 mu) on the polar cone -K, and the
    polar margin -y_0 - ||y[1:]|| (nonnegative means in the domain).  Float64."""
    c = np.asarray(c, np.float64); A = np.asarray(A, np.float64); y = np.asarray(y, np.float64)
    w = A.T @ y[1:] + c
    return float(y[0]) - float(w @ w) / (2.0 * mu), float(-y[0] - np.linalg.norm(y[1:]))


def _valid_bound(lower, p_star):
    return lower is not None and np.isfinite(lower) and lower <= p_star + 1e-6 * abs(p_star)


def test_spatial_full_is_constructed_and_applied_through_one_call():
    prob, op, c, A, mu = ball()
    comp = prob.compile(equilibrate="construct", metric_classes={0: "spatial_full"},
                        tol=1e-4, max_blocks=2000, block=100)
    report = comp.meta["metric_construction"]
    decl = report["declared"][0]
    assert decl["declared_class"] == "spatial_full" and decl["metric_sharing"] == "term"
    assert decl["cone"]["class"] == "spatial_full" and decl["cone"]["algebra"] == "soc"
    rec = report["spatial_full"]["terms"][0]
    k = decl["cone"]["k"]
    assert rec["products"]["total_operator_applications"] == 2 * k      # k adjoint + k forward
    assert rec["products"]["forward_applications"] == k
    assert report["spatial_full"]["measured_scaled_norm_with_carrier"] > 0
    eff = comp.meta["effective_metric"]["terms"][0]
    group = eff["metric_sharing_groups"][0]
    assert group["kind"] == "spatial_full_soc" and group["k"] == k and group["shared"] is True
    assert eff["distinct_dual_steps"] == 1
    measured = comp.meta["block_metrics"]["measured_scaled_norm"]
    assert 0 < measured < 1
    s = comp.solve_instance({})
    assert s.accuracy_criterion == "objective_gap"
    assert s.status in ("solved", "inaccurate"), (s.status, s.accuracy_detail)
    # The primal: feasible and optimal against the independent float64 reference.
    x = np.asarray(s.x, np.float64)
    p_star = exact_optimum(c, A, mu)
    assert np.linalg.norm(np.asarray(A, np.float64) @ x) <= 1.0 + 1e-4
    # Within the REQUESTED tolerance of the float64 optimum: the run may stop at
    # the first check whose certified gap meets tol = 1e-4 (it does under the
    # 0.3.4 witness policy, at the same iteration count as 0.3.2), and a point
    # that is feasible to 1e-4 can sit below p* by that order.  The 1e-5 that
    # stood here was the accuracy of the 102,300-iteration stall of the
    # compounded policy, not the request.
    assert abs(objective(c, mu, x) - p_star) <= 1e-4 * abs(p_star), (objective(c, mu, x), p_star)
    # The certificate: whatever the status, a certified bound is a valid lower bound.
    assert _valid_bound(s.objective_lower_bound, p_star), (s.objective_lower_bound, p_star)
    # The public same-pair check and an independent float64 same-pair check agree
    # on the returned certifying pair.
    val = comp.validate(s.x, s.y, {})
    assert val["objective_lower_bound_available"]
    d, margin = dual_value(c, A, mu, s.y)
    assert margin >= 0.0
    assert abs(float(val["objective_lower_bound"]) - d) <= 1e-5 * abs(p_star)
    assert bool(comp.acceptance(val)["target_met"]) == (
        abs(objective(c, mu, x) - d) <= 1e-4 * max(abs(objective(c, mu, x)), abs(d))
        and float(val["relative_primal_infeasibility"]) <= 1e-4)
    if s.status == "solved":
        assert abs(s.objective_gap) <= 1e-4


def test_the_current_policy_rejects_the_032_accepted_packet():
    """Regression, restated for the ``witness_contraction="once"`` default
    (0.3.4): the pair the 0.3.2 release returned as ``solved`` on this instance
    (tol 1e-4, 3100 iterations) is STILL refused as supplied.  Its float32
    dual is inside the polar cone by less than the conservative envelope of
    the exact conjugate-domain test, so recovery off reports no bound (the
    final membership test is unchanged).  With recovery on, the candidate is
    built with exact projections and the second-order cone inside-pull of
    (4 k + 16) eps32 is applied ONCE on the final candidate, so the recovered
    witness lies two to three envelopes ((2 k + 8) eps32 ||y[1:]||) inside the
    polar cone instead of 64, and its bound is a valid lower bound on p*.  The
    public same-pair verdict on the recovered pair and an independent float64
    same-pair verdict agree.  The packet also records why the supplied pair is
    refused: a positive polar margin below the envelope, with a float64
    discrepancy within tol (the refusal is the frozen conservative rule)."""
    prob, op, c, A, mu = ball()
    z = np.load(os.path.join(REFERENCE, "socp_ball_n24_s3_032_endpoint.npz"))
    assert np.array_equal(z["c"], c) and np.array_equal(z["A"], A) and float(z["tol"]) == 1e-4
    comp = prob.compile(equilibrate="construct", metric_classes={0: "spatial_full"},
                        tol=1e-4, max_blocks=2000, block=100)
    assert comp.meta["witness_contraction"] == "once"
    x, y = jnp.asarray(z["x"]), jnp.asarray(z["y"])
    # 1. Recovery OFF still refuses the float32 iterate (outside dom f_i^* under
    #    the unchanged conservative membership test).
    off = comp.validate(x, y, {})
    assert not off["objective_lower_bound_available"]
    assert "outside dom f_i^*" in off["supplied_dual_status"]
    assert not comp.acceptance(off)["target_met"]
    # 2. Recovery ON: the recovered witness lies 2-3 envelopes inside the polar
    #    cone (one contraction of (4 k + 16) eps32 is exactly two envelopes; the
    #    rest is the supplied dual's own margin) with a valid bound (<= p*).
    on = comp.objective_bound(x, y, {}, recover=True, return_dual=True)
    assert on["objective_lower_bound_available"]
    assert on["witness_contraction"] == "once"
    p_star = exact_optimum(c, A, mu)
    yc = np.asarray(on["certifying_dual"], np.float64)
    d_c, margin_c = dual_value(c, A, mu, yc)
    k = yc.size - 1
    envelope_c = (2 * k + 8) * np.finfo(np.float32).eps * np.linalg.norm(yc[1:])
    ratio = margin_c / envelope_c
    assert 2.0 - 0.05 <= ratio <= 3.0, (margin_c, envelope_c, ratio)
    assert _valid_bound(float(on["objective_lower_bound"]), p_star)
    assert abs(float(on["objective_lower_bound"]) - d_c) <= 1e-5 * abs(p_star), (on["objective_lower_bound"], d_c)
    # 3. The public verdict on the recovered pair agrees with the independent
    #    float64 same-pair verdict on the same pair.
    J = objective(c, mu, z["x"])
    feas = float(off["relative_primal_infeasibility"])
    independent_met = (margin_c >= 0.0
                       and abs(J - d_c) <= 1e-4 * max(abs(J), abs(d_c))
                       and feas <= 1e-4)
    assert bool(comp.acceptance(on)["target_met"]) == independent_met, (comp.acceptance(on), J, d_c, feas)
    assert independent_met, (J, d_c, feas)
    # Why the supplied pair is refused: its own dual sits inside the polar cone
    # by a float64 margin below the conservative float32 envelope, while its
    # float64 discrepancy is within the tolerance.
    d, margin = dual_value(c, A, mu, z["y"])
    envelope = (2 * k + 8) * np.finfo(np.float32).eps * np.linalg.norm(np.asarray(z["y"][1:], np.float64))
    assert 0.0 < margin < envelope, (margin, envelope)
    assert abs(J - d) <= 1e-4 * max(abs(J), abs(d))
    assert feas <= 1e-4
    assert _valid_bound(d, p_star)


def test_one_call_agrees_with_the_supplied_path_on_the_returned_objective():
    prob, op, c, A, mu = ball(blocked=False)
    one = prob.compile(equilibrate="construct", metric_classes={0: "spatial_full"},
                       tol=1e-5, max_blocks=4000, block=100).solve_instance({})
    n = c.size
    k = n + 1
    tau_s, sigma_s, nrm = scalar_steps(op.matvec, op.rmatvec, n)
    tau = jnp.full((n,), F32(tau_s))
    metric, _ = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n, tau,
                                          float(np.sqrt(n / (n + k))), layout=("soc",))
    low, high = ss.metric_amplitude(metric)
    sigma = jnp.full((k,), F32(0.9 / (high * max(nrm, 1e-30))))
    supplied = prob.compile(equilibrate=("given", tau, sigma), block_metrics={0: metric},
                            tol=1e-5, max_blocks=4000, block=100).solve_instance({})
    assert one.status in ("solved", "inaccurate") and supplied.status in ("solved", "inaccurate")
    assert one.status == supplied.status, (one.status, one.accuracy_detail, supplied.status, supplied.accuracy_detail)
    p_star = exact_optimum(c, A, mu)
    a, b = objective(c, mu, one.x), objective(c, mu, supplied.x)
    assert abs(a - b) <= 4e-5 * max(1.0, abs(a)), (a, b)
    assert abs(a - p_star) <= 1e-5 * abs(p_star) and abs(b - p_star) <= 1e-5 * abs(p_star)
    assert _valid_bound(one.objective_lower_bound, p_star) and _valid_bound(supplied.objective_lower_bound, p_star)


def test_unsupported_spatial_full_requests_refuse_by_name():
    n = 12
    rng = np.random.default_rng(0)
    x = ocs.Variable(n)
    prob = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                         ocs.group_l2(ocs.Dense(rng.normal(size=(9, n))) @ x, lam=0.3, group_size=3)])
    with pytest.raises(ValueError, match="not a second-order cone block"):
        prob.compile(equilibrate="construct", metric_classes={0: "spatial_full"}, max_blocks=2)
    batch = ocs.minimize([Term(ocs.Identity(n), ocs.SumSquares().scaled(0.5)),
                          Term(ocs.Dense(rng.normal(size=(8, n))), cones.SOCBatch(2, 3))])
    with pytest.raises(ValueError, match="stacks 2 cones"):
        batch.compile(equilibrate="construct", metric_classes={0: "spatial_full"}, max_blocks=2)
    prob2, *_ = ball(blocked=False)
    with pytest.raises(ValueError, match="takes no metric block size"):
        prob2.compile(equilibrate="construct", metric_classes={0: ("spatial_full", "term", 5)}, max_blocks=2)
    with pytest.raises(ValueError, match="one metric for the"):
        prob2.compile(equilibrate="construct", metric_classes={0: ("spatial_full", "per_block")}, max_blocks=2)
