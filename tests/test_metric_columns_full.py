"""``metric_columns=("full", k)``: one shared k x k column shape, tested where it is applied.

The declaration is one symmetric positive definite ``S`` shared by the ``n / k``
contiguous column blocks of the primal space, so the primal step operator is
``T = c (I kron S)``. It is accepted only on a family with NO distinguished
primal proximal block, where the primal update is the gradient step
``x - T (K^T y + c)`` and ``S`` rides on the adjoint handed to the loop and
folds into the cost.

Every test below inspects what was CONSTRUCTED and APPLIED: the callables and
step arrays handed to ``make_solver`` (intercepted at compile time), the
compiler's own records, and explicit float64 reference objects built in the
test (a 12 by 12 ``T``, a 30 by 12 ``Sigma^1/2 K T^1/2``) that are never
engine objects. Refusals are asserted by their exact message. The fixture is
verified fit (bounded, solvable, and needing a non-diagonal shape) before any
contract assertion.
"""
import math
import re

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.model import compile as cm
from ocs.prox import block_metric as bm
from ocs.prox import cones
from ocs.solver import metric_construct as mc


# --------------------------------------------------------------- the fixture
def mixed_family(n=12, k=3, m1=12, m2=6, seed=0, mix_cond=30.0, cost=True):
    """No Identity term, so no distinguished primal block.

    ``M`` is a k x k SPD matrix with condition number ``mix_cond`` and a random
    rotation. ``K0 = G0 (I kron M)`` feeds a SumSquares term (coercive, so the
    objective is bounded below and a solution exists: the design's fixture
    without it is unbounded at seed 0, refutation D3), ``K1 = G1 (I kron M)``
    a GroupL2 term and ``K2 = G2 (I kron M^-1)`` a NonNeg term, with a nonzero
    random linear cost ``c``.
    """
    rng = np.random.default_rng(seed)
    count = n // k
    Qm, _ = np.linalg.qr(rng.normal(size=(k, k)))
    M = Qm @ np.diag(np.geomspace(1.0, mix_cond, k)) @ Qm.T
    M = 0.5 * (M + M.T)
    Mi = np.linalg.inv(M)
    G0 = rng.normal(size=(n, n)) / np.sqrt(n)
    G1 = rng.normal(size=(m1, n)) / np.sqrt(n)
    G2 = rng.normal(size=(m2, n)) / np.sqrt(n)
    K0 = G0 @ np.kron(np.eye(count), M)
    K1 = G1 @ np.kron(np.eye(count), M)
    K2 = G2 @ np.kron(np.eye(count), Mi)
    x = ocs.Variable(n)
    p0, p1, p2 = ocs.Parameter("l0"), ocs.Parameter("l1"), ocs.Parameter("l2")
    terms = [ocs.sum_squares(ocs.Dense(K0) @ x + p0),
             ocs.group_l2(ocs.Dense(K1) @ x + p1, lam=0.3, group_size=k),
             ocs.nonneg(ocs.Dense(K2) @ x + p2)]
    prob = ocs.minimize(terms, cost=ocs.Parameter("c") if cost else None)
    data = {"l0": jnp.asarray(rng.normal(size=n), jnp.float32),
            "l1": jnp.asarray(rng.normal(size=m1), jnp.float32),
            "l2": jnp.asarray(rng.normal(size=m2), jnp.float32)}
    if cost:
        data["c"] = jnp.asarray(rng.normal(size=n), jnp.float32)
    return prob, data, np.vstack([K0, K1, K2]), n, k


def distinguished_family(block, n, seed=1):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(2 * n, n))
    x = ocs.Variable(n)
    prob = ocs.minimize([Term(ocs.Identity(n), block),
                         ocs.nonneg(ocs.Dense(K) @ x + ocs.Parameter("l"))])
    return prob, {"l": jnp.asarray(rng.normal(size=2 * n), jnp.float32)}


_CAPTURED = []


def _capturing_make_solver(matvec, rmatvec, tau, sigma, prox_primal, prox_dual, *args, **kwargs):
    """Intercept the callables and the steps the compiled loop is actually handed."""
    _CAPTURED.append({"matvec": matvec, "rmatvec": rmatvec,
                      "tau": np.asarray(jax.device_get(tau)),
                      "sigma": np.asarray(jax.device_get(sigma)),
                      "prox_primal": prox_primal, "prox_dual": prox_dual})
    return _capturing_make_solver.wrapped(matvec, rmatvec, tau, sigma, prox_primal, prox_dual,
                                          *args, **kwargs)


@pytest.fixture(autouse=True)
def capture_applied(monkeypatch):
    _CAPTURED.clear()
    _capturing_make_solver.wrapped = cm.make_solver
    monkeypatch.setattr(cm, "make_solver", _capturing_make_solver)
    yield


def build(prob, **kwargs):
    """A construct compile with the capture attached at compile time."""
    kwargs.setdefault("max_blocks", 4)
    kwargs.setdefault("tol", 1e-3)
    kwargs.setdefault("metric_updates", 12)
    _CAPTURED.clear()                          # this compile's captures only
    comp = prob.compile(equilibrate="construct", **kwargs)
    comp.applied = dict(_CAPTURED[0])          # the dense-layout engine's arguments
    comp.applied_last = dict(_CAPTURED[-1])    # the compact engine's, when one was built
    return comp


def shape_from_meta(comp):
    """(S, Q, w) in float64 from the public record of the constructed column shape."""
    rec = comp.meta["metric_construction"]["columns"]["shape"]
    w = np.asarray(rec["w"], np.float64)
    Q = np.eye(w.size) if rec["Q"] is None else np.asarray(rec["Q"], np.float64)
    return Q @ np.diag(w) @ Q.T, Q, w


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# metric-path test: this family has no exact dual witness under the default acceptance (g = 0 coupling with a cost, or a Zero block), so the explicit residual policy is requested here. The default objective-gap acceptance is tested in test_validate.py and test_stopping_contract.py.
SOLVE = dict(accuracy="residual", tol=1e-5, max_blocks=400, metric_updates=20)


# ------------------------------------------------- the fixture is fit for use
def test_the_fixture_is_bounded_and_the_undeclared_compile_solves():
    """Refutation D3: the fixture must be solvable before any declared-versus-
    undeclared comparison. The status is read; a budget-exhausted return
    fails here with that reason and never counts as a solve."""
    prob, data, _K, _n, _k = mixed_family()
    comp = prob.compile(equilibrate="sgd", tol=1e-5, max_blocks=400, accuracy="residual")
    sol = comp.solve_instance(data)
    assert sol.status == "solved", (sol.status, sol.iters, sol.residual)
    assert sol.termination == "criterion", sol.termination


def test_the_fixture_needs_a_non_diagonal_shape():
    """T1. A fixture whose constructed column shape is diagonal would make every
    later test vacuous, so this runs first: the constructed S mixes coordinates."""
    prob, _data, _K, _n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    S, _Q, _w = shape_from_meta(comp)
    d = np.sqrt(np.diag(S))
    correlation = np.abs(S / np.outer(d, d)) - np.eye(S.shape[0])
    # Scale invariant: a diagonal S has zero off-diagonal correlation whatever
    # its anisotropy. (The design's max|offdiag| / max diag is 0.05 on this
    # fixture because the diagonal spans a factor 30, while the largest
    # off-diagonal correlation is 0.30; both numbers are in the receipt.)
    assert np.max(correlation) > 0.1, S
    assert np.linalg.norm(S - np.diag(np.diag(S))) / np.linalg.norm(S) > 0.05, S
    assert comp.meta["metric_construction"]["columns"]["shape"]["kind"] == "full"
    assert comp.meta["metric_construction"]["regime"] == "geodesic"


# ------------------------------------------------------ the applied operator
def test_the_applied_adjoint_and_cost_fold_equal_the_explicit_T():
    """T2. From the capture: the adjoint handed to the loop is (I kron S) K^T
    and the primal prox is x - T c with the nonzero cost. The second assertion
    fails on a build that folds the shape into the adjoint only."""
    prob, data, K, n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    S, _Q, _w = shape_from_meta(comp)
    Sn = np.kron(np.eye(n // k), S)
    tau = float(comp.applied["tau"])
    T = tau * Sn
    l, c = comp._resolve_dense(data)
    c64 = np.asarray(jax.device_get(c), np.float64)
    assert np.linalg.norm(c64) > 0
    rng = np.random.default_rng(5)
    rmv, prox = comp.applied["rmatvec"], comp.applied["prox_primal"]
    for _ in range(8):
        y = rng.normal(size=K.shape[0]).astype(np.float32)
        x = rng.normal(size=n).astype(np.float32)
        applied = np.asarray(jax.device_get(rmv(jnp.asarray(y))))
        assert rel(applied, Sn @ (K.T @ y.astype(np.float64))) < 1e-5
        folded = np.asarray(jax.device_get(prox(jnp.asarray(x), jnp.asarray(tau, jnp.float32),
                                                 (l, c))))
        assert rel(folded, x.astype(np.float64) - T @ c64) < 1e-5
        step = prox(jnp.asarray(x) - tau * rmv(jnp.asarray(y)), jnp.asarray(tau, jnp.float32),
                    (l, c))
        expected = x.astype(np.float64) - T @ (K.T @ y.astype(np.float64) + c64)
        assert rel(np.asarray(jax.device_get(step)), expected) < 1e-5


def test_the_step_condition_is_measured_on_the_applied_operator():
    """T3. The compiler's re-measure against the exact float64 largest singular
    value of the explicit 30 by 12 test object Sigma^1/2 K T^1/2."""
    prob, _data, K, n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    S, _Q, _w = shape_from_meta(comp)
    w, V = np.linalg.eigh(S)
    root_S = V @ np.diag(np.sqrt(w)) @ V.T
    tau = float(comp.applied["tau"])
    sigma = np.broadcast_to(comp.applied["sigma"].astype(np.float64), (K.shape[0],))
    B = np.diag(np.sqrt(sigma)) @ K @ (math.sqrt(tau) * np.kron(np.eye(n // k), root_S))
    smax = float(np.linalg.svd(B, compute_uv=False)[0])
    measured = comp.meta["block_metrics"]["measured_scaled_norm"]
    assert measured < 1.0
    assert measured <= smax * (1 + 1e-4), (measured, smax)
    assert measured >= 0.8 * smax, (measured, smax)


def test_the_re_measure_runs_with_no_row_shape_and_agrees_with_the_constructor():
    """T4. No row term carries a shape here, so before this change the compiler
    never re-measured. It must, and its value must sit at the constructor's
    post-factor target 0.9 up to the design tolerance of five percent."""
    prob, _data, _K, _n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    bmeta = comp.meta["block_metrics"]
    assert bmeta["terms"] == []
    assert bmeta["present"] is False
    assert bmeta["columns"]["present"] is True
    assert bmeta["columns"]["k"] == k and bmeta["columns"]["count"] == _n // k
    norm = comp.meta["metric_construction"]["normalization"]
    assert abs(norm["factor"] * norm["measured_before"] / 0.9 - 1) < 1e-6
    assert abs(bmeta["measured_scaled_norm"] / 0.9 - 1) <= 0.05, bmeta["measured_scaled_norm"]


def test_T_half_is_the_square_root_of_T():
    """T5. The half power the re-measure uses, rebuilt the way the compiler
    builds it, against a float64 sqrtm of the recorded S."""
    prob, _data, _K, n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    S, Q, w = shape_from_meta(comp)
    metric = bm.BlockMetric(w, Q)
    apply = bm.make_shape_apply({0: (k, ((0, n // k, metric),))}, ((0, n),), n)
    root = Q @ np.diag(np.sqrt(w)) @ Q.T
    rng = np.random.default_rng(9)
    for _ in range(4):
        v = rng.normal(size=n).astype(np.float32)
        half = apply(jnp.asarray(v), power=0.5)
        assert rel(jax.device_get(apply(half, power=0.5)), jax.device_get(apply(jnp.asarray(v), power=1))) < 1e-5
        assert rel(jax.device_get(half), np.kron(np.eye(n // k), root) @ v.astype(np.float64)) < 1e-5


# ---------------------------------------------------------- condition domain
def test_the_condition_domain_is_enforced_then_restricted_and_re_measured():
    """T6. A fixture whose constructed cond(S) exceeds MAX_FULL_COND = 100 is
    rejected by default and restricted on request, with the normalization
    re-measured on the restricted shape. Skipped with the number, never passed
    vacuously, if this fixture's constructed cond stays inside the domain."""
    prob, _data, _K, _n, k = mixed_family(mix_cond=2000.0)
    try:
        comp = build(prob, metric_columns=("full", k), metric_box=2.0)
    except ValueError as exc:
        message = str(exc)
        assert "outside the validated numerical domain" in message
        assert f"metric_columns=('full', {k})" in message
        cond = float(re.search(r"condition number ([0-9.e+]+)", message).group(1))
        assert cond > bm.MAX_FULL_COND
    else:
        cond = comp.meta["metric_construction"]["columns"]["shape"]["cond"]
        pytest.skip(f"the constructed column shape has cond {cond:.4g} <= {bm.MAX_FULL_COND}, "
                    "so this fixture does not reach the domain boundary; the domain loop is "
                    "exercised directly by test_the_column_domain_loop_directly")
    comp = build(prob, metric_columns=("full", k), metric_box=2.0, metric_domain="restrict")
    restriction = comp.meta["metric_construction"]["restriction"]
    applied = restriction["applied"][0]
    assert applied["side"] == "columns" and applied["kind"] == "shared_full"
    assert applied["restricted"] is True
    assert applied["cond_after"] <= bm.MAX_FULL_COND
    assert restriction["measured_after_restriction"] > restriction["measured_before_restriction"]
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1.0
    assert comp.meta["metric_construction"]["columns"]["shape"]["cond"] <= bm.MAX_FULL_COND


def test_the_column_domain_loop_directly(monkeypatch):
    """T6b (design open question 4). The domain loop of the constructor with a
    decoded column shape of condition number 1000 substituted, so the reject
    and restrict policies are exercised whatever the fixture constructs."""
    prob, _data, _K, n, k = mixed_family()
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    bad = Q @ np.diag(np.geomspace(1.0, 1000.0, k)) @ Q.T
    bad = 0.5 * (bad + bad.T)
    monkeypatch.setattr(mc, "_decode_columns", lambda groups, state: [("shared_full", bad)])
    with pytest.raises(ValueError) as info:
        build(prob, metric_columns=("full", k))
    assert str(info.value).startswith(
        f"metric_columns=('full', {k}): the constructed full (rotated) column shape has "
        "condition number 1000, outside the validated numerical domain (at most 100).")
    assert "Nothing is clipped for you" in str(info.value)
    comp = build(prob, metric_columns=("full", k), metric_domain="restrict")
    applied = comp.meta["metric_construction"]["restriction"]["applied"][0]
    assert applied["side"] == "columns" and applied["cond_before"] == pytest.approx(1000.0)
    assert applied["cond_after"] <= bm.MAX_FULL_COND
    S, _Q, w = shape_from_meta(comp)
    assert np.allclose(np.sort(w), np.sort(applied["eigenvalues_after"]))
    # The restricted shape is the one the loop applies.
    Sn = np.kron(np.eye(n // k), S)
    y = rng.normal(size=_K.shape[0]).astype(np.float32)
    applied_adjoint = jax.device_get(comp.applied["rmatvec"](jnp.asarray(y)))
    assert rel(applied_adjoint, Sn @ (_K.T @ y.astype(np.float64))) < 1e-5


# -------------------------------------------- every distinguished block refuses
def _core(k, name, cls, count, k0):
    return (f"metric_columns=('full', {k}) is REFUSED on this family: it has a distinguished "
            f"primal proximal block ({name}, scaling class '{cls}', {count} mathematical "
            f"block(s) of {k0} coordinates), and the compiled loop evaluates that block as "
            "g.prox(x, tau) with a scalar or per-coordinate step. No primal proximal block is "
            "routed a full (non-diagonal) metric by this release, so the declaration can be "
            "neither built nor applied. A full column shape is accepted only when no "
            "distinguished primal block exists, where the primal update is the gradient step "
            "x - T (K^T y + c). Declare 'scalar', 'diagonal' or (start, end) column spans for "
            "this family.")


_P6 = np.diag([2.0, 1.5, 1.0, 1.2, 0.8, 1.1])


@pytest.mark.parametrize("block, n, k, kind", [
    (ocs.SumSquares().scaled(0.5), 12, 3, "matrix"),
    (ocs.Quadratic(_P6.astype(np.float32)), 6, 3, "matrix"),
    (ocs.Zero(), 12, 3, "zero"),
    (ocs.NonNeg(), 12, 3, "diagonal"),
    (ocs.L1(0.3), 12, 3, "diagonal"),
    (ocs.L2(0.3), 12, 3, "scalar"),
    (ocs.GroupL2(0.3, group_size=3), 12, 3, "scalar"),
    (ocs.SecondOrderCone(), 6, 2, "symmetric"),
    (ocs.PSDCone(3), 9, 3, "symmetric"),
    (cones.SOCBatch(2, 2), 6, 2, "symmetric"),
    (ocs.ExpCone(), 3, 3, "nonsymmetric"),
    (ocs.PowerCone(0.5), 3, 3, "nonsymmetric"),
], ids=lambda v: type(v).__name__ if hasattr(v, "prox") else str(v))
def test_every_distinguished_block_class_refuses_with_its_reason(block, n, k, kind):
    """T7. One family per class of distinguished block, the exact message."""
    prob, _data = distinguished_family(block, n)
    k0, count, name = mc.primal_block_structure(block, n)
    cls = block.scaling_class
    sentence = {
        "scalar": f"{name} accepts exactly one primal step.",
        "diagonal": (f"{name} accepts a per-coordinate step and has no weighted prox for a "
                     "non-diagonal metric."),
        "matrix": (f"{name} defines prox_matrix, which the compiler does not call on the "
                   "primal side in this release (routing it is SPEC Item 1 territory on the "
                   "row side and is not part of this change)."),
        "zero": (f"{name} defines prox_matrix, which the compiler does not call on the "
                 "primal side in this release (routing it is SPEC Item 1 territory on the "
                 "row side and is not part of this change). Zero is evaluated as a proximal "
                 "block by the compiled loop (ocs/prox/library.py Zero.prox) even though the "
                 "certificate treats it as no block, so it is refused with the others. Route "
                 "the term through an operator to remove the distinguished block."),
        "symmetric": (f"{name} is a symmetric cone: the closed-form metric class that applies "
                      "to it is the automorphism tier (SPEC Item 2, metric_classes "
                      "'automorphism' on the dual side), and its application to a PRIMAL cone "
                      "block is NOT built in this release."),
        "nonsymmetric": (f"{name} is not a symmetric cone and admits no closed-form weighted "
                         "projection beyond the scalar class."),
    }[kind]
    expected = _core(k, name, cls, count, k0) + " " + sentence
    with pytest.raises(ValueError) as info:
        build(prob, metric_columns=("full", k))
    assert str(info.value) == expected
    if kind == "symmetric":
        assert "automorphism tier" in str(info.value) and "NOT built" in str(info.value)


# --------------------------------------------- the certificate never sees T
def test_the_certificate_is_byte_identical_with_and_without_T():
    """T8. The same (x, y, data) scored by a declared and an undeclared compile
    gives byte-identical certificate components: T changes the iterates, not
    the certificate."""
    prob, data, _K, _n, k = mixed_family()
    declared = prob.compile(equilibrate="construct", metric_columns=("full", k), **SOLVE)
    undeclared = prob.compile(equilibrate="sgd", tol=1e-5, max_blocks=400, accuracy="residual")
    sol = declared.solve_instance(data)
    assert sol.status == "solved", (sol.status, sol.iters, sol.residual)
    a = declared.certificate(sol.x, sol.y, data)
    b = undeclared.certificate(sol.x, sol.y, data)
    assert sorted(a) == sorted(b)
    for key in sorted(a):
        va, vb = np.asarray(a[key], np.float64), np.asarray(b[key], np.float64)
        assert va.tobytes() == vb.tobytes(), (key, a[key], b[key])


def test_the_solution_is_invariant_to_the_declaration():
    """T9. Both compiles solve to 1e-5 and agree; a budget-exhausted return on
    either side fails the test with that reason, never counts as agreement."""
    prob, data, _K, _n, k = mixed_family()
    declared = prob.compile(equilibrate="construct", metric_columns=("full", k), **SOLVE)
    undeclared = prob.compile(equilibrate="sgd", tol=1e-5, max_blocks=400, accuracy="residual")
    sd, su = declared.solve_instance(data), undeclared.solve_instance(data)
    for name, sol in (("declared", sd), ("undeclared", su)):
        assert sol.status == "solved", (name, sol.status, sol.iters, sol.residual)
        assert sol.termination == "criterion", (name, sol.termination)
    assert rel(jax.device_get(sd.x), jax.device_get(su.x)) < 5e-3


# ------------------------------------------------------------ operator only
class _Guarded(ocs.LinearOperator):
    """A Dense operator behind a guard: matvec, rmatvec, shape and
    well_conditioned (the LinearOperator interface) are reachable, and ANY
    other attribute access raises AttributeError (the compiler's
    getattr(op, ..., default) patterns need that exact exception) and is
    recorded. No entry of K is reachable through it."""

    def __init__(self, K):
        self._op = ocs.Dense(K)
        super().__init__(self._op.shape)
        self.touched = set()
        self.calls = {"matvec": 0, "rmatvec": 0}

    def matvec(self, x):
        self.calls["matvec"] += 1
        return self._op.matvec(x)

    def rmatvec(self, y):
        self.calls["rmatvec"] += 1
        return self._op.rmatvec(y)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self.touched.add(name)
        raise AttributeError(name)


def test_operator_only():
    """T10, restated per refutation D2: the operators are reached only through
    matvec and rmatvec (any other attribute is an AttributeError, recorded),
    and the REPORTED budget equals 2 x updates x probes + 30 x norm calls.
    Python-level call counts are trace-site counts under jit and are not
    compared with the budget."""
    _prob, data, K, n, k = mixed_family()
    rows = [(0, n), (n, n + 12), (n + 12, n + 18)]
    ops = [_Guarded(K[a:b]) for a, b in rows]
    x = ocs.Variable(n)
    prob = ocs.minimize([ocs.sum_squares(ops[0] @ x + ocs.Parameter("l0")),
                         ocs.group_l2(ops[1] @ x + ocs.Parameter("l1"), lam=0.3, group_size=k),
                         ocs.nonneg(ops[2] @ x + ocs.Parameter("l2"))],
                        cost=ocs.Parameter("c"))
    updates, probes = 12, 2
    comp = build(prob, metric_columns=("full", k), metric_updates=updates, metric_probes=probes)
    budget = comp.meta["metric_construction"]["operator_products"]
    assert budget["normalization"] in (30, 60)          # one or two 15-iteration measurements
    norm_calls = budget["normalization"] // 30
    assert budget["gradient"] == 2 * updates * probes
    assert budget["total"] == 2 * updates * probes + 30 * norm_calls
    assert comp.applied["rmatvec"].__name__ == "iteration_rmatvec"
    for op in ops:
        assert op.touched == set(), op.touched
        assert op.calls["matvec"] > 0 and op.calls["rmatvec"] > 0
    assert comp.meta["metric_construction"]["columns"]["shape"]["k"] == k


# ------------------------------------------------------------- the records
def test_the_recorded_shape_is_the_applied_shape():
    """T11. (Q, w) from the public record reconstructs what the loop applies,
    and (refutation D6) the three places T is recorded agree with each other."""
    prob, _data, K, n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    S, _Q, w = shape_from_meta(comp)
    Sn = np.kron(np.eye(n // k), S)
    y = np.random.default_rng(11).normal(size=K.shape[0]).astype(np.float32)
    assert rel(jax.device_get(comp.applied["rmatvec"](jnp.asarray(y))),
               Sn @ (K.T @ y.astype(np.float64))) < 1e-5
    cond = float(np.max(w) / np.min(w))
    shape = comp.meta["metric_construction"]["columns"]["shape"]
    primal = comp.meta["effective_metric"]["primal"]
    bcol = comp.meta["block_metrics"]["columns"]
    group = primal["metric_sharing_groups"][0]
    assert group["cond"] == cond == shape["cond"] == bcol["cond"]
    assert group["k"] == shape["k"] == bcol["k"] == k
    assert group["blocks"] == [0, n // k] and primal["column_blocks"] == {"k": k, "count": n // k}
    assert bcol["count"] == n // k and group["kind"] == shape["kind"] == bcol["kind"] == "full"
    enforced = comp.meta["metric_construction"]["columns"]["enforced_on_applied_steps"]
    assert enforced["shape_cond"] == cond and enforced["distinct_applied_carriers"] == 1
    assert primal["distinct_steps"] == 1


def test_the_report_and_the_applied_metric_never_contradict_each_other():
    """The invariant of test_metric_columns_contract.py on the full kind."""
    prob, _data, _K, _n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    columns = comp.meta["metric_construction"]["columns"]
    assert columns["metric_sharing_groups"] == 1
    assert comp.meta["effective_metric"]["primal"]["distinct_steps"] <= 1
    assert columns["enforced_on_applied_steps"]
    assert columns["declared"] == ("full", k)
    assert columns["column_blocks"] == {"k": k, "count": _n // k, "sharing": "term"}
    identity = comp.meta["metric_construction"]["restriction_cache_identity"]
    assert ("column_blocks", k, _n // k, "term") in identity
    other = build(prob, metric_columns=("full", 2)).meta["metric_construction"]
    assert other["restriction_cache_identity"] != identity


def test_the_enforcement_branch_for_the_full_kind():
    """T12. Direct calls: a uniform scalar passes and is recorded, a step that
    varies inside a block is refused, two per-block carriers are refused."""
    record = {"kind": "full", "spans": None, "metric_sharing_groups": 1,
              "column_blocks": {"k": 3, "count": 4}, "shape": {"cond": 2.0}}
    ok = dict(record)
    cm._enforce_column_restriction(jnp.asarray(0.25, jnp.float32), 12, ok)
    assert ok["enforced_on_applied_steps"] == {
        "declared_sharing_groups": 1, "column_blocks": 4, "block_size": 3,
        "distinct_applied_carriers": 1, "shape_cond": 2.0, "shape_k": 3}
    varying = np.full(12, 0.25, np.float32); varying[4] = 0.5
    with pytest.raises(ValueError) as info:
        cm._enforce_column_restriction(jnp.asarray(varying), 12, dict(record))
    assert str(info.value) == (
        "the applied primal metric does not honour the declared column restriction: a full "
        "column shape T = c (I kron S) requires one carrier c, uniform inside every column "
        "block of 3 coordinates (4 blocks declared), and the applied steps carry a value that "
        "varies inside a block. The compile fails rather than apply a metric from a class the "
        "caller did not declare.")
    per_block = np.repeat(np.asarray([0.25, 0.5, 0.25, 0.25], np.float32), 3)
    with pytest.raises(ValueError) as info:
        cm._enforce_column_restriction(jnp.asarray(per_block), 12, dict(record))
    assert str(info.value) == (
        "the applied primal metric does not honour the declared column restriction: a full "
        "column shape T = c (I kron S) requires one carrier c, uniform inside every column "
        "block of 3 coordinates (4 blocks declared), and the applied steps carry 2 distinct "
        "values. The compile fails rather than apply a metric from a class the caller did not "
        "declare.")


# --------------------------------------------------------------- refusals
# Read with a fallback so the module still IMPORTS on a tree without the
# change and every case fails on its own assertion (the tests-can-fail run).
SPELLINGS = getattr(mc, "COLUMN_SPELLINGS", "<COLUMN_SPELLINGS is absent on this tree>")
_CAP = getattr(mc, "MAX_COLUMN_FULL_K", 64)


def _malformed(value):
    return (f"metric_columns={value!r} is not a column declaration. A full column shape is "
            "declared as ('full', k) or ('full', k, 'term') with an integer block size k. Tied "
            "spans are declared as a sequence of (start, end) pairs. See COLUMN_CONTRACT.")


@pytest.mark.parametrize("value, expected", [
    ("full", SPELLINGS + " metric_columns='full' without a block size is refused: with no "
     "primal proximal block every primal coordinate is its own block, so a 'full' metric of "
     "block size 1 would be the free per-coordinate diagonal under a wider name. Declare "
     "('full', k) with the block size k that the coordinates to be mixed occupy contiguously."),
    ("matrix", SPELLINGS),
    (("full", 1), "metric_columns=('full', 1) is the free per-coordinate diagonal under a wider "
     "name and is refused. Declare 'diagonal', or a block size of at least 2."),
    (("full", 5), "metric_columns=('full', 5): 12 primal coordinates is not a multiple of the "
     "block size 5. The column blocks are the contiguous groups [i k, (i+1) k) and must tile "
     "[0, 12)."),
    (("full", 3, "per_block"), "metric_columns=('full', 3, 'per_block'): a full column shape "
     "must be shared ('term'). One independent 3x3 shape for each of 4 column blocks is "
     "neither built nor applied by this release: the application layer stores one (Q, w) "
     "pair per metric-sharing group and the constructor's geodesic state has no batched "
     "per-block full kind. Declare ('full', 3) or ('full', 3, 'term')."),
    (("full", _CAP + 1), f"metric_columns=('full', {_CAP + 1}): the block size exceeds "
     f"MAX_COLUMN_FULL_K = {_CAP}. A full column shape is stored once as a {_CAP + 1}x{_CAP + 1} "
     "eigenpair and eigendecomposed inside every constructor update and inside the "
     "step-condition measurement, so the block size is capped. Declare a smaller block, or "
     "tied (start, end) spans."),
    (("full", "term"), _malformed(("full", "term"))),
    (("scalar", 3), _malformed(("scalar", 3))),
    (("full", 3, "shared"), _malformed(("full", 3, "shared"))),
    (("full", 0), _malformed(("full", 0))),
    (["full"], _malformed(["full"])),
], ids=repr)
def test_the_spellings_that_are_refused(value, expected):
    """T13. Each refused spelling with its exact message, on n = 12."""
    prob, _data, _K, n, _k = mixed_family()
    assert n == 12
    with pytest.raises(ValueError) as info:
        build(prob, metric_columns=value)
    assert str(info.value) == expected


def test_the_list_spelling_is_the_same_declaration():
    """Refutation D7: a list ['full', 3] is a class declaration too, never the
    opaque unpack error of the span parser."""
    prob, _data, _K, _n, k = mixed_family()
    comp = build(prob, metric_columns=["full", k])
    assert comp.meta["metric_construction"]["columns"]["declared"] == ("full", k)
    assert comp.meta["metric_construction"]["columns"]["kind"] == "full"


def test_a_column_shape_requires_the_relative_residual():
    """Refusal 8, and the construct-only guard."""
    prob, _data, _K, _n, k = mixed_family()
    with pytest.raises(ValueError) as info:
        build(prob, metric_columns=("full", k), residual_norm="iterate")
    assert str(info.value) == (
        "a constructed column shape requires residual_norm='relative' (metric-free stopping): "
        "the historical 'iterate' and 'data' residuals multiply tau elementwise and cannot take "
        "an operator step.")
    with pytest.raises(ValueError, match="require equilibrate='construct'"):
        prob.compile(max_blocks=2, tol=1e-3, metric_columns=("full", k))


# ------------------------------------------------------- the default is intact
def test_nothing_declared_is_still_byte_identical_to_the_shipped_program():
    """T14. On the no-distinguished-block fixture, equilibrate='sgd' and a
    construct compile with nothing declared hand the solver the same bytes and
    write the same metadata, with no column-shape key anywhere."""
    prob, _data, _K, _n, _k = mixed_family()
    _CAPTURED.clear()
    shipped = prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3)
    shipped_steps = dict(_CAPTURED[-1])
    built = build(prob)
    assert built.meta["metric_construction"]["bit_identical_to_sgd"] is True
    assert built.meta["metric_construction"]["columns"]["declared"] is None
    assert "column_blocks" not in built.meta["metric_construction"]["columns"]
    assert "shape" not in built.meta["metric_construction"]["columns"]
    assert "columns_state" not in built.meta["metric_construction"]
    assert len(built.meta["metric_construction"]["restriction_cache_identity"]) == 5
    assert built.applied["tau"].tobytes() == shipped_steps["tau"].tobytes()
    assert built.applied["sigma"].tobytes() == shipped_steps["sigma"].tobytes()
    # The adjoint handed to the loop is the plain stacked closure, not the
    # shaped one, on both compiles.
    assert built.applied["rmatvec"].__name__ == "rmatvec"
    assert shipped_steps["rmatvec"].__name__ == "rmatvec"
    assert built.meta["scaled_norm"] == shipped.meta["scaled_norm"]
    assert "block_metrics" not in built.meta and "block_metrics" not in shipped.meta
    for comp in (built, shipped):
        primal = comp.meta["effective_metric"]["primal"]
        assert "metric_sharing_groups" not in primal and "column_blocks" not in primal
    assert built.meta["effective_metric"] == shipped.meta["effective_metric"]


# ---------------------------------------------- the two derived residuals
def test_refine_and_diff_use_the_shaped_iteration():
    """T15. refine moves a solved point by less than 1e-3 relative (a refine
    built on the unshaped residual would move it toward a different fixed
    point), and the implicit gradient agrees with a finite difference."""
    prob, data, _K, n, k = mixed_family()
    comp = prob.compile(equilibrate="construct", metric_columns=("full", k), **SOLVE)
    sol = comp.solve_instance(data)
    assert sol.status == "solved", (sol.status, sol.iters, sol.residual)
    xr, _yr = comp.refine(sol.x, sol.y, data, steps=1)
    assert rel(jax.device_get(xr), jax.device_get(sol.x)) < 1e-3
    try:
        def value(c):
            return jnp.sum(comp.solve_diff({**data, "c": c}))
        grad = np.asarray(jax.device_get(jax.grad(value)(data["c"])), np.float64)
    except NotImplementedError as exc:
        pytest.skip(f"solve_diff unavailable on this family for a reason unrelated to T: {exc}")
    h = 1e-2
    c0 = np.asarray(jax.device_get(data["c"]), np.float64)
    for direction in (0, 1):
        e = np.zeros(n); e[direction] = 1.0
        plus = float(value(jnp.asarray(c0 + h * e, jnp.float32)))
        minus = float(value(jnp.asarray(c0 - h * e, jnp.float32)))
        fd = (plus - minus) / (2 * h)
        assert abs(grad[direction] - fd) <= 5e-2 * max(abs(fd), abs(grad[direction]), 1e-3), \
            (direction, grad[direction], fd)


def test_compact_layout_folds_the_cost_through_T():
    """T16. The compact engine's primal prox folds the cost through S too,
    including the absent-cost branch, which gives x - T 0 = x without NaN."""
    prob, data, _K, n, k = mixed_family()
    comp = build(prob, metric_columns=("full", k), data_layout="compact")
    S, _Q, _w = shape_from_meta(comp)
    tau = float(comp.applied_last["tau"])
    T = tau * np.kron(np.eye(n // k), S)
    parts, cost, token = comp._resolve(data)
    c64 = np.asarray(jax.device_get(cost), np.float64)
    prox = comp.applied_last["prox_primal"]
    x = np.random.default_rng(13).normal(size=n).astype(np.float32)
    folded = jax.device_get(prox(jnp.asarray(x), jnp.asarray(tau, jnp.float32), (parts, cost, token)))
    assert rel(folded, x.astype(np.float64) - T @ c64) < 1e-5
    prob0, data0, _K0, _n0, _k0 = mixed_family(cost=False)
    comp0 = build(prob0, metric_columns=("full", k), data_layout="compact")
    parts0, cost0, token0 = comp0._resolve(data0)
    assert cost0 is None
    tau0 = float(comp0.applied_last["tau"])
    out = jax.device_get(comp0.applied_last["prox_primal"](jnp.asarray(x), jnp.asarray(tau0, jnp.float32),
                                                            (parts0, None, token0)))
    assert np.all(np.isfinite(out)) and np.array_equal(out, x)


# ------------------------------- refutation D1: the well-conditioned skip
def test_a_well_conditioned_family_still_builds_the_declared_shape():
    """Refutation D1. On a family whose dual operators all certify
    well_conditioned (FFTMask), the shipped path and the natural construct
    declaration take the scalar skip. A declared ('full', k) must NOT: it is
    built by the geodesic regime and applied, never replaced by one scalar
    under the name of a k x k class."""
    n, k = 16, 4
    rng = np.random.default_rng(21)
    idx1 = np.sort(rng.choice(n, size=6, replace=False))
    idx2 = np.sort(rng.choice(n, size=4, replace=False))
    x = ocs.Variable(n)
    prob = ocs.minimize([ocs.sum_squares(ocs.FFTMask(idx1, n) @ x + ocs.Parameter("a")),
                         ocs.nonneg(ocs.FFTMask(idx2, n) @ x + ocs.Parameter("b"))],
                        cost=ocs.Parameter("c"))
    plain = build(prob)
    assert plain.meta["metric_construction"]["regime"] == "scalar"     # the skip is reached
    assert plain.applied["tau"].ndim == 0
    comp = build(prob, metric_columns=("full", k))
    report = comp.meta["metric_construction"]
    assert report["regime"] == "geodesic"
    assert report["columns"]["kind"] == "full" and report["columns"]["shape"]["k"] == k
    assert comp.meta["block_metrics"]["columns"]["present"] is True
    assert comp.meta["block_metrics"]["measured_scaled_norm"] < 1.0
    assert "re-measured AFTER construction" in comp.meta["block_metrics"]["measured_scaled_norm_method"]
    assert abs(comp.meta["block_metrics"]["measured_scaled_norm"] / 0.9 - 1) <= 0.05
    assert comp.meta["effective_metric"]["primal"]["metric_sharing_groups"][0]["k"] == k
    S, _Q, _w = shape_from_meta(comp)
    Sn = np.kron(np.eye(n // k), S)
    y = rng.normal(size=comp.m).astype(np.float32)
    ops = [t.op for t in prob.terms]
    assert all(op.well_conditioned for op in ops)
    edges = np.cumsum([0] + [op.shape[0] for op in ops])
    plain_adjoint = np.asarray(jax.device_get(
        sum(op.rmatvec(jnp.asarray(y[a:b])) for op, a, b in zip(ops, edges[:-1], edges[1:]))),
        np.float64)
    assert rel(jax.device_get(comp.applied["rmatvec"](jnp.asarray(y))), Sn @ plain_adjoint) < 1e-5


# ------------------------------ the cuda/cpu step-0 divergence, fixed at its origin
def test_the_geodesic_exponential_step_stays_finite_below_the_box_floor():
    """The k=2 column build reached constructor status 2 on cuda and 0 on cpu.
    Measured origin (scripts/diag_k2_nonfinite.py on Mazeppa): at step 0 the
    exponent -eta T of the shared full column step has eigenvalues near -164
    and -100, float32 exp of which is zero on cpu (flushed) and a denormal on
    cuda, and cuSOLVER's eigh of the denormal-scale step matrix returns NaN.
    The construction now floors the exponent at -GEODESIC_EXP_FLOOR per
    eigen-direction. This calls update_side directly with a tangent far below
    the floor and asserts a finite step that lands exactly on the box floor,
    on whichever backend runs the test."""
    k, count, box, mu = 2, 6, 2.0, 1.0
    group = mc._Group("shared_full", k, count)
    state = (jnp.eye(k, dtype=jnp.float32),)
    rng = np.random.default_rng(0)
    eta = 2 / ((mu / 4) * (1 + 20))                      # the step-0 schedule
    # Probe images with energy ~1e4 per coordinate: exponent eigenvalues near
    # -eta * 0.5 * 1e4, far below -GEODESIC_EXP_FLOOR.
    samples = jnp.asarray(100.0 * rng.normal(size=(2, count * k)), jnp.float32)
    new, finite = mc.update_side((group,), state, samples, math.sqrt(2.5), mu, box, eta)
    assert bool(finite), "the floored step must be finite arithmetic on every backend"
    got = np.asarray(jax.device_get(new[0]), np.float64)
    assert np.all(np.isfinite(got))
    floor = math.exp(-2 * box)
    assert rel(got, floor * np.eye(k)) < 1e-5, got
    # A mixed case: one direction far below the floor, one inside the box. The
    # floored direction is projected to the floor and the other is untouched.
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    z = rng.normal(size=(64, k))
    z = z @ Q @ np.diag([300.0, 1.0]) @ Q.T                # anisotropic energy
    samples = jnp.asarray(np.tile(z, (1, count)).reshape(-1, count * k)[:8], jnp.float32)
    new, finite = mc.update_side((group,), state, samples, math.sqrt(2.5), mu, box, eta)
    assert bool(finite)
    eig = np.linalg.eigvalsh(np.asarray(jax.device_get(new[0]), np.float64))
    assert np.all(np.isfinite(eig)) and eig.min() >= floor * (1 - 1e-5)
    assert eig.max() <= math.exp(2 * box) * (1 + 1e-5)
    assert mc.GEODESIC_EXP_FLOOR == 60.0
    assert comp_report_has_floor(mixed_family()[0])


def comp_report_has_floor(prob):
    comp = build(prob, metric_columns=("full", 3))
    rec = comp.meta["metric_construction"]["exponent_floor"]
    return rec["value"] == -mc.GEODESIC_EXP_FLOOR and "cuda" in rec["why"]


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_the_full_column_build_is_finite_and_re_measured_at_every_block_size(k):
    """The probe sweep as a permanent test: every block size that tiles n = 12
    builds with constructor status 0 (finite arithmetic throughout), and on
    every one the step condition is RE-MEASURED after construction by the
    compiler on the decoded shape and the applied steps, recorded in meta,
    below 1, and at the constructor's post-factor target 0.9 within the design
    tolerance of five percent. k = 2 is the case that failed on cuda."""
    prob, _data, K, n, _k = mixed_family()
    comp = build(prob, metric_columns=("full", k))
    report = comp.meta["metric_construction"]
    assert report["constructor_status"] == 0
    assert report["regime"] == "geodesic"
    assert report["columns"]["shape"]["k"] == k
    bmeta = comp.meta["block_metrics"]
    measured = bmeta["measured_scaled_norm"]
    assert math.isfinite(measured) and measured < 1.0
    assert abs(measured / 0.9 - 1) <= 0.05, measured
    assert "re-measured AFTER construction" in bmeta["measured_scaled_norm_method"]
    assert "_power_norm" in bmeta["measured_scaled_norm_method"]
    assert bmeta["columns"] == {"present": True, "k": k, "count": n // k,
                                "cond": report["columns"]["shape"]["cond"], "kind": "full",
                                "carrier": "one scalar",
                                "effective": "T_i = c S, one shared full SPD S = Q diag(w) Q^T, "
                                             "applied to the adjoint and to the cost"}
    # The re-measure is on the operator the loop applies: check it against the
    # exact float64 largest singular value of the explicit test object.
    S, _Q, _w = shape_from_meta(comp)
    w, V = np.linalg.eigh(S)
    root_S = V @ np.diag(np.sqrt(w)) @ V.T
    tau = float(comp.applied["tau"])
    sigma = np.broadcast_to(comp.applied["sigma"].astype(np.float64), (K.shape[0],))
    B = np.diag(np.sqrt(sigma)) @ K @ (math.sqrt(tau) * np.kron(np.eye(n // k), root_S))
    smax = float(np.linalg.svd(B, compute_uv=False)[0])
    assert measured <= smax * (1 + 1e-4) and measured >= 0.8 * smax, (measured, smax)


def test_a_full_declaration_with_no_built_shape_is_refused_by_the_compiler(monkeypatch):
    """Refutation D1, the compiler's side of the repair. If any constructor
    path returns a 'full' column record without its shape carrier (the scalar
    skip would, were it ever to accept the full kind again), the compile
    refuses instead of applying one scalar step under the name of a k x k
    class. Exercised by stripping the carrier from a genuine construction."""
    prob, _data, _K, _n, k = mixed_family()
    original = mc.construct_metric

    def stripped(*args, **kwargs):
        out = original(*args, **kwargs)
        out[4]["columns"].pop("shape_carrier", None)
        return out

    monkeypatch.setattr(mc, "construct_metric", stripped)
    with pytest.raises(ValueError) as info:
        build(prob, metric_columns=("full", k))
    message = str(info.value)
    assert message.startswith(f"metric_columns=('full', {k}) was declared and the constructor "
                              "(regime 'geodesic') returned no column shape")
    assert "refuses rather than apply one scalar step under the name of a k x k column class" in message
