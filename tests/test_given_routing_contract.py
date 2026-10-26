"""The routing contract behind ``equilibrate=("given", tau, sigma)`` (0.3.3).

Since v23 the compiler may fold an explicit nonsingular diagonal least-squares
term into the primal step (``ocs.model.compile.FOLD_POLICY``). It does so only
WITHOUT an explicit metric mapping, so a plain compile and a ``("given", ...)``
compile of the same family can have different dual row counts. This file pins
the contract that makes that explicit:

* every compiled family records ``meta["routing"]`` (declared term -> role ->
  dual term -> rows -> step and metric source -> witness rows of ``sol.y``);
* ``sigma`` may be a dict ``{dual_term_index: scalar or (rows,) array}`` bound
  to the dual terms by position, the same indices as ``block_metrics``;
* a wrong-length vector, a wrong key, a missing term or a wrong-length entry
  is refused by name, and the refusal names the compiled layout and the fold.

The controls: a mixed quadratic/SOC model, the same model with its blocks
reordered, constructed versus supplied steps on the same effective layout
(must agree bit for bit), deliberately wrong shapes (must refuse by name), one
native standard step with the folded ridge against the float64 formula, and
the zero-iteration refusal of the default objective-gap route on the
block-metric family of tests/test_quadratic_block_metric.py.
"""
import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
import ocs.model.compile as cm
from ocs.ir import Term

F32 = np.float32
N = 24


# ------------------------------------------------------------------ helpers
def digest(a):
    a = np.asarray(jax.device_get(a))
    return a.dtype.str, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ball_terms(c, A, mu, ridge="diagonal", order="ridge_first"):
    """min c^T x + (mu/2)||x||^2  s.t. ||A x|| <= 1, the showcase's socp_ball shape.

    ``ridge="diagonal"`` writes the ridge as sum_squares(Diagonal(sqrt(mu)) @ x), the
    fold candidate; ``"identity"`` declares it as the identity-operator primal term
    (mu * (1/2)||x||^2, the same function); ``"dense"`` writes it through a Dense
    diagonal matrix, which the policy never folds (an opaque operator is not inspected).
    """
    n = c.size
    x = ocs.Variable(n)
    cone = ocs.soc(ocs.Dense(np.vstack([np.zeros((1, n), F32), A])) @ x
                   + np.concatenate([[1.0], np.zeros(n)]).astype(F32))
    if ridge == "diagonal":
        quad = ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, F32)) @ x)
    elif ridge == "dense":
        quad = ocs.sum_squares(ocs.Dense(np.sqrt(mu) * np.eye(n, dtype=F32)) @ x)
    else:
        quad = Term(ocs.Identity(n), ocs.SumSquares().scaled(float(mu)))
    terms = [quad, cone] if order == "ridge_first" else [cone, quad]
    return ocs.minimize(terms, cost=jnp.asarray(c, F32))


def ball_data(seed=15, n=N):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(F32), rng.standard_normal((n, n)).astype(F32), F32(1e-2)


_CAPTURED = []


def _capturing_make_solver(matvec, rmatvec, tau, sigma, *args, **kwargs):
    _CAPTURED.append((np.asarray(jax.device_get(tau)), np.asarray(jax.device_get(sigma))))
    return _capturing_make_solver.wrapped(matvec, rmatvec, tau, sigma, *args, **kwargs)


@pytest.fixture(autouse=True)
def capture_applied_steps(monkeypatch):
    _CAPTURED.clear()
    _capturing_make_solver.wrapped = cm.make_solver
    monkeypatch.setattr(cm, "make_solver", _capturing_make_solver)
    yield


def applied(comp):
    tau, sigma = _CAPTURED[-1]
    return (np.ascontiguousarray(np.broadcast_to(np.atleast_1d(tau), (comp.n,))),
            np.ascontiguousarray(np.broadcast_to(np.atleast_1d(sigma), (comp.m,))))


def roles(comp):
    return [t["role"] for t in comp.meta["routing"]["terms"]]


# ================================= 1. the mixed quadratic / SOC model routes
def test_plain_compile_folds_the_diagonal_ridge_and_an_explicit_mapping_does_not():
    c, A, mu = ball_data()
    plain = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=200)
    routing = plain.meta["routing"]
    assert plain.meta["folded_diagonal_quadratic"] is True
    assert routing["folded"] is True and routing["fold_candidate"] == 0
    assert routing["explicit_metric_mapping"] is False
    assert routing["dual_rows"] == plain.m == N + 1
    assert roles(plain) == ["primal_folded", "dual"]
    assert routing["terms"][0]["rows"] is None and routing["terms"][0]["dual_term"] is None
    assert "never formed" in routing["terms"][0]["witness"]
    assert routing["terms"][1] == {
        "declared_term": 1, "op": "Dense", "block": "SecondOrderCone", "offset": True,
        "role": "dual", "dual_term": 0, "rows": [0, N + 1], "step_source": "equilibrate='sgd'",
        "metric_source": "scalar tier: diag(sigma) on these rows", "witness": f"sol.y[0:{N + 1}]"}
    assert routing["fold_policy"] == cm.FOLD_POLICY
    tau = jnp.full((N,), F32(0.01))
    given = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=200,
                                         equilibrate=("given", tau, jnp.full((2 * N + 1,), F32(0.1))))
    routing = given.meta["routing"]
    assert given.meta["folded_diagonal_quadratic"] is False
    assert routing["folded"] is False and routing["fold_candidate"] == 0
    assert routing["explicit_metric_mapping"] is True
    assert routing["dual_rows"] == given.m == 2 * N + 1
    assert roles(given) == ["dual", "dual"]
    assert routing["terms"][0]["rows"] == [0, N] and routing["terms"][0]["step_source"] == f"given sigma[0:{N}]"
    assert routing["terms"][1]["rows"] == [N, 2 * N + 1] and routing["terms"][1]["witness"] == f"sol.y[{N}:{2 * N + 1}]"
    # a family with an identity primal term has no fold candidate on either path
    identity = ball_terms(c, A, mu, ridge="identity").compile(tol=1e-5, max_blocks=200)
    assert identity.meta["routing"]["fold_candidate"] is None and roles(identity) == ["primal", "dual"]
    assert identity.m == N + 1
    # an opaque (Dense) diagonal is never inspected, so it is never folded
    dense = ball_terms(c, A, mu, ridge="dense").compile(tol=1e-5, max_blocks=200)
    assert dense.meta["routing"]["fold_candidate"] is None and roles(dense) == ["dual", "dual"]
    assert dense.m == 2 * N + 1


# ========================================= 2. refusals, deterministic by name
def test_a_sigma_of_the_plain_layout_is_refused_by_name_on_the_explicit_layout():
    c, A, mu = ball_data()
    plain = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=200)
    sigma = np.full(plain.m, F32(0.1))                         # the 25 rows of the FOLDED layout
    with pytest.raises(ValueError) as err:
        ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=200,
                                     equilibrate=("given", jnp.full((N,), F32(0.01)), jnp.asarray(sigma)))
    message = str(err.value)
    for needle in ("sigma has shape (25,)", "broadcast to (49,)", "the compiled model has 49 dual rows",
                   "dual term 0 = declared term 0 (SumSquares over Diagonal, rows 0:24)",
                   "dual term 1 = declared term 1 (SecondOrderCone over Dense, rows 24:49)",
                   "nothing is folded into the primal step",
                   "folds it into the primal step and has 25 dual rows",
                   "meta['routing']"):
        assert needle in message, needle


@pytest.mark.parametrize("mapping, needle", [
    ({0: 0.1}, "sigma has no entry for dual term 1 = declared term 1 (SecondOrderCone over Dense, rows 24:49); it needs a scalar or a (25,) array"),
    ({1: 0.1}, "sigma has no entry for dual term 0 = declared term 0 (SumSquares over Diagonal, rows 0:24); it needs a scalar or a (24,) array"),
    ({0: 0.1, 1: 0.1, 2: 0.1}, "sigma key 2 is not a dual term index in [0, 2)"),
    ({0: 0.1, "soc": 0.1}, "sigma key 'soc' is not a dual term index in [0, 2)"),
    ({0: 0.1, True: 0.1}, "sigma key True is not a dual term index in [0, 2)"),
    ({0: 0.1, 1: np.full(24, 0.1, F32)}, "sigma[1] has shape (24,); dual term 1 = declared term 1 (SecondOrderCone over Dense, rows 24:49) takes a scalar or a (25,) array"),
    ({0: np.full((24, 1), 0.1, F32), 1: 0.1}, "sigma[0] has shape (24, 1); dual term 0 = declared term 0 (SumSquares over Diagonal, rows 0:24) takes a scalar or a (24,) array"),
    ({}, "sigma is an empty dict; the compiled model has 49 dual rows"),
])
def test_wrong_dict_entries_are_refused_by_name(mapping, needle):
    c, A, mu = ball_data()
    with pytest.raises(ValueError, match=re.escape(needle)):
        ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=200,
                                     equilibrate=("given", jnp.full((N,), F32(0.01)), mapping))


def test_dict_entries_still_pass_the_ordinary_given_metric_checks():
    c, A, mu = ball_data()
    tau = jnp.full((N,), F32(0.01))
    with pytest.raises(ValueError, match=r"sigma\[30\] = .* is invalid"):
        ball_terms(c, A, mu).compile(equilibrate=("given", tau, {0: 0.1, 1: np.where(np.arange(25) == 6, -1.0, 0.1).astype(F32)}))
    nonuniform = np.full(25, 0.1, F32)
    nonuniform[3] = 0.2
    with pytest.raises(ValueError, match="sigma is not uniform on dual block 1"):
        ball_terms(c, A, mu).compile(equilibrate=("given", tau, {0: 0.1, 1: nonuniform}))


# ================================ 3. steps bound by position reach their rows
def test_steps_bound_by_dual_term_reach_the_rows_they_name_and_equal_the_vector_form():
    c, A, mu = ball_data()
    tau = jnp.full((N,), F32(0.01))
    s0, s1 = F32(0.01), F32(0.02)
    by_term = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=50, block=10,
                                           equilibrate=("given", tau, {0: s0, 1: s1}))
    _tau, sigma = applied(by_term)
    assert np.all(sigma[:N] == s0) and np.all(sigma[N:] == s1)
    assert by_term.meta["routing"]["terms"][0]["step_source"] == f"given sigma[0] (dict entry bound to rows 0:{N})"
    assert by_term.meta["routing"]["terms"][1]["step_source"] == f"given sigma[1] (dict entry bound to rows {N}:{2 * N + 1})"
    flat = np.concatenate([np.full(N, s0, F32), np.full(N + 1, s1, F32)])
    by_row = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=50, block=10,
                                          equilibrate=("given", tau, jnp.asarray(flat)))
    assert np.array_equal(applied(by_row)[1], sigma)
    a, b = by_term.solve_instance({}), by_row.solve_instance({})
    assert (digest(a.x), digest(a.y), int(a.iters), str(a.status)) == (digest(b.x), digest(b.y), int(b.iters), str(b.status))
    # a per-row array inside the dict is honoured entry by entry on a diagonal-class term
    rows0 = np.linspace(0.02, 0.08, N).astype(F32)
    mixed = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=50, block=10,
                                         equilibrate=("given", tau, {0: rows0, 1: s1}))
    assert np.array_equal(applied(mixed)[1][:N], rows0)


# =========================================================== 4. reordered blocks
def test_reordered_blocks_bind_by_declared_position():
    c, A, mu = ball_data()
    plain = ball_terms(c, A, mu, order="cone_first").compile(tol=1e-5, max_blocks=200)
    routing = plain.meta["routing"]
    assert routing["folded"] is True and routing["fold_candidate"] == 1
    assert roles(plain) == ["dual", "primal_folded"]
    assert routing["terms"][0]["dual_term"] == 0 and routing["terms"][0]["rows"] == [0, N + 1]
    assert plain.m == N + 1
    tau = jnp.full((N,), F32(0.01))
    s_cone, s_ridge = F32(0.02), F32(0.01)
    given = ball_terms(c, A, mu, order="cone_first").compile(
        tol=1e-5, max_blocks=50, block=10, equilibrate=("given", tau, {0: s_cone, 1: s_ridge}))
    routing = given.meta["routing"]
    assert roles(given) == ["dual", "dual"]
    assert routing["terms"][0]["rows"] == [0, N + 1] and routing["terms"][0]["block"] == "SecondOrderCone"
    assert routing["terms"][1]["rows"] == [N + 1, 2 * N + 1] and routing["terms"][1]["block"] == "SumSquares"
    _tau, sigma = applied(given)
    assert np.all(sigma[:N + 1] == s_cone) and np.all(sigma[N + 1:] == s_ridge)
    with pytest.raises(ValueError, match=re.escape("dual term 1 = declared term 1 (SumSquares over Diagonal, rows 25:49)")):
        ball_terms(c, A, mu, order="cone_first").compile(equilibrate=("given", tau, {0: s_cone}))
    # the solve of the reordered declaration lands where the original order lands
    original = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=50, block=10,
                                            equilibrate=("given", tau, {0: s_ridge, 1: s_cone}))
    a, b = original.solve_instance({}), given.solve_instance({})
    # the same iteration up to the order of the adjoint's partial sums, so close, not bitwise
    assert str(a.status) == str(b.status)
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=1e-3, atol=1e-4)


# ===================== 5. constructed vs supplied, same effective layout, agree
def test_constructed_steps_resupplied_on_the_same_layout_reproduce_the_solve_bit_for_bit():
    c, A, mu = ball_data()
    built = ball_terms(c, A, mu, ridge="identity").compile(tol=1e-5, max_blocks=200, block=20,
                                                            equilibrate="sgd")
    tau, sigma = applied(built)
    norm = built.meta["scaled_norm"]
    supplied = ball_terms(c, A, mu, ridge="identity").compile(
        tol=1e-5, max_blocks=200, block=20, equilibrate=("given", jnp.asarray(tau), jnp.asarray(sigma), norm))
    for key in ("dual_rows", "folded", "fold_candidate"):
        assert built.meta["routing"][key] == supplied.meta["routing"][key]
    assert [(t["role"], t["rows"], t["witness"]) for t in built.meta["routing"]["terms"]] == \
           [(t["role"], t["rows"], t["witness"]) for t in supplied.meta["routing"]["terms"]]
    assert np.array_equal(applied(supplied)[0], tau) and np.array_equal(applied(supplied)[1], sigma)
    a, b = built.solve_instance({}), supplied.solve_instance({})
    assert (digest(a.x), digest(a.y), int(a.iters), str(a.status)) == (digest(b.x), digest(b.y), int(b.iters), str(b.status))


def test_the_folded_plain_compile_matches_the_identity_primal_declaration_of_the_same_ridge():
    """Folding writes (1/2)||sqrt(mu) x||^2 into the primal step; declaring the ridge as
    the identity-operator primal term mu * (1/2)||x||^2 is the same splitting. Same
    operator, same equilibration, so the same steps; the same solve up to the
    rounding of the two prox implementations (precomposed vs scaled SumSquares)."""
    c, A, mu = ball_data()
    folded = ball_terms(c, A, mu).compile(tol=1e-5, max_blocks=4000, block=100, accuracy="residual")
    tf, sf = applied(folded)
    declared = ball_terms(c, A, mu, ridge="identity").compile(tol=1e-5, max_blocks=4000, block=100,
                                                              accuracy="residual")
    td, sd = applied(declared)
    assert folded.m == declared.m == N + 1
    assert folded.meta["routing"]["folded"] and not declared.meta["routing"]["folded"]
    assert np.array_equal(tf, td) and np.array_equal(sf, sd)
    a, b = folded.solve_instance({}), declared.solve_instance({})
    assert str(a.status) == str(b.status) == "solved", (a.status, b.status)
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=1e-3, atol=1e-4)


# ================== 6. one native standard step with the folded ridge, float64
def _project_soc(w):
    t, z = w[0], w[1:]
    nz = np.linalg.norm(z)
    if nz <= t:
        return w
    if nz <= -t:
        return np.zeros_like(w)
    a = 0.5 * (1 + t / nz)
    return np.concatenate([[a * nz], a * z])


def test_one_native_standard_step_with_the_folded_ridge_matches_the_float64_formula():
    rng = np.random.default_rng(3)
    n = 6
    d = np.exp(rng.uniform(-0.5, 0.5, n)).astype(F32)
    A = rng.normal(size=(n, n)).astype(F32)
    cost = rng.normal(size=n).astype(F32)
    x = ocs.Variable(n)
    B = np.vstack([np.zeros((1, n), F32), A])
    e = np.concatenate([[1.0], np.zeros(n)]).astype(F32)

    def family(ridge):
        op = ocs.Diagonal(d) if ridge == "diagonal" else ocs.Dense(np.diag(d).astype(F32))
        return ocs.minimize([ocs.sum_squares(op @ x), ocs.soc(ocs.Dense(B) @ x + e)],
                            cost=jnp.asarray(cost))

    x0 = rng.normal(size=n).astype(F32)
    B64, e64, d64, c64, x064 = (np.asarray(v, np.float64) for v in (B, e, d, cost, x0))
    # the folded splitting: the ridge is the primal prox, the cone the only dual term
    folded = family("diagonal").compile(equilibrate="scalar", pdhg_variant="standard", block=1,
                                        max_blocks=1, tol=1e-12, accuracy="residual")
    assert folded.meta["routing"]["folded"] and folded.m == n + 1
    tau, sigma = (float(np.atleast_1d(v)[0]) for v in _CAPTURED[-1])
    y0 = (rng.normal(size=n + 1) * 0.1).astype(F32)
    sol = folded.solve_instance({}, warm_start=(jnp.asarray(x0), jnp.asarray(y0)))
    v = np.asarray(y0, np.float64) + sigma * (B64 @ x064 + e64)
    y1 = v - _project_soc(v)                                   # prox of the polar indicator
    x1 = (x064 - tau * (B64.T @ y1) - tau * c64) / (1 + tau * d64 ** 2)
    assert int(sol.iters) == 1
    np.testing.assert_allclose(np.asarray(sol.x), x1, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sol.y), y1, rtol=1e-4, atol=1e-6)      # cone rows are raw
    # the unfolded splitting (an opaque Dense diagonal is never folded): its own step
    unfolded = family("dense").compile(equilibrate="scalar", pdhg_variant="standard", block=1,
                                       max_blocks=1, tol=1e-12, accuracy="residual")
    assert not unfolded.meta["routing"]["folded"] and unfolded.m == 2 * n + 1
    tau_u, sigma_u = (float(np.atleast_1d(v)[0]) for v in _CAPTURED[-1])
    y0u = (rng.normal(size=2 * n + 1) * 0.1).astype(F32)
    solu = unfolded.solve_instance({}, warm_start=(jnp.asarray(x0), jnp.asarray(y0u)))
    y0u64 = np.asarray(y0u, np.float64)
    y_ridge = (y0u64[:n] + sigma_u * (d64 * x064)) / (1 + sigma_u)
    vu = y0u64[n:] + sigma_u * (B64 @ x064 + e64)
    y_cone = vu - _project_soc(vu)
    x1u = x064 - tau_u * (d64 * y_ridge + B64.T @ y_cone) - tau_u * c64
    np.testing.assert_allclose(np.asarray(solu.x), x1u, rtol=1e-4, atol=1e-6)
    assert not np.allclose(x1u, x1, rtol=1e-3, atol=1e-4)      # a different splitting, a different step
    # the represented problem is the same: both splittings converge to the same point
    a = family("diagonal").compile(tol=1e-6, max_blocks=4000, block=50, accuracy="residual").solve_instance({})
    b = family("dense").compile(tol=1e-6, max_blocks=4000, block=50, accuracy="residual").solve_instance({})
    assert str(a.status) == str(b.status) == "solved", (a.status, b.status)
    np.testing.assert_allclose(np.asarray(b.x), np.asarray(a.x), rtol=2e-3, atol=2e-4)


# ===== 7. the default route on the block-metric family: refused, zero iterations
def test_the_default_objective_gap_route_refuses_the_block_metric_family_before_optimizer_work():
    """The contract test_quadratic_block_metric.py::test_one_native_standard_step...
    runs under: on a family whose Quadratic block declares no conjugate contract, the
    default route returns the warm start untouched (CHANGES.md 0.3.3, v23), and the
    residual route runs the step. The step itself is checked in that test."""
    from test_quadratic_block_metric import quadratic_family, rotation
    from ocs.prox import block_metric as bm
    prob, data, _K = quadratic_family()
    tau = np.full(8, 0.01, F32)
    sigma = np.full(26, 0.01, F32)
    sigma[12:15] = 0.1
    kwargs = dict(equilibrate=("given", tau, sigma, 0.5),
                  block_metrics={0: bm.BlockMetric([1.0, 4.0, 20.0], rotation(5)),
                                 1: bm.BlockMetric([1.0, 1.5, 2.5], rotation(8))},
                  pdhg_variant="standard", block=1, max_blocks=1, tol=1e-12)
    x0, y0 = jnp.asarray(np.random.default_rng(1).normal(size=8), F32), jnp.zeros(26, F32)
    default = prob.compile(**kwargs)
    assert default.meta["objective_bound_capable"] is False
    assert default.meta["objective_bound_missing_contracts"] == ["Quadratic"]
    assert default.meta["routing"]["fold_candidate"] is None and roles(default) == ["primal", "dual", "dual", "dual", "dual"]
    sol = default.solve_instance(data, warm_start=(x0, y0))
    assert sol.status == "unverified" and sol.termination == "unsupported" and int(sol.iters) == 0
    assert "Quadratic" in sol.accuracy_detail
    assert np.array_equal(np.asarray(sol.x), np.asarray(x0))
    residual = prob.compile(accuracy="residual", **kwargs)
    step = residual.solve_instance(data, warm_start=(x0, y0))
    assert int(step.iters) == 1 and step.accuracy_criterion == "fixed_point"
    assert not np.array_equal(np.asarray(step.x), np.asarray(x0))
