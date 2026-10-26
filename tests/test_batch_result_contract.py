"""Public batch results retain per-instance status and iteration semantics."""
import numpy as np
import jax
import jax.numpy as jnp
import ocs


def _equality_family(*, nonnegative=False, max_blocks=0):
    x = ocs.Variable(1)
    op = ocs.LinearOperator.from_forward(lambda v: v, n=1, m=1)
    terms = [ocs.equality(op @ x - ocs.Parameter("b"))]
    if nonnegative:
        terms.insert(0, ocs.nonneg(x))
    return ocs.minimize(terms).compile(
        equilibrate="scalar", block=10, max_blocks=max_blocks, tol=1e-3,
    )


def test_pdhg_batch_statuses_match_independent_zero_iteration_checks():
    compiled = _equality_family()
    data = {"b": jnp.asarray([[0.0], [1.0], [jnp.nan]])}
    results = compiled.solve_many_results(data)
    assert [r.status for r in results] == ["solved", "inaccurate", "numerical_error"]
    assert all(r.iters == 0 for r in results)
    for i, result in enumerate(results):
        single = compiled.solve_instance({"b": data["b"][i]})
        assert result.status == single.status
        if np.isfinite(result.residual):
            assert result.residual == single.residual


def test_batch_has_valid_infeasibility_directions_and_true_iteration_counts():
    compiled = _equality_family(nonnegative=True, max_blocks=100)
    data = {"b": jnp.asarray([[1.0], [-1.0]])}
    raw = compiled.solve_many(data, accuracy="residual")
    assert len(raw) == 4
    # Explicitly select residual stopping on both interfaces. Default tuple
    # batching now follows objective-gap stopping and has separate controls.
    results = compiled.solve_many_results(data, accuracy="residual")
    assert [r.status for r in results] == ["solved", "primal_infeasible"]
    np.testing.assert_allclose(results[0].x, [1.0], atol=2e-3)
    assert results[1].certificate is not None
    # For x>=0 and x=-1, a positive y separates the feasible domains:
    # K.T*y>=0 and (-b)*y>0 under OCS's multiplier sign convention.
    ray = np.asarray(results[1].certificate)
    assert ray[0] > 0 and float(-data["b"][1] @ ray) > 0
    for i, result in enumerate(results):
        assert result.iters == int(raw[2][i]) * compiled.meta["block"]
        assert result.residual == float(raw[3][i])
        assert compiled.solve_instance(
            {"b": data["b"][i]}, accuracy="residual").status == result.status


def test_batch_preserves_invalid_warm_start_in_only_the_affected_row():
    compiled = _equality_family()
    results = compiled.solve_many_results(
        {"b": jnp.zeros((2, 1))},
        warm_start=(jnp.zeros((2, 1)), jnp.asarray([[0.0], [jnp.nan]])),
    )
    assert [r.status for r in results] == ["solved", "numerical_error"]


def _cg_family(iterations):
    x = ocs.Variable(2)
    diagonal = jnp.asarray([1.0, 2.0])
    op = ocs.LinearOperator.from_forward(lambda v: diagonal * v, n=2, m=2)
    return ocs.minimize([
        ocs.least_squares(op @ x - ocs.Parameter("b")),
    ]).compile(method="cg", cg_iters=iterations, tol=1e-4)


def test_cg_batch_has_same_solution_interface_and_keeps_legacy_tuple():
    # `_cg_family` has rho = 0, so the objective is not strongly convex and
    # nothing can be certified at the returned point: see
    # `test_cg_certifies_only_when_strongly_convex`.  This test is about the
    # legacy tuple and the residual statuses, so it asks for that criterion.
    compiled = _cg_family(10)
    data = {"b": jnp.asarray([[2.0, 6.0], [-2.0, 3.0]])}
    raw = compiled.solve_many(data)
    results = compiled.solve_many_results(data, accuracy="residual")
    assert len(raw) == 3
    for i, result in enumerate(results):
        assert result.status == "solved"
        assert result.y is None
        assert result.iters == int(raw[1][i])
        np.testing.assert_allclose(result.x, np.asarray(data["b"][i]) / [1, 2], atol=2e-5)
        assert compiled.solve_instance(
            {"b": data["b"][i]}, accuracy="residual").status == result.status


def test_cg_batch_does_not_label_unfinished_or_nonfinite_rows_solved():
    compiled = _cg_family(0)
    results = compiled.solve_many_results({"b": jnp.asarray([
        [0.0, 0.0], [1.0, 2.0], [jnp.nan, 1.0],
    ])}, accuracy="residual")
    assert [r.status for r in results] == ["solved", "inaccurate", "numerical_error"]
    assert all(r.iters == 0 for r in results)


def test_partial_quadratic_conjugate_capability_disables_gap_safely():
    from ocs.ir import Term, ProblemFamily
    from ocs.operators.common import Identity, Dense
    block = ocs.L1().with_linear_quadratic(e=jnp.asarray([0.0, 1.0]))
    family = ProblemFamily(n=2, terms=[
        Term(Identity(2), block),
        Term(Dense(np.eye(2)), ocs.SumSquares(), lambda data: -jnp.ones(2)),
    ])
    compiled = family.compile(equilibrate="scalar", max_blocks=0)
    assert not compiled.meta["gap_available"]
    result = compiled.solve_instance({})
    report = compiled.certificate(result.x, result.y)
    assert np.isnan(report["gap"])


def _status_fixture(*, with_certificates=False):
    """A prescribed numerical result isolates the public status contract."""
    from ocs.model.compile import CompiledFamily

    calls = {"resolve": 0}

    def resolve(raw):
        calls["resolve"] += 1
        return raw["marker"], raw["residual"]

    def solver(data, x0, y0):
        count = x0.shape[0]
        return x0, y0, jnp.arange(count, dtype=jnp.int32), data[1][:, 0]

    def checks(x, y, data):
        # Contradictory flags intentionally test precedence independently of
        # the domain checker, which ordinarily excludes this combination.
        marker = data[0][0]
        return {
            "primal_infeasible": (marker == 1) | (marker == 3),
            "unbounded": (marker == 2) | (marker == 3),
            "dual_direction": jnp.ones_like(y),
            "primal_direction": jnp.full_like(x, 2),
        }

    # A stub solver returning a PRESCRIBED residual.  There is no problem
    # here to certify an objective gap against, so this fixture exercises the
    # residual status contract, which is what it was written for.
    return CompiledFamily(
        solver, resolve, None, 1, 1, {"block": 7},
        infeas=checks if with_certificates else None, tol=1e-3,
        accuracy="residual",
    ), calls


def test_batched_status_preserves_python_tolerance_boundary_and_resolves_once():
    compiled, calls = _status_fixture()
    rounded_up = np.float32(1e-3)
    below = np.nextafter(rounded_up, np.float32(-np.inf))
    assert float(below) < compiled._tol < float(rounded_up)
    data = {"marker": jnp.zeros((3, 1)),
            "residual": jnp.asarray([[below], [rounded_up], [0]], jnp.float32)}
    results = compiled.solve_many_results(data)
    assert calls["resolve"] == 1
    assert [r.status for r in results] == ["solved", "inaccurate", "solved"]
    assert [r.iters for r in results] == [0, 7, 14]
    assert all(type(r.residual) is float and type(r.iters) is int for r in results)
    assert all(isinstance(r.x, jax.Array) and isinstance(r.y, jax.Array)
               for r in results)
    for i, result in enumerate(results):
        single = compiled.solve_instance({k: v[i] for k, v in data.items()})
        assert result.status == single.status
        assert result.residual == single.residual


def test_batched_status_keeps_all_precedence_branches_and_nonfinite_rules():
    compiled, _ = _status_fixture(with_certificates=True)
    markers = [0, 1, 2, 3, 3, 0, 1, 3, 3, 3, np.nan]
    residuals = [0, 0, 0, 0, 0, np.inf, np.inf, -1, -np.inf, np.nan, 0]
    data = {"marker": jnp.asarray(markers, jnp.float32)[:, None],
            "residual": jnp.asarray(residuals, jnp.float32)[:, None]}
    # Here residuals are also data. Use a finite replacement datum when the
    # prescribed residual is nonfinite, so +inf's scalar semantics are isolated.
    original_solver = compiled._solver
    compiled._solver = lambda d, x, y: (*original_solver(d, x, y)[:3],
                                       jnp.asarray(residuals, jnp.float32))
    data["residual"] = jnp.zeros_like(data["residual"])
    xs = jnp.zeros((len(markers), 1)).at[4, 0].set(jnp.nan)
    ys = jnp.zeros_like(xs)
    results = compiled.solve_many_results(data, warm_start=(xs, ys))
    expected = ["solved", "primal_infeasible", "unbounded", "primal_infeasible",
                "numerical_error", "inaccurate", "primal_infeasible",
                "numerical_error", "numerical_error", "numerical_error",
                "numerical_error"]
    assert [r.status for r in results] == expected
    resolved = jax.vmap(compiled._resolve)(data)
    for i, result in enumerate(results):
        status, ray = compiled._status(
            xs[i], ys[i], tuple(a[i] for a in resolved), i, residuals[i],
        )
        assert result.status == status
        if status in ("primal_infeasible", "unbounded"):
            np.testing.assert_array_equal(result.certificate, ray)
        else:
            assert result.certificate is None


def test_batched_lp_returns_checked_primal_and_dual_rays_in_the_same_batch():
    # x >= 0, x[0] = b; the second coordinate supplies a recession direction.
    variable = ocs.Variable(2)
    operator = ocs.LinearOperator.from_forward(lambda x: x[:1], n=2, m=1)
    compiled = ocs.minimize([
        ocs.nonneg(variable),
        ocs.equality(operator @ variable - ocs.Parameter("b")),
    ], cost=ocs.Parameter("c")).compile(
        equilibrate="scalar", max_blocks=0, tol=1e-3,
    )
    data = {"b": jnp.asarray([[0.], [-1.], [0.], [1.]]),
            "c": jnp.asarray([[0., 0.], [0., 0.], [0., -1.], [0., 0.]])}
    xs = jnp.asarray([[0., 0.], [0., 0.], [0., 2.], [0., 0.]])
    ys = jnp.asarray([[0.], [1.], [0.], [0.]])
    results = compiled.solve_many_results(data, warm_start=(xs, ys))
    assert [r.status for r in results] == [
        "solved", "primal_infeasible", "unbounded", "inaccurate",
    ]
    dual_ray = np.asarray(results[1].certificate)
    assert dual_ray[0] > 0 and float(-np.asarray(data["b"][1]) @ dual_ray) > 0
    primal_ray = np.asarray(results[2].certificate)
    assert np.all(primal_ray >= 0) and primal_ray[0] == 0
    assert float(np.asarray(data["c"][2]) @ primal_ray) < 0
    for i, result in enumerate(results):
        single = compiled.solve_instance(
            {k: v[i] for k, v in data.items()}, warm_start=(xs[i], ys[i]),
        )
        assert result.status == single.status
        if result.certificate is not None:
            np.testing.assert_allclose(result.certificate, single.certificate, rtol=1e-6)


def test_raw_batch_arrays_remain_usable_under_jit():
    compiled = _equality_family()
    data = {"b": jnp.asarray([[0.], [1.]])}
    raw = jax.jit(lambda d: compiled.solve_many(d, accuracy="residual"))(data)
    assert len(raw) == 4
    assert all(isinstance(value, jax.Array) for value in raw)
    np.testing.assert_array_equal(raw[2], [0, 0])


def test_objective_gap_batch_names_host_controller_when_traced():
    import pytest
    compiled = _equality_family()
    with pytest.raises(ValueError, match="host-side assessment"):
        jax.jit(compiled.solve_many)({"b": jnp.asarray([[0.], [1.]])})


# ---------------------------------------------------------------------------
# The criterion that now awards `status`.  These are the tests the switch
# needs; the ones above cover the residual path it replaced as the default.
# ---------------------------------------------------------------------------

def _lasso_family(n=256, lam=0.05, tol=1e-4, max_blocks=400):
    op = ocs.DCTOperator(n)
    x = ocs.Variable(n)
    b = ocs.Parameter("b")
    return op, ocs.minimize([
        ocs.least_squares(op @ x - b),
        ocs.l1(ocs.Identity(n) @ x, lam),
    ]).compile(block=100, max_blocks=max_blocks, tol=tol, equilibrate="sgd")


def _lasso_data(op, n=256, seed=0):
    rng = np.random.default_rng(seed)
    xt = np.zeros(n, np.float32)
    idx = rng.choice(n, n // 20, replace=False)
    xt[idx] = rng.normal(size=idx.size).astype(np.float32)
    return {"b": jnp.asarray(np.asarray(op.matvec(jnp.asarray(xt))))}


def test_default_criterion_is_the_certified_objective_gap():
    op, compiled = _lasso_family()
    sol = compiled.solve_instance(_lasso_data(op))
    assert sol.accuracy_criterion == "objective_gap"
    assert sol.status == "solved"
    # `solved` means THIS, and nothing else sets it.
    assert sol.objective_gap is not None and sol.objective_gap <= 1e-4
    assert sol.objective_lower_bound <= sol.objective_primal_value
    # The residual is still reported and still makes no accuracy claim.
    assert isinstance(sol.residual, float)


def test_residual_criterion_is_reachable_and_labelled_as_itself():
    op, compiled = _lasso_family()
    sol = compiled.solve_instance(_lasso_data(op), accuracy="residual")
    assert sol.accuracy_criterion == "fixed_point"
    assert sol.objective_gap is None
    assert sol.residual <= 1e-4 or sol.status != "solved"


def test_a_point_accepted_on_the_residual_can_miss_the_gap_it_claims():
    """The switch is not cosmetic: measured on this family the legacy point's
    certified gap exceeded the tolerance it reported as met."""
    op, compiled = _lasso_family()
    data = _lasso_data(op)
    legacy = compiled.solve_instance(data, accuracy="residual")
    report = compiled.objective_bound(legacy.x, legacy.y, data)
    if legacy.status == "solved" and report.get("relative_gap_certified", False):
        gap = float(report["relative_gap"])
        # Whatever the exact numbers on this machine, the residual being
        # inside tol must NOT be taken to mean the objective gap is.
        assert legacy.residual <= 1e-4
        assert gap > 0.0


def test_uncertifiable_family_is_unverified_and_never_solved():
    from ocs.ir import Term
    class UndeclaredPinball(ocs.Pinball):
        bound_capable = False
    n = 256
    op = ocs.DCTOperator(n)
    x = ocs.Variable(n)
    b = ocs.Parameter("b")
    compiled = ocs.minimize([
        ocs.least_squares(op @ x - b),
        Term(ocs.Identity(n), UndeclaredPinball(0.5)),
    ]).compile(block=100, max_blocks=50, tol=1e-4, equilibrate="sgd")
    assert not compiled.meta["objective_bound_capable"]
    sol = compiled.solve_instance(_lasso_data(op))
    assert sol.status == "unverified"
    assert sol.accuracy_criterion == "objective_gap"
    assert sol.termination == "unsupported" and sol.iters == 0
    assert sol.objective_gap is None
    assert "UndeclaredPinball" in sol.accuracy_detail


def test_batch_and_scalar_agree_under_the_default_criterion():
    op, compiled = _lasso_family()
    one = _lasso_data(op, seed=0)["b"]
    two = _lasso_data(op, seed=1)["b"]
    rows = compiled.solve_many_results({"b": jnp.stack([one, two])})
    for i, b in enumerate((one, two)):
        single = compiled.solve_instance({"b": b})
        assert rows[i].status == single.status
        assert rows[i].accuracy_criterion == single.accuracy_criterion


def test_cg_certifies_only_when_strongly_convex():
    """rho > 0 makes the CGLS objective strongly convex, which bounds the
    optimum from the gradient alone.  rho = 0 does not, and is refused."""
    d = jnp.asarray([1.0, 2.0, 3.0])
    op = ocs.LinearOperator.from_forward(lambda v: d * v, n=3, m=3)
    x = ocs.Variable(3)
    data = {"b": jnp.asarray([2.0, 6.0, 1.0], jnp.float32)}

    strong = ocs.minimize([ocs.least_squares(op @ x - ocs.Parameter("b"))]
                          ).compile(method="cg", cg_iters=40, tol=1e-4, rho=1.0)
    s = strong.solve_instance(data)
    assert s.accuracy_criterion == "objective_gap"
    assert s.status == "solved"
    assert s.objective_lower_bound <= s.objective_primal_value

    plain = ocs.minimize([ocs.least_squares(op @ x - ocs.Parameter("b"))]
                         ).compile(method="cg", cg_iters=40, tol=1e-4)
    p = plain.solve_instance(data)
    assert p.status == "unverified"
    assert "not strongly convex" in p.accuracy_detail


def test_tolerance_below_float32_resolution_is_refused_not_granted():
    """A gap finer than the arithmetic that produced it is not certified."""
    d = jnp.asarray([1.0, 2.0, 3.0])
    op = ocs.LinearOperator.from_forward(lambda v: d * v, n=3, m=3)
    x = ocs.Variable(3)
    compiled = ocs.minimize([ocs.least_squares(op @ x - ocs.Parameter("b"))]
                            ).compile(method="cg", cg_iters=60, tol=1e-14, rho=1.0)
    sol = compiled.solve_instance({"b": jnp.asarray([2.0, 6.0, 1.0], jnp.float32)})
    assert sol.status == "unverified"
    assert "float32 can resolve" in sol.accuracy_detail


def test_unverified_maps_to_a_cvxpy_status_that_claims_nothing():
    from ocs.cvxpy_harness.backend import _cvxpy_status
    import cvxpy.settings as settings
    assert _cvxpy_status("unverified") == settings.USER_LIMIT
    # A declared-budget contract cannot upgrade a run that certified nothing.
    assert _cvxpy_status("unverified", contract_certificate=True,
                         termination="criterion") == settings.USER_LIMIT
    # It still may upgrade an ordinary strict miss, as documented.
    assert _cvxpy_status("inaccurate", contract_certificate=True,
                         termination="criterion") == settings.OPTIMAL_INACCURATE
