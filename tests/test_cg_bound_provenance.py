"""CGLS explicitly identifies its dual-free primal suboptimality bound."""
import jax.numpy as jnp
import pytest

import ocs


def family(rho, iterations=40):
    x = ocs.Variable(3)
    op = ocs.Diagonal(jnp.array([1., 2., 4.], jnp.float32))
    return ocs.minimize([
        ocs.least_squares(op @ x - ocs.Parameter("b")),
    ]).compile(method="cg", rho=rho, cg_iters=iterations, tol=1e-3)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("rho,iterations,status,source", [
    (1., 40, "solved", "strong_convexity"),
    (1., 0, "inaccurate", "strong_convexity"),
    (0., 40, "unverified", "unavailable"),
    (0., 0, "unverified", "unavailable"),
])
def test_cg_objective_bound_provenance(batched, rho, iterations, status, source):
    compiled = family(rho, iterations)
    b = jnp.array([1., -2., 3.], jnp.float32)
    rows = (compiled.solve_many_results({"b": jnp.stack([b, 2 * b])})
            if batched else [compiled.solve_instance({"b": b})])
    for i, result in enumerate(rows):
        assert result.status == status
        assert result.accuracy_criterion == "objective_gap"
        assert result.y is None and result.y_iterate is None
        assert result.dual_source == "none"
        assert result.objective_bound_source == source
        report = compiled.objective_bound(result.x, data={"b": (i + 1) * b})
        assert report["source"] == source
        assert report["objective_lower_bound_available"] == (rho > 0)
        if rho > 0:
            assert result.objective_lower_bound == report["objective_lower_bound"]


@pytest.mark.parametrize("rho", [0., 1.])
@pytest.mark.parametrize("batched", [False, True])
def test_cg_residual_result_has_no_dual_or_evaluated_bound(rho, batched):
    compiled = family(rho)
    b = jnp.array([1., -2., 3.], jnp.float32)
    rows = (compiled.solve_many_results({"b": b[None]}, accuracy="residual")
            if batched else [compiled.solve_instance({"b": b}, accuracy="residual")])
    for result in rows:
        assert result.status == "solved"
        assert result.y is None and result.y_iterate is None
        assert result.dual_source == "none"
        assert result.objective_bound_source is None
        assert result.accuracy_criterion == "fixed_point"


@pytest.mark.parametrize("batched", [False, True])
def test_cg_numerical_failure_still_has_no_dual(batched):
    compiled = family(1.)
    b = jnp.array([jnp.nan, 1., 2.], jnp.float32)
    rows = (compiled.solve_many_results({"b": b[None]})
            if batched else [compiled.solve_instance({"b": b})])
    for result in rows:
        assert result.status == "numerical_error"
        assert result.y is None and result.y_iterate is None
        assert result.dual_source == "none"
        assert result.objective_bound_source is None
