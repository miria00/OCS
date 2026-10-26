"""The batched certificate evaluator on bound-capable families whose strict
objective bound uses the scoped float64 assessment (Dense / Diagonal guard).

Regression for the 0.3.3 successor failure: ``certificate_many`` raised
``lax.add requires arguments to have the same dtypes, got float64, float32``
on both layouts, because the batched evaluator vmapped the already jitted
scalar report and the pjit batching rule re-traced the scoped float64 block
outside its ``jax.enable_x64()`` scope (the float64 Dense products were
truncated to float32 before meeting the float64 sum).  The batched route now
traces the report BODY under vmap, inside the scope, like the scalar route.

The assertions below are about agreement and declared dtypes.  Nothing about
acceptance, normalization or the stopping rule is touched or asserted here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.ir import ProblemFamily, Term

pytestmark = pytest.mark.usefixtures("residual_criterion")

LAYOUTS = ("dense", "compact")


def dense_guarded(layout, **options):
    """Identity + two Dense terms with a data offset and a data cost."""
    defaults = dict(equilibrate="scalar", block=10, max_blocks=30, tol=1e-4,
                    data_layout=layout, pdhg_variant="standard")
    defaults.update(options)
    return ProblemFamily(2, [
        Term(ocs.Identity(2), ocs.NonNeg()),
        Term(ocs.Dense(np.diag([2., 3.]).astype(np.float32)), ocs.SumSquares(),
             lambda d: d["offset"]),
        Term(ocs.Dense(np.eye(2, dtype=np.float32)), ocs.L1(.1)),
    ], cost_fn=lambda d: d["cost"]).compile(**defaults)


def diagonal_guarded(layout):
    """Diagonal-only guard (no Dense product), zero data."""
    x = ocs.Variable(3)
    p = ocs.minimize([ocs.nonneg(x),
                      ocs.least_squares(ocs.Diagonal(jnp.asarray([1., 2., .5])) @ x
                                        - ocs.Parameter("b"))])
    return p.compile(equilibrate="scalar", max_blocks=0, data_layout=layout)


def rows(data, i):
    return {k: v[i] for k, v in data.items()}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("batch", [1, 5])
def test_batched_report_matches_scalar_rows_and_dtypes(layout, batch):
    comp = dense_guarded(layout)
    assert comp.meta["objective_bound_capable"]
    rng = np.random.default_rng(batch)
    data = {"offset": jnp.asarray(rng.normal(size=(batch, 2)).astype(np.float32)),
            "cost": jnp.asarray(rng.normal(size=(batch, 2)).astype(np.float32))}
    x = jnp.asarray(rng.normal(size=(batch, 2)).astype(np.float32))
    y = jnp.asarray(rng.normal(size=(batch, 4)).astype(np.float32))
    report = comp.certificate_many(x, y, data)
    scalar_dtypes = None
    for i in range(batch):
        scalar = comp.certificate(x[i], y[i], rows(data, i))
        for k in scalar:
            np.testing.assert_allclose(report[k][i], scalar[k], rtol=3e-6, atol=3e-7,
                                       equal_nan=True, err_msg=k)
    # Declared dtypes are fixed across routes: every batched field carries the
    # scalar evaluator's device dtype (float32 / bool / int32), never float64.
    for k, v in report.items():
        assert v.shape[0] == batch
        assert v.dtype in (jnp.float32, jnp.bool_, jnp.int32), (k, v.dtype)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_batched_report_handles_zero_and_nonfinite_rows(layout):
    comp = dense_guarded(layout)
    data = {"offset": jnp.asarray([[-1., -2.], [0., 0.], [jnp.nan, -1.]]),
            "cost": jnp.asarray([[.2, -.1], [0., 0.], [0., 0.]])}
    x = jnp.asarray([[.4, .6], [0., 0.], [0., 0.]])
    y = jnp.asarray([[-.1, 0., .1, .1], [0., 0., 0., 0.], [jnp.nan, -1., 0., 0.]])
    report = comp.certificate_many(x, y, data)
    for i in range(3):
        scalar = comp.certificate(x[i], y[i], rows(data, i))
        for k in scalar:
            np.testing.assert_allclose(report[k][i], scalar[k], rtol=3e-6, atol=3e-7,
                                       equal_nan=True, err_msg=k)
    # The zero row keeps its zero residuals; the nonfinite row never certifies.
    assert float(report["r_primal"][1]) == 0.0
    assert not bool(report["objective_lower_bound_available"][2])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_diagonal_guard_batched_matches_scalar(layout):
    comp = diagonal_guarded(layout)
    assert comp.meta["objective_bound_capable"]
    data = {"b": jnp.asarray([[1., -1., 2.], [0., 0., 0.]])}
    x = jnp.asarray([[0., 0., 0.], [1., .5, 4.]])
    y = jnp.asarray([[0., 0., 0.], [-1., 1., -2.]])
    report = comp.certificate_many(x, y, data)
    for i in range(2):
        scalar = comp.certificate(x[i], y[i], rows(data, i))
        for k in scalar:
            np.testing.assert_allclose(report[k][i], scalar[k], rtol=3e-6, atol=3e-7,
                                       equal_nan=True, err_msg=k)


def dense_qp(layout):
    """nonneg + least squares through a Dense operator: the Dense guard on the
    default objective-gap criterion (the form of ``primal_nonneg`` in
    test_returned_certifying_dual.py, with a Dense operator)."""
    x = ocs.Variable(3)
    k = ocs.Dense(np.array([[1., .2, 0.], [0., 2., .1], [.1, 0., 3.]], np.float32))
    # This file is marked ``residual_criterion`` (the other tests assert
    # residual-level agreement); this family asks for the objective-gap
    # criterion explicitly because the test is about certified lanes.
    return ocs.minimize([ocs.nonneg(x),
                         ocs.least_squares(k @ x - ocs.Parameter("b"))]).compile(
        equilibrate="scalar", block=10, max_blocks=400, tol=1e-3, data_layout=layout,
        accuracy="objective_gap")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_returned_batch_pairs_validate_with_recovery_off(layout):
    """Every lane returned by the batched objective-gap solve certifies with
    recovery OFF on its own returned (x, y), and the batched report agrees."""
    comp = dense_qp(layout)
    assert comp.meta["objective_bound_capable"]
    # Rows with active nonnegativity constraints (a positive optimal value);
    # an instance whose optimal value is exactly zero cannot meet a RELATIVE
    # gap and is not what this test is about.
    data = {"b": jnp.asarray([[-1., 2., 3.], [2., -1., .5], [-2., -1., 1.]])}
    results = comp.solve_many_results(data, accuracy="objective_gap")
    report = comp.certificate_many(jnp.stack([r.x for r in results]),
                                   jnp.stack([r.y for r in results]), data)
    for i, r in enumerate(results):
        assert r.status == "solved", (i, r.status, r.termination)
        assert r.y.dtype == jnp.float32 and r.x.dtype == jnp.float32
        bound = comp.objective_bound(r.x, r.y, rows(data, i), recover=False)
        assert bound["objective_lower_bound_available"], (i, bound)
        assert comp.acceptance(bound)["target_met"], (i, bound)
        assert bool(report["objective_lower_bound_available"][i])
        np.testing.assert_allclose(report["objective_lower_bound"][i],
                                   bound["objective_lower_bound"], rtol=3e-6, atol=3e-7)
