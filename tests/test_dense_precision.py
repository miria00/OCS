"""Native Dense products retain float32 accuracy when batching becomes GEMM.

GPU defaults may use reduced multiplication precision despite float32 arrays.
Independent float64 references and a tight full solve exercise its numerical
consequences. Lowered-oracle assertions also detect a missing precision request
on CPU backends that do not otherwise reproduce the GPU accuracy loss.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.operators.common import Dense
from ocs.operators.structured import (
    BinPack, Kronecker, LeftMatMul, MultiCommodityFlow, Sylvester,
)

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the residual reached in float32 on a dense operator.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


ORACLES = ("matvec", "rmatvec", "sq_matvec", "sq_rmatvec")


@pytest.mark.parametrize("shape", [(96, 64), (64, 96)], ids=["tall", "wide"])
@pytest.mark.parametrize("name", ORACLES)
@pytest.mark.parametrize("precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_batched_dense_oracles_match_float64_references(shape, name, precision):
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=shape).astype(np.float32)
    op = Dense(matrix)
    reference_matrix = matrix.astype(np.float64)
    if name.startswith("sq_"):
        reference_matrix = reference_matrix**2
    if name.endswith("rmatvec"):
        reference_matrix = reference_matrix.T
    batch = rng.normal(size=(16, reference_matrix.shape[1])).astype(np.float32)
    batch *= np.geomspace(1e-4, 1e4, len(batch)).astype(np.float32)[:, None]
    expected = batch.astype(np.float64) @ reference_matrix.T
    previous_setting = jax.config.jax_default_matmul_precision
    with jax.default_matmul_precision(precision):
        actual = jax.jit(jax.vmap(getattr(op, name)))(jnp.asarray(batch))
        single_oracle = jax.jit(getattr(op, name))
        single_results = np.stack([np.asarray(single_oracle(jnp.asarray(row)))
                                   for row in batch])
        assert jax.config.jax_default_matmul_precision == precision
    assert jax.config.jax_default_matmul_precision == previous_setting
    assert actual.dtype == jnp.float32 and op.A.dtype == jnp.float32
    errors = np.linalg.norm(np.asarray(actual, np.float64)-expected, axis=1)
    errors /= np.linalg.norm(expected, axis=1)
    assert np.all(errors < 2e-6), (name, shape, precision, errors)
    batch_single_errors = np.linalg.norm(
        np.asarray(actual, np.float64)-single_results.astype(np.float64), axis=1)
    batch_single_errors /= np.linalg.norm(expected, axis=1)
    assert np.all(batch_single_errors < 2e-6), (name, batch_single_errors)


def _dot_parameters(value):
    """Traverse both older pjit and newer jit nested JAXPR representations."""
    # Newer Jaxpr objects expose .jaxpr as a self-alias. Inspect their real
    # equations before unwrapping ClosedJaxpr/jit containers.
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            if equation.primitive.name == "dot_general":
                yield equation.params
            yield from _dot_parameters(equation.params)
    elif hasattr(value, "jaxpr"):
        yield from _dot_parameters(value.jaxpr)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _dot_parameters(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _dot_parameters(item)


@pytest.mark.parametrize("name", ORACLES)
def test_dense_precision_request_survives_vmap_in_a_reduced_global_context(name):
    op = Dense(np.arange(24, dtype=np.float32).reshape(6, 4)/11)
    dimension = 6 if name.endswith("rmatvec") else 4
    with jax.default_matmul_precision("bfloat16"):
        lowered = jax.make_jaxpr(jax.vmap(getattr(op, name)))(
            jnp.ones((8, dimension), jnp.float32))
    products = list(_dot_parameters(lowered))
    assert products, "the oracle must contain a matrix product"
    assert all(p["precision"] == (jax.lax.Precision.HIGHEST,)*2 for p in products)


@pytest.mark.parametrize("precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_batched_dense_nnls_reaches_requested_tight_tolerance(precision):
    # Same distribution as the profiled 128-instance NNLS failure, reduced
    # dimensions for a routine regression. Noise produces active constraints.
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    n, m, count = 64, 128, 8
    matrix = (rng.normal(size=(m, n))/np.sqrt(m)).astype(np.float32)
    truth = np.maximum(rng.normal(size=(count, n)).astype(np.float32), 0.)
    measured = truth @ matrix.T + .01*rng.normal(size=(count, m)).astype(np.float32)
    matrix64, measured64 = matrix.astype(np.float64), measured.astype(np.float64)
    references = np.stack([optimize.nnls(matrix64, row)[0] for row in measured64])
    assert np.any(references == 0) and np.any(references > 0)
    tolerance = 1e-5
    with jax.default_matmul_precision(precision):
        variable = ocs.Variable(n)
        compiled = ocs.minimize([
            ocs.nonneg(variable),
            ocs.least_squares(Dense(matrix) @ variable - ocs.Parameter("b")),
        ]).compile(tol=tolerance, max_blocks=40, block=100)
        results = compiled.solve_many_results({"b": jnp.asarray(measured)})
        for i, result in enumerate(results):
            assert result.status == "solved", (i, result.status, result.residual)
            assert result.residual <= tolerance
            report = compiled.certificate(result.x, result.y, {"b": measured[i]})
            assert max(report["r_primal"], report["r_dual"]) <= tolerance
            point = np.asarray(result.x, np.float64)
            assert np.min(point) >= 0
            assert np.linalg.norm(point-references[i])/np.linalg.norm(references[i]) < 2e-4
            residual = matrix64 @ point-measured64[i]
            reference_residual = matrix64 @ references[i]-measured64[i]
            objective = .5*float(residual @ residual)
            reference_objective = .5*float(reference_residual @ reference_residual)
            assert reference_objective > 0
            assert abs(objective-reference_objective)/reference_objective < 1e-3
            # Check original-unit stationarity independently of OCS's oracle.
            gradient = matrix64.T @ residual
            projected = np.maximum(point-gradient, 0.)
            scale = np.linalg.norm(matrix64.T @ measured64[i])
            assert np.linalg.norm(point-projected)/scale <= tolerance


def _structured_case(kind):
    rng = np.random.default_rng(81)
    if kind == "kronecker":
        a = rng.normal(size=(48, 32)).astype(np.float32)
        b = rng.normal(size=(40, 24)).astype(np.float32)
        return Kronecker(a, b), np.kron(a.astype(np.float64), b.astype(np.float64))
    if kind == "sylvester":
        a = rng.normal(size=(32, 32)).astype(np.float32)
        b = rng.normal(size=(24, 24)).astype(np.float32)
        expanded = np.kron(a.astype(np.float64), np.eye(24))
        expanded += np.kron(np.eye(32), b.astype(np.float64).T)
        return Sylvester(a, b), expanded
    if kind == "bin_pack":
        resources = rng.uniform(.1, 2., size=(20, 32)).astype(np.float32)
        expanded = np.vstack([np.kron(np.eye(32), np.ones((1, 24))),
                              np.kron(resources.astype(np.float64), np.eye(24))])
        return BinPack(resources, 24), expanded
    if kind == "left_matmul":
        a = rng.normal(size=(48, 32)).astype(np.float32)
        return LeftMatMul(a, 24), np.kron(a.astype(np.float64), np.eye(24))
    # A ring plus additional arcs gives an actual signed incidence matrix.
    incidence = np.zeros((24, 48), dtype=np.float32)
    for edge in range(48):
        tail = edge % 24
        head = (tail + (1 if edge < 24 else 7)) % 24
        incidence[tail, edge], incidence[head, edge] = -1, 1
    return MultiCommodityFlow(incidence, 16), np.kron(incidence.astype(np.float64), np.eye(16))


@pytest.mark.parametrize("kind", ["kronecker", "sylvester", "bin_pack", "left_matmul", "flow"])
@pytest.mark.parametrize("adjoint", [False, True], ids=["forward", "adjoint"])
@pytest.mark.parametrize("precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_structured_batched_products_match_independent_expanded_maps(kind, adjoint, precision):
    op, reference = _structured_case(kind)
    oracle = op.rmatvec if adjoint else op.matvec
    if adjoint:
        reference = reference.T
    rng = np.random.default_rng(82)
    batch = rng.normal(size=(4, reference.shape[1])).astype(np.float32)
    expected = batch.astype(np.float64) @ reference.T
    with jax.default_matmul_precision(precision):
        actual = jax.jit(jax.vmap(oracle))(jnp.asarray(batch))
        single = jax.jit(oracle)
        separate = np.stack([np.asarray(single(jnp.asarray(row))) for row in batch])
        products = list(_dot_parameters(jax.make_jaxpr(jax.vmap(oracle))(jnp.asarray(batch))))
    assert actual.dtype == jnp.float32
    assert products and all(p["precision"] == (jax.lax.Precision.HIGHEST,)*2 for p in products)
    error = np.linalg.norm(np.asarray(actual, np.float64)-expected, axis=1)
    assert np.all(error/np.linalg.norm(expected, axis=1) < 3e-6)
    discrepancy = np.linalg.norm(np.asarray(actual, np.float64)-separate, axis=1)
    assert np.all(discrepancy/np.linalg.norm(expected, axis=1) < 3e-6)


@pytest.mark.parametrize("kind", ["quadratic", "sum_squares", "affine", "singular_affine"])
@pytest.mark.parametrize("precision", [None, "bfloat16"], ids=["default", "reduced"])
def test_batched_quadratic_and_calculus_products_match_original_coordinate_references(kind, precision):
    rng = np.random.default_rng(93)
    dimension, count = 64, 8
    factor = rng.normal(size=(dimension, dimension))
    p = (factor.T @ factor/dimension + .5*np.eye(dimension)).astype(np.float32)
    q = rng.normal(size=dimension).astype(np.float32)
    factor = rng.normal(size=(dimension, dimension))
    metric = (factor.T @ factor/dimension + np.eye(dimension)).astype(np.float32)
    points = rng.normal(size=(count, dimension)).astype(np.float32)
    hessian, linear, constant = p.astype(np.float64), q.astype(np.float64), 0.
    block = ocs.Quadratic(p, q)
    if kind == "sum_squares":
        block = ocs.SumSquares()
        hessian, linear = np.eye(dimension), np.zeros(dimension)
    elif kind in ("affine", "singular_affine"):
        a = rng.uniform(.7, 1.3, size=dimension).astype(np.float32)
        a[::2] *= -1
        if kind == "singular_affine":
            a[0] = 0
        b = rng.normal(size=dimension).astype(np.float32)
        d = rng.normal(size=dimension).astype(np.float32)
        e = rng.uniform(.1, .5, size=dimension).astype(np.float32)
        weight = 1.3
        block = block.scaled(weight).precomposed(a=a, b=b).with_linear_quadratic(d=d, e=e)
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        constant = weight*(.5*bb @ hessian @ bb-linear @ bb)
        linear = weight*aa*(linear-hessian @ bb)+d.astype(np.float64)
        hessian = weight*aa[:, None]*hessian*aa[None, :]+np.diag(e.astype(np.float64))
    points64, metric64 = points.astype(np.float64), metric.astype(np.float64)
    expected_prox = np.linalg.solve(hessian+metric64, (points64 @ metric64.T-linear).T).T
    expected_gradient = points64 @ hessian.T+linear
    expected_value = .5*np.sum(points64*(points64 @ hessian.T), axis=1)+points64 @ linear+constant
    with jax.default_matmul_precision(precision):
        actual_prox = jax.jit(jax.vmap(lambda v: block.prox_matrix(v, jnp.asarray(metric))))(jnp.asarray(points))
        actual_gradient = jax.jit(jax.vmap(block.smooth_grad))(jnp.asarray(points))
        actual_value = jax.jit(jax.vmap(block.value))(jnp.asarray(points))
    assert actual_prox.dtype == actual_gradient.dtype == actual_value.dtype == jnp.float32
    for actual, expected in [(actual_prox, expected_prox), (actual_gradient, expected_gradient)]:
        error = np.linalg.norm(np.asarray(actual, np.float64)-expected, axis=1)
        assert np.all(error/np.linalg.norm(expected, axis=1) < 3e-6), (kind, precision, error)
    np.testing.assert_allclose(actual_value, expected_value, rtol=3e-6, atol=3e-6)
    prox64 = np.asarray(actual_prox, np.float64)
    stationarity = prox64 @ hessian.T+linear+(prox64-points64) @ metric64.T
    scale = np.linalg.norm(points64 @ metric64.T, axis=1)
    assert np.all(np.linalg.norm(stationarity, axis=1)/scale < 3e-6)


def test_jaxpr_walker_discovers_real_precision_attributes_in_all_representations():
    # Two actual dot primitives with different requested precision prevent a
    # vacuous traversal or a helper that manufactures the expected attribute.
    @jax.jit
    def products(a, b):
        dims = (((1,), (0,)), ((), ()))
        high = jax.lax.dot_general(a, b, dims, precision=jax.lax.Precision.HIGHEST)
        default = jax.lax.dot_general(a, b, dims, precision=jax.lax.Precision.DEFAULT)
        return high, default

    closed = jax.make_jaxpr(products)(jnp.ones((2, 3), jnp.float32),
                                      jnp.ones((3, 2), jnp.float32))

    class SelfAlias:
        # Exercise the newer self-alias shape even on older JAX versions,
        # using the actual traced equations, not synthetic dot metadata.
        eqns = closed.jaxpr.eqns

        @property
        def jaxpr(self):
            return self

    expected = [(jax.lax.Precision.HIGHEST,) * 2,
                (jax.lax.Precision.DEFAULT,) * 2]
    for representation in (closed, closed.jaxpr, SelfAlias()):
        found = list(_dot_parameters(representation))
        assert len(found) == 2
        assert [item["precision"] for item in found] == expected
