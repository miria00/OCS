"""Intrinsic SPD-quadratic cone references: accuracy, units, and isolation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.cones import SOCBatch
from ocs.solver.cone_scales import make_cone_reference_scales

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# cone families, which declare no conjugate-domain contract.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


UNITS = [(1., 1.), (2.**-12, 2.**8), (2.**12, 2.**-8)]
P = np.array([[4., 1.], [1., 2.]], dtype=np.float32)
COST = np.array([-5., -3.], dtype=np.float32)


def _family(alpha=1., gamma=1., max_blocks=0, block=None):
    if block is None:
        block = ocs.Quadratic(P)
    operator = ocs.LinearOperator.from_forward(lambda x: alpha*x, n=2, m=2)
    family = ProblemFamily(2, [Term(ocs.Identity(2), block.scaled(gamma)),
                               Term(operator, ocs.NonNeg())],
                           cost_fn=lambda data: gamma*jnp.asarray(COST))
    return family.compile(equilibrate="scalar", tol=1e-5, max_blocks=max_blocks)


@pytest.mark.parametrize("alpha,gamma", UNITS)
@pytest.mark.parametrize("normal_error,accepted", [(1e-8, True), (.1, False)])
def test_inactive_dense_quadratic_multiplier_has_intrinsic_scale(alpha, gamma, normal_error, accepted):
    compiled = _family(alpha, gamma)
    point = jnp.ones(2)
    dual = jnp.full(2, gamma*normal_error/alpha)
    report = compiled.certificate(point, dual)
    result = compiled.solve_instance(warm_start=(point, dual))
    assert compiled.meta["cone_data_scales"]
    assert report["r_domain"] == 0.
    assert (result.status == "solved") == accepted
    assert (max(report["r_primal"], report["r_dual"]) <= 1e-5) == accepted
    assert result.iters == 0


def test_dense_quadratic_nonnegative_program_solves_from_default_start():
    compiled = _family(max_blocks=20)
    result = compiled.solve_instance()
    assert result.status == "solved"
    assert result.residual <= 1e-5
    # Independent analytic optimum: -P^{-1}c=(1,1), strictly inside x>=0.
    expected = np.linalg.solve(P.astype(np.float64), -COST)
    np.testing.assert_allclose(result.x, expected, atol=2e-5, rtol=0)
    point = np.asarray(result.x, dtype=np.float64)
    np.testing.assert_allclose(.5*point@P@point + COST@point + 11., 7., atol=1e-8)


@pytest.mark.parametrize("alpha,gamma", UNITS)
def test_dense_reference_matches_independent_inverse_energy(alpha, gamma):
    # Includes a linear term inside g as well as the separate instance cost.
    q = np.array([.3, -.8], dtype=np.float32)
    K = np.array([[1., -.2], [.4, .7]], dtype=np.float32)
    operator = ocs.LinearOperator.from_forward(lambda x: alpha*jnp.asarray(K)@x, n=2, m=2)
    block = ocs.Quadratic(P, q).scaled(gamma)
    references = make_cone_reference_scales(block, [operator], [ocs.NonNeg()], [(0, 2)])
    cost = np.array([-.9, .2], dtype=np.float32)
    scales, objective = jax.jit(references)((jnp.zeros(2), gamma*jnp.asarray(cost)))
    H, direction = gamma*P.astype(np.float64), gamma*(q.astype(np.float64)+cost)
    expected_point = -np.linalg.solve(H, direction)
    w = alpha*K@expected_point
    adjoint = alpha*K.T@w
    energy = adjoint@np.linalg.solve(H, adjoint)
    expected_dual = w * (w@w)/energy
    np.testing.assert_allclose(scales[0][0], np.linalg.norm(w), rtol=2e-6)
    np.testing.assert_allclose(scales[0][1], np.linalg.norm(expected_dual), rtol=3e-6)
    np.testing.assert_allclose(objective, abs(direction@expected_point), rtol=3e-6)


def test_wrapped_dense_quadratic_reference_uses_transformed_hessian():
    a, b, q = np.array([-2., .5]), np.array([.2, -.4]), np.array([.3, -.1])
    d, e, c = np.array([.1, .2]), np.array([.4, 1.1]), 1.7
    block = ocs.Quadratic(P, q).precomposed(a=a, b=b).scaled(c).with_linear_quadratic(d=d, e=e)
    operator = ocs.LinearOperator.from_forward(lambda x: jnp.sum(x)[None], n=2, m=1)
    references = make_cone_reference_scales(block, [operator], [ocs.NonNeg()], [(0, 1)])
    cost = np.array([-.7, .2])
    H = c*(a[:, None]*P*a[None, :]) + np.diag(e)
    direction = cost + c*a*(q-P@b) + d
    point = -np.linalg.solve(H, direction)
    w = np.sum(point)
    energy = np.ones(2)@np.linalg.solve(H, np.ones(2)) * w*w
    expected_dual = w*w*w/energy
    scales, objective = references((jnp.zeros(1), jnp.asarray(cost, dtype=jnp.float32)))
    np.testing.assert_allclose(scales[0][0], abs(w), rtol=4e-6)
    np.testing.assert_allclose(scales[0][1], abs(expected_dual), rtol=4e-6)
    np.testing.assert_allclose(objective, abs(direction@point), rtol=4e-6)


def test_dense_factor_is_constructed_once_per_family(monkeypatch):
    import ocs.solver.cone_scales as module
    original = module.np.linalg.cholesky
    calls = []
    def tracked(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(module.np.linalg, "cholesky", tracked)
    operator = ocs.LinearOperator.from_forward(lambda x: x, n=2, m=2)
    references = make_cone_reference_scales(ocs.Quadratic(P), [operator], [ocs.NonNeg()], [(0, 2)])
    evaluated = jax.jit(references)
    evaluated((jnp.zeros(2), jnp.asarray(COST)))
    evaluated((jnp.zeros(2), 2*jnp.asarray(COST)))
    assert calls == [(2, 2)]


@pytest.mark.parametrize("base", [ocs.SumSquares().with_linear_quadratic(e=[0., 1.]),
                                  ocs.Quadratic(np.diag([1., 2.])),
                                  ocs.Zero().with_linear_quadratic(e=[1., 2.])])
def test_diagonal_path_does_not_construct_or_factor_dense_hessian(base, monkeypatch):
    import ocs.solver.cone_scales as module
    def forbidden(*args):
        raise AssertionError("diagonal reference must stay O(n)")
    monkeypatch.setattr(module, "_quadratic_coefficients", forbidden)
    monkeypatch.setattr(module.np.linalg, "cholesky", forbidden)
    operator = ocs.LinearOperator.from_forward(lambda x: jnp.sum(x)[None], n=2, m=1)
    references = make_cone_reference_scales(base, [operator], [ocs.NonNeg()], [(0, 1)])
    scales, _ = references((jnp.zeros(1), jnp.array([-1., 1.])))
    np.testing.assert_allclose(scales[0][0], .5, atol=1e-7)
    np.testing.assert_allclose(scales[0][1], 1/3, rtol=2e-6)


@pytest.mark.parametrize("base", [ocs.Quadratic(np.ones((2, 2))),
                                  ocs.Quadratic(P).precomposed(a=[0., 1.])])
def test_singular_dense_quadratic_gets_no_inverse_anchor(base):
    operator = ocs.LinearOperator.from_forward(lambda x: x, n=2, m=2)
    assert make_cone_reference_scales(base, [operator], [ocs.NonNeg()], [(0, 2)]) is None


@pytest.mark.parametrize("indicator", [ocs.Equality(), ocs.NonNeg()])
def test_nonzero_rhs_is_not_hidden_by_large_dense_quadratic_scale(indicator):
    operator = ocs.LinearOperator.from_forward(lambda x: x, n=2, m=2)
    magnitude = 2.**20
    family = ProblemFamily(2, [Term(ocs.Identity(2), ocs.Quadratic(P)),
                               Term(operator, indicator, lambda data: jnp.array([-1., 0.]))],
                           cost_fn=lambda data: jnp.array([-magnitude, 0.]))
    compiled = family.compile(equilibrate="scalar", max_blocks=0, tol=1e-3)
    point, dual = jnp.zeros(2), jnp.array([magnitude, 0.])
    report = compiled.certificate(point, dual)
    assert report["r_domain"] >= 1.-1e-6
    assert compiled.solve_instance(warm_start=(point, dual)).status != "solved"


def test_distinct_cones_do_not_share_a_dense_dual_quotient():
    H = np.eye(4, dtype=np.float32)
    H[2, 3] = H[3, 2] = .5
    operators = [ocs.LinearOperator.from_forward(lambda x: x[:1], n=4, m=1),
                 ocs.LinearOperator.from_forward(lambda x: (2.**-10)*x[2:3], n=4, m=1)]
    references = make_cone_reference_scales(ocs.Quadratic(H), operators,
                                           [ocs.NonNeg(), ocs.NonNeg()], [(0, 1), (1, 2)])
    for magnitude in [1., 2.**30]:
        scales, _ = references((jnp.zeros(2), jnp.array([-1., 0., magnitude, 0.])))
        np.testing.assert_allclose(scales[0][0], 1., atol=1e-7)
        np.testing.assert_allclose(scales[0][1], 1., atol=1e-7)


def test_batched_cones_remain_excluded_for_dense_objective():
    H = np.eye(4, dtype=np.float32)
    H[0, 1] = H[1, 0] = .2
    operator = ocs.LinearOperator.from_forward(lambda x: x, n=4, m=4)
    assert make_cone_reference_scales(ocs.Quadratic(H), [operator], [SOCBatch(2, 1)], [(0, 4)]) is None


def test_zero_cost_does_not_invent_an_absolute_scale():
    operator = ocs.LinearOperator.from_forward(lambda x: x, n=2, m=2)
    references = make_cone_reference_scales(ocs.Quadratic(P), [operator], [ocs.NonNeg()], [(0, 2)])
    scales, objective = references((jnp.zeros(2), jnp.zeros(2)))
    assert float(objective) == 0.
    assert all(np.all(np.asarray(value) == 0) for value in scales[0])
