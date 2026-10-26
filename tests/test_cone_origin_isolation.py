"""Data-derived origin scales must not erase another cone or a nonzero RHS.

These use the public compilation, certificate and solve APIs at supplied
points with zero iteration budget. Operators expose only forward/adjoint maps.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.cones import SOCBatch

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# cone families, which declare no conjugate-domain contract.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


TOLERANCE = 1e-3
UNIT_MAPS = ((1.0, 1.0), (2.0**-12, 2.0**8), (2.0**12, 2.0**-8))


def _compile_quadratic_constraint(n, block, operator, cost, gamma, offset=None, quadratic=None):
    quadratic = ocs.SumSquares() if quadratic is None else quadratic
    family = ProblemFamily(
        n,
        [Term(ocs.Identity(n), quadratic.scaled(gamma)),
         Term(operator, block, offset)],
        cost_fn=lambda _: gamma * cost,
    )
    return family.compile(
        max_blocks=0, tol=TOLERANCE,
        equilibrate=("given", jnp.ones(n), jnp.ones(operator.shape[0])),
    )


@pytest.mark.parametrize("alpha,gamma", UNIT_MAPS)
def test_batched_soc_origin_cannot_borrow_an_unrelated_cones_dual_scale(alpha, gamma):
    # SOCBatch order is (t1,t2,z1,z2). The second independent cone has a
    # small operator scale and a large cost. A global Cauchy quotient for
    # y_ref inflates cone1's dual normalization by approximately 2**20:
    # ||K d||^2 / ||K.T K d||^2. Per-cone norms taken AFTER this global
    # quotient do not restore isolation.
    epsilon, magnitude = 2.0**-10, 2.0**30
    cost = jnp.asarray([-1.0, 0.0, magnitude, 0.0])

    def forward(x):
        return alpha * jnp.stack([x[0], epsilon*x[2], x[1], epsilon*x[3]])

    operator = ocs.LinearOperator.from_forward(forward, n=4, m=4)
    compiled = _compile_quadratic_constraint(4, SOCBatch(2, 1), operator, cost, gamma)
    point = jnp.zeros(4)
    dual = (gamma / alpha) * jnp.asarray([1.0, -magnitude/epsilon, 0.0, 0.0])

    # Exact stationarity and primal feasibility cannot rescue cone1's
    # incorrect normal. The true minimizer is (1,0,0,0), with objective
    # -gamma/2; the proposed origin has objective zero.
    stationarity = gamma * (point + cost) + operator.rmatvec(dual)
    np.testing.assert_array_equal(np.asarray(stationarity), np.zeros(4))
    optimum = np.asarray([1.0, 0.0, 0.0, 0.0])
    optimum_objective = gamma * (0.5*np.dot(optimum, optimum)
                                 + np.dot(np.asarray(cost, dtype=np.float64), optimum))
    assert optimum_objective == -0.5 * gamma
    assert float(dual[0]) > 0.0  # a positive scalar-axis SOC vector is not polar

    report = compiled.certificate(point, dual)
    solution = compiled.solve_instance(warm_start=(point, dual))
    assert report["r_domain"] == 0.0
    assert report["r_dual"] > TOLERANCE
    assert solution.iters == 0
    assert solution.status != "solved"
    assert solution.residual > TOLERANCE
    np.testing.assert_array_equal(np.asarray(solution.x), np.asarray(point))


@pytest.mark.parametrize("alpha,gamma", UNIT_MAPS)
@pytest.mark.parametrize("normal_error,accepted", [(2.0**-23, True), (2.0**-8, False)])
def test_diagonal_quadratic_dual_reference_survives_zero_forward_cost(
        alpha, gamma, normal_error, accepted):
    # H=diag(1,2), d=(-1,1), x_ref=-H^{-1}d=(1,-.5). The unconstrained
    # minimizer is strictly inside x0+x1>=0, with true cone multiplier zero.
    # Kd=0, so a dual reference based only on the cost ray vanishes. In
    # contrast, Kx_ref=.5 and the dual-curvature reference is 1/3 before
    # the alpha/gamma unit map. A tiny supplied normal error should be
    # accepted, while an error of ordinary magnitude must remain rejected.
    cost = jnp.asarray([-1.0, 1.0])
    operator = ocs.LinearOperator.from_forward(
        lambda x: alpha*jnp.sum(x)[None], n=2, m=1,
    )
    quadratic = ocs.SumSquares().with_linear_quadratic(e=jnp.asarray([0.0, 1.0]))
    compiled = _compile_quadratic_constraint(
        2, ocs.NonNeg(), operator, cost, gamma, quadratic=quadratic,
    )
    point = jnp.asarray([1.0, -0.5])
    dual = jnp.asarray([gamma*normal_error/alpha])
    np.testing.assert_array_equal(np.asarray(operator.matvec(cost)), np.zeros(1))
    assert float(operator.matvec(point)[0]) > 0.0

    report = compiled.certificate(point, dual)
    solution = compiled.solve_instance(warm_start=(point, dual))
    assert report["r_domain"] == 0.0
    assert (max(report["r_primal"], report["r_dual"]) <= TOLERANCE) == accepted
    assert solution.iters == 0
    assert (solution.status == "solved") == accepted
    assert (solution.residual <= TOLERANCE) == accepted
    np.testing.assert_array_equal(np.asarray(solution.x), np.asarray(point))


@pytest.mark.parametrize("alpha,gamma", UNIT_MAPS)
def test_nonzero_equality_rhs_is_not_normalized_by_large_quadratic_cost(alpha, gamma):
    # min gamma*(x^2/2 - M*x), subject to alpha*x=alpha. The only feasible
    # point is x=1. At x=0 the supplied multiplier cancels the large cost
    # exactly. If an origin reference x_ref=M enlarged this constraint's
    # primal scale, its complete violation would incorrectly look like 1/M.
    magnitude = 2.0**20
    operator = ocs.LinearOperator.from_forward(lambda x: alpha*x, n=1, m=1)
    compiled = _compile_quadratic_constraint(
        1, ocs.Equality(), operator, jnp.asarray([-magnitude]), gamma,
        offset=lambda _: jnp.asarray([-alpha]),
    )
    point = jnp.zeros(1)
    dual = jnp.asarray([gamma*magnitude/alpha])
    np.testing.assert_array_equal(
        np.asarray(gamma*(point-magnitude) + operator.rmatvec(dual)), np.zeros(1),
    )

    report = compiled.certificate(point, dual)
    solution = compiled.solve_instance(warm_start=(point, dual))
    assert report["r_domain"] == pytest.approx(1.0, rel=2e-6)
    assert report["r_primal"] >= 1.0 - 2e-6
    assert solution.iters == 0
    assert solution.status != "solved"
    assert solution.residual > TOLERANCE
    np.testing.assert_array_equal(np.asarray(solution.x), np.asarray(point))
