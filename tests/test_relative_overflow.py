"""Final regressions for invalid certificate arithmetic and calculus metadata."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.model.expression import _term
from ocs.prox.library import Equality, L1, SumSquares
from ocs.solver.stopping import _balanced_graph

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# relative scaling under extreme magnitudes.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def test_finite_input_with_overflowing_adjoint_norm_cannot_be_solved():
    # Huge, cancelling equality multipliers make each adjoint norm overflow,
    # while their sum and all input entries remain finite. Previously G=inf
    # produced a zero certificate step and falsely certified the primal L1 prox.
    xvar = ocs.Variable(4)
    positive = ocs.LinearOperator.from_forward(
        lambda x: jnp.sum(x)[None], n=4, m=1)
    negative = ocs.LinearOperator.from_forward(
        lambda x: -jnp.sum(x)[None], n=4, m=1)
    family = ocs.minimize([
        ocs.l1(xvar), ocs.equality(positive @ xvar),
        ocs.equality(negative @ xvar),
    ], cost=jnp.asarray([2., 0., 0., 0.]))
    compiled = family.compile(max_blocks=0, equilibrate=(
        "given", jnp.ones(4), jnp.ones(2)))
    x = jnp.asarray([1., -1., 0., 0.])
    y = jnp.asarray([2e38, 2e38])
    assert np.all(np.isfinite(np.asarray(x)))
    assert np.all(np.isfinite(np.asarray(y)))
    # This feasible point has objective 4; the feasible origin has objective 0.
    assert float(2*x[0] + jnp.sum(jnp.abs(x))) == 4.
    report = compiled.certificate(x, y)
    solution = compiled.solve_instance(warm_start=(x, y))
    assert np.isposinf(report["r_primal"])
    assert solution.status != "solved"
    assert np.isposinf(solution.residual)


def test_overflowing_objective_anchor_cannot_erase_complementarity():
    # All entries, graph scales, and even c'x are finite. Only ||c||*||x||
    # overflows. Its real-arithmetic complementarity ratio is ~.0078, so
    # replacing that denominator by infinity would change rejection to solved.
    variable = ocs.Variable(2)
    row = jnp.asarray([128., -1.])
    positive = ocs.LinearOperator.from_forward(
        lambda x: jnp.dot(row, x)[None], n=2, m=1)
    negative = ocs.LinearOperator.from_forward(
        lambda x: -jnp.dot(row, x)[None], n=2, m=1)
    cost = jnp.asarray([2.**66, 0.])
    family = ocs.minimize([
        ocs.nonneg(variable), ocs.equality(positive @ variable),
        ocs.equality(negative @ variable),
    ], cost=cost)
    compiled = family.compile(max_blocks=0, equilibrate=(
        "given", jnp.ones(2), jnp.ones(2)))
    x = jnp.asarray([2.**60, 2.**67])
    y = jnp.asarray([2.**80, 2.**80])
    assert np.isfinite(float(jnp.dot(cost, x)))
    cost64, x64 = np.asarray(cost, dtype=np.float64), np.asarray(x, dtype=np.float64)
    real_ratio = np.dot(cost64, x64) / (np.linalg.norm(cost64)*np.linalg.norm(x64))
    assert real_ratio > 1e-3
    report = compiled.certificate(x, y)
    solution = compiled.solve_instance(warm_start=(x, y))
    assert np.isposinf(report["r_complementarity"])
    assert solution.status != "solved"


@pytest.mark.parametrize("violation,accepted", [(5e-4, True), (2e-3, False)])
def test_routed_primal_indicator_zero_normal_uses_relative_feasibility(
        violation, accepted):
    variable = ocs.Variable(2)
    operator = ocs.LinearOperator.from_forward(lambda x: x[1:2], n=2, m=1)
    family = ocs.minimize([
        ocs.nonneg(variable), ocs.equality(operator @ variable-1.),
    ])
    compiled = family.compile(max_blocks=0, tol=1e-3, equilibrate=(
        "given", jnp.ones(2), jnp.ones(1)))
    point = jnp.asarray([-violation, 1.])
    dual = jnp.zeros(1)
    report = compiled.certificate(point, dual)
    solution = compiled.solve_instance(warm_start=(point, dual))
    expected = violation / np.sqrt(1. + violation**2)
    assert report["r_primal"] == pytest.approx(expected, rel=2e-6)
    assert (solution.status == "solved") == accepted
    assert np.isfinite(solution.residual)


@pytest.mark.parametrize("primal_scale,dual_scale", [
    (1., np.inf), (1., np.nan), (np.inf, 1.), (np.nan, 1.),
])
def test_nonfinite_graph_scales_cannot_pass(primal_scale, dual_scale):
    def check():
        return _balanced_graph(
            L1(), jnp.ones(1), jnp.zeros(1),
            jnp.asarray(primal_scale), jnp.asarray(dual_scale))[0]
    assert np.isposinf(float(jax.jit(check)()))


@pytest.mark.parametrize("kind", ["underflow", "overflow"])
def test_nonrepresentable_certificate_step_cannot_pass(kind):
    if kind == "underflow":
        cone, point, dual, scale = L1(), [1e-30], [-1e20], 1e-30
    else:
        cone, point, dual, scale = Equality(), [0.], [1e-20], 1e20
    def check():
        return _balanced_graph(cone, jnp.asarray(point), jnp.asarray(dual),
                               jnp.asarray(scale))[0]
    assert np.isposinf(float(jax.jit(check)()))


def _wrapped_report(block, argument_units=1., objective_units=1., *, near=False):
    a, gamma = argument_units, objective_units
    variable = ocs.Variable(2)
    diagonal = jnp.asarray([1., 2.])
    operator = ocs.LinearOperator.from_forward(
        lambda x: a * diagonal * x, n=2, m=2)
    family = ocs.minimize([_term(block, operator @ variable-a*diagonal, 2)],
                          cost=None if near else gamma*jnp.asarray([.1, .3]))
    compiled = family.compile(max_blocks=0, equilibrate=(
        "given", jnp.full(2, .1), jnp.full(2, .1)))
    point = jnp.ones(2) if near else jnp.asarray([.5, 1.5])
    y = jnp.full(2, 1e-4) if near else jnp.asarray([-.25, .25])
    return compiled.certificate(point, (gamma/a)*y)


def test_identity_precomposition_preserves_smooth_certificate():
    plain = _wrapped_report(SumSquares(), near=True)
    wrapped = _wrapped_report(SumSquares().precomposed(a=1.), near=True)
    for key in ("r_primal", "r_dual"):
        assert plain[key] < 1e-3
        assert wrapped[key] == pytest.approx(plain[key], rel=3e-6, abs=1e-8)


@pytest.mark.parametrize("kind", ["smooth", "nonsmooth"])
@pytest.mark.parametrize("argument_units,objective_units", [
    (1e-6, 1.), (1e6, 1.), (1., 1e-6), (1., 1e6),
    (1e-6, 1e6), (1e6, 1e-6),
])
def test_public_calculus_preserves_complete_function_units(
        kind, argument_units, objective_units):
    base = SumSquares() if kind == "smooth" else L1(.7)
    block = base.precomposed(a=1/argument_units).scaled(objective_units)
    reference = _wrapped_report(base)
    transformed = _wrapped_report(block, argument_units, objective_units)
    for key in ("r_primal", "r_dual"):
        assert transformed[key] == pytest.approx(reference[key], rel=5e-6, abs=2e-7)


def test_smooth_calculus_gradient_matches_independent_polynomial():
    # h(s)=3*.5*||a*s-b||^2+d's+.5*sum(e*s^2).
    a = jnp.asarray([2., -.5])
    b = jnp.asarray([1., -2.])
    d = jnp.asarray([.4, -.1])
    e = jnp.asarray([.3, .7])
    block = SumSquares().precomposed(a=a, b=b).scaled(3.).with_linear_quadratic(d=d, e=e)
    point = jnp.asarray([.6, -1.3])
    def independent_value(s):
        return 1.5*jnp.sum((a*s-b)**2) + jnp.dot(d, s) + .5*jnp.sum(e*s*s)
    np.testing.assert_allclose(block.smooth_grad(point), jax.grad(independent_value)(point),
                               rtol=2e-6, atol=2e-7)


def test_nonsmooth_calculus_does_not_invent_a_gradient():
    assert L1().precomposed(a=2., b=1.).smooth_grad is None
    assert L1().with_linear_quadratic(d=1., e=2.).smooth_grad is None
