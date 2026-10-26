"""Map functions as well as operators when changing non-conic block units."""
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.model.expression import _term
from ocs.prox.library import L1, SumSquares


def report(kind, argument_units, objective_units):
    a, gamma = argument_units, objective_units
    x = ocs.Variable(2)
    diagonal = jnp.asarray([1., 2.])
    operator = ocs.LinearOperator.from_forward(
        lambda v: a * diagonal * v, n=2, m=2)
    block = (L1(.7 * gamma / a) if kind == "l1" else
             SumSquares().scaled(gamma / a**2))
    family = ocs.minimize(
        [_term(block, operator @ x - a * diagonal, 2)],
        cost=gamma * jnp.asarray([.1, .3]))
    compiled = family.compile(max_blocks=0, equilibrate=(
        "given", jnp.full(2, 1e-14), jnp.full(2, .01)))
    point = jnp.asarray([.5, 1.5])
    dual = (gamma / a) * jnp.asarray([-.25, .25])
    return compiled.certificate(point, dual)


@pytest.mark.parametrize("kind", ["l1", "quadratic"])
@pytest.mark.parametrize("argument_units,objective_units", [
    (1e-6, 1.), (1e6, 1.), (1., 1e-6), (1., 1e6),
    (1e-6, 1e6), (1e6, 1e-6),
])
def test_function_operator_and_dual_transform_preserves_certificate(
        kind, argument_units, objective_units):
    reference = report(kind, 1., 1.)
    transformed = report(kind, argument_units, objective_units)
    for name in ("r_primal", "r_dual"):
        assert np.isfinite(reference[name])
        assert reference[name] > 1e-3
        assert transformed[name] == pytest.approx(reference[name], rel=3e-6, abs=2e-7)
