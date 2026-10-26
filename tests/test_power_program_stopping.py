"""The original cubic-regression plateau through the complete conic solver."""
import jax.numpy as jnp
import numpy as np

import ocs
from ocs.model.expression import _term
from ocs.prox import PowerConeBatch
import pytest

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# power cone families, no conjugate-domain contract.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


class _PowerRegressionMap(ocs.LinearOperator):
    def __init__(self, matrix):
        self.matrix = jnp.asarray(matrix, dtype=jnp.float32)
        self.m, self.n = matrix.shape
        super().__init__((3*self.m, self.n+self.m))

    def matvec(self, point):
        return jnp.concatenate([point[self.n:], jnp.zeros(self.m, point.dtype),
                                self.matrix @ point[:self.n]])

    def rmatvec(self, dual):
        return jnp.concatenate([self.matrix.T @ dual[2*self.m:], dual[:self.m]])


def test_cubic_regression_certifies_and_returns_accurate_epigraph_variables():
    m, n = 200, 60
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(m, n)).astype("f")
    target = (matrix @ rng.normal(size=n).astype("f") + .1*rng.normal(size=m)).astype("f")
    offset = np.r_[np.zeros(m, "f"), np.ones(m, "f"), -target]
    cost = np.r_[np.zeros(n, "f"), np.ones(m, "f")]
    variable = ocs.Variable(m+n)
    term = _term(PowerConeBatch(m, 1/3), _PowerRegressionMap(matrix) @ variable + offset, m+n)
    tolerance = 1e-5
    compiled = ocs.minimize([term], cost=cost).compile(tol=tolerance, max_blocks=200)
    solution = compiled.solve_instance()
    report = compiled.certificate(solution.x, solution.y, {})
    assert solution.status == "solved"
    assert np.isfinite(solution.residual) and solution.residual <= tolerance
    assert max(report["r_primal"], report["r_dual"]) <= tolerance

    point = np.asarray(solution.x, dtype=np.float64)
    original_loss = np.sum(abs(matrix.astype("d") @ point[:n] - target.astype("d"))**3)
    modeled_objective = np.sum(point[n:])
    # Independent float64 SciPy analytic-gradient/Hessian reference, agreeing
    # with Clarabel; gradient norm2.1e-10. Preserve its raw optimizer status in
    # profiling/power_program_20260906/scipy_cubic_reference.json.
    reference = .1471180128816304
    assert abs(original_loss-reference)/(1+abs(reference)) < 1e-3
    # Computing only the reduced regression loss missed errors in the actual
    # epigraph variables. Both objective notions are required in this test.
    assert abs(modeled_objective-reference)/(1+abs(reference)) < 1e-3
