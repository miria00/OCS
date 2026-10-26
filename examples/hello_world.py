"""Small installed-package example: operator input and compile-once batches.

This same self-contained code is in README.md; no repository data are needed.

WHAT ``solved`` MEANS HERE.  It means a strict lower bound on the optimal
objective value was certified at the returned point, and the gap between that
bound and the objective value there met ``tol``.  It is a statement about the
point you get back and the problem you asked about, so another solver can be
held to the same test.  ``residual`` is still reported, and is still the
relative prox-graph residual of this algorithm, but it no longer decides
anything: it is not an objective-error bound and not a duality gap.

Pass ``accuracy="residual"`` to ``solve_instance`` to accept on that residual
instead, which is what earlier versions did by default.
"""
import numpy as np
import jax.numpy as jnp
import ocs

TOL = 1e-3

weights = jnp.array([1.0, 2.0, 3.0])
K = ocs.LinearOperator.from_forward(lambda v: weights * v, n=3, m=3)
x = ocs.Variable(3)
b = ocs.Parameter("b")
family = ocs.minimize([
    ocs.nonneg(x),
    ocs.least_squares(K @ x - b),
])
compiled = family.compile(tol=TOL)
sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
print(sol.status, sol.accuracy_criterion, sol.objective_gap, sol.residual, sol.x)
assert sol.status == "solved"
# The criterion that awarded `solved`, and the quantity it tested.
assert sol.accuracy_criterion == "objective_gap"
assert sol.objective_gap is not None and sol.objective_gap <= TOL
assert sol.objective_lower_bound <= sol.objective_primal_value
np.testing.assert_allclose(sol.x, [0, 1, 1], atol=3e-3)

# The same criterion on a batch: no row is accepted on a test the single
# solve would have refused.
results = compiled.solve_many_results({
    "b": jnp.array([[-1.0, 2.0, 3.0], [2.0, -2.0, 6.0]])
})
for result, expected in zip(results, ([0, 1, 1], [2, 0, 2])):
    print(result.status, result.accuracy_criterion, result.objective_gap,
          result.iters, result.residual, result.x)
    assert result.status == "solved"
    assert result.accuracy_criterion == "objective_gap"
    assert result.objective_gap is not None and result.objective_gap <= TOL
    np.testing.assert_allclose(result.x, expected, atol=3e-3)

# The rejected criterion is still reachable, and says so about itself.
legacy = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])},
                                 accuracy="residual")
print(legacy.status, legacy.accuracy_criterion, legacy.residual)
assert legacy.accuracy_criterion == "fixed_point"
assert legacy.objective_gap is None
