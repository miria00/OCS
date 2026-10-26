"""A scaled block forwards the EXACT conjugate-domain contract.

h = c f has h^*(y) = c f^*(y / c), so dom h^* = c dom f^*.  Before this the
wrapper reported no contract at all, which made every family built with
``.scaled(mu)`` -- the ridge primal of the deblurring families among them --
``unverified`` under the default acceptance, with no bound and no gap.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.prox.library import SumSquares, GroupL2, NonNeg, L1


@pytest.mark.parametrize("base", [SumSquares(), GroupL2(0.7, 3), NonNeg(), L1(0.3)])
def test_scaled_contract_matches_base_on_y_over_c(base):
    c = 2.5
    h = base.scaled(c)
    assert h.bound_capable is True
    assert h.is_indicator == bool(getattr(base, "is_indicator", False))
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = jnp.asarray(rng.normal(size=6) * rng.choice([0.1, 1.0, 5.0]), jnp.float32)
        assert bool(h.conj_in_domain(y)) == bool(base.conj_in_domain(y / c))
        assert float(h.conj_domain_radius(y)) == float(base.conj_domain_radius(y / c))
        p = h.conj_domain_project(y)
        np.testing.assert_allclose(np.asarray(p), c * np.asarray(base.conj_domain_project(y / c)), rtol=1e-6, atol=1e-7)
        assert bool(h.conj_in_domain(p))


def test_unscaled_base_without_contract_stays_without_it():
    class Bare(ocs.ProxBlock if hasattr(ocs, "ProxBlock") else SumSquares):
        bound_capable = False
    assert Bare().scaled(2.0).bound_capable is False


def test_ridge_plus_group_l2_family_is_bound_capable_and_certifies():
    """The deblurring families' composition: ridge primal through .scaled, a
    coupled GroupL2 dual term with an offset.  Must certify under the default
    acceptance, not report 'unverified'."""
    rng = np.random.default_rng(3)
    n, g, k = 60, 20, 3
    A = rng.normal(size=(g * k, n)) / np.sqrt(n)
    b = rng.normal(size=g * k).astype(np.float32)
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(1e-2)),
                         ocs.group_l2(ocs.Dense(A) @ x - bp, lam=0.5, group_size=k)])
    comp = prob.compile(block=50, max_blocks=400, tol=1e-4)
    s = comp.solve_instance({"b": jnp.asarray(b)})
    rep = comp.validate(s.x, s.y, {"b": jnp.asarray(b)})
    assert rep["objective_bound_capable"] is True or rep["objective_bound_capable"] == 1
    assert s.accuracy_criterion == "objective_gap"
    assert s.status == "solved", (s.status, s.accuracy_detail)
