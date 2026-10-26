"""Independent Fenchel references and safe fixed-prox dispatch for OT's block."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs import NonNeg, L1
from ocs.solver.fixed_prox import fixed_nonnegative_quadratic_prox


@pytest.mark.parametrize("vector", [False, True])
def test_positive_orthant_quadratic_fenchel_contract_and_fixed_step(vector):
    rng = np.random.default_rng(712)
    n = 97
    e = rng.uniform(.05, 3, n).astype('f') if vector else .25
    d = rng.normal(size=n).astype('f')
    f = NonNeg().with_linear_quadratic(d=d, e=e)
    y = rng.normal(size=n).astype('f')
    expected = .5*np.sum(np.maximum(y.astype(float)-d, 0)**2/np.asarray(e))
    assert f.bound_capable and bool(f.conj_in_domain(jnp.asarray(y)))
    np.testing.assert_allclose(jax.jit(f.conj_value)(y), expected, rtol=3e-6)
    assert np.isinf(f.conj_domain_radius(y))
    np.testing.assert_array_equal(f.conj_domain_project(y), y)
    tau = jnp.asarray(rng.uniform(.01, 2, n).astype('f'))
    fast = fixed_nonnegative_quadratic_prox(f, tau, n)
    assert fast is not None
    got = jax.jit(fast)(jnp.asarray(y))
    ordinary = jax.jit(lambda v: f.prox(v, tau))(jnp.asarray(y))
    np.testing.assert_array_equal(got, ordinary)
    # Check the independently derived minimizer, not only implementation parity.
    ref = np.maximum((y.astype(float)-np.asarray(tau)*d)/(1+np.asarray(tau)*e), 0)
    np.testing.assert_allclose(got, ref, rtol=4e-6, atol=2e-7)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_conjugate_argument_does_not_supply_a_bound(bad):
    f = NonNeg().with_linear_quadratic(d=0, e=.5)
    y = jnp.asarray([0., bad])
    assert not bool(jax.jit(f.conj_in_domain)(y))
    assert np.isinf(jax.jit(f.conj_value)(y))
    assert float(f.conj_domain_radius(y)) == 0


def test_no_unproved_contract_is_inherited_by_other_wrappers_or_subclasses():
    class ChangedOrthant(NonNeg):
        pass
    for base, e in [(NonNeg(), np.array([0., 1.])), (ChangedOrthant(), 1.)]:
        block = base.with_linear_quadratic(d=0, e=e)
        assert not block.bound_capable
        with pytest.raises(NotImplementedError):
            block.conj_in_domain(jnp.zeros(2))


def test_completed_builtin_linear_and_quadratic_contracts():
    for base,e in [(NonNeg(),0.),(L1(),1.)]:
        block=base.with_linear_quadratic(d=0,e=e)
        assert block.bound_capable
        assert bool(block.conj_in_domain(jnp.zeros(2)))
        assert float(block.conj_value(jnp.zeros(2)))==0.


def test_invalid_mutated_curvature_cannot_leave_a_valid_domain_flag():
    f = NonNeg().with_linear_quadratic(d=0, e=.5)
    f.e = jnp.asarray(0.)
    assert not bool(f.conj_in_domain(jnp.ones(2)))
    assert np.isinf(f.conj_value(jnp.ones(2)))
