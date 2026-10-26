"""Contract and independent optimum checks for optional standard PDHG."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.solver.composite_pdhg import make_solver

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the standard and Halpern variants agreeing.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def family(kind):
    x=ocs.Variable(3)
    op=ocs.LinearOperator.from_forward(lambda z:jnp.array([1.,2.,3.])*z,n=3,m=3)
    terms=[ocs.nonneg(x) if kind=='nonneg' else ocs.l1(x,.2),
           ocs.least_squares(op@x-ocs.Parameter('b'))]
    return ocs.minimize(terms)


@pytest.mark.parametrize('kind',['nonneg','l1'])
@pytest.mark.parametrize('block',[1,7,100])
def test_closed_form_optimum_and_actual_certificate(kind,block):
    a=np.array([1.,2.,3.]);b=np.array([1.,-1.,3.],dtype=np.float32)
    expected=np.maximum(b/a,0) if kind=='nonneg' else np.sign(b/a)*np.maximum(np.abs(b/a)-.2/(a*a),0)
    compiled=family(kind).compile(equilibrate='scalar',pdhg_variant='standard',tol=1e-5,block=block,max_blocks=1000)
    sol=compiled.solve_instance({'b':b})
    assert sol.status=='solved'
    np.testing.assert_allclose(sol.x,expected,rtol=5e-5,atol=3e-6)
    cert=compiled.certificate(sol.x,sol.y,{'b':b})
    assert max(cert['r_primal'],cert['r_dual'])==sol.residual
    objective=lambda z:.5*np.sum((a*z-b)**2)+(0 if kind=='nonneg' else .2*np.abs(z).sum())
    assert abs(objective(np.asarray(sol.x))-objective(expected))<1e-6


def test_reporting_boundaries_preserve_extrapolation():
    # Identical fixed-step trajectory regardless of certificate check interval.
    x=jnp.array([[.7,-.2]]);y=jnp.array([[.4,.1]])
    outputs=[]
    for block in [1,3,9]:
        solver=make_solver(lambda z:2*z,lambda z:2*z,.2,.2,
            lambda z,t,d:jnp.maximum(z,0),lambda z,t,d:(z-t*d)/(1+t),
            lambda x,y,d:jnp.asarray(1.),2,2,block=block,max_blocks=9//block,
            tol=.1,adaptive_omega=False,variant='standard')
        outputs.append(solver(jnp.array([[1.,3.]]),x,y)[:2])
    for result in outputs[1:]:
        for actual,expected in zip(result,outputs[0]): np.testing.assert_array_equal(actual,expected)


def test_batch_stops_per_instance_and_rejects_nonfinite_data():
    compiled=family('nonneg').compile(equilibrate='scalar',pdhg_variant='standard',tol=1e-5,max_blocks=200)
    batch=jnp.array([[0.,0.,0.],[1.,-1.,3.],[jnp.nan,1.,2.]])
    solutions=compiled.solve_many_results({'b':batch})
    assert [s.status for s in solutions]==['solved','solved','numerical_error']
    assert solutions[0].iters==0
    assert solutions[1].iters>0
    np.testing.assert_allclose(solutions[1].x,[1.,0.,1.],atol=1e-4)


@pytest.mark.parametrize('kw',[{'pdhg_variant':'bad'}, {'pdhg_variant':'standard','method':'cg'},
    {'pdhg_variant':'standard','online_warmup':2}, {'pdhg_variant':'standard','adaptive_omega':True}])
def test_unsupported_combinations_rejected(kw):
    with pytest.raises(ValueError): family('nonneg').compile(**kw)


def test_valid_warm_start_is_owned_and_returned_without_iteration():
    compiled=family('nonneg').compile(equilibrate='scalar',pdhg_variant='standard',tol=1e-5,max_blocks=200)
    data={'b':jnp.array([1.,-1.,3.])}
    sol=compiled.solve_instance(data)
    saved=np.asarray(sol.x).copy(),np.asarray(sol.y).copy()
    again=compiled.solve_instance(data,warm_start=(sol.x,sol.y))
    assert again.status=='solved' and again.iters==0
    np.testing.assert_array_equal(sol.x,saved[0]);np.testing.assert_array_equal(sol.y,saved[1])
    np.testing.assert_array_equal(again.x,saved[0]);np.testing.assert_array_equal(again.y,saved[1])
