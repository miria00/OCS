"""Analytic public-solve and invalid-option contracts for fixed primal weights."""
import numpy as np
import pytest
import jax.numpy as jnp
import ocs

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# an algorithm knob, measured by the residual it reaches.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


@pytest.mark.parametrize('variant',['halpern','standard'])
@pytest.mark.parametrize('kind',['nonnegative','primal_scalar','dual_scalar'])
@pytest.mark.parametrize('weight',[.1,1.,10.])
def test_weighted_fixed_metrics_solve_analytic_problems(kind,weight,variant):
    b = np.array([1.,2.,-1.],dtype=np.float32)
    x = ocs.Variable(3); rhs = ocs.Parameter('b')
    k = ocs.Dense(jnp.eye(3,dtype=jnp.float32)); lam=.3
    if kind=='nonnegative':
        terms=[ocs.nonneg(x),ocs.least_squares(k@x-rhs)]
        expected=np.maximum(b,0)
    elif kind=='primal_scalar':
        terms=[ocs.l2(x,lam),ocs.least_squares(k@x-rhs)]
        expected=(1-lam/np.linalg.norm(b))*b
    else:
        terms=[ocs.sum_squares(x),ocs.l2(k@x-rhs,lam)]
        expected=lam*b/np.linalg.norm(b)
    compiled=ocs.minimize(terms).compile(equilibrate='scalar',tol=1e-5,
        max_blocks=500,primal_weight=weight,pdhg_variant=variant)
    result=compiled.solve_instance({'b':jnp.asarray(b)})
    report=compiled.certificate(result.x,result.y,{'b':jnp.asarray(b)})
    assert result.status=='solved'
    assert max(report['r_primal'],report['r_dual'])<=1e-5
    assert np.linalg.norm(np.asarray(result.x)-expected)/np.linalg.norm(expected)<1e-4
    assert compiled.meta['primal_weight']==weight


@pytest.mark.parametrize('weight',[0.,-1.,np.nan,np.inf,1e100,1e-100])
def test_weight_rejects_invalid_or_unrepresentable_steps(weight):
    x=ocs.Variable(2)
    problem=ocs.minimize([ocs.nonneg(x),ocs.least_squares(ocs.Dense(jnp.eye(2))@x-ocs.Parameter('b'))])
    with pytest.raises(ValueError,match='primal_weight'):
        problem.compile(primal_weight=weight)


@pytest.mark.parametrize('options',[{'method':'cg'},{'online_warmup':2}])
def test_nondefault_weight_rejects_unsupported_engines(options):
    x=ocs.Variable(2)
    problem=ocs.minimize([ocs.nonneg(x),ocs.least_squares(ocs.Dense(jnp.eye(2))@x-ocs.Parameter('b'))])
    with pytest.raises(ValueError,match='without online equilibration'):
        problem.compile(primal_weight=2.,**options)
