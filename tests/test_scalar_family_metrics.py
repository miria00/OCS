"""Uniform family metric storage and public numerical-contract controls."""
import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.prox.groups import _verified_scalar_groups,build_groups,grouped_moreau_dual

@pytest.mark.parametrize("coupled",[False,True])
@pytest.mark.parametrize("weight",[.25,1.,4.])
def test_compiler_retains_only_scalar_uniform_metrics(monkeypatch,coupled,weight):
    module=importlib.import_module("ocs.model.compile")
    saved=module.make_solver;captured=[]
    def capture(*args,**kwargs):
        captured.append(args[2:4]);return saved(*args,**kwargs)
    monkeypatch.setattr(module,"make_solver",capture)
    x=ocs.Variable(9)
    op=ocs.Dense(np.diag(np.linspace(.7,1.3,9)))
    terms=[ocs.l2(x,.1) if coupled else ocs.nonneg(x),ocs.least_squares(op@x-ocs.Parameter("b"))]
    comp=ocs.minimize(terms).compile(equilibrate="scalar",primal_weight=weight,pdhg_variant="standard",max_blocks=1)
    # The compiler may build both native scalar and batched engines. Check
    # every metric pair without fixing the private number of factories.
    assert captured
    for steps in captured:
        assert [step.shape for step in steps]==[(),()]
        assert sum(step.nbytes for step in steps)==8
        assert all(np.isfinite(step) and step>0 for step in steps)
    data={"b":np.linspace(-1,2,9,dtype=np.float32)}
    result=comp.solve_instance(data)
    assert np.isfinite(result.x).all() and np.isfinite(result.y).all()
    assert result.residual==max(comp.certificate(result.x,result.y,data)[key] for key in ("r_primal","r_dual"))

@pytest.mark.parametrize("value",[.25,0.,-1.,np.nan,np.inf])
def test_static_scalar_does_not_weaken_metric_validity(value):
    groups=build_groups([ocs.L2(.3)],[(0,4)])
    scalar=jnp.asarray(value,jnp.float32)
    expected=(bool(np.isfinite(value) and value>0),)
    assert _verified_scalar_groups(groups,4,scalar)==expected
    comp=grouped_moreau_dual(groups,4,static_sigma=scalar)
    ordinary=grouped_moreau_dual(groups,4)
    data=jnp.asarray([1.,-.5,.2,2.]);dense=jnp.full(4,value)
    np.testing.assert_array_equal(jax.jit(comp)(data,dense),jax.jit(ordinary)(data,dense))


def test_given_diagonal_metric_remains_dense_and_unchanged(monkeypatch):
    module=importlib.import_module("ocs.model.compile")
    saved=module.make_solver;captured=[]
    def capture(*args,**kwargs):
        captured.append(args[2:4]);return saved(*args,**kwargs)
    monkeypatch.setattr(module,"make_solver",capture)
    x=ocs.Variable(3);op=ocs.Dense(np.eye(3));tau=jnp.array([.1,.2,.3]);sigma=jnp.array([.4,.2,.1])
    ocs.minimize([ocs.nonneg(x),ocs.least_squares(op@x-ocs.Parameter("b"))]).compile(equilibrate=("given",tau,sigma))
    assert captured
    for actual_tau,actual_sigma in captured:
        assert actual_tau.shape==(3,) and actual_sigma.shape==(3,)
        np.testing.assert_array_equal(actual_tau,tau)
        np.testing.assert_array_equal(actual_sigma,sigma)
