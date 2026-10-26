"""Public prepared OT: accuracy, ownership, refusal and finite stopping."""
import dataclasses
import warnings
import copy
import pickle
import numpy as np
import pytest
from ocs.transport import PreparedQuadraticOT, OTConvergenceError


@pytest.mark.parametrize("kwargs", [
    dict(n=1,regularization=1), dict(n=True,regularization=1),
    dict(n=50000,regularization=1), dict(n=4,regularization=0),
    dict(n=4,regularization=-1), dict(n=4,regularization=float('nan')),
    dict(n=4,regularization=float('inf')), dict(n=4,regularization=True),
    dict(n=4,regularization=1e-40), dict(n=4,regularization=1e40),
    dict(n=4,regularization=1,return_capacity=0),
    dict(n=4,regularization=1,return_capacity=4),
    dict(n=4,regularization=1,return_capacity=True),
    dict(n=4,regularization=1,max_iterations=0),
    dict(n=4,regularization=1,max_iterations=2**31),
])
def test_bad_configuration_before_gpu(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError): PreparedQuadraticOT(**kwargs)


@pytest.fixture
def gpu():
    import jax
    pytest.importorskip('cupy')
    if jax.devices()[0].platform != 'gpu': pytest.skip('NVIDIA GPU qualification')


def fixture(n=35):
    rng=np.random.RandomState(42)
    points=rng.normal(size=(2,n,2)).astype(np.float32)
    c=np.sum((points[0,:,None,:]-points[1,None,:,:])**2,axis=-1)
    c=np.ascontiguousarray(c/np.max(c),dtype=np.float32)
    p=np.full(n,1/n,np.float32)
    return c,p,p.copy()


def test_public_accuracy_ownership_and_close(gpu):
    from ocs.transport._assessment import assess
    c,p,q=fixture()
    originals=[z.copy() for z in (c,p,q)]
    session=PreparedQuadraticOT(35,regularization=.7,return_capacity=3)
    with pytest.raises(TypeError): copy.copy(session)
    with pytest.raises(TypeError): copy.deepcopy(session)
    with pytest.raises(TypeError): pickle.dumps(session)
    with pytest.raises(AttributeError): session.n=3
    with pytest.raises(AttributeError): session.regularization=9
    with pytest.raises(ValueError): session.solve(c.astype(np.float64),p,q)
    with pytest.raises(ValueError): session.solve(c.T,p,q)
    session._lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='busy'): session.solve(c,p,q)
        with pytest.raises(RuntimeError,match='busy'): session.close()
    finally: session._lock.release()
    first=session.solve(c,p,q)
    assert first.solved and first.statistics['output_slot']==0
    saved=[z.copy() for z in (first.plan,first.row_potential,first.column_potential)]
    for key in ('gate_pass','arithmetic_valid'):
        assert assess(c,p,q,*saved,np.float32(.7))[key]
    with pytest.raises(TypeError): first.report['gate_pass']=False
    with pytest.raises(dataclasses.FrozenInstanceError): first.plan=c
    with pytest.raises(ValueError): first.plan[0,0]=10
    for _ in range(2):
        other=session.solve(c,p,q)
        assert other.solved and not np.shares_memory(first.plan,other.plan)
        np.testing.assert_array_equal(first.plan,other.plan)
    with pytest.raises(RuntimeError,match='capacity'): session.solve(c,p,q)
    session.close(); session.close()
    with pytest.raises(RuntimeError,match='closed'): session.solve(c,p,q)
    for actual,old in zip((first.plan,first.row_potential,first.column_potential),saved):
        np.testing.assert_array_equal(actual,old)
    for actual,old in zip((c,p,q),originals): np.testing.assert_array_equal(actual,old)


@pytest.mark.parametrize('bad',['nan_cost','negative_cost','mass','negative_mass','zero_mass','infinite_mass'])
def test_value_refusal_consumes_output(gpu,bad):
    c,p,q=fixture(4)
    if bad=='nan_cost': c[0,0]=np.nan
    if bad=='negative_cost': c[0,0]=-1
    if bad=='mass': q[0]+=np.float32(.125)
    if bad=='negative_mass': p[0]=-1
    if bad=='zero_mass': p[:]=0; q[:]=0
    if bad=='infinite_mass': p[0]=np.inf
    with PreparedQuadraticOT(4,regularization=1,return_capacity=1) as session:
        with pytest.raises(ValueError): session.solve(c,p,q)
        with pytest.raises(RuntimeError,match='capacity'): session.solve(c,p,q)


def test_iteration_cap_never_awards_accuracy(gpu):
    c,p,q=fixture()
    with PreparedQuadraticOT(35,regularization=.7,return_capacity=1,max_iterations=1) as session:
        with pytest.raises(OTConvergenceError) as err: session.solve(c,p,q)
        assert err.value.iterations==1 and not err.value.report['gate_pass']
        with pytest.raises(RuntimeError,match='capacity'): session.solve(c,p,q)


def test_analytic_constant_cost(gpu):
    c=np.full((4,4),.25,np.float32);p=np.full(4,.25,np.float32)
    with PreparedQuadraticOT(4,regularization=1,return_capacity=1) as session:
        result=session.solve(c,p,p)
    assert result.solved
    np.testing.assert_allclose(result.plan,np.full((4,4),1/16),rtol=1e-3,atol=1e-5)
    assert abs(result.report['upper']-(.25+1/32))<1e-10


def test_opt_in_import_does_not_import_cupy():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',
        'import sys; import ocs.transport; assert "cupy" not in sys.modules'],check=True)
