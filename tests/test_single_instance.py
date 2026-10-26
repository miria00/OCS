"""Compare standard single-instance execution to the unchanged batch path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.ir import ProblemFamily, Term

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# single and batched paths returning the same point.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def build(layout, dual_layout='flat', loop_unroll=1, metric='scalar', **options):
    defaults = dict(equilibrate=('scalar' if metric=='scalar' else
                    ('given',jnp.asarray([.1,.2]),jnp.asarray([.2,.1,.3,.1]))),
                    block=7,max_blocks=50,tol=1e-4,pdhg_variant='standard',
                    data_layout=layout,dual_layout=dual_layout,loop_unroll=loop_unroll)
    defaults.update(options)
    return ProblemFamily(2,[Term(ocs.Identity(2),ocs.NonNeg()),
        Term(ocs.Dense(np.diag([2.,3.]).astype(np.float32)),ocs.SumSquares(),lambda d:d['offset']),
        Term(ocs.Dense(np.eye(2,dtype=np.float32)),ocs.L1(.1))],
        cost_fn=lambda d:d['cost']).compile(**defaults)


def same(a,b):
    assert a.status==b.status
    assert a.iters==b.iters
    assert a.x.shape==b.x.shape==(2,)
    assert a.y.shape==b.y.shape==(4,)
    for name in ('x','y','residual'):
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=5e-6,atol=5e-7,equal_nan=True)


@pytest.mark.parametrize('layout',['dense','compact'])
@pytest.mark.parametrize('dual_layout',['flat','blocks'])
@pytest.mark.parametrize('unroll',[1,4])
@pytest.mark.parametrize('metric',['scalar','nonuniform'])
def test_cold_warm_batch_and_caller_ownership(layout,dual_layout,unroll,metric):
    c=build(layout,dual_layout,unroll,metric)
    assert c._single_solver is not None and c._single_cold_solver is not None
    data={'offset':jnp.asarray([-1.,-2.]),'cost':jnp.asarray([.2,-.1])}
    batched=jax.tree_util.tree_map(lambda x:x[None],data)
    cold=c.solve_instance(data)
    same(cold,c.solve_many_results(batched)[0])
    raw=c.solve_many(batched)
    assert raw[0].shape==(1,2) and raw[1].shape==(1,4)
    np.testing.assert_allclose(raw[0][0],cold.x,rtol=5e-6,atol=5e-7)
    warm=(jnp.asarray([.2,.3]),jnp.asarray([.1,-.1,.03,.02]))
    saved=[np.asarray(v).copy() for v in (*warm,cold.x,cold.y)]
    result=c.solve_instance(data,warm_start=warm)
    expected=c.solve_many_results(batched,warm_start=(warm[0][None],warm[1][None]))[0]
    same(result,expected)
    for current,before in zip((*warm,cold.x,cold.y),saved):
        np.testing.assert_array_equal(np.asarray(current),before)
    certificate=c.certificate(result.x,result.y,data)
    np.testing.assert_allclose(max(certificate['r_primal'],certificate['r_dual']),result.residual,
                               rtol=5e-6,atol=5e-7)


@pytest.mark.parametrize('layout',['dense','compact'])
@pytest.mark.parametrize('kind',['data','warm'])
def test_nonfinite_returns_same_status_without_extra_blocks(layout,kind):
    c=build(layout,max_blocks=0)
    data={'offset':jnp.asarray([-1.,-2.]),'cost':jnp.asarray([.2,-.1])}
    warm=(jnp.asarray([0.,0.]),jnp.zeros(4))
    if kind=='data': data['offset']=jnp.asarray([jnp.nan,-2.])
    else: warm=(jnp.asarray([jnp.nan,0.]),warm[1])
    result=c.solve_instance(data,warm_start=warm)
    expected=c.solve_many_results(jax.tree_util.tree_map(lambda a:a[None],data),
                                 warm_start=(warm[0][None],warm[1][None]))[0]
    same(result,expected)
    assert result.status=='numerical_error' and result.iters==0


@pytest.mark.parametrize('layout',['dense','compact'])
def test_eager_callback_zero_budget_and_zero_cost(layout):
    calls=[]
    def offset(d):
        calls.append(1)
        return d['a']*d['b']+d['c']
    c=ProblemFamily(2,[Term(ocs.Identity(2),ocs.NonNeg()),
        Term(ocs.Dense(np.eye(2,dtype=np.float32)),ocs.SumSquares(),offset)]).compile(
        equilibrate='scalar',pdhg_variant='standard',data_layout=layout,max_blocks=0)
    data={'a':jnp.asarray([4097.,2.]),'b':jnp.asarray([4097.,3.]),'c':jnp.asarray([-16785408.,-6.])}
    sol=c.solve_instance(data)
    assert calls==[1] and sol.status=='solved' and sol.iters==0 and sol.residual==0
    np.testing.assert_array_equal(sol.x,np.zeros(2))
    assert c.certificate(sol.x,sol.y,data)['r_primal']==0


@pytest.mark.parametrize('layout',['dense','compact'])
def test_all_absent_inputs_single_cold_path(layout):
    x=ocs.Variable(2)
    c=ocs.minimize([ocs.nonneg(x),ocs.least_squares(ocs.Dense(np.eye(2))@x)]).compile(
        equilibrate='scalar',pdhg_variant='standard',data_layout=layout,max_blocks=0)
    result=c.solve_instance({})
    assert result.status=='solved' and result.iters==0
    np.testing.assert_array_equal(result.x,np.zeros(2))


@pytest.mark.parametrize('layout',['dense','compact'])
def test_halpern_remains_on_existing_batched_engine(layout):
    c=build(layout,pdhg_variant='halpern')
    assert c._single_solver is None and c._single_cold_solver is None
    data={'offset':jnp.asarray([-1.,-2.]),'cost':jnp.asarray([.2,-.1])}
    same(c.solve_instance(data),c.solve_many_results(jax.tree_util.tree_map(lambda a:a[None],data))[0])
