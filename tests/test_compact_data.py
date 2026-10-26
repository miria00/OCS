"""Compact input contract: compare actual public APIs against dense consumers."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.ir import ProblemFamily, Term

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# dense and compact data layouts agreeing.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def family(layout, *, cost=True, callback=None, **options):
    offset = callback or (lambda data: data['offset'])
    defaults = dict(equilibrate='scalar', block=10, max_blocks=30, tol=1e-4,
                    data_layout=layout, pdhg_variant='standard')
    defaults.update(options)
    return ProblemFamily(2, [
        Term(ocs.Identity(2), ocs.NonNeg()),
        Term(ocs.Dense(np.diag([2., 3.]).astype(np.float32)), ocs.SumSquares(), offset),
        Term(ocs.Dense(np.eye(2, dtype=np.float32)), ocs.L1(.1)),
    ], cost_fn=(lambda d: d['cost']) if cost else None).compile(**defaults)


def same_solution(actual, expected):
    assert actual.status == expected.status
    assert actual.iters == expected.iters
    for key in ('x', 'y', 'residual'):
        np.testing.assert_allclose(getattr(actual,key), getattr(expected,key),
                                   rtol=4e-6, atol=4e-7, equal_nan=True)
    if expected.certificate is None:
        assert actual.certificate is None
    else:
        np.testing.assert_allclose(actual.certificate, expected.certificate,
                                   rtol=4e-6, atol=4e-7)


@pytest.mark.parametrize('offset', [-1., np.array([-1.],np.float32), np.array([-1.,-2.],np.float32)])
@pytest.mark.parametrize('cost', [False,True])
def test_broadcast_offsets_costs_and_certificate(offset,cost):
    compact,dense=(family(mode,cost=cost) for mode in ('compact','dense'))
    data={'offset':offset,'cost':jnp.asarray([.2,-.1]),'ignored':object(),'ignored_nan':np.nan}
    result=compact.solve_instance(data)
    expected=dense.solve_instance(data)
    same_solution(result,expected)
    report=compact.certificate(result.x,result.y,data)
    reference=dense.certificate(result.x,result.y,data)
    for key in report:
        np.testing.assert_allclose(report[key],reference[key],rtol=3e-6,atol=3e-7,equal_nan=True)


def test_eager_callback_once_raw_python_objects_and_cancellation():
    seen=[]
    sentinel=object()
    def offset(data):
        assert data['sentinel'] is sentinel
        assert data['mode']=='custom'
        seen.append(1)
        return data['a']*data['b']+data['c']
    compiled=family('compact',cost=False,callback=offset,max_blocks=0)
    data={'sentinel':sentinel,'mode':'custom','a':jnp.asarray([4097.,2.]),
          'b':jnp.asarray([4097.,3.]),'c':jnp.asarray([-16785408.,-6.])}
    result=compiled.solve_instance(data)
    assert seen==[1] and result.status=='solved' and result.residual==0
    compact=compiled._resolve(data)
    assert seen==[1,1]
    np.testing.assert_array_equal(compact[0][0],np.zeros(2))
    assert compact[0][1] is compact[1] is None
    starts=(jnp.zeros(2),jnp.zeros(4))
    compiled.solve_instance(data,warm_start=starts)
    compiled.certificate(result.x,result.y,data)
    assert len(seen)==4
    np.testing.assert_array_equal(starts[0],np.zeros(2))
    np.testing.assert_array_equal(starts[1],np.zeros(4))


def test_batched_callbacks_do_not_fuse_cancellation():
    offset=lambda d:d['a']*d['b']+d['c']
    compiled=family('compact',cost=False,callback=offset,max_blocks=0)
    data={'a':jnp.asarray([[4097.,2.],[2.,3.]]),'b':jnp.asarray([[4097.,3.],[3.,4.]]),
          'c':jnp.asarray([[-16785408.,-6.],[-6.,-12.]])}
    resolved=jax.vmap(compiled._resolve)(data)
    np.testing.assert_array_equal(resolved[0][0],np.zeros((2,2)))
    results=compiled.solve_many_results(data)
    reports=compiled.certificate_many(jnp.zeros((2,2)),jnp.zeros((2,4)),data)
    assert all(r.status=='solved' and r.residual==0 for r in results)
    for k in ('r_primal','r_dual','gap'):
        np.testing.assert_array_equal(reports[k],np.zeros(2))


def test_batch_nonfinite_and_warm_ownership():
    compact,dense=(family(mode) for mode in ('compact','dense'))
    data={'offset':jnp.asarray([[-1.,-2.],[-2.,-1.],[jnp.nan,-1.]]),
          'cost':jnp.asarray([[.2,-.1],[.1,-.2],[0.,0.]]),'ignored':jnp.full((3,1),jnp.nan)}
    starts=(jnp.zeros((3,2)),jnp.zeros((3,4)))
    got=compact.solve_many_results(data,warm_start=starts)
    expected=dense.solve_many_results(data,warm_start=starts)
    assert got[-1].status=='numerical_error'
    for a,b in zip(got,expected): same_solution(a,b)
    for a in starts: np.testing.assert_array_equal(a,np.zeros(a.shape))
    direct=compact.solve_many(data,warm_start=starts)
    for i,a in enumerate(got): np.testing.assert_allclose(direct[0][i],a.x,equal_nan=True)
    reports=compact.certificate_many(direct[0],direct[1],data)
    for i in range(3):
        scalar=compact.certificate(direct[0][i],direct[1][i],{k:v[i] for k,v in data.items()})
        for k in scalar: np.testing.assert_allclose(reports[k][i],scalar[k],rtol=3e-6,atol=3e-7,equal_nan=True)


def test_all_zero_data_batch_token_and_optional_certificate():
    x=ocs.Variable(2)
    p=ocs.minimize([ocs.nonneg(x),ocs.least_squares(ocs.Dense(np.eye(2))@x)])
    c=p.compile(equilibrate='scalar',max_blocks=0,data_layout='compact')
    leaves=jax.tree_util.tree_leaves(c._resolve({}))
    assert len(leaves)==1 and leaves[0].shape==() and leaves[0].nbytes==4
    assert c.solve_instance({}).status=='solved'
    assert len(c.solve_many_results({'ignored':jnp.full((3,1),jnp.nan)}))==3
    report=c.certificate_many(jnp.zeros((3,2)),jnp.zeros((3,2)))
    np.testing.assert_array_equal(report['r_primal'],np.zeros(3))


@pytest.mark.parametrize('bad', [np.ones((2,1)),np.ones(3),np.ones((1,2))])
def test_bad_offset_shapes_fail_even_zero_iterations(bad):
    for layout in ('compact','dense'):
        c=family(layout,cost=False,max_blocks=0)
        with pytest.raises(ValueError): c.solve_instance({'offset':bad})


def test_cost_callback_once_and_invalid_value():
    seen=[]; sentinel=object()
    def cost(d):
        assert d['sentinel'] is sentinel
        seen.append(1)
        return jnp.asarray(d['cost'],jnp.float32)
    c=ProblemFamily(1,[Term(ocs.Identity(1),ocs.NonNeg()),
        Term(ocs.Dense(np.ones((1,1))),ocs.SumSquares())],cost_fn=cost).compile(
        equilibrate='scalar',max_blocks=0,data_layout='compact')
    assert c.solve_instance({'sentinel':sentinel,'cost':[np.nan]}).status=='numerical_error'
    assert len(seen)==1
    assert c.solve_instance({'sentinel':sentinel,'cost':[0.]}).status=='solved'
    assert len(seen)==2


@pytest.mark.parametrize('direction', ['primal_infeasible','unbounded'])
def test_status_directions_and_invalid_warm_point(direction):
    x=ocs.Variable(1)
    if direction=='primal_infeasible':
        p=ocs.minimize([ocs.nonneg(x),ocs.equality(ocs.Dense(np.ones((1,1)))@x-ocs.Parameter('b'))])
        data={'b':jnp.array([[-1.],[-1.]])}
        starts=(jnp.array([[0.],[jnp.nan]]),jnp.ones((2,1)))
    else:
        p=ProblemFamily(1,[Term(ocs.Identity(1),ocs.NonNeg()),Term(ocs.Dense(np.zeros((1,1))),ocs.Zero())],
                        cost_fn=lambda d:d['cost'])
        data={'cost':-jnp.ones((2,1))}
        starts=(jnp.array([[1.],[jnp.nan]]),jnp.zeros((2,1)))
    families=[p.compile(equilibrate='scalar',max_blocks=0,data_layout=l) for l in ('compact','dense')]
    got,expected=(c.solve_many_results(data,warm_start=starts) for c in families)
    assert [s.status for s in got]==[direction,'numerical_error']
    for a,b in zip(got,expected): same_solution(a,b)
    assert np.isnan(np.asarray(starts[0])[1,0])


@pytest.mark.parametrize('options', [
    {'pdhg_variant':'halpern'}, {'pdhg_variant':'halpern','adaptive_omega':True},
    {'dual_layout':'blocks'}, {'loop_unroll':4}, {'residual_norm':'data'},
    {'residual_norm':'iterate'}, {'pdhg_variant':'halpern','online_warmup':2},
    {'pdhg_variant':'halpern','online_warmup':2,'online_method':'osgm'},
    {'equilibrate':('given',jnp.asarray([.1,.2]),jnp.asarray([.2,.1,.3,.1]))},
])
def test_solver_options_keep_certificate_and_solution_contract(options):
    compact,dense=(family(mode,**options) for mode in ('compact','dense'))
    data={'offset':jnp.array([-1.,-2.]),'cost':jnp.array([.2,-.1])}
    same_solution(compact.solve_instance(data),dense.solve_instance(data))


def test_dense_refinement_and_differentiation_contract():
    # Smooth strongly convex analytic reference avoids nonsmooth derivatives.
    p=ProblemFamily(2,[Term(ocs.Identity(2),ocs.SumSquares()),
                      Term(ocs.Dense(np.diag([2.,3.])),ocs.SumSquares(),lambda d:-d['b'])])
    cs=[p.compile(equilibrate='scalar',data_layout=l,pdhg_variant='standard',tol=1e-5,
                  block=20,max_blocks=80) for l in ('compact','dense')]
    b=jnp.array([1.,2.])
    compact,dense=cs
    sol=compact.solve_instance({'b':b})
    refined=[c.refine(sol.x,sol.y,{'b':b},steps=1,cg_iters=10) for c in cs]
    for a,e in zip(*refined): np.testing.assert_allclose(a,e,rtol=1e-6,atol=1e-7)
    diffs=[jax.jacobian(lambda v:c.solve_diff({'b':v}))(b) for c in cs]
    np.testing.assert_allclose(diffs[0],diffs[1],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(diffs[0],np.diag([2/5,3/10]),rtol=1e-4,atol=1e-5)


def test_input_memory_eliminates_only_absent_buffers():
    compact,dense=(family(mode,cost=False,max_blocks=0) for mode in ('compact','dense'))
    data={'offset':jnp.asarray([-1.,-2.])}
    count=lambda d:sum(a.nbytes for a in jax.tree_util.tree_leaves(d))
    assert count(compact._resolve(data))==2*4+4
    assert count(dense._resolve(data))==(4+2)*4
    # Explicit zero callbacks stay dynamic and finite checked.
    explicit=compact._resolve({'offset':jnp.zeros(2)})
    assert explicit[0][0] is not None and explicit[0][0].shape==(2,)


@pytest.mark.parametrize('bad', ['auto',None,False])
def test_invalid_layout_is_rejected(bad):
    with pytest.raises(ValueError,match='data_layout'): family(bad)


def test_none_from_explicit_cost_callback_is_not_structural_absence():
    p=ProblemFamily(1,[Term(ocs.Identity(1),ocs.NonNeg()),
                      Term(ocs.Dense(np.ones((1,1))),ocs.SumSquares())],
                    cost_fn=lambda d:None)
    for layout in ('compact','dense'):
        c=p.compile(equilibrate='scalar',max_blocks=0,data_layout=layout)
        with pytest.raises((TypeError,ValueError)):
            c.solve_instance({})


@pytest.mark.parametrize('kind',['exp','power'])
def test_cone_supplied_point_certificate_matches_dense_at_origin_and_violation(kind):
    x=ocs.Variable(3)
    cone=ocs.exp_cone if kind=='exp' else ocs.pow_cone
    p=ocs.minimize([ocs.sum_squares(x),cone(ocs.Diagonal(jnp.ones(3))@x)],cost=ocs.Parameter('c'))
    cs=[p.compile(equilibrate='scalar',max_blocks=0,data_layout=l) for l in ('compact','dense')]
    normal=np.array([0.,0.,-1.] if kind=='exp' else [-1.,-1.,0.],np.float32)
    scales=np.array([1e-6,1.,1e6],np.float32)[:,None]
    yy=jnp.asarray(np.concatenate([scales*normal,scales*normal]))
    xx=jnp.asarray(np.concatenate([np.zeros((3,3),np.float32),.05*scales*normal]))
    data={'c':-yy}
    reports=[c.certificate_many(xx,yy,data) for c in cs]
    for key in reports[0]:
        np.testing.assert_allclose(reports[0][key],reports[1][key],rtol=1e-5,atol=1e-6,equal_nan=True)
    r=reports[0]
    assert np.max(np.maximum(r['r_primal'][:3],r['r_dual'][:3]))<1e-5
    assert np.all(np.maximum(r['r_primal'][3:],r['r_dual'][3:])>1e-3)
