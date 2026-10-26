"""Domain boundaries, wrapper algebra, public dispatch and actual witnesses."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.library import Pinball, Logistic, SumSquares, L1, NonNeg
from ocs.prox.cones import PSDCone, SecondOrderCone

@pytest.mark.parametrize('block,lo,hi',[(Pinball(.75),-.25,.75),(Logistic(),0.,1.)])
def test_closed_domain_no_slack(block,lo,hi):
    y=jnp.asarray([lo,hi],jnp.float32)
    assert bool(block.conj_in_domain(y)) and np.isfinite(float(block.conj_value(y)))
    for bad in (np.nextafter(np.float32(lo),np.float32(-np.inf)),
                np.nextafter(np.float32(hi),np.float32(np.inf)), np.nan, np.inf):
        v=jnp.asarray([bad],jnp.float32)
        assert not bool(block.conj_in_domain(v))
        assert np.isinf(float(block.conj_value(v)))

def test_logistic_continuous_endpoints():
    assert float(Logistic().conj_value(jnp.array([0.,1.]))) == 0.
    assert float(Logistic().conj_value(jnp.array([.5]))) == pytest.approx(-np.log(2),abs=1e-7)

@pytest.mark.parametrize('base,y',[(Pinball(.75),[-.1,.2]),(Logistic(),[.2,.8]),
                                (SumSquares(),[-2.,3.]),(SecondOrderCone(),[-3.,1.])])
def test_scaling_offset_negative_precomposition(base,y):
    a,b,c=-2.,.25,3.
    wrapped=base.scaled(c).precomposed(a=a,b=b)
    yy=jnp.asarray(y)*a*c
    expected=c*float(base.conj_value(jnp.asarray(y)))+float(jnp.sum((yy/a)*b))
    assert wrapped.bound_capable and bool(wrapped.conj_in_domain(yy))
    assert float(wrapped.conj_value(yy)) == pytest.approx(expected,rel=2e-6,abs=2e-6)

@pytest.mark.parametrize('base', [Pinball(.75),Logistic(),SumSquares(),L1(.7),NonNeg()])
def test_compound_quadratic_fenchel_equality(base):
    wrapped=base.with_linear_quadratic(d=.125,e=.5)
    yy=jnp.asarray([-.7,.2,1.3],jnp.float32)
    point=base.prox((yy-.125)/.5,jnp.full_like(yy,2.))
    expected=float(jnp.dot(yy,point)-wrapped.value(point))
    assert wrapped.bound_capable and bool(wrapped.conj_in_domain(yy))
    assert float(wrapped.conj_value(yy)) == pytest.approx(expected,rel=2e-6,abs=2e-6)
    shifted=base.with_linear_quadratic(d=.125)
    candidate=base.conj_domain_project(yy)+.125
    assert shifted.bound_capable
    assert bool(shifted.conj_in_domain(candidate))

def test_psd_polar_sign_and_skew():
    block=PSDCone(2)
    negative=jnp.array([[-1.,2.],[-2.,-3.]]).ravel()
    assert bool(block.conj_in_domain(negative))
    assert float(block.conj_value(negative)) == 0.
    assert not bool(block.conj_in_domain(-negative))
    assert not bool(block.conj_in_domain(jnp.diag(jnp.array([1e-12,-1.])).ravel()))
    assert bool(block.conj_in_domain(block.conj_domain_project(-negative)))

@pytest.mark.parametrize('magnitude',[1.0001,2.1,100.])
def test_dual_ray_search_includes_interval_above_last_feasible_halving(magnitude):
    x=ocs.Variable(1)
    op=ocs.LinearOperator.from_forward(lambda v:v,n=1,m=1)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(op@x-ocs.Parameter('b'))],
                      cost=jnp.ones(1)).compile(equilibrate='scalar',max_blocks=0)
    report=comp.objective_bound(jnp.ones(1),jnp.array([-magnitude]),
                                {'b':jnp.ones(1)},rounds=0,return_dual=True)
    assert report['objective_lower_bound_available']
    assert .9999 < report['objective_lower_bound'] <= 1.
    # The supplied multiplier is infeasible; only its actually returned,
    # checked scaling may provide the bound. There are zero recovery rounds.
    witness=np.asarray(report['certifying_dual'],dtype=np.float64)
    assert np.all(witness+1. >= 0.)
    assert report['n_corrected_dual_rounds']==0

def test_valid_dual_survives_cost_below_lower_at_approximately_feasible_point():
    x=ocs.Variable(1)
    op=ocs.LinearOperator.from_forward(lambda v:v,n=1,m=1)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(op@x-ocs.Parameter('b'))],
                      cost=jnp.ones(1)).compile(equilibrate='scalar',max_blocks=0,tol=1e-3)
    report=comp.objective_bound(jnp.array([.999999]),jnp.array([-1.]),
                                {'b':jnp.ones(1)},recover=False)
    assert report['objective_lower_bound_available']
    assert report['objective_lower_bound']==1.
    assert report['gap_at_returned_point'] < 0.
    decision=comp.acceptance(report)
    assert decision['target_met']
    assert not decision['objective_bracket_valid']

def test_dense_orthant_coupling_refuses_cancelled_negative_slack():
    # The exact represented adjoint is -1; a float32 reduction may return 0.
    A=np.array([[-1e8],[-1.],[1e8]],dtype='f')
    x=ocs.Variable(1)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(ocs.Dense(A)@x)]).compile(max_blocks=0)
    report=comp.objective_bound(jnp.zeros(1),jnp.ones(3),{},recover=False)
    assert not report['objective_lower_bound_available']
    assert report['n_dense_absolute_adjoints']==1
    assert report['arithmetic']=='mixed'
    assert report['dense_assessment_matrix_dtype']=='float64'
    assert not jax.config.x64_enabled
    zero=comp.objective_bound(jnp.zeros(1),jnp.zeros(3),{},recover=False)
    assert zero['objective_lower_bound_available']

def test_dense_guard_recovers_interior_and_same_witness_independently():
    A=np.array([[1.,-.3],[.2,.7]],dtype='f')
    c=np.array([1.,1.],dtype='f')
    x=ocs.Variable(2)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(ocs.Dense(A)@x)],cost=c).compile(max_blocks=0)
    for scale in (1.,1.0001,2.,100.):
        y=np.linalg.solve(A.astype('d').T,-c.astype('d')).astype('f')*np.float32(scale)
        report=comp.objective_bound(jnp.zeros(2),jnp.asarray(y),{},rounds=0,return_dual=True)
        assert report['objective_lower_bound_available']
        witness=np.asarray(report['certifying_dual'],dtype='d')
        assert np.all(A.astype('d').T@witness+c.astype('d') >= 0)

def test_structured_orthant_coupling_refuses_cancelled_negative_slack():
    # The BinPack resource adjoint is -1 in represented real arithmetic,
    # although its f32 reduction can cancel it. No full operator is formed.
    R=np.array([[-1e8],[-1.],[1e8]],dtype='f')
    x=ocs.Variable(1)
    op=ocs.Weighted(ocs.BinPack(R,1),1.)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(op@x),
                      ocs.sum_squares(ocs.Diagonal(jnp.ones(1))@x)]).compile(max_blocks=0)
    y=jnp.array([0.,1.,1.,1.,0.],dtype=jnp.float32)
    report=comp.objective_bound(jnp.zeros(1),y,{},recover=False)
    assert not report['objective_lower_bound_available']
    assert report['n_structured_reference_adjoints']==1
    assert report['n_dense_reference_adjoints']==0
    assert report['coupling_arithmetic_guard']=='binpack_float64_adjoint_envelope'
    assert report['arithmetic']=='mixed'
    assert report['dense_assessment_matrix_dtype'] is None
    zero=comp.objective_bound(jnp.zeros(1),jnp.zeros(5),{},recover=False)
    assert zero['objective_lower_bound_available']
    assert not jax.config.x64_enabled

def test_dense_mixed_assessment_scalar_batch_context():
    A=np.array([[1.,.2],[.3,1.]],dtype='f');c=np.ones(2,'f')
    x=ocs.Variable(2)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(ocs.Dense(A)@x-ocs.Parameter('b'))],cost=c).compile(
        tol=1e-3,max_blocks=100,block=10,equilibrate='scalar')
    data={'b':jnp.ones(2)}
    single=comp.solve_instance(data)
    batch=comp.solve_many_results({'b':jnp.ones((2,2))})
    for s in [single]+batch:
        assert s.status=='solved'
        assert s.x.dtype==s.y.dtype==jnp.float32
        xx=np.asarray(s.x,dtype='d');yy=np.asarray(s.y,dtype='d')
        assert np.all(A.astype('d').T@yy+c.astype('d')>=0)
        assert abs(c.astype('d')@xx+yy.sum())/abs(yy.sum()) <= 1e-3
        report=comp.objective_bound(s.x,s.y,data,recover=False)
        assert comp.acceptance(report)['target_met']
    assert not jax.config.x64_enabled

def test_recovered_dense_return_rechecks_without_ray_materialization_charge():
    rng=np.random.default_rng(801)
    A=rng.normal(size=(8,16)).astype('f')
    xf=np.abs(rng.normal(size=16)).astype('f')
    b=(A@xf).astype('f'); c=np.abs(rng.normal(size=16)).astype('f')
    x=ocs.Variable(16)
    comp=ocs.minimize([ocs.nonneg(x),ocs.equality(ocs.Dense(A)@x-ocs.Parameter('b'))],
                      cost=ocs.Parameter('c')).compile(tol=1e-3,max_blocks=1500,block=100)
    data={'b':jnp.asarray(b),'c':jnp.asarray(c)}
    sol=comp.solve_instance(data)
    assert sol.status=='solved'
    report=comp.objective_bound(sol.x,sol.y,data,recover=False)
    assert comp.acceptance(report)['target_met']
    assert np.min(A.astype('d').T@np.asarray(sol.y,dtype='d')+c.astype('d'))>=0

def test_public_no_recovery_and_zero_feasibility():
    # A pure PSD feasibility program with an offset: x >= I in symmetric order.
    n=4; op=ocs.LinearOperator.from_forward(lambda v:v,n=n,m=n)
    fam=ProblemFamily(n,[Term(op,PSDCone(2),lambda d:-jnp.eye(2).ravel())])
    comp=fam.compile(equilibrate='scalar',max_blocks=1,tol=1e-3)
    x=2*jnp.eye(2).ravel(); bad=jnp.ones(4)
    raw=comp.objective_bound(x,bad,{},recover=False,return_dual=True)
    assert not raw['objective_lower_bound_available']
    assert raw['n_ray_searches']==raw['n_corrected_dual_rounds']==0
    assert sum(raw['operator_applications_breakdown'].values()) == raw['operator_applications']
    good=comp.objective_bound(x,bad,{},return_dual=True)
    assert good['objective_lower_bound_available'] and good['objective_lower_bound']==0
    np.testing.assert_array_equal(good['certifying_dual'],np.zeros(4))
    accepted=comp.acceptance(good)
    assert accepted['target_met']
    reject=comp.acceptance(comp.objective_bound(jnp.zeros(4),jnp.zeros(4),{}))
    assert not reject['target_met']

def test_psd_anchor_rechecks_actual_witness_and_refuses_missing_interior():
    for weights in (np.array([1.,1.],'f'),np.array([1.,-1.],'f')):
        op=ocs.LinearOperator.from_forward(lambda v:jnp.asarray(weights)*jnp.diag(v.reshape(2,2)),n=4,m=2)
        x=ocs.Variable(4);cost=np.array([-.2,.1,.1,-.3],'f')
        comp=ocs.minimize([ocs.psd(x,2),ocs.equality(op@x-ocs.Parameter('b'))],cost=cost).compile(max_blocks=0)
        report=comp.objective_bound(jnp.eye(2).ravel(),jnp.zeros(2),{'b':jnp.ones(2)},rounds=0,return_dual=True)
        if weights[1]>0:
            assert report['anchor_available']
            yy=np.asarray(report['certifying_dual'],dtype='d')
            slack=np.diag(weights.astype('d')*yy)+cost.astype('d').reshape(2,2)
            assert np.linalg.eigvalsh(slack).min()>=0
        else:
            assert not report['anchor_available']

def test_diagonal_quadratic_public_route():
    x=ocs.Variable(2)
    fam=ocs.minimize([ocs.sum_squares(ocs.Diagonal(jnp.array([.1,.2]))@x),
                      ocs.pinball(ocs.Dense(jnp.eye(2))@x, .75)])
    comp=fam.compile(max_blocks=1)
    assert comp.meta['folded_diagonal_quadratic']
    assert comp.m==2
    report=comp.objective_bound(jnp.zeros(2),jnp.zeros(2),{},recover=False)
    assert report['objective_lower_bound_available']

def test_rof_original_objective_and_dual_identity():
    h,w=4,5; n=h*w
    rng=np.random.default_rng(42); b=rng.normal(size=n).astype('f'); alpha=.2
    g=SumSquares().precomposed(b=b).scaled(1/alpha)
    G=ocs.Gradient2D(h,w)
    fam=ProblemFamily(n,[Term(ocs.Identity(n),g),Term(G,L1(1.))])
    comp=fam.compile(max_blocks=1)
    x=jnp.asarray(rng.normal(size=n).astype('f'))
    y=jnp.asarray(rng.uniform(-.9,.9,size=2*n).astype('f'))
    report=comp.objective_bound(x,y,{},recover=False,return_dual=True)
    gt=np.asarray(G.rmatvec(y)).astype('d'); gx=np.asarray(G.matvec(x)).astype('d')
    xx=np.asarray(x).astype('d'); bb=b.astype('d'); yy=np.asarray(y).astype('d')
    lower=bb@gt-alpha*.5*(gt@gt)
    primal=.5/alpha*np.sum((xx-bb)**2)+np.abs(gx).sum()
    decomposed=.5/alpha*np.sum((xx-bb+alpha*gt)**2)+np.abs(gx).sum()-yy@gx
    assert report['objective_lower_bound_available']
    assert report['objective_lower_bound']==pytest.approx(lower,rel=2e-5,abs=2e-5)
    assert report['objective_at_returned_point']==pytest.approx(primal,rel=2e-6)
    assert primal-lower==pytest.approx(decomposed,rel=2e-6)

def test_unsupported_request_dispatches_no_solver_scalar_or_batch():
    class Unspecified(Logistic):
        bound_capable=False
    fam=ProblemFamily(2,[Term(ocs.Diagonal(jnp.ones(2)),Unspecified(),lambda d:d['b'])])
    comp=fam.compile(max_blocks=100000)
    def forbidden(*args,**kwargs):
        raise AssertionError('unsupported request dispatched solver')
    comp._single_solver=comp._single_cold_solver=comp._cold_solver=comp._run=forbidden
    comp._drive=comp._single_drive=forbidden
    single=comp.solve_instance({'b':jnp.zeros(2)})
    batch=comp.solve_many_results({'b':jnp.zeros((2,2))})
    for sol in [single]+batch:
        assert sol.status=='unverified' and sol.termination=='unsupported'
        assert sol.iters==0 and 'Unspecified' in sol.accuracy_detail

def test_finite_raw_cost_controls_acceptance_even_when_indicator_value_infinite():
    comp=ProblemFamily(2,[Term(ocs.Identity(2),SumSquares()),
                          Term(ocs.Diagonal(jnp.ones(2)),NonNeg())]).compile(max_blocks=0)
    packet=dict(objective_at_returned_point=1.,objective_lower_bound=.9999,
                objective_lower_bound_available=True,objective_primal_value=float('inf'),
                relative_primal_infeasibility=1e-6)
    from ocs.model.compile import Solution
    sol=Solution(jnp.zeros(2),jnp.zeros(2),0,0.)
    assert comp.acceptance(packet)['target_met']
    assert comp._award(sol,packet,None).status=='solved'

def test_public_batch_tuple_honors_requested_accuracy(monkeypatch):
    comp=ProblemFamily(2,[Term(ocs.Identity(2),SumSquares()),
                          Term(ocs.Diagonal(jnp.ones(2)),NonNeg(),lambda d:d['b'])]).compile(max_blocks=0)
    from ocs.model.compile import Solution
    expected=[Solution(jnp.ones(2),-jnp.ones(2),100,.25,status='inaccurate')]
    calls=[]
    def checked(*args): calls.append('checked');return expected
    monkeypatch.setattr(comp,'_solve_many_on_objective_gap',checked)
    xx,yy,nn,rr=comp.solve_many({'b':jnp.zeros((1,2))})
    assert calls==['checked']
    np.testing.assert_array_equal(xx,[[1.,1.]])
    np.testing.assert_array_equal(yy,[[-1.,-1.]])
    assert int(nn[0])==1 and float(rr[0])==.25
    calls.clear()
    comp.solve_many({'b':jnp.zeros((1,2))},accuracy='residual')
    assert calls==[]
