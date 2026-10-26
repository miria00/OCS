"""Actual returned-pair recovery; scalar report remains an independent check.

Run against a selected isolated source via PYTHONPATH. The zero-iteration
warm starts isolate recovery from outer-solver convergence, and analytic
normals give independent expected primal/dual pairs.
"""
import numpy as np
import jax.numpy as jnp
import pytest
import ocs
from ocs.ir import Term, ProblemFamily
from ocs.prox.cones import ExpConeBatch, PowerConeBatch

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# dual recovery, measured through the residual.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def arr(v):
    return jnp.asarray(v, dtype=jnp.float32)


def _compile(cone, K, offset, cost, tol=1e-6):
    n=K.shape[1]
    family=ProblemFamily(n=n, terms=[
        Term(ocs.Dense(np.eye(n,dtype=np.float32)),ocs.SumSquares()),
        Term(ocs.Dense(K),cone,lambda d:d['l'])],cost_fn=lambda d:d['c'])
    return family.compile(equilibrate='scalar',max_blocks=0,tol=tol), {'l':arr(offset),'c':arr(cost)}


def _check_report(compiled, solution, data):
    report=compiled.certificate(solution.x,solution.y,data)
    residual=max(float(report['r_primal']),float(report['r_dual']))
    np.testing.assert_allclose(solution.residual,residual,rtol=2e-5,atol=2e-7)
    assert (solution.status=='solved') == (residual <= compiled._tol)
    return report


@pytest.mark.parametrize('kind', ['exp','power'])
def test_interior_zero_normal_is_returned_and_certified(kind):
    cone=ocs.ExpCone() if kind=='exp' else ocs.PowerCone(.3)
    offset=[0,1,2] if kind=='exp' else [1,1,0]
    compiled,data=_compile(cone,np.eye(3,dtype=np.float32),offset,[0,0,0])
    y=arr([0,0,0,1e-7,2e-7,3e-7])
    before=compiled.certificate(arr([0,0,0]),y,data)
    assert max(float(before['r_primal']),float(before['r_dual'])) > 1e-3
    solution=compiled.solve_instance(data,warm_start=(arr([0,0,0]),y))
    assert solution.status=='solved'
    assert solution.iters==0
    np.testing.assert_array_equal(solution.y,np.zeros(6))
    _check_report(compiled,solution,data)


@pytest.mark.parametrize('kind', ['exp','power'])
def test_batch_recovery_preserves_active_normal(kind):
    # Block zero lies on a curved face; block one is strictly interior.
    cone=ExpConeBatch(2) if kind=='exp' else PowerConeBatch(2,.3)
    normal=np.array([1,1,-1] if kind=='exp' else [-.3,-.7,1],np.float32)
    boundary=np.array([0,1,1] if kind=='exp' else [1,1,1],np.float32)
    interior=np.array([0,1,2] if kind=='exp' else [1,1,0],np.float32)
    K=np.repeat(np.eye(3,dtype=np.float32),2,axis=0)
    offset=np.stack((boundary,interior),axis=1).reshape(-1)
    cone_y=np.stack((normal,np.full(3,1e-7,np.float32)),axis=1).reshape(-1)
    expected=np.stack((normal,np.zeros(3,np.float32)),axis=1).reshape(-1)
    compiled,data=_compile(cone,K,offset,-K.T@expected)
    solution=compiled.solve_instance(data,warm_start=(arr([0,0,0]),arr(np.r_[np.zeros(3),cone_y])))
    assert solution.status=='solved'
    np.testing.assert_allclose(solution.y[3:],expected,atol=0,rtol=0)
    _check_report(compiled,solution,data)


def test_psd_relative_interior_retains_nonzero_skew_normal():
    # Symmetry forces x=0; K.T does not annihilate skew multipliers.
    K=np.array([[0],[1],[-1],[0]],np.float32)
    skew=np.array([0,-.5,.5,0],np.float32)
    offset=np.eye(2,dtype=np.float32).reshape(-1)
    compiled,data=_compile(ocs.PSDCone(2),K,offset,[1])
    y=arr(np.r_[0.,skew+.05*offset])
    solution=compiled.solve_instance(data,warm_start=(arr([0]),y))
    assert solution.status=='solved'
    np.testing.assert_array_equal(solution.y[1:],skew)
    np.testing.assert_array_equal(K.T@np.asarray(solution.y[1:]),[-1])
    report=_check_report(compiled,solution,data)
    assert float(report['r_primal'])==0
    # Discarding skew would fail stationarity, despite primal positive definiteness.
    bad=compiled.certificate(solution.x,jnp.zeros_like(solution.y),data)
    assert float(bad['r_primal']) >= .99


def _near_boundary_data(kind):
    delta=1e-4
    if kind=='exp':
        cone=ocs.ExpCone();offset=[0,1,1+delta];normal=[1,1,-1]
    elif kind=='power':
        cone=ocs.PowerCone(.3);offset=[1,1,1-delta];normal=[-.3,-.7,1]
    else:
        cone=ocs.PSDCone(2);offset=[delta,0,0,1];normal=[-1,0,0,0]
    return cone,np.asarray(offset,np.float32),np.asarray(normal,np.float32)


def _score(report):
    return max(float(report['r_primal']),float(report['r_dual']))


@pytest.mark.parametrize('kind', ['exp','power','psd'])
def test_coupled_stationarity_prevents_erasing_needed_near_boundary_normal(kind):
    cone,offset,normal=_near_boundary_data(kind);n=normal.size
    # A nonzero point gives a nonzero objective reference. This makes retaining
    # the multiplier strictly preferable, rather than relying on a rounded tie
    # between (r_primal,r_dual)=(0,1) and (1,0).
    point=np.zeros(n,np.float32);point[0]=1
    compiled,data=_compile(cone,np.eye(n,dtype=np.float32),offset-point,
                           -normal-point,tol=1e-6)
    x=arr(point);y=arr(np.r_[point,normal])
    solution=compiled.solve_instance(data,warm_start=(x,y))
    assert solution.status=='inaccurate'
    np.testing.assert_array_equal(solution.y,y)
    report=_check_report(compiled,solution,data)
    assert float(report['r_primal']) < 1e-6
    assert 1e-6 < float(report['r_dual']) < 1e-3
    # The actual proposed candidate preserves the exact smooth multiplier.
    # Only the cone block changes; zeroing every dual block would test a
    # different pair and obscure the stationarity comparison.
    candidate=y.at[n:].set(0.)
    candidate_report=compiled.certificate(x,candidate,data)
    assert float(candidate_report['r_primal']) >= .5
    assert _score(report) < .01*_score(candidate_report)


@pytest.mark.parametrize('kind', ['exp','power','psd'])
def test_zero_objective_score_tie_preserves_truthful_status_and_certificate(kind):
    cone,offset,normal=_near_boundary_data(kind);n=normal.size
    compiled,data=_compile(cone,np.eye(n,dtype=np.float32),offset,-normal,tol=1e-6)
    x=jnp.zeros(n,jnp.float32);y=arr(np.r_[np.zeros(n),normal])
    candidate=y.at[n:].set(0.)
    original_report=compiled.certificate(x,y,data)
    candidate_report=compiled.certificate(x,candidate,data)
    scores=[_score(original_report),_score(candidate_report)]
    # A small complementarity product also supplies the zero-objective
    # normalization scale. Both full coupled scores are mathematically one.
    np.testing.assert_allclose(scores,[1.,1.],rtol=2e-6,atol=2e-7)
    solution=compiled.solve_instance(data,warm_start=(x,y))
    assert solution.status=='inaccurate'
    np.testing.assert_array_equal(solution.x,x)
    report=_check_report(compiled,solution,data)
    roundoff=8*np.finfo(np.float32).eps*max(scores)
    assert _score(report) <= min(scores)+roundoff


def test_psd_cancellation_keeps_active_block_while_repairing_inactive_block():
    # The second cone has tiny positive slack formed by order-one cancellation.
    # Its nonzero multiplier is needed for stationarity. The first cone is
    # genuinely inactive; its symmetric roundoff can be removed independently.
    eye=np.eye(2,dtype=np.float32).reshape(-1)
    K0=np.zeros((4,1),np.float32);K1=eye[:,None]
    problem=ProblemFamily(n=1,terms=[
        Term(ocs.Dense(np.ones((1,1),np.float32)),ocs.SumSquares()),
        Term(ocs.Dense(K0),ocs.PSDCone(2),lambda d:d['l0']),
        Term(ocs.Dense(K1),ocs.PSDCone(2),lambda d:d['l1'])],cost_fn=lambda d:d['c'])
    data={'l0':arr(eye),'l1':arr((-1+1e-7)*eye),'c':arr([1])}
    compiled=problem.compile(equilibrate='scalar',max_blocks=0,tol=1e-6)
    skew=np.array([0,1e-6,-1e-6,0],np.float32)
    y=arr(np.r_[1,skew-4e-8*eye,-eye]);x=arr([1])
    solution=compiled.solve_instance(data,warm_start=(x,y))
    assert solution.status=='solved'
    np.testing.assert_array_equal(solution.y[1:5],skew)
    np.testing.assert_array_equal(solution.y[5:],-eye)
    np.testing.assert_array_equal(solution.x,x)
    _check_report(compiled,solution,data)
