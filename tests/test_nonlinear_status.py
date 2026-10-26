"""Independent positive/negative controls for exp/power exceptional statuses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ocs.prox.cones import ExpCone,ExpConeBatch,PowerCone,PowerConeBatch
from ocs.prox.library import Equality,SumSquares,NonNeg
from ocs.prox.base import ProxBlock
from ocs.solver.certificates import make_status_checks,_domain,_nonlinear_cone_distance


def block_and_vectors(kind,q=1,alpha=.3):
    if kind=='exp':
        block=ExpCone() if q==1 else ExpConeBatch(q)
        inside=np.array([1.,1.,4.]);outside=np.array([0.,1.,-1.]);dual=np.array([0.,0.,1.]);ray=np.array([0.,0.,1.])
    else:
        block=PowerCone(alpha) if q==1 else PowerConeBatch(q,alpha)
        inside=np.array([1.,1.,0.]);outside=np.array([-1.,-1.,0.]);dual=np.array([alpha,1-alpha,0.]);ray=np.array([1.,0.,0.])
    layout=lambda v:jnp.asarray(np.repeat(v,q),jnp.float32)
    return block,*[layout(v) for v in (inside,outside,dual,ray)]


def independently_in_polar(kind,value,q,alpha=.3,tol=2e-6):
    rows=np.asarray(value,float).reshape(3,q).T
    for a,b,c in rows:
        if kind=='exp':
            if a>tol:
                if c>=0 or b>a*np.log(-np.e*c/a)+tol:return False
            elif abs(a)<=tol:
                if b>tol or c>tol:return False
            else:return False
        else:
            u,v=-a/alpha,-b/(1-alpha)
            if u < -tol or v < -tol or abs(c)>max(u,0)**alpha*max(v,0)**(1-alpha)+tol:return False
    return True


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('scale',[1e-20,1e-6,1.,1e6,1e20])
def test_known_farkas_directions_and_independent_polar_margin(kind,q,scale):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    n=len(inside);check=make_status_checks(lambda x:x,lambda y:y,cone,[Equality()],[(0,n)],n,n,operator_norm=1.)
    result=check(jnp.zeros(n),dual*scale,(-outside*scale,jnp.zeros(n)))
    assert result['primal_infeasibility_supported']
    assert result['primal_infeasible'] and not result['unbounded']
    direction=np.asarray(result['dual_direction'],float)
    assert independently_in_polar(kind,-direction,q)
    assert -np.asarray(outside,float)@direction>0
    assert float(result['primal_infeasibility_residual'])<=1e-6


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_actual_unbounded_cone_program_has_feasible_base_and_ray(kind,q,scale):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    n=len(inside);check=make_status_checks(lambda x:x,lambda y:y,None,[cone],[(0,n)],n,n,operator_norm=1.)
    result=check(ray*scale,jnp.zeros(n),(jnp.zeros(n),-ray*scale))
    assert result['unbounded'] and not result['primal_infeasible']
    assert float(result['base_feasibility_residual'])==0.
    assert float(result['recession_residual'])<=1e-6
    assert float(result['recession_margin'])>.9


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
def test_bad_projector_cannot_make_feasible_problem_farkas(kind,q):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    cone.prox=lambda value,step:jnp.zeros_like(value)
    n=len(inside);check=make_status_checks(lambda x:x,lambda y:y,cone,[Equality()],[(0,n)],n,n,operator_norm=1.)
    # Positive separation alone is misleading: this multiplier is not polar.
    bad=-jnp.ones(n)
    result=check(inside,bad,(-inside,jnp.zeros(n)))
    assert not result['primal_infeasible']
    assert float(result['primal_infeasibility_residual'])>1e-3


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
def test_bad_projector_cannot_make_invalid_recession_or_base_valid(kind,q):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    cone.prox=lambda value,step:value
    n=len(inside)
    # A bad identity projector would accept every recession direction.
    check=make_status_checks(lambda x:jnp.zeros(1),lambda y:jnp.zeros(n),cone,[Equality()],[(0,1)],n,1,operator_norm=0.)
    result=check(outside,jnp.zeros(1),(jnp.zeros(1),-outside))
    assert not result['unbounded'] and float(result['recession_residual'])>1e-3
    # Both zero and current base have a negative second cone coordinate.
    offset=np.zeros((3,q),np.float32);offset[1]=-1
    check=make_status_checks(lambda x:x,lambda y:y,None,[cone],[(0,n)],n,n,operator_norm=1.)
    result=check(ray,jnp.zeros(n),(jnp.asarray(offset.reshape(-1)),-ray))
    assert not result['unbounded'] and float(result['base_feasibility_residual'])>1e-3


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_ray_never_certifies(kind,q,bad):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    n=len(inside);check=make_status_checks(lambda x:x,lambda y:y,cone,[Equality()],[(0,n)],n,n,operator_norm=1.)
    result=check(jnp.zeros(n),dual.at[0].set(bad),(-outside,jnp.zeros(n)))
    assert not result['primal_infeasible'] and not result['unbounded']


@pytest.mark.parametrize('kind',['exp','power'])
@pytest.mark.parametrize('q',[1,3])
def test_zero_and_closure_faces_have_no_artificial_unit_floor(kind,q):
    cone,inside,outside,dual,ray=block_and_vectors(kind,q)
    zero=jnp.zeros(len(inside))
    assert float(_nonlinear_cone_distance(cone,zero))==0.
    assert float(_nonlinear_cone_distance(cone,zero,polar=True))==0.
    if kind=='exp':face=jnp.asarray(np.repeat([-2.,0.,3.],q),jnp.float32)
    else:face=jnp.asarray(np.repeat([2.,0.,0.],q),jnp.float32)
    assert float(_nonlinear_cone_distance(cone,face))==0.
    check=make_status_checks(lambda x:x,lambda y:y,None,[cone],[(0,len(zero))],len(zero),len(zero),operator_norm=1.)
    result=check(zero,zero,(zero,zero))
    assert not result['unbounded'] and not result['primal_infeasible']


@pytest.mark.parametrize('block',[ProxBlock(),NonNeg().precomposed(b=1.),PowerCone(0.),PowerCone(1.)])
def test_unknown_shifted_and_invalid_domains_remain_unsupported(block):
    assert _domain(block)[0][0]=='unsupported'


def test_nonlinear_objective_recession_is_still_not_claimed():
    cone=ExpCone();check=make_status_checks(lambda x:x,lambda y:y,SumSquares(),[cone],[(0,3)],3,3,operator_norm=1.)
    result=check(jnp.ones(3),jnp.zeros(3),(jnp.zeros(3),-jnp.ones(3)))
    assert result['primal_infeasibility_supported']
    assert not result['recession_supported'] and not result['unbounded']


@pytest.mark.parametrize('q',[1,3])
@pytest.mark.parametrize('point,expected', [
    ((1.,-1.,-1.), True),  # Strict polar interior: b < a*log(-e*c/a).
    ((-1.,-1.,-1.), False), # Violates the closed-face sign constraint.
    ((1.,2.,-1.), False),   # Correct signs but violates the curved polar face.
])
def test_exponential_polar_interior_and_sign_controls(q,point,expected):
    block=ExpCone() if q==1 else ExpConeBatch(q)
    value=np.repeat(np.asarray(point,np.float32),q)
    assert independently_in_polar('exp',value,q)==expected
    distance=float(_nonlinear_cone_distance(block,jnp.asarray(value),polar=True))
    assert (distance==0)==expected
    if not expected:
        assert distance > .1
    # An independent primal witness rules out either invalid multiplier.
    if point[0] < 0:
        primal=np.array([-10.,1.,np.exp(-10.)])
        assert np.dot(np.asarray(point),primal)>0
    elif not expected:
        primal=np.array([0.,1.,1.])
        assert np.dot(np.asarray(point),primal)>0
