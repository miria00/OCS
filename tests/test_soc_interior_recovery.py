"""Actual SOC multiplier recovery, with unchanged coupled KKT acceptance."""
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.ir import ProblemFamily, Term
from ocs.prox.cones import SecondOrderCone, SOCBatch


def family(block, point, cost, alpha=1., gamma=1., offset=None):
    n=len(point)
    problem=ProblemFamily(n, [
        Term(ocs.Identity(n), ocs.SumSquares().scaled(gamma)),
        Term(ocs.Diagonal(jnp.full(n, alpha)), block,
             None if offset is None else lambda _: jnp.asarray(offset))],
        cost_fn=lambda _: gamma*jnp.asarray(cost))
    return problem.compile(max_blocks=0, tol=1e-4, accuracy='residual',
        equilibrate=('given', jnp.full(n, .01/gamma), jnp.full(n, .01*gamma/alpha**2)))


@pytest.mark.parametrize('alpha,gamma', [(1.,1.), (2.**-12,2.**8), (2.**12,2.**-8)])
@pytest.mark.parametrize('batch', [False, True])
def test_interior_zero_normal_and_active_boundary_remain_distinct(alpha,gamma,batch):
    # First cone is strictly interior with exact multiplier zero; the
    # second is on its boundary with a necessary nonzero normal.
    point=np.array([2.,1.,.25,1.], np.float32) if batch else np.array([2.,.25],np.float32)
    normal=np.array([0.,-1.,0.,1.],np.float32) if batch else np.zeros(2,np.float32)
    block=SOCBatch(2,1) if batch else SecondOrderCone()
    cost=-point-normal
    comp=family(block,point,cost,alpha,gamma)
    supplied=(gamma/alpha)*(normal+np.array([2.**-25,0.,0.,0.],np.float32) if batch
                            else np.array([2.**-25,0.],np.float32))
    sol=comp.solve_instance(warm_start=(jnp.asarray(point),jnp.asarray(supplied)))
    assert sol.status=='solved' and sol.iters==0
    np.testing.assert_array_equal(np.asarray(sol.x),point)
    np.testing.assert_array_equal(np.asarray(sol.y),(gamma/alpha)*normal)
    report=comp.certificate(sol.x,sol.y)
    assert max(report['r_primal'],report['r_dual'])<=1e-4
    wrong=point.copy();wrong[0]+=.1
    miss=comp.solve_instance(warm_start=(jnp.asarray(wrong),jnp.asarray(supplied)))
    assert miss.status!='solved'  # recovery must not erase stationarity error


def test_near_boundary_after_offset_cancellation_keeps_required_multiplier():
    point=np.array([1.,0.],np.float32)
    slack=np.array([2.**-20,0.],np.float32)
    normal=np.array([-1.,0.],np.float32)
    comp=family(SecondOrderCone(),point,-point-normal,offset=slack-point)
    sol=comp.solve_instance(warm_start=(jnp.asarray(point),jnp.asarray(normal)))
    # Relative-to-data interior guard does not erase a potentially active
    # normal just because the remaining slack is positive.
    np.testing.assert_array_equal(np.asarray(sol.y),normal)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_interior_recovery_does_not_hide_nonfinite_supplied_multiplier(bad):
    point=np.array([2.,.25],np.float32)
    comp=family(SecondOrderCone(),point,-point)
    sol=comp.solve_instance(warm_start=(jnp.asarray(point),jnp.asarray([bad,0.])))
    assert sol.status=='numerical_error'
    assert not np.isfinite(np.asarray(sol.y)).all()
