"""Public option validation and analytic solves for standard-loop unrolling."""
import numpy as np
import jax.numpy as jnp
import pytest
import ocs


def family():
    x=ocs.Variable(3)
    k=ocs.Dense(jnp.diag(jnp.array([1.,2.,3.])))
    return ocs.minimize([ocs.nonneg(x),ocs.least_squares(k@x-ocs.Parameter('b'))])


@pytest.mark.parametrize('value',[True,False,0,-1,1.0,4.0,np.nan,np.inf,'4',None])
def test_invalid_loop_unroll_rejected(value):
    with pytest.raises(ValueError,match='loop_unroll must be a positive integer'):
        family().compile(loop_unroll=value,pdhg_variant='standard')


@pytest.mark.parametrize('kwargs',[
    {},{'pdhg_variant':'halpern'},{'method':'cg'},
    {'pdhg_variant':'standard','method':'cg'},
    {'pdhg_variant':'standard','online_warmup':2},
    {'pdhg_variant':'standard','adaptive_omega':True},
])
def test_nondefault_loop_unroll_rejects_unsupported_engines(kwargs):
    with pytest.raises(ValueError,match='nondefault loop_unroll requires standard PDHG'):
        family().compile(loop_unroll=4,**kwargs)


@pytest.mark.parametrize('block',[1,3,5,100])
@pytest.mark.parametrize('unroll',[1,4,np.int64(7)])
def test_public_unrolled_solve_has_analytic_answer_and_checked_pair(block,unroll):
    compiled=family().compile(loop_unroll=unroll,pdhg_variant='standard',
        equilibrate='scalar',block=block,max_blocks=500,tol=1e-5)
    data={'b':jnp.array([-1.,2.,6.])}
    result=compiled.solve_instance(data)
    assert result.status=='solved'
    np.testing.assert_allclose(result.x,[0.,1.,2.],rtol=3e-5,atol=1e-6)
    report=compiled.certificate(result.x,result.y,data)
    assert max(report['r_primal'],report['r_dual'])<=1e-5
    np.testing.assert_allclose(result.residual,max(report['r_primal'],report['r_dual']),rtol=2e-6,atol=1e-8)
    assert compiled.meta['loop_unroll']==int(unroll)
    assert result.iters%block==0


@pytest.mark.parametrize('variant',['halpern','standard'])
def test_default_loop_unroll_is_one(variant):
    compiled=family().compile(pdhg_variant=variant,equilibrate='scalar',max_blocks=0)
    assert compiled.meta['loop_unroll']==1
