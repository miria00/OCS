"""Structural data facts must not imply unproved proximal/objective facts."""
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.ir import ProblemFamily,Term


@pytest.mark.parametrize('block',[
    ocs.L1(.3).precomposed(b=np.asarray([-1.,2.],np.float32)),
    ocs.SumSquares().precomposed(b=np.asarray([-1.,2.],np.float32)),
    ocs.Logistic(),
])
def test_absent_offset_keeps_nonzero_prox_gradient_and_value_anchor(block):
    p=ProblemFamily(2,[Term(ocs.Identity(2),ocs.SumSquares()),
                      Term(ocs.Dense(np.eye(2,dtype=np.float32)),block),
                      Term(ocs.Dense(np.diag([2.,3.]).astype(np.float32)),ocs.SumSquares(),lambda d:d['b'])])
    cs=[p.compile(equilibrate='scalar',pdhg_variant='standard',data_layout=l,
                  block=20,max_blocks=60,tol=1e-4) for l in ('compact','dense')]
    raw={'b':jnp.asarray([-.4,.8])}
    solutions=[c.solve_instance(raw) for c in cs]
    assert solutions[0].status==solutions[1].status
    assert solutions[0].iters==solutions[1].iters
    for key in ('x','y','residual'):
        np.testing.assert_allclose(getattr(solutions[0],key),getattr(solutions[1],key),rtol=5e-6,atol=5e-7)
    # The origin is not generally an optimum merely because l and c are absent.
    origin=[c.certificate(jnp.zeros(2),jnp.zeros(4),raw) for c in cs]
    assert max(origin[0]['r_primal'],origin[0]['r_dual'])>1e-3
    for key in origin[0]:
        np.testing.assert_allclose(origin[0][key],origin[1][key],rtol=2e-6,atol=2e-7,equal_nan=True)


@pytest.mark.parametrize('slot',['offset','cost'])
def test_explicit_zero_callback_remains_dynamic_and_nonfinite_checked(slot):
    p=ProblemFamily(1,[Term(ocs.Identity(1),ocs.NonNeg()),
                      Term(ocs.Dense(np.ones((1,1))),ocs.SumSquares(),
                           (lambda d:d['value']) if slot=='offset' else None)],
                    cost_fn=(lambda d:d['value']) if slot=='cost' else None)
    c=p.compile(equilibrate='scalar',data_layout='compact',max_blocks=0)
    assert c.solve_instance({'value':jnp.zeros(1)}).status=='solved'
    assert c.solve_instance({'value':jnp.full(1,jnp.nan)}).status=='numerical_error'
    assert c.solve_instance({'value':jnp.full(1,jnp.inf)}).status=='numerical_error'
