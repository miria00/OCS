"""Batch reporting must evaluate the same resolved instance as scalar solving."""
import numpy as np
import jax
import jax.numpy as jnp
import ocs
from ocs.ir import Term, ProblemFamily


def test_certificate_does_not_fuse_arbitrary_data_callback_arithmetic():
    # Legal finite float32 data with cancellation. Separate multiply/add round
    # 4097*4097 to16785408, then add-16785408 toproduce0. FMA may instead
    # produce1. Moving this callback into a larger jit changes the instance.
    def offset(data):
        return data['a']*data['b']+data['c']

    problem=ProblemFamily(n=1,terms=[Term(ocs.Dense(np.ones((1,1),np.float32)),
                                        ocs.SumSquares(),offset)])
    compiled=problem.compile(equilibrate='scalar',max_blocks=0)
    data={'a':jnp.asarray([[4097.],[2.]],jnp.float32),
          'b':jnp.asarray([[4097.],[3.]],jnp.float32),
          'c':jnp.asarray([[-16785408.],[-6.]],jnp.float32)}
    x=y=jnp.zeros((2,1),jnp.float32)
    resolved=jax.vmap(compiled._resolve)(data)
    np.testing.assert_array_equal(resolved[0],np.zeros((2,1)))
    batch=jax.device_get(compiled.certificate_many(x,y,data))
    for i in range(2):
        datum={k:v[i] for k,v in data.items()}
        solution=compiled.solve_instance(datum,warm_start=(x[i],y[i]))
        scalar=compiled.certificate(x[i],y[i],datum)
        assert solution.status=='solved' and solution.residual==0.
        assert scalar['r_primal']==scalar['r_dual']==scalar['gap']==0.
        for key,value in scalar.items():
            np.testing.assert_equal(batch[key][i],value,err_msg=key)
