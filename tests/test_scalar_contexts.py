"""Additional compiler contexts requested by independent review."""
import importlib
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from test_cold_sumsquares import pair, identical


@pytest.mark.parametrize('layout', ['dense','compact'])
@pytest.mark.parametrize('metric,adaptive', [('given',True),('scalar',False),('scalar',True)])
def test_scalar_equilibration_and_adaptive_scalar_omega_match_fallback(monkeypatch,layout,metric,adaptive):
    module = importlib.import_module('ocs.model.compile')
    original, observed = module.grouped_moreau_dual, []
    def observe(*args,**kwargs):
        observed.append(kwargs['compact_scalar_sumsquares'])
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'grouped_moreau_dual',observe)
    x = ocs.Variable(6)
    family = ocs.minimize([ocs.nonneg(x),ocs.least_squares(
        ocs.Dense(np.diag(np.linspace(.5,1.5,6)))@x-ocs.Parameter('b'))])
    equil = ('given',jnp.float32(.2),jnp.float32(.3)) if metric == 'given' else 'scalar'
    options = dict(equilibrate=equil,data_layout=layout,block=2,max_blocks=3,
                   tol=1e-5,adaptive_omega=adaptive,pdhg_variant='halpern')
    candidate = family.compile(**options)
    # In particular the original two GPU-failing scalar/False fixtures now
    # execute the predecessor's full-vector SumSquares dispatch.
    assert observed == [False]
    actual = module.grouped_moreau_dual
    def fallback(*args,**kwargs):
        kwargs['compact_scalar_sumsquares'] = False
        return actual(*args,**kwargs)
    monkeypatch.setattr(module,'grouped_moreau_dual',fallback)
    baseline = family.compile(**options)
    for data in [{'b':jnp.linspace(-1.,2.,6)},{'b':jnp.linspace(1.,-.3,6)}]:
        left,right=baseline.solve_instance(data),candidate.solve_instance(data)
        identical(left.x,right.x);identical(left.y,right.y)
        assert (left.status,left.iters,left.residual)==(right.status,right.iters,right.residual)


def test_multiple_contiguous_sumsquares_blocks_match_original_dispatch():
    _,base,candidate=pair([ocs.SumSquares(),ocs.SumSquares()],[(0,4),(4,8)],jnp.float32(.3))
    v=jnp.linspace(-1.,2.,8);sw=jnp.full(8,.3)
    identical(base(v,sw),candidate(v,sw))
