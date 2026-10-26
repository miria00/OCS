"""Batched reports preserve scalar KKT/guard/gap contracts at supplied points."""
from functools import lru_cache
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import ocs
from ocs.ir import Term, ProblemFamily
from ocs.model.compile import CompiledFamily


def compare(compiled, x, y=None, data=None):
    batch = jax.device_get(compiled.certificate_many(x, y, data))
    assert all(value.shape == (len(x),) for value in batch.values())
    for i in range(len(x)):
        row_data = None if data is None else jax.tree_util.tree_map(lambda a: a[i], data)
        scalar = compiled.certificate(x[i], None if y is None else y[i], row_data)
        assert set(scalar) == set(batch)
        for key, value in scalar.items():
            np.testing.assert_allclose(batch[key][i], value, rtol=3e-4, atol=3e-7,
                                       equal_nan=True, err_msg=key)
            if not key.startswith('r_') and key != 'gap':
                assert bool(batch[key][i]) == bool(value), key
    return batch


@lru_cache(None)
def diagonal_family(norm='relative', primal='nonneg', method='pdhg', constant=False):
    x = ocs.Variable(3)
    operator = ocs.Diagonal(jnp.array([1., 2., 4.]))
    offset = jnp.array([1., -2., 4.]) if constant else ocs.Parameter('b')
    terms = [ocs.least_squares(operator @ x-offset)]
    if primal == 'nonneg': terms.insert(0, ocs.nonneg(x))
    elif primal == 'quadratic': terms.insert(0, ocs.sum_squares(x))
    elif primal == 'missing':
        block = ocs.L1().with_linear_quadratic(e=jnp.array([0., 1., 1.]))
        terms.insert(0, Term(ocs.Identity(3), block))
    return ocs.minimize(terms).compile(equilibrate='scalar', max_blocks=0,
        residual_norm=norm, method=method, rho=.25 if method == 'cg' else 0.)


@pytest.mark.parametrize('norm', ['relative','iterate','data'])
@pytest.mark.parametrize('primal', ['nonneg','quadratic','absent','missing'])
def test_same_fields_values_and_gap_availability(norm, primal):
    c = diagonal_family(norm, primal)
    x = jnp.array([[0., 0., 0.], [1., 2., -1.], [1e-12, -2e-12, 3e-12],
                   [1e12, -2e12, 3e12]])
    y = jnp.array([[0., 0., 0.], [3., -1., .5], [2e-12, 0., -1e-12],
                   [2e12, 0., -1e12]])
    data = {'b': jnp.array([[0., 0., 0.], [2., -1., 3.], [0., 0., 0.], [1., 2., 3.]])}
    r = compare(c, x, y, data)
    if primal == 'missing':
        assert np.all(np.isnan(r['gap']))
        assert not np.any(r['gap_domains_finite'])
        assert not np.any(r['gap_exact'])
    if primal == 'nonneg':
        assert not np.any(r['gap_exact'])  # toleranced indicators are conservative


@pytest.mark.parametrize('part', ['x','y','b'])
@pytest.mark.parametrize('invalid', [np.nan,np.inf,-np.inf])
def test_nonfinite_row_does_not_contaminate_other_rows(part, invalid):
    c = diagonal_family()
    x, y, b = [jnp.zeros((3,3)) for _ in range(3)]
    values = {'x':x,'y':y,'b':b}
    values[part] = values[part].at[1,0].set(invalid)
    r = compare(c, values['x'], values['y'], {'b':values['b']})
    assert max(r['r_primal'][0], r['r_dual'][0]) == 0
    assert max(r['r_primal'][2], r['r_dual'][2]) == 0
    assert np.isinf(max(r['r_primal'][1], r['r_dual'][1]))


def test_independent_nnls_kkt_pair_and_supplied_bad_dual():
    c = diagonal_family()
    diag = np.array([1.,2.,4.], dtype=np.float32)
    b = np.array([[1.,-2.,3.], [-1.,3.,0.]], dtype=np.float32)
    x = np.maximum(b/diag, 0.)
    y = x*diag-b
    # Independently constructed exact NNLS solution: x>=0, gradient>=0,
    # and x*gradient=0. Perturb only y; diagnostics must not repair it.
    np.testing.assert_array_equal(x*(y*diag), 0.)
    assert np.all(y*diag >= 0.)
    r = compare(c, jnp.asarray(x), jnp.asarray(y), {'b':jnp.asarray(b)})
    assert np.max(np.maximum(r['r_primal'],r['r_dual'])) < 1e-6
    bad = jnp.asarray(y).at[0,0].add(.2)
    rb = compare(c, jnp.asarray(x), bad, {'b':jnp.asarray(b)})
    assert max(rb['r_primal'][0],rb['r_dual'][0]) > 1e-3
    assert max(rb['r_primal'][1],rb['r_dual'][1]) < 1e-6


@pytest.mark.parametrize('kind', ['exp','power'])
def test_cone_origins_and_material_feasibility_across_data_magnitudes(kind):
    xvar = ocs.Variable(3)
    cone = ocs.exp_cone if kind == 'exp' else ocs.pow_cone
    terms = [ocs.sum_squares(xvar), cone(ocs.Diagonal(jnp.ones(3)) @ xvar)]
    c = ocs.minimize(terms,cost=ocs.Parameter('c')).compile(equilibrate='scalar',max_blocks=0)
    normal = np.array([0.,0.,-1.] if kind == 'exp' else [-1.,-1.,0.], np.float32)
    scales = np.array([1e-6,1.,1e6], np.float32)[:,None]
    yy = np.concatenate([scales*normal, scales*normal])
    xx = np.concatenate([np.zeros((3,3),np.float32), .05*scales*normal])
    data = {'c':jnp.asarray(-yy)}
    r = compare(c,jnp.asarray(xx),jnp.asarray(yy),data)
    assert np.max(np.maximum(r['r_primal'][:3],r['r_dual'][:3])) < 1e-5
    assert np.all(np.maximum(r['r_primal'][3:],r['r_dual'][3:]) > 1e-3)
    assert not np.any(r['gap_exact'])


@pytest.mark.parametrize('method', ['pdhg','cg'])
@pytest.mark.parametrize('batch_size',[0,1,3])
def test_empty_batches_constant_data_and_jit_composition(method,batch_size):
    c = diagonal_family(primal='absent',method=method,constant=True)
    x = jnp.zeros((batch_size,3))
    y = None if method == 'cg' else jnp.zeros_like(x)
    out = c.certificate_many(x,y)
    assert all(isinstance(value,jax.Array) and value.shape == (batch_size,) for value in out.values())
    wrapped = jax.jit(lambda xx,yy: c.certificate_many(xx,yy))(x,y)
    # A wider JIT boundary can constant-fold the fixed datum while the
    # numerator is evaluated on device. Neither API promises bitwise reports
    # across compiler boundaries. Keep masks and flags exact, and check both
    # evaluations against an independent analytic answer for this fixture.
    epsilon = np.finfo(np.float32).eps
    for key in out:
        ordinary, outer_jit = np.asarray(out[key]), np.asarray(wrapped[key])
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(mask(ordinary), mask(outer_jit))
        if ordinary.dtype == np.bool_:
            np.testing.assert_array_equal(ordinary, outer_jit)
        else:
            np.testing.assert_allclose(outer_jit, ordinary, rtol=0.,
                                       atol=2*epsilon, equal_nan=True)
    # x=y=0 and b=(1,-2,4): PDHG r_dual=||b||/||b||=1;
    # primal objective is21/2, dual objective is0, so gap=21/23.
    # CG instead has r_primal=||A.T*b||/||A.T*b||=1 and no gap.
    expected = ({'r_primal':0., 'r_dual':1., 'gap':21./23.}
                if method == 'pdhg' else
                {'r_primal':1., 'r_dual':0., 'gap':np.nan})
    for evaluated in (out, wrapped):
        for key,value in expected.items():
            np.testing.assert_allclose(evaluated[key],np.full(batch_size,value),
                                       rtol=0.,atol=2*epsilon,equal_nan=True)
        if method == 'pdhg':
            np.testing.assert_array_equal(evaluated['gap_exact'],np.ones(batch_size,bool))
            np.testing.assert_array_equal(evaluated['gap_domains_finite'],np.ones(batch_size,bool))
    compare(c,x,y)


@pytest.mark.parametrize('invalid', [0.,np.nan,np.inf])
def test_cg_same_keys_and_independent_gradient_ratio(invalid):
    c = diagonal_family(primal='absent',method='cg')
    x = jnp.array([[0.,0.,0.],[1.,2.,3.],[invalid,1.,2.]])
    b = jnp.array([[0.,0.,0.],[2.,-1.,4.],[1.,2.,3.]])
    r = compare(c,x,data={'b':b})
    assert set(r) == {'r_primal','r_dual','gap'}
    assert np.all(np.isnan(r['gap']))
    assert r['r_primal'][0] == 0
    diag = np.array([1.,2.,4.]); point=np.array(x[1]); datum=np.array(b[1])
    grad = diag*(diag*point-datum)
    reference = np.linalg.norm(grad+.25*point)/max(np.linalg.norm(diag*datum),np.linalg.norm(grad),np.linalg.norm(.25*point))
    np.testing.assert_allclose(r['r_primal'][1],reference,rtol=2e-6)
    if not np.isfinite(invalid): assert np.isinf(r['r_primal'][2])


@pytest.mark.parametrize('method',['pdhg','cg'])
@pytest.mark.parametrize('bad', ['x_rank','x_width','data_scalar','data_count','y_count','y_width'])
def test_new_api_rejects_inconsistent_batch_axes(method,bad):
    if method == 'cg' and bad.startswith('y_'): pytest.skip('CG has no dual array')
    c = diagonal_family(primal='absent',method=method)
    x=y=jnp.zeros((2,3));data={'b':jnp.zeros((2,3))}
    if bad=='x_rank':x=jnp.zeros(3)
    if bad=='x_width':x=jnp.zeros((2,4))
    if bad=='data_scalar':data={'b':jnp.array(1.)}
    if bad=='data_count':data={'b':jnp.zeros((1,3))}
    if bad=='y_count':y=jnp.zeros((1,3))
    if bad=='y_width':y=jnp.zeros((2,4))
    with pytest.raises(ValueError):c.certificate_many(x,y,data)


def test_direct_constructor_without_device_report_fails_explicitly():
    c=CompiledFamily(None,lambda d:d,None,1,1,{'block':1})
    with pytest.raises(NotImplementedError):
        c.certificate_many(jnp.zeros((1,1)),jnp.zeros((1,1)))
