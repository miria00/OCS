"""Narrow scalar least-squares dispatch controls; no changed proximal formula."""
import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ocs
from ocs.prox.groups import build_groups, grouped_moreau_dual, _verified_scalar_groups

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# cold-start behaviour and the residual it reaches.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def pair(blocks, spans, sigma):
    groups = build_groups(blocks, spans)
    m = spans[-1][1]
    baseline = grouped_moreau_dual(groups, m, static_sigma=sigma)
    candidate = grouped_moreau_dual(groups, m, static_sigma=sigma,
                                   compact_scalar_sumsquares=True)
    return groups, jax.jit(baseline), jax.jit(candidate)


def identical(left, right):
    a, b = np.asarray(left), np.asarray(right)
    assert a.shape == b.shape and a.dtype == b.dtype
    # NaN payloads are not a public contract; finite entries must be bit exact.
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    np.testing.assert_array_equal(a[mask].view(np.uint32), b[mask].view(np.uint32))


@pytest.mark.parametrize('sigma', [.125, .3, 2.])
@pytest.mark.parametrize('omega', [.1, 1., 4.])
def test_scalar_steps_keep_original_moreau_bits(sigma, omega):
    sigma = jnp.asarray(sigma, jnp.float32)
    groups, base, candidate = pair([ocs.SumSquares()], [(0, 17)], sigma)
    assert _verified_scalar_groups(groups, 17, sigma, compact_scalar_sumsquares=True) == (True,)
    v = jnp.asarray(np.random.default_rng(10).normal(size=17), jnp.float32)
    sw = jnp.broadcast_to(sigma*jnp.float32(omega), v.shape)
    identical(base(v, sw), candidate(v, sw))
    np.testing.assert_allclose(candidate(v, sw), np.asarray(v)/(1+np.asarray(sw)),
                               rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf])
def test_invalid_static_metrics_fall_back_and_remain_nan(sigma):
    sigma = jnp.asarray(sigma, jnp.float32)
    groups, base, candidate = pair([ocs.SumSquares()], [(0, 5)], sigma)
    assert _verified_scalar_groups(groups, 5, sigma, compact_scalar_sumsquares=True) == (False,)
    v, sw = jnp.ones(5), jnp.broadcast_to(sigma, (5,))
    identical(base(v, sw), candidate(v, sw))
    assert np.isnan(np.asarray(candidate(v, sw))).all()


@pytest.mark.parametrize('omega', [0., -1., np.nan, np.inf])
def test_invalid_runtime_multiplier_still_reaches_metric_validation(omega):
    _, base, candidate = pair([ocs.SumSquares()], [(0, 5)], jnp.float32(.3))
    v, sw = jnp.ones(5), jnp.full(5, jnp.float32(.3)*jnp.float32(omega))
    identical(base(v, sw), candidate(v, sw))
    assert np.isnan(np.asarray(candidate(v, sw))).all()


@pytest.mark.parametrize('sigma', [np.finfo(np.float32).tiny,
                                  np.nextafter(np.float32(0), np.float32(1)),
                                  np.finfo(np.float32).max])
def test_extreme_scalar_reciprocal_behavior_is_unchanged(sigma):
    sigma = jnp.asarray(sigma, jnp.float32)
    _, base, candidate = pair([ocs.SumSquares()], [(0, 5)], sigma)
    v, sw = jnp.asarray([0., -1., 1., 1e-12, 1e12]), jnp.broadcast_to(sigma, (5,))
    identical(base(v, sw), candidate(v, sw))


@pytest.mark.parametrize('metric', [[.2]*6, [.2,.3,.4,.5,.6,.7], [.2,.3,0.,.5,.6,.7]])
def test_even_uniform_vector_metrics_retain_full_existing_contract(metric):
    sigma = jnp.asarray(metric, jnp.float32)
    groups, base, candidate = pair([ocs.SumSquares()], [(0, 6)], sigma)
    assert _verified_scalar_groups(groups, 6, sigma, compact_scalar_sumsquares=True) == (False,)
    identical(base(jnp.arange(6.), sigma), candidate(jnp.arange(6.), sigma))


@pytest.mark.parametrize('kind', ['subclass', 'instance_prox', 'extra_attribute', 'scaled', 'affine', 'quadratic'])
def test_custom_instances_and_calculus_wrappers_do_not_opt_in(kind):
    class Custom(ocs.SumSquares):
        pass
    if kind == 'subclass':
        block = Custom()
    elif kind == 'instance_prox':
        block = ocs.SumSquares()
        block.prox = lambda v, t: v/(2+t)
    elif kind == 'extra_attribute':
        block = ocs.SumSquares()
        block.user_tag = 'preserve-dispatch'
    elif kind == 'scaled':
        block = ocs.SumSquares().scaled(.3)
    elif kind == 'affine':
        block = ocs.SumSquares().precomposed(jnp.asarray([1.,2.,3.,4.]), .1)
    else:
        block = ocs.SumSquares().with_linear_quadratic(.1, jnp.asarray([.1,.2,.3,.4]))
    groups, base, candidate = pair([block], [(0, 4)], jnp.float32(.3))
    assert _verified_scalar_groups(groups, 4, jnp.float32(.3), compact_scalar_sumsquares=True) == (False,)
    identical(base(jnp.arange(4.), jnp.full(4,.3)), candidate(jnp.arange(4.), jnp.full(4,.3)))


def test_mixed_noncontiguous_groups_preserve_coordinates_and_scalar_tv():
    blocks = [ocs.SumSquares(), ocs.GroupL2(.2, 2), ocs.SumSquares()]
    spans = [(0, 4), (4, 8), (8, 12)]
    groups, base, candidate = pair(blocks, spans, jnp.float32(.3))
    assert _verified_scalar_groups(groups, 12, jnp.float32(.3), compact_scalar_sumsquares=True) == (True, True)
    v, sw = jnp.linspace(-2., 3., 12), jnp.full(12, .3)
    identical(base(v, sw), candidate(v, sw))


def test_default_callable_remains_arbitrary_vector_capable():
    groups = build_groups([ocs.SumSquares()], [(0, 4)])
    grouped_moreau_dual(groups, 4, static_sigma=jnp.float32(.3), compact_scalar_sumsquares=True)
    ordinary = jax.jit(grouped_moreau_dual(groups, 4))
    values, sw = jnp.arange(4.), jnp.asarray([.1,.2,.3,.4])
    np.testing.assert_allclose(ordinary(values, sw), np.asarray(values)/(1+np.asarray(sw)), atol=2e-7)


def test_lowered_validation_reduces_one_scalar_instead_of_full_measurement_vector():
    size = 8192
    groups = build_groups([ocs.SumSquares()], [(0, size)])
    def widths(enable):
        apply = grouped_moreau_dual(groups, size, static_sigma=jnp.float32(.3),
                                   compact_scalar_sumsquares=enable)
        jp = jax.make_jaxpr(apply)(jnp.ones(size), jnp.full(size,.3)).jaxpr
        return [int(np.prod(eqn.invars[0].aval.shape)) for eqn in jp.eqns
                if eqn.primitive.name == 'reduce_and']
    assert widths(False) == [size]
    assert widths(True) == [1]


@pytest.mark.parametrize('layout', ['dense', 'compact'])
@pytest.mark.parametrize('variant,dual_layout', [('standard','flat'),('standard','blocks'),('halpern','flat')])
def test_full_public_solver_matches_fallback_for_cold_and_warm_starts(monkeypatch, layout, dual_layout, variant):
    module = importlib.import_module('ocs.model.compile')
    original, observed = module.grouped_moreau_dual, []
    def observe(*args, **kwargs):
        observed.append(kwargs['compact_scalar_sumsquares'])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'grouped_moreau_dual', observe)
    x = ocs.Variable(9)
    family = ocs.minimize([ocs.nonneg(x),
        ocs.least_squares(ocs.Dense(np.diag(np.linspace(.5,1.5,9)))@x-ocs.Parameter('b')),
        ocs.group_l2(ocs.Identity(9)@x, lam=.02, group_size=3)])
    options = dict(equilibrate=('given',jnp.float32(.2),jnp.float32(.3)),
                   pdhg_variant=variant, data_layout=layout, dual_layout=dual_layout,
                   block=3, max_blocks=3, tol=1e-5, adaptive_omega=False)
    candidate = family.compile(**options)
    assert observed == [variant == 'standard']
    actual = module.grouped_moreau_dual
    def fallback(*args, **kwargs):
        kwargs['compact_scalar_sumsquares'] = False
        return actual(*args, **kwargs)
    monkeypatch.setattr(module, 'grouped_moreau_dual', fallback)
    baseline = family.compile(**options)
    data = {'b': jnp.linspace(-1.,2.,9)}
    for warm in [None, (jnp.linspace(.1,.9,9), jnp.linspace(-.2,.3,18))]:
        old = baseline.solve_instance(data, warm_start=warm)
        new = candidate.solve_instance(data, warm_start=warm)
        identical(old.x, new.x); identical(old.y, new.y)
        assert (old.status, old.iters, old.residual) == (new.status, new.iters, new.residual)
        left, right = baseline.certificate(old.x, old.y, data), candidate.certificate(new.x, new.y, data)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_equal(left[key], right[key])


@pytest.mark.parametrize('method', ['jacobi', 'osgm'])
def test_online_coordinate_adaptation_never_enables_new_specialization(monkeypatch, method):
    module = importlib.import_module('ocs.model.compile')
    actual, seen = module.grouped_moreau_dual, []
    def capture(*args, **kwargs):
        seen.append(kwargs['compact_scalar_sumsquares'])
        return actual(*args, **kwargs)
    monkeypatch.setattr(module, 'grouped_moreau_dual', capture)
    x = ocs.Variable(4)
    family = ocs.minimize([ocs.nonneg(x),ocs.least_squares(ocs.Dense(np.eye(4))@x-ocs.Parameter('b'))])
    result = family.compile(equilibrate='scalar', online_warmup=1, online_method=method,
        block=2, max_blocks=2).solve_instance({'b':jnp.asarray([.2,-1.,.4,1.])})
    assert seen == [False]
    assert np.isfinite(np.asarray(result.x)).all()
