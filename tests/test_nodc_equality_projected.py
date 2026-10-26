"""No-DC Fourier/TV imaging: the equality-projected witness route."""
import numpy as np
import jax.numpy as jnp
import pytest
import ocs
from ocs.solver import nodc_witness


def _family(idx, n=64, shape=(4, 4, 4)):
    x = ocs.Variable(n); F = ocs.FFTMask(idx, n); G = ocs.Gradient3D(*shape)
    return ocs.minimize([ocs.nonneg(x), ocs.least_squares(F @ x - ocs.Parameter('b')),
                         ocs.group_l2(G @ x, lam=.02, group_size=3)])


def _independent(idx, n, shape, b, y, lam=float(np.float32(.02))):
    """Route A certificate in NumPy float64: implicit correction bound through s_D."""
    m = len(idx); u = y[:2 * m].astype('d'); w = y[2 * m:].astype('d').reshape(*shape, 3)
    f = np.zeros(n, 'complex128'); f[idx] = u[:m] + 1j * u[m:]; q = np.fft.ifft(f, norm='ortho').real.reshape(shape)
    for k in range(3):
        lo = [slice(None)] * 3; hi = lo.copy(); lo[k] = slice(0, -1); hi[k] = slice(1, None); lo = tuple(lo); hi = tuple(hi)
        q[lo] -= w[..., k][lo]; q[hi] += w[..., k][lo]
    s_D = 2 * np.sin(np.pi / (2 * max(shape)))
    eps = np.finfo('d').eps
    eta = np.linalg.norm(q) + 8 * np.log2(n) * eps * np.linalg.norm(u) + 8 * eps * np.linalg.norm(w)
    peak = np.linalg.norm(w, axis=-1).max()
    d = -b.astype('d') @ u - .5 * (u @ u)
    return dict(q_mean=float(q.mean()), q_2=float(np.linalg.norm(q)), balls_with_correction=bool(peak + eta / s_D <= lam), d=float(d))


def test_route_eligibility_is_an_exact_mask_fact():
    n = 64
    comp_dc = _family(np.array([0, 3, 7, 11])).compile(max_blocks=0)
    comp_nodc = _family(np.array([2, 3, 7, 11])).compile(max_blocks=0)
    assert comp_dc.meta['equality_projected_route']['eligible'] is False
    assert 'DC' in comp_dc.meta['equality_projected_route']['reason']
    assert comp_nodc.meta['equality_projected_route']['eligible'] is True
    assert comp_nodc.meta['equality_projected_route']['representation'] == nodc_witness.REPRESENTATION


def test_nodc_bound_is_named_valid_and_independently_checked():
    n = 64; shape = (4, 4, 4); idx = np.array([2, 3, 7, 11]); m = len(idx)
    fam = _family(idx); comp = fam.compile(max_blocks=0)
    rng = np.random.default_rng(7)
    phantom = np.abs(rng.normal(size=n)).astype('f')
    F = ocs.FFTMask(idx, n)
    b = np.asarray(F.matvec(jnp.asarray(phantom))).astype('f')
    y = np.r_[rng.normal(size=2 * m), .01 * rng.normal(size=3 * n)].astype('f')
    x = np.maximum(phantom + .1 * rng.normal(size=n), 0).astype('f')
    report = comp.objective_bound(jnp.asarray(x), jnp.asarray(y), {'b': jnp.asarray(b)}, return_dual=True)
    assert report['objective_lower_bound_available']
    assert report['source'] == 'equality_projected'
    assert report['bound_kind'] == 'strict_dual'
    assert report['witness_representation'] == nodc_witness.REPRESENTATION
    assert report['arithmetic'] == 'mixed'
    route = report['equality_projected']
    assert route['valid'] and route['delta'] < 1e-9
    assert route['localized']['bound_kind'] == 'localized' and route['localized']['dual_feasible'] is False
    assert report['operator_applications'] >= route['operator_applications'] > 0
    yy = np.asarray(report['certifying_dual'])
    assert yy.dtype == np.float64
    ind = _independent(idx, n, shape, b, yy)
    assert ind['balls_with_correction']
    assert abs(ind['d'] - report['objective_lower_bound']) <= 1e-9 * max(1., abs(ind['d']))
    # the equality-projected value never exceeds the primal value at a feasible x
    assert report['objective_lower_bound'] <= report['objective_primal_value'] + 1e-9
    # certify-only (validate) on the returned float64 pair keeps the same semantics and moves nothing
    v = comp.validate(jnp.asarray(x), yy, {'b': jnp.asarray(b)})
    assert v['source'] == 'equality_projected' and v['equality_projected']['improved'] is False
    assert abs(v['objective_lower_bound'] - report['objective_lower_bound']) <= 1e-9 * max(1., abs(ind['d']))


def test_supplied_strict_witness_is_refused_without_route_when_disabled():
    n = 64; idx = np.array([2, 3, 7, 11]); m = len(idx)
    fam = _family(idx); comp = fam.compile(max_blocks=0)
    rng = np.random.default_rng(3)
    y = np.r_[rng.normal(size=2 * m), rng.normal(size=3 * n)].astype('f'); b = rng.normal(size=2 * m).astype('f')
    nodc_witness.configure(enabled=False)
    try:
        report = comp.objective_bound(jnp.ones(n), jnp.asarray(y), {'b': jnp.asarray(b)})
        # the generic routes may still find a trivial valid bound (the zero
        # ray); whatever they find, the named route is absent and not claimed
        assert 'equality_projected' not in report
        assert report['source'] != 'equality_projected'
        assert report['arithmetic'] == 'float32'
    finally:
        nodc_witness.configure(enabled=True)
    report = comp.objective_bound(jnp.ones(n), jnp.asarray(y), {'b': jnp.asarray(b)})
    assert report['source'] == 'equality_projected'


def test_small_live_solve_awards_solved_on_the_named_route():
    L = 6; n = L ** 3; shape = (L, L, L)
    idx = np.sort(np.random.default_rng(0).choice(np.arange(1, n), int(.4 * n), replace=False))
    assert idx[0] != 0
    x = ocs.Variable(n); F = ocs.FFTMask(idx, n); G = ocs.Gradient3D(*shape)
    ph = np.zeros(shape, 'f'); ph[1:4, 1:5, 2:5] = 1.
    b = np.asarray(F.matvec(jnp.asarray(ph.reshape(-1)))).astype('f')
    fam = ocs.minimize([ocs.nonneg(x), ocs.least_squares(F @ x - ocs.Parameter('b')), ocs.group_l2(G @ x, lam=.02, group_size=3)])
    comp = fam.compile(max_blocks=400, tol=1e-3)
    s = comp.solve_instance({'b': jnp.asarray(b)})
    assert s.status == 'solved'
    assert s.objective_bound_source == 'equality_projected'
    assert s.dual_source == 'certifying' and np.asarray(s.y).dtype == np.float64
    ind = _independent(idx, n, shape, b, np.asarray(s.y))
    assert ind['balls_with_correction']
    P = float(s.objective_primal_value)
    assert abs(P - ind['d']) / max(abs(P), abs(ind['d'])) <= 1e-3
