"""PiecewiseLinearConvex: the conjugate and domain contract (0.3.5; order decided on the bits, 0.3.6).

Per coordinate g(u) = max_j (s_j u + r_j).  Its conjugate g^*(y) = sup_u (y u -
g(u)) is finite exactly on [s_min, s_max], where it is the lower convex envelope
of the points (s_j, -r_j).  Every reference below is independent of the block's
evaluation and runs in float64 on the represented float32 data:

  * ``kink_reference``: the supremum of the concave piecewise-linear y u - g(u)
    is attained at a kink of g or, at an end of the slope range, in the limit
    u -> -inf / +inf; evaluate y u - g(u) at every pairwise intersection
    u_jk = (r_k - r_j)/(s_j - s_k), s_j != s_k, add the two end limits, take
    the maximum;
  * ``lp_reference``: the linear program over convex combinations
    min -r^T lam  s.t.  s^T lam = y, 1^T lam = 1, lam >= 0  (HiGHS);
  * ``grid_reference``: brute force on a fine grid of u, a lower estimate.

The block's value must never lie below the exact conjugate (its rounding is
conservative: the dual value built from it is never overstated) and must agree
with it to float32 resolution.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ocs
from ocs.prox.library import PiecewiseLinearConvex
from ocs.prox.groups import build_groups, grouped_moreau_dual

F32 = np.float32
EPS32 = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------
# Independent float64 references
# ---------------------------------------------------------------------------
def _g64(s, r, u):
    u = np.asarray(u, dtype=np.float64)
    return np.max(np.multiply.outer(u, s) + r, axis=-1)


def kink_reference(s, r, y):
    """float64 sup_u (y u - g(u)) per coordinate (+inf outside [s_min, s_max])."""
    s = np.asarray(s, np.float64); r = np.asarray(r, np.float64)
    y = np.atleast_1d(np.asarray(y, np.float64))
    lo, hi = s.min(), s.max()
    kinks = [(r[k] - r[j]) / (s[j] - s[k])
             for j in range(len(s)) for k in range(len(s)) if s[j] != s[k]]
    kinks = np.asarray(kinks, np.float64)
    out = np.full(y.shape, np.inf)
    for i, yi in enumerate(y):
        if not np.isfinite(yi) or yi < lo or yi > hi:
            continue
        best = -np.inf
        if kinks.size:
            best = np.max(yi * kinks - _g64(s, r, kinks))
        if yi == lo:
            best = max(best, -r[s == lo].max())
        if yi == hi:
            best = max(best, -r[s == hi].max())
        out[i] = best
    return out


def lp_reference(s, r, y):
    linprog = pytest.importorskip("scipy.optimize").linprog
    s = np.asarray(s, np.float64); r = np.asarray(r, np.float64)
    out = []
    for yi in np.atleast_1d(np.asarray(y, np.float64)):
        res = linprog(-r, A_eq=np.vstack([s, np.ones_like(s)]), b_eq=[yi, 1.0],
                      bounds=[(0, None)] * len(s), method="highs")
        out.append(res.fun if res.status == 0 else np.inf)
    return np.asarray(out)


def grid_reference(s, r, y, width=None, points=400001):
    if width is None:
        # wide enough to contain every kink, so the endpoint suprema are reached
        s64, r64 = np.asarray(s, np.float64), np.asarray(r, np.float64)
        kinks = [abs((r64[k] - r64[j]) / (s64[j] - s64[k])) for j in range(len(s64))
                 for k in range(len(s64)) if s64[j] != s64[k]]
        width = 2.0 * max(kinks, default=0.0) + 2.0
    u = np.linspace(-width, width, points)
    gu = _g64(s, r, u)
    return np.asarray([np.max(yi * u - gu) for yi in np.atleast_1d(np.asarray(y, np.float64))])


def per_coordinate(block, y):
    """One-coordinate conjugate values, through jit and vmap (the batched route)."""
    evaluate = jax.jit(jax.vmap(lambda v: block.conj_value(v[None])))
    return np.asarray(evaluate(jnp.asarray(np.asarray(y, F32))), np.float64)


def random_instance(rng, pieces):
    s = rng.normal(size=pieces).astype(F32)
    r = rng.normal(size=pieces).astype(F32)
    return s, r


def domain_points(rng, s, count):
    lo, hi = float(np.min(s)), float(np.max(s))
    y = rng.uniform(lo, hi, size=count).astype(F32)
    y = np.clip(y, F32(lo), F32(hi))
    return np.concatenate([y, np.asarray([lo, hi], F32), np.asarray(s, F32)])


def assert_conservative_and_tight(got, ref, scale):
    got = np.asarray(got, np.float64); ref = np.asarray(ref, np.float64)
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
    # Never below the exact conjugate (float64 reference error ~1e-15 scale).
    assert np.all(got >= ref - 1e-12 * scale), np.max(ref - got)
    # Within float32 resolution above it.
    assert np.all(got - ref <= 8 * EPS32 * scale), np.max(got - ref)


# ---------------------------------------------------------------------------
# 1. Randomized instances against the independent references
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pieces", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conjugate_matches_kink_lp_and_grid_references(pieces, seed):
    rng = np.random.default_rng(1000 * pieces + seed)
    s, r = random_instance(rng, pieces)
    block = PiecewiseLinearConvex(s, r)
    y = domain_points(rng, s, 24)
    got = per_coordinate(block, y)
    ref = kink_reference(s, r, y)
    scale = 1.0 + np.max(np.abs(r)) + np.max(np.abs(s))
    assert_conservative_and_tight(got, ref, scale)
    lp = lp_reference(s, r, y)
    np.testing.assert_allclose(lp, ref, rtol=1e-7, atol=1e-7 * scale)
    if pieces > 1:
        grid = grid_reference(s, r, y)
        # A grid lower estimate: never above the supremum, close to it.
        assert np.all(grid <= ref + 1e-9 * scale)
        assert np.all(ref - grid <= 1e-2 * scale)


@pytest.mark.parametrize("seed", [3, 4])
def test_vector_value_is_the_conservative_sum(seed):
    rng = np.random.default_rng(seed)
    s, r = random_instance(rng, 6)
    block = PiecewiseLinearConvex(s, r)
    y = domain_points(rng, s, 5000)
    total = float(block.conj_value(jnp.asarray(y)))
    ref = kink_reference(s, r, y)
    exact = float(np.sum(ref))
    magnitude = float(np.sum(np.abs(ref)))
    assert total >= exact - 1e-12 * magnitude
    assert total - exact <= 4 * EPS32 * (abs(exact) + magnitude)
    assert block.conj_value(jnp.asarray(y)).dtype == jnp.float32
    assert not jax.config.x64_enabled


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------
def test_single_piece_is_a_linear_function_with_a_point_domain():
    block = PiecewiseLinearConvex([0.7], [-0.2])
    s = F32(0.7)
    assert bool(block.conj_in_domain(jnp.asarray([s, s])))
    assert float(block.conj_value(jnp.asarray([s]))) == float(-F32(-0.2))
    for bad in (np.nextafter(s, F32(np.inf)), np.nextafter(s, F32(-np.inf)), F32(0.)):
        assert not bool(block.conj_in_domain(jnp.asarray([bad])))
        assert np.isinf(float(block.conj_value(jnp.asarray([bad]))))
    assert float(block.conj_domain_radius(jnp.asarray([s]))) == 1.0
    assert float(block.conj_domain_radius(jnp.asarray([0.35], jnp.float32))) == pytest.approx(2.0)
    np.testing.assert_array_equal(
        np.asarray(block.conj_domain_project(jnp.asarray([-3., 0.7, 9.], jnp.float32))),
        np.full(3, s))


def test_repeated_slopes_keep_the_largest_intercept():
    # g(u) = max(u + 0.5, u - 3, -u, -u + 0.25)  ->  pieces (1, .5) and (-1, .25) matter.
    s = np.asarray([1., 1., -1., -1.], F32); r = np.asarray([.5, -3., 0., .25], F32)
    block = PiecewiseLinearConvex(s, r)
    assert float(block.conj_value(jnp.asarray([1.], jnp.float32))) == -0.5
    assert float(block.conj_value(jnp.asarray([-1.], jnp.float32))) == -0.25
    y = np.linspace(-1, 1, 41).astype(F32)
    assert_conservative_and_tight(per_coordinate(block, y), kink_reference(s, r, y), 2.0)
    reduced = PiecewiseLinearConvex([1., -1.], [.5, .25])
    np.testing.assert_array_equal(per_coordinate(block, y), per_coordinate(reduced, y))


def test_endpoints_are_the_limits_at_infinity():
    # At y = s_min the supremum is approached as u -> -inf: -max{r_j : s_j = s_min}.
    s = np.asarray([-2., .5, 3., -2., 3.], F32); r = np.asarray([1., 0., -1., -4., 2.], F32)
    block = PiecewiseLinearConvex(s, r)
    assert float(block.conj_value(jnp.asarray([-2.], jnp.float32))) == -1.0
    assert float(block.conj_value(jnp.asarray([3.], jnp.float32))) == -2.0
    # The grid supremum creeps toward these values from below as the grid widens.
    for y, limit in ((-2., -1.), (3., -2.)):
        assert grid_reference(s, r, [y], width=1e3, points=20001)[0] <= limit + 1e-9
        assert kink_reference(s, r, [y])[0] == pytest.approx(limit, abs=1e-12)


def test_unsorted_pieces_give_identical_values():
    rng = np.random.default_rng(7)
    s, r = random_instance(rng, 7)
    y = domain_points(rng, s, 64)
    base = PiecewiseLinearConvex(s, r)
    for trial in range(3):
        order = rng.permutation(7)
        shuffled = PiecewiseLinearConvex(s[order], r[order])
        np.testing.assert_array_equal(per_coordinate(base, y), per_coordinate(shuffled, y))
        assert float(base.conj_value(jnp.asarray(y))) == float(shuffled.conj_value(jnp.asarray(y)))


def test_redundant_pieces_never_change_the_conjugate():
    # |u| plus pieces that are never active (below the max everywhere).
    s = np.asarray([-1., 1.], F32); r = np.asarray([0., 0.], F32)
    padded_s = np.asarray([-1., 0., 1., .5, -.25], F32)
    padded_r = np.asarray([0., -10., 0., -2., -3.], F32)
    y = np.linspace(-1, 1, 81).astype(F32)
    base = PiecewiseLinearConvex(s, r)
    padded = PiecewiseLinearConvex(padded_s, padded_r)
    np.testing.assert_array_equal(per_coordinate(base, y), per_coordinate(padded, y))
    assert np.all(per_coordinate(base, y) == 0.0)        # f = |u|, f^* = indicator of [-1, 1]
    u = np.linspace(-3, 3, 61)
    np.testing.assert_array_equal(_g64(padded_s, padded_r, u), _g64(s, r, u))


def test_closed_domain_has_no_slack():
    block = PiecewiseLinearConvex([-.25, .5, .75], [0., -.1, -.3])
    lo, hi = F32(-.25), F32(.75)
    y = jnp.asarray([lo, hi], jnp.float32)
    assert bool(block.conj_in_domain(y)) and np.isfinite(float(block.conj_value(y)))
    for bad in (np.nextafter(lo, F32(-np.inf)), np.nextafter(hi, F32(np.inf)), np.nan, np.inf, -np.inf):
        v = jnp.asarray([.1, bad], jnp.float32)
        assert not bool(block.conj_in_domain(v))
        assert np.isinf(float(block.conj_value(v))) and float(block.conj_value(v)) > 0


# ---------------------------------------------------------------------------
# 3. Radius and projection
# ---------------------------------------------------------------------------
def _brute_radius(s, y, scale=4.0, points=200001):
    lo, hi = float(np.min(s)), float(np.max(s))
    a = np.linspace(0.0, scale, points)
    inside = np.all((np.multiply.outer(a, y) >= lo) & (np.multiply.outer(a, y) <= hi), axis=1)
    return a[inside].max() if inside.any() else None


def test_radius_when_zero_is_inside_the_domain():
    s = np.asarray([-1., .5, 2.], F32)
    block = PiecewiseLinearConvex(s, np.zeros(3, F32))
    y = jnp.asarray([.5, -.25], jnp.float32)
    a = float(block.conj_domain_radius(y))
    assert a == pytest.approx(4.0)
    assert bool(block.conj_in_domain(F32(a * (1 - 1e-6)) * y))
    assert not bool(block.conj_in_domain(F32(a * (1 + 1e-5)) * y))
    assert np.isinf(float(block.conj_domain_radius(jnp.zeros(3))))
    assert float(block.conj_domain_radius(jnp.asarray([np.nan, 1.], jnp.float32))) == 0.0


def test_radius_when_zero_lies_outside_the_domain():
    # dom g^* = [1, 3]: scaling y toward the origin leaves the domain.
    s = np.asarray([1., 3., 2.], F32)
    block = PiecewiseLinearConvex(s, np.asarray([0., -2., -.5], F32))
    y = jnp.asarray([2., 1.], jnp.float32)
    a = float(block.conj_domain_radius(y))
    assert a == pytest.approx(1.5)
    assert a == pytest.approx(_brute_radius(s, np.asarray(y)), abs=1e-4)
    assert bool(block.conj_in_domain(F32(a) * y))
    assert not bool(block.conj_in_domain(0.5 * y))        # [0, a] is NOT admissible
    assert not bool(block.conj_in_domain(jnp.zeros(2)))
    # No admissible scaling: a wrong sign, a zero coordinate, an empty interval.
    for bad in ([1., -1.], [1., 0.], [10., 1.], [0., 0.]):
        assert float(block.conj_domain_radius(jnp.asarray(bad, jnp.float32))) == 0.0
    negative = PiecewiseLinearConvex([-3., -1.], [0., 0.])
    assert float(negative.conj_domain_radius(jnp.asarray([-2., -1.], jnp.float32))) == pytest.approx(1.5)
    assert float(negative.conj_domain_radius(jnp.asarray([2., -1.], jnp.float32))) == 0.0


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_radius_matches_brute_force_on_random_directions(seed):
    rng = np.random.default_rng(seed)
    s, r = random_instance(rng, 4)
    block = PiecewiseLinearConvex(s, r)
    for _ in range(20):
        y = rng.normal(size=3).astype(F32)
        a = float(block.conj_domain_radius(jnp.asarray(y)))
        brute = _brute_radius(s, y, scale=max(4.0, 2 * a if np.isfinite(a) else 4.0))
        if brute is None:
            assert a == 0.0
        else:
            assert a == pytest.approx(brute, rel=1e-3, abs=1e-4)
            assert bool(block.conj_in_domain(F32(a * (1 - 1e-6)) * jnp.asarray(y)))


def test_projection_is_the_clip_and_lands_in_the_domain():
    rng = np.random.default_rng(5)
    s, r = random_instance(rng, 5)
    block = PiecewiseLinearConvex(s, r)
    y = (3 * rng.normal(size=200)).astype(F32)
    p = np.asarray(block.conj_domain_project(jnp.asarray(y)))
    np.testing.assert_array_equal(p, np.clip(y, s.min(), s.max()))
    assert bool(block.conj_in_domain(jnp.asarray(p)))
    np.testing.assert_array_equal(np.asarray(block.conj_domain_project(jnp.asarray(p))), p)
    np.testing.assert_array_equal(np.asarray(block.conj_domain_project_exact(jnp.asarray(y))), p)
    assert block.bound_capable and not block.is_indicator


def test_wrappers_forward_the_contract():
    s = np.asarray([-.5, .25, 1.5], F32); r = np.asarray([.2, 0., -.7], F32)
    base = PiecewiseLinearConvex(s, r)
    a, b, c = -2., .25, 3.
    wrapped = base.scaled(c).precomposed(a=a, b=b)
    y = np.asarray([-.3, .1, 1.2], F32)
    yy = jnp.asarray(y) * a * c
    expected = c * float(base.conj_value(jnp.asarray(y))) + float(jnp.sum((yy / a) * b))
    assert wrapped.bound_capable and bool(wrapped.conj_in_domain(yy))
    assert float(wrapped.conj_value(yy)) == pytest.approx(expected, rel=2e-6, abs=2e-6)
    shifted = base.with_linear_quadratic(d=.125)
    assert shifted.bound_capable
    assert bool(shifted.conj_in_domain(base.conj_domain_project(jnp.asarray(5 * y)) + .125))
    assert float(shifted.conj_value(jnp.asarray(y) + .125)) == float(base.conj_value(jnp.asarray(y)))


# ---------------------------------------------------------------------------
# 4. Proximal identity, Moreau decomposition, fixed points
# ---------------------------------------------------------------------------
def _conjugate_prox64(s, r, v, sigma, iters=200):
    """float64 argmin_w sigma g^*(w) + (w - v)^2 / 2 on [s_min, s_max] (golden section)."""
    lo0, hi0 = float(np.min(s)), float(np.max(s))
    out = []
    for vi, si in zip(v, sigma):
        lo, hi = lo0, hi0
        phi = (np.sqrt(5) - 1) / 2
        f = lambda w: si * kink_reference(s, r, [w])[0] + .5 * (w - vi) ** 2
        for _ in range(iters):
            if hi - lo <= 1e-15 * (1 + abs(lo) + abs(hi)):
                break
            a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
            if f(a) <= f(b):
                hi = b
            else:
                lo = a
        out.append(.5 * (lo + hi))
    return np.asarray(out)


@pytest.mark.parametrize("seed", [21, 22])
def test_prox_moreau_and_fixed_points_agree_with_the_conjugate(seed):
    rng = np.random.default_rng(seed)
    s = np.asarray([-1.5, -.25, .5, 2.], F32); r = np.asarray([.3, -.1, 0., -1.2], F32)
    order = rng.permutation(4)
    s, r = s[order], r[order]
    block = PiecewiseLinearConvex(s, r)
    n = 24
    v = (2 * rng.normal(size=n)).astype(F32)
    t = rng.uniform(.2, 3., size=n).astype(F32)
    # Primal prox and its subgradient w = (v - u)/t in d g(u).
    u = np.asarray(block.prox(jnp.asarray(v), jnp.asarray(t)), np.float64)
    w = (v.astype(np.float64) - u) / t.astype(np.float64)
    assert np.all(w >= s.min() - 1e-5) and np.all(w <= s.max() + 1e-5)
    w32 = np.clip(w, s.min(), s.max()).astype(F32)
    # Fenchel-Young equality g(u) + g^*(w) = u w, per coordinate, with the block's conjugate.
    gu = _g64(s, r, u)
    conj = per_coordinate(block, w32)
    defect = gu + conj - u * w32.astype(np.float64)
    assert np.all(defect >= -1e-5) and np.all(np.abs(defect) <= 1e-4 * (1 + np.abs(gu) + np.abs(u)))
    # Moreau: prox_{sigma f^*}(z) through the solver's grouped dual machinery,
    # against an independent float64 minimization of sigma g^* + |w - z|^2 / 2.
    sigma = (1 / t).astype(F32)
    groups = build_groups([block], [(0, n)])
    z = (v / t).astype(F32)
    dual = np.asarray(grouped_moreau_dual(groups, n)(jnp.asarray(z), jnp.asarray(sigma)), np.float64)
    ref = _conjugate_prox64(s, r, z.astype(np.float64), sigma.astype(np.float64))
    np.testing.assert_allclose(dual, ref, atol=2e-5)
    # Moreau decomposition z = prox_{f/sigma... } : v = prox_{tf}(v) + t prox_{f^*/t}(v/t).
    moreau = np.abs(u + t * dual - v) / (1 + np.abs(v))
    assert np.all(moreau <= 2e-5), moreau.max()
    np.testing.assert_allclose(dual, w, atol=2e-5)
    # Fixed points: u = prox_{tf}(u + t w)  and  w = prox_{sigma f^*}(w + sigma u).
    back = np.asarray(block.prox(jnp.asarray((u + t * w).astype(F32)), jnp.asarray(t)), np.float64)
    fixed = np.abs(back - u) / (1 + np.abs(u))
    assert np.all(fixed <= 2e-5), fixed.max()
    again = np.asarray(grouped_moreau_dual(groups, n)(jnp.asarray((w32 + sigma * u).astype(F32)),
                                                      jnp.asarray(sigma)), np.float64)
    np.testing.assert_allclose(again, w32, atol=2e-5)


# ---------------------------------------------------------------------------
# 5. A family with a PWL block, solved under criterion (7)
# ---------------------------------------------------------------------------
def _family(slopes, intercepts, seed, m=40, n=8):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)).astype(F32)
    b = (A @ rng.normal(size=n) + .3 * rng.normal(size=m)).astype(F32)
    d = F32(np.sqrt(.05)); mu = float(d) ** 2
    x = ocs.Variable(n)
    fam = ocs.minimize([ocs.sum_squares(ocs.Diagonal(np.full(n, d, F32)) @ x),
                        ocs.pwl(ocs.Dense(-A) @ x + ocs.Parameter("b"), slopes, intercepts)])
    return fam, A, b, mu


def _same_pair64(A, b, mu, s, r, x, y):
    """Independent float64 criterion (7) at the returned pair (x, y)."""
    xx, yy = np.asarray(x, np.float64), np.asarray(y, np.float64)
    A64, b64 = A.astype(np.float64), b.astype(np.float64)
    P = np.sum(_g64(np.asarray(s, np.float64), np.asarray(r, np.float64), b64 - A64 @ xx)) + .5 * mu * (xx @ xx)
    adj = A64.T @ yy
    D = b64 @ yy - np.sum(kink_reference(s, r, yy)) - .5 / mu * (adj @ adj)
    domain = bool(np.all((yy >= np.min(s)) & (yy <= np.max(s))))
    discrepancy = abs(P - D) / max(abs(P), abs(D))
    return P, D, domain, discrepancy


@pytest.mark.parametrize("case", ["zero_inside", "zero_outside"])
def test_pwl_family_is_solved_under_criterion_7_and_rechecks(case):
    if case == "zero_inside":
        # unsorted, a duplicate slope and a never-active piece
        s = np.asarray([.75, -1., .25, -1., 0.], F32); r = np.asarray([-.5, 0., 0., -2., -5.], F32)
    else:
        # increasing cost: dom g^* = [.5, 2] excludes 0
        s = np.asarray([2., .5], F32); r = np.asarray([-1., 0.], F32)
    fam, A, b, mu = _family(s, r, seed=31 if case == "zero_inside" else 32)
    tol = 1e-3
    comp = fam.compile(tol=tol, max_blocks=3000, block=100)
    assert comp.meta["objective_bound_capable"]
    data = {"b": jnp.asarray(b)}
    sol = comp.solve_instance(data)
    assert sol.status == "solved", (sol.status, sol.termination, sol.accuracy_detail)
    assert sol.x.dtype == sol.y.dtype == jnp.float32
    # Recovery-off validation of the returned pair.
    report = comp.objective_bound(sol.x, sol.y, data, recover=False)
    assert report["objective_lower_bound_available"]
    assert comp.acceptance(report)["target_met"]
    # Independent float64 same-pair check.
    assert comp.meta.get("folded_diagonal_quadratic") and comp.m == A.shape[0]
    P, D, domain, discrepancy = _same_pair64(A, b, mu, s, r, sol.x, sol.y)
    assert domain and discrepancy <= tol, (P, D, discrepancy)
    assert report["objective_lower_bound"] <= D + 1e-6 * abs(D)     # never above the f64 value
    # Batched route (the scoped float64 assessment under vmap), changed data.
    b2 = (b + .1 * np.random.default_rng(9).normal(size=b.shape)).astype(F32)
    lanes = comp.solve_many_results({"b": jnp.stack([jnp.asarray(b), jnp.asarray(b2)])})
    for lane, bb in zip(lanes, (b, b2)):
        assert lane.status == "solved"
        rep = comp.objective_bound(lane.x, lane.y, {"b": jnp.asarray(bb)}, recover=False)
        assert comp.acceptance(rep)["target_met"]
        _P, _D, dom, disc = _same_pair64(A, bb, mu, s, r, lane.x, lane.y)
        assert dom and disc <= tol
    assert not jax.config.x64_enabled


# ---------------------------------------------------------------------------
# 6. Order decided on the bits, not on a flushing comparison (0.3.6)
# ---------------------------------------------------------------------------
# XLA:CPU reads subnormal operands as zero, and not uniformly between eager
# and jitted code.  0.3.5's float comparison therefore admitted y = -1.4e-45
# with s_min = 0 (eagerly) and returned the value -0.  Every case below must
# hold on every backend and in both modes.
NEG_SUB, POS_SUB = F32(-1.4e-45), F32(1.4e-45)


def _bits(v):
    return np.asarray(v, F32).view(np.uint32)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_subnormal_below_a_zero_endpoint_is_outside(mode):
    wrap = (lambda f: f) if mode == "eager" else jax.jit
    block = PiecewiseLinearConvex([0., 1., 2.], [0., -.5, -2.])         # s_min = 0
    inside, value = wrap(block.conj_in_domain), wrap(block.conj_value)
    assert not bool(inside(jnp.asarray([NEG_SUB])))
    assert not bool(inside(jnp.asarray([.5, NEG_SUB], jnp.float32)))
    assert float(value(jnp.asarray([NEG_SUB]))) == np.inf
    for y in (F32(0.), F32(-0.), POS_SUB, F32(2.)):
        assert bool(inside(jnp.asarray([y]))), y
    # g^*(y) = y / 2 near 0: the value at the positive subnormal is never below it
    assert float(value(jnp.asarray([POS_SUB]))) >= 0.5 * float(POS_SUB)
    assert float(value(jnp.asarray([F32(0.)]))) == 0.0                  # exact stays exact
    p = wrap(block.conj_domain_project)(jnp.asarray([NEG_SUB, POS_SUB, F32(-1.), F32(3.)]))
    assert bool(block.conj_in_domain(p))
    # an inside subnormal stays put, or (where the clip flushes it) moves to 0,
    # which is inside too; the outside points take the endpoints
    assert int(_bits(p)[1]) in (int(_bits(POS_SUB)), 0)
    assert int(_bits(p)[0]) in (0, 0x80000000)
    assert float(p[2]) == 0.0 and float(p[3]) == 2.0


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_subnormal_endpoint_is_kept_exactly(mode):
    wrap = (lambda f: f) if mode == "eager" else jax.jit
    block = PiecewiseLinearConvex([POS_SUB, 1.], [0., -.5])            # s_min = +1.4e-45
    inside, value = wrap(block.conj_in_domain), wrap(block.conj_value)
    for y in (F32(0.), F32(-0.), NEG_SUB):
        assert not bool(inside(jnp.asarray([y]))), y
        assert float(value(jnp.asarray([y]))) == np.inf
    assert bool(inside(jnp.asarray([POS_SUB])))
    assert np.isfinite(float(value(jnp.asarray([POS_SUB]))))
    p = wrap(block.conj_domain_project)(jnp.asarray([F32(0.), NEG_SUB, F32(-1.), F32(5.)]))
    assert list(_bits(p)[:3]) == [int(_bits(POS_SUB))] * 3 and float(p[3]) == 1.0
    assert bool(block.conj_in_domain(p))


def test_order_keys_agree_with_the_float_order_on_normal_data():
    rng = np.random.default_rng(21)
    for _ in range(5):
        s, r = random_instance(rng, 4)
        block = PiecewiseLinearConvex(s, r)
        lo, hi = s.min(), s.max()
        y = np.r_[rng.uniform(lo - 1, hi + 1, 50), lo, hi, np.nextafter(lo, F32(-np.inf)),
                  np.nextafter(hi, F32(np.inf)), 0., -0.].astype(F32)
        for v in y:
            assert bool(block.conj_in_domain(jnp.asarray([v]))) == bool(lo <= v <= hi), v
        np.testing.assert_array_equal(np.asarray(block.conj_domain_project(jnp.asarray(y))),
                                      np.clip(y, lo, hi))


def test_wider_dtype_with_a_subnormal_slope_refuses_the_ambiguous_band():
    block = PiecewiseLinearConvex([POS_SUB, 1.], [0., -.5])
    with jax.enable_x64():
        f64 = jnp.float64
        assert not bool(block.conj_in_domain(jnp.asarray([1e-40], f64)))    # conservative
        assert bool(block.conj_in_domain(jnp.asarray([0.5], f64)))
        assert not bool(block.conj_in_domain(jnp.asarray([-0.5], f64)))
    normal = PiecewiseLinearConvex([0., 1.], [0., -.5])
    with jax.enable_x64():
        assert bool(normal.conj_in_domain(jnp.asarray([1e-300], jnp.float64)))
        assert not bool(normal.conj_in_domain(jnp.asarray([-1e-300], jnp.float64)))
