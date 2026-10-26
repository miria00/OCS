"""The spatial-full weighted SOC projection (ocs.prox.spatial_soc).

Gate for stage 2 of candidates/spatial_full_soc_20260930, whose SPEC.md is
frozen at sha256
31405f48f9e1150c593d7cd6a36a768fb3aab6c3881ccddfd695ed9ebc090c48. Application
only: no metric is constructed here, no released block's scaling class changes,
and nothing is reached through the public supplied path.

Four independent legs, deliberately not one:

1. **Two hand-derived algebraic controls.** ``M = diag(1, 100)`` at
   ``v = (-1, 1/50)`` has ``v`` in ``-Q`` but ``M v`` outside it, so the answer
   is ``(1/101, 1/101)`` and NOT zero; ``M = diag(100, 1)`` at
   ``v = (-1/50, 1)`` has ``v`` outside ``-Q`` but ``M v`` inside it, so the
   answer IS zero. A polar test written on ``v`` instead of ``M v`` fails both.
   Both were derived from the optimality conditions, not measured.
2. **The optimality conditions themselves**, evaluated in float64 in the
   ORIGINAL coordinates against the represented metric: ``z`` in ``Q``,
   ``n = M (v - z)`` in ``-Q``, ``z^T n = 0``. This is the reference, and it is
   solver-free.
3. **The certificate.** ``||z - z*||_M <= sqrt(|z^T n|)`` whenever the first two
   conditions hold, so the distance to the exact projection is bounded by a
   measured number rather than assumed small.
4. **The archived FAIR reference projector**, byte-identical in
   ``tests/reference/`` with its provenance recorded there. Regression evidence
   only, and only to root accuracy: this module returns ``||u(theta)||`` for the
   radius where the reference returns ``a t / (a - theta)``, which differ off the
   exact root. The reference is never edited to agree.

No generic nonlinear solver is used as a reference. In this project's own
records an SLSQP reference at n = 512 returned an INFEASIBLE point and made a
correct bound look violated, and the lesson taken was that reference outputs are
checked objects too. The KKT conditions in float64 need no convergence.

Tolerances are stated per assertion and the worst measured values are printed
for the record, so no tolerance here is doing acceptance work silently.
"""
from __future__ import annotations

import importlib.util
import os
import zlib
from contextlib import contextmanager
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox import cones
from ocs.prox import spatial_soc as ss

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)

# The acceptance rule is DERIVED, not tuned: one noise model, one safety factor.
#
# The three defects are evaluated at a float32 or float64 point, so each carries
# the roundoff of the products that form it. With ``top = max(a, max_j b_j)``:
#
#   n = M (v - z)  is formed to absolute accuracy  eps top ||v||,  so the polar
#     defect cannot be resolved below that;
#   z^T n          inherits it scaled by ||z||, so the complementarity cannot be
#     resolved below  eps ||z|| top ||v|| ;
#   the radius is a norm, so the primal defect floor is  eps ||z|| .
#
# A defect is accepted when it is within SAFETY times its own floor. Everything
# in the model is a measured quantity of the case at hand, so an ill-conditioned
# metric or a near-boundary point widens its OWN allowance and nothing else's,
# and the worst measured ratio of defect to allowance is printed for the record.
SAFETY = 16.0


@contextmanager
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _reference():
    """The archived counted copy of the E-reference projector, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get("OCS_FAIR_REFERENCE", ""),
                 os.path.join(here, "reference", "fair_projx.py"),
                 os.path.expanduser(
                     "~/OCSolver/candidates/positive_cone_metric_20260922/"
                     "fair/scripts/fair_projx.py")):
        if path and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("_fair_projx", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


# ---------------------------------------------------------------------------
# the independent float64 checker: the optimality conditions, in the original
# coordinates, from the represented factors, with no import of the kernel
# ---------------------------------------------------------------------------
def kkt(v, z, a, b, Q):
    """The three ABSOLUTE defects and the scales they are read against.

    Returns a dict with ``primal``, ``polar``, ``complementarity``, the derived
    floor of each, ``energy`` = ``z^T M z``, and ``certified``, the relative
    M-distance to the exact projection that the complementarity certifies.

    The certificate's normalizer is ``z^T M z`` and NOT ``||n||``. Near the cone
    boundary ``n = M (v - z)`` is a difference of nearly equal terms, so its
    DIRECTION is noise and an angle normalized by ``||n||`` reaches order one at
    a point correct to unit roundoff. Which of the two it is gets settled against
    the exact rational root, not by choosing a flattering normalizer.
    """
    v = np.asarray(v, np.float64)
    z = np.asarray(z, np.float64)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    Q = np.asarray(Q, np.float64)
    if v.ndim == 1:
        v, z = v[None, :], z[None, :]
        a, b, Q = np.atleast_1d(a), np.atleast_2d(b), Q[None, ...]
    d = v - z
    spatial = np.einsum("cij,cj->ci", Q, b * np.einsum("cji,cj->ci", Q, d[:, 1:]))
    n = np.concatenate([(a * d[:, 0])[:, None], spatial], axis=1)
    primal = np.maximum(np.linalg.norm(z[:, 1:], axis=1) - z[:, 0], 0.0)
    polar = np.maximum(np.linalg.norm(n[:, 1:], axis=1) + n[:, 0], 0.0)
    comp = np.abs(np.sum(z * n, axis=1))
    norm_v = np.linalg.norm(v, axis=1)
    norm_z = np.linalg.norm(z, axis=1)
    zt = np.einsum("cji,cj->ci", Q, z[:, 1:])
    energy = a * z[:, 0] ** 2 + np.sum(b * zt * zt, axis=1)
    top = np.maximum(a, b.max(axis=1))
    return {"primal": primal, "polar": polar, "complementarity": comp,
            "primal_floor": norm_z, "polar_floor": top * norm_v,
            "complementarity_floor": norm_z * top * norm_v,
            "energy": energy,
            "certified": np.sqrt(comp / np.where(energy > 0, energy, 1.0))}


def _seed(*parts, offset=0):
    """A seed that does NOT move between processes.

    ``hash()`` of anything containing a string is randomized per interpreter, so
    a fixture seeded with it draws different cases on every run and a gate built
    on it is not reproducible. One such seed in this file did hide a marginal
    float32 case for exactly one run, so the rule here is crc32 and nothing else.
    """
    key = "|".join(str(p) for p in parts).encode()
    return (zlib.crc32(key) + int(offset)) % (2 ** 31)


def _gamma(k):
    """``k u / (1 - k u)``, the standard summation allowance, rounded up.

    Same form and the same constant as ``ocs.verify.restore._gamma``, so the
    validated arithmetic below reads like the rest of this project's checkers.
    """
    ku = int(k) * (EPS64 / 2.0)
    assert ku < 0.5, "too many terms for a validated bound"
    return float(np.nextafter(np.float64(ku / (1.0 - ku)), np.inf))


def _up(value):
    value = float(value)
    return value if value == 0.0 else float(np.nextafter(np.float64(value),
                                                         np.inf))


def _norm_up(v):
    """A validated UPPER bound on a Euclidean norm: the sum and root both round."""
    v = np.asarray(v, np.float64).ravel()
    if v.size == 0:
        return 0.0
    value = float(np.linalg.norm(v))
    if value == 0.0:
        return 0.0
    return _up(value * (1.0 + _gamma(v.size + 2)))


def validated_distance_bound(v, z, a, b, Q):
    """A VALIDATED upper bound on ``||z - z*||_M`` for one cone, host float64.

    The naive certificate ``||z - z*||_M <= sqrt(|z^T n|)`` is exact in real
    arithmetic and is NOT a bound once evaluated: the float ``z^T n`` can cancel
    to exactly zero at a point that is a measurable distance from the exact
    projection, which this file measures. Two things are needed to make it one.

    First, an upper bound on the exact ``|z^T n|`` rather than its float value.
    Every intermediate magnitude is dominated by the positive surrogate
    ``N = |Q| ( b * ( |Q|^T |v - z| ) )``, so charging ``gamma`` on that, plus
    ``gamma`` on the dot product itself, bounds it.

    Second, NEITHER remaining hypothesis can be established as stated, and
    voiding the bound would be useless. ``n in -Q`` fails near either boundary,
    where ``n`` is noise. ``z in Q`` fails ALWAYS, and for a reason worth stating:
    the returned point is ON the cone boundary, ``z0`` is the float norm of its
    own spatial part, and a validated upper bound on that norm necessarily
    exceeds it. A point on the boundary cannot be certified to be in the cone.

    So both are carried as residuals instead. With ``eps_p >= dist(z, Q)``,
    ``rho >= dist(n, -Q)``, ``top = max(a, max_j b_j)``,
    ``amin = min(a, min_j b_j)`` the extreme eigenvalues of ``M``, and ``z*`` in
    ``Q`` with ``n* = M (v - z*)`` in ``-Q`` exactly, being the true projection:

        ||z - z*||_M^2  <=  |z^T n| + z^T n* + z*^T n
        z^T n*  <=  eps_p ||n*||  <=  eps_p top ( ||v - z|| + D / sqrt(amin) )
        z*^T n  <=  rho ||z*||    <=  rho ( ||z|| + D / sqrt(amin) )

    which is ``D^2 <= c0 + c1 D`` with everything measured, so
    ``D <= (c1 + sqrt(c1^2 + 4 c0)) / 2``. Nothing is assumed away, the bound
    always applies, and it tightens to the ideal one as the defects vanish.

    Returns ``(bound, detail)`` with every part of the model in ``detail``.
    """
    v = np.asarray(v, np.float64).ravel()
    z = np.asarray(z, np.float64).ravel()
    a = float(a)
    b = np.asarray(b, np.float64).ravel()
    Q = np.asarray(Q, np.float64)
    d = b.size
    w = v - z
    absQ = np.abs(Q)
    surrogate = absQ @ (b * (absQ.T @ np.abs(w[1:])))
    surrogate0 = a * abs(w[0])
    n_spatial = Q @ (b * (Q.T @ w[1:]))
    n = np.concatenate([[a * w[0]], n_spatial])
    charge = _gamma(2 * d + 6)

    # |z^T n_exact| <= |fl(z^T n)| + gamma(k+1) |z|^T |n| + charge |z|^T N
    comp = abs(float(z @ n))
    comp_up = _up(comp
                  + _gamma(d + 2) * float(np.abs(z) @ np.abs(n))
                  + charge * float(np.abs(z) @ np.concatenate([[surrogate0],
                                                               surrogate])))
    # rho >= dist(n, -Q).  Outside both cones the distance is
    # (||nbar|| + n0)/sqrt(2); dividing a violation by the float64 sqrt(2),
    # which is below the true value, can only enlarge it.
    n0_up = _up(n[0] + charge * surrogate0)
    nbar_up = _up(_norm_up(n_spatial) + charge * _norm_up(surrogate))
    rho = _up(max(0.0, nbar_up + n0_up) / np.sqrt(2.0))

    # eps_p >= dist(z, Q), by the same one-sided formula on the other cone.
    eps_p = _up(max(0.0, _norm_up(z[1:]) - z[0]) / np.sqrt(2.0))
    amin = min(a, float(b.min()))
    top = max(a, float(b.max()))
    root_amin = float(np.nextafter(np.sqrt(amin), -np.inf))
    c1 = _up((_up(eps_p * top) + rho) / root_amin)
    c0 = _up(comp_up + _up(eps_p * top * _norm_up(w)) + _up(rho * _norm_up(z)))
    bound = _up(0.5 * (c1 + np.sqrt(_up(c1 * c1 + 4.0 * c0))))
    return bound, {"complementarity_float": comp, "complementarity_up": comp_up,
                   "rho": rho, "eps_primal": eps_p, "amin": amin, "top": top,
                   "c0": c0, "c1": c1}


def m_norm(x, a, b, Q):
    """``||x||_M`` from the factors, host float64."""
    x = np.asarray(x, np.float64)
    single = x.ndim == 1
    X = x[None, :] if single else x
    a, b = np.atleast_1d(a), np.atleast_2d(b)
    Qm = Q[None, ...] if np.ndim(Q) == 2 else Q
    yt = np.einsum("cji,cj->ci", Qm, X[:, 1:])
    out = np.sqrt(a * X[:, 0] ** 2 + np.sum(b * yt * yt, axis=1))
    return out[0] if single else out


def assert_conditions(v, z, a, b, Q, eps, label, safety=SAFETY):
    """Each defect against its own derived floor, with the ratios printed."""
    out = kkt(v, z, a, b, Q)
    ratios = {}
    for key in ("primal", "polar", "complementarity"):
        allowance = safety * eps * out[f"{key}_floor"]
        ratios[key] = float(np.max(out[key] / np.where(allowance > 0, allowance,
                                                       1.0)))
    print(f"{label}: of allowance, primal {ratios['primal']:.3f} "
          f"polar {ratios['polar']:.3f} "
          f"complementarity {ratios['complementarity']:.3f}; "
          f"worst certified relative distance {float(out['certified'].max()):.3e}")
    assert ratios["primal"] <= 1.0, \
        f"{label}: z left the cone by {ratios['primal']:.3f} of its floor"
    assert ratios["polar"] <= 1.0, \
        f"{label}: n left the polar by {ratios['polar']:.3f} of its floor"
    assert ratios["complementarity"] <= 1.0, \
        f"{label}: complementarity at {ratios['complementarity']:.3f} of its floor"
    return ratios


# ---------------------------------------------------------------------------
# leg 1: the two algebraic controls that discriminate the weighted polar branch
# ---------------------------------------------------------------------------
def test_control_one_polar_test_is_on_the_metric_image():
    """v IS in -Q, M v is NOT, so the answer is not zero."""
    with x64():
        metric = ss.SpatialFullSOCMetric(jnp.float64(1.0),
                                         jnp.array([100.0], jnp.float64),
                                         jnp.eye(1, dtype=jnp.float64))
        v = jnp.array([-1.0, 1.0 / 50.0], jnp.float64)
        assert v[0] <= -abs(v[1]), "the fixture must put v inside -Q"
        mv = metric.apply(v)
        assert not bool(mv[0] <= -abs(mv[1])), "the fixture must put M v outside -Q"
        z, status = ss.project(v, metric)
        assert int(status) == ss.STATUS_ROOT, ss.STATUS_NAMES[int(status)]
        expected = np.array([1.0 / 101.0, 1.0 / 101.0])
        assert np.allclose(np.asarray(z), expected, rtol=0.0, atol=1e-15), z
        assert_conditions(v, z, 1.0, [100.0], np.eye(1), EPS64, "control 1")


def test_control_two_polar_branch_fires_on_the_metric_image():
    """v is NOT in -Q, M v IS, so the answer is exactly zero."""
    with x64():
        metric = ss.SpatialFullSOCMetric(jnp.float64(100.0),
                                         jnp.array([1.0], jnp.float64),
                                         jnp.eye(1, dtype=jnp.float64))
        v = jnp.array([-1.0 / 50.0, 1.0], jnp.float64)
        assert not bool(v[0] <= -abs(v[1])), "the fixture must put v outside -Q"
        mv = metric.apply(v)
        assert bool(mv[0] <= -abs(mv[1])), "the fixture must put M v inside -Q"
        z, status = ss.project(v, metric)
        assert int(status) == ss.STATUS_POLAR, ss.STATUS_NAMES[int(status)]
        assert np.array_equal(np.asarray(z), np.zeros(2)), z


def test_a_polar_test_written_on_v_would_fail_control_one():
    """The mutation this control exists to catch, stated as a test.

    Not a check of the kernel: a check that the fixture DISCRIMINATES. If the
    two conditions ever agree on control 1, the control has stopped working and
    the branch could be written either way undetected.
    """
    with x64():
        metric = ss.SpatialFullSOCMetric(jnp.float64(1.0),
                                         jnp.array([100.0], jnp.float64),
                                         jnp.eye(1, dtype=jnp.float64))
        v = jnp.array([-1.0, 1.0 / 50.0], jnp.float64)
        on_v = bool(v[0] <= -abs(v[1]))
        on_mv = bool(metric.apply(v)[0] <= -abs(metric.apply(v)[1]))
        assert on_v != on_mv, "control 1 no longer discriminates the polar test"


# ---------------------------------------------------------------------------
# reductions: a scalar metric must not change the Euclidean answer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [1.0, 0.125, 8.0, 1e-3, 1e3])
@pytest.mark.parametrize("k", [2, 5, 33])
def test_scalar_reduction_matches_the_released_euclidean_projection(c, k):
    """P_{cI}(v) = Pi_Q(v) for every c > 0, and Pi_Q is the released prox."""
    with x64():
        rng = np.random.default_rng(20260930 + k)
        for kind in ("inside", "polar", "root_pos", "root_neg", "apex"):
            v = _one_case(rng, k - 1, kind)
            metric = ss.SpatialFullSOCMetric(
                jnp.asarray(c, jnp.float64),
                jnp.full((k - 1,), c, jnp.float64),
                jnp.eye(k - 1, dtype=jnp.float64))
            z, status = ss.project(jnp.asarray(v, jnp.float64), metric)
            assert int(status) != ss.STATUS_FAILED, kind
            euclid = np.asarray(cones.SecondOrderCone().prox(
                jnp.asarray(v, jnp.float64), jnp.asarray(1.0, jnp.float64)))
            assert np.allclose(np.asarray(z), euclid, rtol=1e-12, atol=1e-12), \
                f"{kind}: scalar metric changed the Euclidean projection"


def test_a_common_rescaling_of_the_metric_leaves_the_point_alone():
    """P_{cM} = P_M for c > 0: the argmin is invariant to a positive scaling."""
    with x64():
        rng = np.random.default_rng(4242)
        a, b, Q = _one_metric(rng, 16)
        v = _one_case(rng, 16, "root_neg")
        base = ss.SpatialFullSOCMetric(jnp.asarray(a), jnp.asarray(b), jnp.asarray(Q))
        z0, s0 = ss.project(jnp.asarray(v), base)
        for c in (1e-6, 1e-2, 7.0, 1e5):
            scaled = ss.SpatialFullSOCMetric(jnp.asarray(a * c),
                                             jnp.asarray(b * c), jnp.asarray(Q))
            zc, sc = ss.project(jnp.asarray(v), scaled)
            assert int(sc) == int(s0)
            rel = float(np.linalg.norm(np.asarray(zc) - np.asarray(z0))
                        / max(1e-300, np.linalg.norm(np.asarray(z0))))
            assert rel <= 1e-11, f"c = {c}: the point moved by {rel:.3e}"


# ---------------------------------------------------------------------------
# fixtures: the archived case families
# ---------------------------------------------------------------------------
def _one_metric(rng, d, conditioning=100.0):
    la = rng.uniform(-4.0, 4.0)
    a = float(np.exp(la))
    lb = np.clip(la + rng.uniform(-np.log(conditioning), np.log(conditioning), d),
                 -4.0, 4.0)
    b = np.exp(lb)
    if d == 1:
        Q = np.ones((1, 1))
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return a, b, Q


def _one_case(rng, d, kind, a=1.0, b=None, Q=None):
    y = rng.standard_normal(d) * float(np.exp(rng.uniform(-3.0, 3.0)))
    R = float(np.linalg.norm(y))
    if kind == "inside":
        t = R * float(rng.uniform(1.0, 3.0)) + 1e-9
    elif kind == "polar":
        by = R if b is None else float(np.linalg.norm(
            np.asarray(b) * (np.asarray(Q).T @ y if Q is not None else y)))
        t = -(by / a) * float(rng.uniform(1.0, 3.0)) - 1e-9
    elif kind == "root_pos":
        t = R * float(rng.uniform(0.05, 0.95))
    elif kind == "root_neg":
        by = R if b is None else float(np.linalg.norm(
            np.asarray(b) * (np.asarray(Q).T @ y if Q is not None else y)))
        t = -(by / a) * float(rng.uniform(0.05, 0.95))
    elif kind == "apex":
        t = 0.0
    elif kind == "near_feasible":
        t = R * (1.0 - 10.0 ** float(rng.uniform(-12, -3)))
    elif kind == "near_polar":
        by = R if b is None else float(np.linalg.norm(
            np.asarray(b) * (np.asarray(Q).T @ y if Q is not None else y)))
        t = -(by / a) * (1.0 - 10.0 ** float(rng.uniform(-12, -3)))
    else:
        raise AssertionError(kind)
    return np.concatenate([[t], y])


def _family(rng, n, d, kind, conditioning=100.0):
    a = np.empty(n)
    b = np.empty((n, d))
    Q = np.empty((n, d, d))
    V = np.empty((n, d + 1))
    for i in range(n):
        a[i], b[i], Q[i] = _one_metric(rng, d, conditioning)
        V[i] = _one_case(rng, d, kind, a[i], b[i], Q[i])
    return V, a, b, Q


@pytest.mark.parametrize("d", [1, 2, 4, 32, 128])
@pytest.mark.parametrize("kind", ["inside", "polar", "root_pos", "root_neg",
                                  "apex", "near_feasible", "near_polar"])
def test_optimality_conditions_hold_in_float64(d, kind):
    """Leg 2 on every branch and every spatial size, with a rotation."""
    with x64():
        rng = np.random.default_rng(_seed(d, kind))
        V, a, b, Q = _family(rng, 24, d, kind)
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        assert not np.any(np.asarray(status) == ss.STATUS_FAILED), \
            np.asarray(status)
        assert_conditions(V, z, a, b, Q, EPS64, f"float64 d={d} {kind}")


@pytest.mark.parametrize("d", [1, 4, 128])
@pytest.mark.parametrize("kind", ["root_pos", "root_neg", "apex",
                                  "near_feasible", "near_polar"])
def test_optimality_conditions_in_float32(d, kind):
    """The float32 domain, reported against its own frozen bound.

    The float64 reference tolerance is NOT the float32 production tolerance, so
    these are separate numbers, frozen on these development fixtures.
    """
    rng = np.random.default_rng(_seed(d, kind, offset=1000))
    V, a, b, Q = _family(rng, 24, d, kind, conditioning=10.0)
    metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float32),
                                     jnp.asarray(b, jnp.float32),
                                     jnp.asarray(Q, jnp.float32))
    z, status = ss.project(jnp.asarray(V, jnp.float32), metric)
    assert not np.any(np.asarray(status) == ss.STATUS_FAILED), np.asarray(status)
    assert np.asarray(z).dtype == np.float32
    assert_conditions(V, z, a, b, Q, EPS32, f"float32 d={d} {kind}")


def test_extreme_permitted_conditioning_is_reported_not_hidden():
    """A wide spread of b is allowed here, and its cost is measured."""
    with x64():
        rng = np.random.default_rng(777)
        for spread in (1e2, 1e4, 1e6, 1e8):
            V, a, b, Q = _family(rng, 16, 8, "root_neg", conditioning=spread)
            metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                             jnp.asarray(b, jnp.float64),
                                             jnp.asarray(Q, jnp.float64))
            z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
            assert not np.any(np.asarray(status) == ss.STATUS_FAILED)
            assert_conditions(V, z, a, b, Q, EPS64, f"spread {spread:.0e}")


def test_repeated_spatial_eigenvalues():
    """Ties in b, where a root solve that sorts or divides by gaps would break."""
    with x64():
        rng = np.random.default_rng(31337)
        d = 12
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        b = np.repeat(np.array([0.5, 0.5, 0.5, 4.0, 4.0, 4.0]), 2)
        a = 1.25
        for kind in ("root_pos", "root_neg", "apex"):
            v = _one_case(rng, d, kind, a, b, Q)
            metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                             jnp.asarray(b, jnp.float64),
                                             jnp.asarray(Q, jnp.float64))
            z, status = ss.project(jnp.asarray(v, jnp.float64), metric)
            assert int(status) != ss.STATUS_FAILED, kind
            assert_conditions(v, z, a, b, Q, EPS64, f"repeated {kind}")


# ---------------------------------------------------------------------------
# the branch decisions themselves
# ---------------------------------------------------------------------------
def test_a_point_in_the_cone_is_returned_bitwise():
    with x64():
        rng = np.random.default_rng(99)
        V, a, b, Q = _family(rng, 32, 9, "inside")
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        assert np.all(np.asarray(status) == ss.STATUS_INSIDE)
        assert np.array_equal(np.asarray(z), V), "the inside branch must be exact"


def test_the_projection_is_idempotent():
    """P_M(P_M(v)) = P_M(v) bitwise, because the result is in the cone."""
    with x64():
        rng = np.random.default_rng(1234)
        V, a, b, Q = _family(rng, 32, 7, "root_neg")
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z1, s1 = ss.project(jnp.asarray(V, jnp.float64), metric)
        z2, s2 = ss.project(z1, metric)
        assert np.all(np.asarray(s2) == ss.STATUS_INSIDE), np.asarray(s2)
        assert np.array_equal(np.asarray(z2), np.asarray(z1))


def test_the_apex_branch_is_the_closed_form():
    """t = 0 forces theta = a, so the answer is u_j = b_j ytilde_j/(b_j + a)."""
    with x64():
        rng = np.random.default_rng(606)
        d = 20
        a, b, Q = _one_metric(rng, d)
        y = rng.standard_normal(d)
        v = np.concatenate([[0.0], y])
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(v, jnp.float64), metric)
        assert int(status) == ss.STATUS_APEX, ss.STATUS_NAMES[int(status)]
        u = b * (Q.T @ y) / (b + a)
        expected = np.concatenate([[np.linalg.norm(u)], Q @ u])
        assert np.allclose(np.asarray(z), expected, rtol=1e-14, atol=1e-15)
        assert_conditions(v, z, a, b, Q, EPS64, "apex closed form")


def test_a_batch_mixing_branches_agrees_row_by_row_with_single_calls():
    """One cone's branch may not disturb another's.

    What agrees exactly and what agrees to roundoff is the whole content of this
    test, and it took a GPU to get it right. A batched call and a one-cone call
    are different kernels, and on a GPU their dot products associate differently,
    so a cone placed within roundoff of a branch boundary is decided differently
    by the two. That is not a defect: at such a point the two branches give the
    SAME answer to roundoff, which is asserted for every row. The branch label is
    therefore required to agree exactly only where the case is further than its
    own roundoff from both boundaries, and the fixture deliberately contains rows
    that are not.
    """
    with x64():
        rng = np.random.default_rng(555)
        d = 6
        kinds = ["inside", "polar", "root_pos", "root_neg", "apex",
                 "near_feasible", "near_polar"] * 3
        a = np.empty(len(kinds))
        b = np.empty((len(kinds), d))
        Q = np.empty((len(kinds), d, d))
        V = np.empty((len(kinds), d + 1))
        for i, kind in enumerate(kinds):
            a[i], b[i], Q[i] = _one_metric(rng, d)
            V[i] = _one_case(rng, d, kind, a[i], b[i], Q[i])
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        seen, ambiguous = set(), []
        for i, kind in enumerate(kinds):
            one = ss.SpatialFullSOCMetric(jnp.asarray(a[i], jnp.float64),
                                          jnp.asarray(b[i], jnp.float64),
                                          jnp.asarray(Q[i], jnp.float64))
            zi, si = ss.project(jnp.asarray(V[i], jnp.float64), one)
            # ABSOLUTE, against the size of the input. A near-polar row's answer
            # is itself of size delta, and its relative accuracy is eps/delta by
            # the law measured elsewhere in this file, so two independent
            # evaluations of it may differ by that much RELATIVELY while agreeing
            # to roundoff absolutely. A relative test here was measured failing
            # at 8.3e-06 on a GPU for exactly that reason, on a row whose
            # absolute agreement was one unit of roundoff.
            gap = float(np.abs(np.asarray(zi) - np.asarray(z[i])).max()
                        / max(1e-300, float(np.linalg.norm(V[i]))))
            assert gap <= SAFETY * EPS64, \
                f"row {i} ({kind}) point differs by {gap:.3e} of the input size"
            yt = Q[i].T @ V[i][1:]
            to_cone = abs(V[i][0] - float(np.linalg.norm(V[i][1:])))
            to_polar = abs(a[i] * V[i][0] + float(np.linalg.norm(b[i] * yt)))
            top = max(a[i], float(b[i].max()))
            decided = (to_cone > SAFETY * EPS64 * float(np.linalg.norm(V[i]))
                       and to_polar > SAFETY * EPS64 * top
                       * float(np.linalg.norm(V[i])))
            if decided:
                assert int(si) == int(status[i]), \
                    f"row {i} ({kind}) status, and it is not a boundary case"
            ambiguous.append(not decided)
            seen.add(int(si))
        assert {ss.STATUS_INSIDE, ss.STATUS_POLAR, ss.STATUS_ROOT,
                ss.STATUS_APEX} <= seen, f"the batch missed a branch: {seen}"
        print(f"rows within roundoff of a branch boundary: {sum(ambiguous)} "
              f"of {len(kinds)}")


def test_a_shared_metric_means_one_metric_for_every_cone():
    with x64():
        rng = np.random.default_rng(8080)
        d = 5
        a, b, Q = _one_metric(rng, d)
        V = np.stack([_one_case(rng, d, kind, a, b, Q)
                      for kind in ("root_pos", "root_neg", "apex", "inside",
                                   "polar")])
        shared = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        per_cone = ss.SpatialFullSOCMetric(
            jnp.asarray(np.repeat(a, len(V)), jnp.float64),
            jnp.asarray(np.tile(b, (len(V), 1)), jnp.float64),
            jnp.asarray(np.tile(Q, (len(V), 1, 1)), jnp.float64))
        z1, s1 = ss.project(jnp.asarray(V, jnp.float64), shared)
        z2, s2 = ss.project(jnp.asarray(V, jnp.float64), per_cone)
        assert np.array_equal(np.asarray(s1), np.asarray(s2))
        assert np.array_equal(np.asarray(z1), np.asarray(z2))


# ---------------------------------------------------------------------------
# leg 3: the certificate
# ---------------------------------------------------------------------------
def test_the_validated_bound_covers_the_float32_point_on_a_whole_family():
    """Both points carry their own validated bound, so the pair can be checked.

    Neither the float32 nor the float64 answer is the exact projection, so what
    can be asserted is the triangle inequality through it: the distance between
    them is at most the sum of the two validated bounds. The float32 bound is
    also required to be smaller than the point, since a bound the size of the
    answer certifies nothing.
    """
    rng = np.random.default_rng(24680)
    d = 24
    V, a, b, Q = _family(rng, 32, d, "root_neg", conditioning=10.0)
    m32 = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float32),
                                  jnp.asarray(b, jnp.float32),
                                  jnp.asarray(Q, jnp.float32))
    z32, s32 = ss.project(jnp.asarray(V, jnp.float32), m32)
    assert not np.any(np.asarray(s32) == ss.STATUS_FAILED)
    with x64():
        m64 = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                      jnp.asarray(b, jnp.float64),
                                      jnp.asarray(Q, jnp.float64))
        z64, s64 = ss.project(jnp.asarray(V, jnp.float64), m64)
        assert np.array_equal(np.asarray(s64), np.asarray(s32))
    z32h = np.asarray(z32, np.float64)
    z64h = np.asarray(z64, np.float64)
    actual = m_norm(z32h - z64h, a, b, Q)
    own = m_norm(z64h, a, b, Q)
    worst_ratio, worst_share = 0.0, 0.0
    for i in range(len(V)):
        b32, _d32 = validated_distance_bound(V[i], z32h[i], a[i], b[i], Q[i])
        b64, _d64 = validated_distance_bound(V[i], z64h[i], a[i], b[i], Q[i])
        assert actual[i] <= b32 + b64, \
            f"cone {i}: the validated bounds missed the distance"
        worst_ratio = max(worst_ratio, actual[i] / max(b32 + b64, 1e-300))
        worst_share = max(worst_share, b32 / max(own[i], 1e-300))
    print(f"validated bound: the distance uses at most {worst_ratio:.3e} of the "
          f"pair of bounds, and the float32 bound is at most {worst_share:.3e} "
          f"of the point")
    assert worst_share < 1.0, "a bound the size of the point certifies nothing"


def test_the_estimate_is_quoted_only_beside_its_hypotheses():
    """The primal and polar defects are what the estimate rests on."""
    with x64():
        rng = np.random.default_rng(13579)
        V, a, b, Q = _family(rng, 16, 10, "root_pos")
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, _ = ss.project(jnp.asarray(V, jnp.float64), metric)
        defects = ss.optimality_defects(jnp.asarray(V, jnp.float64), z, metric)
        for key in ("primal", "polar", "complementarity"):
            assert key in defects
            assert f"{key}_scale" in defects
        rel_polar = np.asarray(defects["polar"]) / np.maximum(
            np.asarray(defects["polar_scale"]), 1e-300)
        print(f"polar defect relative worst {rel_polar.max():.3e}")
        assert rel_polar.max() <= SAFETY * EPS64, \
            "the hypothesis n in -Q is not satisfied, so the bound is not one"
        certified = np.asarray(ss.relative_distance_estimate(defects))
        assert np.all(np.isfinite(certified))
        assert np.all(certified >= 0.0)
        print(f"certified relative distance worst {certified.max():.3e}")


# ---------------------------------------------------------------------------
# an exact reference: the root equation is rational, so Fraction closes it
# ---------------------------------------------------------------------------
def exact_root(t, yt, a, b, digits=80):
    """``theta*`` to ``digits`` bits, in exact rational arithmetic.

    Every input is a binary rational, and squaring the root equation clears the
    only irrationality, so

        S(theta) = sum_j (b_j ytilde_j / (b_j + theta))^2   against
        R(theta) = (a t / (a - theta))^2

    is a comparison of two exact Fractions at every trial point. On the t > 0
    bracket S is decreasing and R increasing, and on the t < 0 bracket
    ``phi^2 S(a + phi)`` is increasing against the constant ``(a t)^2``, so each
    is monotone and plain bisection converges with no arithmetic error at all.

    This is the reference that decides whether a near-boundary limitation is in
    the returned POINT or only in the residual used to measure it.
    """
    T, A = Fraction(float(t)), Fraction(float(a))
    B = [Fraction(float(x)) for x in b]
    Y = [Fraction(float(x)) for x in yt]

    def spatial(theta):
        return sum((bj * yj / (bj + theta)) ** 2 for bj, yj in zip(B, Y))

    if t > 0:
        lo, hi = Fraction(0), A
        for _ in range(digits):
            mid = (lo + hi) / 2
            if spatial(mid) * (A - mid) ** 2 > (A * T) ** 2:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo, hi = Fraction(0), Fraction(1)
    while spatial(A + hi) * hi ** 2 < (A * T) ** 2:
        hi *= 2
    for _ in range(digits):
        mid = (lo + hi) / 2
        if spatial(A + mid) * mid ** 2 < (A * T) ** 2:
            lo = mid
        else:
            hi = mid
    return A + (lo + hi) / 2


def exact_point(t, y, a, b, Q, digits=80):
    """``z*`` in the ORIGINAL coordinates, from the exact root."""
    yt = np.asarray(Q, np.float64).T @ np.asarray(y, np.float64)
    theta = exact_root(t, yt, a, b, digits)
    u = [Fraction(float(bj)) * Fraction(float(yj)) / (Fraction(float(bj)) + theta)
         for bj, yj in zip(b, yt)]
    uf = np.array([float(x) for x in u], np.float64)
    spatial = np.asarray(Q, np.float64) @ uf
    return np.concatenate([[float(np.linalg.norm(spatial))], spatial]), theta


@pytest.mark.parametrize("delta", [1e-1, 1e-3, 1e-6, 1e-9, 1e-12])
@pytest.mark.parametrize("side", ["feasible", "polar"])
def test_what_a_near_boundary_case_costs_measured_against_the_exact_root(delta,
                                                                        side):
    """What degrades near each boundary, and what does not, decided exactly.

    Write the distance to the boundary as ``delta``. Approaching either one, the
    root equation compares two quantities that agree to ``delta``, so the float
    difference carries relative error ``eps / delta``: the information is not in
    the subtraction. The consequences differ by side, and the exact rational root
    is what separates them rather than an argument:

    * NEAR-FEASIBLE. The answer tends to ``v``, which is of full size, and the
      root enters the point only through ``b_j + theta`` with ``theta`` tending
      to zero. Relative accuracy of the POINT is preserved.
    * NEAR-POLAR. The answer tends to ZERO like ``delta``, and the root enters as
      ``1/theta``, so ``eps/delta`` passes straight into the point's RELATIVE
      error. Its ABSOLUTE error stays at ``eps`` times the size of ``v``, which
      is what a following iteration sees.

    So the assertion is on the absolute error against ``||v||_M``, and the
    relative error is asserted against the derived ``eps/delta``, which is a
    statement about the arithmetic and not a tolerance chosen to pass. Both
    numbers are printed at every delta so the trend is in the record.
    """
    with x64():
        rng = np.random.default_rng(int(-np.log10(delta)) * 31 + len(side))
        d = 3
        a, b, Q = _one_metric(rng, d, conditioning=20.0)
        y = rng.standard_normal(d)
        yt = Q.T @ y
        if side == "feasible":
            t = float(np.linalg.norm(y)) * (1.0 - delta)
        else:
            t = -float(np.linalg.norm(b * yt)) / a * (1.0 - delta)
        v = np.concatenate([[t], y])
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(v, jnp.float64), metric)
        assert int(status) in (ss.STATUS_ROOT, ss.STATUS_APEX), \
            ss.STATUS_NAMES[int(status)]
        z_star, theta_star = exact_point(t, y, a, b, Q)
        moved = float(m_norm(np.asarray(z, np.float64) - z_star, a, b, Q))
        own = float(m_norm(z_star, a, b, Q))
        ambient = float(m_norm(v, a, b, Q))
        absolute = moved / max(ambient, 1e-300)
        relative = moved / max(own, 1e-300)
        out = kkt(v, z, a, b, Q)
        estimate = float(out["certified"][0])
        bound, detail = validated_distance_bound(v, np.asarray(z, np.float64),
                                                 a, b, Q)
        print(f"{side} delta {delta:.0e}: theta* {float(theta_star):.6e}, "
              f"||z*||/||v|| {own / max(ambient, 1e-300):.3e}, error against "
              f"||v||_M {absolute:.3e}, against ||z*||_M {relative:.3e}, "
              f"naive estimate {estimate:.3e}, validated bound "
              f"{bound / max(own, 1e-300):.3e} relative, float "
              f"complementarity {detail['complementarity_float']:.3e}")
        assert absolute <= SAFETY * EPS64, \
            f"the absolute error reached {absolute:.3e} of the ambient size"
        assert relative <= SAFETY * EPS64 / delta, \
            f"the relative error {relative:.3e} exceeds the derived eps/delta"
        assert moved <= bound, \
            f"the VALIDATED bound {bound:.3e} missed the distance {moved:.3e}"
        if side == "feasible":
            assert relative <= SAFETY * EPS64, \
                "the near-feasible side must keep its relative accuracy"


def test_the_exact_rational_reference_reproduces_control_one():
    """The reference is a checked object too, so it is checked on a known case."""
    z_star, theta_star = exact_point(-1.0, np.array([1.0 / 50.0]), 1.0,
                                     np.array([100.0]), np.eye(1))
    assert abs(float(theta_star) - 102.0) <= 1e-12, float(theta_star)
    assert np.allclose(z_star, [1.0 / 101.0, 1.0 / 101.0], rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# the Moreau pair: one form is production, the other is the test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 3, 64])
def test_the_two_dual_forms_agree(d):
    """w - Sigma P_Sigma(Sigma^{-1} w) = -P_{Sigma^{-1}}(-w)."""
    with x64():
        rng = np.random.default_rng(2026 + d)
        V, a, b, Q = _family(rng, 20, d, "root_pos")
        sigma = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                        jnp.asarray(b, jnp.float64),
                                        jnp.asarray(Q, jnp.float64))
        w = jnp.asarray(V, jnp.float64)
        one, s1 = ss.project_polar_step(w, sigma)
        two, s2 = ss.project_polar_step_moreau(w, sigma)
        rel = float(np.linalg.norm(np.asarray(one) - np.asarray(two))
                    / max(1e-300, np.linalg.norm(np.asarray(one))))
        print(f"moreau d={d}: relative difference {rel:.3e}")
        assert rel <= 1e-10, f"the two exact forms disagree by {rel:.3e}"


def test_supplying_the_metric_where_its_inverse_belongs_is_caught():
    """The mutation the Moreau pair exists to catch.

    Swapping ``Sigma`` for ``Sigma^{-1}`` in one form and not the other must
    make the pair disagree, otherwise the pair is not a test.
    """
    with x64():
        rng = np.random.default_rng(4711)
        V, a, b, Q = _family(rng, 12, 5, "root_pos")
        sigma = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                        jnp.asarray(b, jnp.float64),
                                        jnp.asarray(Q, jnp.float64))
        w = jnp.asarray(V, jnp.float64)
        good, _ = ss.project_polar_step(w, sigma)
        wrong, _ = ss.project_polar_step_moreau(w, sigma.inverse())
        rel = float(np.linalg.norm(np.asarray(good) - np.asarray(wrong))
                    / max(1e-300, np.linalg.norm(np.asarray(good))))
        assert rel > 1e-3, \
            f"the inverse swap was invisible, relative difference {rel:.3e}"


def test_the_inverse_is_in_the_class_and_round_trips():
    with x64():
        rng = np.random.default_rng(321)
        a, b, Q = _one_metric(rng, 9)
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        v = jnp.asarray(rng.standard_normal(10), jnp.float64)
        back = metric.inverse().apply(metric.apply(v))
        rel = float(np.linalg.norm(np.asarray(back) - np.asarray(v))
                    / np.linalg.norm(np.asarray(v)))
        assert rel <= 1e-13, rel
        dense = np.asarray(metric.dense()[0], np.float64)
        assert np.allclose(dense, dense.T, rtol=0, atol=1e-14)
        assert np.allclose(dense[0, 1:], 0.0, rtol=0, atol=1e-14), \
            "the class has no time-space cross terms"


# ---------------------------------------------------------------------------
# leg 4: the archived reference projector
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 4, 32, 128])
@pytest.mark.parametrize("kind", ["root_pos", "root_neg", "apex", "inside",
                                  "polar", "near_feasible"])
def test_agrees_with_the_archived_reference_projector(d, kind):
    """Regression evidence, to root accuracy and not bitwise.

    The reference returns a t/(a - theta) for the radius; this module returns
    ||u(theta)||. They agree at the exact root, so a relative difference at the
    level of the root error is expected and a larger one is a defect.
    """
    reference = _reference()
    if reference is None:
        pytest.skip("the archived FAIR reference projector is not on this host")
    with x64():
        rng = np.random.default_rng(5150 + d)
        V, a, b, Q = _family(rng, 24, d, kind)
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        assert not np.any(np.asarray(status) == ss.STATUS_FAILED)
        # The reference works in ROTATED spatial coordinates, so rotate in and
        # out around it.  Nothing about the reference itself is changed.
        Wr = np.einsum("cji,cj->ci", Q, V[:, 1:])
        Tp, Zr = reference.proj_block_rows(V[:, 0].copy(), Wr.copy(), a.copy(),
                                           b.copy())
        ref = np.concatenate([Tp[:, None], np.einsum("cij,cj->ci", Q, Zr)],
                             axis=1)
        num = np.linalg.norm(np.asarray(z) - ref, axis=1)
        den = np.maximum(np.linalg.norm(ref, axis=1), 1e-300)
        rel = num / den
        print(f"reference d={d} {kind}: worst relative difference {rel.max():.3e}")
        assert rel.max() <= 1e-8, f"worst relative difference {rel.max():.3e}"
        # And the reference's own point, through the same independent checker,
        # allowed its OWN declared stopping tolerance rather than machine
        # roundoff: fair_projx converges Newton to NEWTON_RTOL, and its radius
        # is a t/(a - theta), so its primal defect is the root error and not
        # zero. The constant is the reference's, not one chosen here.
        assert_conditions(V, ref, a, b, Q, float(reference.NEWTON_RTOL),
                          f"reference point d={d} {kind}")


# ---------------------------------------------------------------------------
# failure is a status, and it is per cone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("break_it", ["a_zero", "a_negative", "b_zero",
                                      "b_negative", "a_nan", "b_inf"])
def test_a_metric_that_is_not_positive_fails_that_cone_only(break_it):
    with x64():
        rng = np.random.default_rng(963)
        d = 4
        V, a, b, Q = _family(rng, 5, d, "root_pos")
        if break_it == "a_zero":
            a[2] = 0.0
        elif break_it == "a_negative":
            a[2] = -1.0
        elif break_it == "b_zero":
            b[2, 1] = 0.0
        elif break_it == "b_negative":
            b[2, 1] = -3.0
        elif break_it == "a_nan":
            a[2] = np.nan
        else:
            b[2, 1] = np.inf
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        status = np.asarray(status)
        assert status[2] == ss.STATUS_FAILED, f"{break_it}: cone 2 was accepted"
        assert np.all(np.isnan(np.asarray(z)[2])), "a failed cone must be NaN"
        others = [i for i in range(5) if i != 2]
        assert not np.any(status[others] == ss.STATUS_FAILED), \
            "a neighbour was dragged down"
        assert np.all(np.isfinite(np.asarray(z)[others]))
        assert_conditions(V[others], np.asarray(z)[others], a[others],
                          b[others], Q[others], EPS64, f"neighbours {break_it}")


def test_a_non_finite_input_fails_that_cone_only():
    with x64():
        rng = np.random.default_rng(1597)
        V, a, b, Q = _family(rng, 4, 3, "root_neg")
        V[1, 2] = np.nan
        V[3, 0] = np.inf
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        status = np.asarray(status)
        assert status[1] == ss.STATUS_FAILED, "NaN input was accepted"
        assert not np.any(status[[0, 2]] == ss.STATUS_FAILED)
        assert np.all(np.isfinite(np.asarray(z)[[0, 2]]))


# ---------------------------------------------------------------------------
# shapes are refused, not reinterpreted
# ---------------------------------------------------------------------------
def test_unsupported_layouts_are_refused():
    with x64():
        eye = jnp.eye(3, dtype=jnp.float64)
        b3 = jnp.ones((3,), jnp.float64)
        with pytest.raises(ValueError, match="both be shared or both be per cone"):
            ss.SpatialFullSOCMetric(1.0, b3, jnp.tile(eye, (2, 1, 1)))
        with pytest.raises(ValueError, match="both be shared or both be per cone"):
            ss.SpatialFullSOCMetric(1.0, jnp.ones((2, 3), jnp.float64), eye)
        with pytest.raises(ValueError, match="to match b"):
            ss.SpatialFullSOCMetric(1.0, b3, jnp.eye(4, dtype=jnp.float64))
        with pytest.raises(ValueError, match="to match b"):
            ss.SpatialFullSOCMetric(1.0, jnp.ones((2, 3), jnp.float64),
                                    jnp.ones((2, 3, 4), jnp.float64))
        with pytest.raises(ValueError, match=r"a must be \(2,\) to match b"):
            ss.SpatialFullSOCMetric(jnp.ones((3,), jnp.float64),
                                    jnp.ones((2, 3), jnp.float64),
                                    jnp.tile(eye, (2, 1, 1)))
        with pytest.raises(ValueError, match="Q has 3 cones and b has 2"):
            ss.SpatialFullSOCMetric(jnp.ones((2,), jnp.float64),
                                    jnp.ones((2, 3), jnp.float64),
                                    jnp.tile(eye, (3, 1, 1)))
        metric = ss.SpatialFullSOCMetric(1.0, b3, eye)
        with pytest.raises(ValueError, match="k = 4, not 3"):
            ss.project(jnp.ones((3,), jnp.float64), metric)
        with pytest.raises(ValueError, match=r"\(k,\) or \(cones, k\)"):
            ss.project(jnp.ones((2, 2, 4), jnp.float64), metric)
        per_cone = ss.SpatialFullSOCMetric(jnp.ones((2,), jnp.float64),
                                          jnp.ones((2, 3), jnp.float64),
                                          jnp.tile(eye, (2, 1, 1)))
        with pytest.raises(ValueError, match="carries 2 cones, not 5"):
            ss.project(jnp.ones((5, 4), jnp.float64), per_cone)


def test_the_supplied_factors_are_not_modified():
    """A supplied metric is a checked object: it is used as given."""
    with x64():
        rng = np.random.default_rng(2718)
        a, b, Q = _one_metric(rng, 6)
        # a Q that is orthogonal only to a tolerance, which must NOT be repaired
        Q = Q + 1e-7 * rng.standard_normal(Q.shape)
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        before = (np.asarray(metric.a).copy(), np.asarray(metric.b).copy(),
                  np.asarray(metric.Q).copy())
        v = _one_case(rng, 6, "root_neg", a, b, Q)
        ss.project(jnp.asarray(v, jnp.float64), metric)
        assert np.array_equal(np.asarray(metric.a), before[0])
        assert np.array_equal(np.asarray(metric.b), before[1])
        assert np.array_equal(np.asarray(metric.Q), before[2]), \
            "the supplied Q was reorthogonalized behind the caller's back"


def test_imperfect_orthogonality_shows_up_in_the_reported_defects():
    """Rotation error is part of the contract, so it must be visible."""
    with x64():
        rng = np.random.default_rng(1123)
        d = 8
        a, b, Q0 = _one_metric(rng, d, conditioning=10.0)
        worst = {}
        for scale in (0.0, 1e-10, 1e-6, 1e-3):
            Q = Q0 + scale * rng.standard_normal(Q0.shape)
            v = _one_case(rng, d, "root_pos", a, b, Q)
            metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                             jnp.asarray(b, jnp.float64),
                                             jnp.asarray(Q, jnp.float64))
            z, status = ss.project(jnp.asarray(v, jnp.float64), metric)
            assert int(status) != ss.STATUS_FAILED
            out = kkt(v, z, a, b, Q)
            worst[scale] = (float(out["primal"][0] / out["primal_floor"][0]),
                            float(out["polar"][0] / out["polar_floor"][0]),
                            float(out["complementarity"][0]
                                  / out["complementarity_floor"][0]))
            print(f"orthogonality {scale:.0e}, as multiples of unit roundoff: "
                  f"primal {worst[scale][0] / EPS64:.3e} "
                  f"polar {worst[scale][1] / EPS64:.3e} "
                  f"complementarity {worst[scale][2] / EPS64:.3e}")
        # The primal and polar defects are ONE-SIDED, so a perturbation that
        # moves the point inward leaves them at zero and they cannot carry this.
        # The complementarity is two-sided, and it is what has to grow.
        assert worst[0.0][2] <= SAFETY * EPS64, \
            "an exact Q must leave the complementarity at its roundoff floor"
        assert (worst[1e-10][2] < worst[1e-6][2] < worst[1e-3][2]), \
            f"a non-orthogonal Q left no trace in the defects: {worst}"
        assert worst[1e-3][2] > 1e4 * EPS64, \
            "a Q wrong in the third digit must be visible above roundoff"


# ---------------------------------------------------------------------------
# the kernel is jittable, vmappable and deterministic
# ---------------------------------------------------------------------------
def test_jit_vmap_and_determinism():
    with x64():
        rng = np.random.default_rng(8191)
        d = 5
        V, a, b, Q = _family(rng, 16, d, "root_neg")
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))

        def run(v):
            return ss.project(v, metric)[0]

        plain = np.asarray(run(jnp.asarray(V, jnp.float64)))
        jitted = np.asarray(jax.jit(run)(jnp.asarray(V, jnp.float64)))
        # Bitwise only WITHIN one transform. Across transforms the backend is
        # free to fuse differently, which is why the claim here is agreement to
        # roundoff and determinism on repetition, not one number for both.
        rel = float(np.abs(plain - jitted).max()
                    / max(1e-300, np.abs(plain).max()))
        print(f"jit against eager: worst relative difference {rel:.3e}")
        assert rel <= 1e-14, f"jit moved the point by {rel:.3e}"
        again = np.asarray(jax.jit(run)(jnp.asarray(V, jnp.float64)))
        assert np.array_equal(jitted, again), "the kernel is not deterministic"
        third = np.asarray(run(jnp.asarray(V, jnp.float64)))
        assert np.array_equal(plain, third), "the eager path is not deterministic"

        shared = ss.SpatialFullSOCMetric(jnp.asarray(a[0], jnp.float64),
                                         jnp.asarray(b[0], jnp.float64),
                                         jnp.asarray(Q[0], jnp.float64))
        stack = jnp.asarray(np.stack([V, V + 0.5, V - 0.25]), jnp.float64)
        mapped = jax.vmap(lambda M: ss.project(M, shared)[0])(stack)
        for i in range(3):
            one = np.asarray(ss.project(stack[i], shared)[0])
            rel = float(np.abs(np.asarray(mapped[i]) - one).max()
                        / max(1e-300, np.abs(one).max()))
            assert rel <= 1e-14, \
                f"vmap disagreed with the direct call on slab {i} by {rel:.3e}"


def test_the_step_counts_are_fixed_and_named():
    """A bounded, deterministic root solve is part of the contract."""
    assert isinstance(ss.BISECTION_STEPS, int) and ss.BISECTION_STEPS >= 30
    assert isinstance(ss.EXPANSION_STEPS, int) and ss.EXPANSION_STEPS >= 8
    assert set(ss.STATUS_NAMES) == {ss.STATUS_INSIDE, ss.STATUS_POLAR,
                                    ss.STATUS_ROOT, ss.STATUS_APEX,
                                    ss.STATUS_FAILED, ss.STATUS_POLAR_LIMIT}
    assert ss.STATUS_FAILED not in ss.STATUS_ACCEPTED
    assert set(ss.STATUS_ACCEPTED) | {ss.STATUS_FAILED} == set(ss.STATUS_NAMES)


def test_no_case_next_to_the_polar_boundary_is_ever_refused():
    """Walk up to the polar boundary one float32 step at a time.

    This is the property that matters and the reason the fold exists: a cone one
    representable step off the polar boundary must come back with an answer, on
    every backend, and must not be refused because a dot product associated
    differently. Which branch answers it is allowed to vary; that it answers is
    not. Where the fold does fire, its stated bound
    ``||z*|| <= ||bw|| (1 - kappa) / cmin`` is checked against the exact root.
    """
    d = 6
    rng = np.random.default_rng(90210)
    a, b, Q = _one_metric(rng, d, conditioning=10.0)
    y = rng.standard_normal(d)
    yt = Q.T @ y
    norm_bw = float(np.linalg.norm(b * yt))
    folded = 0
    for steps in (0, 1, 2, 4, 8, 64, 4096):
        # a t just above -||bw||, by a few float32 steps: not polar, but as close
        # to it as float32 can express
        target = np.float32(-norm_bw)
        for _ in range(steps):
            target = np.nextafter(target, np.float32(np.inf))
        t = float(target) / a
        v = np.concatenate([[t], y])
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float32),
                                         jnp.asarray(b, jnp.float32),
                                         jnp.asarray(Q, jnp.float32))
        z, status = ss.project(jnp.asarray(v, jnp.float32), metric)
        assert int(status) != ss.STATUS_FAILED, \
            f"{steps} steps off the polar boundary was refused"
        if int(status) == ss.STATUS_POLAR_LIMIT:
            folded += 1
            assert np.array_equal(np.asarray(z), np.zeros(d + 1)), \
                "the fold must return exactly zero"
            with x64():
                z_star, _theta = exact_point(t, y, a, b, Q)
            kappa = abs(a * t) / norm_bw
            allowance = norm_bw * abs(1.0 - kappa) / (a + float(b.min()))
            got = float(np.linalg.norm(z_star))
            print(f"{steps} steps: folded, ||z*|| {got:.3e} against the stated "
                  f"bound {allowance:.3e}, ||v|| {np.linalg.norm(v):.3e}")
            assert got <= allowance * (1.0 + 1e-9), \
                "the fold discarded more than its bound allows"
        else:
            print(f"{steps} steps: {ss.STATUS_NAMES[int(status)]}")
    # The fold is a guard against a marginal sign, so whether it fires on THIS
    # backend with THIS data is not something to assert. Its behaviour is tested
    # through its seam below, and what is asserted here is that nothing was
    # refused.
    print(f"the polar-limit fold fired {folded} times on this backend")


def test_the_polar_limit_fold_returns_zero_and_touches_nothing_else():
    """The fold's behaviour, exercised at its seam rather than by float luck.

    ``1 - kappa`` reaching its roundoff floor makes the sign of ``G`` at the far
    end of the bracket marginal, and on the backends measured here it has come
    out negative, so the fold does not fire. A branch that cannot be reached is a
    branch that cannot be trusted, so it is driven directly: one cone is reported
    unreachable and the fold's contract is checked on it.
    """
    with x64():
        rng = np.random.default_rng(555111)
        V, a, b, Q = _family(rng, 5, 4, "root_neg")
        metric = ss.SpatialFullSOCMetric(jnp.asarray(a, jnp.float64),
                                         jnp.asarray(b, jnp.float64),
                                         jnp.asarray(Q, jnp.float64))
        plain, plain_status = ss.project(jnp.asarray(V, jnp.float64), metric)
        assert np.all(np.asarray(plain_status) == ss.STATUS_ROOT)

        real_brackets = ss._brackets

        def one_cone_unreachable(*args, **kwargs):
            lo, up, established = real_brackets(*args, **kwargs)
            return lo, up, established.at[2].set(False)

        ss._brackets = one_cone_unreachable
        try:
            z, status = ss.project(jnp.asarray(V, jnp.float64), metric)
        finally:
            ss._brackets = real_brackets
        status = np.asarray(status)
        assert status[2] == ss.STATUS_POLAR_LIMIT, ss.STATUS_NAMES[int(status[2])]
        assert np.array_equal(np.asarray(z)[2], np.zeros(5)), \
            "the fold must return exactly zero"
        others = [0, 1, 3, 4]
        assert np.all(status[others] == ss.STATUS_ROOT), status
        assert np.array_equal(np.asarray(z)[others],
                              np.asarray(plain)[others]), \
            "one cone's fold disturbed another cone's answer"
        # and a NON-finite input must still be REFUSED and never folded
        Vn = V.copy()
        Vn[2, 1] = np.nan
        ss._brackets = one_cone_unreachable
        try:
            zn, statusn = ss.project(jnp.asarray(Vn, jnp.float64), metric)
        finally:
            ss._brackets = real_brackets
        assert np.asarray(statusn)[2] == ss.STATUS_FAILED, \
            "a NaN cone was folded into an answer"
        assert np.all(np.isnan(np.asarray(zn)[2]))
