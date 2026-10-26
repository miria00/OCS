"""PSD witnesses need a resolved sign (0.3.6 patch, 2026-10-23).

The objective-bound evaluator decides membership of PSD-dual blocks and of a
PSD primal block's coupling with ``PSDCone.conj_in_domain_resolved``: the
float32 test ``conj_in_domain`` AND, on the represented float32 values,
lambda_max(sym) <= -c k eps32 ||sym||_2 in scoped float64.  The ray search's
chosen point is re-validated as materialized, through its own adjoint.

The controls are the saved bytes of four boundary cases in which 0.3.4/0.3.6
admitted a PSD witness that exact rational arithmetic proves to be OUTSIDE
the cone, and of the boundary packet whose production and recheck verdicts
disagreed (``tests/reference/psd_signfix_controls.npz``, provenance in
``tests/reference/PROVENANCE.md``).  Stricter, never looser: every admission
must be exactly valid, and an unresolved sign is refused.
"""
import os
from fractions import Fraction as Fr

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.prox.base import _Scaled
from ocs.prox.cones import PSD_RESOLVED_SIGN_C, PSDCone, SecondOrderCone
from ocs.prox.library import NonNeg

EPS = float(np.finfo(np.float32).eps)
HERE = os.path.dirname(__file__)
CONTROLS = os.path.join(HERE, "reference", "psd_signfix_controls.npz")


# --------------------------------------------------------------------------
# exact rational helpers (every float32 value is an exact dyadic rational)
# --------------------------------------------------------------------------
def _F(v):
    return Fr(float(v))


def _ldl_pd(A):
    n = len(A); L = [[Fr(0)] * n for _ in range(n)]; D = [Fr(0)] * n
    for j in range(n):
        D[j] = A[j][j] - sum(L[j][q] ** 2 * D[q] for q in range(j))
        if D[j] <= 0:
            return False
        for i in range(j + 1, n):
            L[i][j] = (A[i][j] - sum(L[i][q] * L[j][q] * D[q] for q in range(j))) / D[j]
    return True


def _exactly_negative_definite(vec_rational, k):
    S = [[(vec_rational[i * k + j] + vec_rational[j * k + i]) / 2 for j in range(k)] for i in range(k)]
    return _ldl_pd([[-S[i][j] for j in range(k)] for i in range(k)])


def _resolved64(vec32, k, factor=1.0):
    M = np.asarray(vec32, np.float32).astype(np.float64).reshape(k, k)
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    return bool(w[-1] <= -factor * PSD_RESOLVED_SIGN_C * k * EPS * np.max(np.abs(w)))


def _family_A(k):                     # PSD dual-handled
    x = ocs.Variable(k * k)
    return ocs.minimize([ocs.sum_squares(x), ocs.psd(x, k)], cost=ocs.Parameter("nv")).compile(
        max_blocks=0, tol=1e-5)


def _family_B(k):                     # PSD primal-handled: coupling z = V - y
    x = ocs.Variable(k * k)
    return ocs.minimize([ocs.psd(x, k), ocs.sum_squares(x)], cost=ocs.Parameter("nv")).compile(
        max_blocks=0, tol=1e-5)


def _maxcut(c):
    k = int(round(np.sqrt(c.size)))
    op = ocs.LinearOperator.from_forward(lambda v: jnp.diag(v.reshape(k, k)), n=k * k, m=k)
    x = ocs.Variable(k * k)
    return k, ocs.minimize([ocs.psd(x, k), ocs.equality(op @ x - ocs.Parameter("b"))], cost=c).compile(
        max_blocks=0, tol=1e-3)


@pytest.fixture(scope="module")
def controls():
    return dict(np.load(CONTROLS))


# --------------------------------------------------------------------------
# the membership method
# --------------------------------------------------------------------------
def test_resolved_membership_is_never_looser_and_refuses_unresolved_signs():
    blk = PSDCone(2)
    cases = {
        "zero": ([0.0, 0.0, 0.0, 0.0], True, True),               # ||sym|| = 0: exactly in the cone
        "minus_identity": ([-1.0, 0.0, 0.0, -1.0], True, True),
        "singular_boundary": ([0.0, 0.0, 0.0, -2.0], True, False),  # lambda_max = 0 exactly: unresolved
        "tiny_positive": ([1e-9, 0.0, 0.0, -2.0], False, False),
        "skew_only_part": ([-1.0, 5.0, -5.0, -1.0], True, True),   # skew part is irrelevant
    }
    for name, (v, f32, resolved) in cases.items():
        y = jnp.asarray(v, jnp.float32)
        assert bool(blk.conj_in_domain(y)) is f32, name
        assert bool(blk.conj_in_domain_resolved(y)) is resolved, name
        assert bool(jax.jit(blk.conj_in_domain_resolved)(y)) is resolved, name
    # the envelope boundary, on exactly represented data: diag(-t, -1)
    gamma = PSD_RESOLVED_SIGN_C * 2 * EPS
    inside = jnp.asarray([-4.0 * gamma, 0.0, 0.0, -1.0], jnp.float32)
    within = jnp.asarray([-0.5 * gamma, 0.0, 0.0, -1.0], jnp.float32)
    assert bool(blk.conj_in_domain_resolved(inside))
    assert not bool(blk.conj_in_domain_resolved(within))
    assert bool(blk.conj_in_domain(within))                     # the float32 test alone admits it
    # scale invariance (powers of two are exact)
    for s in (2.0 ** -40, 2.0 ** -3, 2.0 ** 7, 2.0 ** 60):
        assert bool(blk.conj_in_domain_resolved(inside * s))
        assert not bool(blk.conj_in_domain_resolved(within * s))
    # batched: same decisions under vmap
    batch = jnp.stack([inside, within, jnp.zeros(4, jnp.float32)])
    with jax.enable_x64():
        got = np.asarray(jax.vmap(blk.conj_in_domain_resolved)(batch))
    assert got.tolist() == [True, False, True]


def test_only_psd_blocks_and_their_wrappers_take_the_resolved_test():
    assert PSDCone(3).resolved_sign_domain
    assert _Scaled(PSDCone(2), 2.5).resolved_sign_domain
    assert not bool(_Scaled(PSDCone(2), 2.5).conj_in_domain_resolved(
        jnp.asarray([0.0, 0.0, 0.0, -2.0], jnp.float32)))
    assert "resolved_sign_domain" not in vars(_Scaled(PSDCone(2), 2.5))   # wrappers keep their fields
    assert not SecondOrderCone().resolved_sign_domain
    assert not NonNeg().resolved_sign_domain
    x = ocs.Variable(3)
    lp = ocs.minimize([ocs.nonneg(x), ocs.equality(ocs.Dense(np.ones((1, 3), np.float32)) @ x - np.ones(1, np.float32))],
                      cost=np.asarray([1.0, 2.0, 3.0], np.float32)).compile(max_blocks=0)
    assert lp.meta["psd_resolved_sign"] == {"active": False}
    rep = lp.objective_bound(jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([-1.0]), {})
    # Materialized-point revalidation now applies to every recovery ray;
    # PSD resolved-sign membership remains exclusive to PSD blocks above.
    assert rep["operator_applications_breakdown"]["ray_search_materialized_revalidation"] == 3
    assert sum(rep["operator_applications_breakdown"].values()) == rep["operator_applications"]


# --------------------------------------------------------------------------
# controls on saved bytes: the boundary admissions are closed
# --------------------------------------------------------------------------
def test_b_seed300_gpu_witness_no_longer_passes(controls):
    k = 3
    x, y, V = controls["b300_x"], controls["b300_y"], controls["b300_V"]
    Vf = V.reshape(-1)
    z = [_F(a) - _F(b) for a, b in zip(Vf, y)]
    assert not _exactly_negative_definite(z, k)                 # the saved coupling is outside (exact)
    comp, data = _family_B(k), {"nv": -Vf}
    v = comp.validate(jnp.asarray(x), jnp.asarray(y), data)
    assert not bool(np.asarray(v["objective_lower_bound_available"]))
    assert "coupling" in str(v["supplied_dual_status"])
    assert not comp.acceptance(v)["target_met"]
    rep = comp.objective_bound(jnp.asarray(x), jnp.asarray(y), data, return_dual=True, recover=True)
    if bool(np.asarray(rep["objective_lower_bound_available"])):
        cert = np.asarray(rep["certifying_dual"])
        zc = [_F(a) - _F(b) for a, b in zip(Vf, cert)]
        assert _exactly_negative_definite(zc, k)
        assert _resolved64(np.asarray(Vf, np.float32) - cert, k)
        # the solve-time certificate re-checks with recovery off
        assert bool(np.asarray(comp.validate(jnp.asarray(x), jnp.asarray(cert), data)["objective_lower_bound_available"]))


@pytest.mark.parametrize("which", ["a301_check2", "a302_cpu"])
def test_a_boundary_witnesses_are_no_longer_admitted(controls, which):
    k = 3
    x, y, V = controls[which + "_x"], controls[which + "_y"], controls[which + "_V"]
    assert not _exactly_negative_definite([_F(v) for v in y], k)  # outside the polar cone, exactly
    comp, data = _family_A(k), {"nv": -V.reshape(-1)}
    v = comp.validate(jnp.asarray(x), jnp.asarray(y), data)
    assert not bool(np.asarray(v["objective_lower_bound_available"]))
    assert "outside dom f_i^*" in str(v["supplied_dual_status"])


def test_maxcut_10_1002_no_longer_passes_with_an_infeasible_certificate(controls):
    c = controls["maxcut1002_c"]
    k, comp = _maxcut(c)
    data = {"b": jnp.ones(k, jnp.float32)}
    x, y = controls["maxcut1002_x"], controls["maxcut1002_y_iterate"]
    rep = comp.objective_bound(jnp.asarray(x), jnp.asarray(y), data, return_dual=True)
    if not bool(np.asarray(rep["objective_lower_bound_available"])):
        return                                                  # unverified: acceptable
    cert = np.asarray(rep["certifying_dual"])
    # the coupling for g = PSD:  z = -K^T y - c = -diag(y) - C must be NSD, i.e.
    # C + diag(y) PSD -- proved exactly on the certificate's represented numbers
    C = c.reshape(k, k)
    M = [[_F(C[i, j]) + (_F(cert[i]) if i == j else 0) for j in range(k)] for i in range(k)]
    S = [[(M[i][j] + M[j][i]) / 2 for j in range(k)] for i in range(k)]
    assert _ldl_pd(S)
    assert bool(np.asarray(comp.validate(jnp.asarray(x), jnp.asarray(cert), data)["objective_lower_bound_available"]))


def test_the_ninth_packet_is_never_a_false_pass_and_is_now_refused_as_unresolved(controls, record_testsuite_property):
    k = 8
    x, y, V = controls["b800_x"], controls["b800_y"], controls["b800_V"]
    Vf = V.reshape(-1)
    z = [_F(a) - _F(b) for a, b in zip(Vf, y)]
    assert _exactly_negative_definite(z, k)                     # exactly feasible ...
    assert not _resolved64(np.asarray(Vf - y, np.float32), k)   # ... but inside the envelope
    comp, data = _family_B(k), {"nv": -Vf}
    v = comp.validate(jnp.asarray(x), jnp.asarray(y), data)
    # The new verdict, the same on every host: the sign is unresolved, so the
    # supplied witness is refused (a conservative refusal, never a false pass).
    assert not bool(np.asarray(v["objective_lower_bound_available"]))
    assert "coupling" in str(v["supplied_dual_status"])
    rep = comp.objective_bound(jnp.asarray(x), jnp.asarray(y), data, return_dual=True, recover=True)
    record_testsuite_property("ninth_packet_recovery_on_source", str(rep.get("source")))
    if bool(np.asarray(rep["objective_lower_bound_available"])):
        cert = np.asarray(rep["certifying_dual"])
        zc = [_F(a) - _F(b) for a, b in zip(Vf, cert)]
        assert _exactly_negative_definite(zc, k)
        assert _resolved64(np.asarray(Vf, np.float32) - cert, k)
        assert bool(np.asarray(comp.validate(jnp.asarray(x), jnp.asarray(cert), data)["objective_lower_bound_available"]))


# --------------------------------------------------------------------------
# a recovered certificate is what a recheck admits; feasible witnesses pass
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k,seed", [(3, 300), (5, 501), (8, 800)])
def test_recovered_certificates_recheck_and_validate_exactly(k, seed):
    r = np.random.default_rng(100 * k + seed % 100)
    V = r.normal(size=(k, k)).astype("f"); V = (V + V.T) / 2
    w, U = np.linalg.eigh(V.astype(np.float64))
    Pplus = (U * np.maximum(w, 0)) @ U.T
    data = {"nv": -V.reshape(-1)}
    for fam, y0 in ((_family_A(k), (U * np.minimum(w, 0)) @ U.T + 1e-4 * np.eye(k)),   # A: dual just outside
                    (_family_B(k), Pplus - 1e-4 * np.eye(k))):                          # B: coupling just outside
        x = Pplus.astype(np.float32).reshape(-1)
        y = y0.astype(np.float32).reshape(-1)
        rep = fam.objective_bound(jnp.asarray(x), jnp.asarray(y), data, return_dual=True)
        assert fam.meta["psd_resolved_sign"]["active"]
        bd = rep["operator_applications_breakdown"]
        assert bd["ray_search_materialized_revalidation"] in (2, 3)
        assert sum(bd.values()) == rep["operator_applications"]
        assert bool(np.asarray(rep["objective_lower_bound_available"]))
        cert = np.asarray(rep["certifying_dual"])
        v = fam.validate(jnp.asarray(x), jnp.asarray(cert), data)
        assert bool(np.asarray(v["objective_lower_bound_available"]))          # re-checks
        np.testing.assert_allclose(float(np.asarray(v["objective_lower_bound"])),
                                   float(np.asarray(rep["objective_lower_bound"])), rtol=1e-6)


def test_a_strictly_feasible_witness_is_still_accepted():
    k = 4
    r = np.random.default_rng(4)
    V = r.normal(size=(k, k)).astype("f"); V = (V + V.T) / 2
    data = {"nv": -V.reshape(-1)}
    y = (-np.eye(k)).astype(np.float32).reshape(-1)            # well inside the polar cone
    comp = _family_A(k)
    v = comp.validate(jnp.asarray(np.zeros(k * k, np.float32)), jnp.asarray(y), data)
    assert bool(np.asarray(v["objective_lower_bound_available"]))
    assert v["supplied_dual_status"].startswith("available")


def test_batched_bounds_agree_with_the_scalar_path():
    k = 3
    r = np.random.default_rng(7)
    V = r.normal(size=(k, k)).astype("f"); V = (V + V.T) / 2
    comp = _family_B(k)
    w, U = np.linalg.eigh(V.astype(np.float64))
    x = ((U * np.maximum(w, 0)) @ U.T).astype(np.float32).reshape(-1)
    ys = np.stack([(V.astype(np.float64) + np.eye(k)).astype(np.float32).reshape(-1),      # inside
                   (V.astype(np.float64) - 1e-9 * np.eye(k)).astype(np.float32).reshape(-1)])  # at the boundary
    data = {"nv": jnp.stack([-V.reshape(-1)] * 2)}
    # the batched report body (vmap inside the scoped float64 assessment)
    many = comp.certificate_many(jnp.stack([x, x]), jnp.asarray(ys), data)
    for i in range(2):
        one = comp.validate(jnp.asarray(x), jnp.asarray(ys[i]), {"nv": -V.reshape(-1)})
        assert bool(np.asarray(many["objective_lower_bound_available"])[i]) == bool(one["objective_lower_bound_available"])
    # the batched evaluator of solve_many_results (scoped float64 vmap) runs and
    # every lane it certifies re-checks through the scalar recovery-off gate
    x8 = ocs.Variable(k * k)
    fam = ocs.minimize([ocs.psd(x8, k), ocs.sum_squares(x8)], cost=ocs.Parameter("nv")).compile(
        max_blocks=4, tol=1e-5)
    sols = fam.solve_many_results(data)
    for s in sols:
        if s.status == "solved":
            v = fam.validate(s.x, s.y, {"nv": -V.reshape(-1)})
            assert bool(np.asarray(v["objective_lower_bound_available"]))
