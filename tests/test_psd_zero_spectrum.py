"""The PSD cone on the exactly zero matrix (since 0.3.8).

The GPU eigensolver can return NaN for an exactly zero matrix, in some calls
and not in others, and the dual of a strictly feasible PSD block converges to
exactly zero.  A NaN eigenvalue made the conjugate-domain test refuse the zero
dual that had certified a Lyapunov solve (k = 64 to 512 on an RTX 5090).
``sym_eigvalsh`` and ``sym_eigh`` never pass a zero slice to the eigensolver,
and every other slice takes the unchanged call.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.prox.cones import PSDCone, sym_eigh, sym_eigvalsh
from ocs.prox import cone_metric


@pytest.mark.parametrize("k", [16, 64, 128, 512])
def test_zero_matrix_has_zero_spectrum_on_every_call(k):
    Z = jnp.zeros((k, k), jnp.float32)
    jitted = jax.jit(sym_eigvalsh)
    for _ in range(8):
        for w in (sym_eigvalsh(Z), jitted(Z), sym_eigh(Z)[0]):
            assert np.all(np.asarray(w) == 0.0)
    w, U = sym_eigh(Z)
    np.testing.assert_array_equal(np.asarray(U), np.eye(k, dtype=np.float32))
    with jax.enable_x64():
        w64 = sym_eigvalsh(jnp.zeros((k, k), jnp.float64))
        assert w64.dtype == jnp.float64 and np.all(np.asarray(w64) == 0.0)


@pytest.mark.parametrize("k", [16, 128])
def test_zero_dual_is_in_the_polar_cone_on_every_call(k):
    blk = PSDCone(k)
    y = jnp.zeros(k * k, jnp.float32)
    for _ in range(8):
        assert bool(blk.conj_in_domain(y))
        assert bool(blk.conj_in_domain_resolved(y))
        assert bool(jax.jit(blk.conj_in_domain_resolved)(y))
        assert float(blk.conj_value(y)) == 0.0
    assert np.all(np.asarray(blk.conj_domain_project(y)) == 0.0)
    assert np.all(np.asarray(blk.prox(y, jnp.ones_like(y))) == 0.0)
    assert float(blk.domain_distance(y)) == 0.0
    assert float(blk.value(y)) == 0.0
    M = jnp.zeros((3, k, k), jnp.float32)
    assert np.all(np.asarray(cone_metric.psd_project(M)) == 0.0)


def test_nonzero_slices_are_bit_identical_to_the_plain_call():
    rng = np.random.default_rng(0)
    k = 64
    G = rng.normal(size=(3, k, k)).astype(np.float32)
    S = jnp.asarray(0.5 * (G + np.swapaxes(G, -1, -2)))
    np.testing.assert_array_equal(np.asarray(sym_eigvalsh(S)),
                                  np.asarray(jnp.linalg.eigvalsh(S)))
    w, U = sym_eigh(S)
    w0, U0 = jnp.linalg.eigh(S)
    np.testing.assert_array_equal(np.asarray(w), np.asarray(w0))
    np.testing.assert_array_equal(np.asarray(U), np.asarray(U0))
    mixed = S.at[1].set(0.0)
    wm = np.asarray(sym_eigvalsh(mixed))
    assert np.all(wm[1] == 0.0)
    np.testing.assert_array_equal(wm[[0, 2]], np.asarray(jnp.linalg.eigvalsh(S))[[0, 2]])


def test_nonfinite_slice_still_propagates():
    S = jnp.zeros((4, 4), jnp.float32).at[0, 0].set(jnp.nan)
    assert not np.all(np.isfinite(np.asarray(sym_eigvalsh(S))))
    assert not bool(PSDCone(4).conj_in_domain(S.reshape(-1)))


def test_strictly_feasible_psd_solve_returns_a_dual_that_rechecks():
    """A strictly feasible Lyapunov-type family returns the zero dual, and the
    recovery-off bound on the returned pair accepts it as the supplied
    witness (the ladder's re-check that failed at k = 64 to 512)."""
    k = 16
    rng = np.random.default_rng(3)
    A = (-np.eye(k) + 0.2 * rng.normal(size=(k, k)) / np.sqrt(k)).astype(np.float32)
    z = ocs.Variable(k * k)
    b = ocs.Parameter("b")

    def sym_view(x):
        X = x.reshape(k, k)
        return (0.5 * (X + X.T)).reshape(-1)

    def lyap(x, a=jnp.asarray(A)):
        P = x.reshape(k, k)
        P = 0.5 * (P + P.T)
        M = -a.T @ P
        return (M + M.T).reshape(-1)

    S = ocs.LinearOperator.from_forward(sym_view, n=k * k, m=k * k)
    L = ocs.LinearOperator.from_forward(lyap, n=k * k, m=k * k)
    comp = ocs.minimize([ocs.psd(S @ z - b, k), ocs.psd(L @ z - b, k)]).compile(tol=1e-5)
    data = {"b": jnp.eye(k, dtype=jnp.float32).ravel()}
    s = comp.solve_instance(data)
    assert s.status == "solved"
    for _ in range(4):
        rep = comp.objective_bound(s.x, s.y, data, recover=False)
        assert rep["objective_lower_bound_available"]
        assert comp.acceptance(rep)["target_met"]
