"""The PSD cone's feasibility distance, read from its spectrum (since 0.3.8).

``PSDCone.domain_distance(v)`` is ||P(v) - v||_F, the distance the
projection moves v, which the objective-bound evaluator reports as the
block's primal infeasibility.  It is computed from float64 eigenvalues of the
represented float32 matrix, so a strictly interior matrix reads zero, where
the float32 reconstruction of the projection does not.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from ocs.prox.cones import PSDCone


def _reference(v, k):
    M = np.asarray(v, np.float64).reshape(k, k)
    sym, skew = 0.5 * (M + M.T), 0.5 * (M - M.T)
    w = np.linalg.eigvalsh(sym)
    return float(np.sqrt(np.sum(skew ** 2) + np.sum(np.minimum(w, 0.0) ** 2)))


@pytest.mark.parametrize("k", [3, 16, 64])
def test_distance_matches_float64_reference_and_projection(k):
    rng = np.random.default_rng(k)
    blk = PSDCone(k)
    for symmetric in (True, False):
        G = rng.normal(size=(k, k))
        M = 0.5 * (G + G.T) if symmetric else G
        v = jnp.asarray(M.reshape(-1).astype(np.float32))
        d = float(blk.domain_distance(v))
        ref = _reference(v, k)
        assert d >= ref
        assert d <= ref * (1 + 1e-6) + 1e-30
        moved = float(jnp.linalg.norm(blk.prox(v, jnp.ones_like(v)) - v))
        np.testing.assert_allclose(d, moved, rtol=1e-4)


def test_strictly_interior_matrix_reads_zero_at_k_512():
    k = 512
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    M = ((Q * np.geomspace(1e-3, 10.0, k)) @ Q.T).astype(np.float32)
    M = np.triu(M) + np.triu(M, 1).T                     # exactly symmetric in float32
    v = jnp.asarray(M.reshape(-1))
    blk = PSDCone(k)
    assert float(blk.domain_distance(v)) == 0.0
    # the float32 reconstructed projection moves the same interior matrix
    assert float(jnp.linalg.norm(blk.prox(v, jnp.ones_like(v)) - v)) > 0.0


def test_skew_part_and_nonfinite_input():
    blk = PSDCone(2)
    v = jnp.asarray([1.0, 3.0, 1.0, 1.0], jnp.float32)    # sym [[1,2],[2,1]], skew 1
    ref = _reference(v, 2)
    assert float(blk.domain_distance(v)) >= ref
    np.testing.assert_allclose(float(blk.domain_distance(v)), np.sqrt(2.0 + 1.0), rtol=1e-6)
    assert float(blk.domain_distance(jnp.asarray([np.nan, 0., 0., 1.], jnp.float32))) == np.inf
