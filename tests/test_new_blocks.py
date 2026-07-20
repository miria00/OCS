"""Tests for the blocks / paths added in the reliability + tier work:
pinball, logistic (safeguarded prox), the matrix scaling tier, the CG path, and
the multicommodity-flow operators."""
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

import ocs


def _brute_prox(f, v, t):
    return np.array([minimize_scalar(lambda u: f(u) + (u - vi) ** 2 / (2 * t)).x for vi in v])


def test_pinball_prox():
    v = np.linspace(-3, 3, 21).astype("f")
    for tau in (0.1, 0.5, 0.85):
        blk = ocs.Pinball(tau); rho = lambda u: tau * max(u, 0) + (1 - tau) * max(-u, 0)
        got = np.asarray(blk.prox(jnp.asarray(v), 0.7)); ref = _brute_prox(rho, v, 0.7)
        assert np.max(np.abs(got - ref)) < 1e-5, tau
    print("  pinball prox OK")


def test_logistic_prox_all_steps():
    # the safeguarded Newton-bisection must be accurate even for large t
    v = np.linspace(-6, 6, 21).astype("f"); sp = lambda u: np.log1p(np.exp(u))
    blk = ocs.Logistic()
    for t in (0.5, 50.0, 5000.0):
        got = np.asarray(blk.prox(jnp.asarray(v), float(t))); ref = _brute_prox(sp, v, t)
        assert np.max(np.abs(got - ref)) < 1e-4, t
    print("  logistic prox OK at all step sizes")


def test_pwl_prox():
    from ocs.prox.library import PiecewiseLinearConvex
    s = [0.5, 1.0, 2.0, 4.0]; rr = [0.0, -0.25, -1.25, -4.25]
    blk = PiecewiseLinearConvex(s, rr); g = lambda u: max(sj * u + rj for sj, rj in zip(s, rr))
    v = np.linspace(-1, 3, 21).astype("f")
    for t in (0.3, 1.0, 3.0):
        got = np.asarray(blk.prox(jnp.asarray(v), t))
        ref = _brute_prox(g, v, t)
        assert np.max(np.abs(got - ref)) < 1e-5, t
    print("  PWL-convex prox OK")


def test_matrix_tier_prox():
    r = np.random.default_rng(0); k = 6
    Mm = r.normal(size=(k, k)); D = jnp.asarray((Mm @ Mm.T + 2 * np.eye(k)).astype("f"))
    Pm = r.normal(size=(k, k)); P = jnp.asarray((Pm @ Pm.T + 0.5 * np.eye(k)).astype("f"))
    q = jnp.asarray(r.normal(size=k).astype("f")); v = jnp.asarray(r.normal(size=k).astype("f"))
    u = ocs.Quadratic(P, q).prox_matrix(v, D)              # (P+D)u = Dv - q
    assert np.linalg.norm(np.asarray((P + D) @ u - (D @ v - q))) < 1e-4
    u2 = ocs.SumSquares().prox_matrix(v, D)                # (I+D)u = Dv
    assert np.linalg.norm(np.asarray((np.eye(k) + D) @ u2 - D @ v)) < 1e-4
    print("  matrix-tier prox OK")


def test_cg_path():
    # pure least squares solved by the CG path matches a dense solve
    r = np.random.default_rng(0); m, n = 80, 40
    A = r.normal(size=(m, n)).astype("f"); b = r.normal(size=m).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    comp = ocs.minimize([ocs.least_squares(ocs.Dense(A) @ x - bp)]).compile(method="cg", cg_iters=200, tol=1e-10)
    sol = comp.solve_instance({"b": b})
    xref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.linalg.norm(np.asarray(sol.x) - xref) / np.linalg.norm(xref) < 1e-3
    print(f"  CG path OK (matvecs={sol.iters})")


def test_transforms():
    r = np.random.default_rng(0); n = 64
    def adj(op):
        x = jnp.asarray(r.normal(size=n).astype("f")); y = jnp.asarray(r.normal(size=op.shape[0]).astype("f"))
        return abs(float(jnp.dot(op.matvec(x), y)) - float(jnp.dot(x, op.rmatvec(y))))
    # orthonormal: adjoint = inverse (roundtrip too); self-adjoint smoothers: adjoint only
    for op in [ocs.DCTOperator(n), ocs.HaarWavelet(n), ocs.GaussianBlur(n, 2.0), ocs.InverseLaplacian(n, 1.0)]:
        assert adj(op) < 1e-4, type(op).__name__
    x = jnp.asarray(r.normal(size=n).astype("f"))
    for op in [ocs.DCTOperator(n), ocs.HaarWavelet(n)]:                 # orthonormal roundtrip
        assert float(jnp.max(jnp.abs(op.rmatvec(op.matvec(x)) - x))) < 1e-4, type(op).__name__
    print("  transforms (DCT/Haar/GaussianBlur/InverseLaplacian) OK")


def test_multicommodity_adjoint():
    r = np.random.default_rng(0)
    N = r.normal(size=(12, 40)).astype("f")
    for op, n, mm in [(ocs.MultiCommodityFlow(N, 4), 40 * 4, 12 * 4),
                      (ocs.CommoditySum(40, 4), 40 * 4, 40)]:
        xx = jnp.asarray(r.normal(size=n).astype("f")); yy = jnp.asarray(r.normal(size=mm).astype("f"))
        err = abs(float(jnp.dot(op.matvec(xx), yy)) - float(jnp.dot(xx, op.rmatvec(yy))))
        assert err < 1e-3, err
    print("  multicommodity operators adjoint OK")


if __name__ == "__main__":
    print("running OCS new-block tests...")
    test_pinball_prox(); test_logistic_prox_all_steps(); test_matrix_tier_prox()
    test_pwl_prox(); test_cg_path(); test_transforms(); test_multicommodity_adjoint()
    print("all new-block tests passed.")
