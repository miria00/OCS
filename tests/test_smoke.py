"""tests: operator adjoint, the FFT LASSO, and a nonnegative least squares
(constraint handled as a primal indicator block).  Run: python tests/test_smoke.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import ocs


def test_fft_adjoint():
    K = ocs.FFTMask.random(2048, ratio=0.25, seed=3)
    err = K.adjoint_test(jax.random.PRNGKey(0))
    assert err < 1e-3, f"adjoint mismatch {err}"
    print(f"  fft adjoint error = {err:.2e}  OK")


def test_lasso():
    n = 2048
    K = ocs.FFTMask.random(n, ratio=0.25, seed=0)
    x = ocs.Variable(n); y = ocs.Parameter("y")
    prob = ocs.minimize([ocs.l1(x, 1e-3), ocs.least_squares(K @ x - y)])
    comp = prob.compile(max_blocks=80)
    rng = np.random.default_rng(1)
    xt = np.zeros(n); xt[rng.choice(n, n // 50, replace=False)] = rng.normal(size=n // 50)
    ymeas = np.asarray(K.matvec(jnp.asarray(xt, jnp.float32)))
    sol = comp.solve_instance({"y": ymeas})
    assert sol.residual < 1e-3, f"lasso residual {sol.residual}"
    print(f"  lasso residual = {sol.residual:.2e}  OK")


def test_nnls():
    # min 1/2||A x - b||^2  s.t.  x >= 0     (nonneg is a primal indicator block)
    m, n = 60, 40
    rng = np.random.default_rng(2)
    A = rng.normal(size=(m, n)).astype(np.float32)
    xt = np.abs(rng.normal(size=n)).astype(np.float32)
    b = A @ xt
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([ocs.nonneg(x), ocs.least_squares(ocs.Dense(A) @ x - bp)])
    comp = prob.compile(max_blocks=120)
    sol = comp.solve_instance({"b": b})
    xr = np.asarray(sol.x)
    assert sol.residual < 1e-3, f"nnls residual {sol.residual}"
    assert xr.min() > -1e-3, f"nonneg violated: min {xr.min()}"
    print(f"  nnls residual = {sol.residual:.2e}  min(x) = {xr.min():.2e}  OK")


if __name__ == "__main__":
    print("running OCS smoke tests...")
    test_fft_adjoint()
    test_lasso()
    test_nnls()
    print("all smoke tests passed.")
