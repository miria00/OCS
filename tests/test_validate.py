"""Validate OCS against SCS through the CVXPY harness.  Run: python tests/test_validate.py"""
import numpy as np


def test_lp():
    import cvxpy as cp
    from ocs.cvxpy_harness import validate_against_cvxpy
    rng = np.random.default_rng(0)
    m, n = 6, 10
    A = rng.normal(size=(m, n)); xt = np.abs(rng.normal(size=n)); b = A @ xt
    c = np.abs(rng.normal(size=n))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c @ x), [A @ x == b, x >= 0])
    r = validate_against_cvxpy(prob)
    print(f"  LP:   OCS={r['ocs_obj']:.4f}  SCS={r['ref_obj']:.4f}  "
          f"rel_err={r['rel_err']:.2e}  gap={r['cert']['gap']:.2e}")
    assert r["rel_err"] < 5e-2, r


def test_socp():
    import cvxpy as cp
    from ocs.cvxpy_harness import validate_against_cvxpy
    rng = np.random.default_rng(1)
    n = 8
    c = rng.normal(size=n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c @ x), [cp.norm(x, 2) <= 1])   # SOCP
    r = validate_against_cvxpy(prob)
    print(f"  SOCP: OCS={r['ocs_obj']:.4f}  SCS={r['ref_obj']:.4f}  "
          f"rel_err={r['rel_err']:.2e}")
    assert r["rel_err"] < 5e-2, r


if __name__ == "__main__":
    print("validating OCS against SCS via CVXPY...")
    test_lp()
    test_socp()
    print("validation passed.")
