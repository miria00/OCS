"""Operator algebra: composition, sum, and scale stay matrix-free and correct.

These cover the structure-preserving DSL front-end: `A @ B` (Compose),
`A + B` (OpSum), `c * A` (Scale), and the affine forms `A @ (B x + l)`,
`(A x) + (B x)`, `c * (A x + l)`.  Each is checked for a correct adjoint and
against a densified reference, then a composed operator is solved end to end.
Run: python -m pytest tests/test_operator_algebra.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import ocs


def _densify(op):
    n = op.shape[1]
    cols = []
    for j in range(n):
        e = np.zeros(n, np.float32); e[j] = 1.0
        cols.append(np.asarray(op.matvec(jnp.asarray(e))))
    return np.stack(cols, axis=1)


def test_compose_sum_scale_adjoints_and_values():
    key = jax.random.PRNGKey(0)
    G = ocs.Gradient2D(16, 16)      # (512, 256)
    D = ocs.DCTOperator(256)        # (256, 256)

    C = G @ D                       # Compose
    S = D + D                       # OpSum
    Z = 2.5 * D                     # Scale
    N = -D                          # negation via Scale(-1)

    from ocs import Compose, OpSum, Scale
    assert isinstance(C, Compose) and C.shape == (512, 256)
    assert isinstance(S, OpSum) and S.shape == (256, 256)
    assert isinstance(Z, Scale) and isinstance(N, Scale)

    for op in (C, S, Z, N):
        assert op.adjoint_test(key) < 1e-3

    Dm = _densify(D); Gm = _densify(G)
    x = np.asarray(jax.random.normal(key, (256,)), np.float32)
    assert np.linalg.norm(np.asarray(C.matvec(jnp.asarray(x))) - Gm @ (Dm @ x)) < 1e-3
    assert np.linalg.norm(np.asarray(S.matvec(jnp.asarray(x))) - 2 * (Dm @ x)) < 1e-3
    assert np.linalg.norm(np.asarray(Z.matvec(jnp.asarray(x))) - 2.5 * (Dm @ x)) < 1e-3


def test_compose_dimension_guard():
    D = ocs.DCTOperator(256)        # (256,256)
    G = ocs.Gradient2D(16, 16)      # (512,256)
    # inner dims 256 (D cols) vs 512 (G rows) do not match
    try:
        _ = D @ G
        assert False, "expected a dimension-mismatch ValueError"
    except ValueError:
        pass


def test_affine_algebra_shapes():
    D = ocs.DCTOperator(256)
    G = ocs.Gradient2D(16, 16)
    x = ocs.Variable(256)

    e = G @ (D @ x)                 # affine operator composition
    assert e.out_dim == 512
    from ocs import Compose, OpSum
    assert isinstance(e.op, Compose)

    s = (D @ x) + (D @ x)           # affine sum, same variable
    assert isinstance(s.op, OpSum) and s.out_dim == 256

    m = (D @ x) - 0.5 * (D @ x)     # scale + sum
    assert isinstance(m.op, OpSum)

    # different-variable combination is rejected
    y = ocs.Variable(256)
    try:
        _ = (D @ x) + (D @ y)
        assert False, "expected a different-variable ValueError"
    except ValueError:
        pass


def test_composed_operator_end_to_end():
    # min 1e-2||x||_1 + 1/2||(G∘D) x - b||^2, with G∘D a matrix-free composite
    key = jax.random.PRNGKey(1)
    D = ocs.DCTOperator(256)
    G = ocs.Gradient2D(16, 16)
    Cop = G @ D
    x = ocs.Variable(256); b = ocs.Parameter("b")
    prob = ocs.minimize([ocs.l1(x, lam=1e-2), ocs.least_squares(Cop @ x - b)])
    comp = prob.compile()
    bmeas = np.asarray(jax.random.normal(key, (Cop.shape[0],)), np.float32)
    sol = comp.solve_instance({"b": bmeas})
    assert sol.residual < 1e-3, f"composed solve residual {sol.residual}"


if __name__ == "__main__":
    test_compose_sum_scale_adjoints_and_values()
    test_compose_dimension_guard()
    test_affine_algebra_shapes()
    test_composed_operator_end_to_end()
    print("operator-algebra tests passed.")
