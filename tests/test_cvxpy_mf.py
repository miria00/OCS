"""Matrix-free CVXPY bridge: walker semantics and an end-to-end solve.

Skips when cvxpy (or the reduction APIs the bridge needs) is unavailable in
the running environment; the full validation suites live in
OCS_cvxpy/validate_walker.py and validate_solve_mf*.py.
Run: python -m pytest tests/test_cvxpy_mf.py
"""
import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
try:
    from ocs.cvxpy_mf.solve import solve_mf, _supports_symbolic_perspective
    _HAVE_SOLVE = True
    _HAVE_SYMBOLIC_PERSPECTIVE = _supports_symbolic_perspective()
except Exception:
    _HAVE_SOLVE = False
    _HAVE_SYMBOLIC_PERSPECTIVE = False

from ocs.cvxpy_mf import make_affine_forward, make_affine_operator


def _id_map(variables):
    id_to_col, off = {}, 0
    for v in sorted(variables, key=lambda v: v.id):
        id_to_col[v.id] = off
        off += int(np.prod(v.shape)) if v.shape else 1
    return id_to_col, off


def test_walker_matches_cvxpy_numeric():
    rng = np.random.default_rng(0)
    n, m = 7, 5
    A = rng.normal(size=(m, n)).astype(np.float32)
    kern = rng.normal(size=4).astype(np.float32)
    X = cp.Variable(n)
    exprs = [A @ X + rng.normal(size=m).astype(np.float32),
             cp.convolve(kern, X),
             cp.hstack([X, 2.0 * X])]
    for expr in exprs:
        id_to_col, x_len = _id_map(expr.variables())
        fwd = make_affine_forward([expr.canonical_form[0]], id_to_col, x_len)
        x = rng.normal(size=x_len).astype(np.float32)
        for v in expr.variables():
            v.value = x[id_to_col[v.id]: id_to_col[v.id] + v.shape[0]]
        ours = np.asarray(fwd(x))
        truth = np.asarray(expr.value).flatten(order="F")
        assert np.linalg.norm(ours - truth) < 1e-4 * (1 + np.linalg.norm(truth))


def test_walker_adjoint_identity():
    rng = np.random.default_rng(1)
    n, m = 6, 9
    A = rng.normal(size=(m, n)).astype(np.float32)
    X = cp.Variable(n)
    expr = A @ X - 1.0
    id_to_col, x_len = _id_map(expr.variables())
    mv, rmv, b = make_affine_operator([expr.canonical_form[0]], id_to_col, x_len)
    x = rng.normal(size=x_len).astype(np.float32)
    y = rng.normal(size=m).astype(np.float32)
    lhs = float(np.dot(np.asarray(mv(x)), y))
    rhs = float(np.dot(x, np.asarray(rmv(y))))
    assert abs(lhs - rhs) < 1e-4 * (1 + abs(lhs))


@pytest.mark.skipif(not _HAVE_SOLVE, reason="needs newer cvxpy reduction APIs")
def test_solve_mf_lasso_matches_reference():
    rng = np.random.default_rng(2)
    m, n = 30, 18
    A = rng.normal(size=(m, n))
    b = A @ (rng.random(n) < 0.2) + 0.01 * rng.normal(size=m)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x)))
    ref = prob.solve()
    val, sol = solve_mf(prob, tol=1e-4, max_blocks=800)
    assert abs(val - ref) / (1 + abs(ref)) < 1e-2


@pytest.mark.skipif(not _HAVE_SOLVE, reason="needs newer cvxpy reduction APIs")
def test_matrix_is_never_formed():
    """The no-materialization invariant: stub out CVXPY's matrix constructor
    so any attempt to stuff a matrix raises, then solve end to end."""
    import cvxpy.cvxcore.python.canonInterface as ci

    def _boom(*a, **k):
        raise AssertionError("get_problem_matrix was called: a matrix was formed")

    orig = ci.get_problem_matrix
    ci.get_problem_matrix = _boom
    try:
        rng = np.random.default_rng(4)
        m, n = 25, 15
        A = rng.normal(size=(m, n))
        b = A @ rng.normal(size=n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x)))
        val, sol = solve_mf(prob, tol=1e-4, max_blocks=800)
        assert np.isfinite(val)
    finally:
        ci.get_problem_matrix = orig


def test_sparse_constant_is_applied_without_densification(monkeypatch):
    """A scipy sparse constant in a CVXPY expression must reach the walker's
    matmul paths as coordinate arrays: the forward map and its adjoint
    agree with the sparse product, and the walker never expands the entries."""
    import scipy.sparse as sp
    import jax, jax.numpy as jnp
    from ocs.cvxpy_mf import LinOpWalker
    from ocs.cvxpy_mf.linop_walker import _SparseConstant
    def refuse_dense(*args, **kwargs):
        raise AssertionError("sparse matrix constant was expanded")
    monkeypatch.setattr(_SparseConstant, "todense", refuse_dense)
    rng = np.random.default_rng(3)
    m, n = 20000, 20000                      # dense expansion would be 1.6 GB
    A = sp.random(m, n, density=2e-5, format="csr", random_state=7, dtype=np.float64)
    X = cp.Variable(n)
    for expr in (A @ X, X @ A.T):
        id_to_col, x_len = _id_map(expr.variables())
        lin = expr.canonical_form[0]
        walker = LinOpWalker(id_to_col, x_len)
        # The constant of the mul/rmul node is delivered sparse.
        node = lin
        while node.type not in ("mul", "rmul"):
            node = node.args[0]
        assert isinstance(walker.const_value(node.data, keep_sparse=True), _SparseConstant)
        fwd = make_affine_forward([lin], id_to_col, x_len)
        x = rng.normal(size=x_len).astype(np.float32)
        out = np.asarray(fwd(jnp.asarray(x)))
        want = A @ x if node.type == "mul" else x @ A.T
        assert np.allclose(out, want.astype(np.float32), rtol=2e-5, atol=1e-4)
        # Adjoint by vjp equals the sparse transpose product.
        y = rng.normal(size=out.shape[0]).astype(np.float32)
        _, vjp = jax.vjp(fwd, jnp.asarray(x))
        adj = np.asarray(vjp(jnp.asarray(y))[0])
        want_t = A.T @ y                          # both maps are x -> A x
        assert np.allclose(adj, want_t.astype(np.float32), rtol=2e-5, atol=1e-4)


@pytest.mark.skipif(not _HAVE_SOLVE, reason="solve_mf needs a newer cvxpy")
def test_solve_mf_sparse_constant_matches_dense_constant():
    import scipy.sparse as sp
    from ocs.cvxpy_mf.solve import solve_mf
    rng = np.random.default_rng(5)
    m, n = 60, 40
    A = sp.random(m, n, density=0.1, format="csr", random_state=11).astype(np.float64)
    b = rng.normal(size=m)
    sols = []
    for Aform in (A, A.toarray()):
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(Aform @ x - b) + 0.5 * cp.norm1(x)))
        solve_mf(prob, tol=1e-6)
        sols.append(np.asarray(x.value, dtype=np.float64))
    assert np.allclose(sols[0], sols[1], atol=2e-3)


def test_walker_integer_index_drops_axes():
    """An integer index into a 3-D variable drops that axis in CVXPY; the
    walker must return the LinOp's declared shape (a trace over the wrong
    axes produced a length-3 objective before this was fixed)."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    try:
        X = cp.Variable((2, 3, 3))
        exprs = [cp.trace(X[0]) + cp.trace(X[1]), X[0, 0, 1], X[1, :, 2], cp.sum(X[:, 1, :])]
    except ValueError as exc:                    # CVXPY without N-D expressions
        pytest.skip(f"this CVXPY lacks 3-D variables: {exc}")
    xv = rng.normal(size=(2, 3, 3))
    for e in exprs:
        fwd = make_affine_forward([e.canonical_form[0]], {X.id: 0}, 18)
        X.value = xv
        ref = np.asarray(e.value).reshape(-1, order="F")
        got = np.asarray(fwd(jnp.asarray(xv.reshape(-1, order="F"), jnp.float32)))
        assert got.shape == ref.shape and np.allclose(got, ref, atol=1e-5), str(e)


def test_walker_hstack_is_column_major_concatenation():
    """CVXPY's backends define hstack as the concatenation of the arguments'
    column-major vectors read back in the node's shape, so a (1, 1) block and
    a (1,) block stack (Complex2Real builds exactly that for a 1 x 1 Hermitian
    parameter); numpy's hstack refuses mixed dimensions."""
    import jax.numpy as jnp
    import cvxpy.lin_ops.lin_utils as lu
    a = lu.create_var((1, 1))
    b = lu.create_var((1,))
    node = lu.hstack([a, b], (1, 2))
    fwd = make_affine_forward([node], {a.data: 0, b.data: 1}, 2)
    out = np.asarray(fwd(jnp.asarray([3.0, 5.0], jnp.float32)))
    assert np.allclose(out, [3.0, 5.0])
    # a 2-D pair keeps the ordinary meaning
    X = cp.Variable((2, 2)); y = cp.Variable((2, 2))
    expr = cp.hstack([X, 2 * y])
    id_to_col, x_len = _id_map(expr.variables())
    fwd = make_affine_forward([expr.canonical_form[0]], id_to_col, x_len)
    xv = np.arange(8, dtype=np.float32)
    X.value = xv[id_to_col[X.id]:id_to_col[X.id] + 4].reshape(2, 2, order="F")
    y.value = xv[id_to_col[y.id]:id_to_col[y.id] + 4].reshape(2, 2, order="F")
    assert np.allclose(np.asarray(fwd(jnp.asarray(xv))), np.asarray(expr.value).flatten(order="F"))


def test_matvec_is_exact_next_to_a_large_offset():
    """The linear part is the linearization of the affine map, never
    forward(x) - forward(0): with an offset of 1e5 the latter rounds A x to
    float32's spacing there (7.8e-3) and a 1e-3 entry would vanish."""
    import jax.numpy as jnp
    x = cp.Variable(3)
    expr = x + 1e5 * np.ones(3)
    mv, rmv, b = make_affine_operator([expr.canonical_form[0]], {x.id: 0}, 3)
    xv = jnp.asarray([1e-3, -2e-3, 3e-3], jnp.float32)
    assert np.array_equal(np.asarray(mv(xv)), np.asarray(xv))
    assert np.allclose(np.asarray(b), 1e5)


@pytest.mark.skipif(not _HAVE_SYMBOLIC_PERSPECTIVE,
                   reason="needs the patched symbolic perspective canonicalizer and matrix-free OCS interface")
def test_perspective_is_canonicalized_without_a_matrix():
    """The perspective atom's canonicalizer stuffs an auxiliary problem for
    its solver; under the OCS solver context it takes the symbolic route
    (offsets scaled by s), so no matrix is formed anywhere.  Objective
    against SCS at 1e-6, on CVXPY's own test_quad_atom instance."""
    import cvxpy.cvxcore.python.canonInterface as ci
    from ocs.cvxpy_mf.solve import solve_mf

    def _boom(*a, **k):
        raise AssertionError("get_problem_matrix was called: a matrix was formed")

    def build():
        x = cp.Variable(); s = cp.Variable(nonneg=True)
        obj = cp.perspective(cp.square(x) + 2 * x - 4, s)
        return cp.Problem(cp.Minimize(obj), [s <= .5, x >= 2]), x, s
    prob, _, _ = build()
    ref = prob.solve(solver="SCS", eps=1e-9)
    prob, x, s = build()
    orig = ci.get_problem_matrix
    ci.get_problem_matrix = _boom
    try:
        val, sol = solve_mf(prob, tol=1e-6, max_blocks=2000)
    finally:
        ci.get_problem_matrix = orig
    assert abs(val - ref) < 1e-3 * (1 + abs(ref)), (val, ref)
    assert abs(float(x.value) - 2.0) < 1e-3 and abs(float(s.value) - 0.5) < 1e-3


@pytest.mark.skipif(not _HAVE_SOLVE, reason="solve_mf needs a newer cvxpy")
def test_perspective_without_symbolic_capability_is_explicitly_rejected(monkeypatch):
    """The unsupported path must fail before any auxiliary matrix stuffing."""
    import importlib
    import cvxpy.cvxcore.python.canonInterface as ci
    bridge = importlib.import_module("ocs.cvxpy_mf.solve")
    monkeypatch.setattr(bridge, "_supports_symbolic_perspective", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("unsupported perspective attempted matrix stuffing")

    monkeypatch.setattr(ci, "get_problem_matrix", _boom)
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.perspective(cp.square(x), s)),
                      [s <= 1, x >= 2])
    with pytest.raises(NotImplementedError, match="symbolic perspective canonicalizer"):
        solve_mf(prob)
    with pytest.raises(NotImplementedError, match="no-matrix contract"):
        bridge.compile_mf(prob)


@pytest.mark.skipif(not _HAVE_SOLVE, reason="solve_mf needs a newer cvxpy")
def test_primal_violation_by_operators_only():
    """The guard behind the inaccurate infeasibility verdict: a nearly
    feasible returned point refutes a loose Farkas ray.  Measured with the
    terms' operators and cone projections only."""
    from ocs.cvxpy_mf.solve import _cone_lower, stuff_mf
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1, cp.norm(x, 2) <= 3, x[0] + x[1] == 2.5])
    lowered, _ = _cone_lower(prob)
    st = stuff_mf(lowered, tol=1e-6, max_blocks=200)
    cols = {v.id: c for v, c in ((v, st.id_to_col[v.id]) for v in lowered.variables())}
    xs = np.full(st.x_len, 2.0, np.float32)     # the norm's epigraph variable: 2 >= |x| = 1.8
    xs[cols[x.id]:cols[x.id] + 2] = [1.0, 1.5]
    assert st.primal_violation(xs, {}) < 1e-6
    xs[cols[x.id]:cols[x.id] + 2] = [5.0, 5.0]
    assert st.primal_violation(xs, {}) > 0.1


@pytest.mark.skipif(not _HAVE_SOLVE, reason="solve_mf needs a newer cvxpy")
def test_parameter_times_variable_is_evaluated_one_shot():
    """A parameter that multiplies the variable changes the operator.  The
    one-shot ``solve_mf`` substitutes the parameters' current values into
    the tree (CVXPY's EvalParams, as for non-DPP problems) and rebuilds the
    closure on the next call, so two solves with different values give the
    objectives SCS gives; the compile-once family keeps rejecting it."""
    from ocs.cvxpy_mf.solve import compile_mf, solve_mf
    rng = np.random.default_rng(6)
    m, k, n = 3, 4, 5
    P = cp.Parameter((m, k))
    X = cp.Variable((k, n))
    target = rng.normal(size=(m, n))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target) + 0.1 * cp.sum_squares(X)))
    assert prob.is_dpp()
    points = []
    for value in (rng.normal(size=(m, k)), 3.0 * rng.normal(size=(m, k))):
        P.value = value
        ref = prob.solve(solver="SCS", eps=1e-9)
        val, _ = solve_mf(prob, tol=1e-6, max_blocks=2000)
        assert abs(val - ref) < 1e-3 * (1 + abs(ref)), (val, ref)
        points.append(np.array(X.value))
    assert not np.allclose(points[0], points[1])   # the operator was rebuilt
    with pytest.raises(NotImplementedError, match="parameter multiplies the variable"):
        compile_mf(prob)


@pytest.mark.skipif(not _HAVE_SOLVE or "OCS" not in cp.installed_solvers(),
                    reason="needs the CVXPY clone with the OCS interface")
def test_cvxpy_interface_reoptimizes_a_parametric_operator():
    """``problem.solve(solver="OCS")`` twice with different values of a
    parameter that multiplies the variable: each solve rebuilds the operator
    from the current value (objective against SCS)."""
    P = cp.Parameter((2, 2))
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ x - np.ones(2))), [x >= 1])
    for value in (np.eye(2), 2.0 * np.eye(2)):
        P.value = value
        ref = prob.solve(solver="SCS", eps=1e-9)
        val = prob.solve(solver="OCS", tol=1e-6, max_blocks=2000)
        assert abs(val - ref) < 1e-3 * (1 + abs(ref)), (val, ref)
        assert prob.solution.attr.get("ocs_operator_parameters_evaluated") is True
    # an offset-only parameter keeps the same-operator regime
    b = cp.Parameter(2, value=np.ones(2))
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(x - b)), [x >= 0])
    prob2.solve(solver="OCS", tol=1e-6, max_blocks=2000)
    assert prob2.solution.attr.get("ocs_operator_parameters_evaluated") is False
