"""Sparse CVXPY constants without an experimental sparse dependency."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from ocs.cvxpy_mf.linop_walker import _sparse_constant, LinOpWalker


def _matrix(empty):
    # Rectangular, unsorted, repeated coordinates, explicit zero, empty rows.
    rows = np.array([4, 0, 4, 1, 0, 6, 4])
    cols = np.array([3, 1, 3, 0, 1, 4, 2])
    values = np.array([2., 3., -1., 0., 5., -.25, 4.], np.float32)
    return sp.coo_matrix((7, 5), dtype=np.float32) if empty else sp.coo_matrix(
        (values, (rows, cols)), shape=(7, 5))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_products_jit_and_adjoint(empty, right, rank, monkeypatch):
    matrix = _matrix(empty)
    dense = matrix.toarray().astype(np.float64)
    constant = _sparse_constant(matrix)
    def refuse_dense(*a, **kw):
        raise AssertionError("matrix product expanded the sparse constant")
    monkeypatch.setattr(type(constant), "todense", refuse_dense)
    size = 7 if right else 5
    shape = ((size,) if rank == 1 else
             (3, size) if right and rank == 2 else
             (size, 3) if rank == 2 else
             (2, 3, size) if right else (2, size, 3))
    x = np.random.default_rng(18).normal(size=shape).astype(np.float32)
    expected = x @ dense if right else dense @ x
    fn = constant.rmatmul if right else constant.matmul
    result = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    y = np.random.default_rng(23).normal(size=result.shape).astype(np.float32)
    adjoint = jax.jit(lambda z: jax.vjp(fn, jnp.asarray(x))[1](z)[0])
    actual = np.asarray(adjoint(jnp.asarray(y)))
    reference = y @ dense.T if right else dense.T @ y
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)


def test_sparse_vmap_matches_individual_product():
    matrix = _matrix(False)
    const = _sparse_constant(matrix)
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    out = jax.jit(jax.vmap(const.matmul))(jnp.asarray(x))
    np.testing.assert_allclose(out, (matrix @ x.T).T)


def test_scalar_and_constant_product_fallbacks():
    matrix = _matrix(False)
    const = _sparse_constant(matrix)
    dense = matrix.toarray()
    for left, right in ((const, jnp.float32(2)), (jnp.float32(2), const)):
        np.testing.assert_array_equal(LinOpWalker._matmul(left, right, (7, 5)), 2*dense)
    other = _sparse_constant(matrix.T)
    np.testing.assert_allclose(LinOpWalker._matmul(const, other, (7, 7)), dense @ dense.T)


@pytest.mark.parametrize("right", [False, True])
def test_wrong_product_shape_is_refused(right):
    constant = _sparse_constant(_matrix(False))
    fn = constant.rmatmul if right else constant.matmul
    with pytest.raises(ValueError, match="incompatible shape"):
        fn(jnp.ones(3, jnp.float32))


def test_sparse_index_range_is_refused_before_downcast():
    huge = sp.coo_matrix((1, 2**31), dtype=np.float32)
    with pytest.raises(ValueError, match="int32 shape"):
        _sparse_constant(huge)


def test_bridge_works_when_experimental_sparse_imports_are_forbidden():
    # A fresh interpreter catches lazy imports even if other tests have
    # already imported JAX. Record attempts even when production catches them.
    import os
    import subprocess
    import sys
    code = r'''
import importlib.abc, sys
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("jax.experimental.sparse", "jax.sparse.experimental")):
            attempts.append(fullname)
            raise ImportError("forbidden experimental sparse import")
sys.meta_path.insert(0, Guard())
import jax, jax.numpy as jnp, numpy as np, scipy.sparse as sp, cvxpy as cp
from ocs.cvxpy_mf import make_affine_operator
x = cp.Variable(3)
matrix = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], np.float32))
mv, rmv, offset = make_affine_operator([(matrix @ x).canonical_form[0]], {x.id: 0}, 3)
np.testing.assert_array_equal(jax.jit(mv)(jnp.ones(3)), [3, 3])
np.testing.assert_array_equal(jax.jit(rmv)(jnp.ones(2)), [1, 3, 2])
assert not attempts, attempts
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
