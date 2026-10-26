"""Independent analytic checks for the public proximal calculus contract."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox.base import ProxBlock, SCALAR, DIAGONAL, MATRIX, _has_oracle
from ocs.prox.library import SumSquares, Quadratic, Zero, L1, L2, Box, NonNeg, Equality, Logistic, Pinball

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the prox calculus wrappers, through the residual.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


@pytest.mark.parametrize("coefficient", [0, -1, np.inf, np.nan, 1e300, 1e-60, [1, 2]])
def test_scaling_requires_representable_positive_scalar(coefficient):
    with pytest.raises(ValueError):
        L1().scaled(coefficient)


@pytest.mark.parametrize("construct", [
    lambda: L2().precomposed(a=[1., 2.]),
    lambda: L2().precomposed(a=[0., 1.]),
    lambda: L2().precomposed(a=0.),
    lambda: L2().with_linear_quadratic(e=[0., 9.]),
    lambda: L1().with_linear_quadratic(e=-1.),
    lambda: L1().with_linear_quadratic(d=np.nan),
    lambda: L1().precomposed(a=np.inf),
    lambda: L1().precomposed(b=np.nan),
    lambda: L1().precomposed(a=[[1., 2.]]),
    lambda: L1().precomposed(a=[1., 2.], b=[1., 2., 3.]),
    lambda: Quadratic(np.eye(2)).precomposed(a=np.ones(3)),
    lambda: Quadratic(np.eye(2)).with_linear_quadratic(d=np.ones(3)),
])
def test_invalid_calculus_rejected_at_construction(construct):
    with pytest.raises(ValueError):
        construct()


def test_coupled_affine_signed_uniform_magnitudes_stationarity():
    a, b = np.array([2., -2.]), np.array([.3, -.1])
    v, t = np.array([2., 1.]), .7
    block = L2(.8).precomposed(a=a, b=b)
    point = np.asarray(jax.jit(block.prox)(jnp.asarray(v), t))
    z = a * point - b
    residual = point - v + t * .8 * a * z / np.linalg.norm(z)
    np.testing.assert_allclose(residual, 0, atol=4e-7)
    assert block.scaling_class == SCALAR


def test_coupled_uniform_quadratic_stationarity_and_conjugate():
    d = np.array([.2, -.4])
    block = L2(.8).with_linear_quadratic(d=d, e=2.)
    v, t = np.array([2., 1.]), .7
    point = np.asarray(jax.jit(block.prox)(jnp.asarray(v), t))
    gradient = .8 * point / np.linalg.norm(point) + d + 2 * point
    np.testing.assert_allclose(point - v + t * gradient, 0, atol=4e-7)
    # Independent Fenchel optimizer for lambda||x|| + e||x||²/2 + dᵀx.
    y = np.array([3., -.7])
    expected = max(np.linalg.norm(y - d) - .8, 0.) ** 2 / 4.
    np.testing.assert_allclose(block.conj_value(jnp.asarray(y)), expected, rtol=2e-6)


def test_zero_affine_coordinates_remain_free_and_keep_constant_value():
    a, b = np.array([0., 2.]), np.array([3., 4.])
    v, t = np.array([5., -2.]), np.array([.4, .7])
    block = SumSquares().precomposed(a=a, b=b)
    expected = (v + t * a * b) / (1 + t * a * a)
    point = jax.jit(block.prox)(jnp.asarray(v), jnp.asarray(t))
    np.testing.assert_allclose(point, expected, rtol=2e-6)
    np.testing.assert_allclose(block.value(point), .5 * np.sum((a * expected - b) ** 2), rtol=2e-6)
    y = jnp.array([0., .4])
    np.testing.assert_allclose(block.conj_value(y), .4**2 / 8 + 2 * .4 - 4.5, rtol=2e-6)
    assert np.isinf(block.conj_value(jnp.array([.1, .4])))


def test_tiny_affine_quadratic_coefficient_keeps_resolvable_update():
    block = SumSquares().precomposed(a=[1e-4, 1e-12], b=[1., 1.])
    v = np.array([1., 2.])
    expected = (v + np.array([1e-4, 1e-12])) / (1 + np.array([1e-8, 1e-24]))
    np.testing.assert_allclose(block.prox(jnp.asarray(v), 1.), expected, atol=7e-8, rtol=0)


def test_separable_singular_conjugate_uses_no_dense_matrix(monkeypatch):
    import ocs.prox.base as calculus
    block = SumSquares().precomposed(a=jnp.array([0., 2.]), b=.3)
    def forbidden_dense(*args):
        raise AssertionError("separable conjugate must not construct a dense Hessian")
    monkeypatch.setattr(calculus, "_quadratic_coefficients", forbidden_dense)
    assert np.isfinite(block.conj_value(jnp.array([0., .2])))


@pytest.mark.parametrize("base,b", [
    (NonNeg(), 1e-8), (NonNeg().scaled(2.), 1e-8),
    (Equality(), 1e-8), (Box(0., 1.), 1e-8), (Box(0., 1.), -1.00001),
])
def test_zero_affine_coefficients_reject_fixed_infeasible_domain(base, b):
    with pytest.raises(ValueError, match="domain"):
        base.precomposed(a=0., b=b)


def test_separable_fixed_coordinate_and_unavailable_conjugate():
    block = Box(lo=jnp.array([-2., 1.]), hi=jnp.array([3., 4.])).precomposed(
        a=[0., 2.], b=[1., .2])
    actual = jax.jit(block.prox)(jnp.array([7., -3.]), .5)
    np.testing.assert_allclose(actual, [7., .6], atol=1e-7)
    assert _has_oracle(block, "value")
    assert not _has_oracle(block, "conj_value")
    with pytest.raises(NotImplementedError):
        block.conj_value(jnp.zeros(2))


def test_scalar_zero_affine_on_dimensioned_quadratic():
    block = Quadratic(jnp.diag(jnp.array([1., 2.]))).precomposed(a=0., b=1.)
    v = jnp.array([2., -3.])
    np.testing.assert_allclose(block.prox(v, .3), v)
    np.testing.assert_allclose(block.value(v), 1.5)
    np.testing.assert_allclose(block.conj_value(jnp.zeros(2)), -1.5)


@pytest.mark.parametrize("zero_coefficient", [False, True])
def test_nested_matrix_calculus_matches_explicit_quadratic(zero_coefficient):
    P = np.array([[2., .4], [.4, 1.3]])
    q = np.array([.3, -.5])
    a = np.array([0. if zero_coefficient else -1.5, 2.])
    b, d, e, c = np.array([.2, -.4]), np.array([.1, .2]), np.array([.4, 1.1]), 1.7
    base = Quadratic(P, q)
    block = base.precomposed(a=a, b=b).scaled(c).with_linear_quadratic(d=d, e=e)
    H = c * (a[:, None] * P * a[None, :]) + np.diag(e)
    linear = c * a * (q - P @ b) + d
    constant = c * (.5 * b @ P @ b - q @ b)
    v, D = np.array([2., -1.]), np.array([[2.5, .2], [.2, 1.2]])
    expected = np.linalg.solve(H + D, D @ v - linear)
    actual = jax.jit(block.prox_matrix)(jnp.asarray(v), jnp.asarray(D))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(block.value(actual), .5 * expected @ H @ expected + linear @ expected + constant, rtol=3e-6)
    np.testing.assert_allclose(block.smooth_grad(actual), H @ expected + linear, rtol=3e-6, atol=3e-7)
    y = np.array([.4, -.7])
    conjugate = .5 * (y - linear) @ np.linalg.solve(H, y - linear) - constant
    np.testing.assert_allclose(block.conj_value(jnp.asarray(y)), conjugate, rtol=3e-6, atol=3e-7)
    t = np.array([.3, .8])
    expected_diagonal = np.linalg.solve(H + np.diag(1 / t), v / t - linear)
    np.testing.assert_allclose(jax.jit(block.prox)(jnp.asarray(v), jnp.asarray(t)), expected_diagonal, rtol=3e-6, atol=3e-7)
    assert block.scaling_class == MATRIX
    assert block.eval_kind == "linear_solve"


def test_separable_matrix_wrappers_retain_full_matrix_capability():
    block = SumSquares().scaled(2.).precomposed(a=[0., -2.], b=[.5, 1.])
    v = np.array([1., -2.])
    D = np.array([[2., .4], [.4, 1.]])
    H, linear = np.diag([0., 8.]), np.array([0., 4.])
    expected = np.linalg.solve(H + D, D @ v - linear)
    np.testing.assert_allclose(block.prox_matrix(jnp.asarray(v), jnp.asarray(D)), expected, rtol=3e-6)


def test_invertible_affine_conjugate_sign_and_scale():
    # max_z yᵀz - c/2 ||a*z-b||² has a coordinatewise closed form.
    a, b, c = np.array([-2., .5]), np.array([.3, -.7]), 1.4
    y = np.array([.2, -.6])
    block = SumSquares().precomposed(a=a, b=b).scaled(c)
    expected = np.sum(y * b / a + y * y / (2 * c * a * a))
    np.testing.assert_allclose(block.conj_value(jnp.asarray(y)), expected, rtol=2e-6)


def test_linear_only_conjugate_and_value_preserved():
    block = L1(.7).scaled(2.).precomposed(a=-2., b=.3).with_linear_quadratic(d=.1)
    y = jnp.array([.2, -.4])
    expected = jnp.sum((y - .1) / -2. * .3)
    np.testing.assert_allclose(block.conj_value(y), expected, rtol=2e-6)
    assert np.isinf(block.conj_value(jnp.array([4., 0.])))


def test_partial_zero_curvature_declares_missing_general_conjugate():
    block = L1().with_linear_quadratic(e=[0., 2.])
    assert block.value_available and not block.conj_value_available
    with pytest.raises(NotImplementedError):
        block.conj_value(jnp.ones(2))
    nested = block.scaled(2.).precomposed(a=2.)
    assert not _has_oracle(nested, "conj_value")


def test_partial_zero_quadratic_curvature_conjugate_available():
    block = SumSquares().with_linear_quadratic(d=[.1, -.2], e=[0., 2.])
    y = np.array([.7, -.4])
    expected = .5 * np.sum((y - [.1, -.2])**2 / np.array([1., 3.]))
    np.testing.assert_allclose(block.conj_value(jnp.asarray(y)), expected, rtol=2e-6)
    assert _has_oracle(block, "conj_value")


def test_eval_kind_and_smooth_grad_survive_nested_calculus():
    block = Logistic().scaled(2.).precomposed(a=[-2., .5], b=[.3, -.1]).with_linear_quadratic(d=.2, e=.4)
    point = jnp.array([.7, -.2])
    np.testing.assert_allclose(block.smooth_grad(point), jax.grad(block.value)(point), rtol=2e-6)
    assert block.eval_kind == "newton"
    assert block.scaling_class == DIAGONAL


def test_unknown_matrix_declaration_does_not_invent_missing_oracle():
    class DeclaredMatrix(ProxBlock):
        scaling_class = MATRIX
        def prox(self, v, t):
            return v
    block = DeclaredMatrix().scaled(2.)
    assert block.scaling_class == DIAGONAL
    assert not _has_oracle(block, "value")
    assert not _has_oracle(block, "conj_value")
    with pytest.raises(NotImplementedError):
        block.prox_matrix(jnp.ones(2), jnp.eye(2))
    with pytest.raises(ValueError):
        DeclaredMatrix().precomposed(a=[0., 1.])


def test_unknown_separable_singular_affine_downgrades_matrix_tier():
    class CustomZero(ProxBlock):
        scaling_class = MATRIX
        coordinate_separable = True
        def prox(self, v, t):
            return v
        def prox_matrix(self, v, D):
            return v
        def value(self, z):
            return jnp.array(0.)
    block = CustomZero().precomposed(a=[0., 1.])
    assert block.scaling_class == DIAGONAL
    np.testing.assert_allclose(block.prox(jnp.array([1., 2.]), 1.), [1., 2.])
    with pytest.raises(NotImplementedError):
        block.prox_matrix(jnp.ones(2), jnp.eye(2))


def test_diagonal_metric_capability_does_not_imply_separability():
    class CoupledDiagonalOracle(ProxBlock):
        scaling_class = DIAGONAL
    with pytest.raises(ValueError, match="separable"):
        CoupledDiagonalOracle().precomposed(a=[0., 1.])


@pytest.mark.parametrize("step", [0., -1., np.inf, [1., 2.]])
def test_coupled_wrapper_rejects_invalid_eager_metric(step):
    block = L2().scaled(2.)
    with pytest.raises(ValueError):
        block.prox(jnp.ones(2), step)


def test_invalid_traced_step_does_not_silently_project_indicator():
    block = NonNeg().scaled(2.)
    result = jax.jit(block.prox)(jnp.array([-1., 2.]), -1.)
    assert np.all(np.isnan(result))


@pytest.mark.parametrize("metric", [np.diag([1., 0.]), np.diag([1., -1.]), np.array([[1., .3], [.2, 1.]])])
def test_wrapper_rejects_invalid_matrix_metric(metric):
    block = SumSquares().scaled(2.)
    with pytest.raises(ValueError):
        block.prox_matrix(jnp.ones(2), jnp.asarray(metric))


def test_wrapped_zero_function_has_full_matrix_prox():
    block = Zero().scaled(3.).precomposed(a=[0., 2.], b=[.2, .3]).with_linear_quadratic(d=[.1, -.2], e=[0., .5])
    D, v = np.array([[2., .2], [.2, 1.]]), np.array([.4, -.7])
    expected = np.linalg.solve(D + np.diag([0., .5]), D @ v - [.1, -.2])
    np.testing.assert_allclose(block.prox_matrix(jnp.asarray(v), jnp.asarray(D)), expected, rtol=2e-6)


def test_missing_wrapper_conjugate_does_not_break_public_compile():
    from ocs.ir import ProblemFamily, Term
    from ocs.operators.common import Identity, Diagonal
    block = L1().with_linear_quadratic(e=[0., 2.])
    family = ProblemFamily(2, [Term(Identity(2), SumSquares()),
                               Term(Diagonal(jnp.ones(2)), block,
                                    lambda data: jnp.array([-2., 3.]))])
    compiled = family.compile(equilibrate="scalar", max_blocks=20, tol=1e-3)
    assert not compiled.meta["gap_available"]
    solution = compiled.solve_instance({})
    assert solution.status == "solved"
    assert np.isfinite(solution.residual)


def _two_piece_scalar_reference(v, t, a, b, left, right):
    """Float64 convex reference from the stationary point of each segment."""
    v, t, a, b, left, right = map(float, (v, t, a, b, left, right))
    if a == 0:
        return v
    candidates = [b/a]
    negative_root = v - t*a*left
    positive_root = v - t*a*right
    if a*negative_root-b <= 0:
        candidates.append(negative_root)
    if a*positive_root-b >= 0:
        candidates.append(positive_root)
    def objective(point):
        z = a*point-b
        return (left*z if z <= 0 else right*z) + .5*(point-v)**2/t
    return min(candidates, key=objective)


@pytest.mark.parametrize("a", [1e-6, -1e-6, 1e-8, -1e-8])
@pytest.mark.parametrize("b", [-1., 1.])
@pytest.mark.parametrize("kind", ["l1", "scaled_l1", "pinball", "scaled_pinball"])
def test_tiny_affine_two_piece_loss_matches_scalar_convex_reference(a, b, kind):
    base = L1() if "l1" in kind else Pinball(.2)
    left, right = (-1., 1.) if "l1" in kind else (-.8, .2)
    if kind.startswith("scaled"):
        base = base.scaled(2.).scaled(1.5)
        left, right = 3*left, 3*right
    block = base.precomposed(a=a, b=b)
    # The transformed prox displacement is tiny, but its original-coordinate
    # effect is order one. Use the represented float32 inputs in the oracle.
    v = np.array([1., -2., .3], dtype=np.float32)
    step = np.float32(1/abs(a))
    aa, bb = np.float32(a), np.float32(b)
    expected = [_two_piece_scalar_reference(point, step, aa, bb, left, right) for point in v]
    actual = jax.jit(block.prox)(jnp.asarray(v), step)
    np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=2e-7)


@pytest.mark.parametrize("a", [1e-8, -1e-8])
@pytest.mark.parametrize("tau", [0., .2, 1.])
def test_affine_pinball_clips_to_kink_and_handles_flat_side(a, tau):
    a, step = np.float32(a), np.float32(1/abs(a))
    b = np.float32(2*a)
    points = np.array([1.7, 2., 2.1], dtype=np.float32)
    block = Pinball(tau).precomposed(a=a, b=b)
    expected = [_two_piece_scalar_reference(point, step, a, b, tau-1., tau) for point in points]
    np.testing.assert_allclose(jax.jit(block.prox)(jnp.asarray(points), step), expected, atol=3e-7, rtol=0)


def test_grouped_affine_small_coefficients_use_each_nested_parameter():
    from ocs.prox.groups import build_groups, grouped_moreau_dual
    coefficients = [1e-8, -1e-8, 1e-6]
    offsets = [1., 1., -1.]
    weights = [(1., 2.), (2., .5), (.3, 3.)]
    blocks = [L1(lam).scaled(scale).precomposed(a=a, b=b)
              for a, b, (lam, scale) in zip(coefficients, offsets, weights)]
    groups = build_groups(blocks, [(0, 1), (1, 2), (2, 3)])
    assert len(groups) == 1
    # Moreau dispatch evaluates v/sigma with proximal step 1/sigma.
    sigma = np.asarray([abs(a) for a in coefficients], dtype=np.float32)
    v = sigma.copy()
    actual = jax.jit(grouped_moreau_dual(groups, 3))(jnp.asarray(v), jnp.asarray(sigma))
    primal = [_two_piece_scalar_reference(1., 1/float(s), np.float32(a), np.float32(b),
                                           -float(np.float32(lam))*float(np.float32(scale)),
                                           float(np.float32(lam))*float(np.float32(scale)))
              for a, b, s, (lam, scale) in zip(coefficients, offsets, sigma, weights)]
    expected = v.astype(np.float64)-sigma.astype(np.float64)*np.asarray(primal)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-14)
