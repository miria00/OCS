"""Independent checks that grouped proximal dispatch preserves each function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.prox.base import ProxBlock
from ocs.prox.groups import build_groups, grouped_moreau_dual
from ocs.prox.library import Quadratic, PiecewiseLinearConvex

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# prox grouping, measured through the residual.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def _apply(blocks, sizes, v, steps):
    ends = np.cumsum([0, *sizes])
    slices = list(zip(ends[:-1], ends[1:]))
    groups = build_groups(blocks, slices)
    result = jax.jit(grouped_moreau_dual(groups, int(ends[-1])))(
        jnp.asarray(v, jnp.float32), jnp.asarray(steps, jnp.float32))
    return np.asarray(result), groups


def _soft(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)


def test_distinct_quadratics_share_a_group_without_losing_coefficients():
    matrices = [np.array([[2., .4], [.4, 1.]]),
                np.array([[.7, -.2], [-.2, 3.]]), np.array([[4., .2], [.2, 4.]])]
    linear = [np.array([1., -2.]), np.array([-.3, .8]), np.array([2., .5])]
    blocks = [Quadratic(P, q) for P, q in zip(matrices, linear)]
    v = np.array([[3., -1.], [1.2, 4.], [-2., 1.]])
    steps = np.array([[.2, .7], [1.3, .4], [.9, 2.]])
    expected = np.array([row - step * np.linalg.solve(P + np.diag(step), row-q)
                         for P, q, row, step in zip(matrices, linear, v, steps)])
    actual, groups = _apply(blocks, [2]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected.ravel(), rtol=3e-6, atol=1e-6)
    assert len(groups) == 1 and groups[0].count == 3


def test_pwl_equal_sums_do_not_identify_equal_functions():
    # Both parameter sums collide in the old group key, although these
    # functions are |u| and 2|u-1|, respectively.
    blocks = [PiecewiseLinearConvex([-1., 1.], [0., 0.]),
              PiecewiseLinearConvex([-2., 2.], [2., -2.])]
    v = np.array([[1.5, -2., .1], [1.5, -2., .1]])
    steps = np.array([[.2, .7, 1.1], [.5, .3, .8]])
    expected = []
    for row, step, lam, center in zip(v, steps, [1., 2.], [0., 1.]):
        primal = center + _soft(row / step - center, lam / step)
        expected.extend(row - step * primal)
    actual, groups = _apply(blocks, [3, 3], v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert len(groups) == 1


def test_nested_calculus_parameters_are_batched_and_preserved():
    params = [(1., 2., -1., .4, -.2, .3),
              (.7, .5, 2., -.6, .8, 1.2),
              (2., 3., .5, .3, -.1, .8)]
    blocks = [ocs.L1(lam).scaled(c).precomposed(a, b).with_linear_quadratic(d, e)
              for lam, c, a, b, d, e in params]
    v = np.array([[2., -1., .3], [-.2, 4., -3.], [1., 2., -2.]])
    steps = np.array([[.3, .8, .5], [1.2, .2, .7], [.4, .9, 1.1]])
    expected = []
    for row, step, (lam, c, a, b, d, e) in zip(v, steps, params):
        # Prox of c*lam*|a*u-b| + d*u + e*u^2/2, in closed form.
        center = b/a
        shifted = (row-d)/(step+e)
        primal = center + _soft(shifted-center, c*lam*abs(a)/(step+e))
        expected.extend(row-step*primal)
    actual, groups = _apply(blocks, [3]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    assert len(groups) == 1 and groups[0].count == 3


def test_wrappers_with_different_base_functions_do_not_share_representatives():
    blocks = [ocs.L1(2.).scaled(.3), ocs.SumSquares().scaled(4.),
              ocs.L1(.2).scaled(3.)]
    v = np.array([[2., -.4], [2., -.4], [2., -.4]])
    steps = np.array([[.3, 2.], [.8, .4], [2., .3]])
    expected = np.concatenate([np.clip(v[0], -.6, .6),
                               4*v[1]/(steps[1]+4), np.clip(v[2], -.6, .6)])
    actual, groups = _apply(blocks, [2]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert sorted(g.count for g in groups) == [1, 2]


def test_vector_box_bounds_and_scalar_bounds_support_distinct_parameters():
    bounds = [([-2., 0., 1.], [-1., 2., 3.]),
              ([0., -3., -1.], [1., -.2, 4.]), (-2., 3.), (.5, 2.)]
    blocks = [ocs.Box(lo, hi) for lo, hi in bounds]
    v = np.array([[2., -1., 4.], [-3., 3., 1.], [2., -1., 4.], [-3., 3., 1.]])
    steps = np.array([[.4, .2, 1.], [.5, 2., .3], [.7, .9, .3], [2., .8, .4]])
    expected = np.concatenate([row-step*np.clip(row/step, lo, hi)
                               for row, step, (lo, hi) in zip(v, steps, bounds)])
    actual, groups = _apply(blocks, [3]*4, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert len(groups) <= 2


def test_weights_and_levels_group_by_shapes_not_parameter_values():
    blocks = [ocs.L1(float(k)/16) for k in range(1, 65)]
    blocks += [ocs.Pinball(float(k)/65) for k in range(1, 65)]
    v = np.tile([-.8, .3, 1.4], (128, 1))
    steps = np.linspace(.1, 2., v.size).reshape(v.shape)
    expected = [np.clip(row, -k/16, k/16) for k, row in enumerate(v[:64], 1)]
    expected += [np.clip(row, k/65-1, k/65) for k, row in enumerate(v[64:], 1)]
    actual, groups = _apply(blocks, [3]*128, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, np.asarray(expected).ravel(), atol=2e-6)
    assert len(groups) == 2
    assert all(g.count == 64 for g in groups)


def test_custom_blocks_with_colliding_keys_keep_their_own_functions():
    class ShiftedSquare(ProxBlock):
        def __init__(self, center):
            self.center = center

        def group_key(self):
            return "deliberately-colliding"

        def prox(self, v, t):
            return (v+t*self.center)/(1+t)

    first, second = ShiftedSquare(1.), ShiftedSquare(-2.)
    blocks = [first, second, first]
    v = np.array([[2., -1.], [2., -1.], [1., 3.]])
    steps = np.array([[.3, .8], [1.2, .4], [.2, 2.]])
    expected = (v-steps*np.array([1., -2., 1.])[:, None])/(1+steps)
    actual, groups = _apply(blocks, [2]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected.ravel(), atol=2e-6)
    assert sorted(g.count for g in groups) == [1, 2]


def test_builtin_subclasses_do_not_inherit_a_parameter_packing_contract():
    class TiltedL1(ocs.L1):
        def __init__(self, tilt):
            super().__init__(1.)
            self.tilt = tilt

        def prox(self, v, t):
            return super().prox(v-t*self.tilt, t)

    blocks = [TiltedL1(.4), TiltedL1(-.7)]
    v = np.array([[2., -.8], [2., -.8]])
    steps = np.array([[.3, 2.], [.8, .4]])
    expected = np.array([np.clip(row-tilt, -1., 1.)+tilt
                         for row, tilt in zip(v, [.4, -.7])])
    actual, groups = _apply(blocks, [2, 2], v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected.ravel(), atol=2e-6)
    assert len(groups) == 2


def test_same_kind_different_sizes_keep_gather_scatter_order():
    blocks = [ocs.L1(.2), ocs.L1(1.3), ocs.L1(.7), ocs.L1(2.)]
    sizes = [2, 3, 2, 3]
    v = np.linspace(-2., 3., sum(sizes))
    steps = np.linspace(.2, 1.5, sum(sizes))
    expected = np.concatenate([np.clip(row, -lam, lam) for row, lam in
                               zip(np.split(v, np.cumsum(sizes)[:-1]), [.2, 1.3, .7, 2.])])
    actual, groups = _apply(blocks, sizes, v, steps)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert sorted((g.size, g.count) for g in groups) == [(2, 2), (3, 2)]


def test_grouped_program_solves_the_requested_distinct_functions():
    # Both non-identity terms must stay dual blocks. Their weights affect
    # the exact solution: min .5*(x-1)^2 + 4.5*(x-3)^2 has optimum 2.8.
    op = ocs.LinearOperator.from_forward(lambda x: x, n=1, m=1)
    from ocs.ir import ProblemFamily, Term
    terms = [Term(op, ocs.SumSquares().scaled(1.), lambda data: jnp.array([-1.])),
             Term(op, ocs.SumSquares().scaled(9.), lambda data: jnp.array([-3.]))]
    family = ProblemFamily(n=1, terms=terms)
    compiled = family.compile(tol=1e-5, max_blocks=200, block=20, equilibrate="scalar")
    solution = compiled.solve_instance({})
    np.testing.assert_allclose(solution.x, [2.8], atol=5e-5)
    assert solution.status == "solved"
    certificate = compiled.certificate(solution.x, solution.y, {})
    assert max(certificate["r_primal"], certificate["r_dual"]) <= 1e-5


def test_affine_quadratics_keep_zero_pattern_branch_and_matrix_parameters():
    matrices = [np.array([[2., .6], [.6, 3.]]),
                np.array([[1., -.2], [-.2, 2.]]),
                np.array([[3., .7], [.7, 4.]])]
    coefficients = [np.array([0., 2.]), np.array([1., 0.]), np.array([1., -2.])]
    offsets = [np.array([1., -.4]), np.array([-.2, 2.]), np.array([.3, 1.])]
    linear = [np.array([.2, -1.]), np.array([1., 2.]), np.array([-1., .4])]
    blocks = [Quadratic(P, q).precomposed(a, b)
              for P, q, a, b in zip(matrices, linear, coefficients, offsets)]
    v = np.array([[2., -1.], [1., 3.], [-2., .4]])
    steps = np.array([[.3, .9], [2., .2], [.4, .7]])
    expected = []
    for P, q, a, b, row, step in zip(matrices, linear, coefficients, offsets, v, steps):
        H = a[:, None]*P*a[None, :]
        linear_term = a*(q-P@b)
        primal = np.linalg.solve(H+np.diag(step), row-linear_term)
        expected.extend(row-step*primal)
    actual, groups = _apply(blocks, [2]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    # The singular and invertible affine paths have distinct static code,
    # while the two singular maps with different zero locations share a group.
    assert sorted(g.count for g in groups) == [1, 2]


def test_power_exponents_are_batched_using_independent_normal_ray_projections():
    from ocs.prox.cones import PowerCone, PowerConeBatch
    alphas = [.2, .5, .8]
    blocks, sizes, values, steps, expected = [], [], [], [], []
    for batched in [False, True]:
        for alpha, step in zip(alphas, [.2, .7, 1.3]):
            points, normals = [], []
            for x, y in [(1.2, .7), (.4, 1.8)][:2 if batched else 1]:
                z = x**alpha*y**(1-alpha)
                point = np.array([x, y, z])
                # Gradient of |z|-x^alpha*y^(1-alpha) is an outward normal.
                normal = np.array([-alpha*z/x, -(1-alpha)*z/y, 1.])
                points.append(point)
                normals.append(normal)
            point = np.array(points).T.ravel() if batched else points[0]
            normal = np.array(normals).T.ravel() if batched else normals[0]
            blocks.append(PowerConeBatch(2, alpha) if batched else PowerCone(alpha))
            sizes.append(len(point))
            values.extend(step*(point+.4*normal))
            steps.extend([step]*len(point))
            expected.extend(step*.4*normal)
    actual, groups = _apply(blocks, sizes, values, steps)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
    assert len(groups) == 2 and all(g.count == 3 for g in groups)


def test_piece_counts_are_structural_not_a_parameter_collision():
    # A redundant middle line leaves the second function equal to 2|u-1|.
    blocks = [PiecewiseLinearConvex([-1., 1.], [0., 0.]),
              PiecewiseLinearConvex([-2., 0., 2.], [2., 0., -2.]),
              PiecewiseLinearConvex([-3., 0., 3.], [-3., 0., 3.])]
    v = np.array([[2., -.4], [-2., .8], [.4, -2.]])
    steps = np.array([[.4, 1.], [.2, 2.], [1.1, .3]])
    expected = []
    for row, step, lam, center in zip(v, steps, [1., 2., 3.], [0., 1., -1.]):
        primal = center+_soft(row/step-center, lam/step)
        expected.extend(row-step*primal)
    actual, groups = _apply(blocks, [2]*3, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, expected, atol=3e-6)
    assert sorted(g.count for g in groups) == [1, 2]


def test_repeated_identical_objects_share_parameters_without_replication():
    block = Quadratic([[2., .2], [.2, 3.]], [.5, -.8])
    blocks = [block]*20
    v = np.linspace(-1., 3., 40).reshape(20, 2)
    steps = np.linspace(.1, 2., 40).reshape(20, 2)
    expected = [row-step*np.linalg.solve(np.array([[2., .2], [.2, 3.]])+np.diag(step),
                                         row-np.array([.5, -.8]))
                for row, step in zip(v, steps)]
    actual, groups = _apply(blocks, [2]*20, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, np.asarray(expected).ravel(), atol=2e-6)
    assert len(groups) == 1 and groups[0].count == 20
    assert not jax.tree_util.tree_leaves(groups[0].parameters)


def test_group_builder_rejects_missing_slices_instead_of_dropping_functions():
    with pytest.raises(ValueError, match="one slice"):
        build_groups([ocs.L1(), ocs.L1(2)], [(0, 2)])


def test_distinct_wrappers_share_large_base_parameters_without_replication():
    matrix = np.array([[2., .3], [.3, 1.]])
    linear = np.array([.2, -1.])
    base = Quadratic(matrix, linear)
    weights = np.arange(1, 13)/4
    blocks = [base.scaled(float(weight)) for weight in weights]
    v = np.linspace(-1., 3., 24).reshape(12, 2)
    steps = np.linspace(.2, 2., 24).reshape(12, 2)
    expected = [row-step*np.linalg.solve(weight*matrix+np.diag(step), row-weight*linear)
                for row, step, weight in zip(v, steps, weights)]
    actual, groups = _apply(blocks, [2]*12, v.ravel(), steps.ravel())
    np.testing.assert_allclose(actual, np.asarray(expected).ravel(), atol=2e-6)
    assert len(groups) == 1
    leaves = jax.tree_util.tree_leaves(groups[0].parameters)
    assert any(leaf is base.P for leaf in leaves)
    assert sum(leaf.size for leaf in leaves) == len(weights)+matrix.size+linear.size


def test_large_contiguous_groups_store_only_span_bounds(monkeypatch):
    import ocs.prox.groups as module
    def no_coordinate_indices(*args, **kwargs):
        raise AssertionError("contiguous groups must not allocate coordinate indices")
    monkeypatch.setattr(module.np, "arange", no_coordinate_indices)
    # Construct metadata only: this must not allocate multi-gigabyte arrays.
    n = 10**9
    groups = build_groups([ocs.SumSquares(), ocs.GroupL2(.02, 3)],
                          [(0, n), (n, 4*n)])
    assert [group.span for group in groups] == [(0, n), (n, 4*n)]
    assert all(group.idx is None for group in groups)


def test_imaging_partition_has_no_indexed_dispatch_in_jaxpr():
    blocks = [ocs.SumSquares(), ocs.GroupL2(.2, 3)]
    groups = build_groups(blocks, [(0, 8), (8, 20)])
    apply = grouped_moreau_dual(groups, 20)
    v = np.linspace(-1.5, 2., 20, dtype=np.float32)
    steps = np.r_[np.linspace(.2, .9, 8), np.full(12, .4)].astype(np.float32)
    rows = v[8:].reshape(-1, 3)
    projected = rows*np.minimum(1., .2/np.linalg.norm(rows, axis=1, keepdims=True))
    expected = np.r_[v[:8]/(1+steps[:8]), projected.ravel()]
    np.testing.assert_allclose(jax.jit(apply)(jnp.asarray(v), jnp.asarray(steps)),
                               expected, rtol=3e-6, atol=2e-7)
    representation = str(jax.make_jaxpr(apply)(jnp.asarray(v), jnp.asarray(steps)))
    assert "gather[" not in representation
    assert "scatter" not in representation
    assert "concatenate[" in representation


def test_contiguous_members_keep_distinct_parameters_without_indices():
    blocks = [ocs.L1(.2), ocs.L1(.8), ocs.L1(1.3)]
    groups = build_groups(blocks, [(0, 2), (2, 4), (4, 6)])
    assert len(groups) == 1 and groups[0].idx is None
    assert groups[0].span == (0, 6)
    v = np.array([-1., .4, 2., -.3, -.8, 3.], dtype=np.float32)
    expected = np.clip(v.reshape(3, 2), -np.array([.2, .8, 1.3])[:, None],
                       np.array([.2, .8, 1.3])[:, None]).ravel()
    result = jax.jit(grouped_moreau_dual(groups, 6))(jnp.asarray(v), jnp.ones(6))
    np.testing.assert_allclose(result, expected, atol=2e-7)


def test_slice_updates_preserve_gaps_overlap_and_group_order():
    blocks = [ocs.SumSquares(), ocs.L1(.4), ocs.Pinball(.7)]
    spans = [(3, 5), (0, 2), (1, 3)]
    groups = build_groups(blocks, spans)
    assert all(group.idx is None for group in groups)
    v = np.array([-.8, .6, 1.4, -.2, 2., 3.], dtype=np.float32)
    steps = np.linspace(.2, 1., 6, dtype=np.float32)
    expected = np.zeros(6, dtype=np.float32)
    expected[3:5] = v[3:5]/(1+steps[3:5])
    expected[0:2] = np.clip(v[0:2], -.4, .4)
    expected[1:3] = np.clip(v[1:3], -.3, .7)
    result = jax.jit(grouped_moreau_dual(groups, 6))(jnp.asarray(v), jnp.asarray(steps))
    np.testing.assert_allclose(result, expected, atol=2e-7)


def test_out_of_order_members_keep_indexed_parameter_order():
    groups = build_groups([ocs.L1(.2), ocs.L1(.8)], [(4, 6), (0, 2)])
    assert len(groups) == 1 and groups[0].span is None
    np.testing.assert_array_equal(groups[0].idx, [4, 5, 0, 1])
    v = np.array([-2., 1.4, 3., -4., -.7, .5], dtype=np.float32)
    expected = np.r_[np.clip(v[:2], -.8, .8), [0., 0.], np.clip(v[4:], -.2, .2)]
    result = jax.jit(grouped_moreau_dual(groups, 6))(jnp.asarray(v), jnp.ones(6))
    np.testing.assert_allclose(result, expected, atol=2e-7)


def test_overlapping_members_keep_indexed_fallback():
    block = ocs.L1(.6)
    groups = build_groups([block, block], [(0, 3), (2, 5)])
    assert len(groups) == 1 and groups[0].span is None
    np.testing.assert_array_equal(groups[0].idx, [0, 1, 2, 2, 3, 4])
    v = jnp.array([-2., 1.4, 3., -4., -.7])
    result = jax.jit(grouped_moreau_dual(groups, 5))(v, jnp.ones(5))
    np.testing.assert_allclose(result, np.clip(v, -.6, .6), atol=2e-7)


def test_out_of_range_span_preserves_original_gather_clip_and_scatter_drop():
    groups = build_groups([ocs.L2(.6)], [(2, 6)])
    v = np.array([-2., 1., .3, .7], dtype=np.float32)
    # Original indexed gather repeats the last entry for indices 4 and 5.
    # They affect this coupled projection, even though their outputs are dropped.
    gathered = v[[2, 3, 3, 3]]
    projected = gathered*min(1., .6/np.linalg.norm(gathered))
    expected = np.r_[0., 0., projected[:2]]
    result = jax.jit(grouped_moreau_dual(groups, 4))(jnp.asarray(v), jnp.ones(4))
    np.testing.assert_allclose(result, expected, atol=2e-7)


def test_empty_dispatch_still_returns_zero_vector():
    result = jax.jit(grouped_moreau_dual([], 3))(jnp.ones(3), jnp.ones(3))
    np.testing.assert_array_equal(result, np.zeros(3))
