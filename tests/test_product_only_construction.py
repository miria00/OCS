"""The product-only spatial-full constructor (product_only_metric_20260930).

Gate for `SPEC.md`, sha256
29401a6ab8c30a8801a3d9f151d8d380a5399157757d2ae05ea934e822dbc489, frozen before
any of it was written.

THE CLAIM UNDER TEST. The archived construction reads its operator through
exactly one object, the cone's scaled Gram `G = A T A^T`, consumed as `G[0,0]`
and `eigh(G[1:,1:])`. The archive forms it from `K.toarray()`, which its own code
marks REFERENCE ONLY. This module obtains the same `G` from `k` adjoint and `k`
forward applications and runs the archive's selection rule unchanged.

HOW "PRODUCT ONLY" IS ESTABLISHED HERE. Not by reading the source and asserting
it. The constructor is run against an operator object that exposes ONLY `matvec`
and `rmatvec`, counts both, and RAISES on every attribute an entry reader would
reach. A constructor that quietly retrieves a dense norm from metadata at its
last step fails this and passes a source inspection.

FLOAT32 IS THE PRODUCTION TARGET. What is compared against the archive is the
metric's ACTION and the construction's energy. Eigenvector signs, eigenvalue
order inside a repeated eigenspace and entrywise float64 agreement are NOT
compared, because none of them is what the metric does.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox import spatial_construct as sc
from ocs.prox import spatial_soc as ss

F32 = np.float32


class EntryAccessBlocked(AssertionError):
    """Raised when anything reaches for the operator's entries."""


class ProductOnlyOperator:
    """An operator that is ONLY a pair of callables, and says so loudly.

    Every attribute an entry reader would reach raises. That includes the ones a
    numpy or scipy object would answer: `A`, `toarray`, `todense`, `data`,
    `indices`, `indptr`, `shape` beyond the declared one, `__array__`,
    `__getitem__`. If the constructor works through this object, it worked
    through products.
    """

    _ALLOWED = {"matvec", "rmatvec", "k", "n", "forward", "adjoint", "_A",
                "_count_forward", "_count_adjoint"}

    def __init__(self, A):
        object.__setattr__(self, "_A", np.asarray(A, np.float64))
        object.__setattr__(self, "k", int(A.shape[0]))
        object.__setattr__(self, "n", int(A.shape[1]))
        object.__setattr__(self, "_count_forward", 0)
        object.__setattr__(self, "_count_adjoint", 0)

    def matvec(self, x):
        object.__setattr__(self, "_count_forward", self._count_forward + 1)
        return jnp.matmul(jnp.asarray(self._A, jnp.float32), jnp.asarray(x, jnp.float32), precision=jax.lax.Precision.HIGHEST)

    def rmatvec(self, y):
        object.__setattr__(self, "_count_adjoint", self._count_adjoint + 1)
        return jnp.matmul(jnp.asarray(self._A, jnp.float32).T, jnp.asarray(y, jnp.float32), precision=jax.lax.Precision.HIGHEST)

    def __getattr__(self, name):
        raise EntryAccessBlocked(
            f"the constructor reached for {name!r} on the operator. Only matvec "
            "and rmatvec are available, which is the whole point of this test.")

    def __array__(self, *a, **k):
        raise EntryAccessBlocked("the constructor tried to array-convert the operator")

    def __getitem__(self, item):
        raise EntryAccessBlocked(f"the constructor tried to index the operator at {item!r}")


def instance(rng, n, k):
    """One cone's operator block and a fixed primal scaling."""
    A = rng.standard_normal((k, n)).astype(F32)
    tau = np.exp(rng.uniform(-0.5, 0.5, n)).astype(F32)
    return A, tau


def dense_moment(A, tau):
    """`A T A^T` formed with entries, the REFERENCE side only."""
    A64 = np.asarray(A, np.float64)
    return (A64 * np.asarray(tau, np.float64)[None, :]) @ A64.T


# ---------------------------------------------------------------------------
# the moment itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [3, 8, 33])
def test_the_block_moment_is_the_gram_and_costs_exactly_2k_applications(k):
    rng = np.random.default_rng(100 + k)
    n = 4 * k
    A, tau = instance(rng, n, k)
    op = ProductOnlyOperator(A)
    count = sc.ProductCount()
    G = sc.block_moment(op.matvec, op.rmatvec, k, n, jnp.asarray(tau), count=count)
    reference = dense_moment(A, tau)
    rel = float(np.linalg.norm(np.asarray(G, np.float64) - reference)
                / np.linalg.norm(reference))
    print(f"k={k}: block moment against the dense Gram, {rel:.3e} relative, "
          f"{count.record()}")
    assert rel <= 2e-6, "the product-only moment is not the Gram"
    assert count.adjoint == k and count.forward == k, count.record()
    # The operator's OWN counter counts Python calls, and the vectorized path
    # traces the function once for all k columns. The mathematical count is k
    # either way, which is what ProductCount records, and the two numbers mean
    # different things: k applications, and how many times the callable was
    # entered. Both are reported rather than one being passed off as the other.
    if count.record()["vectorized"]:
        assert op._count_adjoint == 1 and op._count_forward == 1
    else:
        assert op._count_adjoint == k and op._count_forward == k
    assert np.allclose(np.asarray(G), np.asarray(G).T, rtol=0, atol=0), \
        "the moment must be symmetrized exactly, not nearly"


def test_the_moment_never_touches_an_entry():
    """The wrapper raises on everything an entry reader would reach."""
    rng = np.random.default_rng(7)
    A, tau = instance(rng, 40, 10)
    op = ProductOnlyOperator(A)
    for name in ("toarray", "todense", "data", "indices", "indptr", "T", "A"):
        with pytest.raises(EntryAccessBlocked):
            getattr(op, name)
    with pytest.raises(EntryAccessBlocked):
        np.asarray(op)
    with pytest.raises(EntryAccessBlocked):
        op[0, 0]
    # and the constructor runs through it without tripping any of them
    sc.block_moment(op.matvec, op.rmatvec, 10, 40, jnp.asarray(tau))


def test_a_cone_above_the_supported_block_size_is_refused_with_its_cost():
    rng = np.random.default_rng(11)
    A, tau = instance(rng, 8, 4)
    op = ProductOnlyOperator(A)
    with pytest.raises(sc.ConstructionRefused, match="above the supported"):
        sc.block_moment(op.matvec, op.rmatvec, sc.MAX_CONSTRUCT_K + 1, 8,
                        jnp.asarray(tau))


class NumpyOnlyOperator(ProductOnlyOperator):
    """A legitimate operator whose callable is NOT traceable.

    A caller's ``matvec`` may convert to numpy internally, which cannot be
    vmapped. Refusing such an operator would be refusing a legitimate one, so the
    constructor falls back to the column loop and says so.
    """

    def matvec(self, x):
        object.__setattr__(self, "_count_forward", self._count_forward + 1)
        return np.asarray(self._A, np.float32) @ np.asarray(x, np.float32)

    def rmatvec(self, y):
        object.__setattr__(self, "_count_adjoint", self._count_adjoint + 1)
        return np.asarray(self._A, np.float32).T @ np.asarray(y, np.float32)


def test_an_untraceable_operator_falls_back_and_says_so():
    rng = np.random.default_rng(77)
    n, k = 40, 10
    A, tau = instance(rng, n, k)
    fast = ProductOnlyOperator(A)
    slow = NumpyOnlyOperator(A)
    c_fast, c_slow = sc.ProductCount(), sc.ProductCount()
    G_fast = sc.block_moment(fast.matvec, fast.rmatvec, k, n, jnp.asarray(tau),
                             count=c_fast)
    G_slow = sc.block_moment(slow.matvec, slow.rmatvec, k, n, jnp.asarray(tau),
                             count=c_slow)
    assert c_fast.record()["vectorized"] is True
    assert c_slow.record()["vectorized"] is False
    assert c_slow.adjoint == k and c_slow.forward == k
    # The attempt is paid for, and WHERE it fails decides what it cost: the
    # column function applies the adjoint first, so an untraceable callable is
    # entered once on the adjoint and never reaches the forward.
    assert c_slow.record()["failed_vectorization_probe"] is True
    assert slow._count_adjoint == k + 1, slow._count_adjoint
    assert slow._count_forward == k, slow._count_forward
    rel = float(np.linalg.norm(np.asarray(G_fast) - np.asarray(G_slow))
                / max(1e-30, np.linalg.norm(np.asarray(G_fast))))
    print(f"vectorized against the fallback: {rel:.3e} relative")
    assert rel <= 1e-6, "the two paths must compute the same moment"


def test_the_temporary_never_materializes_the_operators_transpose():
    """The storage claim, not just the access claim.

    `[A^T e_1, ..., A^T e_k] = A^T`, so mapping all `k` columns at once would
    hold the operator's transpose as a temporary. That would still be oracle
    access, no entry was read, but it would destroy the point of keeping a `k` by
    `k` metric instead of a `k` by `n` operator. The microbatch has to bound the
    temporary at `O(n q + k^2)`, and the default must not be `q = k` when that
    would exceed the budget.
    """
    rng = np.random.default_rng(303)
    # n large relative to k, which is the regime the storage claim is about, and
    # a STRUCTURED operator so that regime costs nothing to test: a selection of
    # k coordinates followed by a small dense mix. Its own storage is O(k^2 + k)
    # whatever n is, which is exactly the case where materializing A^T would be
    # the whole cost.
    n, k = 200000, 64
    sel = np.sort(rng.choice(n, size=k, replace=False))
    mix = rng.standard_normal((k, k)).astype(F32)
    tau = np.exp(rng.uniform(-0.5, 0.5, n)).astype(F32)

    class Structured(ProductOnlyOperator):
        def __init__(self):
            object.__setattr__(self, "k", k)
            object.__setattr__(self, "n", n)
            object.__setattr__(self, "_count_forward", 0)
            object.__setattr__(self, "_count_adjoint", 0)

        def matvec(self, x):
            object.__setattr__(self, "_count_forward", self._count_forward + 1)
            return jnp.matmul(jnp.asarray(mix), jnp.asarray(x)[jnp.asarray(sel)], precision=jax.lax.Precision.HIGHEST)

        def rmatvec(self, y):
            object.__setattr__(self, "_count_adjoint", self._count_adjoint + 1)
            small = jnp.matmul(jnp.asarray(mix).T, jnp.asarray(y), precision=jax.lax.Precision.HIGHEST)
            return jnp.zeros((n,), jnp.float32).at[jnp.asarray(sel)].set(small)

    op = Structured()
    count = sc.ProductCount()
    sc.block_moment(op.matvec, op.rmatvec, k, n, jnp.asarray(tau), count=count)
    rec = count.record()
    full_transpose = n * k
    print(f"n={n} k={k}: microbatch {rec['microbatch_columns']}, peak temporary "
          f"{rec['peak_temporary_elements']} elements against {full_transpose} "
          f"for the full transpose")
    assert rec["microbatch_columns"] < k, \
        "the default mapped every column at once, so it held A^T"
    assert rec["peak_temporary_elements"] <= sc.MOMENT_TEMPORARY_ELEMENTS + k * k
    assert rec["peak_temporary_elements"] < full_transpose, \
        "the temporary is no smaller than the operator's transpose"
    # and the applications are still exactly 2k, the microbatch is not a cost
    assert count.adjoint == k and count.forward == k


def test_a_small_operator_may_map_every_column_at_once():
    """When `n k` is inside the budget there is nothing to bound, and `q = k`."""
    rng = np.random.default_rng(304)
    n, k = 40, 12
    A, tau = instance(rng, n, k)
    op = ProductOnlyOperator(A)
    count = sc.ProductCount()
    sc.block_moment(op.matvec, op.rmatvec, k, n, jnp.asarray(tau), count=count)
    assert count.record()["microbatch_columns"] == k
    assert n * k <= sc.MOMENT_TEMPORARY_ELEMENTS


def test_batching_does_not_change_the_moment():
    """The columns may be taken in batches, and the answer may not depend on it."""
    rng = np.random.default_rng(13)
    n, k = 60, 12
    A, tau = instance(rng, n, k)
    op = ProductOnlyOperator(A)
    full = np.asarray(sc.block_moment(op.matvec, op.rmatvec, k, n,
                                      jnp.asarray(tau)))
    for batch in (1, 3, 5, k):
        got = np.asarray(sc.block_moment(op.matvec, op.rmatvec, k, n,
                                         jnp.asarray(tau), batch=batch))
        # To ROUNDOFF, not bitwise: a vectorized map over one column and over k
        # of them are different kernels, and the backend associates their sums
        # differently. The moment is the same operator either way.
        rel = float(np.linalg.norm(got - full) / max(1e-30, np.linalg.norm(full)))
        assert rel <= 1e-6, f"batch {batch} changed the moment by {rel:.3e}"


# ---------------------------------------------------------------------------
# the selection rule, ported unchanged
# ---------------------------------------------------------------------------
def reference_solve_cone(G, r):
    """The archive's rule, re-implemented in float64 from its published source.

    `metrics.solve_cone(G, "spatial", r)` with `analyze.level_solve` and
    `analyze.clipped_roots`, at the archive's constants. Written here from the
    mathematics so the test does not import the code under test.
    """
    G = np.asarray(G, np.float64)
    k = G.shape[0]
    evz, Vz = np.linalg.eigh(0.5 * (G[1:, 1:] + G[1:, 1:].T))
    g = np.maximum(np.concatenate([[G[0, 0]], evz]), 0.0)

    def phi(x, gg):
        return 0.5 * gg * np.exp(x) - 0.5 * r * x + sc.MU * x * x / 8.0

    def h(x, gg):
        return gg * np.exp(x) + 0.5 * sc.MU * x - r

    lo = np.full(g.shape, sc.LO)
    hi = np.full(g.shape, sc.HI)
    for _ in range(sc.ROOT_STEPS):
        mid = 0.5 * (lo + hi)
        pos = h(mid, g) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    roots = 0.5 * (lo + hi)
    roots = np.where(h(np.full(g.shape, sc.LO), g) >= 0, sc.LO, roots)
    roots = np.where(h(np.full(g.shape, sc.HI), g) <= 0, sc.HI, roots)

    if roots.max() - roots.min() <= sc.LCAP:
        x = roots
    else:
        def dphi(xx, gg):
            return 0.5 * gg * np.exp(xx) - 0.5 * r + sc.MU * xx / 4.0

        def dF(m):
            low = roots < m
            high = roots > m + sc.LCAP
            return (np.sum(dphi(m, g[low])) + np.sum(dphi(m + sc.LCAP, g[high])))

        a, b = roots.min(), roots.max() - sc.LCAP
        for _ in range(sc.LEVEL_STEPS):
            mid = 0.5 * (a + b)
            if dF(mid) > 0:
                b = mid
            else:
                a = mid
        level = 0.5 * (a + b)
        x = np.clip(roots, level, level + sc.LCAP)
    return np.exp(x[0]), np.exp(x[1:]), Vz, float(np.sum(phi(x, g)))


def metric_action(a, b, Q, v):
    """`M v` in the original coordinates, host float64."""
    a = float(a)
    b = np.asarray(b, np.float64)
    Q = np.asarray(Q, np.float64)
    v = np.asarray(v, np.float64)
    return np.concatenate([[a * v[0]], Q @ (b * (Q.T @ v[1:]))])


@pytest.mark.parametrize("k", [3, 9, 33, 65])
def test_the_selection_rule_matches_the_archives_on_the_metric_action(k):
    """Compared by ACTION and ENERGY, never by eigenvectors.

    The metric is `diag(a, Q diag(b) Q^T)`. Two eigendecompositions of the same
    spatial block may differ in sign and, inside a repeated eigenspace, in basis,
    while defining the SAME metric. So the comparison is what the metric does.
    """
    rng = np.random.default_rng(200 + k)
    n = 3 * k
    A, tau = instance(rng, n, k)
    r = float(math.sqrt(n / (n + k)))
    op = ProductOnlyOperator(A)
    metric, record = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n,
                                               jnp.asarray(tau), r,
                                               layout=("soc",))
    ra, rb, rQ, renergy = reference_solve_cone(dense_moment(A, tau), r)

    worst = 0.0
    for _ in range(30):
        v = rng.standard_normal(k)
        mine = metric_action(metric.a, metric.b, metric.Q, v)
        theirs = metric_action(ra, rb, rQ, v)
        worst = max(worst, float(np.linalg.norm(mine - theirs)
                                 / max(1e-30, np.linalg.norm(theirs))))
    energy_rel = abs(record["construction_energy"] - renergy) / max(1e-30,
                                                                   abs(renergy))
    print(f"k={k}: action {worst:.3e} relative, energy {energy_rel:.3e} relative, "
          f"{record['products']['total_operator_applications']} applications")
    assert worst <= 5e-4, f"the constructed metric ACTS differently: {worst:.3e}"
    assert energy_rel <= 5e-5, f"the construction energy differs: {energy_rel:.3e}"


def test_the_condition_cap_is_enforced_exactly_as_the_archive_does():
    """A moment with a wide spectrum must come back inside the cap."""
    rng = np.random.default_rng(31)
    k, n = 12, 48
    # a deliberately ill-conditioned block: the spread must exceed LCAP
    scale = np.exp(np.linspace(-6.0, 6.0, k)).astype(F32)
    A = (rng.standard_normal((k, n)) * scale[:, None]).astype(F32)
    tau = np.ones(n, F32)
    r = float(math.sqrt(n / (n + k)))
    op = ProductOnlyOperator(A)
    metric, record = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n,
                                               jnp.asarray(tau), r,
                                               layout=("soc",))
    values = np.concatenate([[float(metric.a)], np.asarray(metric.b, np.float64)])
    spread = float(np.log(values.max() / values.min()))
    print(f"log spread {spread:.4f} against the cap {sc.LCAP:.4f}")
    assert spread <= sc.LCAP + 1e-3, "the condition cap was not enforced"
    ra, rb, _rQ, _re = reference_solve_cone(dense_moment(A, tau), r)
    ref_values = np.concatenate([[ra], rb])
    ref_spread = float(np.log(ref_values.max() / ref_values.min()))
    assert abs(spread - ref_spread) <= 1e-3, \
        "the cap is enforced differently from the archive"


def test_the_moment_floor_is_reported_rather_than_hidden():
    """A slightly indefinite moment is floored, and the floor is a reported fact."""
    rng = np.random.default_rng(41)
    # n < k, so A T A^T has rank at most n and the moment is genuinely singular:
    # with n = 6 and k = 10 four of its eigenvalues are zero up to float32 noise,
    # and some of those come out slightly NEGATIVE, which is the case the floor
    # exists for.
    k, n = 10, 6
    A, tau = instance(rng, n, k)
    r = float(math.sqrt(n / (n + k)))
    op = ProductOnlyOperator(A)
    _metric, record = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n,
                                                jnp.asarray(tau), r,
                                                layout=("soc",))
    print(f"floor active on {record['moment_floor_active']} of {k} energies, "
          f"worst raw eigenvalue {record['moment_min_eigenvalue']:.3e}")
    assert "moment_floor_active" in record
    assert "moment_min_eigenvalue" in record
    assert record["moment_floor_active"] > 0, \
        "this fixture no longer produces a moment the floor has to act on"
    assert record["moment_min_eigenvalue"] < 0.0, \
        "the floor is reported but nothing was actually negative"


def test_the_record_says_the_construction_is_conditional_on_T():
    rng = np.random.default_rng(51)
    k, n = 6, 24
    A, tau = instance(rng, n, k)
    op = ProductOnlyOperator(A)
    _m, record = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n,
                                           jnp.asarray(tau), 0.7,
                                           layout=("soc",))
    assert "conditional" in record and "FIXED" in record["conditional"]
    assert record["dtype"] == "float32", record["dtype"]


# ---------------------------------------------------------------------------
# the whole path, with entries blocked throughout
# ---------------------------------------------------------------------------
def test_the_constructed_metric_drives_the_existing_weighted_update():
    """`(A, A^T) -> (a, b, Q) -> the existing weighted SOC update`, no entries."""
    rng = np.random.default_rng(61)
    k, n = 17, 51
    A, tau = instance(rng, n, k)
    r = float(math.sqrt(n / (n + k)))
    op = ProductOnlyOperator(A)
    metric, record = sc.construct_spatial_full(op.matvec, op.rmatvec, k, n,
                                               jnp.asarray(tau), r,
                                               layout=("soc",))
    v = jnp.asarray(rng.standard_normal(k), jnp.float32)
    out = ss.spatial_full_moreau(None, v, jnp.float32(1.0), ((0, 1, metric),))
    assert np.all(np.isfinite(np.asarray(out)))
    # the same update through the primitive's own entry point, as a control
    direct, _status = ss.project_polar_step(jnp.reshape(v, (1, -1)), metric)
    assert np.allclose(np.asarray(out), np.asarray(direct)[0], rtol=1e-5,
                       atol=1e-6)
    assert record["products"]["total_operator_applications"] == 2 * k
    print(f"constructed and applied with {2 * k} operator applications, "
          f"energy {record['construction_energy']:.6f}")
