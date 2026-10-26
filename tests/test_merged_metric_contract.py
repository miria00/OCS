"""The ONE returned contract of ``normalize_term_metrics``, across all carriers.

The three metric tiers of this release reach the compiled loop through one
return value, ``(k, ranges)``. Eleven call sites in ``ocs/prox`` and
``ocs/model`` read that pair and none of them type-checks every field, so the
invariant is pinned here once instead of being implied by those call sites.

Host only, float64, no solve: every assertion below is about validation code
that runs on the host at compile time.
"""
import numpy as np
import pytest

import ocs
from ocs.prox import block_metric as bm
from ocs.prox import cone_metric as cm
from ocs.prox import cones


def assert_pair_contract(k, ranges, size):
    """I1 to I6 of work/merged/CONTRACT.md, on one returned pair."""
    # I1: k is a positive divisor of the term's row count.
    assert isinstance(k, int) and k > 0 and size % k == 0
    nblocks = size // k
    # I2: a non-empty tuple of 3-tuples with int bounds.
    assert isinstance(ranges, tuple) and ranges
    for entry in ranges:
        assert isinstance(entry, tuple) and len(entry) == 3
        assert isinstance(entry[0], int) and isinstance(entry[1], int)
    # I3: sorted, contiguous from 0, covering exactly [0, nblocks).
    cursor = 0
    for start, end, _carrier in ranges:
        assert start == cursor and end > start
        cursor = end
    assert cursor == nblocks
    # I4: one carrier family per term, never mixed.
    carriers = [entry[2] for entry in ranges]
    if isinstance(carriers[0], cm.ConeAutomorphism):
        assert len(carriers) == 1
        # I6: the whole term is one plumbing block.
        assert k == size and nblocks == 1 and ranges[0][:2] == (0, 1)
        assert carriers[0].cones is not None and carriers[0].cones >= 1
    else:
        assert all(isinstance(c, bm.BlockMetric) for c in carriers)
        # I5: every shape carrier agrees with k.
        assert all(c.k == k for c in carriers)


def test_norm_shape_carrier_satisfies_the_pair_contract():
    metric = bm.BlockMetric([1.0, 2.0, 3.0])
    k, ranges = bm.normalize_term_metrics(metric, 0, ocs.GroupL2(1.0, 3), 30)
    assert_pair_contract(k, ranges, 30)
    assert k == 3 and len(ranges) == 1 and ranges[0][:2] == (0, 10)
    assert not isinstance(ranges[0][2], bm.QuadraticShape)


def test_quadratic_shape_carrier_satisfies_the_pair_contract():
    metric = bm.BlockMetric([1.0, 2.0])
    k, ranges = bm.normalize_term_metrics(metric, 0, ocs.SumSquares(), 8)
    assert_pair_contract(k, ranges, 8)
    assert k == 2 and isinstance(ranges[0][2], bm.QuadraticShape)


def test_cone_carrier_satisfies_the_pair_contract():
    rng = np.random.default_rng(3)
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u)
    w = np.stack([np.r_[1.3, 0.4 * u], np.r_[1.1, 0.2 * u]])
    k, ranges = bm.normalize_term_metrics(cm.ConeAutomorphism.soc(w), 0, cones.SOCBatch(2, 3), 8)
    assert_pair_contract(k, ranges, 8)
    carrier = ranges[0][2]
    # Every cone-level fact is on the carrier, never on k or on a range bound.
    assert carrier.cones == 2 and carrier.count == 2 and carrier.k == 4
    assert carrier.algebra == "soc" and carrier.layout == ("socbatch", 2, 3)


def test_a_column_pair_built_by_hand_satisfies_the_same_contract():
    """The declared column metric hand-builds the pair; it is the same object."""
    shape = bm.BlockMetric.from_matrix(np.array([[2.0, 0.5], [0.5, 1.5]]))
    n = 8
    pair = (shape.k, ((0, n // shape.k, shape),))
    assert_pair_contract(pair[0], pair[1], n)
    apply = bm.make_shape_apply({0: pair}, ((0, n),), n)
    v = np.arange(1.0, n + 1.0)
    out = np.asarray(apply(v, power=1), dtype=np.float64)
    # (I kron S) v on contiguous blocks of k coordinates; S is k x k and host-side.
    expected = (v.reshape(-1, shape.k) @ np.asarray(shape.matrix(), np.float64).T).ravel()
    assert out.shape == (n,)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_a_symmetric_cone_block_still_refuses_a_block_metric():
    with pytest.raises(ValueError, match="not a supported") as err:
        bm.normalize_term_metrics(bm.BlockMetric([1.0, 2.0, 3.0, 4.0]), 0,
                                  cones.SecondOrderCone(), 4)
    message = str(err.value)
    assert "not a supported rotation-invariant" in message
    assert "not a supported quadratic block" in message
    assert "declare a ConeAutomorphism instead" in message


def test_a_norm_block_refusal_carries_no_cone_hint():
    with pytest.raises(ValueError, match="not a supported") as err:
        bm.normalize_term_metrics(bm.BlockMetric([1.0, 2.0]), 0, ocs.NonNeg(), 4)
    assert "declare a ConeAutomorphism instead" not in str(err.value)


def test_a_list_of_cone_carriers_is_refused_by_name():
    rng = np.random.default_rng(5)
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u)
    w = np.r_[1.2, 0.3 * u]
    with pytest.raises(ValueError, match="takes exactly ONE ConeAutomorphism"):
        bm.normalize_term_metrics([cm.ConeAutomorphism.soc(w)], 0, cones.SecondOrderCone(), 4)


def test_one_compile_carries_every_items_disclosure_keys():
    """Contract risk 3: the three-way merge of ``_describe_block_metrics`` must not
    drop any item's keys. One construct compile declares a quadratic shared shape
    (item 1), a cone automorphism (item 2) and a full column shape (item 3), and
    every item's public keys must be present in the one record. Compile only: this
    test makes no claim about a solve.

    This exercises the public opt-in constructor without a private switch."""
    from ocs.ir import Term
    rng = np.random.default_rng(11)
    n, k = 12, 3
    K0 = rng.normal(size=(n, n)) / np.sqrt(n)
    K1 = rng.normal(size=(8, n)) / np.sqrt(n)
    K2 = rng.normal(size=(6, n)) / np.sqrt(n)
    x = ocs.Variable(n)
    prob = ocs.minimize([ocs.sum_squares(ocs.Dense(K0) @ x + ocs.Parameter("l0")),
                         Term(ocs.Dense(K1), cones.SOCBatch(2, 3), lambda d: d["l1"]),
                         ocs.nonneg(ocs.Dense(K2) @ x + ocs.Parameter("l2"))],
                        cost=ocs.Parameter("c"))
    comp = prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3, metric_updates=12,
                        metric_classes={0: ("full", "term", k), 1: "automorphism"},
                        metric_columns=("full", k))
    record = comp.meta["block_metrics"]
    assert record["present"] is True
    by_term = {t["term"]: t for t in record["terms"]}
    # item 1: the quadratic shape and its domain constant
    assert by_term[0]["block_family"] == "quadratic"
    assert "max_quadratic_hessian_ratio" in record["numerical_domain"]
    # item 2: the cone carrier, its cone count read from the carrier, its domain constant
    cone = by_term[1]
    assert cone["block_family"] == "cone" and cone["mathematical_blocks"] == 2
    assert cone["cone_structure"]["cones"] == 2 and cone["k"] == 4 and cone["rows"] == 8
    assert cone["sharing_groups"][0]["groups"] == [0, 2]
    assert "max_automorphism_cond" in record["numerical_domain"]
    assert "automorphism_domain_status" in record["numerical_domain"]
    # item 3: the column record, and the carrier text keeps the cone description
    assert record["columns"]["present"] is True and record["columns"]["k"] == k
    assert "Jordan element per cone" in record["carrier"]
    assert record["carrier"].endswith("the column shape is one (Q, w) pair applied to the primal vector")
    assert 0 < record["measured_scaled_norm"] < 1
    assert comp.meta["eval_kinds"][1] == cones.SOCBatch(2, 3).eval_kind
