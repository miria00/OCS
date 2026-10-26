"""The objective-gap check schedule is an explicit, recorded instrument.

The default is the shipped geometric schedule 1, 2, 4, 8, ... blocks, at most
64 checks, summing to the block budget.  ``gap_check_growth`` and
``gap_check_first`` change WHEN the criterion is asked, never what it accepts,
and the schedule in force is recorded in ``meta["gap_check_schedule"]`` and
followed by the drive loop, whose per-check history reports the segments.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.model.compile import _check_schedule
from ocs.prox.library import SumSquares


def test_default_schedule_is_geometric_and_sums_to_the_budget():
    s = _check_schedule(1000)
    assert s[:7] == [1, 2, 4, 8, 16, 32, 64]
    assert sum(s) == 1000 and len(s) <= 64


def test_finer_growth_resolves_more_finely_and_still_sums_to_the_budget():
    s = _check_schedule(1000, first=1, growth=1.25, max_checks=64)
    assert sum(s) == 1000 and len(s) <= 64
    assert max(s[:-1]) < 64                      # finer than the default's seventh segment
    u = _check_schedule(40, first=4, growth=1.0, max_checks=64)
    assert u == [4] * 10


def _family():
    rng = np.random.default_rng(5)
    n, g, k = 40, 12, 3
    A = rng.normal(size=(g * k, n)) / np.sqrt(n)
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(1e-2)),
                         ocs.group_l2(ocs.Dense(A) @ x - bp, lam=0.5, group_size=k)])
    return prob, {"b": jnp.asarray(rng.normal(size=g * k).astype(np.float32))}


def test_the_loop_follows_the_declared_schedule_and_records_it():
    prob, data = _family()
    default = prob.compile(block=20, max_blocks=400, tol=1e-4)
    fine = prob.compile(block=20, max_blocks=400, tol=1e-4, gap_check_growth=1.25)
    assert default.meta["gap_check_schedule"]["growth"] == 2.0
    assert default.meta["gap_check_schedule"]["segments_blocks"][:4] == [1, 2, 4, 8]
    assert fine.meta["gap_check_schedule"]["growth"] == 1.25
    for comp in (default, fine):
        s = comp.solve_instance(data)
        segs = [h["segment_blocks"] for h in (s.gap_history or [])]
        declared = [0] + comp.meta["gap_check_schedule"]["segments_blocks"]
        assert segs == declared[:len(segs)], (segs, declared[:len(segs)])
    # the acceptance decision is the same function: both runs are solved on the gap.
    # NO monotonicity is asserted between the two schedules: in this release every
    # segment boundary re-enters the iteration from (x, repaired y), which restarts
    # the Halpern anchor, the block averages and the primal weight, so a finer
    # schedule can stop LATER.  Recorded in RC2_INTEGRATION_NOTE.md as a limitation.
    a, b = default.solve_instance(data), fine.solve_instance(data)
    assert a.status == b.status == "solved" and a.accuracy_criterion == b.accuracy_criterion == "objective_gap"


def test_bad_growth_is_refused():
    prob, _ = _family()
    with pytest.raises(ValueError, match="gap_check_growth"):
        prob.compile(block=20, max_blocks=400, tol=1e-4, gap_check_growth=0.5)
