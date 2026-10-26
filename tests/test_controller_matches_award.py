"""The live controller stops successfully exactly when the award would.

Before 0.3.0rc3 the scalar drive loop stopped on ``gap <= tol`` while
``_award`` refused a gap within the float32 resolution of ``tol`` as
``unverified``, so a lane could halt on a point the award then declined to
decide.  One predicate, ``_gap_accepts``, now serves the award, the scalar
loop and the batched loop.  These tests place packets immediately below,
inside and above the ACTUAL float32 boundary region, computed from the same
resolution floor the award applies, and exercise the controller on them.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import ocs
from ocs.model.compile import (
    _FLOAT32_EPS, _gap_accepts, _gap_resolution, Solution,
)

TOL = 1e-5
PRIMAL = 1.5          # an ordinary objective scale: the floor is 4 ulps there


def _family(max_blocks=40):
    weights = jnp.array([1.0, 2.0, 3.0])
    K = ocs.LinearOperator.from_forward(lambda v: weights * v, n=3, m=3)
    x = ocs.Variable(3)
    b = ocs.Parameter("b")
    return ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - b)]).compile(
        tol=TOL, block=10, max_blocks=max_blocks, equilibrate="scalar",
        gap_check_first=1, gap_check_growth=2.0)


def _packet(gap, *, primal=PRIMAL, infeasibility=0.0, available=True,
            finite_primal=True):
    """A bound report whose REALIZED acceptance gap is as close to ``gap`` as
    float32 values allow.  ``gap`` is the scale-invariant discrepancy
    (primal - lower) / max(|primal|, |lower|)."""
    p = float(np.float32(primal))
    lower = p * (1.0 - gap) if gap >= 0.0 else p / (1.0 + gap)
    lower = float(np.float32(lower))
    return {"objective_lower_bound_available": bool(available),
            "relative_gap_certified": bool(available and finite_primal),
            "objective_lower_bound": lower if available else float("nan"),
            "objective_primal_value": p if finite_primal else float("inf"),
            "objective_at_returned_point": p if finite_primal else float("inf"),
            "objective_primal_upper_bound": p if finite_primal else float("inf"),
            "relative_primal_infeasibility": float(infeasibility),
            "unavailable_reason": "test packet: no witness",
            "gap_unavailable_reason": "test packet: cost not finite"}


def _realized(packet):
    p, l = packet["objective_primal_value"], packet["objective_lower_bound"]
    return (p - l) / max(abs(p), abs(l))


FLOOR = _gap_resolution(PRIMAL)
assert FLOOR == 4.0 * _FLOAT32_EPS           # |primal| >= 1: four ulps

BOUNDARY = {
    "well below": TOL - 3.0 * FLOOR,
    "just below the band": TOL - 1.5 * FLOOR,
    "inside the band, below": TOL - 0.5 * FLOOR,
    "at the tolerance": TOL,
    "inside the band, above": TOL + 0.5 * FLOOR,
    "just above the band": TOL + 1.5 * FLOOR,
    "well above": TOL + 3.0 * FLOOR,
    "negative, within": -(TOL - 3.0 * FLOOR),
    "negative, beyond": -(TOL + 3.0 * FLOOR),
}


def _award_status(compiled, packet):
    s = Solution(jnp.zeros(3), jnp.zeros(3), 0, 0.0, status="solved")
    return compiled._award(s, packet, None)


# ---- the predicate IS the award --------------------------------------------

@pytest.mark.parametrize("name", sorted(BOUNDARY))
def test_predicate_agrees_with_award_on_the_boundary(name):
    compiled = _family()
    packet = _packet(BOUNDARY[name])
    gap = _realized(packet)
    awarded = _award_status(compiled, packet)
    accepts = _gap_accepts(True, True, gap, 0.0, packet["objective_primal_value"], TOL)
    assert (awarded.status == "solved") == accepts
    if abs(gap - TOL) < FLOOR:
        # Neither met nor missed: refused by name, never decided by the loop.
        assert awarded.status == "unverified"
        assert "float32 can resolve" in awarded.accuracy_detail
    elif abs(gap) <= TOL:
        assert awarded.status == "solved"
    else:
        assert awarded.status == "inaccurate"


def test_band_packets_really_lie_inside_the_float32_resolution():
    inside = [_realized(_packet(BOUNDARY[k])) for k in
              ("inside the band, below", "at the tolerance", "inside the band, above")]
    outside = [_realized(_packet(BOUNDARY[k])) for k in
               ("just below the band", "just above the band")]
    assert all(abs(g - TOL) < FLOOR for g in inside)
    assert all(abs(g - TOL) >= FLOOR for g in outside)


def test_feasibility_is_tested_before_the_sign_of_the_discrepancy():
    """A negative discrepancy at an infeasible point bounds nothing."""
    compiled = _family()
    packet = _packet(-(TOL / 2), infeasibility=10 * TOL)
    awarded = _award_status(compiled, packet)
    assert awarded.status == "inaccurate"
    assert "infeasibility" in awarded.accuracy_detail
    assert not compiled.acceptance(packet)["target_met"]
    assert not _gap_accepts(True, True, _realized(packet), 10 * TOL, PRIMAL, TOL)


def test_invalid_or_unavailable_witness_is_never_a_success():
    compiled = _family()
    no_witness = _packet(TOL / 2, available=False)
    assert _award_status(compiled, no_witness).status == "unverified"
    assert not _gap_accepts(False, False, float("nan"), 0.0, PRIMAL, TOL)
    not_finite = _packet(TOL / 2, finite_primal=False)
    assert _award_status(compiled, not_finite).status == "inaccurate"
    assert not _gap_accepts(True, False, float("nan"), 0.0, PRIMAL, TOL)


# ---- scalar, batch and comparator decide the same packet the same way -------

def test_scalar_batch_and_comparator_decide_one_packet_alike():
    compiled = _family()
    packets = [_packet(BOUNDARY[k]) for k in sorted(BOUNDARY)]
    packets += [_packet(TOL / 2, infeasibility=10 * TOL),
                _packet(TOL / 2, available=False)]
    scalar = [_gap_accepts(p["objective_lower_bound_available"],
                           p["relative_gap_certified"], _realized(p),
                           p["relative_primal_infeasibility"],
                           p["objective_primal_value"], TOL) for p in packets]
    # The batched loop evaluates the identical expression on arrays.
    arr = lambda k, dt: jnp.asarray([p[k] for p in packets], dt)
    gaps = jnp.asarray([_realized(p) for p in packets], jnp.float64
                       if jnp.zeros(1).dtype == jnp.float64 else jnp.float32)
    batched = _gap_accepts(arr("objective_lower_bound_available", bool),
                           arr("relative_gap_certified", bool), gaps,
                           arr("relative_primal_infeasibility", jnp.float32),
                           arr("objective_primal_value", jnp.float32), TOL)
    awarded = [_award_status(compiled, p).status == "solved" for p in packets]
    comparator = [compiled.acceptance(p)["target_met"] for p in packets]
    for p, s, b, a, c in zip(packets, scalar, np.asarray(batched), awarded, comparator):
        assert s == bool(b) == a, p
        # The comparator now shares the controller's unresolved boundary.
        assert c == a, p


# ---- the live controller --------------------------------------------------

class _Scripted:
    """Replaces the bound evaluator with a scripted sequence of packets and
    records the exact pair it assessed at every check."""

    def __init__(self, packets):
        self.packets, self.calls, self.pairs = packets, 0, []

    def __call__(self, x, y, dense, rounds, return_dual=False):
        i = min(self.calls, len(self.packets) - 1)
        self.calls += 1
        self.pairs.append((np.asarray(x), np.asarray(y)))
        packet = dict(self.packets[i])
        if return_dual:
            packet["certifying_dual"] = y
        return packet


def test_controller_continues_past_a_band_point_and_returns_the_assessed_pair():
    compiled = _family()
    band, accept = _packet(BOUNDARY["inside the band, below"]), _packet(BOUNDARY["well below"])
    # Nominally `gap <= tol` at the first two checks, but inside the band:
    # the former loop stopped there and was then refused as unverified.
    script = _Scripted([band, band, accept])
    compiled._objective_bound = script
    sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
    assert script.calls == 3
    assert sol.status == "solved"
    assert sol.objective_gap == pytest.approx(_realized(accept))
    assert sol.objective_lower_bound == accept["objective_lower_bound"]
    assert len(sol.gap_history) == 3
    assert sol.iters == sol.gap_history[-1]["iterations"] > 0
    # The returned pair is the one the accepting check assessed.
    np.testing.assert_array_equal(np.asarray(sol.x), script.pairs[2][0])
    np.testing.assert_array_equal(np.asarray(sol.y), script.pairs[2][1])
    assert compiled._award(sol, accept, None).status == "solved"


def test_controller_retains_stronger_lower_with_current_cost_and_old_witness():
    compiled = _family()
    first = _packet(.5, primal=2.)
    weaker = _packet(.2, primal=1.000001)
    script = _Scripted([first, weaker])
    compiled._objective_bound = script
    sol = compiled.solve_instance({'b':jnp.array([-1.,2.,3.])})
    assert script.calls == 2
    assert sol.status == 'solved'
    assert sol.objective_bound_source == 'retained_witness'
    assert sol.objective_lower_bound == first['objective_lower_bound']
    assert sol.objective_gap == pytest.approx((weaker['objective_primal_value']-1.)/weaker['objective_primal_value'])
    np.testing.assert_array_equal(np.asarray(sol.y),script.pairs[0][1])
    assert sol.gap_history[-1]['retained_previous_lower']


def test_batched_retention_keeps_each_lane_lower_but_current_feasibility():
    from ocs.model.compile import _retain_stronger_lower
    old=dict(objective_lower_bound=jnp.array([1.,2.]),
             objective_lower_bound_available=jnp.array([True,False]),source_code=jnp.array([4.,0.]))
    now=dict(objective_lower_bound=jnp.array([.5,3.]),
             objective_lower_bound_available=jnp.array([True,True]),source_code=jnp.array([1.,1.]),
             objective_primal_value=jnp.array([1.000001,3.]),
             relative_primal_infeasibility=jnp.array([.1,.2]))
    out,mask=_retain_stronger_lower(now,old,batched=True)
    np.testing.assert_array_equal(mask,[True,False])
    np.testing.assert_array_equal(out['objective_lower_bound'],[1.,3.])
    np.testing.assert_array_equal(out['relative_primal_infeasibility'],now['relative_primal_infeasibility'])
    np.testing.assert_array_equal(out['source_code'],[6.,1.])


def test_controller_at_its_work_limit_reports_the_unmet_target_honestly():
    compiled = _family(max_blocks=30)
    band = _packet(BOUNDARY["at the tolerance"])
    script = _Scripted([band])
    compiled._objective_bound = script
    sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
    assert sol.status == "unverified"
    assert "float32 can resolve" in sol.accuracy_detail
    assert sol.iters == sol.gap_history[-1]["iterations"]
    # Every check on the schedule was consumed and none succeeded: the target
    # is reported as not established, never as met.  (The inner iteration
    # may converge on its residual before the block budget is spent, which
    # is why the iteration count, not the schedule, can end below budget.)
    assert script.calls == len(sol.gap_history) >= 3
    assert sol.objective_gap is not None and sol.status != "solved"


def test_controller_never_succeeds_on_an_invalid_witness():
    compiled = _family(max_blocks=30)
    for packet in (_packet(TOL / 2, available=False),
                   _packet(TOL / 2, finite_primal=False),
                   _packet(TOL / 2, infeasibility=10 * TOL)):
        script = _Scripted([packet])
        compiled._objective_bound = script
        sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
        assert sol.status != "solved", packet
        assert script.calls == len(sol.gap_history) > 1


def test_batched_controller_stops_each_lane_exactly_where_its_award_accepts():
    compiled = _family(max_blocks=30)
    band, accept = _packet(BOUNDARY["inside the band, above"]), _packet(BOUNDARY["well below"])
    none = _packet(TOL / 2, available=False)
    # lane 0: band, band, accept; lane 1: accept at once; lane 2: never valid.
    rows = [[band, accept, none], [band, accept, none], [accept, accept, none]]
    keys = [k for k in band if not k.endswith("_reason")]

    class Many:
        calls = 0

        def __call__(self, x, y, dense, rounds, return_dual=False):
            i = min(Many.calls, len(rows) - 1)
            Many.calls += 1
            out = {}
            for k in keys:
                vals = [rows[i][lane][k] for lane in range(3)]
                out[k] = jnp.asarray(vals, bool if isinstance(vals[0], bool) else jnp.float32)
            return (out, y) if return_dual else out

    compiled._objective_bound_many = Many()
    b = jnp.array([[-1.0, 2.0, 3.0]] * 3)
    results = compiled.solve_many_results({"b": b})
    assert [r.status for r in results] == ["solved", "solved", "unverified"]
    assert results[1].iters == 0
    assert results[0].iters > 0
    assert results[2].iters > results[0].iters
    assert results[0].objective_gap == pytest.approx(_realized(accept))
    assert Many.calls >= 3


def test_an_ordinary_solve_is_accepted_through_the_same_predicate():
    compiled = _family()
    sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
    assert sol.status == "solved"
    last = sol.gap_history[-1]
    assert _gap_accepts(last["bound_available"], True, last["relative_gap"],
                        last["relative_primal_infeasibility"],
                        sol.objective_primal_value, TOL)
    np.testing.assert_allclose(np.asarray(sol.x), [0, 1, 1], atol=3e-3)
