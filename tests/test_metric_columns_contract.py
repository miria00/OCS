"""The ``metric_columns`` contract, tested where the option must have an effect.

The contract, stated once in ``ocs.solver.metric_construct.COLUMN_CONTRACT`` and
in ``docs/scaling_contract.md``:

    ``metric_columns="scalar"`` declares ONE SCALAR STEP PER DECLARED PRIMAL
    PROXIMAL BLOCK. It never means one scalar for the whole primal space; that
    coarser restriction is declared as one span, ``metric_columns=[(0, n)]``.

Every test below inspects the metric that was CONSTRUCTED and APPLIED -- the
compiler's own ``meta["effective_metric"]["primal"]`` record and the step array
itself -- never the arm name, the keyword, or the constructor's declaration
echo. The fixture is deliberately anisotropic and its nonuniformity is verified
first, so the fixture can tell the contracts apart; a fixture whose optimal
metric is already scalar would prove nothing and is not used.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import ocs
from ocs.ir import Term
from ocs.model import compile as cm
from ocs.prox import L2, GroupL2, SumSquares
from ocs.solver import equilibrate_sgd as es
from ocs.solver import metric_construct as mc


# --------------------------------------------------------------- the fixtures
def anisotropic_family(n=12, seed=0, spread=100.0):
    """A separable primal block (one coordinate per primal block) and an operator
    whose column energies are spread over two orders of magnitude, so the free
    diagonal construction is strongly nonuniform."""
    rng = np.random.default_rng(seed)
    scale = np.geomspace(1.0, spread, n)
    K1 = rng.normal(size=(12, n)) * scale
    K2 = rng.normal(size=(6, n)) * scale[::-1]
    x = ocs.Variable(n)
    p1, p2 = ocs.Parameter("l1"), ocs.Parameter("l2")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(0.5)),
                         ocs.group_l2(ocs.Dense(K1) @ x + p1, lam=0.9, group_size=3),
                         ocs.nonneg(ocs.Dense(K2) @ x + p2)])
    data = {"l1": jnp.asarray(rng.normal(size=12), jnp.float32),
            "l2": jnp.asarray(rng.normal(size=6), jnp.float32)}
    return prob, data, n


def coupled_primal_family(primal, n=9, seed=1):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(12, n)) * np.geomspace(1.0, 100.0, n)
    x = ocs.Variable(n)
    p = ocs.Parameter("l")
    prob = ocs.minimize([Term(ocs.Identity(n), primal),
                         ocs.nonneg(ocs.Dense(K) @ x + p)])
    return prob, {"l": jnp.asarray(rng.normal(size=12), jnp.float32)}, n


_CAPTURED = []


def _capturing_make_solver(matvec, rmatvec, tau, sigma, *args, **kwargs):
    """Intercept the steps the compiled loop is actually handed."""
    _CAPTURED.append((np.asarray(jax.device_get(tau)),
                      np.asarray(jax.device_get(sigma))))
    return _capturing_make_solver.wrapped(matvec, rmatvec, tau, sigma, *args, **kwargs)


@pytest.fixture(autouse=True)
def capture_applied_steps(monkeypatch):
    _CAPTURED.clear()
    _capturing_make_solver.wrapped = cm.make_solver
    monkeypatch.setattr(cm, "make_solver", _capturing_make_solver)
    yield


def _attach(comp):
    """Attach the primal step ARRAY this compile handed to the solver, taken at
    compile time: reading it later would read a later compile's metric."""
    comp.applied_tau = np.ascontiguousarray(
        np.broadcast_to(np.atleast_1d(_CAPTURED[-1][0]), (comp.n,)))
    return comp


def build(prob, **kwargs):
    return _attach(prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3,
                                metric_updates=12, **kwargs))


def steps(comp):
    """The primal steps the compiled loop will use: the ARRAY it was handed."""
    return comp.applied_tau.tobytes(), comp.applied_tau


# ------------------------------------------------- the fixture is fit for use
def test_the_fixture_is_anisotropic_enough_to_separate_the_contracts():
    """Verified BEFORE anything else: on this operator the free diagonal
    construction is demonstrably nonuniform, so a test that a declared tie
    changes the applied metric is a test that can fail."""
    prob, _data, n = anisotropic_family()
    primal = build(prob, metric_columns="diagonal").meta["effective_metric"]["primal"]
    assert primal["distinct_steps"] > 1
    assert primal["steps_max"] / primal["steps_min"] > 3.0, primal


# --------------------------------------------------- the contract, per block
def test_declared_scalar_is_one_scalar_per_primal_block_not_one_for_the_space():
    """The primal block here is separable: one coordinate per block. The
    per-block scalar class and the per-coordinate diagonal class are then the
    SAME set of metrics, so the declaration is honoured by the free diagonal --
    an identity of feasible sets, not a reduction of one -- and the record must
    say so with n sharing groups, not one."""
    prob, _data, n = anisotropic_family()
    comp = build(prob, metric_columns="scalar")
    columns = comp.meta["metric_construction"]["columns"]
    assert columns["declared"] == "scalar"
    assert columns["primal_blocks"] == {"block": "_Scaled", "k": 1, "count": n}
    assert columns["metric_sharing_groups"] == n
    assert "free per-coordinate diagonal" in columns["applied"]
    # ... and the applied metric agrees with the record.
    assert comp.meta["effective_metric"]["primal"]["distinct_steps"] <= n


@pytest.mark.parametrize("declaration", [{}, dict(metric_columns="scalar"),
                                         dict(metric_columns="diagonal")])
def test_the_report_and_the_applied_metric_never_contradict_each_other(declaration):
    """The defect this file exists for: a declaration reported as ONE sharing
    group while the applied metric had 2,389,602 distinct primal steps. The
    record and the steps are compared on every declaration."""
    prob, _data, _n = anisotropic_family()
    comp = build(prob, **declaration)
    columns = comp.meta["metric_construction"]["columns"]
    applied = comp.meta["effective_metric"]["primal"]["distinct_steps"]
    assert applied <= columns["metric_sharing_groups"], (columns, applied)
    assert columns["enforced_on_applied_steps"]


def test_a_whole_space_tie_is_a_span_and_it_changes_the_applied_metric():
    """One scalar for the WHOLE primal space is spelled [(0, n)]. On the
    anisotropic fixture it is a different metric from the free diagonal, and
    the difference is read off the applied steps."""
    prob, _data, n = anisotropic_family()
    free = build(prob, metric_columns="diagonal")
    tied = build(prob, metric_columns=[(0, n)])
    assert tied.meta["metric_construction"]["columns"]["metric_sharing_groups"] == 1
    assert tied.meta["effective_metric"]["primal"]["distinct_steps"] == 1
    assert free.meta["effective_metric"]["primal"]["distinct_steps"] > 1
    assert steps(free)[0] != steps(tied)[0]
    lo, hi = (tied.meta["effective_metric"]["primal"][k] for k in ("steps_min", "steps_max"))
    assert lo == hi


def test_one_coupled_primal_block_makes_scalar_one_step_built_with_the_tie():
    """L2 over the whole primal space is ONE primal block, so one scalar per
    block is one scalar for the space -- and it is constructed with the tie in
    place (the shipped program's per-step projection onto the constant
    subspace), not by collapsing a free diagonal afterwards."""
    prob, _data, n = coupled_primal_family(L2(0.4))
    comp = build(prob, metric_columns="scalar")
    columns = comp.meta["metric_construction"]["columns"]
    assert columns["primal_blocks"] == {"block": "L2", "k": n, "count": 1}
    assert columns["metric_sharing_groups"] == 1
    assert comp.meta["metric_construction"]["tie_columns"] is True
    assert comp.meta["effective_metric"]["primal"]["distinct_steps"] == 1
    assert comp.meta["effective_metric"]["primal"]["tie"] == "construction"


def test_per_block_scalar_on_several_coupled_blocks_is_refused_not_replaced():
    """GroupL2 over 9 coordinates in groups of 3 is THREE primal blocks, so the
    declaration asks for three distinct primal steps. This release applies
    exactly one step to a coupled primal prox, so the declaration is refused.
    Before this contract it was accepted and one step was applied silently."""
    prob, _data, _n = coupled_primal_family(GroupL2(0.5, group_size=3))
    with pytest.raises(ValueError, match="neither built nor applied"):
        build(prob, metric_columns="scalar")


# ------------------------------------------------------ order, and the caches
@pytest.mark.parametrize("order", [("diagonal", "tie", "diagonal", "tie"),
                                   ("tie", "diagonal", "tie", "diagonal"),
                                   ("scalar", "diagonal", "scalar")])
def test_declarations_survive_each_other_in_one_process(order):
    """Every compile in one process, through the public compiler and its
    caches. Each declaration must get its own metric every time it is asked
    for, whatever ran before it: a cached executable or metric built under the
    previous declaration would show up as a repeat that does not match."""
    prob, _data, n = anisotropic_family()
    seen, records = {}, {}
    for name in order:
        kwargs = {"tie": dict(metric_columns=[(0, n)])}.get(
            name, dict(metric_columns=name))
        comp = build(prob, **kwargs)
        digest, _tau = steps(comp)
        columns = comp.meta["metric_construction"]["columns"]
        record = (columns["declared"], columns["kind"], columns["metric_sharing_groups"],
                  comp.meta["effective_metric"]["primal"]["distinct_steps"])
        if name in seen:
            assert digest == seen[name], f"{name} changed after another declaration ran"
            assert record == records[name]
        seen[name], records[name] = digest, record
    if "tie" in seen and "diagonal" in seen:
        assert seen["tie"] != seen["diagonal"]


def test_the_smfe_program_cache_key_carries_the_restriction():
    """White box, because the cache is what the defect could hide behind: two
    runs that differ ONLY in the declared column restriction must not share a
    cached program, and two runs that agree on it must."""
    rng = np.random.default_rng(3)
    n, m = 12, 16
    op = ocs.Dense(rng.normal(size=(m, n)) * np.geomspace(1.0, 100.0, n))
    args = dict(iters=8, probes=1, _cache_operators=(op,))
    before = dict(es._SMFE_CACHE_STATS)

    def run(**extra):
        tau, _sigma, _nrm = es.smfe_equilibration(op.matvec, op.rmatvec, n, m, (),
                                                  **args, **extra)
        return np.asarray(jax.device_get(tau))

    free = run()
    tied = run(tie_columns=True)
    tagged = run(_cache_tag=("declaration", "A"))
    assert es._SMFE_CACHE_STATS["bypasses"] == before["bypasses"], "the cache did not engage"
    assert es._SMFE_CACHE_STATS["builds"] - before["builds"] == 3
    repeat = run()
    assert es._SMFE_CACHE_STATS["hits"] > before["hits"]
    assert repeat.tobytes() == free.tobytes()
    # The restriction is real, not only a key: the tie is exactly constant and
    # the free solution on this anisotropic operator is not.
    assert float(np.ptp(tied)) == 0.0
    assert float(np.ptp(free)) > 0.0
    assert tagged.tobytes() == free.tobytes()   # same program, different identity


def test_the_cache_identity_is_recorded_and_separates_declarations():
    prob, _data, n = anisotropic_family()
    identities = {}
    for name, kwargs in (("diagonal", dict(metric_columns="diagonal")),
                         ("tie", dict(metric_columns=[(0, n)])),
                         ("rows_diagonal_term", dict(metric_classes={0: ("diagonal", "term")}))):
        report = build(prob, **kwargs).meta["metric_construction"]
        identities[name] = report["restriction_cache_identity"]
        assert isinstance(hash(identities[name]), int)
    assert len(set(identities.values())) == len(identities)


# ------------------------------------------- the enforcement on applied steps
def test_the_enforcement_rejects_steps_outside_the_declared_class():
    whole = {"kind": "scalar", "spans": None, "metric_sharing_groups": 1}
    cm._enforce_column_restriction(jnp.ones(4), 4, dict(whole))
    with pytest.raises(ValueError, match="does not honour the declared column restriction"):
        cm._enforce_column_restriction(jnp.asarray([1., 1., 2., 1.]), 4, dict(whole))
    spans = {"kind": "spans", "spans": ((0, 2), (2, 4)), "metric_sharing_groups": 2}
    cm._enforce_column_restriction(jnp.asarray([1., 1., 2., 2.]), 4, dict(spans))
    with pytest.raises(ValueError, match="does not honour the declared column restriction"):
        cm._enforce_column_restriction(jnp.asarray([1., 1., 2., 3.]), 4, dict(spans))
    free = {"kind": "diagonal", "spans": None, "metric_sharing_groups": 4}
    record = dict(free)
    cm._enforce_column_restriction(jnp.asarray([1., 2., 3., 4.]), 4, record)
    assert "vacuous" in record["enforced_on_applied_steps"]
    # The full kind: T = c (I kron S) needs ONE carrier, uniform inside every
    # column block. Never vacuous, even for a scalar tau.
    full = {"kind": "full", "spans": None, "metric_sharing_groups": 1,
            "column_blocks": {"k": 2, "count": 2}, "shape": {"cond": 2.0}}
    record = dict(full)
    cm._enforce_column_restriction(jnp.asarray(0.5), 4, record)
    assert record["enforced_on_applied_steps"]["distinct_applied_carriers"] == 1
    with pytest.raises(ValueError, match="does not honour the declared column restriction"):
        cm._enforce_column_restriction(jnp.asarray([1., 2., 1., 1.]), 4, dict(full))
    with pytest.raises(ValueError, match="does not honour the declared column restriction"):
        cm._enforce_column_restriction(jnp.asarray([1., 1., 2., 2.]), 4, dict(full))


def test_the_block_structure_helper_matches_the_row_side_definition():
    assert mc.primal_block_structure(None, 7) == (1, 7, "none")
    assert mc.primal_block_structure(SumSquares(), 7) == (1, 7, "SumSquares")
    assert mc.primal_block_structure(L2(0.3), 7) == (7, 1, "L2")
    assert mc.primal_block_structure(GroupL2(0.3, group_size=3), 9) == (3, 3, "GroupL2")


# ------------------------------------------------------- the default is intact
def test_nothing_declared_is_still_byte_identical_to_the_shipped_program():
    prob, _data, _n = anisotropic_family()
    shipped = _attach(prob.compile(equilibrate="sgd", max_blocks=4, tol=1e-3))
    built = _attach(prob.compile(equilibrate="construct", max_blocks=4, tol=1e-3))
    assert built.meta["metric_construction"]["bit_identical_to_sgd"] is True
    assert built.meta["metric_construction"]["columns"]["declared"] is None
    assert steps(built)[0] == steps(shipped)[0]
    assert built.meta["scaled_norm"] == shipped.meta["scaled_norm"]
