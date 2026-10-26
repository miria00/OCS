"""A scalar-class primal block gets the tied SMFE construction, not min(tau).

Before this change the compiler built a free per-column SMFE metric and then
reduced it with ``jnp.min(tau)``. Columns the constraints never touch receive
a large step from the construction but contribute nothing to the spectral
bound, so the minimum is dragged down to the most constrained column's step:
safe, but a primal weight off by orders of magnitude on conic standard form.
The construction now solves its problem with the column tie in place and
measures the norm of that tied metric.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
import ocs.model.compile as module
from ocs.solver.equilibrate import _power_norm
from ocs.solver.equilibrate_sgd import smfe_equilibration


def _selector_operator(k=6, seed=0):
    # a maxcut-like diagonal extraction plus a little dense coupling: most
    # columns carry zero energy, the free SMFE tau spreads widely
    rng = np.random.default_rng(seed)
    A = np.zeros((k + 2, k * k), np.float32)
    for i in range(k):
        A[i, i * k + i] = 1.
    A[k:, :k] = rng.standard_normal((2, k))
    return ocs.Dense(A), A


def _scaled_norm(A, tau, sigma, iters=400):
    M = np.sqrt(np.asarray(sigma, np.float64))[:, None] * A * np.sqrt(np.asarray(tau, np.float64))[None, :]
    return float(np.linalg.norm(M, 2))


def test_tied_construction_returns_one_primal_step_at_the_measured_bound():
    op, A = _selector_operator()
    n, m = A.shape[1], A.shape[0]
    free_tau, free_sigma, _ = smfe_equilibration(op.matvec, op.rmatvec, n, m)
    tau, sigma, norm = smfe_equilibration(op.matvec, op.rmatvec, n, m, tie_columns=True)
    tau, sigma = np.asarray(tau), np.asarray(sigma)
    assert tau.shape == (n,) and np.all(tau == tau[0])
    assert np.all(np.isfinite(sigma) & (sigma > 0))
    assert np.isfinite(norm) and norm > 0
    # the shipped normalization is measured on THIS metric: its own estimate
    # sits at the 0.9 safety factor and the exact norm respects the bound 1
    st, sg = jnp.sqrt(jnp.asarray(tau)), jnp.sqrt(jnp.asarray(sigma))
    estimate = float(_power_norm(lambda x: sg * op.matvec(st * x),
                                 lambda y: st * op.rmatvec(sg * y), n, iters=15, seed=7))
    assert estimate == pytest.approx(0.9, rel=1e-4)
    assert _scaled_norm(A, tau, sigma) <= 1.0
    # and it is not the collapse: the tied step exceeds the free minimum
    assert tau[0] > float(np.min(np.asarray(free_tau)))


def test_default_construction_is_unchanged_by_the_new_option():
    op, A = _selector_operator(seed=1)
    n, m = A.shape[1], A.shape[0]
    a = smfe_equilibration(op.matvec, op.rmatvec, n, m, [(0, 3)])
    b = smfe_equilibration(op.matvec, op.rmatvec, n, m, [(0, 3)], tie_columns=False)
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert a[2] == b[2]


@pytest.mark.parametrize("schedule", ["sgd", "bradley"])
def test_tie_is_honoured_with_row_spans_and_both_schedules(schedule):
    op, A = _selector_operator(seed=2)
    n, m = A.shape[1], A.shape[0]
    tau, sigma, _ = smfe_equilibration(op.matvec, op.rmatvec, n, m, [(0, 4)],
                                       tie_columns=True, schedule=schedule)
    tau, sigma = np.asarray(tau), np.asarray(sigma)
    assert np.all(tau == tau[0])
    assert np.all(sigma[:4] == sigma[0])
    assert _scaled_norm(A, tau, sigma) <= 1.0


def _capture(monkeypatch):
    captured = []
    original = module.make_solver

    def capture(matvec, rmatvec, tau, sigma, *args, **kwargs):
        captured.append((np.asarray(tau), np.asarray(sigma)))
        return original(matvec, rmatvec, tau, sigma, *args, **kwargs)
    monkeypatch.setattr(module, "make_solver", capture)
    return captured


def _psd_family(k=6):
    x = ocs.Variable(k * k)
    op, _A = _selector_operator(k)
    rng = np.random.default_rng(3)
    C = rng.standard_normal((k, k)).astype(np.float32)
    cost = jnp.asarray((C + C.T).reshape(-1))
    b = jnp.asarray(np.r_[np.ones(k), np.zeros(2)].astype(np.float32))
    return ocs.minimize([ocs.psd(x, k), ocs.equality(op @ x - b)], cost=cost)


def test_compile_uses_the_tied_construction_for_a_scalar_primal_block(monkeypatch):
    family = _psd_family()
    captured = _capture(monkeypatch)
    compiled = family.compile(block=20, max_blocks=5)
    assert compiled.meta["primal_scalar_metric"]
    assert compiled.meta["primal_metric_tie"] == "construction"
    tau, sigma = captured[0]
    assert np.all(tau == tau[0]) and np.all(np.isfinite(sigma) & (sigma > 0))
    _op, A = _selector_operator()
    assert _scaled_norm(A, tau, sigma) <= 1.0


def test_given_metric_on_a_scalar_primal_block_keeps_the_minimum(monkeypatch):
    family = _psd_family()
    captured = _capture(monkeypatch)
    tau = jnp.linspace(.1, .5, 36)
    compiled = family.compile(equilibrate=("given", tau, jnp.full(8, .1), .9),
                              block=20, max_blocks=5)
    assert compiled.meta["primal_metric_tie"] == "minimum"
    np.testing.assert_array_equal(captured[0][0], np.full(36, np.float32(.1)))


def test_non_scalar_primal_blocks_do_not_request_a_tie(monkeypatch):
    calls = []
    original = module.smfe_equilibration

    def spy(*args, **kwargs):
        calls.append(kwargs.get("tie_columns"))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "smfe_equilibration", spy)
    x = ocs.Variable(4)
    A = ocs.Dense(np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32))
    compiled = ocs.minimize([ocs.nonneg(x), ocs.least_squares(A @ x - ocs.Parameter("b"))]).compile(
        block=20, max_blocks=5)
    assert calls == [False]
    assert compiled.meta["primal_metric_tie"] is None
    assert not compiled.meta["primal_scalar_metric"]
