"""Executable reuse must preserve changed operator data and numerical results."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.operators import Dense, Diagonal
from ocs.solver import equilibrate_sgd as eq


@pytest.fixture(autouse=True)
def clear_cache():
    eq._SMFE_RUN_CACHE.clear()
    eq._SMFE_CACHE_STATS.update(builds=0, hits=0, bypasses=0)
    yield
    eq._SMFE_RUN_CACHE.clear()


def run(op, cached=True, **kwargs):
    # Fresh stacked closures mimic each public family.compile call.
    mv = lambda x: jnp.concatenate([op.matvec(x)])
    rmv = lambda y: op.rmatvec(y)
    return eq.smfe_equilibration(mv, rmv, op.shape[1], op.shape[0],
        iters=8, probes=2, _cache_operators=(op,) if cached else None, **kwargs)


def equal(a, b):
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    if len(a) == 4:
        assert a[3] == b[3]


@pytest.mark.parametrize("schedule", ["sgd", "bradley"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_fresh_closures_reuse_program_and_match_uncached(schedule, diagnostics):
    op = Dense(jnp.array([[1., 2.], [.3, 4.], [2., -.1]], jnp.float32))
    a = run(op, schedule=schedule, diagnostics=diagnostics)
    b = run(op, schedule=schedule, diagnostics=diagnostics)
    c = run(op, cached=False, schedule=schedule, diagnostics=diagnostics)
    equal(a, b)
    equal(a, c)
    assert eq._SMFE_CACHE_STATS == {"builds": 1, "hits": 1, "bypasses": 1}


def test_replacing_array_on_same_operator_invalidates_program():
    op = Dense(jnp.array([[1., 2.], [3., 4.]], jnp.float32))
    original = run(op)
    op.A = jnp.array([[20., .1], [.1, .3]], jnp.float32)
    changed = run(op)
    equal(changed, run(op, cached=False))
    assert eq._SMFE_CACHE_STATS["builds"] == 2
    assert not np.allclose(original[0], changed[0])


def test_mutable_numpy_array_never_reuses_program():
    op = Dense(jnp.eye(2))
    op.A = np.array([[1., 2.], [3., 4.]], np.float32)
    first = run(op)
    op.A *= 10
    changed = run(op)
    equal(changed, run(op, cached=False))
    assert eq._SMFE_CACHE_STATS["builds"] == 0
    assert not np.allclose(first[0], changed[0])


def test_custom_callback_closure_is_not_cached():
    from ocs.operators.base import LinearOperator
    state = {"scale": 1.}
    op = LinearOperator.from_forward(lambda x: state["scale"] * x, 2, 2)
    run(op)
    state["scale"] = 3.
    run(op)
    assert eq._SMFE_CACHE_STATS["builds"] == 0
    assert eq._SMFE_CACHE_STATS["bypasses"] == 2


def test_changed_seed_and_scalar_spans_do_not_share_program():
    op = Dense(jnp.array([[1., 2.], [3., .1]], jnp.float32))
    for kwargs in ({"seed": 1}, {"seed": 2}, {"seed": 1, "scalar_spans": ((0, 2),)}):
        equal(run(op, **kwargs), run(op, cached=False, **kwargs))
    assert eq._SMFE_CACHE_STATS["builds"] == 3


def test_cache_is_bounded():
    for seed in range(eq._SMFE_CACHE_LIMIT + 2):
        run(Diagonal(jnp.array([1., 2.])), seed=seed)
    assert len(eq._SMFE_RUN_CACHE) == eq._SMFE_CACHE_LIMIT


def test_large_captured_arrays_bypass_retention(monkeypatch):
    monkeypatch.setattr(eq, "_SMFE_CACHE_ARRAY_BYTES", 1)
    op = Dense(jnp.eye(2))
    equal(run(op), run(op, cached=False))
    assert not eq._SMFE_RUN_CACHE
    assert eq._SMFE_CACHE_STATS["bypasses"] == 2


def test_mutated_operator_cannot_retain_new_large_array(monkeypatch):
    monkeypatch.setattr(eq, "_SMFE_CACHE_ARRAY_BYTES", 32)
    op = Dense(jnp.eye(2))
    run(op)
    assert len(eq._SMFE_RUN_CACHE) == 1
    op.A = jnp.eye(3)
    op.shape = (3, 3)
    run(op)
    assert not eq._SMFE_RUN_CACHE
    assert eq._SMFE_CACHE_STATS["bypasses"] == 1


def test_monkeypatched_builtin_method_bypasses_cache(monkeypatch):
    state = {"scale": 1.}
    monkeypatch.setattr(Dense, "matvec", lambda self, x: state["scale"] * (self.A @ x))
    monkeypatch.setattr(Dense, "rmatvec", lambda self, y: state["scale"] * (self.A.T @ y))
    op = Dense(jnp.eye(2))
    a = run(op)
    state["scale"] = 10.
    b = run(op)
    equal(b, run(op, cached=False))
    assert not np.allclose(a[0], b[0])
    assert eq._SMFE_CACHE_STATS["builds"] == 0


def test_public_compile_reuses_setup_and_invalidates_replaced_data():
    import ocs
    x = ocs.Variable(2)
    op = Dense(jnp.array([[1., 2.], [3., 4.]], jnp.float32))
    family = ocs.minimize([ocs.least_squares(op @ x)])
    options = dict(smfe_iters=8, max_blocks=2, block=10)
    first = family.compile(**options)
    second = family.compile(**options)
    assert eq._SMFE_CACHE_STATS["hits"] == 1
    op.A = jnp.array([[20., .1], [.1, .3]], jnp.float32)
    changed = family.compile(**options)
    assert eq._SMFE_CACHE_STATS["builds"] == 2
    assert first.meta["scaled_norm"] == second.meta["scaled_norm"]
    assert first.meta["scaled_norm"] != changed.meta["scaled_norm"]
