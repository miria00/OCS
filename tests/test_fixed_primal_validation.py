"""Independent formulas and negative controls for fixed iteration validation."""
import importlib
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.ir import Term
from ocs.solver.fixed_prox import fixed_nonnegative_quadratic_prox


@contextmanager
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("vector_step", [False, True])
@pytest.mark.parametrize("vector_coefficients", [False, True])
def test_formula_and_generic_match(dtype, vector_step, vector_coefficients):
    with x64():
        n = 9
        t = jnp.linspace(.1, .9, n, dtype=jnp.float32) if vector_step else jnp.float32(.3)
        d = np.linspace(-.2, .3, n, dtype=np.float32) if vector_coefficients else .03
        e = np.linspace(0., .3, n, dtype=np.float32) if vector_coefficients else .1
        block = ocs.NonNeg().with_linear_quadratic(d=d, e=e)
        candidate = fixed_nonnegative_quadratic_prox(block, t, n)
        assert candidate is not None
        v = jnp.asarray(np.linspace(-1., 2., n), dtype=dtype)
        generic = jax.jit(lambda z: block.prox(z, jnp.broadcast_to(t, (n,))))(v)
        actual = jax.jit(candidate)(v)
        expected = np.maximum((np.asarray(v, dtype=np.float64)-np.asarray(t)*np.asarray(block.d))
                              /(1+np.asarray(t, dtype=np.float64)*np.asarray(block.e)), 0)
        np.testing.assert_allclose(actual, generic, rtol=3e-7, atol=1e-8)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-8)


@pytest.mark.parametrize("step,e", [
    (0., .1), (-1., .1), (np.nan, .1), (np.inf, .1),
    (np.finfo(np.float32).tiny, .1), (np.finfo(np.float32).max, 2.),
    (1e20, 1e20), (1e-20, 1e38), ([.2, 0., .2, .2], .1),
    ([.2, np.nan, .2, .2], .1),
])
def test_invalid_or_extreme_transformed_metrics_fall_back(step, e):
    block = ocs.NonNeg().with_linear_quadratic(e=e)
    assert fixed_nonnegative_quadratic_prox(block, jnp.asarray(step, jnp.float32), 4) is None


@pytest.mark.parametrize("kind", ["subclass", "base_subclass", "instance", "base_instance",
                                 "tag", "other_base", "other_wrapper", "none"])
def test_only_unmodified_exact_builtins_are_specialized(kind):
    from ocs.prox.base import _LinQuad
    block = ocs.NonNeg().with_linear_quadratic(e=.1)
    if kind == "subclass":
        class Custom(_LinQuad):
            pass
        block = Custom(ocs.NonNeg(), 0., .1)
    elif kind == "base_subclass":
        class Custom(ocs.NonNeg):
            pass
        block = Custom().with_linear_quadratic(e=.1)
    elif kind == "instance":
        block.prox = lambda v, t: v
    elif kind == "base_instance":
        block.base.prox = lambda v, t: v
    elif kind == "tag":
        block.user_tag = True
    elif kind == "other_base":
        block = ocs.Box().with_linear_quadratic(e=.1)
    elif kind == "other_wrapper":
        block = ocs.NonNeg().scaled(2.)
    else:
        block = None
    assert fixed_nonnegative_quadratic_prox(block, jnp.float32(.3), 4) is None


def test_runtime_metric_is_not_assumed_fixed():
    block = ocs.NonNeg().with_linear_quadratic(e=.1)
    def factory(t):
        assert fixed_nonnegative_quadratic_prox(block, t, 4) is None
        return block.prox(jnp.arange(4.), t)
    assert np.isnan(jax.jit(factory)(jnp.asarray(0.))).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_points_and_signed_zero_keep_generic_behavior(dtype):
    with x64():
        v = jnp.asarray([np.nan, np.inf, -np.inf, -0., 0., -1., 1.], dtype=dtype)
        block = ocs.NonNeg().with_linear_quadratic(e=.1)
        t = jnp.float32(.2)
        candidate = fixed_nonnegative_quadratic_prox(block, t, 7)
        actual = np.asarray(jax.jit(candidate)(v))
        reference = np.asarray(jax.jit(lambda z: block.prox(z, jnp.broadcast_to(t, (7,))))(v))
        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(reference))


def test_iteration_jaxpr_has_no_metric_reductions():
    block = ocs.NonNeg().with_linear_quadratic(e=.1)
    t = jnp.ones(8192, jnp.float32)*.2
    candidate = fixed_nonnegative_quadratic_prox(block, t, 8192)
    graph = jax.make_jaxpr(candidate)(jnp.ones(8192)).jaxpr
    assert not any(e.primitive.name == "reduce_and" for e in graph.eqns)


@pytest.mark.parametrize("layout", ["dense", "compact"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_trajectory_status_certificate_and_batches_match(monkeypatch, layout, dtype):
    module = importlib.import_module("ocs.model.compile")
    with x64():
        x = ocs.Variable(4)
        problem = ocs.minimize([
            Term(ocs.Identity(4), ocs.NonNeg().with_linear_quadratic(e=.1)),
            ocs.equality(ocs.Dense(np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]], np.float32))@x
                         - ocs.Parameter("b")),
        ], cost=ocs.Parameter("c"))
        kwargs = dict(equilibrate=("given", jnp.float32(.2), jnp.float32(.2)),
                      pdhg_variant="standard", data_layout=layout, block=10, max_blocks=100,
                      tol=1e-5)
        new = problem.compile(**kwargs)
        assert new.meta["fixed_primal_validation"]
        monkeypatch.setattr(module, "fixed_nonnegative_quadratic_prox", lambda *a: None)
        old = problem.compile(**kwargs)
        data = {"b": jnp.asarray([1., 2.], jnp.float32),
                "c": jnp.asarray([.1, .2, -.2, .3], jnp.float32)}
        start = (jnp.zeros(4, dtype=dtype), jnp.zeros(2, dtype=dtype))
        left, right = old.solve_instance(data, start), new.solve_instance(data, start)
        assert (left.status, left.iters) == (right.status, right.iters)
        np.testing.assert_allclose(left.x, right.x, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(left.y, right.y, rtol=2e-6, atol=2e-7)
        for k, value in old.certificate(left.x, left.y, data).items():
            np.testing.assert_allclose(value, new.certificate(right.x, right.y, data)[k],
                                       rtol=2e-5, atol=2e-7, equal_nan=True)
        # Independent analytic constrained optimum: all mass goes to the
        # cheapest coordinate except the first pair's boundary tie.
        np.testing.assert_allclose(right.x, [1., 0., 2., 0.], atol=1e-4)
        batch = {k: jnp.stack([v, v]) for k, v in data.items()}
        for a, b in zip(old.solve_many(batch), new.solve_many(batch)):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7, equal_nan=True)


def test_halpern_keeps_generic_validation():
    x = ocs.Variable(4)
    family = ocs.minimize([
        Term(ocs.Identity(4), ocs.NonNeg().with_linear_quadratic(e=.1)),
        ocs.least_squares(ocs.Identity(4)@x - ocs.Parameter("b")),
    ]).compile(max_blocks=1)
    assert family.meta["fixed_primal_validation"] is False
