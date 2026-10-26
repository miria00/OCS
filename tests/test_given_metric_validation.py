"""A user-supplied ("given", tau, sigma) metric is validated at compile time.

Before this check, an unhonourable metric (for example a non-uniform step on a
scalar-class GroupL2 dual block) reached the prox traced inside the compiled
loop, where ``_validate_step`` can only propagate NaN: the solve returned
``numerical_error`` with an infinite residual and no explanation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
import ocs.model.compile as module
from ocs.ir import ProblemFamily, Term
from ocs.prox.base import _validate_step


def _tv_like_family():
    # primal SumSquares (matrix class) + a GroupL2 dual block (scalar class)
    # + a SumSquares dual block (matrix class), like an isotropic-TV model.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 4)).astype(np.float32)
    G = rng.standard_normal((4, 4)).astype(np.float32)
    return ProblemFamily(n=4, terms=[
        Term(ocs.Identity(4), ocs.SumSquares()),
        Term(ocs.Dense(G), ocs.GroupL2(.3, 2)),
        Term(ocs.Dense(A), ocs.SumSquares(), lambda data: -data["b"]),
    ]), {"b": jnp.asarray(rng.standard_normal(6), jnp.float32)}


def test_nonuniform_step_on_scalar_class_dual_block_raises_at_compile():
    family, _ = _tv_like_family()
    sigma = jnp.concatenate([jnp.asarray([.2, .2, .2, .4]), jnp.full(6, .3)])
    with pytest.raises(ValueError) as error:
        family.compile(equilibrate=("given", jnp.full(4, .1), sigma, .9))
    message = str(error.value)
    assert "dual block 0" in message
    assert "GroupL2" in message
    assert "scaling class 'scalar'" in message
    assert "rows 0:4" in message
    assert "2 distinct values" in message


def test_nonuniform_step_on_matrix_class_dual_block_is_accepted():
    family, data = _tv_like_family()
    sigma = jnp.concatenate([jnp.full(4, .2), jnp.linspace(.1, .6, 6)])
    compiled = family.compile(equilibrate=("given", jnp.full(4, .1), sigma, .9),
                              block=20, max_blocks=50)
    result = compiled.solve_instance(data)
    assert np.isfinite(result.residual)
    assert result.status != "numerical_error"


@pytest.mark.parametrize("name", ["tau", "sigma"])
@pytest.mark.parametrize("bad", [0., -1., np.inf, np.nan])
def test_nonfinite_or_nonpositive_entries_raise_and_name_the_array(name, bad):
    family, _ = _tv_like_family()
    tau, sigma = np.full(4, .1, np.float32), np.full(10, .2, np.float32)
    (tau if name == "tau" else sigma)[3] = bad
    with pytest.raises(ValueError, match=rf"{name}\[3\]"):
        family.compile(equilibrate=("given", jnp.asarray(tau), jnp.asarray(sigma)))


def test_sigma_with_an_overflowing_reciprocal_raises():
    family, _ = _tv_like_family()
    sigma = np.full(10, .2, np.float32)
    sigma[5] = np.float32(1e-40)            # positive, but 1/sigma is inf in float32
    with pytest.raises(ValueError, match="finite reciprocal"):
        family.compile(equilibrate=("given", jnp.full(4, .1), jnp.asarray(sigma)))


@pytest.mark.parametrize("which,shape", [("tau", (3,)), ("sigma", (9,)), ("tau", (4, 1))])
def test_wrong_metric_shape_raises(which, shape):
    family, _ = _tv_like_family()
    tau, sigma = jnp.full(4, .1), jnp.full(10, .2)
    if which == "tau":
        tau = jnp.full(shape, .1)
    else:
        sigma = jnp.full(shape, .2)
    with pytest.raises(ValueError, match=f"{which} has shape"):
        family.compile(equilibrate=("given", tau, sigma))


def test_given_tuple_without_sigma_raises_a_value_error():
    family, _ = _tv_like_family()
    with pytest.raises(ValueError, match="needs a tau and a sigma"):
        family.compile(equilibrate=("given", jnp.full(4, .1)))


def test_valid_given_metrics_reach_the_engine_unchanged(monkeypatch):
    family, data = _tv_like_family()
    captured = []
    original = module.make_solver

    def capture(matvec, rmatvec, tau, sigma, *args, **kwargs):
        captured.append((tau, sigma))
        return original(matvec, rmatvec, tau, sigma, *args, **kwargs)

    monkeypatch.setattr(module, "make_solver", capture)
    tau = jnp.asarray([.1, .2, .3, .4])
    sigma = jnp.concatenate([jnp.full(4, .25), jnp.linspace(.1, .6, 6)])
    compiled = family.compile(equilibrate=("given", tau, sigma, .9), block=20, max_blocks=50)
    assert captured
    for actual_tau, actual_sigma in captured:
        np.testing.assert_array_equal(np.asarray(actual_tau), np.asarray(tau))
        np.testing.assert_array_equal(np.asarray(actual_sigma), np.asarray(sigma))
    assert compiled.meta["scaled_norm"] == pytest.approx(.9)
    assert np.isfinite(compiled.solve_instance(data).residual)


def test_scalar_steps_and_scalar_primal_blocks_are_still_accepted():
    # Scalar tau/sigma, and a non-uniform tau on a scalar-class PRIMAL block
    # (which the compiler reduces to one value), remain valid inputs.
    x = ocs.Variable(9)
    cost = jnp.eye(3).reshape(-1)
    diagonal = ocs.Dense(np.eye(9)[[0, 4, 8]])
    family = ocs.minimize([ocs.psd(x, 3), ocs.equality(diagonal @ x - jnp.ones(3))], cost=cost)
    tau = jnp.where(jnp.eye(3).reshape(-1) > 0, .1, 5.)
    compiled = family.compile(equilibrate=("given", tau, jnp.full(3, .1), .1),
                              block=20, max_blocks=5)
    assert compiled.meta["primal_scalar_metric"]
    family.compile(equilibrate=("given", jnp.float32(.1), jnp.float32(.1)), block=20, max_blocks=5)


def test_traced_prox_backstop_still_propagates_nan():
    # The eager check does not replace the in-loop safeguard.
    block = ocs.GroupL2(.3, 2)
    evaluate = jax.jit(lambda v, t: block.prox(v, t))
    out = evaluate(jnp.ones(4), jnp.asarray([1., 1., 1., 2.]))
    assert np.isnan(np.asarray(out)).all()
    with pytest.raises(ValueError):
        _validate_step(jnp.ones(4), jnp.asarray([1., 1., 1., 2.]), "scalar")
