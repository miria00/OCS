"""Compact SMFE state preserves projection order and weighted log averaging."""
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.solver.equilibrate_sgd import _scalar_span_layout, smfe_equilibration


def test_imaging_layout_is_metadata_only_and_keeps_single_scalar_per_span():
    n = 1_000_000_000
    fft_rows = 800_000_000
    layout, size = _scalar_span_layout(fft_rows+3*n, ((fft_rows, fft_rows+3*n),))
    assert size == fft_rows+1
    assert layout == ((0, fft_rows, 0, False), (fft_rows, fft_rows+3*n, fft_rows, True))
    # Both the iterate and its weighted accumulator avoid replicated TV rows.
    assert 2*4*((fft_rows+3*n)-size) == 24*n-8


@pytest.mark.parametrize("spans", [(), ((0, 1),), ((1, 4), (3, 5)),
                                   ((3, 5), (0, 2)), ((-1, 3),), ((1, 6),)])
def test_uncompressed_or_unusual_spans_keep_legacy_layout(spans):
    assert _scalar_span_layout(5, spans) == (None, 5)


def _diagonal_reference(weights, spans, iters, schedule, gamma, bound):
    # Every Rademacher squared row/column product is exact for a diagonal
    # operator. This gives a deterministic oracle independent of RNG/probes.
    u, v = np.zeros_like(weights), np.zeros_like(weights)
    ubar, vbar, total_weight = u.copy(), v.copy(), 0.
    for t in range(iters):
        energies = np.exp(2*(u+v))*weights**2
        if schedule == "sgd":
            eta = 2/(gamma*(t+2))
            w_u = u-eta*(energies-1+gamma*u)          # tentative, unclipped
            u = np.clip(w_u, -bound, bound)
            v = np.clip(v-eta*(energies-1+gamma*v), -bound, bound)
        else:
            damping = max(2**(-np.floor(np.log2(t+1))), 1/16)
            change = -.5*damping*np.log(energies+1e-30)
            w_u = u+change                            # tentative, unclipped
            u = np.clip(w_u, -bound, bound)
            v = np.clip(v+change, -bound, bound)
        # Euclidean projection onto {a*1 : |a| <= bound}: mean the UNCLIPPED
        # tentative point, then clip the scalar. mean(clip(.)) is a different
        # map, and was what this reference and the solver used to compute.
        for a, b in spans:
            u[a:b] = np.clip(np.mean(w_u[a:b]), -bound, bound)
        weight = t+1
        ubar += weight*u
        vbar += weight*v
        total_weight += weight
    return (u, v) if schedule == "bradley" else (ubar/total_weight, vbar/total_weight)


@pytest.mark.parametrize("schedule", ["sgd", "bradley"])
@pytest.mark.parametrize("spans", [((0, 5),), ((1, 4),), ((0, 2), (2, 5))])
@pytest.mark.parametrize("iters", [1, 9])
def test_block_mean_projection_and_iterate_averaging(schedule, spans, iters):
    weights = np.array([.1, .3, 1., 3., 10.], np.float64)
    device_weights = jnp.asarray(weights, jnp.float32)
    operator = lambda x: device_weights*x
    gamma, bound, safety = .7, .2, .9
    tau, sigma, norm = smfe_equilibration(operator, operator, 5, 5, spans,
        iters=iters, probes=2, gamma=gamma, M=bound, safety=safety, seed=4,
        schedule=schedule)
    u, v = _diagonal_reference(weights, spans, iters, schedule, gamma, bound)
    factor = safety/(norm+1e-30)
    np.testing.assert_allclose(np.asarray(tau)/factor, np.exp(2*v), rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(np.asarray(sigma)/factor, np.exp(2*u), rtol=3e-6, atol=3e-7)
    for a, b in spans:
        np.testing.assert_array_equal(sigma[a:b], np.full(b-a, np.asarray(sigma)[a]))
    if iters == 1 and spans == ((0, 5),):
        # The case that separates the two maps. Averaging the already-clipped
        # rows lets the positive and negative ones cancel, giving exp(0) = 1 on
        # either schedule. The projection averages the unclipped tentative
        # point instead, which on this operator is far below -bound under the
        # sgd step and slightly positive under the bradley step. Either way it
        # is not 1, which is what the previous implementation returned.
        got = np.asarray(sigma)/factor
        assert not np.allclose(got, np.ones(5), rtol=1e-3), (
            "block value cancelled to 1: this is mean(clip(.)), not the projection")
        if schedule == "sgd":
            np.testing.assert_allclose(got, np.full(5, np.exp(-2*bound)), rtol=1e-6)
