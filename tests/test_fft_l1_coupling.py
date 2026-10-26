"""The l1 coupling through a subsampled Fourier adjoint (0.3.6 fold-in).

g = lam ||x||_1 with K = FFTMask makes the dual domain ||K^T y||_inf <= lam.
0.3.4 decided it on the float32 adjoint with no rounding allowance, and a
Gaussian right-hand side at N = 1024 returned `solved` with a witness 4.4e-6
lam outside the domain in float64. These tests check, against NumPy float64 on
the REPRESENTED float32 witness:

* the coupling decision (recover=False) is the float64 decision, including on
  witnesses the float32 test accepts and float64 rejects (old false passes);
* the decision, the float64 ratio and the float32 allowance agree under
  eager (disable_jit), jit, vmap and jit(vmap), and are rounded upward;
* every certifying dual the evaluator returns (recover=True) is inside the
  float64 domain, scaled fallback included, in every transform;
* solve_instance / solve_many_results never return `solved` with an
  out-of-domain witness on such an instance;
* other families do not trace the guard; global x64 stays off.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs

N = 1024
LAM = float(np.float32(1e-3))
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def _idx(n=N, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n, n // 4, replace=False))


def _adj64(idx, n, y):
    y = np.asarray(y, np.float64)
    m = len(idx)
    spec = np.zeros(n, np.complex128)
    spec[idx] = y[:m] + 1j * y[m:]
    return np.fft.ifft(spec, norm="ortho").real


def _fwd64(idx, x):
    c = np.fft.fft(np.asarray(x, np.float64), norm="ortho")[idx]
    return np.r_[c.real, c.imag]


def _gamma(n, eps):
    return (16 * max(1, math.ceil(math.log2(max(2, n)))) + 64) * eps


def _numpy_assess(idx, n, y, lam=LAM):
    """The same quantities in NumPy float64 on the represented witness."""
    y64 = np.asarray(y, np.float64)
    adj = _adj64(idx, n, y64)
    mass = np.abs(y64).sum() / math.sqrt(n)
    peak = np.abs(adj).max()
    env64 = _gamma(n, EPS64) * mass + 3 * EPS64 * peak
    env32 = _gamma(n, EPS32) * mass + 3 * EPS32 * peak
    return peak / lam, (peak + env64) / lam, env32 / lam


def _family(idx, n, lam=LAM):
    x = ocs.Variable(n)
    F = ocs.FFTMask(idx, n)
    fam = ocs.minimize([ocs.l1(x, lam), ocs.least_squares(F @ x - ocs.Parameter("b"))])
    return fam


def _ista(idx, n, b, lam=LAM, iters=3000):
    """Near-optimal lasso primal in float64 (F has orthonormal rows: step 1)."""
    x = np.zeros(n)
    for _ in range(iters):
        g = _adj64(idx, n, _fwd64(idx, x) - b)
        v = x - g
        x = np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
    return x


def _float32_old_test(idx, n, y, lam=LAM):
    """0.3.4's decision: max|-K^T y| <= lam on the jitted float32 adjoint."""
    F = ocs.FFTMask(idx, n)
    kty = jax.jit(F.rmatvec)(jnp.asarray(y, jnp.float32))
    return bool(jnp.all(jnp.isfinite(kty)) & (jnp.max(jnp.abs(-kty)) <= jnp.float32(lam)))


@pytest.fixture(scope="module")
def instance():
    """A Gaussian right-hand side (lam/lam_max ~ 6e-4) and a witness that the
    float32 test accepts while float64 places it outside the domain, built the
    way the ray search builds its top: a = (lam / peak32) (1 - 64 eps32) along
    a residual direction of a lasso iterate (heavy cancellation in K^T y)."""
    idx = _idx(N, 0)
    rm = jax.jit(ocs.FFTMask(idx, N).rmatvec)
    for seed in range(8):
        b = np.random.default_rng(1000 + seed).standard_normal(2 * len(idx)).astype(np.float32)
        b64 = b.astype(np.float64)
        x = np.zeros(N)
        for t in range(600):
            v = x - _adj64(idx, N, _fwd64(idx, x) - b64)
            x = np.sign(v) * np.maximum(np.abs(v) - LAM, 0.0)
            if t < 30:
                continue
            ydir = (_fwd64(idx, x) - b64).astype(np.float32)
            peak32 = np.float32(jnp.max(jnp.abs(rm(jnp.asarray(ydir)))))
            a = np.float32(np.float32(LAM) / peak32) * np.float32(1.0 - 64.0 * EPS32)
            y = (a * ydir).astype(np.float32)
            if _float32_old_test(idx, N, y) and _numpy_assess(idx, N, y)[0] > 1.0:
                s0 = LAM / np.abs(_adj64(idx, N, ydir)).max()
                return dict(idx=idx, b=b, x=x.astype(np.float32), y_bad=y, ydir=ydir,
                            s0=s0, seed=seed, iterate=t)
    pytest.fail("no float32-accepted, float64-rejected witness found")


def _pack(instance):
    fam = _family(instance["idx"], N)
    comp = fam.compile(tol=1e-3, max_blocks=50)
    data = {"b": jnp.asarray(instance["b"])}
    return comp, data


def test_guard_is_traced_and_labelled(instance):
    comp, data = _pack(instance)
    rep = comp.objective_bound(instance["x"], instance["y_bad"], data, recover=False)
    assert rep["coupling_arithmetic_guard"] == "fftmask_l1_float64_adjoint_envelope"
    assert rep["assessment_uses_float64"] is True
    assert rep["arithmetic"] == "mixed"
    assert not jax.config.jax_enable_x64


def test_old_false_pass_is_refused(instance):
    comp, data = _pack(instance)
    y = instance["y_bad"]
    assert _float32_old_test(instance["idx"], N, y)          # 0.3.4 accepted it
    plain, ratio64, allow = _numpy_assess(instance["idx"], N, y)
    assert plain > 1.0
    rep = comp.objective_bound(instance["x"], y, data, recover=False)
    assert rep["objective_lower_bound_available"] is False
    assert "coupling" in rep["supplied_dual_status"]
    assert rep["fft_l1_supplied_ratio_float64"] >= ratio64
    assert rep["fft_l1_supplied_ratio_float64"] <= ratio64 * (1 + 1e-6)
    assert rep["fft_l1_supplied_allowance_float32"] >= allow
    assert plain - 1.0 <= allow                               # rounding-level, inside the band


def _witnesses(instance):
    """Represented witnesses around the boundary, with their NumPy truth."""
    ys, truth = [], []
    for k in (-300, -40, -8, 8, 40, 300):
        y = (instance["s0"] * (1.0 + k * 1e-6) * instance["ydir"]).astype(np.float32)
        plain, ratio64, allow = _numpy_assess(instance["idx"], N, y)
        ys.append(y)
        truth.append((ratio64 <= 1.0, ratio64, allow))
    ys.append(instance["y_bad"])
    truth.append((False,) + _numpy_assess(instance["idx"], N, instance["y_bad"])[1:])
    return ys, truth


def _decisions_scalar(comp, x, ys, data):
    return [comp.objective_bound(x, y, data, recover=False) for y in ys]


def test_decision_and_allowance_agree_under_eager_jit_vmap(instance):
    comp, data = _pack(instance)
    x = jnp.asarray(instance["x"])
    ys, truth = _witnesses(instance)
    with jax.disable_jit():
        eager = _decisions_scalar(comp, x, ys, data)
    jitted = _decisions_scalar(comp, x, ys, data)
    count = len(ys)
    xb = jnp.stack([x] * count)
    yb = jnp.stack([jnp.asarray(y) for y in ys])
    dense = jax.vmap(comp._resolve_dense)({"b": jnp.stack([data["b"]] * count)})
    vm, _dual = comp._objective_bound_many(xb, yb, dense, 32, return_dual=True)
    jvm, _jdual = jax.jit(lambda a, b_, d: comp._objective_bound_many(a, b_, d, 32, return_dual=True))(xb, yb, dense)
    for i, (inside, ratio64, allow) in enumerate(truth):
        # The decision is the float64 decision AND the unchanged float32 test.
        # Batched transforms may round the float32 FFT differently, so only
        # the float64 half is required to agree across all four transforms.
        assert eager[i]["supplied_dual_available"] == jitted[i]["supplied_dual_available"], i
        if jitted[i]["supplied_dual_available"]:
            assert inside, i
        if inside and _float32_old_test(instance["idx"], N, ys[i]):
            assert jitted[i]["supplied_dual_available"] is True, i
        for name, rep in (("eager", eager[i]), ("jit", jitted[i])):
            assert rep["fft_l1_supplied_ratio_float64"] >= ratio64, (name, i)
            assert rep["fft_l1_supplied_ratio_float64"] <= ratio64 * (1 + 1e-6), (name, i)
            assert rep["fft_l1_supplied_allowance_float32"] >= allow, (name, i)
        for name, rep in (("vmap", vm), ("jit_vmap", jvm)):
            got = bool(rep["supplied_dual_available"][i])
            if got:
                assert inside, (name, i)
            assert float(rep["fft_l1_supplied_ratio_float64"][i]) >= ratio64, (name, i)
            assert float(rep["fft_l1_supplied_ratio_float64"][i]) <= ratio64 * (1 + 1e-6), (name, i)
            assert float(rep["fft_l1_supplied_allowance_float32"][i]) >= allow, (name, i)
        if not inside:
            assert eager[i]["supplied_dual_available"] is False
    assert not jax.config.jax_enable_x64


def _check_dual(instance, dual, lower):
    dual = np.asarray(dual)
    assert dual.dtype == np.float32
    plain, ratio64, _allow = _numpy_assess(instance["idx"], N, dual)
    assert ratio64 <= 1.0
    d64 = -instance["b"].astype(np.float64) @ dual.astype(np.float64) - .5 * float(dual.astype(np.float64) @ dual.astype(np.float64))
    assert lower <= d64 + 1e-5 * max(1.0, abs(d64))


def test_returned_certifying_duals_are_inside_in_every_transform(instance):
    comp, data = _pack(instance)
    x = jnp.asarray(instance["x"])
    ys, _truth = _witnesses(instance)
    with jax.disable_jit():
        eager = [comp.objective_bound(x, y, data, return_dual=True) for y in ys]
    jitted = [comp.objective_bound(x, y, data, return_dual=True) for y in ys]
    count = len(ys)
    xb = jnp.stack([x] * count)
    yb = jnp.stack([jnp.asarray(y) for y in ys])
    dense = jax.vmap(comp._resolve_dense)({"b": jnp.stack([data["b"]] * count)})
    vm, vdual = comp._objective_bound_many(xb, yb, dense, 32, return_dual=True)
    jvm, jdual = jax.jit(lambda a, b_, d: comp._objective_bound_many(a, b_, d, 32, return_dual=True))(xb, yb, dense)
    engaged = 0
    for i in range(count):
        for rep in (eager[i], jitted[i]):
            if rep["objective_lower_bound_available"]:
                _check_dual(instance, rep["certifying_dual"], rep["objective_lower_bound"])
            engaged += int(rep["fft_l1_scaled_engaged"]) + int(rep["fft_l1_corrected_engaged"])
        for rep, dual in ((vm, vdual), (jvm, jdual)):
            if bool(rep["objective_lower_bound_available"][i]):
                _check_dual(instance, dual[i], float(rep["objective_lower_bound"][i]))
            engaged += int(np.sum(np.asarray(rep["fft_l1_scaled_engaged"][i])))
            engaged += int(np.sum(np.asarray(rep["fft_l1_corrected_engaged"][i])))
    # Every witness here sits at the float32 boundary; the bound must still be
    # available from at least one of them.
    assert any(r["objective_lower_bound_available"] for r in jitted)
    assert jitted[-1]["objective_lower_bound_available"]      # the old false pass, recovered inside
    print("scaled-fallback engagements across transforms:", engaged)
    assert not jax.config.jax_enable_x64


def test_solves_never_return_an_out_of_domain_witness(instance):
    comp = _family(instance["idx"], N).compile(tol=1e-3, max_blocks=2000)
    b = instance["b"]
    s = comp.solve_instance({"b": jnp.asarray(b)})
    if s.status == "solved":
        _check_dual(instance, s.y, s.objective_lower_bound)
        rep = comp.objective_bound(s.x, s.y, {"b": jnp.asarray(b)}, recover=False)
        assert rep["objective_lower_bound_available"] is True
    b2 = (b + np.float32(0.01) * np.random.default_rng(7).standard_normal(b.shape).astype(np.float32)).astype(np.float32)
    rows = comp.solve_many_results({"b": jnp.stack([jnp.asarray(b), jnp.asarray(b2)])})
    for row, bb in zip(rows, (b, b2)):
        if row.status == "solved":
            inst = dict(instance, b=bb)
            _check_dual(inst, row.y, row.objective_lower_bound)
    assert not jax.config.jax_enable_x64


def test_other_families_do_not_trace_the_guard():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((12, 8)).astype(np.float32)
    b = rng.standard_normal(12).astype(np.float32)
    x = ocs.Variable(8)
    fam = ocs.minimize([ocs.l1(x, 0.1), ocs.least_squares(ocs.Dense(A) @ x - ocs.Parameter("b"))])
    comp = fam.compile(tol=1e-3, max_blocks=20)
    rep = comp.objective_bound(jnp.zeros(8), jnp.zeros(12), {"b": jnp.asarray(b)}, recover=False)
    assert rep["coupling_arithmetic_guard"] == "operator_callback_arithmetic"
    assert not any(k.startswith("fft_l1_") for k in rep)
    idx = _idx(64, 0)
    v = ocs.Variable(64)
    fam2 = ocs.minimize([ocs.nonneg(v), ocs.least_squares(ocs.FFTMask(idx, 64) @ v - ocs.Parameter("b"))])
    comp2 = fam2.compile(tol=1e-3, max_blocks=20)
    rep2 = comp2.objective_bound(jnp.zeros(64), jnp.zeros(2 * len(idx)), {"b": jnp.zeros(2 * len(idx), jnp.float32)}, recover=False)
    assert rep2["coupling_arithmetic_guard"] == "operator_callback_arithmetic"
    assert not any(k.startswith("fft_l1_") for k in rep2)


def test_real_fft_path_and_index_contract():
    """The assessment's float64 adjoint is the operator's adjoint on the real
    FFT implementation too; a non-unique index set does not trace the guard."""
    n = 512
    idx = _idx(n, 1)
    rng = np.random.default_rng(11)
    y = (1e-4 * rng.standard_normal(2 * len(idx))).astype(np.float32)
    for real_fft in (False, True):
        x = ocs.Variable(n)
        F = ocs.FFTMask(idx, n, real_fft=real_fft)
        k32 = np.asarray(jax.jit(F.rmatvec)(jnp.asarray(y)), np.float64)
        assert np.max(np.abs(k32 - _adj64(idx, n, y))) <= 1e-6 * np.max(np.abs(k32))
        fam = ocs.minimize([ocs.l1(x, LAM), ocs.least_squares(F @ x - ocs.Parameter("b"))])
        comp = fam.compile(tol=1e-3, max_blocks=20)
        rep = comp.objective_bound(jnp.zeros(n), jnp.asarray(y), {"b": jnp.zeros(2 * len(idx), jnp.float32)},
                                   recover=False)
        _plain, ratio64, allow = _numpy_assess(idx, n, y)
        assert rep["coupling_arithmetic_guard"] == "fftmask_l1_float64_adjoint_envelope"
        assert ratio64 <= rep["fft_l1_supplied_ratio_float64"] <= ratio64 * (1 + 1e-6)
        assert rep["fft_l1_supplied_allowance_float32"] >= allow
        assert rep["supplied_dual_available"] == bool(ratio64 <= 1.0 and _float32_old_test(idx, n, y))
    dup = np.r_[idx, idx[:3]]
    x = ocs.Variable(n)
    fam = ocs.minimize([ocs.l1(x, LAM), ocs.least_squares(ocs.FFTMask(dup, n) @ x - ocs.Parameter("b"))])
    comp = fam.compile(tol=1e-3, max_blocks=20)
    rep = comp.objective_bound(jnp.zeros(n), jnp.zeros(2 * len(dup), jnp.float32),
                               {"b": jnp.zeros(2 * len(dup), jnp.float32)}, recover=False)
    assert rep["coupling_arithmetic_guard"] == "operator_callback_arithmetic"
