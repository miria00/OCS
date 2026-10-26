"""Planner failures and reuse must not silently change the execution contract."""
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ocs
from ocs.experimental import compile_with_plan
from ocs.experimental.plan import PlanningError, family_signature, plan_family

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the setup planner and device pinning, and its GPU case checks
# `sol.residual <= tol` directly. Under the default, status is awarded by a
# certified objective gap instead, and the two differ: that case is accepted
# at gap 7.10e-05 with a prox-graph residual of 4.20e-03. See the
# `residual_criterion` fixture in tests/conftest.py. No assertion is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")

planner = importlib.import_module("ocs.experimental.plan")


def family_and_data():
    x = ocs.Variable(4)
    family = ocs.minimize([
        ocs.nonneg(x),
        ocs.least_squares(ocs.Diagonal(jnp.asarray([1., 2., 3., 4.])) @ x - ocs.Parameter("b")),
    ])
    return family, {"b": jnp.asarray([1., 2., 3., 4.])}


class MemoryDevice:
    platform = "gpu"
    device_kind = "test-budget-device"
    id = 0
    process_index = 0
    uuid = "test-budget-device-uuid"

    def __init__(self, peak=1000, limit=10**9):
        self.peak = peak
        self.limit = limit

    def memory_stats(self):
        return {"bytes_limit": self.limit, "peak_bytes_in_use": self.peak}


@pytest.fixture
def measured(monkeypatch):
    device = MemoryDevice()
    calls = []
    monkeypatch.setattr(planner, "device_info", lambda dev=None: (device, device.device_kind, device.limit))
    monkeypatch.setattr(planner, "_source_identity", lambda: "test-source")
    # Device-placement has a separate real-device test. These tests isolate
    # budget policy without making arrays on an imaginary backend.
    monkeypatch.setattr(planner, "_array_metadata", lambda family, dev: 4)
    def probe(family, data, candidate, fixed, block, dev):
        calls.append((data, candidate, fixed, block, dev))
        return .01, device.peak, None
    monkeypatch.setattr(planner, "_probe", probe)
    return device, calls


def test_cache_hit_and_new_data_and_block_are_distinct(tmp_path, measured):
    _, calls = measured
    family, data = family_and_data()
    kwargs = dict(cache_dir=tmp_path, candidates=[{"data_layout": "dense"}], mode="cached")
    first = plan_family(family, data, block=7, **kwargs)
    again = plan_family(family, data, block=7, **kwargs)
    assert first.key == again.key and len(calls) == 1
    assert first.compile_kwargs["block"] == 7
    new_data = {"b": data["b"] + 1}
    changed = plan_family(family, new_data, block=7, **kwargs)
    block = plan_family(family, data, block=9, **kwargs)
    assert len({first.key, changed.key, block.key}) == 3
    assert len(calls) == 3


def test_source_runtime_dtype_and_candidate_policy_change_keys(tmp_path, measured, monkeypatch):
    family, data = family_and_data()
    def make(**extra):
        return plan_family(family, data, cache_dir=tmp_path, probe=False, **extra)
    first = make(candidates=[{"data_layout": "dense"}])
    candidate = make(candidates=[{"data_layout": "compact"}])
    monkeypatch.setattr(planner, "_source_identity", lambda: "changed-source")
    source = make(candidates=[{"data_layout": "dense"}])
    with jax.enable_x64():
        runtime = make(candidates=[{"data_layout": "dense"}])
        dtype = plan_family(family, {"b": data["b"].astype(jnp.float64)},
                            cache_dir=tmp_path, probe=False)
    assert len({first.key, candidate.key, source.key, runtime.key, dtype.key}) == 5
    assert dtype.estimated_bytes > first.estimated_bytes


def test_operator_values_and_block_parameters_are_not_shape_only():
    family, _ = family_and_data()
    original = family_signature(family, {})
    family.terms[1].op.d = family.terms[1].op.d * 2
    changed_operator = family_signature(family, {})
    family.terms[1].block = ocs.L1(2.)
    changed_block = family_signature(family, {})
    family.terms[1].block.lam = 7.
    assert len({original, changed_operator, changed_block, family_signature(family, {})}) == 4


def test_runtime_custom_class_oracle_disables_reuse(monkeypatch, tmp_path, measured):
    family, _ = family_and_data()
    monkeypatch.setattr(type(family.terms[1].op), "matvec", lambda self, x: x)
    result = plan_family(family, None, cache_dir=tmp_path, probe=False)
    assert result.cache_scope == "disabled" and "custom class oracle" in result.note


def test_mutable_data_and_custom_callbacks_bypass_cache(tmp_path, measured):
    _, calls = measured
    family, _ = family_and_data()
    data = {"b": np.ones(4, np.float32)}
    first = plan_family(family, data, cache_dir=tmp_path, candidates=[{}], mode="cached")
    data["b"][0] = 3
    second = plan_family(family, data, cache_dir=tmp_path, candidates=[{}], mode="cached")
    assert first.cache_scope == second.cache_scope == "disabled"
    assert len(calls) == 2 and not list(tmp_path.glob("*.json"))
    family.cost_fn = lambda data: jnp.ones(4)
    opaque = plan_family(family, {"b": jnp.ones(4)}, cache_dir=tmp_path, probe=False)
    assert opaque.cache_scope == "disabled" and "custom callback" in opaque.note


@pytest.mark.parametrize("failure", ["runtime", "overbudget", "missing_peak"])
def test_failed_probes_never_return_estimate_fallback(tmp_path, measured, monkeypatch, failure):
    device, _ = measured
    family, data = family_and_data()
    result = {"runtime": (None, None, "deliberate compile failure"),
              "overbudget": (.01, device.limit, None),
              "missing_peak": (.01, None, None)}[failure]
    monkeypatch.setattr(planner, "_probe", lambda *a: result)
    with pytest.raises(PlanningError, match="no measured candidate") as err:
        plan_family(family, data, cache_dir=tmp_path, candidates=[{}])
    assert len(err.value.candidates) == 1
    assert not list(tmp_path.glob("*.json"))


def test_late_lifetime_peak_invalidates_earlier_success(tmp_path, measured, monkeypatch):
    device, _ = measured
    family, data = family_and_data()
    def probe(*args):
        if args[2]["data_layout"] == "dense":
            return .01, 1000, None
        device.peak = device.limit
        return .02, device.peak, None
    monkeypatch.setattr(planner, "_probe", probe)
    with pytest.raises(PlanningError, match="prior allocations"):
        plan_family(family, data, cache_dir=tmp_path)


def test_cached_measurement_rechecked_against_current_lifetime_peak(tmp_path, measured):
    device, _ = measured
    family, data = family_and_data()
    plan_family(family, data, cache_dir=tmp_path, candidates=[{}])
    device.peak = device.limit
    with pytest.raises(PlanningError):
        plan_family(family, data, cache_dir=tmp_path, candidates=[{}], mode="cached")


def test_all_estimates_over_budget_reject_without_probe(tmp_path, measured):
    device, calls = measured
    device.limit = 1
    family, data = family_and_data()
    with pytest.raises(PlanningError, match="screening estimates"):
        plan_family(family, data, cache_dir=tmp_path, probe=False)
    assert not calls


def test_fixed_choices_and_empty_candidates(tmp_path, measured):
    family, data = family_and_data()
    fixed = plan_family(family, data, cache_dir=tmp_path, probe=False,
                        data_layout="compact", loop_unroll=1, pdhg_variant="standard")
    assert fixed.compile_kwargs["data_layout"] == "compact"
    assert fixed.compile_kwargs["loop_unroll"] == 1
    assert len(fixed.candidates) == 1
    with pytest.raises(ValueError, match="at least one"):
        plan_family(family, data, cache_dir=tmp_path, candidates=[])
    with pytest.raises(ValueError, match="conflicts"):
        plan_family(family, data, cache_dir=tmp_path, data_layout="dense",
                    candidates=[{"data_layout": "compact"}])
    with pytest.raises(ValueError, match="only data_layout"):
        plan_family(family, data, cache_dir=tmp_path, candidates=[{"tol": .2}])


def test_corrupt_cache_is_a_miss(tmp_path, measured):
    _, calls = measured
    family, data = family_and_data()
    first = plan_family(family, data, cache_dir=tmp_path, candidates=[{}])
    path = Path(tmp_path) / (first.key + ".json")
    path.write_text("{broken")
    second = plan_family(family, data, cache_dir=tmp_path, candidates=[{}], mode="cached")
    assert second.key == first.key and len(calls) == 2
    assert json.loads(path.read_text())["schema"] == planner._SCHEMA


def test_gpu_probe_and_public_calls_pin_selected_device(tmp_path):
    gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)
    if gpu is None:
        pytest.skip("requires a GPU for cross-backend placement check")
    with jax.default_device(gpu):
        family, data = family_and_data()
    cpu = jax.devices("cpu")[0]
    host_data = jax.device_put(data, cpu)
    with jax.default_device(cpu):
        comp = compile_with_plan(family, host_data, device=gpu, cache_dir=tmp_path,
                                 candidates=[{"data_layout": "dense"}],
                                 equilibrate="scalar", tol=1e-3, max_blocks=50)
        sol = comp.solve_instance(host_data)
    assert sol.x.devices() == sol.y.devices() == {gpu}
    assert sol.status == "solved" and sol.residual <= 1e-3
    np.testing.assert_allclose(np.asarray(sol.x), np.ones(4), atol=.004, rtol=.004)
    assert comp.plan.compile_kwargs["block"] == 100
    assert comp.plan.memory_scope.startswith("device process-lifetime")
    with jax.default_device(gpu):
        direct = family.compile(**comp.plan.compile_kwargs).solve_instance(data)
    np.testing.assert_array_equal(np.asarray(sol.x), np.asarray(direct.x))
    np.testing.assert_array_equal(np.asarray(sol.y), np.asarray(direct.y))
    assert (sol.status, sol.iters, sol.residual) == (direct.status, direct.iters, direct.residual)


def test_captured_other_device_array_is_rejected(tmp_path):
    gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)
    if gpu is None:
        pytest.skip("requires a GPU for cross-backend placement check")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(gpu):
        family, data = family_and_data()
    family.terms[1].op.d = jax.device_put(family.terms[1].op.d, cpu)
    with pytest.raises(ValueError, match="another device"):
        plan_family(family, data, device=gpu, cache_dir=tmp_path, probe=False)


@pytest.mark.parametrize("entry", ["certificate_many", "solve_diff"])
def test_planned_public_array_entries_preserve_jax_transformations(tmp_path, entry):
    family, data = family_and_data()
    comp = compile_with_plan(family, data, cache_dir=tmp_path, probe=False,
                             equilibrate="scalar", tol=1e-5, max_blocks=50)
    if entry == "certificate_many":
        x = jnp.ones((1, 4))
        y = jnp.zeros((1, 4))
        batch = {"b": data["b"][None]}
        eager = comp.certificate_many(x, y, batch)
        traced = jax.jit(comp.certificate_many)(x, y, batch)
        for name in eager:
            np.testing.assert_array_equal(np.asarray(eager[name]), np.asarray(traced[name]))
        gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)
        if gpu is not None:
            cpu = jax.devices("cpu")[0]
            cpu_args = jax.device_put((x, y, batch), cpu)
            for default in (None, cpu):
                try:
                    with jax.default_device(default):
                        cross_backend = jax.jit(comp.certificate_many)(*cpu_args)
                except ValueError as exc:
                    assert "device" in str(exc).lower()
                else:
                    assert all(value.devices() == {gpu} for value in cross_backend.values())
    else:
        gradient = jax.jit(jax.grad(lambda b: jnp.sum(comp.solve_diff({"b": b}))))(data["b"])
        np.testing.assert_allclose(np.asarray(gradient), 1 / np.arange(1, 5), atol=2e-5, rtol=2e-5)
