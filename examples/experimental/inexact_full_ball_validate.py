"""Bounded Mazeppa GPU validation; independent NumPy/SciPy float64 references.

Run this file only on the authorized Mazeppa RTX 5090 under the cooperative
GPU lock. This is numerical evidence for a new experimental helper, not a
speed comparison, public OCS integration, or a rigorous floating enclosure.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from inexact_full_ball_prox import (
    adaptive_ball_prox, prepare_metric, prepare_budget, summable_budget, STATUS_NAMES,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    if isinstance(obj, np.generic):
        return safe_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def float64_reference(Q, sigma, v, radius):
    """Diagonalize ACTUAL rounded Sigma afresh; solve its scalar root in float64.

    This deliberately does not reuse the runtime approximate inverse or its
    root bracket. The stored numerical reference is moved inward before use.
    """
    Q, sigma, v = [np.asarray(a, np.float64) for a in (Q, sigma, v)]
    S = (Q * sigma[None, :]) @ Q.T
    D = np.linalg.inv(S)
    if radius == 0:
        return np.zeros_like(v), D, 0.
    if np.linalg.norm(v) <= radius:
        return v.copy(), D, 0.
    values, basis = np.linalg.eigh(S)
    w = basis.T @ v
    fn = lambda eta: np.linalg.norm(w / (1 + values * eta)) - radius
    high = max(1., (np.linalg.norm(v) / radius - 1) / min(values))
    while fn(high) > 0:
        high *= 2
    eta = brentq(fn, 0., high, xtol=np.nextafter(0., 1.),
                 rtol=4 * np.finfo(float).eps, maxiter=256)
    point = basis @ (w / (1 + values * eta))
    point *= min(1., radius * (1 - 8 * np.finfo(float).eps) / np.linalg.norm(point))
    return point, D, eta


def rotations(rng, B, d):
    return np.stack([np.linalg.qr(rng.normal(size=(d, d)))[0] for _ in range(B)]).astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert platform.node() == "Mazeppa", "Numerical work is authorized only on Mazeppa"
    assert not args.output.exists(), "Preserve prior runs; choose a fresh output directory"
    args.output.mkdir(parents=True)
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
    ], text=True).strip()
    assert "78b0d21c-fe02-946b-2e74-6930e848f338" in gpu and "5090" in gpu
    jax.config.update("jax_enable_x64", True)
    assert jax.default_backend() == "gpu"
    start_campaign = time.perf_counter()
    report = dict(host=platform.node(), gpu=gpu, jax=jax.__version__,
                  devices=list(map(str, jax.devices())),
                  helper_sha256=digest(Path(__file__).with_name("inexact_full_ball_prox.py")),
                  harness_sha256=digest(__file__), runtime_dtype="float32",
                  reference="CPU NumPy/SciPy float64 on actual rounded Sigma",
                  scope="Bounded experimental prox validation; no solver integration or speed claim",
                  cases=[], guards=[])
    rng = np.random.default_rng(20260915)
    input_block_hashes = set()

    def run_case(name, Q, sigma, v, radius, limits, expected_success,
                 max_iterations=80, total_override=None):
        Q, sigma, v, radius = [np.asarray(a, np.float32) for a in (Q, sigma, v, radius)]
        B = len(sigma)
        for b in range(B):
            identity = hashlib.sha256(str(Q[b].shape).encode())
            for array in (Q[b], sigma[b], v[b], radius[b]):
                identity.update(np.asarray(array).tobytes())
            input_block_hashes.add(identity.hexdigest())
        limits = np.asarray(limits, float)
        requested_total = math.fsum(limits) * (1 + 1e-14)
        budget = prepare_budget(requested_total, B, per_block=limits)
        metric = prepare_metric(Q, sigma)
        runtime_total = budget.total if total_override is None else jnp.asarray(total_override, jnp.float32)
        inputs_path = args.output / (name + ".inputs.npz")
        np.savez_compressed(inputs_path, Q=Q, sigma=sigma, v=v, radius=radius,
                            requested_per_block=limits, effective_per_block=np.asarray(budget.per_block),
                            runtime_total=np.asarray(runtime_total))
        fn = jax.jit(partial(adaptive_ball_prox, max_iterations=max_iterations))
        begin = time.perf_counter()
        result = fn(metric, jnp.asarray(v), jnp.asarray(radius), runtime_total, budget.per_block)
        jax.block_until_ready(result)
        cold = time.perf_counter() - begin
        begin = time.perf_counter()
        repeat = fn(metric, jnp.asarray(v), jnp.asarray(radius), runtime_total, budget.per_block)
        jax.block_until_ready(repeat)
        warm = time.perf_counter() - begin
        data = {k: np.asarray(val) for k, val in result.items()}
        endpoint_path = args.output / (name + ".endpoint.npz")
        np.savez_compressed(endpoint_path, **data)
        blocks = []
        for b in range(B):
            reference, D, eta = float64_reference(Q[b], sigma[b], v[b], float(radius[b]))
            y = data["y"][b].astype(float)
            vv = v[b].astype(float)
            lam = float(radius[b])
            q = D @ (vv - y)
            measured = lam * np.linalg.norm(q) - q @ y
            fp64_allowance = 64 * np.finfo(float).eps * (lam * np.linalg.norm(q) + abs(q @ y))
            # Physical norm is independently checked; no runtime y is replaced.
            feasible = np.linalg.norm(y) <= lam
            upper_ok = measured <= float(data["error_upper_estimate"][b]) + fp64_allowance
            per_pass = measured <= float(limits[b]) + fp64_allowance
            primal_gap = .5 * ((y-vv) @ D @ (y-vv) - (reference-vv) @ D @ (reference-vv))
            dist = np.sqrt(max(0., (y-reference) @ D @ (y-reference)))
            blocks.append(dict(block=b, norm=float(np.linalg.norm(y)), radius=lam,
                               independently_feasible=bool(feasible), measured_type2=float(measured),
                               estimated_type2=float(data["error_upper_estimate"][b]),
                               estimate_covers_measurement=bool(upper_ok),
                               independently_within_block_budget=bool(per_pass),
                               float64_reference_feasible=bool(np.linalg.norm(reference) <= lam),
                               reference_eta=float(eta), reference_distance_D=float(dist),
                               reference_prox_objective_gap=float(primal_gap),
                               iterations=int(data["iterations"][b]),
                               status=STATUS_NAMES[int(data["block_status"][b])]))
        actual_sum = math.fsum(max(0., z["measured_type2"]) for z in blocks)
        block_estimate_sum = math.fsum(float(x) for x in data["error_upper_estimate"])
        sums_estimates = float(data["total_error_upper_estimate"]) >= block_estimate_sum
        success = bool(data["success"])
        accepted_points_valid = (all(b["independently_feasible"] and b["estimate_covers_measurement"]
                                     and b["independently_within_block_budget"] for b in blocks)
                                 and actual_sum <= float(runtime_total))
        case_pass = (sums_estimates
                     and all(b["independently_feasible"] and b["estimate_covers_measurement"] for b in blocks)
                     and (not success or accepted_points_valid)
                     and (expected_success is None or success == expected_success))
        row = dict(name=name, blocks=B, expected_success=expected_success, success=success,
                   global_status=STATUS_NAMES[int(data["global_status"])], passed=bool(case_pass),
                   cold_inclusive_seconds=cold, warm_inclusive_seconds=warm,
                   requested_total=requested_total, runtime_total=float(runtime_total),
                   actual_type2_sum=actual_sum,
                   block_estimate_sum=block_estimate_sum,
                   total_estimate_covers_block_sum=bool(sums_estimates),
                   total_upper_estimate=float(data["total_error_upper_estimate"]),
                   max_iterations=max_iterations, block_results=blocks,
                   inputs_sha256=digest(inputs_path), endpoint_sha256=digest(endpoint_path))
        report["cases"].append(row)
        (args.output / "progress.json").write_text(json.dumps(safe_json(report), indent=2) + "\n")
        print(json.dumps(safe_json({k: val for k, val in row.items() if k != "block_results"})), flush=True)

    # Algebraic zero-error branches, even with a zero requested budget.
    Q = rotations(rng, 4, 3)
    sigma = np.tile([.1, 1., 10.], (4, 1))
    run_case("exact_branches", Q, sigma,
             [[3, -2, 1], [0, 0, 0], [.1, .2, .1], [-.1, 0, .1]],
             [0, 1, 1, .5], [0] * 4, True)

    # Varied rotations, spectra, magnitude regimes and just-outside boundaries.
    for d in (1, 3, 8):
        B = 18
        Q = rotations(rng, B, d)
        sigma = np.exp(rng.uniform(np.log(.1), np.log(10), size=(B, d))).astype("float32")
        radius = np.geomspace(1e-3, 1e3, B).astype("float32")
        v = rng.normal(size=(B, d))
        v /= np.linalg.norm(v, axis=1)[:, None]
        factors = np.resize([1 + 4 * np.finfo("float32").eps, 1.2, 3.], B)
        v *= (radius * factors)[:, None]
        limits = 1e-3 * radius.astype(float) ** 2 / sigma.min(axis=1)
        run_case(f"rotated_d{d}", Q, sigma, v, radius, limits, True)

    Q = rotations(rng, 6, 3)
    Q[:, 0, 0] += np.float32(1e-5)
    sigma = np.tile([.1, 1., 10.], (6, 1))
    v = rng.normal(size=(6, 3)) * 2
    run_case("nonorthogonal_rounded_basis", Q, sigma, v, np.ones(6),
             [1e-2] * 6, True)

    # Unequal metrics must not collapse to an unweighted radial projection.
    Q = rotations(rng, 4, 3)
    sigma = np.tile([1e-4, 1, 1e4], (4, 1))
    v = np.tile([2., -1., .5], (4, 1))
    run_case("condition_1e8_diagnostic", Q, sigma, v, np.ones(4),
             [1e-3] * 4, None)

    Q = np.tile(np.eye(3, dtype="float32"), (3, 1, 1))
    sigma = np.tile([.1, 1., 10.], (3, 1))
    v = np.tile([2., -1., .5], (3, 1))
    run_case("finite_cap_failure", Q, sigma, v, np.ones(3), [1e-3] * 3, False,
             max_iterations=0)
    run_case("zero_active_budget_failure", Q, sigma, v, np.ones(3), [0.] * 3, False)
    run_case("allocation_over_total_failure", Q, sigma, v, np.ones(3), [1e-3] * 3, False,
             total_override=1e-4)

    # Counter values preserve chunk history; restarting at zero is observably different.
    first = [summable_budget(k, 1e-2, .5) for k in range(4)]
    continued = [summable_budget(k, 1e-2, .5) for k in range(4, 8)]
    report["counter_schedule"] = dict(first_chunk=first, continued_chunk=continued,
                                       reset_would_be=first, passed=continued[0] < first[-1])
    checks = [
        ("negative_sigma", lambda: prepare_metric(np.eye(2)[None], [[1, -1]])),
        ("nonorthogonal_Q", lambda: prepare_metric((2 * np.eye(2))[None], [[1, 1]])),
        ("bad_metric_shape", lambda: prepare_metric(np.eye(2), [[1, 1]])),
        ("overallocated_budget", lambda: prepare_budget(1, 2, per_block=[1, 1])),
        ("negative_budget", lambda: prepare_budget(-1, 1)),
        ("unsupported_budget_dtype", lambda: prepare_budget(.1, 1, dtype=np.float16)),
        ("negative_counter", lambda: summable_budget(-1, 1., 1.)),
        ("nonsummable_schedule", lambda: summable_budget(0, 1., 0.)),
    ]
    for name, fn in checks:
        try:
            fn()
        except ValueError as error:
            report["guards"].append(dict(name=name, passed=True, message=str(error)))
        else:
            report["guards"].append(dict(name=name, passed=False))
    jax.config.update("jax_enable_x64", False)
    try:
        prepare_budget(.1, 1, dtype=np.float64)
    except ValueError as error:
        report["guards"].append(dict(name="x64_disabled_budget", passed=True, message=str(error)))
    else:
        report["guards"].append(dict(name="x64_disabled_budget", passed=False))
    finally:
        jax.config.update("jax_enable_x64", True)
    metric = prepare_metric(np.eye(3, dtype="float32")[None], np.ones((1, 3), "float32"))
    budget = prepare_budget(1e-3, 1)
    fn = jax.jit(partial(adaptive_ball_prox, max_iterations=8))
    for name, vv, rr in [
        ("negative_radius", [[1., 2., 3.]], [-1.]),
        ("nonfinite_radius", [[1., 2., 3.]], [float("nan")]),
        ("nonfinite_v", [[float("nan"), 2., 3.]], [1.]),
    ]:
        out = fn(metric, jnp.asarray(vv, dtype=jnp.float32),
                 jnp.asarray(rr, dtype=jnp.float32), budget.total, budget.per_block)
        jax.block_until_ready(out)
        passed = not bool(out["success"]) and int(out["global_status"]) == 3
        report["guards"].append(dict(name=name, passed=passed,
                                    status=STATUS_NAMES[int(out["global_status"])]))
    report["elapsed_seconds"] = time.perf_counter() - start_campaign
    report["passed"] = (all(row["passed"] for row in report["cases"])
                        and all(row["passed"] for row in report["guards"])
                        and report["counter_schedule"]["passed"])
    report["block_evaluations"] = sum(row["blocks"] for row in report["cases"])
    report["distinct_input_blocks"] = len(input_block_hashes)
    (args.output / "results.json").write_text(json.dumps(safe_json(report), indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "block_evaluations", "distinct_input_blocks", "elapsed_seconds")}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
