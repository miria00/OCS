"""One shared squared-loss FFT-LASSO input and acceptance contract.

All adapters load the same NPZ; none regenerates rows, measurements or weights.
NumPy float64 FFTs provide a separate evaluator for the exact mathematical DFT
operator. OCS may execute that operator using float32 GPU FFTs. This module is
an experimental benchmark harness, not part of the public solver package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

CONTRACT = "squared_fft_lasso_v1"


def generate(path, n, seed=0, ratio=.25, weight=1e-3):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if platform.node() != "Mazeppa":
        raise RuntimeError("Generate experimental inputs only on the authorized Mazeppa host")
    m = max(1, int(n * ratio))
    idx = np.sort(np.random.default_rng(seed).choice(n, m, replace=False)).astype(np.int32)
    # Independent RNG reset matches the historical OCS instance, now shared.
    y = np.random.default_rng(seed).normal(size=2 * m).astype(np.float32)
    np.savez(path, n=np.int64(n), idx=idx, y=y, weight=np.float32(weight),
             seed=np.int64(seed), contract=np.array(CONTRACT))
    return load(path)


def load(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    assert str(data["contract"]) == CONTRACT
    n, idx, y, weight = int(data["n"]), data["idx"], data["y"], data["weight"]
    assert idx.dtype == np.int32 and y.dtype == np.float32 and weight.dtype == np.float32
    assert n > 0 and len(idx) > 0 and y.shape == (2 * len(idx),)
    assert np.all((0 <= idx) & (idx < n)) and np.all(np.diff(idx) > 0)
    assert np.isfinite(y).all() and np.isfinite(weight) and weight > 0
    data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    data["path"] = str(path.resolve())
    return data


def forward(data, x):
    c = np.fft.fft(np.asarray(x, np.float64), norm="ortho")[data["idx"]]
    return np.concatenate([c.real, c.imag])


def adjoint(data, u):
    m = len(data["idx"])
    u = np.asarray(u, np.float64)
    z = np.zeros(int(data["n"]), np.complex128)
    z[data["idx"]] = u[:m] + 1j * u[m:]
    return np.fft.ifft(z, norm="ortho").real


def dense_matrix(data, max_bytes):
    """Dense float64 realization of the same DFT; explicit allocation guard.

    This bound covers the real output and complex construction temporary only;
    it is neither a process/GPU peak-memory estimate nor an OOM measurement.
    """
    n, m = int(data["n"]), len(data["idx"])
    estimate = 32 * m * n
    if estimate > max_bytes:
        raise MemoryError(f"construction guard: {estimate} bytes > {max_bytes}; not a measured OOM")
    c = np.exp((-2j * np.pi / n) * np.outer(data["idx"], np.arange(n))) / np.sqrt(n)
    return np.concatenate([c.real, c.imag], axis=0)


def audit(data, x, tolerance=1e-3):
    """Evaluate the saved returned point, without optimizing/refining it.

    Dual: maximize -.5 ||nu||^2 - y^T nu, subject to ||K^T nu||inf <= weight.
    A radially scaled residual supplies a feasible dual candidate. The reported
    float64 bound is a numerical independent check, not interval arithmetic.
    """
    x = np.asarray(x, np.float64)
    assert x.shape == (int(data["n"]),) and np.isfinite(x).all()
    y, lam = data["y"].astype(np.float64), float(data["weight"])
    residual = forward(data, x) - y
    primal = float(.5 * np.dot(residual, residual) + lam * np.abs(x).sum())
    nu = residual.copy()
    dual_norm = float(np.max(np.abs(adjoint(data, nu))))
    if dual_norm:
        nu *= min(1., lam / dual_norm) * (1. - 32 * np.finfo(np.float64).eps)
    checked_norm = float(np.max(np.abs(adjoint(data, nu))))
    dual = float(-.5 * np.dot(nu, nu) - np.dot(nu, y))
    # nu=0 is always feasible and can improve a poor residual-derived bound.
    if dual < 0:
        nu[:] = 0
        dual, checked_norm = 0., 0.
    gap = primal - dual
    relative_gap = gap / primal if primal > 0 else (0. if gap == 0 else float("inf"))
    feasible = checked_norm <= lam * (1 + 8 * np.finfo(np.float64).eps)
    valid_bound = feasible and gap >= -64 * np.finfo(np.float64).eps * max(abs(primal), abs(dual))
    return {"contract": CONTRACT, "input_sha256": data["sha256"],
            "primal_objective_float64": primal, "dual_bound_float64": dual,
            "duality_gap": gap, "relative_gap_over_primal": relative_gap,
            "dual_inf_norm": checked_norm, "weight_exact_float32_value": lam,
            "dual_feasible": bool(feasible), "valid_numerical_bound": bool(valid_bound),
            "quality_tolerance": tolerance,
            "independent_quality_pass": bool(valid_bound and 0 <= relative_gap <= tolerance),
            "zero_case": "P=0 requires exact zero gap; generated Gaussian inputs are nonzero"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = generate(args.output, args.n, args.seed)
    print(json.dumps({"contract": CONTRACT, "n": int(data["n"]),
                      "input_sha256": data["sha256"], "path": data["path"]}))
