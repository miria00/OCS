"""Tiny exact-reference mixed-metric research; never imported by the OCS engine.

References use NumPy/SciPy float64 on the authorized remote host. Downstream
ordinary PDHG uses explicitly typed float32 JAX arrays on that host's GPU.
This is a quadratic research prototype, not a measurement of public OCS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp


def sym_function(a, fun):
    v, q = np.linalg.eigh((a + a.T) / 2)
    return (q * fun(v)) @ q.T


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def potential(a, b, k, mu):
    la = np.log(np.linalg.eigvalsh(a))
    lb = np.log(np.diag(b))
    return float(.5 * np.trace(a @ k @ b @ k.T) - .5 * (la.sum() + lb.sum())
                 + mu / 8 * (la @ la + lb @ lb))


def gradients(a, b, k, mu):
    ga = .5 * a @ k @ b @ k.T @ a - .5 * a + mu / 4 * a @ sym_function(a, np.log)
    gb = .5 * b @ k.T @ a @ k @ b - .5 * b + mu / 4 * b @ sym_function(b, np.log)
    return (ga + ga.T) / 2, np.diag(np.diag(gb))


def exp_step(a, grad, eta, box):
    ah = sym_function(a, np.sqrt)
    ai = sym_function(a, lambda v: 1 / np.sqrt(v))
    z = ah @ sym_function(-eta * ai @ grad @ ai, np.exp) @ ah
    return sym_function(z, lambda v: np.clip(v, np.exp(-2 * box), np.exp(2 * box)))


def grad_mapping(a, b, k, mu, box, arm):
    ga, gb = gradients(a, b, k, mu)
    if arm == "scalar":
        ga = np.eye(2) * np.trace(ga) / 2
    elif arm == "diagonal":
        ga = np.diag(np.diag(ga))
    eta = 1e-3
    aa = exp_step(a, ga, eta, box)
    bb = exp_step(b, gb, eta, box)
    total = 0.
    for old, new in ((a, aa), (b, bb)):
        inv = sym_function(old, lambda v: 1 / np.sqrt(v))
        total += np.sum(np.log(np.linalg.eigvalsh(inv @ new @ inv)) ** 2) / eta**2
    return float(np.sqrt(total))


def exact_reference(k, arm, mu, box):
    """Optimize log eigenvalues + rotation (2D only); audit intrinsic gradient.

    Multiple angle starts avoid the eigen-coordinate singularity at scalar A.
    The geodesic convexity proof, together with small intrinsic projected
    gradient, is the global-optimality rationale; multistart alone is not one.
    """
    dim_a = {"scalar": 1, "diagonal": 2, "full": 3}[arm]

    def decode(z):
        if arm == "scalar":
            av = np.repeat(np.exp(z[0]), 2)
            q = np.eye(2)
        elif arm == "diagonal":
            av = np.exp(z[:2]); q = np.eye(2)
        else:
            av = np.exp(z[:2]); q = rotation(z[2])
        return (q * av) @ q.T, np.diag(np.exp(z[dim_a:])), av, q

    def fun(z):
        a, b, av, q = decode(z)
        logs_a = np.log(av)
        da = .5 * av * np.diag(q.T @ k @ b @ k.T @ q) - .5 + mu / 4 * logs_a
        db = .5 * np.diag(b) * np.diag(k.T @ a @ k) - .5 + mu / 4 * z[dim_a:]
        if arm == "scalar":
            d = np.r_[da.sum(), db]
        elif arm == "diagonal":
            d = np.r_[da, db]
        else:
            qp = q @ np.array([[0., -1.], [1., 0.]])
            ap = (qp * av) @ q.T + (q * av) @ qp.T
            d = np.r_[da, .5 * np.trace(ap @ k @ b @ k.T), db]
        return potential(a, b, k, mu), d

    starts = []
    for theta in ([0., np.pi / 8, np.pi / 4, -np.pi / 4] if arm == "full" else [0.]):
        z = np.zeros(dim_a + 2)
        if arm == "full":
            z[2] = theta
        bounds = [(-2 * box, 2 * box)] * len(z)
        if arm == "full":
            bounds[2] = (-np.pi, np.pi)
        result = minimize(fun, z, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 5000,
                                   "maxls": 50})
        a, b, _, _ = decode(result.x)
        starts.append((result, a, b))
    result, a, b = min(starts, key=lambda t: t[0].fun)
    return a, b, {"objective": float(result.fun), "optimizer_success": bool(result.success),
                  "optimizer_message": str(result.message), "optimizer_iterations": int(result.nit),
                  "projected_riemannian_gradient_norm": grad_mapping(a, b, k, mu, box, arm),
                  "start_objectives": [float(z[0].fun) for z in starts]}


def pdhg_matrices(k, bdata, a, b, alpha, safety=.95, weight=1.):
    norm = np.linalg.norm(sym_function(a, np.sqrt) @ k @ sym_function(b, np.sqrt), 2)
    sigma = a * safety / norm * weight
    tau = b * safety / norm / weight
    p = np.linalg.inv(np.eye(2) + alpha * tau)
    q = np.linalg.inv(np.eye(2) + sigma)
    px = p
    py = -p @ tau @ k.T
    yx = q @ sigma @ k
    yy = q
    c = -q @ sigma @ bdata
    jmat = np.block([[px, py], [yx @ (2 * px - np.eye(2)), yy + 2 * yx @ py]])
    return (px, py, yx, yy, c), jmat, sigma, tau


@jax.jit
def solve_pdhg(px, py, yx, yy, c, k, bdata, alpha, tol, maxiter):
    def residual(x, y):
        gx, aty, kx = alpha * x, k.T @ y, k @ x
        rp = jnp.linalg.norm(gx + aty) / jnp.maximum(jnp.linalg.norm(gx) + jnp.linalg.norm(aty), 1e-30)
        rd = jnp.linalg.norm(y - kx + bdata) / jnp.maximum(jnp.linalg.norm(y) + jnp.linalg.norm(kx) + jnp.linalg.norm(bdata), 1e-30)
        return jnp.maximum(rp, rd)

    def cond(state):
        it, x, y, res = state
        return (it < maxiter) & (res > tol)

    def body(state):
        it, x, y, res = state
        xn = px @ x + py @ y
        yn = yx @ (2 * xn - x) + yy @ y + c
        it = it + 1
        res = jax.lax.cond((it % 25 == 0) | (it == maxiter),
                           lambda _: residual(xn, yn), lambda _: res, operand=None)
        return it, xn, yn, res

    return jax.lax.while_loop(cond, body, (jnp.int32(0), jnp.zeros(2, jnp.float32),
                                         jnp.zeros(2, jnp.float32), jnp.float32(jnp.inf)))


def independent_check(x, y, k, b, alpha, tol):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    h = k.T @ k + alpha * np.eye(2)
    xs = np.linalg.solve(h, k.T @ b)
    f = lambda z: .5 * np.linalg.norm(k @ z - b)**2 + alpha / 2 * np.linalg.norm(z)**2
    obj, optimum = f(x), f(xs)
    gx, aty, kx = alpha * x, k.T @ y, k @ x
    rp = np.linalg.norm(gx + aty) / max(np.linalg.norm(gx) + np.linalg.norm(aty), 1e-30)
    rd = np.linalg.norm(y - kx + b) / max(np.linalg.norm(y) + np.linalg.norm(kx) + np.linalg.norm(b), 1e-30)
    ferr = np.linalg.norm(x - xs) / max(np.linalg.norm(xs), 1e-30)
    gap = max(obj - optimum, 0.) / max(abs(optimum), 1e-30)
    return {"rp": float(rp), "rd": float(rd), "objective": float(obj),
            "reference_objective": float(optimum), "relative_objective_error": float(gap),
            "relative_forward_error": float(ferr),
            "accepted": bool(max(rp, rd) <= tol * 1.1 and gap <= tol and ferr <= tol * 10)}


def run_measurement(k, bdata, a, b, alpha, tol, maxiter, warm_repeats=1):
    mats, jmat, sigma, tau = pdhg_matrices(k, bdata, a, b, alpha)
    args = [jnp.asarray(v, jnp.float32) for v in (*mats, k, bdata, alpha, tol)]
    args.append(jnp.asarray(maxiter, jnp.int32))
    start = time.perf_counter()
    result = solve_pdhg(*args)
    jax.block_until_ready(result)
    cold = time.perf_counter() - start
    timings = []
    for _ in range(warm_repeats):
        start = time.perf_counter(); result = solve_pdhg(*args); jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    it, x, y, res = result
    return {"iterations": int(it), "reported_residual": float(res), "cold_seconds": cold,
            "warm_seconds": timings, "spectral_radius": float(max(abs(np.linalg.eigvals(jmat)))),
            "coupling_norm": float(np.linalg.norm(sym_function(sigma, np.sqrt) @ k @ sym_function(tau, np.sqrt), 2)),
            "sigma": sigma.tolist(), "tau": tau.tolist(), "x": np.asarray(x).tolist(),
            "y": np.asarray(y).tolist(), "independent": independent_check(x, y, k, bdata, alpha, tol)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mu", type=float, default=.1)
    parser.add_argument("--box", type=float, default=2.)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--warm-repeats", type=int, default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("This research pilot requires the authorized remote GPU")
    records = []
    config = {"mu": args.mu, "M": args.box, "r": 1., "s": 1., "safety": .95, "weight": 1.,
              "tol": args.tol, "maxiter": args.maxiter, "warm_repeats": args.warm_repeats,
              "reference_dtype": "float64 CPU",
              "iteration_dtype": "float32 GPU", "B_structure": "diagonal",
              "solver": "research ordinary primal-first extrapolated PDHG; not public OCS",
              "host": platform.node(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for alpha in (.01, .0001):
        for name, condition, angle in (("isotropic", 1., 0.), ("aligned100", 100., 0.),
                                        ("rotated10", 10., np.pi / 4), ("rotated100", 100., np.pi / 4)):
            q = rotation(angle)
            k = q @ np.diag([1., 1. / condition]) @ q.T
            # Define the problem by the exact rounded float32 data actually supplied to the GPU.
            k = k.astype(np.float32).astype(np.float64)
            bdata = np.array([1., -.7], np.float32).astype(np.float64)
            alpha_actual = float(np.float32(alpha))
            for arm in ("scalar", "diagonal", "full"):
                start = time.perf_counter()
                a, b, ref = exact_reference(k, arm, args.mu, args.box)
                setup = time.perf_counter() - start
                row = {"case": name, "alpha": alpha_actual, "condition": condition, "arm": arm,
                       "K": k.tolist(), "b": bdata.tolist(), "A": a.tolist(), "B": b.tolist(),
                       "construction_seconds_cpu_f64": setup, "reference": ref,
                       "solve": run_measurement(k, bdata, a, b, alpha_actual, args.tol, args.maxiter, args.warm_repeats)}
                records.append(row)
                args.output.write_text(json.dumps({"config": config, "records": records}, indent=2) + "\n")
                print(json.dumps({"case": name, "alpha": alpha, "arm": arm,
                                  "pg": ref["projected_riemannian_gradient_norm"],
                                  "rho": row["solve"]["spectral_radius"],
                                  "iterations": row["solve"]["iterations"],
                                  "accepted": row["solve"]["independent"]["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
