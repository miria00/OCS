"""Float64 tiny oracle-reference constructor; float32 GPU downstream checks.

This CPU reference verifies the exact manifold algorithm before a GPU
implementation. It is not a throughput claim or a public OCS constructor.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import jax

from geodesic_metric_pilot import (sym_function, rotation, potential, gradients,
                                   exp_step, exact_reference, grad_mapping,
                                   run_measurement)


def geometric_mean(a, b, weight):
    ah = sym_function(a, np.sqrt)
    ai = sym_function(a, lambda v: 1 / np.sqrt(v))
    return ah @ sym_function(ai @ b @ ai, lambda v: v**weight) @ ah


def tangent_norm_squared(a, g):
    ai = sym_function(a, lambda v: 1 / np.sqrt(v))
    return float(np.linalg.norm(ai @ g @ ai, "fro")**2)


def oracle_checks(mu=.1, box=2.):
    # Noncommuting, fixed inputs; these checks are deliberately not the easy
    # rotated diagonal path used by the decisive downstream example.
    k = rotation(.37) @ np.diag([1., .07]) @ rotation(-.23)
    a = np.array([[2., .3], [.3, .7]])
    b = np.diag([.8, 1.4])
    ga, gb = gradients(a, b, k, mu)
    ah, bh = sym_function(a, np.sqrt), sym_function(b, np.sqrt)
    probes = [np.array(p, float) for p in itertools.product((-1., 1.), repeat=2)]
    gram_a = sum(np.outer(k @ bh @ z, k @ bh @ z) for z in probes) / 4
    gram_b = sum(np.outer(k.T @ ah @ z, k.T @ ah @ z) for z in probes) / 4
    oracle_error = max(np.linalg.norm(gram_a - k @ b @ k.T),
                       np.linalg.norm(gram_b - k.T @ a @ k))
    u = np.array([[.3, -.2], [-.2, .4]])
    v = np.diag([-.1, .2])
    ai, bi = np.linalg.inv(ah), np.linalg.inv(bh)
    derivative = np.trace(ai @ ga @ ai @ u) + np.trace(bi @ gb @ bi @ v)
    def along(t):
        at = ah @ sym_function(t * u, np.exp) @ ah
        bt = bh @ sym_function(t * v, np.exp) @ bh
        return potential(at, bt, k, mu)
    step = 1e-5
    numerical = (along(step) - along(-step)) / (2 * step)
    derivative_error = abs(derivative - numerical) / max(1., abs(derivative))
    a1 = np.array([[.7, -.15], [-.15, 1.6]])
    b1 = np.diag([1.7, .6])
    d2 = sum(np.sum(np.log(np.linalg.eigvalsh(inv @ end @ inv))**2)
             for inv, end in ((ai, a1), (bi, b1)))
    violations = []
    for t in (.1, .3, .5, .8):
        lhs = potential(geometric_mean(a, a1, t), geometric_mean(b, b1, t), k, mu)
        rhs = (1-t)*potential(a, b, k, mu) + t*potential(a1, b1, k, mu) - mu/8*t*(1-t)*d2
        violations.append(float(lhs-rhs))
    # Verify scalar block multiplicity against a finite difference of log(a).
    aa = np.eye(2) * 1.3
    gas, _ = gradients(aa, b, k, mu)
    intrinsic_scalar_derivative = np.trace(gas) / 1.3
    scalar_fd = (potential(aa*np.exp(step), b, k, mu)
                 - potential(aa*np.exp(-step), b, k, mu))/(2*step)
    scalar_error = abs(intrinsic_scalar_derivative-scalar_fd)
    return {"oracle_enumeration_error": float(oracle_error),
            "geodesic_derivative_relative_error": float(derivative_error),
            "scalar_multiplicity_derivative_error": float(scalar_error),
            "strong_convexity_inequality_lhs_minus_rhs": violations,
            "passed": bool(oracle_error < 1e-12 and derivative_error < 1e-8
                            and scalar_error < 1e-8 and max(violations) <= 1e-10)}


def constructor(forward, adjoint, seed, budgets, mu, box, shift):
    """Only callbacks apply K. No Gram matrix or matrix K is inspected here."""
    rng = np.random.default_rng(seed)
    a, b = np.eye(2), np.eye(2)
    abar, bbar = None, None
    weight_sum = 0.
    mug = mu / 4
    start = time.perf_counter()
    endpoint = {}
    max_norm = 0.
    for t in range(1, max(budgets)+1):
        # Both fresh samples are computed from the same pre-update (A,B).
        xi = rng.choice([-1., 1.], 2)
        zeta = rng.choice([-1., 1.], 2)
        w = forward(sym_function(b, np.sqrt) @ xi)
        z = adjoint(sym_function(a, np.sqrt) @ zeta)
        ga = .5*a @ np.outer(w, w) @ a - .5*a + mu/4*a @ sym_function(a, np.log)
        gb = .5*b @ np.outer(z, z) @ b - .5*b + mu/4*b @ sym_function(b, np.log)
        gb = np.diag(np.diag(gb))
        ga = (ga+ga.T)/2
        # Average the pre-update iterates z_t, as in the proof, not z_(t+1).
        weight = t+shift-1
        weight_sum += weight
        if abar is None:
            abar, bbar = a.copy(), b.copy()
        else:
            abar = geometric_mean(abar, a, weight/weight_sum)
            bbar = geometric_mean(bbar, b, weight/weight_sum)
        max_norm = max(max_norm, tangent_norm_squared(a, ga)+tangent_norm_squared(b, gb))
        eta = 2/(mug*(t+shift))
        a = exp_step(a, ga, eta, box)
        b = exp_step(b, gb, eta, box)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise FloatingPointError(f"Nonfinite exact exponential/projection at t={t}")
        if t in budgets:
            endpoint[t] = {"A": abar.copy(), "B": bbar.copy(),
                           "last_A": a.copy(), "last_B": b.copy(),
                           "constructor_seconds_cpu_f64": time.perf_counter()-start,
                           "operator_calls": 2*t, "maximum_sample_tangent_norm_squared": max_norm}
    return endpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budgets", default="50,200,1000")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--shift", type=int, default=100)
    p.add_argument("--mu", type=float, default=.1)
    p.add_argument("--box", type=float, default=2.)
    args = p.parse_args()
    budgets = sorted(int(x) for x in args.budgets.split(","))
    seeds = [int(x) for x in args.seeds.split(",")]
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Authorized remote GPU required for downstream PDHG")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {"mu": args.mu, "M": args.box, "shift": args.shift,
                         "budgets": budgets, "seeds": seeds, "host": platform.node(),
                         "constructor_dtype_backend": "float64 CPU reference prototype",
                         "pdhg_dtype_backend": "float32 GPU",
                         "averaging": "weighted geodesic mean of pre-update iterates",
                         "gradient": "simultaneous conditional independent Rademacher samples",
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "oracle_checks": oracle_checks(args.mu, args.box), "records": []}
    if not output["oracle_checks"]["passed"]:
        args.output.write_text(json.dumps(output, indent=2)+"\n")
        raise AssertionError("Independent construction/oracle checks failed")
    for name, condition, angle in (("isotropic", 1., 0.), ("aligned100", 100., 0.),
                                    ("rotated10", 10., np.pi/4), ("rotated100", 100., np.pi/4)):
        q = rotation(angle)
        k = (q @ np.diag([1., 1/condition]) @ q.T).astype(np.float32).astype(np.float64)
        bdata = np.array([1., -.7], np.float32).astype(np.float64)
        alpha = float(np.float32(.01))
        _, _, exact = exact_reference(k, "full", args.mu, args.box)
        for seed in seeds:
            endpoints = constructor(lambda x: k @ x, lambda y: k.T @ y,
                                    seed, budgets, args.mu, args.box, args.shift)
            for budget in budgets:
                point = endpoints[budget]
                a, b = point.pop("A"), point.pop("B")
                last_a, last_b = point.pop("last_A"), point.pop("last_B")
                objective = potential(a,b,k,args.mu)
                row = {"case": name, "seed": seed, "budget": budget, **point,
                       "A": a.tolist(), "B": b.tolist(), "last_A": last_a.tolist(),
                       "last_B": last_b.tolist(), "objective": objective,
                       "exact_objective": exact["objective"],
                       "construction_objective_gap": objective-exact["objective"],
                       "last_construction_objective_gap": potential(last_a,last_b,k,args.mu)-exact["objective"],
                       "projected_gradient_norm": grad_mapping(a,b,k,args.mu,args.box,"full"),
                       "solve": run_measurement(k,bdata,a,b,alpha,1e-3,5000,warm_repeats=1)}
                output["records"].append(row)
                args.output.write_text(json.dumps(output, indent=2)+"\n")
                print(json.dumps({"case": name, "seed": seed, "budget": budget,
                                  "gap": row["construction_objective_gap"],
                                  "iterations": row["solve"]["iterations"],
                                  "accepted": row["solve"]["independent"]["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
