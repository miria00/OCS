"""Numerical conformance of the factored cone projections and applied metrics.

Specification: candidates/gate1b_20260921/SPEC_FROZEN.md section 3 (A), sha256
in SPEC_FROZEN.sha256. Every bar below is the frozen formula, derived there
from float32 error analysis. No bar was set from an observed discrepancy.

This file replaces tests/test_cone_automorphism.py::
test_factored_projections_are_bitwise_the_shipped_prox. That test asserted
bitwise equality of the jitted factored copy and the eager shipped classes on
every backend. On the KEVIN RTX 5090 it failed in 13 of 14 processes (XLA's
fusion autotuner picks, per process, an emitter that sums a 3-element norm in
a different order, and fixture point 688 is 0.43 ulp outside the cone), and on
the Mazeppa RTX 5090 it failed once in 5 at the PSD pin. The record is kept in
SPEC_FROZEN.md section 1 and in candidates/gate1_20260921/.

What is asserted here:
* A3: the two byte claims on the CPU device (the tested CPU configuration).
* A1, A2: on the default device, every path is within a derived bound of a
  float64 reference, every pair of paths is within twice that bound, the
  outputs meet cone feasibility, polar feasibility, complementarity and the
  proximal objective with stated bars, and every verdict outside a declared
  band equals the float64 verdict. Points inside the band form the boundary
  stratum. They stay in every count, meet every accuracy bar, and any verdict
  difference is recorded as "unresolved at float32".
* A4: applied metrics against float64 references.
* A5: the condition cap as a decision, against exact rational arithmetic.
Records go to the JUnit XML as testsuite properties "spec_record::<test>" and to stdout
(lines starting "SPEC-RECORD ").
"""
import json
import math
import os
import platform
from decimal import Decimal, getcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocs.prox import cone_metric as cm
from ocs.prox import cones

U = 2.0 ** -24                     # float32 unit roundoff (SPEC notation u)


# ======================================================================= utils
def _environment():
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except Exception:              # pragma: no cover
        jaxlib_version = "unknown"
    dev = jax.devices()[0]
    return {"backend": jax.default_backend(), "device_kind": getattr(dev, "device_kind", str(dev)),
            "jax": jax.__version__, "jaxlib": jaxlib_version,
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"), "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"), "machine": platform.machine()}


def _record(record_testsuite_property, name, payload):
    text = json.dumps({"test": name, "environment": _environment(), **payload},
                      sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    record_testsuite_property(f"spec_record::{name}", text)
    print("SPEC-RECORD " + text)


def _ratio_max(err, bound):
    err, bound = np.asarray(err, np.float64), np.asarray(bound, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(bound > 0, err / np.where(bound > 0, bound, 1.0), np.where(err == 0, 0.0, np.inf))
    return float(np.max(r)) if r.size else 0.0


# ================================================================== fixtures
def _original_fixture():
    """Exactly the draws of the replaced test: 1000 points in L^4, then 200
    random 3 x 3 matrices from the same stream (default_rng(6))."""
    rng = np.random.default_rng(6)
    k = 4
    pts = rng.standard_normal((1000, k)).astype(np.float32)
    pts[:300, 0] = np.linalg.norm(pts[:300, 1:], axis=1) + rng.uniform(0, 0.5, 300)       # inside
    pts[300:600, 0] = -np.linalg.norm(pts[300:600, 1:], axis=1) - rng.uniform(0, 0.5, 300)  # below
    pts[600:700, 0] = np.linalg.norm(pts[600:700, 1:], axis=1)                            # boundary
    mats = rng.standard_normal((200, 9)).astype(np.float32)
    return pts, mats


def _extra_soc(d, rng):
    n = 300
    P = rng.standard_normal((n, 1 + d)).astype(np.float32)
    nz = lambda A: np.linalg.norm(A[:, 1:], axis=1)
    P[:90, 0] = nz(P[:90]) + rng.uniform(0, 0.5, 90)
    P[90:180, 0] = -nz(P[90:180]) - rng.uniform(0, 0.5, 90)
    P[180:210, 0] = nz(P[180:210])
    if d == 16:
        Q = rng.standard_normal((30, 1 + d)).astype(np.float32)
        Q[:, 0] = -nz(Q)                                                    # polar boundary
        P = np.concatenate([P, Q])
    return P


def _soc_sets():
    pts, _ = _original_fixture()
    rng = np.random.default_rng(61)
    sets = [("original_d3", pts),
            ("original_d3_x1e-3", (pts * np.float32(1e-3)).astype(np.float32)),
            ("original_d3_x1e3", (pts * np.float32(1e3)).astype(np.float32))]
    for d in (1, 16):
        sets.append((f"extra_d{d}", _extra_soc(d, rng)))
    return sets


def _orth(rng, k):
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


def _psd_sets():
    _, mats = _original_fixture()
    rng = np.random.default_rng(62)
    sign = lambda: rng.choice([-1.0, 1.0])
    mag = lambda: rng.uniform(0.5, 2.0)

    def build(k, lams, skew=False):
        Q = _orth(rng, k)
        M = Q @ np.diag(lams) @ Q.T
        M = 0.5 * (M + M.T)
        if skew:
            G = rng.standard_normal((k, k))
            M = M + 0.25 * (G - G.T)
        return M.astype(np.float32).reshape(-1)

    boundary3 = [build(3, [0.0, sign() * mag(), sign() * mag()], skew=(i < 30)) for i in range(100)]
    repeated3 = []
    for _ in range(50):
        a = sign() * mag()
        repeated3.append(build(3, [a, a, sign() * mag()]))
    rank1 = [build(3, [mag(), 0.0, 0.0]) for _ in range(50)]
    random5 = [rng.standard_normal(25).astype(np.float32) for _ in range(100)]
    boundary5 = [build(5, [0.0] + [sign() * mag() for _ in range(4)]) for _ in range(30)]
    return [("original_k3", 3, mats),
            ("boundary_k3", 3, np.stack(boundary3)),
            ("repeated_k3", 3, np.stack(repeated3)),
            ("rank1_k3", 3, np.stack(rank1)),
            ("random_k5", 5, np.stack(random5)),
            ("boundary_k5", 5, np.stack(boundary5))]


# ============================================================= A1 references
def _soc_proj64(V):
    V = np.asarray(V, np.float64)
    s, z = V[:, 0], V[:, 1:]
    n = np.linalg.norm(z, axis=1)
    out = np.zeros_like(V)
    inside = n <= s
    below = n <= -s
    mid = ~inside & ~below
    a = 0.5 * (s + n)
    out[inside] = V[inside]
    out[mid, 0] = a[mid]
    out[mid, 1:] = (a[mid] / n[mid])[:, None] * z[mid]
    verdict = np.where(inside, "inside", np.where(below, "zero", "scaled"))
    return out, verdict, n


def _dist_soc(X):
    P, _, _ = _soc_proj64(X)
    return np.linalg.norm(np.asarray(X, np.float64) - P, axis=1)


def _soc_verdict(out32, in32):
    same = np.all(out32 == in32, axis=1)
    zero = np.all(out32 == 0, axis=1)
    return np.where(same, "inside", np.where(zero, "zero", "scaled"))


# ============================================================= A1 test
@pytest.mark.parametrize("name", ["original_d3", "original_d3_x1e-3", "original_d3_x1e3",
                                  "extra_d1", "extra_d16"])
def test_a1_soc_projection_conformance(name, record_testsuite_property):
    pts = dict(_soc_sets())[name]
    count, k = pts.shape
    d = k - 1
    soc = cones.SecondOrderCone()
    S1 = np.stack([np.asarray(soc.prox(jnp.asarray(p), 1.0)) for p in pts])
    layout = ("socbatch", count, d)
    flat = cm.unrows(layout, pts, np)
    batched = np.asarray(cones.SOCBatch(count, d).prox(jnp.asarray(flat), 1.0))
    S2 = np.asarray(cm.rows(layout, jnp.asarray(batched)))
    F = np.asarray(jax.jit(cm.soc_project)(jnp.asarray(pts)))
    paths = {"F": F, "S1": S1, "S2": S2}

    V = pts.astype(np.float64)
    vnorm = np.linalg.norm(V, axis=1)
    p64, ref_verdict, n64 = _soc_proj64(V)
    Bc = (d + 10) * U                                   # SPEC 3.1, B_soc(d)
    B = Bc * vnorm
    nu = (d / 2 + 2) * U
    boundary = (np.abs(n64 - V[:, 0]) <= 2 * nu * n64) | (np.abs(n64 + V[:, 0]) <= 2 * nu * n64)
    dist_v = np.linalg.norm(p64 - V, axis=1)

    failures, ratios, verdicts = [], {}, {}
    for key, P in paths.items():
        P64 = P.astype(np.float64)
        checks = {
            "accuracy": (np.linalg.norm(P64 - p64, axis=1), B),
            "feasibility": (_dist_soc(P64), B),
            "polar": (_dist_soc(P64 - V), B),
            "complementarity": (np.abs(np.sum(P64 * (P64 - V), axis=1)), (2 * Bc + Bc ** 2) * vnorm ** 2),
            "objective": (np.linalg.norm(P64 - V, axis=1), dist_v + B),
        }
        for cname, (err, bound) in checks.items():
            ratios[f"{key}.{cname}"] = _ratio_max(err if cname != "objective" else
                                                  np.maximum(err - dist_v, 0.0), bound if cname != "objective" else B)
            bad = np.nonzero(err > bound)[0]
            if bad.size:
                failures.append((key, cname, bad[:10].tolist(), int(bad.size)))
        verdicts[key] = _soc_verdict(P, pts)
    for a, b in (("F", "S1"), ("F", "S2"), ("S1", "S2")):
        err = np.linalg.norm(paths[a].astype(np.float64) - paths[b].astype(np.float64), axis=1)
        ratios[f"parity.{a}-{b}"] = _ratio_max(err, 2 * B)
        bad = np.nonzero(err > 2 * B)[0]
        if bad.size:
            failures.append((f"{a}-{b}", "parity", bad[:10].tolist(), int(bad.size)))

    interior_disagree, unresolved = [], []
    for i in range(count):
        vs = {key: str(verdicts[key][i]) for key in paths}
        agree = all(v == ref_verdict[i] for v in vs.values())
        if agree:
            continue
        if boundary[i]:
            unresolved.append({"index": i, "input": V[i].tolist(), "verdicts": vs,
                               "reference_verdict": str(ref_verdict[i]),
                               "outputs": {key: paths[key][i].astype(np.float64).tolist() for key in paths},
                               "reference_output": p64[i].tolist(),
                               "signed_gap_n_minus_s": float(n64[i] - V[i, 0]),
                               "status": "unresolved at float32"})
        else:
            interior_disagree.append({"index": i, "verdicts": vs, "reference_verdict": str(ref_verdict[i])})

    # membership oracle consistency (policy ||z|| <= s + 1e-5, absolute)
    oracle_f = jax.jit(jax.vmap(soc.value))
    oracle_bad, oracle_band_disagree, oracle_inf = [], 0, {}
    for key, P in paths.items():
        oracle = np.asarray(oracle_f(jnp.asarray(P)))
        P64 = P.astype(np.float64)
        zn = np.linalg.norm(P64[:, 1:], axis=1)
        policy = zn <= P64[:, 0] + 1e-5
        oracle_band = np.abs(zn - P64[:, 0] - 1e-5) <= nu * zn
        oracle_bad += [(key, int(i)) for i in np.nonzero(((oracle == 0) != policy) & ~oracle_band)[0]]
        oracle_band_disagree += int(np.sum(((oracle == 0) != policy) & oracle_band))
        oracle_inf[key] = int(np.sum(oracle != 0))

    _record(record_testsuite_property, f"A1[{name}]", {
        "points": count, "d": d, "B_soc_over_u": d + 10, "band_over_u_times_n": 2 * (d / 2 + 2),
        "strata": {"interior": int(np.sum(~boundary)), "boundary": int(np.sum(boundary))},
        "max_error_over_bound": ratios,
        "interior_verdict_disagreements": interior_disagree,
        "boundary_unresolved_at_float32": unresolved,
        "boundary_unresolved_count": len(unresolved),
        "oracle": {"disagreements_outside_band": oracle_bad,
                   "disagreements_inside_band": oracle_band_disagree,
                   "reports_inf": oracle_inf},
        "failures": failures})
    assert not failures, failures
    assert not interior_disagree, interior_disagree
    assert not oracle_bad, oracle_bad[:10]


# ============================================================= A2 test
def _psd_ref(m, k):
    M = np.asarray(m, np.float64).reshape(k, k)
    Ms = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(Ms)
    return Ms, (V * np.maximum(w, 0.0)) @ V.T, w


def _neg_part(X):
    return float(np.linalg.norm(np.minimum(np.linalg.eigvalsh(0.5 * (X + X.T)), 0.0)))


@pytest.mark.parametrize("name", ["original_k3", "boundary_k3", "repeated_k3", "rank1_k3",
                                  "random_k5", "boundary_k5"])
def test_a2_psd_projection_conformance(name, record_testsuite_property):
    entry = {n: (k, m) for n, k, m in _psd_sets()}[name]
    k, mats = entry
    count = mats.shape[0]
    psd = cones.PSDCone(k)
    P1 = np.stack([np.asarray(psd.prox(jnp.asarray(m), 1.0)) for m in mats])
    F1 = np.stack([np.asarray(cm.psd_project(jnp.asarray(m).reshape(1, k, k))).reshape(-1) for m in mats])
    F2 = np.asarray(jax.jit(cm.psd_project)(jnp.asarray(mats).reshape(count, k, k))).reshape(count, -1)
    paths = {"P1": P1, "F1": F1, "F2": F2}
    Bc = math.ceil(12 * k ** 2 + math.sqrt(k) * (12 * k ** 2 + 2 * k + 3) + 1) * U   # SPEC 3.2
    failures, ratios = [], {}
    worst = {}
    interior_disagree, unresolved = [], []
    boundary_count = 0
    oracle_bad, oracle_band_disagree = [], 0
    oracle_f = jax.jit(jax.vmap(psd.value))
    oracle = {key: np.asarray(oracle_f(jnp.asarray(P))) for key, P in paths.items()}

    def note(key, err, bound, i):
        r = _ratio_max(err, bound)
        worst[key] = max(worst.get(key, 0.0), r)
        if err > bound:
            failures.append((key, i, float(err), float(bound)))

    for i in range(count):
        Ms, P64, w = _psd_ref(mats[i], k)
        sn = float(np.linalg.norm(Ms))
        B = Bc * sn
        in_band = bool(np.any(np.abs(w) <= 2 * B))
        boundary_count += in_band
        ref_rank = int(np.sum(w > 0))
        ranks = {}
        for key, P in paths.items():
            Ph = P[i].astype(np.float64).reshape(k, k)
            X = 0.5 * (Ph + Ph.T)
            note(f"{key}.accuracy", float(np.linalg.norm(Ph - P64)), B, i)
            note(f"{key}.symmetry", float(np.linalg.norm(Ph - Ph.T)) / 2, B, i)
            note(f"{key}.feasibility", _neg_part(X), B, i)
            note(f"{key}.polar", _neg_part(X - Ms), B, i)
            note(f"{key}.complementarity", abs(float(np.sum(X * (X - Ms)))), (2 * Bc + Bc ** 2) * sn ** 2, i)
            excess = max(float(np.linalg.norm(X - Ms)) - float(np.linalg.norm(P64 - Ms)), 0.0)
            note(f"{key}.objective", excess, B, i)
            lam = np.linalg.eigvalsh(X)
            ranks[key] = int(np.sum(lam > B))
            # membership oracle (policy lam_min(sym) >= -1e-4, absolute)
            policy = lam.min() >= -1e-4
            if (oracle[key][i] == 0) != policy:
                if abs(lam.min() + 1e-4) <= B:
                    oracle_band_disagree += 1
                else:
                    oracle_bad.append((key, i))
        for a, b in (("F1", "P1"), ("F2", "P1"), ("F1", "F2")):
            err = float(np.linalg.norm(paths[a][i].astype(np.float64) - paths[b][i].astype(np.float64)))
            note(f"parity.{a}-{b}", err, 2 * B, i)
        if any(r != ref_rank for r in ranks.values()):
            item = {"index": i, "ranks": ranks, "reference_rank": ref_rank,
                    "reference_eigenvalues": w.tolist(), "scale_frobenius": sn}
            if in_band:
                item.update({"status": "unresolved at float32",
                             "outputs": {key: paths[key][i].astype(np.float64).tolist() for key in paths},
                             "reference_output": P64.reshape(-1).tolist()})
                unresolved.append(item)
            else:
                interior_disagree.append(item)

    _record(record_testsuite_property, f"A2[{name}]", {
        "matrices": count, "k": k, "B_psd_over_u": round(Bc / U),
        "strata": {"interior": count - boundary_count, "boundary": boundary_count},
        "max_error_over_bound": worst,
        "interior_verdict_disagreements": interior_disagree,
        "boundary_unresolved_at_float32": unresolved,
        "boundary_unresolved_count": len(unresolved),
        "oracle": {"disagreements_outside_band": oracle_bad,
                   "disagreements_inside_band": oracle_band_disagree},
        "failures": failures[:20], "failure_count": len(failures)})
    assert not failures, failures[:20]
    assert not interior_disagree, interior_disagree
    assert not oracle_bad, oracle_bad


# ============================================================= A3 test
_TESTED_CPU = {"jax": "0.11.1", "jaxlib": "0.11.1", "machine": "x86_64"}


def test_a3_cpu_byte_parity_under_the_tested_cpu_configuration(record_testsuite_property):
    pts, mats = _original_fixture()
    cpu = jax.devices("cpu")[0]
    soc, psd = cones.SecondOrderCone(), cones.PSDCone(3)
    with jax.default_device(cpu):
        shipped = np.stack([np.asarray(soc.prox(jnp.asarray(p), 1.0)) for p in pts])
        factored = np.asarray(jax.jit(cm.soc_project)(jnp.asarray(pts)))
        shipped_psd = np.stack([np.asarray(psd.prox(jnp.asarray(m), 1.0)) for m in mats])
        factored_psd = np.stack([np.asarray(cm.psd_project(jnp.asarray(m).reshape(1, 3, 3))).reshape(-1)
                                 for m in mats])
    env = _environment()
    tested = (env["jax"] == _TESTED_CPU["jax"] and env["jaxlib"] == _TESTED_CPU["jaxlib"]
              and env["machine"] == _TESTED_CPU["machine"] and env["JAX_PLATFORMS"] == "cpu"
              and env["XLA_FLAGS"] in (None, "", "--xla_force_host_platform_device_count=1")
              and env["JAX_ENABLE_X64"] in (None, "", "0", "false", "False"))
    soc_equal = factored.tobytes() == shipped.tobytes()
    psd_equal = factored_psd.tobytes() == shipped_psd.tobytes()
    _record(record_testsuite_property, "A3", {
        "in_tested_configuration": tested, "cpu_device": str(cpu),
        "soc_bitwise_equal": soc_equal, "psd_bitwise_equal": psd_equal,
        "note": "a failure outside the tested configuration means the CPU byte claim does "
                "not carry over to that configuration. It is not widened or relabelled"})
    assert soc_equal
    assert psd_equal


# ============================================================= A4 test
def _soc_element(ratio, scale, k, rng):
    lam2, lam1 = scale, ratio * scale
    direction = rng.standard_normal(k - 1)
    direction /= np.linalg.norm(direction)
    return np.concatenate([[0.5 * (lam1 + lam2)], 0.5 * (lam1 - lam2) * direction]).astype(np.float32)


def test_a4_applied_metrics_against_float64(record_testsuite_property):
    rng = np.random.default_rng(64)
    worst, failures, cases = {}, [], 0
    jit_soc, jit_psd = jax.jit(cm.apply_soc), jax.jit(cm.apply_psd)
    for k in (3, 4, 9):
        J = -np.eye(k)
        J[0, 0] = 1.0
        for ratio in (1.0, 3.0, 10.0 * (1 - 1e-9)):
            for scale in (1e-3, 1.0, 1e3):
                a = _soc_element(ratio, scale, k, rng)
                a64 = a.astype(np.float64)
                det = np.float32(a64[0] ** 2 - a64[1:] @ a64[1:])
                R = rng.standard_normal((40, k)).astype(np.float32)
                R64 = R.astype(np.float64)
                ref = 2.0 * np.outer(R64 @ a64, a64) - float(det) * (R64 @ J)
                bound = ((2 * k + 4) * (a64 @ a64) + 2 * abs(float(det))) * U * np.linalg.norm(R64, axis=1)
                outs = {"eager": np.asarray(cm.apply_soc(jnp.asarray(a), jnp.asarray(det), jnp.asarray(R))),
                        "jit": np.asarray(jit_soc(jnp.asarray(a), jnp.asarray(det), jnp.asarray(R)))}
                for key, out in outs.items():
                    err = np.linalg.norm(out.astype(np.float64) - ref, axis=1)
                    worst[f"soc.{key}"] = max(worst.get(f"soc.{key}", 0.0), _ratio_max(err, bound))
                    if np.any(err > bound):
                        failures.append(("soc", key, k, ratio, scale))
                err = np.linalg.norm(outs["eager"].astype(np.float64) - outs["jit"].astype(np.float64), axis=1)
                worst["soc.parity"] = max(worst.get("soc.parity", 0.0), _ratio_max(err, 2 * bound))
                if np.any(err > 2 * bound):
                    failures.append(("soc", "parity", k, ratio, scale))
                cases += 1
    for k in (2, 3, 5):
        for cond in (1.0, 3.0, 10.0):
            for scale in (1e-3, 1.0, 1e3):
                lams = scale * np.geomspace(1.0, cond, k)
                Q = _orth(rng, k)
                A = (Q * lams) @ Q.T
                A = np.triu(A) + np.triu(A, 1).T
                A = A.astype(np.float32)
                A = np.triu(A) + np.triu(A, 1).T                   # exactly symmetric float32
                A64 = A.astype(np.float64)
                for j in range(20):
                    R = rng.standard_normal((k, k))
                    if j < 10:
                        R = 0.5 * (R + R.T)
                    R = R.astype(np.float32)
                    R64 = R.astype(np.float64)
                    ref = A64 @ R64 @ A64
                    bound = (2 * k + 1) * U * np.linalg.norm(A64) ** 2 * np.linalg.norm(R64)
                    outs = {"eager": np.asarray(cm.apply_psd(jnp.asarray(A), jnp.asarray(R))),
                            "jit": np.asarray(jit_psd(jnp.asarray(A), jnp.asarray(R)))}
                    for key, out in outs.items():
                        err = float(np.linalg.norm(out.astype(np.float64) - ref))
                        worst[f"psd.{key}"] = max(worst.get(f"psd.{key}", 0.0), _ratio_max(err, bound))
                        if err > bound:
                            failures.append(("psd", key, k, cond, scale, j))
                    err = float(np.linalg.norm(outs["eager"].astype(np.float64) - outs["jit"].astype(np.float64)))
                    worst["psd.parity"] = max(worst.get("psd.parity", 0.0), _ratio_max(err, 2 * bound))
                    if err > 2 * bound:
                        failures.append(("psd", "parity", k, cond, scale, j))
                    cases += 1
    _record(record_testsuite_property, "A4", {"cases": cases, "max_error_over_bound": worst,
                                    "failures": failures})
    assert not failures, failures


# ============================================================= A5 test
DELTAS = (-1e-3, -1e-9, -1e-13, -1e-15, 0.0, 1e-15, 1e-13, 1e-9, 1e-3)
CAP_BAND = 1e-12                                          # SPEC 3.5


def _constructor_verdict(algebra, element):
    try:
        cm.ConeAutomorphism(algebra, element)
    except ValueError as exc:
        if "condition number at most" in str(exc):
            return "refused", str(exc)[:80]
        raise
    return "accepted", ""


def test_a5_condition_cap_decision_against_exact_arithmetic(record_testsuite_property):
    getcontext().prec = 60
    rng = np.random.default_rng(65)
    rows, interior_bad, unresolved = [], [], []
    for k in (2, 3, 9):
        direction = np.ones(1) if k == 2 else rng.standard_normal(k - 1)
        direction = direction / np.linalg.norm(direction)
        for delta in DELTAS:
            lam2, lam1 = 1.0, 10.0 * (1.0 + delta)
            w = np.concatenate([[0.5 * (lam1 + lam2)], 0.5 * (lam1 - lam2) * direction])
            w0 = Fraction(float(w[0]))
            nb2 = sum(Fraction(float(x)) ** 2 for x in w[1:])
            exact_accept = 121 * nb2 <= 81 * w0 * w0
            to_dec = lambda f: Decimal(f.numerator) / Decimal(f.denominator)
            nb = to_dec(nb2).sqrt()
            cond = ((to_dec(w0) + nb) / (to_dec(w0) - nb)) ** 2
            gap = float(abs(cond / Decimal(100) - 1))
            verdict, _msg = _constructor_verdict("soc", w)
            row = {"algebra": "soc", "k": k, "delta": delta, "cond_exact_minus_100": float(cond - 100),
                   "relative_gap": gap, "exact": "accepted" if exact_accept else "refused",
                   "implementation": verdict, "stratum": "boundary" if gap <= CAP_BAND else "interior"}
            rows.append(row)
            if row["implementation"] != row["exact"]:
                (unresolved if gap <= CAP_BAND else interior_bad).append(row)
    for k in (2, 3, 5):
        perm = rng.permutation(k)
        for delta in DELTAS:
            lam_min, lam_max = 1.0, 10.0 * (1.0 + delta)
            diag = np.linspace(lam_min, lam_max, k)
            diag[0], diag[-1] = lam_min, lam_max
            S = np.diag(diag)[perm][:, perm]
            exact_accept = Fraction(lam_max) <= 10 * Fraction(lam_min)
            gap = abs((Fraction(lam_max) / Fraction(lam_min)) ** 2 / 100 - 1)
            gap = float(gap)
            verdict, _msg = _constructor_verdict("psd", S)
            row = {"algebra": "psd", "k": k, "delta": delta, "relative_gap": gap,
                   "exact": "accepted" if exact_accept else "refused", "implementation": verdict,
                   "stratum": "boundary" if gap <= CAP_BAND else "interior"}
            rows.append(row)
            if row["implementation"] != row["exact"]:
                (unresolved if gap <= CAP_BAND else interior_bad).append(row)
    for row in unresolved:
        row["status"] = "unresolved at float64 evaluation"
    _record(record_testsuite_property, "A5", {
        "cases": len(rows), "strata": {"interior": sum(r["stratum"] == "interior" for r in rows),
                                       "boundary": sum(r["stratum"] == "boundary" for r in rows)},
        "rows": rows, "interior_disagreements": interior_bad, "boundary_unresolved": unresolved,
        "not_covered": "rotated (non-diagonal) PSD elements have no exact reference here"})
    assert not interior_bad, interior_bad
