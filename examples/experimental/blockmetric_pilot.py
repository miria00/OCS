"""Block-metric pilot: do small full block metrics on coupled Euclidean-norm
blocks, built from operator products only and frozen during the solve, reduce
TOTAL time to a common certified accuracy in preconditioned PDHG?

Research loop outside the package.  The shipped engine (ocs/) is untouched: it
is one arm (A0) and it supplies the only acceptance test, the compiled
family's public certificate  CompiledFamily.certificate(x, y, data)  whose
max(r_primal, r_dual) <= tol decides every arm.

Metric contract (paper, "Admissible metrics" / Moreau in a metric):
    prox_f^D(v) = argmin_u f(u) + 1/2 (u-v)^T D (u-v)
    prox_{f^*}^{D^{-1}}(v) = v - D prox_f^D(D^{-1} v)              (generalized Moreau)
Dual block i carries metric Sigma_i (scalar | diagonal | full SPD), the primal
metric T is the shipped diagonal, and the fixed-step iteration is
    y_i <- prox_{f_i^*}^{Sigma_i^{-1}}(y_i + Sigma_i (K_i xbar + l_i))
    x   <- prox_g^{T^{-1}}(x - T K^T y),   xbar <- 2 x - x_old
with ||Sigma^{1/2} K T^{1/2}||_2 < 1 enforced by one global factor.  Plain
PDHG: no anchor, no restart, no adaptive weight.

Arms
  A0    shipped engine, default path (Halpern + restarts), tol 1e-5
  A0std shipped engine, pdhg_variant="standard" (loop-equality reference)
  A1    this loop with the shipped scalar/tied metric (loop vs metric)
  A2    per-row diagonal metric from the block Gram diagonal (eta = 0)
  A3    full block metric from EXACT Grams (eta = 1)
  A3s   full block metric from SKETCHED Grams, q in {k, 4k, 16k} probes
  A4    Liu-Xu-Yin iPrePDHG: M1 = I/tau, M2 = tau K K^T by products, p inner
        proximal-gradient steps with closed-form ball projection
  A5    (n <= 400 only) optimal two-sided diagonal reference by an SDP
        (cvxpy + Clarabel on the formed small matrix; reference only)

Usage (always on a remote GPU host; see the README in the profiling folder):
  python blockmetric_pilot.py --validate
  python blockmetric_pilot.py --family F1 --n 4000 --rho 0.99 --loopcheck
  python blockmetric_pilot.py --family F1 --n 200000 --rho 0.99 --arm A3 --out results.csv
"""
from __future__ import annotations

import os, sys, time, json, csv, math, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp

csv.field_size_limit(10 ** 9)      # the residual history is a long field; never let a read of it kill a run


def _source_hash():
    """Commit-equivalent hash of the prototype source actually running."""
    import hashlib
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(("blockmetric_pilot.py", "blockmetric_models.py")):
        p = os.path.join(here, name)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

NEWTON_ITERS = int(os.environ.get("BM_NEWTON_ITERS", 20))   # safeguarded Newton/bisection iterations (diagonal-metric l2 prox)
NEWTON_UNROLL = os.environ.get("BM_NEWTON_UNROLL", "1") == "1"   # unroll the Newton loop (kernel fusion)
GAMMA_SAFETY = 0.9         # global step factor applied to the power-iteration norm estimate
POWER_ITERS = 50           # power iterations for ||Sigma^{1/2} K T^{1/2}||
KAPPA_MAX = 100.0          # eigenvalue clip of R_i to [1/kappa, kappa]
DELTA_FRAC = 1e-3          # delta = DELTA_FRAC * tr(W)/k  ridge inside W_i


# ============================================================================ prox routines
def prox_l2_diag(v, d, lam):
    """prox of lam*||z||_2 under the diagonal metric diag(d), batched over rows.

    v, d: (B, k); lam: scalar or (B,) or (B,1).  Zero iff ||d*v||_2 <= lam;
    otherwise z_j = d_j v_j r / (d_j r + lam) with r = ||z|| the root of
    phi(r) = sum_j (d_j v_j / (d_j r + lam))^2 - 1 on (0, ||v||].  The root is
    found by Newton on the secular form  g(r) = 1 / ||q(r)||_2 - 1,
    q_j = d_j v_j / (d_j r + lam),  which is increasing and concave on [0, ||v||]
    (exactly linear when d is uniform), started at r = 0 so the iterates increase
    monotonically to the root, inside a bracket safeguard (bisection whenever a
    step would leave the bracket).  NEWTON_ITERS fixed steps; validated against
    float64 KKT solves (see the pilot README for the measured accuracy).
    """
    lam = jnp.reshape(jnp.asarray(lam, v.dtype), (-1, 1)) if jnp.ndim(lam) else lam
    dv = d * v
    active = jnp.sum(dv * dv, axis=1, keepdims=True) > lam * lam
    hi = jnp.sqrt(jnp.sum(v * v, axis=1, keepdims=True))
    lo = jnp.zeros_like(hi)
    r = jnp.zeros_like(hi)

    def secular(r):
        den = d * r + lam
        q = dv / den
        n2 = jnp.sum(q * q, axis=1, keepdims=True)
        n = jnp.sqrt(n2)
        g = 1.0 / jnp.maximum(n, 1e-30) - 1.0                       # < 0 left of the root
        gp = jnp.sum(q * q * d / den, axis=1, keepdims=True) / jnp.maximum(n2 * n, 1e-30)
        return g, gp

    def body(_, st):
        lo, hi, r = st
        g, gp = secular(r)
        lo = jnp.where(g < 0, r, lo)
        hi = jnp.where(g < 0, hi, r)
        rn = r - g / jnp.maximum(gp, 1e-30)
        inside = (rn > lo) & (rn < hi)
        r = jnp.where(inside, rn, 0.5 * (lo + hi))
        return lo, hi, r

    lo, hi, r = jax.lax.fori_loop(0, NEWTON_ITERS, body, (lo, hi, r), unroll=NEWTON_UNROLL)
    z = dv * r / (d * r + lam)
    return jnp.where(active, z, 0.0)


PROX_IMPL = os.environ.get("BM_PROX_IMPL", "array")   # "array" | "components"


def prox_l2_diag_components(vs, ds, lam):
    """Same prox as prox_l2_diag, written over k separate (B,) arrays.

    prox_l2_diag reduces along the block axis twice per Newton step
    (sum_j q_j^2 and the derivative sum).  A reduction is a fusion boundary in
    XLA, so the (B,k) intermediates are materialized once per step and the cost
    scales as one memory pass PER NEWTON STEP.  With k static and tiny, the same
    arithmetic written as explicit sums over k component arrays contains no
    reduction at all: every operation is elementwise on (B,) arrays, so the
    whole unrolled Newton solve can fuse into one kernel that keeps the working
    set in registers across steps.  Mathematically identical, validated against
    the same float64 KKT references.
    """
    k = len(vs)
    dvs = tuple(d * v for d, v in zip(ds, vs))
    nrm2_dv = sum(dv * dv for dv in dvs)
    active = nrm2_dv > lam * lam
    hi = jnp.sqrt(sum(v * v for v in vs))
    lo = jnp.zeros_like(hi)
    r = jnp.zeros_like(hi)
    tiny = jnp.asarray(1e-30, hi.dtype)
    for _ in range(NEWTON_ITERS):                  # python loop: fully unrolled, no reductions
        dens = tuple(d * r + lam for d in ds)
        qs = tuple(dv / den for dv, den in zip(dvs, dens))
        n2 = sum(q * q for q in qs)
        n = jnp.sqrt(n2)
        g = 1.0 / jnp.maximum(n, tiny) - 1.0
        gp = sum(q * q * d / den for q, d, den in zip(qs, ds, dens)) / jnp.maximum(n2 * n, tiny)
        lo = jnp.where(g < 0, r, lo)
        hi = jnp.where(g < 0, hi, r)
        rn = r - g / jnp.maximum(gp, tiny)
        inside = (rn > lo) & (rn < hi)
        r = jnp.where(inside, rn, 0.5 * (lo + hi))
    zs = tuple(dv * r / (d * r + lam) for dv, d in zip(dvs, ds))
    return tuple(jnp.where(active, z, 0.0) for z in zs)


def _split_k(a):
    return tuple(a[:, j] for j in range(a.shape[1]))


def _stack_k(parts):
    return jnp.stack(parts, axis=1)


def prox_l2_diag_any(v, d, lam):
    """Dispatch to whichever implementation BM_PROX_IMPL selects (same math)."""
    if PROX_IMPL == "components":
        return _stack_k(prox_l2_diag_components(_split_k(v), _split_k(d), lam))
    return prox_l2_diag(v, d, lam)


def prox_l2_full(v, Q, w, lam):
    """prox of lam*||z||_2 under M = Q diag(w) Q^T (batched):  Q prox^{diag(w)}(Q^T v)."""
    vt = jnp.einsum("bji,bj->bi", Q, v)
    zt = prox_l2_diag(vt, w, lam)
    return jnp.einsum("bij,bj->bi", Q, zt)


def moreau_dual_diag(v, d, lam):
    """prox_{f^*}^{diag(d)^{-1}}(v) = v - d * prox_f^{diag(d)}(v / d),  f = lam||.||_2."""
    return v - d * prox_l2_diag(v / d, d, lam)


def moreau_dual_full(v, Q, w, lam):
    """prox_{f^*}^{M^{-1}}(v) = v - M prox_f^{M}(M^{-1} v),  M = Q diag(w) Q^T."""
    Minv_v = jnp.einsum("bij,bj->bi", Q, jnp.einsum("bji,bj->bi", Q, v) / w)
    p = prox_l2_full(Minv_v, Q, w, lam)
    return v - jnp.einsum("bij,bj->bi", Q, w * jnp.einsum("bji,bj->bi", Q, p))


def ball_project(v, lam):
    """Euclidean projection onto {||y||_2 <= lam} per row (scalar-metric closed form)."""
    nrm = jnp.sqrt(jnp.sum(v * v, axis=1, keepdims=True))
    return v * jnp.minimum(1.0, lam / jnp.maximum(nrm, 1e-30))


# ============================================================================ validation
def _cvx_solve(prob):
    import cvxpy as cp
    for kw in (dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=500), {}):
        try:
            prob.solve(solver="CLARABEL", **kw)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status
        except cp.error.SolverError:
            pass
    prob.solve(solver="SCS", eps=1e-10, max_iters=200000)
    return prob.status


def _brute_prox_cvxpy(v, D, lam):
    """argmin lam||z|| + 1/2 (z-v)^T D (z-v) by cvxpy.  The problem is solved in
    normalized units (||v|| = 1, max diag = 1) using positive homogeneity:
    prox^{cD}_{lam}(s u) = s prox^{D}_{lam/(c s)}(u)."""
    import cvxpy as cp
    s = max(np.linalg.norm(v), 1e-300); c = np.max(np.diag(D))
    z = cp.Variable(v.shape[0])
    obj = (lam / (c * s)) * cp.norm(z, 2) + 0.5 * cp.quad_form(z - v / s, cp.psd_wrap(D / c))
    _cvx_solve(cp.Problem(cp.Minimize(obj)))
    return s * np.asarray(z.value, np.float64)


def _brute_prox_scalar_eq(v, d, lam):
    """float64 KKT: brentq on phi(r) (independent of the jax Newton path)."""
    from scipy.optimize import brentq
    dv = d * v
    if np.linalg.norm(dv) <= lam:
        return np.zeros_like(v)
    phi = lambda r: np.sum((dv / (d * r + lam)) ** 2) - 1.0
    r = brentq(phi, 0.0, np.linalg.norm(v) * (1 + 1e-12), xtol=1e-15, rtol=1e-14, maxiter=500)
    return dv * r / (d * r + lam)


def _brute_dual_projection(v, M, lam):
    """prox_{f^*}^{M^{-1}}(v): projection of v onto the lam-ball in the M^{-1} metric
    (cvxpy, normalized units: proj_lam(s u) = s proj_{lam/s}(u), metric scale free)."""
    import cvxpy as cp
    s = max(np.linalg.norm(v), 1e-300)
    Minv = np.linalg.inv(M); Minv = 0.5 * (Minv + Minv.T); Minv = Minv / np.max(np.abs(Minv))
    y = cp.Variable(v.shape[0])
    obj = 0.5 * cp.quad_form(y - v / s, cp.psd_wrap(Minv))
    _cvx_solve(cp.Problem(cp.Minimize(obj), [cp.norm(y, 2) <= lam / s]))
    return s * np.asarray(y.value, np.float64)


def _brute_prox_full_kkt(v, M, lam):
    """float64 KKT for prox^M of lam||.||: M(z-v) + lam z/||z|| = 0, so
    z(r) = (M + (lam/r) I)^{-1} M v with r = ||z|| the root of ||z(r)|| - r (brentq)."""
    from scipy.optimize import brentq
    Mv = M @ v
    if np.linalg.norm(v) == 0 or np.linalg.norm(np.linalg.solve(M, Mv)) == 0:
        return np.zeros_like(v)
    w, Q = np.linalg.eigh(M)
    if np.linalg.norm(Mv) <= lam:                         # 0 optimal iff ||M v||_2 <= lam
        return np.zeros_like(v)
    z = lambda r: Q @ ((Q.T @ Mv) / (w + lam / r))
    g = lambda r: np.linalg.norm(z(r)) - r
    lo = 1e-14 * np.linalg.norm(v)
    r = brentq(g, lo, np.linalg.norm(v) * (1 + 1e-12), xtol=1e-300, rtol=1e-15, maxiter=1000)
    return z(r)


def _brute_dual_projection_kkt(v, M, lam):
    """float64 KKT for the projection of v onto {||y|| <= lam} in the M^{-1} metric:
    y = (I + nu M)^{-1} v with nu >= 0 the root of ||y(nu)|| = lam (brentq), no Moreau."""
    from scipy.optimize import brentq
    if np.linalg.norm(v) <= lam:
        return v.copy()
    w, Q = np.linalg.eigh(M)
    y = lambda nu: Q @ ((Q.T @ v) / (1.0 + nu * w))
    g = lambda nu: np.linalg.norm(y(nu)) - lam
    hi = 1.0
    while g(hi) > 0:
        hi *= 10.0
    nu = brentq(g, 0.0, hi, xtol=1e-300, rtol=1e-15, maxiter=1000)
    return y(nu)


def validate(cases=300, seed=0, log=print, dlog=3.0):
    """Prox (diag + full) against cvxpy/Clarabel and a float64 brentq KKT solve,
    plus the generalized Moreau identity against a direct metric projection."""
    rng = np.random.default_rng(seed)
    x64 = jax.config.read("jax_enable_x64")
    dtype = jnp.float64 if x64 else jnp.float32
    rows = []
    worst = dict(diag_vs_cvx=0.0, diag_vs_kkt=0.0, full_vs_cvx=0.0, full_vs_kkt=0.0, moreau_diag=0.0, moreau_full=0.0,
                 moreau_full_rotated=0.0, moreau_diag_vs_kkt=0.0, moreau_full_rotated_vs_kkt=0.0,
                 components_vs_kkt=0.0)
    for c in range(cases):
        k = int(rng.integers(3, 9))
        lam = float(np.exp(rng.uniform(np.log(0.1), np.log(10))))
        d = np.exp(rng.uniform(-dlog * np.log(10), dlog * np.log(10), size=k))   # metric spread 10^(2 dlog)
        kind = c % 4
        v = rng.normal(size=k)
        if kind == 1:                       # near-zero solution: ||D v|| just above lam
            v = v / np.linalg.norm(d * v) * lam * (1 + 10 ** rng.uniform(-6, -2))
        elif kind == 2:                     # inactive: ||D v|| below lam
            v = v / np.linalg.norm(d * v) * lam * rng.uniform(0.2, 0.999)
        elif kind == 3:                     # large v
            v = v * 1e3
        # diagonal metric
        z_ref = _brute_prox_cvxpy(v, np.diag(d), lam)
        z_kkt = _brute_prox_scalar_eq(v, d, lam)
        z_jax = np.asarray(prox_l2_diag(jnp.asarray(v[None], dtype), jnp.asarray(d[None], dtype), lam))[0]
        sc = max(1.0, np.linalg.norm(v))
        e1 = np.linalg.norm(z_jax - z_ref) / sc; e2 = np.linalg.norm(z_jax - z_kkt) / sc
        # full metric  M = Q diag(w) Q^T
        Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
        M = Q @ np.diag(d) @ Q.T
        zf_ref = _brute_prox_cvxpy(v, M, lam)
        zf_jax = np.asarray(prox_l2_full(jnp.asarray(v[None], dtype), jnp.asarray(Q[None], dtype),
                                         jnp.asarray(d[None], dtype), lam))[0]
        e3 = np.linalg.norm(zf_jax - zf_ref) / sc
        e3k = np.linalg.norm(zf_jax - _brute_prox_full_kkt(v, M, lam)) / sc
        # Moreau identity in the metric: dual prox by Moreau vs direct metric projection
        yd_ref = _brute_dual_projection(v, np.diag(d), lam)
        yd_jax = np.asarray(moreau_dual_diag(jnp.asarray(v[None], dtype), jnp.asarray(d[None], dtype), lam))[0]
        yf_ref = _brute_dual_projection(v, M, lam)
        yf_jax = np.asarray(moreau_dual_full(jnp.asarray(v[None], dtype), jnp.asarray(Q[None], dtype),
                                             jnp.asarray(d[None], dtype), lam))[0]
        e4 = np.linalg.norm(yd_jax - yd_ref) / max(1.0, np.linalg.norm(v))
        e5 = np.linalg.norm(yf_jax - yf_ref) / max(1.0, np.linalg.norm(v))
        # componentwise implementation (the fusion-enabling rewrite) vs the same KKT solve
        zc = np.asarray(_stack_k(prox_l2_diag_components(
            _split_k(jnp.asarray(v[None], dtype)), _split_k(jnp.asarray(d[None], dtype)), lam)))[0]
        e7 = np.linalg.norm(zc - z_kkt) / sc
        # the rotated form the loop actually runs: ytilde = Q^T v - w*prox^{diag w}(Q^T v / w), y = Q ytilde
        Qj, wj = jnp.asarray(Q[None], dtype), jnp.asarray(d[None], dtype)
        vt = jnp.einsum("bji,bj->bi", Qj, jnp.asarray(v[None], dtype))
        yr = np.asarray(jnp.einsum("bij,bj->bi", Qj, vt - wj * prox_l2_diag(vt / wj, wj, lam)))[0]
        e6 = np.linalg.norm(yr - yf_ref) / max(1.0, np.linalg.norm(v))
        e4k = np.linalg.norm(yd_jax - _brute_dual_projection_kkt(v, np.diag(d), lam)) / max(1.0, np.linalg.norm(v))
        e6k = np.linalg.norm(yr - _brute_dual_projection_kkt(v, M, lam)) / max(1.0, np.linalg.norm(v))
        rows.append(dict(case=c, k=k, lam=lam, kind=["random", "near_zero", "inactive", "large"][kind],
                         dmin=d.min(), dmax=d.max(), diag_vs_cvx=e1, diag_vs_kkt=e2, full_vs_cvx=e3, full_vs_kkt=e3k,
                         moreau_diag=e4, moreau_full=e5, moreau_full_rotated=e6, moreau_diag_vs_kkt=e4k,
                         moreau_full_rotated_vs_kkt=e6k, components_vs_kkt=e7, znorm=np.linalg.norm(z_ref)))
        for key, val in zip(worst, (e1, e2, e3, e3k, e4, e5, e6, e4k, e6k, e7)):
            worst[key] = max(worst[key], float(val))
    log(f"validation ({cases} cases, metric entries in [1e-{dlog:g}, 1e{dlog:g}], jax dtype {dtype.__name__ if hasattr(dtype,'__name__') else dtype}): "
        f"worst relative errors {json.dumps({k: f'{v:.2e}' for k, v in worst.items()})}")
    return rows, worst


# ============================================================================ shipped scaling (B0)
def shipped_scaling(spec):
    """Reproduce the shipped diagonal/tied scaling: the package's own SMFE
    equilibration on the same stacked operator with the same scalar spans and
    seeds as compile() uses (the compiled object does not expose tau/sigma)."""
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    ops = [d["op"] for d in spec.dual_blocks]
    dims = [op.shape[0] for op in ops]
    slices, a = [], 0
    for dim in dims:
        slices.append((a, a + dim)); a += dim
    m = a
    scalar_spans = [sl for d, sl in zip(spec.dual_blocks, slices) if d["kind"] == "group_l2"]
    matvec = lambda x: jnp.concatenate([op.matvec(x) for op in ops])
    rmatvec = lambda y: sum(op.rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))
    t0 = time.perf_counter()
    tau, sigma, scaled_norm, diag = smfe_equilibration(matvec, rmatvec, spec.n, m, scalar_spans,
                                                       diagnostics=True, probes=2, iters=80, schedule="sgd")
    jax.block_until_ready((tau, sigma))
    return dict(tau=tau, sigma=sigma, slices=slices, m=m, scaled_norm=scaled_norm, diag=diag,
                seconds=time.perf_counter() - t0)


# ============================================================================ metric construction
_SKETCH_CACHE = {}


def sketch_grams(op, t0, s0, k, q, seed=0):
    """G_hat_i = (1/q) sum_r v_i^(r) v_i^(r)^T with v^(r) = s0^{1/2} K T0^{1/2} xi^(r),
    xi Rademacher, shared across blocks (one forward product per probe).  One
    compiled program per (operator, k, q); t0 and s0 are arguments."""
    n = op.shape[1]
    B = op.shape[0] // k
    key_ = (id(op), k, q, seed)
    if key_ not in _SKETCH_CACHE:
        @jax.jit
        def run(t0, s0):
            st = jnp.sqrt(jnp.broadcast_to(t0, (n,)))

            def body(i, carry):
                key, G = carry
                key, kk = jax.random.split(key)
                xi = jnp.where(jax.random.bernoulli(kk, 0.5, (n,)), 1.0, -1.0).astype(jnp.float32)
                v = (jnp.sqrt(s0) * op.matvec(st * xi)).reshape(-1, k)
                return key, G + jnp.einsum("bi,bj->bij", v, v)
            _, G = jax.lax.fori_loop(0, q, body, (jax.random.PRNGKey(seed + 1234), jnp.zeros((B, k, k), jnp.float32)))
            return G / q
        _SKETCH_CACHE[key_] = run
    return _SKETCH_CACHE[key_](jnp.asarray(t0, jnp.float32), jnp.float32(s0))


EIGH_CHUNK = int(os.environ.get("BM_EIGH_CHUNK", 1 << 20))


def batched_eigh(W):
    """Symmetric eigendecomposition of a large batch of small matrices.

    cuSolver's batched Jacobi path (gpusolverDnSsyevjBatched) fails with an
    invalid-value error above roughly a million matrices (measured: 1.05M blocks
    fine, 4.19M blocks fail on a 5090), so the batch is split into chunks.
    """
    B = W.shape[0]
    if B <= EIGH_CHUNK:
        return jnp.linalg.eigh(W)
    parts = [jnp.linalg.eigh(W[i:i + EIGH_CHUNK]) for i in range(0, B, EIGH_CHUNK)]
    return (jnp.concatenate([p[0] for p in parts]),
            jnp.concatenate([p[1] for p in parts]))


def build_block_metric(G, s0, eta, kappa_max=None, delta_frac=DELTA_FRAC):
    kappa_max = KAPPA_MAX if kappa_max is None else float(kappa_max)
    """W = (1-eta) diag(G) + eta G + delta I;  R = (tr W/k) W^{-1} clipped to
    [1/kappa, kappa];  Sigma_i = s0 R_i (the tied Sigma0 is scalar on coupled blocks).
    Returns ("diag", d (B,k)) for eta = 0, else ("full", Q, w)."""
    G = jnp.asarray(G, jnp.float32)
    B, k, _ = G.shape
    tr_k = jnp.trace(G, axis1=1, axis2=2) / k                            # (B,)
    floor = 1e-6 * jnp.maximum(jnp.median(tr_k), 1e-30)
    delta = delta_frac * jnp.maximum(tr_k, floor)                           # (B,)
    dG = jnp.diagonal(G, axis1=1, axis2=2)
    if eta == 0.0:
        Wd = dG + delta[:, None]
        R = jnp.clip((jnp.mean(Wd, axis=1, keepdims=True)) / Wd, 1.0 / kappa_max, kappa_max)
        return ("diag", s0 * R)
    W = (1.0 - eta) * jnp.einsum("bi,ij->bij", dG, jnp.eye(k)) + eta * G + delta[:, None, None] * jnp.eye(k)
    W = 0.5 * (W + jnp.swapaxes(W, 1, 2))
    omega, U = batched_eigh(W)
    trW_k = jnp.sum(omega, axis=1, keepdims=True) / k
    r = jnp.clip(trW_k / jnp.maximum(omega, 1e-30), 1.0 / kappa_max, kappa_max)
    return ("full", U, s0 * r)


def block_gram_condition(G_exact, R):
    """Conditioning of the preconditioned block Grams R^{1/2} G R^{1/2}.

    A condition number is only meaningful for a positive definite Gram.  Blocks
    whose operator has a null direction (the 3-D TV gradient annihilates
    constants, so its block Gram is singular) produce a minimum eigenvalue that
    is zero or slightly negative in floating point, and the ratio lam_max/lam_min
    is then a saturated artifact, not a large condition number.  Such cases are
    reported as rank deficient with NaN condition numbers, and lam_min / lam_max
    are always returned so a reader can see why.
    """
    G = np.asarray(G_exact, np.float64)
    if R[0] == "scalar":
        P = float(R[1]) * G
    elif R[0] == "diag":
        h = np.sqrt(np.asarray(R[1], np.float64))
        P = h[:, :, None] * G * h[:, None, :]
    else:
        Q, r = np.asarray(R[1], np.float64), np.asarray(R[2], np.float64)
        half = np.einsum("bij,bj,bkj->bik", Q, np.sqrt(r), Q)
        P = np.einsum("bij,bjk,bkl->bil", half, G, half)
    ev = np.linalg.eigvalsh(P)
    scale = ev[:, -1].max()
    nz = ev[:, -1] > 1e-12 * scale                      # blocks that are not entirely zero
    ev = ev[nz]
    lam_max = float(ev[:, -1].max())
    lam_min = float(ev[:, 0].min())
    # Relative to the spectral scale, a minimum eigenvalue at or below the
    # eigensolver's roundoff is a null direction, not a small positive value.
    floor = 1e-12 * lam_max
    deficient = bool(lam_min <= floor)
    if deficient:
        return dict(kappa_global=float("nan"), kappa_block_median=float("nan"),
                    kappa_block_max=float("nan"), lam_max=lam_max, lam_min=lam_min,
                    rank_deficient=True)
    per = ev[:, -1] / ev[:, 0]
    return dict(kappa_global=lam_max / lam_min, kappa_block_median=float(np.median(per)),
                kappa_block_max=float(per.max()), lam_max=lam_max, lam_min=lam_min,
                rank_deficient=False)


# ============================================================================ research PDHG loop

def make_checker(comp, data):
    """Stopping check through the public batched certificate (one device->host
    scalar per block); acceptance is then confirmed by the scalar public call
    comp.certificate(x, y, data) on the same pair."""
    data_b = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], data)

    @jax.jit
    def reduce_(cert):
        return jnp.maximum(cert["r_primal"][0], cert["r_dual"][0])

    def check(x, y):
        return float(reduce_(comp.certificate_many(x[None], y[None], data_b)))
    return check


def _split_metrics(metrics):
    """Static kinds (closed over) and the array leaves (passed as jit arguments)."""
    kinds = tuple(m[0] for m in metrics)
    arrays = tuple(tuple(m[1:]) for m in metrics)
    return kinds, arrays


def _smooth_repair(spec, x, ys_orig):
    """The shipped engine returns smooth dual blocks as their gradient (make_dual_repair);
    the prototype applies the same map to its returned pair so the certificate sees
    the same kind of point (only F3's sum_squares block is affected)."""
    out = []
    for j, d in enumerate(spec.dual_blocks):
        if d["kind"] == "sum_squares":
            out.append(d["op"].matvec(x) + jnp.asarray(d["offset"], jnp.float32))
        else:
            out.append(ys_orig[j])
    return jnp.concatenate(out)


class Loop:
    """Fixed-metric PDHG on a Spec.  metrics: list per dual block of
    ("scalar", s) | ("diag", d) | ("full", Q, w) | ("vec", sigma_vector) (sum_squares rows).
    T, the metric arrays and the offsets are jit ARGUMENTS (not baked constants), so
    one compiled block serves every instance of a family (F4)."""

    def __init__(self, spec, T, metrics, block=100):
        self.spec, self.block = spec, int(block)
        self.T = jnp.asarray(T, jnp.float32)
        self.kinds, self.marrays = _split_metrics(metrics)
        self.offs = tuple(jnp.asarray(d["offset"], jnp.float32) for d in spec.dual_blocks)
        n = spec.n
        ops = [d["op"] for d in spec.dual_blocks]
        bkinds = [d["kind"] for d in spec.dual_blocks]
        ks = [d["k"] for d in spec.dual_blocks]
        lams = [float(d["lam"]) for d in spec.dual_blocks]
        mu = float(spec.mu)
        primal = spec.primal
        kinds = self.kinds

        def prox_primal(v, T):
            if primal == "ridge":
                return v / (1.0 + T * mu)
            return jnp.maximum(v, 0.0)

        def dual_update(j, ystate, r, ma):
            kind, k, lam = bkinds[j], ks[j], lams[j]
            if kind == "sum_squares":                       # f = 1/2||.||^2, diagonal metric s
                s = ma[0]
                return (ystate + s * r) / (1.0 + s)
            R = r.reshape(-1, k)
            if kinds[j] == "scalar":
                return ball_project(ystate + ma[0] * R, lam)
            if kinds[j] == "diag":
                v = ystate + ma[0] * R
                return v - ma[0] * prox_l2_diag_any(v / ma[0], ma[0], lam)
            Q, w = ma                                       # rotated dual state ytilde = Q^T y
            vt = ystate + w * jnp.einsum("bji,bj->bi", Q, R)
            return vt - w * prox_l2_diag_any(vt / w, w, lam)

        def to_original(j, ystate, ma):
            if bkinds[j] == "sum_squares":
                return ystate
            if kinds[j] == "full":
                return jnp.einsum("bij,bj->bi", ma[0], ystate).reshape(-1)
            return ystate.reshape(-1)

        def step(st, T, marrays, offs):
            x, ys, xbar = st
            new_ys, KTy = [], jnp.zeros((n,), jnp.float32)
            for j in range(len(ops)):
                r = ops[j].matvec(xbar) + offs[j]
                yn = dual_update(j, ys[j], r, marrays[j])
                KTy = KTy + ops[j].rmatvec(to_original(j, yn, marrays[j]))
                new_ys.append(yn)
            xn = prox_primal(x - T * KTy, T)
            return xn, tuple(new_ys), 2.0 * xn - x

        @jax.jit
        def run_block(x, ys, xbar, T, marrays, offs):
            return jax.lax.fori_loop(0, self.block, lambda _, st: step(st, T, marrays, offs), (x, ys, xbar))

        @jax.jit
        def assemble(ys, marrays):
            return jnp.concatenate([to_original(j, ys[j], marrays[j]) for j in range(len(ops))])

        @jax.jit
        def prox_bench(ys, rs, marrays):                    # timing attribution: dual proxes alone
            def body(_, st):
                ys, rs = st
                return tuple(dual_update(j, ys[j], rs[j], marrays[j]) for j in range(len(ops))), rs
            return jax.lax.fori_loop(0, self.block, body, (ys, rs))[0]

        @jax.jit
        def ops_bench(x, ys, marrays, offs):                # timing attribution: operator products alone
            def body(_, st):
                x, ys, acc = st
                KTy = jnp.zeros((n,), jnp.float32)
                for j in range(len(ops)):
                    r = ops[j].matvec(x) + offs[j]
                    acc = acc + jnp.sum(r)                  # keeps the forward product live
                    KTy = KTy + ops[j].rmatvec(to_original(j, ys[j], marrays[j]))
                return x + 1e-9 * KTy, ys, acc
            return jax.lax.fori_loop(0, self.block, body, (x, ys, jnp.float32(0.0)))[0]

        self._run_block, self._assemble, self._prox_bench, self._ops_bench = run_block, assemble, prox_bench, ops_bench

    def init_state(self):
        n = self.spec.n
        x = jnp.zeros((n,), jnp.float32)
        ys = []
        for j, d in enumerate(self.spec.dual_blocks):
            m_j = d["op"].shape[0]
            ys.append(jnp.zeros((m_j,), jnp.float32) if d["kind"] == "sum_squares"
                      else jnp.zeros((m_j // d["k"], d["k"]), jnp.float32))
        return x, tuple(ys), x

    def returned_pair(self, x, ys):
        y = self._assemble(ys, self.marrays)
        if any(d["kind"] == "sum_squares" for d in self.spec.dual_blocks):
            parts, a = [], 0
            for d in self.spec.dual_blocks:
                parts.append(y[a:a + d["op"].shape[0]]); a += d["op"].shape[0]
            y = _smooth_repair(self.spec, x, parts)
        return y

    def run(self, comp, data, tol, max_blocks, max_seconds, log=None, offs=None):
        offs = self.offs if offs is None else offs
        check = make_checker(comp, data)
        x, ys, xbar = self.init_state()
        t_iter = t_ver = 0.0
        history, cert, res = [], None, float("inf")
        t_start = time.perf_counter()
        blocks = 0
        status = "max_blocks"
        while blocks < max_blocks:
            t0 = time.perf_counter()
            x, ys, xbar = self._run_block(x, ys, xbar, self.T, self.marrays, offs)
            jax.block_until_ready(x)
            t_iter += time.perf_counter() - t0
            blocks += 1
            t0 = time.perf_counter()
            y = self.returned_pair(x, ys)
            res = check(x, y)
            if res <= tol:                                  # confirm with the scalar public call
                cert = comp.certificate(x, y, data)
                res = max(cert["r_primal"], cert["r_dual"])
            t_ver += time.perf_counter() - t0
            history.append((blocks * self.block, res, time.perf_counter() - t_start))
            if log and (blocks % 20 == 0 or res <= tol):
                log(f"    it={blocks * self.block:7d} r={res:.3e}")
            if not math.isfinite(res):
                status = "nonfinite"; break
            if res <= tol:
                status = "solved"; break
            if time.perf_counter() - t_start > max_seconds:
                status = "time_cap"; break
        total_s = time.perf_counter() - t_start
        y = self.returned_pair(x, ys)
        if cert is None:
            cert = comp.certificate(x, y, data); res = max(cert["r_primal"], cert["r_dual"])
        return dict(x=x, y=y, iters=blocks * self.block, blocks=blocks, residual=res,
                    cert=cert, status=status, iter_s=t_iter, verify_s=t_ver, total_s=total_s, history=history)

    def bench(self, reps=5):
        """Steady-state seconds per block for: full block, dual proxes only, operator products only."""
        x, ys, xbar = self.init_state()
        rs = tuple(jnp.ones_like(y) for y in ys)
        out = {}
        for name, fn in (("block", lambda: self._run_block(x, ys, xbar, self.T, self.marrays, self.offs)),
                         ("prox", lambda: self._prox_bench(ys, rs, self.marrays)),
                         ("ops", lambda: self._ops_bench(x, ys, self.marrays, self.offs))):
            jax.block_until_ready(fn())
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); jax.block_until_ready(fn()); ts.append(time.perf_counter() - t0)
            out[name] = float(np.median(ts))
        return out


# ============================================================================ Liu-Xu-Yin arm (A4)
class LXYLoop:
    """iPrePDHG (Liu, Xu, Yin 2019): M1 = I/tau, M2 = tau K K^T (products only),
    z-subproblem  argmin_z g^*(z) - <z - z_k, K xbar + l> + 1/2||z - z_k||^2_{M2}
    solved by p proximal-gradient steps with step gamma = 0.95 / (tau ||K||^2),
    prox_{gamma g^*} the closed-form ball projection (group_l2) or v/(1+gamma) (sum_squares)."""

    def __init__(self, spec, tau, p, normK, block=100):
        self.spec, self.block, self.p = spec, int(block), int(p)
        n = spec.n
        ops = [d["op"] for d in spec.dual_blocks]
        self.offs = tuple(jnp.asarray(d["offset"], jnp.float32) for d in spec.dual_blocks)
        kinds = [d["kind"] for d in spec.dual_blocks]
        ks = [d["k"] for d in spec.dual_blocks]
        lams = [float(d["lam"]) for d in spec.dual_blocks]
        mu = float(spec.mu)
        tau = float(tau)
        gamma = 0.95 / (tau * float(normK) ** 2)
        self.tau, self.gamma = tau, gamma
        p_inner = self.p

        def prox_primal(v):
            return v / (1.0 + tau * mu) if spec.primal == "ridge" else jnp.maximum(v, 0.0)

        def prox_gstar(j, u):
            if kinds[j] == "sum_squares":
                return u / (1.0 + gamma)
            return ball_project(u.reshape(-1, ks[j]), lams[j]).reshape(-1)

        def KKt(zs):                                    # K K^T z on the stacked dual
            KTz = sum(ops[j].rmatvec(zs[j]) for j in range(len(ops)))
            return [ops[j].matvec(KTz) for j in range(len(ops))]

        def step(st, offs):
            x, zs, xbar = st
            KTz = sum(ops[j].rmatvec(zs[j]) for j in range(len(ops)))
            xn = prox_primal(x - tau * KTz)
            xbar = 2.0 * xn - x
            rs = [ops[j].matvec(xbar) + offs[j] for j in range(len(ops))]
            zk = zs
            z = zs
            for _i in range(p_inner):
                Mdz = KKt([z[j] - zk[j] for j in range(len(ops))])
                z = tuple(prox_gstar(j, z[j] - gamma * (-rs[j] + tau * Mdz[j])) for j in range(len(ops)))
            return xn, z, xbar

        @jax.jit
        def run_block(x, zs, xbar, offs):
            return jax.lax.fori_loop(0, self.block, lambda _, st: step(st, offs), (x, zs, xbar))
        self._run_block = run_block
        self.n = n

    def init_state(self):
        x = jnp.zeros((self.n,), jnp.float32)
        zs = tuple(jnp.zeros((d["op"].shape[0],), jnp.float32) for d in self.spec.dual_blocks)
        return x, zs, x

    def returned_pair(self, x, zs):
        if any(d["kind"] == "sum_squares" for d in self.spec.dual_blocks):
            return _smooth_repair(self.spec, x, list(zs))
        return jnp.concatenate(zs)

    def run(self, comp, data, tol, max_blocks, max_seconds, log=None, offs=None):
        offs = self.offs if offs is None else offs
        check = make_checker(comp, data)
        x, zs, xbar = self.init_state()
        t_iter = t_ver = 0.0
        history, cert, res, blocks, status = [], None, float("inf"), 0, "max_blocks"
        t_start = time.perf_counter()
        while blocks < max_blocks:
            t0 = time.perf_counter()
            x, zs, xbar = self._run_block(x, zs, xbar, offs)
            jax.block_until_ready(x)
            t_iter += time.perf_counter() - t0
            blocks += 1
            t0 = time.perf_counter()
            y = self.returned_pair(x, zs)
            res = check(x, y)
            if res <= tol:
                cert = comp.certificate(x, y, data)
                res = max(cert["r_primal"], cert["r_dual"])
            t_ver += time.perf_counter() - t0
            history.append((blocks * self.block, res, time.perf_counter() - t_start))
            if log and (blocks % 20 == 0 or res <= tol):
                log(f"    it={blocks * self.block:7d} r={res:.3e}")
            if not math.isfinite(res):
                status = "nonfinite"; break
            if res <= tol:
                status = "solved"; break
            if time.perf_counter() - t_start > max_seconds:
                status = "time_cap"; break
        total_s = time.perf_counter() - t_start
        y = self.returned_pair(x, zs)
        if cert is None:
            cert = comp.certificate(x, y, data); res = max(cert["r_primal"], cert["r_dual"])
        return dict(x=x, y=y, iters=blocks * self.block, blocks=blocks, residual=res,
                    cert=cert, status=status, iter_s=t_iter, verify_s=t_ver, total_s=total_s, history=history)

    def bench(self, reps=5):
        x, zs, xbar = self.init_state()
        jax.block_until_ready(self._run_block(x, zs, xbar, self.offs))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(self._run_block(x, zs, xbar, self.offs)); ts.append(time.perf_counter() - t0)
        return {"block": float(np.median(ts)), "prox": float("nan"), "ops": float("nan")}


# ============================================================================ norm estimate
_POWER_CACHE = {}


def power_norm_metric(spec, T, metrics, iters=POWER_ITERS, seed=0):
    """Power iteration estimate of ||Sigma^{1/2} K T^{1/2}||_2 (a lower bound that
    converges from below; GAMMA_SAFETY covers the remaining gap).  The compiled
    program is cached per (spec, metric kinds); T and the metric arrays are arguments."""
    kinds, arrays = _split_metrics(metrics)
    key = (id(spec), kinds, iters, seed)
    if key in _POWER_CACHE:
        return float(_POWER_CACHE[key](jnp.asarray(T, jnp.float32), arrays))
    n = spec.n
    ops = [d["op"] for d in spec.dual_blocks]
    def apply_sigma(j, r, ma):                              # Sigma_j r
        if spec.dual_blocks[j]["kind"] == "sum_squares":
            return ma[0] * r
        k = spec.dual_blocks[j]["k"]
        R = r.reshape(-1, k)
        if kinds[j] in ("scalar", "diag"):
            return (ma[0] * R).reshape(-1)
        Q, w = ma
        return jnp.einsum("bij,bj->bi", Q, w * jnp.einsum("bji,bj->bi", Q, R)).reshape(-1)

    @jax.jit
    def run(T, arrays):
        sT = jnp.sqrt(jnp.broadcast_to(T, (n,)))

        def AtA(v):                                         # T^{1/2} K^T Sigma K T^{1/2} v
            u = sT * v
            out = jnp.zeros((n,), jnp.float32)
            for j in range(len(ops)):
                out = out + ops[j].rmatvec(apply_sigma(j, ops[j].matvec(u), arrays[j]))
            return sT * out
        v = jax.random.normal(jax.random.PRNGKey(seed + 5), (n,), jnp.float32)
        v = v / jnp.linalg.norm(v)

        def body(_, st):
            v, s = st
            w = AtA(v)
            s = jnp.linalg.norm(w) + 1e-30
            return w / s, s
        _, s = jax.lax.fori_loop(0, iters, body, (v, jnp.asarray(1.0, jnp.float32)))
        return jnp.sqrt(s)
    _POWER_CACHE[key] = run
    return float(run(jnp.asarray(T, jnp.float32), arrays))


def scale_metrics(metrics, T, factor):
    out = []
    for met in metrics:
        if met[0] in ("scalar", "diag", "vec"):
            out.append((met[0], met[1] * factor))
        else:
            out.append(("full", met[1], met[2] * factor))
    return out, T * factor


def metric_summary(met):
    if met[0] == "scalar":
        return f"scalar {float(met[1]):.4g}"
    if met[0] in ("diag", "vec"):
        a = np.asarray(met[1]); return f"{met[0]} [{a.min():.3g}, {a.max():.3g}]"
    w = np.asarray(met[2]); return f"full w[{w.min():.3g}, {w.max():.3g}]"


# ============================================================================ arms

def float64_objective(spec, x):
    """Objective of the ORIGINAL problem at a returned point, evaluated in float64.

    The solve runs in float32; the point is promoted and the objective summed in
    float64 on the host, so two arms can be compared on the problem they both
    claim to have solved rather than only on their certificates.  Operator
    products are applied by the operator itself (float32, as the solver sees it)
    and promoted afterwards, which is stated rather than hidden: this measures
    agreement of the returned points, not an exact float64 optimum.
    """
    xf = np.asarray(x, np.float64)
    obj = 0.0
    if spec.primal == "ridge":
        obj += 0.5 * float(spec.mu) * float(xf @ xf)
    elif spec.primal == "nonneg":
        if xf.min() < -1e-6 * max(1.0, float(np.abs(xf).max())):
            return float("nan"), float(xf.min())
    for d in spec.dual_blocks:
        r = np.asarray(d["op"].matvec(jnp.asarray(x, jnp.float32)), np.float64) \
            + np.asarray(d["offset"], np.float64)
        if d["kind"] == "sum_squares":
            obj += 0.5 * float(r @ r)
        else:
            obj += float(d["lam"]) * float(np.linalg.norm(r.reshape(-1, d["k"]), axis=1).sum())
    return obj, (float(xf.min()) if spec.primal == "nonneg" else float("nan"))


def device_peak_bytes():
    try:
        st = jax.local_devices()[0].memory_stats()
        return int(st.get("peak_bytes_in_use", -1))
    except Exception:
        return -1


def build_family(args):
    import blockmetric_models as bm
    if args.family in ("F1", "F4"):
        return bm.blockmetric_f1(n=args.n, rho=args.rho, k=args.k, variant=args.variant, seed=args.seed, mu=args.mu)
    if args.family == "F2":
        return bm.blockmetric_f2(L=args.L, mixing=args.variant, seed=args.seed, mu=args.mu,
                                 sigma_blur=args.blur, channels=args.channels)
    if args.family == "F3":
        return bm.blockmetric_f3(L=args.L, seed=args.seed)
    raise ValueError(args.family)


def run_shipped(prob, data, tol, max_blocks, variant, log, warm=False):
    """A0 / A0std: the shipped engine.  Compile once (setup), then solve twice from
    the zero start: the first call carries XLA compilation, the second is the warm time."""
    t0 = time.perf_counter()
    kw = dict(tol=tol, max_blocks=max_blocks, block=100)
    if variant == "standard":
        kw["pdhg_variant"] = "standard"
    comp = prob.compile(**kw)
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
    cold_s = time.perf_counter() - t0
    if cold_s > 200.0:
        warm_s = cold_s
        log(f"  cold solve took {cold_s:.1f}s > 200s: reporting the cold solve (includes XLA compile)")
    else:
        t0 = time.perf_counter()
        sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
        warm_s = time.perf_counter() - t0
    cert = comp.certificate(sol.x, sol.y, data)
    log(f"  shipped[{variant}] iters={sol.iters} status={sol.status} residual={sol.residual:.3e} "
        f"cert={max(cert['r_primal'], cert['r_dual']):.3e} compile={setup_s:.2f}s cold={cold_s:.2f}s warm={warm_s:.2f}s")
    return comp, sol, cert, dict(setup_s=setup_s, cold_s=cold_s, warm_s=warm_s)


def prototype_metrics(spec, sc, arm, eta, source, q, log):
    """Build (T, metrics, setup timings, diagnostics) for A1/A2/A3/A3s."""
    tau, sigma, slices = sc["tau"], sc["sigma"], sc["slices"]
    T = tau
    metrics, timings, diag = [], dict(gram_s=0.0, eig_s=0.0, norm_s=0.0), {}
    for j, d in enumerate(spec.dual_blocks):
        s, e = slices[j]
        sig_j = sigma if sigma.ndim == 0 else sigma[s:e]
        if d["kind"] == "sum_squares":
            metrics.append(("vec", jnp.broadcast_to(sig_j, (e - s,))))
            continue
        s0 = float(sig_j if sig_j.ndim == 0 else sig_j[0])
        if sig_j.ndim and not bool(jnp.all(sig_j == sig_j[0])):
            raise RuntimeError("tied block expected a uniform shipped sigma")
        if arm == "A1":
            metrics.append(("scalar", jnp.asarray(s0, jnp.float32)))
            continue
        k = d["k"]
        t0 = time.perf_counter()
        if source == "exact":
            G = jnp.asarray(d["exact_gram"](np.asarray(tau), s0), jnp.float32)
        else:
            G = sketch_grams(d["op"], tau, s0, k, q)
        jax.block_until_ready(G)
        timings["gram_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        met = build_block_metric(G, s0, eta, kappa_max=KAPPA_MAX)
        jax.block_until_ready(met[1])
        timings["eig_s"] += time.perf_counter() - t0
        metrics.append(met)
        if d["exact_gram"] is not None:
            Gx = d["exact_gram"](np.asarray(tau), s0)
            diag[f"block{j}_kappa_B0"] = block_gram_condition(Gx, ("scalar", 1.0))
            Rm = (("diag", np.asarray(met[1]) / s0) if met[0] == "diag"
                  else ("full", np.asarray(met[1]), np.asarray(met[2]) / s0))
            diag[f"block{j}_kappa_new"] = block_gram_condition(Gx, Rm)
            if source != "exact":
                Gs = np.asarray(G, np.float64)
                diag[f"block{j}_gram_sketch_relerr"] = float(np.linalg.norm(Gs - Gx) / np.linalg.norm(Gx))
    if arm == "A1":
        return T, metrics, timings, diag, dict(norm_est=sc["scaled_norm"], norm_exact=None, norm_upper=None, factor=1.0)
    t0 = time.perf_counter()
    nu = power_norm_metric(spec, T, metrics)
    timings["norm_s"] = time.perf_counter() - t0
    factor = GAMMA_SAFETY / nu
    metrics, T = scale_metrics(metrics, T, factor)
    nu_exact = None
    if spec.exact_norm is not None:
        j = [i for i, d in enumerate(spec.dual_blocks) if d["kind"] == "group_l2"][0]
        met = metrics[j]
        if met[0] == "scalar":
            sig = float(met[1])
        elif met[0] == "diag":
            sig = np.asarray(met[1], np.float64)
        else:
            Q, w = np.asarray(met[1], np.float64), np.asarray(met[2], np.float64)
            sig = np.einsum("bij,bj,bkj->bik", Q, w, Q)
        nu_exact = spec.exact_norm(sig, np.asarray(T))
        nu_upper = spec.norm_upper(sig, np.asarray(T)) if getattr(spec, "norm_upper", None) is not None else None
    else:
        nu_upper = None
    log(f"  norm estimate {nu:.5f} -> scaled by {factor:.4f}; exact after scaling: "
        f"{'n/a' if nu_exact is None else f'{nu_exact:.5f}'}"
        f"{'' if nu_upper is None else f', upper bound {nu_upper:.5f}'} (must be < 1)")
    return T, metrics, timings, diag, dict(norm_est=nu, norm_exact=nu_exact, norm_upper=nu_upper, factor=factor)


def optimal_diagonal_reference(spec, sc, log):
    """A5: maximize t s.t. t H <= P <= H, P diagonal, H = K T0 K^T + delta I,
    by cvxpy + Clarabel on the formed small matrix (tiny instances only)."""
    import cvxpy as cp
    n = spec.n
    op = spec.dual_blocks[0]["op"]
    m = op.shape[0]
    t0 = time.perf_counter()
    Kd = np.asarray(jax.vmap(op.matvec)(jnp.eye(n, dtype=jnp.float32))).T.astype(np.float64)   # forming: reference only
    tau = np.broadcast_to(np.asarray(sc["tau"], np.float64), (n,))
    H = Kd @ np.diag(tau) @ Kd.T
    H = 0.5 * (H + H.T)
    delta = 1e-6 * np.trace(H) / m
    H = H + delta * np.eye(m)
    p = cp.Variable(m); t = cp.Variable()
    P = cp.diag(p)
    prob = cp.Problem(cp.Maximize(t), [P << H, P >> t * H, p >= 1e-9])
    prob.solve(solver="CLARABEL", verbose=False)
    sdp_s = time.perf_counter() - t0
    log(f"  A5 SDP: status {prob.status}, t = {t.value:.4e} (diagonal-achievable kappa = {1 / t.value:.2f}), {sdp_s:.1f}s")
    pv = np.asarray(p.value, np.float64)
    k = spec.dual_blocks[0]["k"]
    sig = (1.0 / pv).reshape(-1, k)
    metrics = [("diag", jnp.asarray(sig, jnp.float32))]
    T = sc["tau"]
    nu_exact = spec.exact_norm(sig, np.asarray(T))
    factor = GAMMA_SAFETY / nu_exact
    metrics, T = scale_metrics(metrics, T, factor)
    # condition of the whole preconditioned Gram, exactly, for the record
    Sh = np.sqrt(sig.reshape(-1) * factor)
    Pm = Sh[:, None] * (H - delta * np.eye(m)) * Sh[None, :] * factor
    ev = np.linalg.eigvalsh(Pm)
    return T, metrics, dict(sdp_s=sdp_s, t_opt=float(t.value), kappa_diag_opt=float(1 / t.value),
                            kappa_check=float(ev[-1] / ev[0]), norm_exact=nu_exact * factor, factor=factor)


def run_arm(args, prob, spec, comp, data, log):
    """One arm in this process.  Returns the CSV row."""
    tol, max_blocks, max_s = args.tol, args.max_blocks, args.max_seconds
    row = dict(family=spec.family, variant=spec.variant, name=spec.name, n=spec.n, k=spec.k, rho=spec.rho,
               mu=spec.mu, kappa_max=KAPPA_MAX, note=getattr(spec, "note", ""),
               seed=args.seed, src_hash=_source_hash(),
               timer_defs="iter_s=loop blocks only; total_warm_s=setup+loop+verify; setup_s=equil+gram+eig+norm",
               newton_iters=NEWTON_ITERS, prox_impl=PROX_IMPL, arm=args.arm, eta=args.eta if args.arm in ("A2", "A3", "A3s") else "",
               q=args.q if args.arm == "A3s" else "", p=args.p if args.arm == "A4" else "",
               tau_mult=args.tau_mult if args.arm == "A4" else "", tol=tol,
               host=os.uname().nodename, device=jax.devices()[0].device_kind)
    if args.arm in ("A0", "A0std"):
        variant = "standard" if args.arm == "A0std" else "halpern"
        comp0, sol, cert, tm = run_shipped(prob, data, tol, max_blocks, variant, log)
        try:
            obj, minx = float64_objective(spec, sol.x)
            row["objective_f64"] = obj
            row["min_x"] = minx
        except Exception as exc:
            row["objective_f64"] = f"ERROR {type(exc).__name__}"
        row.update(iters=sol.iters, status=sol.status, engine_residual=sol.residual,
                   r_primal=cert["r_primal"], r_dual=cert["r_dual"], residual=max(cert["r_primal"], cert["r_dual"]),
                   certified=int(max(cert["r_primal"], cert["r_dual"]) <= tol),
                   setup_s=tm["setup_s"], iter_s=tm["warm_s"], prox_s=float("nan"), ops_s=float("nan"),
                   verify_s=0.0, total_warm_s=tm["setup_s"] + tm["warm_s"], cold_solve_s=tm["cold_s"],
                   norm_est=comp0.meta["scaled_norm"], norm_exact="", factor=1.0)
        row["peak_bytes"] = device_peak_bytes()
        return row

    sc = shipped_scaling(spec)
    log(f"  shipped scaling reproduced in {sc['seconds']:.2f}s: ||B0||~{sc['scaled_norm']:.4f}, "
        f"tau [{float(jnp.min(sc['tau'])):.3g},{float(jnp.max(sc['tau'])):.3g}], "
        f"sigma [{float(jnp.min(sc['sigma'])):.3g},{float(jnp.max(sc['sigma'])):.3g}]")
    diag = {}
    if args.arm in ("A1", "A2", "A3", "A3s"):
        eta = {"A1": 0.0, "A2": 0.0, "A3": 1.0, "A3s": 1.0}[args.arm] if args.eta is None else args.eta
        source = "sketch" if args.arm == "A3s" else "exact"
        q = args.q * spec.k if args.arm == "A3s" else 0
        # warm-up pass (XLA compile of gram/eigh/power), then the timed pass
        T, metrics, timings, diag, nrm = prototype_metrics(spec, sc, args.arm, eta, source, q, lambda *_: None)
        T, metrics, timings, diag, nrm = prototype_metrics(spec, sc, args.arm, eta, source, q, log)
        setup_s = sc["seconds"] + timings["gram_s"] + timings["eig_s"] + timings["norm_s"]
        loop = Loop(spec, T, metrics, block=100)
        row.update(eta=eta, q=q)
    elif args.arm == "A4":
        from ocs.solver.equilibrate import operator_norm
        ops = [d["op"] for d in spec.dual_blocks]
        matvec = lambda x: jnp.concatenate([o.matvec(x) for o in ops])
        rmatvec = lambda y: sum(o.rmatvec(y[s:e]) for o, (s, e) in zip(ops, sc["slices"]))
        t0 = time.perf_counter()
        normK = float(operator_norm(matvec, rmatvec, spec.n, iters=30))
        tau = args.tau_mult * 0.99 / normK
        timings = dict(gram_s=0.0, eig_s=0.0, norm_s=time.perf_counter() - t0)
        setup_s = timings["norm_s"]
        nrm = dict(norm_est=normK, norm_exact=None, factor=1.0)
        log(f"  LXY: ||K||~{normK:.4f}, tau={tau:.4g}, p={args.p}")
        loop = LXYLoop(spec, tau, args.p, normK, block=100)
    elif args.arm == "A5":
        T, metrics, ref = optimal_diagonal_reference(spec, sc, log)
        timings = dict(gram_s=0.0, eig_s=ref["sdp_s"], norm_s=0.0)
        setup_s = sc["seconds"] + ref["sdp_s"]
        nrm = dict(norm_est=ref["norm_exact"], norm_exact=ref["norm_exact"], factor=ref["factor"])
        diag = {"A5": ref}
        loop = Loop(spec, T, metrics, block=100)
    else:
        raise ValueError(args.arm)

    # cold run (XLA compile) then the timed warm run from the same zero start
    cold = loop.run(comp, data, tol, max_blocks, max_s, log=None)
    if cold["total_s"] > 200.0:
        res, timed_run = cold, "cold"
        log(f"  cold run took {cold['total_s']:.1f}s > 200s: reporting the cold run (includes XLA compile)")
    else:
        res, timed_run = loop.run(comp, data, tol, max_blocks, max_s, log=log), "warm"
    bench = loop.bench()
    prox_s = bench["prox"] * res["blocks"] if math.isfinite(bench["prox"]) else float("nan")
    ops_s = bench["ops"] * res["blocks"] if math.isfinite(bench["ops"]) else float("nan")
    log(f"  {args.arm}: iters={res['iters']} status={res['status']} residual={res['residual']:.3e} "
        f"setup={setup_s:.2f}s loop={res['iter_s']:.2f}s (prox {prox_s:.2f}s, ops {ops_s:.2f}s) verify={res['verify_s']:.2f}s "
        f"total_warm={setup_s + res['total_s']:.2f}s cold_loop={cold['total_s']:.2f}s")
    try:
        obj, minx = float64_objective(spec, res["x"])
        row["objective_f64"] = obj
        row["min_x"] = minx
    except Exception as exc:
        row["objective_f64"] = f"ERROR {type(exc).__name__}"
    row.update(iters=res["iters"], status=res["status"], engine_residual="",
               r_primal=res["cert"]["r_primal"], r_dual=res["cert"]["r_dual"], residual=res["residual"],
               certified=int(res["residual"] <= tol), setup_s=setup_s, iter_s=res["iter_s"], prox_s=prox_s, ops_s=ops_s,
               verify_s=res["verify_s"], total_warm_s=setup_s + res["total_s"], cold_solve_s=cold["total_s"],
               norm_est=nrm["norm_est"], norm_exact="" if nrm["norm_exact"] is None else nrm["norm_exact"],
               norm_upper="" if nrm.get("norm_upper") is None else nrm["norm_upper"],
               factor=nrm["factor"], gram_s=timings["gram_s"], eig_s=timings["eig_s"], norm_s=timings["norm_s"],
               equil_s=sc["seconds"], block_s=bench["block"], prox_block_s=bench["prox"], ops_block_s=bench["ops"],
               timed_run=timed_run, peak_bytes=device_peak_bytes())
    for key, val in diag.items():
        if isinstance(val, dict):
            for kk, vv in val.items():
                row[f"{key}_{kk}"] = vv
        else:
            row[key] = val
    # Subsample: a 500k-iteration run has 5000 blocks, and a multi-megabyte CSV
    # field is what made an earlier header extension fail and silently drop rows.
    hist = res["history"]
    step = max(1, len(hist) // 200)
    kept = hist[::step]
    if kept and kept[-1] is not hist[-1]:
        kept.append(hist[-1])
    row["history"] = json.dumps([(it, float(r), round(t, 3)) for it, r, t in kept])
    row["history_blocks"] = len(hist)
    return row


def loopcheck(args, prob, spec, data, log):
    """A1 must reproduce the shipped standard-PDHG certificate at equal iterations."""
    sc = shipped_scaling(spec)
    T, metrics, _, _, _ = prototype_metrics(spec, sc, "A1", 0.0, "exact", 0, log)
    loop = Loop(spec, T, metrics, block=100)
    out = []
    for nb in (1, 5, 20, 50):
        comp = prob.compile(tol=1e-12, max_blocks=nb, block=100, pdhg_variant="standard")
        sol = comp.solve_instance(data)
        c0 = comp.certificate(sol.x, sol.y, data)
        x, ys, xbar = loop.init_state()
        for _ in range(nb):
            x, ys, xbar = loop._run_block(x, ys, xbar, loop.T, loop.marrays, loop.offs)
        y = loop.returned_pair(x, ys)
        c1 = comp.certificate(x, y, data)
        r0, r1 = max(c0["r_primal"], c0["r_dual"]), max(c1["r_primal"], c1["r_dual"])
        dx = float(jnp.linalg.norm(x - sol.x) / (jnp.linalg.norm(sol.x) + 1e-30))
        dy = float(jnp.linalg.norm(y - sol.y) / (jnp.linalg.norm(sol.y) + 1e-30))
        log(f"  loopcheck {nb * 100:5d} iters: shipped-standard r={r0:.4e}  A1 r={r1:.4e}  "
            f"rel diff {abs(r1 - r0) / r0:.2e}   ||dx||/||x||={dx:.2e} ||dy||/||y||={dy:.2e}")
        out.append(dict(iters=nb * 100, r_shipped=r0, r_A1=r1, rel=abs(r1 - r0) / r0, dx=dx, dy=dy))
    return out


def repeated_solves(args, prob, spec, comp, data, log):
    """F4: same K, N instances with different b; mean time per instance with setup amortized."""
    import blockmetric_models as bm
    rng = np.random.default_rng(args.seed + 999)
    op = spec.dual_blocks[0]["op"]
    n = spec.n
    bs = []
    for i in range(args.repeats):
        x_true = rng.normal(size=n)
        b = np.asarray(jax.jit(op.matvec)(jnp.asarray(x_true, jnp.float32))) + 0.3 * rng.normal(size=n)
        bs.append(b.astype(np.float32))
    tol, max_blocks, max_s = args.tol, args.max_blocks, args.max_seconds
    rows = []
    if args.arm in ("A0", "A0std"):
        t0 = time.perf_counter()
        kw = dict(tol=tol, max_blocks=max_blocks, block=100)
        if args.arm == "A0std":
            kw["pdhg_variant"] = "standard"
        comp0 = prob.compile(**kw)
        setup_s = time.perf_counter() - t0
        cum = setup_s
        for i, b in enumerate(bs):
            d = {"b": jnp.asarray(b)}
            t0 = time.perf_counter()
            sol = comp0.solve_instance(d); jax.block_until_ready(sol.x)
            dt = time.perf_counter() - t0
            cum += dt
            cert = comp0.certificate(sol.x, sol.y, d)
            rows.append(dict(instance=i, iters=sol.iters, solve_s=dt, cum_s=cum, setup_s=setup_s,
                             residual=max(cert["r_primal"], cert["r_dual"]), status=sol.status))
            if i in (0, 9, 99) or i == len(bs) - 1:
                log(f"  {args.arm} N={i + 1}: mean per instance {cum / (i + 1):.3f}s (setup {setup_s:.2f}s)")
    else:
        sc = shipped_scaling(spec)
        eta = {"A1": 0.0, "A2": 0.0, "A3": 1.0, "A3s": 1.0}[args.arm] if args.eta is None else args.eta
        source = "sketch" if args.arm == "A3s" else "exact"
        q = args.q * spec.k if args.arm == "A3s" else 0
        t0 = time.perf_counter()
        T, metrics, timings, diag, nrm = prototype_metrics(spec, sc, args.arm, eta, source, q, log)
        loop = Loop(spec, T, metrics, block=100)
        setup_s = time.perf_counter() - t0 + sc["seconds"]
        cum = setup_s
        for i, b in enumerate(bs):
            d = {"b": jnp.asarray(b)}
            offs = (jnp.asarray(-b),)
            t0 = time.perf_counter()
            res = loop.run(comp, d, tol, max_blocks, max_s, offs=offs)
            dt = time.perf_counter() - t0
            cum += dt
            rows.append(dict(instance=i, iters=res["iters"], solve_s=dt, cum_s=cum, setup_s=setup_s,
                             residual=res["residual"], status=res["status"]))
            if i in (0, 9, 99) or i == len(bs) - 1:
                log(f"  {args.arm} N={i + 1}: mean per instance {cum / (i + 1):.3f}s (setup {setup_s:.2f}s)")
    return rows


# ============================================================================ main
def write_row(path, row):
    """Append one row, extending the header when a run introduces new columns.

    Reading existing rows back is what previously failed on a multi-megabyte
    history field and dropped completed measurements, so the limit is raised at
    import and the history itself is subsampled.
    """
    fields = list(row)
    exists = os.path.exists(path)
    if exists:
        with open(path) as fh:
            header = next(csv.reader(fh))
        missing = [f for f in fields if f not in header]
        if missing:
            with open(path) as fh:
                old_rows = list(csv.DictReader(fh))
            fields = header + missing
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader(); w.writerows(old_rows)
        else:
            fields = header
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--gain-only", action="store_true",
                    help="build the metric and report the conditioning gain; do not solve")
    ap.add_argument("--dlog", type=float, default=3.0, help="validation metric entries span [10^-dlog, 10^dlog]")
    ap.add_argument("--loopcheck", action="store_true")
    ap.add_argument("--family", default="F1", choices=["F1", "F2", "F3", "F4"])
    ap.add_argument("--n", type=int, default=200_000); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.99); ap.add_argument("--variant", default="equal")
    ap.add_argument("--L", type=int, default=256); ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--channels", type=int, default=3, help="F2 block size c (3 = development family)")
    ap.add_argument("--mu", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--eta", type=float, default=None); ap.add_argument("--q", type=int, default=4, help="probes as a multiple of k")
    ap.add_argument("--p", type=int, default=2); ap.add_argument("--tau-mult", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--kappa-max", type=float, default=100.0, help="eigenvalue clip of the block correction R_i")
    ap.add_argument("--max-blocks", type=int, default=1000); ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--out", default=None); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global KAPPA_MAX
    KAPPA_MAX = float(args.kappa_max)
    import ocs
    log = lambda s: print(s, flush=True)
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} jax {jax.__version__} ocs {ocs.__version__} at {ocs.__file__}")

    if args.validate:
        rows, worst = validate(log=log, dlog=args.dlog)
        if args.out:
            with open(args.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            with open(args.out.replace(".csv", "_worst.json"), "w") as fh:
                json.dump(dict(worst=worst, x64=bool(jax.config.read("jax_enable_x64")), cases=len(rows), dlog=args.dlog,
                               newton_iters=NEWTON_ITERS, newton_unroll=NEWTON_UNROLL, jax=jax.__version__,
                               host=os.uname().nodename), fh, indent=1)
        return

    t0 = time.perf_counter()
    prob, spec = build_family(args)
    data = spec.data
    log(f"# {spec.name}: n={spec.n} built in {time.perf_counter() - t0:.1f}s")

    if args.loopcheck:
        out = loopcheck(args, prob, spec, data, log)
        if args.out:
            with open(args.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
        return

    if args.gain_only:
        sc = shipped_scaling(spec)
        _T, _m, _t, diag, _n = prototype_metrics(spec, sc, "A3", 1.0, "exact", 0, lambda *_: None)
        out = {"name": spec.name, "family": spec.family, "variant": spec.variant, "n": spec.n,
               "k": spec.k, "rho": spec.rho, "mu": spec.mu}
        for key, val in diag.items():
            if isinstance(val, dict):
                for kk, vv in val.items():
                    out[f"{key}_{kk}"] = vv
            else:
                out[key] = val
        for blk in ("block0", "block1"):
            kb, kn = out.get(f"{blk}_kappa_B0_kappa_global"), out.get(f"{blk}_kappa_new_kappa_global")
            lb = out.get(f"{blk}_kappa_B0_lam_min")
            if kb is not None and kn:
                out[f"{blk}_gain"] = (float("nan") if (lb is not None and lb <= 0) or not kn > 0
                                      else kb / kn)
        print("GAIN " + json.dumps(out, default=str), flush=True)
        if args.out:
            write_row(args.out, out)
        return

    # certificate family: compiled once per process, used for acceptance of every arm
    t0 = time.perf_counter()
    comp = prob.compile(tol=args.tol, max_blocks=1, block=100)
    log(f"# certificate family compiled in {time.perf_counter() - t0:.1f}s: {comp}")

    if args.family == "F4":
        rows = repeated_solves(args, prob, spec, comp, data, log)
        for r in rows:
            r.update(arm=args.arm, name=spec.name, tag=args.tag, eta=args.eta, q=args.q)
        if args.out:
            exists = os.path.exists(args.out)
            with open(args.out, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                if not exists: w.writeheader()
                w.writerows(rows)
        return

    row = run_arm(args, prob, spec, comp, data, log)
    row["tag"] = args.tag
    log("RESULT " + json.dumps({k: v for k, v in row.items() if k != "history"}, default=str))
    if args.out:
        try:
            write_row(args.out, row)
        except Exception as exc:                          # a completed measurement is never lost to CSV trouble
            fallback = args.out.replace(".csv", "") + "_rows.jsonl"
            with open(fallback, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            print(f"CSV WRITE FAILED ({type(exc).__name__}: {exc}); row appended to {fallback}", flush=True)


if __name__ == "__main__":
    main()
