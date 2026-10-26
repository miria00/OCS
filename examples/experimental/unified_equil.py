"""Unified block-metric equilibration: ONE equilibration over a block-diagonal
dual metric M (blocks M_i, each declaring its class) and a diagonal primal
scaling E, alternating exact block minimizations.

Research prototype, outside the shipped package.  `ocs/` is untouched: the
shipped engine is one arm and supplies the ONLY acceptance test, the compiled
family's public certificate  CompiledFamily.certificate(x, y, data).

--------------------------------------------------------------------------
THE PROBLEM
--------------------------------------------------------------------------
The shipped stochastic matrix-free equilibration (Diamond & Boyd, JOTA 2017,
`ocs/solver/equilibrate_sgd.py`) minimizes over log row/column scalings

    f(u,v) = 1/2 sum_ij A_ij^2 e^{2u_i+2v_j} - r 1^T u - s 1^T v + (g/2)||(u,v)||^2

with r = sqrt(n/m), s = sqrt(m/n)  (so m r = n s, the feasibility condition),
rows of a coupled block tied to one scalar.  Writing Sigma = diag(e^{2u}) and
T = diag(e^{2v}) that objective is, up to the ridge,

    F(Sigma, T) = 1/2 tr(Sigma K T K^T) - (r/2) sum_i log Sigma_ii
                                        - (s/2) sum_j log T_jj .

REPLACE the diagonal Sigma by a BLOCK-DIAGONAL Sigma with SPD blocks Sigma_i of
size k, each constrained to a declared class, and the same objective reads

    minimize   F(Sigma, T) = 1/2 tr(Sigma K T K^T)
                             - (r/2) sum_i logdet(Sigma_i)
                             - (s/2) sum_j log T_jj
    over       T = diag(t) > 0,  Sigma_i in C_i,

    C_i = { s I_k : s > 0 }              "scalar"   (one scalar per k-block)
        = { diag(d) : d > 0 }            "diag"     (one per row)
        = { S : S = S^T > 0 }            "full"     (a small k x k metric)
        and, at term granularity, one scalar for a whole term  "tied"
        (what the shipped compile does today).

`logdet` is the exact generalization of `sum log Sigma_ii`: for a diagonal block
the two coincide, and Sigma_i = e^{2 U_i} makes -(r/2) logdet Sigma_i = -r tr(U_i),
the block form of -r 1^T u.

--------------------------------------------------------------------------
THE ALGORITHM.  Two blocks, each with a CLOSED FORM.
--------------------------------------------------------------------------
Write G_i(T) = K_i T K_i^T, the k x k Gram of block i's rows under the current
primal scaling, and c_j(Sigma) = (K^T Sigma K)_jj, the squared column norms of
Sigma^{1/2} K.

  Sigma-step (T fixed).  The blocks do not interact; each solves
      minimize  1/2 tr(Sigma_i G_i) - (r/2) logdet Sigma_i   over  Sigma_i in C_i
  whose unique solution is
      full    Sigma_i = r G_i^{-1}                     (inverse of the Gram)
      diag    Sigma_i = r diag(G_i)^{-1}               (inverse of its diagonal)
      scalar  Sigma_i = (r k / tr G_i) I               (1 / mean eigenvalue)
      tied    Sigma   = (r m_term / sum_i tr G_i) I    (over the whole term)
  i.e. M_i = Sigma_i^{1/2} is the inverse square root of the Gram, of its
  diagonal, or of its mean eigenvalue: exactly the three closed forms.  The
  "mean eigenvalue" is tr(G_i)/k (stated, as asked).

  T-step (Sigma fixed).  Separable in j, minimizer  t_j = s / c_j(Sigma).
  This is the exact minimizer of the same convex column-scaling problem the
  shipped method solves by SGD, with the row side now measured THROUGH Sigma.

  Gauge.  (Sigma, T) -> (e^{2a} Sigma, e^{-2a} T) leaves both the objective
  (because m r = n s) and the preconditioned operator Sigma^{1/2} K T^{1/2}
  unchanged, but not the primal/dual weight PDHG sees.  The shipped ridge
  selects a point on that ray; here the ridge is dropped (the block steps are
  exact) and the ray is fixed explicitly to its minimum-log-norm point,
      a = (sum_j (1/2) log t_j - sum over all dual eigenvalues of (1/2) log lam)
          / (m + n),
  which is what the ridge selects in the limit g -> 0.  `--gauge match_shipped`
  is an ablation that instead reproduces the shipped method's weight.

WHAT IS GUARANTEED AND WHAT IS NOT.  Each half-step is the exact unique
minimizer of a convex problem in its block, so F is non-increasing along the
alternation and, being bounded below on the gauge-fixed ray, converges.  That
is ALL that is claimed.  Joint convexity holds when every class is scalar or
diagonal (then this is exactly the Diamond-Boyd problem in log coordinates and
the alternation is block coordinate descent on a jointly convex function); for
the full class the Sigma-block is convex in Sigma_i and the T-block is convex in
log t, but joint convexity is NOT claimed and neither is convergence of the
ITERATES.  Whether they converge, cycle or drift is measured, not assumed
(`--diag-only`).

--------------------------------------------------------------------------
EVERY QUANTITY COMES FROM PRODUCTS
--------------------------------------------------------------------------
  G_i(T)  : q_gram shared Rademacher FORWARD probes, v = K T^{1/2} xi,
            G_hat_i = (1/q) sum_r v_i v_i^T   (`sketch_grams`, from the pilot).
            One forward application of the stacked K per probe.
  c_j     : q_col Rademacher ADJOINT probes, z = K^T Sigma^{1/2} s,
            c_hat = (1/q) sum_r z^2.  Unbiased: E z_j^2 = (K^T Sigma K)_jj.
            One adjoint application of the stacked K per probe.
  ||B||   : 50 power iterations on T^{1/2} K^T Sigma K T^{1/2}, 1 forward and
            1 adjoint each; both steps then scaled by 0.9 / estimate.
  F       : 1/2 tr(Sigma K T K^T) from 8 fixed-seed forward probes (diagnostic
            only, reported separately from the method's budget).
Small per-block k x k matrices ARE formed (eigendecomposition of a (B,k,k)
batch).  K itself is never formed, densified, factored or indexed.

--------------------------------------------------------------------------
ARMS
--------------------------------------------------------------------------
  A0     shipped engine, default path (Halpern + restarts)
  S      research loop, shipped SMFE scalings alone           (brief's arm 1)
  TS     shipped SMFE, then the pilot's whitening on top      (brief's arm 2)
  TScf   shipped SMFE, then ONE closed-form Sigma-step on top (decomposition)
  U      the unified alternation, --rounds R                  (brief's arm 3)
All prototype arms run the SAME loop, the SAME prox, the SAME tolerance and the
SAME 0.9/||B|| rule, so the only difference between them is the metric.

Usage (always on a remote GPU host):
  python unified_equil.py --family F1 --variant overlap --n 200000 --rho 0.99 \
      --arm U --rounds 3 --classes full --out results.csv
  python unified_equil.py --family F1M --diag-only --rounds 30 --out conv.csv
"""
from __future__ import annotations

import os, sys, time, json, csv, math, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import blockmetric_pilot as BP
from blockmetric_pilot import (sketch_grams, batched_eigh, shipped_scaling, Loop,
                               power_norm_metric, scale_metrics, float64_objective,
                               device_peak_bytes, block_gram_condition, write_row,
                               _split_metrics)

DELTA_FRAC = 1e-3        # ridge inside the Gram: delta = DELTA_FRAC * tr(G)/k
KAPPA_MAX = 100.0        # condition cap on each block's metric (pilot value)
LOG_BOX = 6.0            # |log-scaling| <= LOG_BOX, the shipped SMFE's box M
GAMMA_SAFETY = 0.9       # step factor on the power-iteration norm estimate
POWER_ITERS = 50
OBJ_PROBES = 8           # diagnostic probes for tr(Sigma K T K^T)
METRIC_F64 = False       # build the per-block metric in float64 on the host (diagnostic)

CLASS_NAMES = ("scalar", "diag", "full")


def _source_hash():
    import hashlib
    h = hashlib.sha256()
    for name in sorted(("unified_equil.py", "unified_equil_models.py",
                        "blockmetric_pilot.py", "blockmetric_models.py")):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


# ============================================================ class declarations
def resolve_classes(spec, spec_str):
    """-> list per dual term of either a term-level string ("tied") or a (B,) int
    array of per-block codes (0 scalar, 1 diag, 2 full).

    `spec_str` is a comma list, one entry per dual term, or ONE entry applied to
    every group_l2 term (sum_squares terms are always per-row diagonal, which is
    what the shipped compile gives them).  The entry "mixed" uses the family's
    own declaration (`spec.block_classes`) and fails loudly if it has none.
    """
    parts = [p.strip() for p in spec_str.split(",")]
    out = []
    gl2 = 0
    for j, d in enumerate(spec.dual_blocks):
        if d["kind"] == "sum_squares":
            out.append("rowdiag")
            continue
        want = parts[j] if len(parts) == len(spec.dual_blocks) else parts[min(gl2, len(parts) - 1)]
        gl2 += 1
        B = d["op"].shape[0] // d["k"]
        if want == "tied":
            out.append("tied")
        elif want in CLASS_NAMES:
            out.append(np.full(B, CLASS_NAMES.index(want), np.int32))
        elif want == "mixed":
            bc = getattr(spec, "block_classes", None)
            if bc is None:
                raise SystemExit(f"family {spec.family} declares no per-block classes; "
                                 "'mixed' is not available for it")
            codes = np.asarray(bc[j], np.int32)
            if codes.shape != (B,):
                raise SystemExit(f"block_classes[{j}] has shape {codes.shape}, expected {(B,)}")
            out.append(codes)
        else:
            raise SystemExit(f"unknown class '{want}'")
    return out


def class_tag(cls):
    if isinstance(cls, str):
        return cls
    cnt = np.bincount(cls, minlength=3)
    if (cnt > 0).sum() == 1:
        return CLASS_NAMES[int(np.argmax(cnt))]
    return "mixed(" + "/".join(f"{CLASS_NAMES[i]}:{int(cnt[i])}" for i in range(3)) + ")"


def needs_full(cls):
    return (not isinstance(cls, str)) and bool(np.any(cls == 2))


# ============================================================ closed-form metric
def closed_form_metric(G, cls, r_tgt, kappa_max=None, delta_frac=None,
                       log_box=LOG_BOX, delta_abs=None):
    """Exact minimizer of  1/2 tr(Sigma G) - (r/2) logdet Sigma  over the declared
    class, per block, with a ridge and a condition cap.

    W_i = G_i + delta_i I,   delta_i = delta_frac * max(tr(G_i)/k, floor),
    mbar_i = tr(W_i)/k (the mean eigenvalue).  Writing the metric's eigenvalues
    as w = (r/mbar) * clip(mbar/x, 1/kappa, kappa):

        full   x = eigenvalues of W_i        (Sigma_i = r W_i^{-1})
        diag   x = diagonal of W_i           (Q_i = I)
        scalar x = mbar_i                    (Q_i = I, w = r/mbar_i, no clipping
                                              can bind: mbar/mbar = 1)
        tied   one scalar over the term, r * (B k) / sum_i tr(W_i)

    Returns ("diag", w (B,k)) when no block is full, else ("full", Q (B,k,k),
    w (B,k)), plus a small diagnostics dict.  The clip and the box are the only
    regularization; both are reported when they bind.
    """
    kappa_max = KAPPA_MAX if kappa_max is None else float(kappa_max)
    delta_frac = DELTA_FRAC if delta_frac is None else float(delta_frac)
    G = jnp.asarray(G, jnp.float32)
    B, k, _ = G.shape
    if delta_abs is None:
        trk = jnp.trace(G, axis1=1, axis2=2) / k
        floor = 1e-6 * jnp.maximum(jnp.median(trk), 1e-30)
        delta = delta_frac * jnp.maximum(trk, floor)
    else:
        delta = jnp.asarray(delta_abs, jnp.float32)
    eye = jnp.eye(k, dtype=jnp.float32)
    W = 0.5 * (G + jnp.swapaxes(G, 1, 2)) + delta[:, None, None] * eye
    mbar = jnp.trace(W, axis1=1, axis2=2) / k                       # (B,)
    base = r_tgt / jnp.maximum(mbar, 1e-30)                         # (B,)

    if isinstance(cls, str) and cls == "tied":
        s = r_tgt * (B * k) / jnp.maximum(jnp.sum(jnp.trace(W, axis1=1, axis2=2)), 1e-30)
        w = jnp.broadcast_to(s, (B, k))
        lo, hi = math.exp(-2 * log_box), math.exp(2 * log_box)
        box_frac = float(jnp.mean((w < lo) | (w > hi)))
        return ("diag", jnp.clip(w, lo, hi)), dict(clip_frac=0.0, box_frac=box_frac, delta=delta)
    codes = jnp.asarray(cls, jnp.int32)
    any_full = bool(np.any(np.asarray(cls) == 2))
    dW = jnp.diagonal(W, axis1=1, axis2=2)                          # (B,k)
    if any_full and METRIC_F64:
        # Diagnostic path: the k x k eigendecomposition in float64 on the host.
        # The blocks are small, so this is a few hundred ms; it exists to separate
        # "the alternation drifts" from "float32 cannot invert this block".
        Wn = np.asarray(W, np.float64)
        om_n, U_n = np.linalg.eigh(Wn)
        omega = jnp.asarray(om_n, jnp.float32)
        U = jnp.asarray(U_n, jnp.float32)
        mbar_n = np.asarray(mbar, np.float64)
        dW_n = np.asarray(dW, np.float64)
        cd = np.asarray(cls)
        x_n = np.where(cd[:, None] == 2, om_n, np.where(cd[:, None] == 1, dW_n, mbar_n[:, None]))
        ratio_n = mbar_n[:, None] / np.maximum(x_n, 1e-300)
        if math.isfinite(kappa_max):
            ratio_n = np.clip(ratio_n, 1.0 / kappa_max, kappa_max)
        w64 = (r_tgt / np.maximum(mbar_n, 1e-300))[:, None] * ratio_n
        lo, hi = math.exp(-2 * log_box), math.exp(2 * log_box)
        box_frac = float(np.mean((w64 < lo) | (w64 > hi)))
        w = jnp.asarray(np.clip(w64, lo, hi), jnp.float32)
        Q = jnp.where((jnp.asarray(cls, jnp.int32) == 2)[:, None, None], U,
                      jnp.broadcast_to(eye, (B, k, k)))
        return ("full", Q, w), dict(clip_frac=0.0, box_frac=box_frac, delta=delta)
    if any_full:
        omega, U = batched_eigh(W)                                  # ascending
    else:
        omega, U = dW, jnp.broadcast_to(eye, (B, k, k))
    # per-class x (the quantity the metric inverts)
    x = jnp.where(codes[:, None] == 2, omega,
                  jnp.where(codes[:, None] == 1, dW, mbar[:, None]))
    ratio = mbar[:, None] / jnp.maximum(x, 1e-30)
    ratio_c = ratio if not math.isfinite(kappa_max) else jnp.clip(ratio, 1.0 / kappa_max, kappa_max)
    clip_frac = float(jnp.mean(jnp.abs(ratio_c - ratio) > 1e-6 * jnp.maximum(ratio, 1e-30)))
    w = base[:, None] * ratio_c
    lo, hi = math.exp(-2 * log_box), math.exp(2 * log_box)
    box_frac = float(jnp.mean((w < lo) | (w > hi)))
    w = jnp.clip(w, lo, hi)
    # a block that is not "full" keeps Q = I, so its metric is genuinely diagonal
    if any_full:
        Q = jnp.where((codes == 2)[:, None, None], U, jnp.broadcast_to(eye, (B, k, k)))
        return ("full", Q, w), dict(clip_frac=clip_frac, box_frac=box_frac, delta=delta)
    return ("diag", w), dict(clip_frac=clip_frac, box_frac=box_frac, delta=delta)


def pilot_whiten(G, s0, cls, kappa_max=None, delta_frac=None):
    """The pilot's two-stage rule, per class: R_i = (tr W_i/k) W_i^{-1} with the
    eigenvalues clipped to [1/kappa, kappa], Sigma_i = s0 R_i.  It is
    TRACE-PRESERVING per block, so a scalar or tied declaration leaves the
    shipped tied sigma untouched (R_i = I) -- which is exactly the pilot's
    behaviour and the reason it is a two-stage heuristic rather than an
    equilibration."""
    kappa_max = KAPPA_MAX if kappa_max is None else float(kappa_max)
    delta_frac = DELTA_FRAC if delta_frac is None else float(delta_frac)
    G = jnp.asarray(G, jnp.float32)
    B, k, _ = G.shape
    trk = jnp.trace(G, axis1=1, axis2=2) / k
    floor = 1e-6 * jnp.maximum(jnp.median(trk), 1e-30)
    delta = delta_frac * jnp.maximum(trk, floor)
    eye = jnp.eye(k, dtype=jnp.float32)
    W = 0.5 * (G + jnp.swapaxes(G, 1, 2)) + delta[:, None, None] * eye
    mbar = jnp.trace(W, axis1=1, axis2=2) / k
    dW = jnp.diagonal(W, axis1=1, axis2=2)
    if isinstance(cls, str):                                   # tied -> nothing to do
        return ("diag", jnp.broadcast_to(jnp.asarray(s0, jnp.float32), (B, k))), dict(clip_frac=0.0)
    codes = jnp.asarray(cls, jnp.int32)
    any_full = bool(np.any(np.asarray(cls) == 2))
    if any_full:
        omega, U = batched_eigh(W)
    else:
        omega, U = dW, jnp.broadcast_to(eye, (B, k, k))
    x = jnp.where(codes[:, None] == 2, omega,
                  jnp.where(codes[:, None] == 1, dW, mbar[:, None]))
    ratio = mbar[:, None] / jnp.maximum(x, 1e-30)
    if math.isfinite(kappa_max):
        ratio = jnp.clip(ratio, 1.0 / kappa_max, kappa_max)
    w = jnp.asarray(s0, jnp.float32) * ratio
    if any_full:
        Q = jnp.where((codes == 2)[:, None, None], U, jnp.broadcast_to(eye, (B, k, k)))
        return ("full", Q, w), dict(clip_frac=0.0)
    return ("diag", w), dict(clip_frac=0.0)


# ============================================================ probe estimators
_COL_CACHE = {}


def colnorm_estimate(spec, metrics, q, seed=0):
    """c_hat_j = (1/q) sum_r ((K^T Sigma^{1/2} s^(r))_j)^2, s Rademacher over the
    m dual rows.  Unbiased for (K^T Sigma K)_jj = squared column norm of
    Sigma^{1/2} K.  Cost: q ADJOINT applications of the stacked operator."""
    kinds, arrays = _split_metrics(metrics)
    key = (id(spec), kinds, q, seed)
    if key not in _COL_CACHE:
        ops = [d["op"] for d in spec.dual_blocks]
        dims = [op.shape[0] for op in ops]
        bkinds = [d["kind"] for d in spec.dual_blocks]
        ks = [d["k"] for d in spec.dual_blocks]
        n, m = spec.n, sum(dims)

        @jax.jit
        def run(arrays):
            def body(_i, carry):
                key_, acc = carry
                key_, kk = jax.random.split(key_)
                sgn = jnp.where(jax.random.bernoulli(kk, 0.5, (m,)), 1.0, -1.0).astype(jnp.float32)
                z = jnp.zeros((n,), jnp.float32)
                a = 0
                for j, op in enumerate(ops):
                    sj = sgn[a:a + dims[j]]; a += dims[j]
                    if bkinds[j] == "sum_squares":
                        u = jnp.sqrt(arrays[j][0]) * sj
                    else:
                        R = sj.reshape(-1, ks[j])
                        if kinds[j] in ("scalar", "diag"):
                            u = (jnp.sqrt(arrays[j][0]) * R).reshape(-1)
                        else:
                            Q, w = arrays[j]
                            u = jnp.einsum("bij,bj->bi", Q,
                                           jnp.sqrt(w) * jnp.einsum("bji,bj->bi", Q, R)).reshape(-1)
                    z = z + op.rmatvec(u)
                return key_, acc + z * z
            _k, acc = jax.lax.fori_loop(0, q, body,
                                        (jax.random.PRNGKey(seed + 7777), jnp.zeros((n,), jnp.float32)))
            return acc / q
        _COL_CACHE[key] = run
    return _COL_CACHE[key](arrays)


_TR_CACHE = {}


def trace_estimate(spec, T, metrics, q=OBJ_PROBES, seed=999):
    """tr(Sigma K T K^T) = E ||Sigma^{1/2} K T^{1/2} xi||^2, xi Rademacher.
    DIAGNOSTIC only (the equilibration objective); a FIXED seed is used so the
    same probes measure every round and every arm (common random numbers)."""
    kinds, arrays = _split_metrics(metrics)
    key = (id(spec), kinds, q, seed)
    if key not in _TR_CACHE:
        ops = [d["op"] for d in spec.dual_blocks]
        bkinds = [d["kind"] for d in spec.dual_blocks]
        ks = [d["k"] for d in spec.dual_blocks]
        n = spec.n

        @jax.jit
        def run(T, arrays):
            st = jnp.sqrt(jnp.broadcast_to(T, (n,)))

            def body(_i, carry):
                key_, acc = carry
                key_, kk = jax.random.split(key_)
                xi = jnp.where(jax.random.bernoulli(kk, 0.5, (n,)), 1.0, -1.0).astype(jnp.float32)
                tot = jnp.float32(0.0)
                for j, op in enumerate(ops):
                    v = op.matvec(st * xi)
                    if bkinds[j] == "sum_squares":
                        tot = tot + jnp.sum(arrays[j][0] * v * v)
                    else:
                        R = v.reshape(-1, ks[j])
                        if kinds[j] in ("scalar", "diag"):
                            tot = tot + jnp.sum(arrays[j][0] * R * R)
                        else:
                            Q, w = arrays[j]
                            Rt = jnp.einsum("bji,bj->bi", Q, R)
                            tot = tot + jnp.sum(w * Rt * Rt)
                return key_, acc + tot
            _k, acc = jax.lax.fori_loop(0, q, body, (jax.random.PRNGKey(seed), jnp.float32(0.0)))
            return acc / q
        _TR_CACHE[key] = run
    return float(_TR_CACHE[key](jnp.asarray(T, jnp.float32), arrays))


def metric_logdet_sum(metrics, spec):
    """sum_i logdet Sigma_i over every dual row, and the eigenvalue range.

    A term carried as ("scalar", s) is ONE number standing for the whole term's
    rows, so its contribution is m_term * log s, not log s.  Getting this wrong
    would mis-set the gauge for the shipped arm."""
    tot, lo, hi, cnt = 0.0, np.inf, -np.inf, 0
    for j, met in enumerate(metrics):
        if met[0] == "scalar":
            mrows = spec.dual_blocks[j]["op"].shape[0]
            a = np.full(mrows, float(np.asarray(met[1], np.float64)))
        elif met[0] == "vec":
            a = np.asarray(met[1], np.float64).reshape(-1)
        elif met[0] == "diag":
            a = np.asarray(met[1], np.float64).reshape(-1)
        else:
            a = np.asarray(met[2], np.float64).reshape(-1)
        tot += float(np.sum(np.log(np.maximum(a, 1e-300))))
        lo, hi, cnt = min(lo, float(a.min())), max(hi, float(a.max())), cnt + a.size
    return tot, lo, hi, cnt


def exact_trace(spec, metrics, Gex):
    """tr(Sigma K T K^T) = sum_i tr(Sigma_i G_i), in float64, from exact Grams."""
    tot = 0.0
    for j, met in enumerate(metrics):
        G = np.asarray(Gex[j], np.float64)
        if met[0] == "vec":
            tot += float(np.sum(np.asarray(met[1], np.float64) * G.reshape(-1)))
        elif met[0] == "diag":
            tot += float(np.sum(np.asarray(met[1], np.float64) *
                                np.einsum("bii->bi", G)))
        else:
            Q, w = np.asarray(met[1], np.float64), np.asarray(met[2], np.float64)
            tot += float(np.sum(w * np.einsum("bji,bjl,bli->bi", Q, G, Q)))
    return tot


def equil_objective(spec, T, metrics, r_tgt, s_tgt, probes=OBJ_PROBES, Gex=None):
    """F = 1/2 tr(Sigma K T K^T) - (r/2) sum_i logdet Sigma_i - (s/2) sum_j log t_j."""
    tr = exact_trace(spec, metrics, Gex) if Gex is not None else trace_estimate(spec, T, metrics, q=probes)
    ld, lo, hi, _c = metric_logdet_sum(metrics, spec)
    t = np.asarray(jnp.broadcast_to(jnp.asarray(T), (spec.n,)), np.float64)
    lt = float(np.sum(np.log(np.maximum(t, 1e-300))))
    return dict(F=0.5 * tr - 0.5 * r_tgt * ld - 0.5 * s_tgt * lt,
                tr=tr, logdet_sigma=ld, logsum_tau=lt, sig_min=lo, sig_max=hi,
                tau_min=float(t.min()), tau_max=float(t.max()))


# ============================================================ exact references
def _block_chain(op):
    """Decompose a (stacked) block-select operator into [(perm, C), ...] if it is
    one.  Diagnostic reference only; returns None for anything else."""
    if hasattr(op, "K1") and hasattr(op, "K2"):
        a, b = _block_chain(op.K1), _block_chain(op.K2)
        return None if (a is None or b is None) else a + b
    if hasattr(op, "perm") and hasattr(op, "dense_blocks"):
        return [(np.asarray(op.perm), op.dense_blocks())]
    return None


def _f2_exact_coldiag(op, Sall):
    """F2 (`ChannelBlur`): column j = (channel c, pixel p) of K has
    (Sigma^{1/2} K)_{(p',c'),(c,p)} = [Sigma_{p'}^{1/2} M]_{c'c} h_{p'p}, so
    (K^T Sigma K)_{jj} = sum_{p'} h_{p'p}^2 (M^T Sigma_{p'} M)_{cc}: one
    convolution with the SQUARED kernel, whose spectrum the operator already
    carries as H2.  h is even, so h^2 is a symmetric circulant and the same
    convolution serves both directions.  Diagnostic reference only."""
    M = np.asarray(op.M, np.float64)
    W = np.einsum("ic,pij,jd->pcd", M, np.asarray(Sall, np.float64), M)
    field = np.einsum("pcc->pc", W).T.reshape(op.c, op.L, op.L)
    e = np.real(np.fft.ifft2(np.fft.fft2(field) * np.asarray(op.H2)[None]))
    return e.reshape(-1)


def _sigma_blocks(met, k):
    if met[0] == "diag":
        w = np.asarray(met[1], np.float64)
        S = np.zeros((w.shape[0], k, k)); S[:, np.arange(k), np.arange(k)] = w
        return S
    Q, w = np.asarray(met[1], np.float64), np.asarray(met[2], np.float64)
    return np.einsum("bij,bj,blj->bil", Q, w, Q)


def exact_coldiag(spec, metrics):
    """diag(K^T Sigma K) exactly, when the family allows it.  Diagnostic
    reference for the adjoint-probe estimator; never used inside an arm."""
    if getattr(spec, "purity_guarded", False):
        return None
    if spec.family == "F2" and len(spec.dual_blocks) == 1:
        return _f2_exact_coldiag(spec.dual_blocks[0]["op"], _sigma_blocks(metrics[0], spec.k))
    if getattr(spec, "exact_coldiag", None) is not None and len(spec.dual_blocks) == 1:
        met = metrics[0]
        B = spec.dual_blocks[0]["op"].shape[0] // spec.k
        k = spec.k
        if met[0] == "diag":
            w = np.asarray(met[1], np.float64)
            S = np.zeros((B, k, k)); S[:, np.arange(k), np.arange(k)] = w
        else:
            Q, w = np.asarray(met[1], np.float64), np.asarray(met[2], np.float64)
            S = np.einsum("bij,bj,blj->bil", Q, w, Q)
        return spec.exact_coldiag(S)
    chain = _block_chain(spec.dual_blocks[0]["op"]) if len(spec.dual_blocks) == 1 else None
    if chain is None:
        return None
    met = metrics[0]
    k = spec.k
    if met[0] == "diag":
        w = np.asarray(met[1], np.float64)
        Sall = np.zeros((w.shape[0], k, k)); Sall[:, np.arange(k), np.arange(k)] = w
    else:
        Q, w = np.asarray(met[1], np.float64), np.asarray(met[2], np.float64)
        Sall = np.einsum("bij,bj,blj->bil", Q, w, Q)
    out = np.zeros(spec.n); a = 0
    for perm, C in chain:
        B = C.shape[0]
        S = Sall[a:a + B]; a += B
        A = np.einsum("bji,bjl,blm->bim", C, S, C)
        d = np.einsum("bii->bi", A).reshape(-1)
        np.add.at(out, perm, d)
    return out


def exact_gram_all(spec, tau):
    """Per-term exact Grams K_j T K_j^T when the family exposes them, else None."""
    out = []
    for d in spec.dual_blocks:
        if d["kind"] == "sum_squares" or d["exact_gram"] is None:
            return None
        out.append(np.asarray(d["exact_gram"](np.asarray(tau), 1.0), np.float64))
    return out


# ============================================================ the alternation
def unified_equilibration(spec, classes, rounds, q_gram, q_col, *, seed=0,
                          probe_mode="fixed", gauge="canonical", tau_ref=None,
                          source="probe", log=print, record=True, obj_probes=OBJ_PROBES,
                          delta_mode="fixed"):
    """delta_mode
      "perround"  delta_i = delta_frac * tr(G_i)/k recomputed every round (the
                  block-metric pilot's rule).  The Sigma-step then minimizes a
                  perturbed objective whose perturbation MOVES with T, so the
                  alternation is no longer exact block minimization of any one
                  function and the objective can drift upward.  Measured.
      "fixed"     delta_i frozen at its round-1 value.  The pair then exactly
                  minimizes  1/2 tr(Sigma (K T K^T + Delta)) - (r/2) sum logdet
                  Sigma_i - (s/2) sum log t_j  in each block (the T-step is
                  unchanged because Delta does not involve t), so the
                  alternation is exact block coordinate descent again.
      "zero"      no ridge.  Exact, but fails on a singular block Gram (the
                  3-D TV gradient annihilates constants).

    R rounds of [Sigma-step ; T-step], from T = I.  Returns
    (T, metrics, history, counts, timings).
    """
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    r_tgt = float(np.sqrt(n / m))
    s_tgt = float(np.sqrt(m / n))
    tau = jnp.ones((n,), jnp.float32)
    metrics = None
    counts = dict(fwd=0, adj=0)
    t_gram = t_metric = t_col = 0.0
    hist = []
    prev_logtau = np.zeros(n)
    prev_logw = None
    delta_frozen = [None] * len(spec.dual_blocks)

    for rd in range(1, rounds + 1):
        gseed = seed if probe_mode == "fixed" else seed + 1000 * rd
        # ---- Sigma-step
        Gs = None
        if source == "exact":
            Gs = exact_gram_all(spec, tau)
            if Gs is None:
                raise SystemExit("--source exact: this family exposes no exact Grams")
        new_metrics, diags = [], []
        for j, d in enumerate(spec.dual_blocks):
            k = d["k"]
            t0 = time.perf_counter()
            if source == "exact":
                G = jnp.asarray(Gs[j], jnp.float32)
            else:
                G = sketch_grams(d["op"], tau, 1.0, k, q_gram, seed=gseed)
            jax.block_until_ready(G)
            t_gram += time.perf_counter() - t0
            t0 = time.perf_counter()
            cls = classes[j]
            if isinstance(cls, str) and cls == "rowdiag":   # sum_squares rows: per-row
                dg = jnp.maximum(jnp.diagonal(G, axis1=1, axis2=2).reshape(-1), 0.0)
                fl = DELTA_FRAC * jnp.maximum(jnp.mean(dg), 1e-30)
                w = jnp.clip(r_tgt / (dg + fl), math.exp(-2 * LOG_BOX), math.exp(2 * LOG_BOX))
                new_metrics.append(("vec", w)); diags.append(dict(clip_frac=0.0, box_frac=0.0))
            else:
                dabs = None if delta_mode == "perround" else delta_frozen[j]
                dfrac = 0.0 if delta_mode == "zero" else None
                met, dg = closed_form_metric(G, cls, r_tgt, delta_frac=dfrac, delta_abs=dabs)
                if delta_frozen[j] is None:
                    delta_frozen[j] = dg["delta"]
                new_metrics.append(met); diags.append(dg)
            jax.block_until_ready(new_metrics[-1][1])
            t_metric += time.perf_counter() - t0
        if source != "exact":
            counts["fwd"] += q_gram
        metrics = new_metrics
        F_sigma = (equil_objective(spec, tau, metrics, r_tgt, s_tgt, probes=obj_probes,
                                   Gex=Gs)["F"] if record else float("nan"))

        # ---- T-step
        t0 = time.perf_counter()
        if source == "exact":
            c = exact_coldiag(spec, metrics)
            if c is None:
                raise SystemExit("--source exact: this family has no exact column diagonal")
            c = jnp.asarray(c, jnp.float32)
        else:
            c = colnorm_estimate(spec, metrics, q_col, seed=gseed)
            counts["adj"] += q_col
        jax.block_until_ready(c)
        floor = 1e-12 * jnp.maximum(jnp.mean(c), 1e-30)
        tau = jnp.clip(s_tgt / jnp.maximum(c, floor),
                       math.exp(-2 * LOG_BOX), math.exp(2 * LOG_BOX))
        t_col += time.perf_counter() - t0

        # ---- gauge
        a = _apply_gauge_value(spec, tau, metrics, m, n, gauge, tau_ref)
        tau = tau * math.exp(-2 * a)
        metrics, _ = scale_metrics(metrics, tau, math.exp(2 * a))
        metrics = metrics if isinstance(metrics, list) else list(metrics)

        if record:
            rec = dict(round=rd, gauge_a=a, fwd=counts["fwd"], adj=counts["adj"], F_after_sigma=F_sigma)
            Gnew = exact_gram_all(spec, tau) if source == "exact" else None
            rec.update(equil_objective(spec, tau, metrics, r_tgt, s_tgt, probes=obj_probes, Gex=Gnew))
            rec["F_after_tau"] = rec["F"]
            lt = np.log(np.maximum(np.asarray(tau, np.float64), 1e-300))
            lw = np.concatenate([np.log(np.maximum(np.asarray(mt[1] if mt[0] in ("vec", "diag") else mt[2],
                                                              np.float64).reshape(-1), 1e-300))
                                 for mt in metrics])
            rec["dlog_tau_inf"] = float(np.max(np.abs(lt - prev_logtau)))
            rec["dlog_tau_rms"] = float(np.sqrt(np.mean((lt - prev_logtau) ** 2)))
            rec["dlog_sigma_inf"] = float("nan") if prev_logw is None else float(np.max(np.abs(lw - prev_logw)))
            rec["dlog_sigma_rms"] = float("nan") if prev_logw is None else float(np.sqrt(np.mean((lw - prev_logw) ** 2)))
            rec["clip_frac"] = float(np.max([d["clip_frac"] for d in diags]))
            rec["box_frac"] = float(np.max([d.get("box_frac", 0.0) for d in diags]))
            prev_logtau, prev_logw = lt, lw
            hist.append(rec)
            if log:
                log(f"   round {rd:3d}  F_sig={rec['F_after_sigma']:.6e} F={rec['F']:.6e}  "
                    f"dlog_tau={rec['dlog_tau_inf']:.3e} "
                    f"dlog_sig={rec['dlog_sigma_inf']:.3e}  gauge={a:+.4f} "
                    f"clip={rec['clip_frac']:.3f} box={rec['box_frac']:.3f}")
    return tau, metrics, hist, counts, dict(gram_s=t_gram, metric_s=t_metric, col_s=t_col)


def _apply_gauge_value(spec, tau, metrics, m, n, gauge, tau_ref):
    if gauge == "none":
        return 0.0
    lt = np.log(np.maximum(np.asarray(jnp.broadcast_to(tau, (n,)), np.float64), 1e-300))
    if gauge == "match_shipped":
        if tau_ref is None:
            raise SystemExit("--gauge match_shipped needs the shipped scaling")
        lr = np.log(np.maximum(np.asarray(jnp.broadcast_to(jnp.asarray(tau_ref), (n,)), np.float64), 1e-300))
        return 0.5 * (float(np.mean(lt)) - float(np.mean(lr)))
    ld, _lo, _hi, _c = metric_logdet_sum(metrics, spec)
    Lv = 0.5 * float(np.sum(lt))
    Lu = 0.5 * ld
    return (Lv - Lu) / (m + n)


# ============================================================ arm construction
def shipped_metrics(spec, sc, classes):
    """The shipped scalings as the loop's metric list."""
    sigma, slices = sc["sigma"], sc["slices"]
    metrics = []
    for j, d in enumerate(spec.dual_blocks):
        s, e = slices[j]
        sig_j = sigma if sigma.ndim == 0 else sigma[s:e]
        if d["kind"] == "sum_squares":
            metrics.append(("vec", jnp.broadcast_to(sig_j, (e - s,))))
        else:
            s0 = float(sig_j if sig_j.ndim == 0 else sig_j[0])
            if sig_j.ndim and not bool(jnp.all(sig_j == sig_j[0])):
                raise RuntimeError("tied block expected a uniform shipped sigma")
            metrics.append(("scalar", jnp.asarray(s0, jnp.float32)))
    return sc["tau"], metrics


def build_arm(args, spec, classes, log):
    """-> dict(T, metrics, setup_s, counts, timings, hist, extra)."""
    arm = args.arm
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    r_tgt, s_tgt = float(np.sqrt(n / m)), float(np.sqrt(m / n))
    counts = dict(equil_fwd=0, equil_adj=0, norm_fwd=0, norm_adj=0)
    hist, extra, timings = [], {}, dict(equil_s=0.0, gram_s=0.0, metric_s=0.0, col_s=0.0, norm_s=0.0)

    if arm in ("S", "TS", "TScf"):
        sc = shipped_scaling(spec)
        timings["equil_s"] = sc["seconds"]
        # shipped SMFE: iters * probes * (1 fwd + 1 adj) + 15 power iterations
        counts["equil_fwd"] += 80 * 2 + 15
        counts["equil_adj"] += 80 * 2 + 15
        extra["shipped_scaled_norm"] = sc["scaled_norm"]
        extra.update({f"smfe_{k_}": v for k_, v in sc["diag"].items()})
        T, metrics = shipped_metrics(spec, sc, classes)
        if arm == "S" and args.gauge == "canonical" and args.regauge_shipped:
            # control: the shipped metric, moved to the unified method's gauge.
            # It separates "the unified pair is better" from "the unified pair
            # sits at a better primal/dual weight".
            a = _apply_gauge_value(spec, T, metrics, m, n, "canonical", sc["tau"])
            T = T * math.exp(-2 * a)
            metrics, _ = scale_metrics(metrics, T, math.exp(2 * a))
            metrics = list(metrics)
            extra["gauge_a"] = a
        if arm != "S":
            new_metrics = []
            for j, d in enumerate(spec.dual_blocks):
                cls = classes[j]
                if d["kind"] == "sum_squares":
                    new_metrics.append(metrics[j]); continue
                s0 = float(metrics[j][1])
                t0 = time.perf_counter()
                if arm == "TS":
                    G = sketch_grams(d["op"], T, s0, d["k"], args.qgram_blocks, seed=args.seed)
                else:
                    G = sketch_grams(d["op"], T, 1.0, d["k"], args.qgram_blocks, seed=args.seed)
                jax.block_until_ready(G)
                timings["gram_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                met, _dg = (pilot_whiten(G, s0, cls) if arm == "TS"
                            else closed_form_metric(G, cls, r_tgt))
                jax.block_until_ready(met[1])
                timings["metric_s"] += time.perf_counter() - t0
                new_metrics.append(met)
            counts["equil_fwd"] += args.qgram_blocks
            metrics = new_metrics
            if arm == "TScf":
                a = _apply_gauge_value(spec, T, metrics, m, n, args.gauge, sc["tau"])
                T = T * math.exp(-2 * a)
                metrics, _ = scale_metrics(metrics, T, math.exp(2 * a))
                extra["gauge_a"] = a
    elif arm == "U":
        tau_ref = None
        if args.gauge == "match_shipped":
            sc = shipped_scaling(spec)
            tau_ref = sc["tau"]
            log("  NOTE: --gauge match_shipped runs the shipped SMFE purely to read its weight "
                "(an ablation; its cost is reported in equil_fwd/equil_adj)")
            counts["equil_fwd"] += 80 * 2 + 15; counts["equil_adj"] += 80 * 2 + 15
            timings["equil_s"] = sc["seconds"]
        T, metrics, hist, cnt, tm = unified_equilibration(
            spec, classes, args.rounds, args.qgram_blocks, args.qcol, seed=args.seed,
            probe_mode=args.probe_mode, gauge=args.gauge, tau_ref=tau_ref,
            source=args.source, log=log, record=args.record_alt, delta_mode=args.delta_mode)
        counts["equil_fwd"] += cnt["fwd"]; counts["equil_adj"] += cnt["adj"]
        timings.update(gram_s=tm["gram_s"], metric_s=tm["metric_s"], col_s=tm["col_s"])
        if hist:
            extra["gauge_a"] = hist[-1]["gauge_a"]
            extra["clip_frac"] = hist[-1]["clip_frac"]
            extra["box_frac"] = hist[-1]["box_frac"]
    else:
        raise ValueError(arm)

    # one common stability rule for every prototype arm
    t0 = time.perf_counter()
    nu = power_norm_metric(spec, T, metrics, iters=POWER_ITERS)
    timings["norm_s"] = time.perf_counter() - t0
    counts["norm_fwd"] += POWER_ITERS; counts["norm_adj"] += POWER_ITERS
    factor = GAMMA_SAFETY / nu
    metrics, T = scale_metrics(metrics, T, factor)
    extra.update(norm_est=nu, factor=factor)
    setup_s = sum(timings.values())
    return dict(T=T, metrics=list(metrics), setup_s=setup_s, counts=counts,
                timings=timings, hist=hist, extra=extra, r_tgt=r_tgt, s_tgt=s_tgt)


# ============================================================ operator purity
class PurityProxy:
    """Exposes ONLY matvec, rmatvec and shape.  Any other attribute access
    raises, so a run under `--purity-guard` cannot reach the operator's
    internals (`dense_blocks`, `perm`, `H2`, `M`, ...).  `--source exact` and the
    exact column-diagonal reference therefore fail loudly under the guard, which
    is the point: it converts "the method is operator-only" from a claim about
    the source into something the run itself checks."""

    __slots__ = ("_op", "shape")

    def __init__(self, op):
        object.__setattr__(self, "_op", op)
        object.__setattr__(self, "shape", op.shape)

    def matvec(self, x):
        return self._op.matvec(x)

    def rmatvec(self, y):
        return self._op.rmatvec(y)

    def __getattr__(self, name):
        raise AttributeError(
            f"purity guard: the equilibration asked the operator for '{name}'. "
            "Only matvec, rmatvec and shape are allowed.")


def apply_purity_guard(spec):
    spec.purity_guarded = True
    for d in spec.dual_blocks:
        d["op"] = PurityProxy(d["op"])
        d["exact_gram"] = None
    spec.exact_norm = None
    if hasattr(spec, "norm_upper"):
        spec.norm_upper = None
    if hasattr(spec, "exact_coldiag"):
        spec.exact_coldiag = None
    return spec


# ============================================================ families
def build_family(args):
    import blockmetric_models as bm
    if args.family == "F1":
        return bm.blockmetric_f1(n=args.n, rho=args.rho, k=args.k, variant=args.variant,
                                 seed=args.seed, mu=args.mu)
    if args.family == "F2":
        return bm.blockmetric_f2(L=args.L, mixing=args.variant, seed=args.seed, mu=args.mu,
                                 sigma_blur=args.blur, channels=args.channels)
    if args.family == "F3":
        return bm.blockmetric_f3(L=args.L, seed=args.seed)
    if args.family == "F1M":
        import unified_equil_models as um
        return um.unified_f1_mixed(n=args.n, k=args.k, seed=args.seed, mu=args.mu, rho=args.rho)
    raise ValueError(args.family)


# ============================================================ main
def base_row(args, spec, classes):
    return dict(family=spec.family, variant=spec.variant, name=spec.name, n=spec.n, k=spec.k,
                rho=spec.rho, mu=spec.mu, note=getattr(spec, "note", ""), seed=args.seed,
                src_hash=_source_hash(), arm=args.arm, rounds=args.rounds if args.arm == "U" else "",
                classes=args.classes, class_tag="|".join(class_tag(c) for c in classes),
                qgram=args.qgram_blocks, qcol=args.qcol if args.arm == "U" else "",
                probe_mode=args.probe_mode, gauge=args.gauge, source=args.source,
                tol=args.tol, kappa_max=KAPPA_MAX, log_box=LOG_BOX, delta_frac=DELTA_FRAC,
                metric_f64=int(METRIC_F64), delta_mode=args.delta_mode,
                purity_guard=int(args.purity_guard), regauge=int(args.regauge_shipped),
                host=os.uname().nodename, device=jax.devices()[0].device_kind,
                tag=args.tag,
                timer_defs="setup_s=equil+gram+metric+col+norm; iter_s=loop blocks only; "
                           "total_warm_s=setup+loop+verify")


def main():
    global KAPPA_MAX, DELTA_FRAC, METRIC_F64
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="F1", choices=["F1", "F2", "F3", "F1M"])
    ap.add_argument("--variant", default="overlap")
    ap.add_argument("--n", type=int, default=200_000); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024); ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--mu", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="U", choices=["A0", "A0std", "S", "TS", "TScf", "U"])
    ap.add_argument("--classes", default="full",
                    help="tied | scalar | diag | full | mixed, or one per dual term")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--qgram", type=int, default=16, help="Gram probes as a multiple of k")
    ap.add_argument("--qcol", type=int, default=16, help="adjoint probes for the column diagonal")
    ap.add_argument("--probe-mode", default="fixed", choices=["fixed", "fresh"])
    ap.add_argument("--gauge", default="canonical", choices=["canonical", "match_shipped", "none"])
    ap.add_argument("--source", default="probe", choices=["probe", "exact"])
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=2000)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--diag-only", action="store_true", help="run the alternation only, no solve")
    ap.add_argument("--record-alt", action="store_true",
                    help="record the per-round equilibration objective inside a SOLVE arm.  Off by "
                         "default: the objective costs 8 extra forward probes per round, which would "
                         "be charged to the arm's setup time without being part of the method.")
    ap.add_argument("--repeats", type=int, default=1, help="paired repeated solves of the SAME metric")
    ap.add_argument("--kappa-max", type=float, default=100.0)
    ap.add_argument("--regauge-shipped", action="store_true",
                    help="arm S only: move the shipped scalings to the canonical gauge")
    ap.add_argument("--purity-guard", action="store_true",
                    help="wrap every operator so only matvec/rmatvec/shape can be reached")
    ap.add_argument("--delta-mode", default="fixed", choices=["perround", "fixed", "zero"],
                    help="how the per-block ridge is set across outer rounds")
    ap.add_argument("--metric-f64", action="store_true",
                    help="build the per-block metric with a float64 host eigendecomposition")
    ap.add_argument("--obj-probes", type=int, default=OBJ_PROBES,
                    help="probes for the diagnostic objective (ignored when --source exact, "
                         "which evaluates the trace exactly in float64)")
    ap.add_argument("--delta-frac", type=float, default=DELTA_FRAC,
                    help="ridge inside the Gram, delta = this * tr(G)/k")
    ap.add_argument("--out", default=None); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    KAPPA_MAX = float(args.kappa_max)
    DELTA_FRAC = float(args.delta_frac)
    METRIC_F64 = bool(args.metric_f64)
    BP.KAPPA_MAX = KAPPA_MAX
    import ocs
    log = lambda s: print(s, flush=True)
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} jax {jax.__version__} "
        f"ocs {ocs.__version__} at {ocs.__file__} src {_source_hash()}")

    t0 = time.perf_counter()
    prob, spec = build_family(args)
    data = spec.data
    args.qgram_blocks = args.qgram * spec.k
    if args.purity_guard:
        spec = apply_purity_guard(spec)
        log("# purity guard ON: operators expose only matvec / rmatvec / shape")
    classes = resolve_classes(spec, args.classes)
    log(f"# {spec.name}: n={spec.n} built in {time.perf_counter() - t0:.1f}s; "
        f"classes {'|'.join(class_tag(c) for c in classes)}")

    if args.diag_only:
        T, metrics, hist, counts, tm = unified_equilibration(
            spec, classes, args.rounds, args.qgram_blocks, args.qcol, seed=args.seed,
            probe_mode=args.probe_mode, gauge=args.gauge, source=args.source, log=log,
            obj_probes=args.obj_probes, delta_mode=args.delta_mode)
        # conditioning of the preconditioned block Grams, exactly where possible
        Gx = exact_gram_all(spec, T)
        if Gx is not None:
            for j, met in enumerate(metrics):
                if met[0] == "vec":
                    continue
                R = (("diag", np.asarray(met[1])) if met[0] == "diag"
                     else ("full", np.asarray(met[1]), np.asarray(met[2])))
                kb = block_gram_condition(Gx[j], ("scalar", 1.0))
                kn = block_gram_condition(Gx[j], R)
                log(f"  exact block-Gram conditioning term {j}: before {kb} after {kn}")
                for rec in hist[-1:]:
                    rec.update({f"t{j}_kappa_before": kb["kappa_global"],
                                f"t{j}_kappa_after": kn["kappa_global"],
                                f"t{j}_kblock_med_before": kb["kappa_block_median"],
                                f"t{j}_kblock_med_after": kn["kappa_block_median"]})
        # probe estimator vs exact column diagonal, where available
        ce = exact_coldiag(spec, metrics)
        if ce is not None:
            ch = np.asarray(colnorm_estimate(spec, metrics, args.qcol, seed=args.seed + 31), np.float64)
            rel = float(np.linalg.norm(ch - ce) / max(np.linalg.norm(ce), 1e-300))
            log(f"  column-diagonal probe estimator relative error at q={args.qcol}: {rel:.4e}")
            if hist:
                hist[-1]["coldiag_relerr"] = rel
        if args.out:
            for rec in hist:
                row = base_row(args, spec, classes); row.update(rec)
                row.update(equil_fwd=counts["fwd"], equil_adj=counts["adj"],
                           gram_s=tm["gram_s"], metric_s=tm["metric_s"], col_s=tm["col_s"])
                write_row(args.out, row)
        log("DIAG " + json.dumps(hist[-1], default=str))
        return

    t0 = time.perf_counter()
    comp = prob.compile(tol=args.tol, max_blocks=1, block=100)
    log(f"# certificate family compiled in {time.perf_counter() - t0:.1f}s")

    row = base_row(args, spec, classes)

    if args.arm in ("A0", "A0std"):
        variant = "standard" if args.arm == "A0std" else "halpern"
        comp0, sol, cert, tm = BP.run_shipped(prob, data, args.tol, args.max_blocks, variant, log)
        obj, minx = float64_objective(spec, sol.x)
        res = max(cert["r_primal"], cert["r_dual"])
        row.update(iters=sol.iters, status=sol.status, residual=res, r_primal=cert["r_primal"],
                   r_dual=cert["r_dual"], certified=int(res <= args.tol), objective_f64=obj, min_x=minx,
                   setup_s=tm["setup_s"], iter_s=tm["warm_s"], verify_s=0.0,
                   total_warm_s=tm["setup_s"] + tm["warm_s"], cold_solve_s=tm["cold_s"],
                   equil_fwd=80 * 2 + 15, equil_adj=80 * 2 + 15, norm_fwd=0, norm_adj=0,
                   peak_bytes=device_peak_bytes())
        log("RESULT " + json.dumps(row, default=str))
        if args.out:
            write_row(args.out, row)
        return

    # Setup is timed WARM: the metric is built once to pay XLA compilation (of the
    # equilibration, the Gram sketch, the column probe and the power iteration),
    # then built again and that second build is what setup_s reports.  Every arm
    # gets the same treatment, including the shipped one.  The first build's wall
    # time is kept as setup_cold_s.
    cold_built = build_arm(args, spec, classes, lambda *_: None)
    built = build_arm(args, spec, classes, log)
    built["extra"]["setup_cold_s"] = cold_built["setup_s"]
    T, metrics = built["T"], built["metrics"]
    log(f"  metric: " + " | ".join(BP.metric_summary(mt) for mt in metrics))
    log(f"  norm estimate {built['extra']['norm_est']:.5f} -> factor {built['extra']['factor']:.4f}")
    loop = Loop(spec, T, metrics, block=100)

    cold = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
    if cold["total_s"] > 200.0:
        res, timed = cold, "cold"
        log(f"  cold run {cold['total_s']:.1f}s > 200s: reporting the cold run")
    else:
        res, timed = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=log), "warm"
    reps = []
    for _ in range(max(0, args.repeats - 1)):
        rr = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
        reps.append(rr["total_s"])
    bench = loop.bench()
    obj, minx = float64_objective(spec, res["x"])
    c = built["counts"]
    row.update(iters=res["iters"], status=res["status"], residual=res["residual"],
               r_primal=res["cert"]["r_primal"], r_dual=res["cert"]["r_dual"],
               certified=int(res["residual"] <= args.tol), objective_f64=obj, min_x=minx,
               setup_s=built["setup_s"], iter_s=res["iter_s"], verify_s=res["verify_s"],
               total_warm_s=built["setup_s"] + res["total_s"], loop_total_s=res["total_s"],
               cold_solve_s=cold["total_s"], timed_run=timed,
               block_s=bench["block"], prox_block_s=bench["prox"], ops_block_s=bench["ops"],
               prox_s=bench["prox"] * res["blocks"], ops_s=bench["ops"] * res["blocks"],
               repeat_total_s=json.dumps([round(x, 5) for x in reps]),
               equil_fwd=c["equil_fwd"], equil_adj=c["equil_adj"],
               norm_fwd=c["norm_fwd"], norm_adj=c["norm_adj"],
               equil_apps=c["equil_fwd"] + c["equil_adj"],
               total_apps=c["equil_fwd"] + c["equil_adj"] + c["norm_fwd"] + c["norm_adj"],
               peak_bytes=device_peak_bytes())
    row.update({f"t_{k_}": v for k_, v in built["timings"].items()})
    row.update({k_: v for k_, v in built["extra"].items()})
    if built["hist"]:
        row["alt_F"] = json.dumps([round(h["F"], 6) for h in built["hist"]])
        row["alt_dlog_tau"] = json.dumps([float(f"{h['dlog_tau_inf']:.4g}") for h in built["hist"]])
    hist = res["history"]
    step = max(1, len(hist) // 200)
    kept = hist[::step]
    if kept and kept[-1] is not hist[-1]:
        kept.append(hist[-1])
    row["history"] = json.dumps([(it, float(r), round(t, 3)) for it, r, t in kept])
    log(f"  {args.arm}: iters={res['iters']} status={res['status']} residual={res['residual']:.3e} "
        f"setup={built['setup_s']:.2f}s loop={res['iter_s']:.2f}s verify={res['verify_s']:.2f}s "
        f"total={built['setup_s'] + res['total_s']:.2f}s apps={row['total_apps']}")
    log("RESULT " + json.dumps({k_: v for k_, v in row.items() if k_ != "history"}, default=str))
    if args.out:
        try:
            write_row(args.out, row)
        except Exception as exc:
            fb = args.out.replace(".csv", "") + "_rows.jsonl"
            with open(fb, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            print(f"CSV WRITE FAILED ({type(exc).__name__}: {exc}); row appended to {fb}", flush=True)


if __name__ == "__main__":
    main()
