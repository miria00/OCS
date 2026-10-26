"""Stochastic matrix-free equilibration (Diamond & Boyd, JOTA 2017).

Instead of a one-shot estimate of per-coordinate steps, solve the equilibration
problem itself by projected stochastic gradient over row/column log-scalings.
Minimize (their problem 5)

    f(u,v) = 1/2 sum_ij A_ij^2 e^{2u_i+2v_j} - r 1^T u - s 1^T v + (g/2)||(u,v)||^2
    s.t.  ||u||_inf <= M, ||v||_inf <= M,  u_{B_i} constant (scalar-metric blocks),

using only matvec/rmatvec.  Use squared-norm targets r = sqrt(n/m)
(rows) and s = sqrt(m/n) (columns), which satisfy m r = n s; the necessary
condition for the row and column targets to be jointly attainable on a rectangular
operator (unit targets r = s = 1 are infeasible unless m = n).  The gradient in u
is the row-norms^2 of DAE (D=diag e^u, E=diag e^v) minus r plus g*u; its unbiased
estimate is |D A E s|^2 for a random sign vector s (one matvec).  Iterate-averaged
projected SGD converges O(1/t).

The equilibrated scalings give the PDHG diagonal steps tau = e^{2 vbar} (primal,
per column) and sigma = e^{2 ubar} (dual, per row); a final power iteration on the
scaled operator applies the global rescale for the PDHG stability bound.

Rows of a coupled (scalar-class) block are averaged in log space each step, so the
dual scaling stays uniform over that block; the block-admissible coercion done
inside the optimization, not fighting it afterwards.

``tie_columns=True`` applies the same coercion to ALL the COLUMNS: the column logs
are projected onto the constant subspace every step (the Euclidean projection
onto {box} cap {v = a*1}: the mean of the UNCLIPPED tentative logs, clipped),
so the construction chooses the single primal step, and the row scalings that
go with it, for a scalar-class primal block. The final power-iteration rescale
is then measured on that tied metric. The estimator, schedule, averaging,
oracle budget and rescale are unchanged; only the column projection is added.
"""
from __future__ import annotations

from collections import OrderedDict
import numbers
import threading

import jax
import jax.numpy as jnp

from .equilibrate import _power_norm
from .._operands import operand_jit


class _IdentityKey:
    """Retain immutable array/operator identities while a cached program uses them."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.value is other.value


_UNSAFE_STATE = object()
_SMFE_RUN_CACHE = OrderedDict()
_SMFE_CACHE_LIMIT = 8
_SMFE_CACHE_ARRAY_BYTES = 128 * 1024 * 1024
_SMFE_CACHE_LOCK = threading.RLock()
_SMFE_CACHE_STATS = {"builds": 0, "hits": 0, "bypasses": 0}


def _retained_array_bytes(keys):
    """Bound retained input storage from metadata, without reading entries."""
    from ..operators.base import LinearOperator
    arrays, seen = {}, set()

    def visit(key):
        if id(key) in seen:
            return
        seen.add(id(key))
        if isinstance(key, _IdentityKey):
            if isinstance(key.value, jax.Array):
                arrays[id(key.value)] = key.value.nbytes
            elif isinstance(key.value, LinearOperator):
                visit(vars(key.value))
        elif isinstance(key, jax.Array):
            arrays[id(key)] = key.nbytes
        elif isinstance(key, dict):
            for part in key.values():
                visit(part)
        elif isinstance(key, (tuple, list)):
            for part in key:
                visit(part)
    for key in keys:
        visit(key)
    return sum(arrays.values())


def _operator_state_key(value, visiting=None):
    """Conservative metadata key; never reads operator or array entries.

    Built-in operators expose their state as attributes. JAX arrays are
    immutable, so retaining their identity detects replacement without copying
    their contents. Mutable arrays, custom callbacks and opaque state bypass
    reuse. Container contents and nested operators are checked on every call.
    """
    from ..operators.base import LinearOperator

    if value is None or isinstance(value, (str, bytes, bool, numbers.Number)):
        return (type(value), value)
    if isinstance(value, jax.Array):
        return ("array", _IdentityKey(value), value.shape, str(value.dtype),
                str(value.sharding))
    visiting = set() if visiting is None else visiting
    if id(value) in visiting:
        return _UNSAFE_STATE
    visiting.add(id(value))
    try:
        if isinstance(value, (tuple, list)):
            parts = tuple(_operator_state_key(v, visiting) for v in value)
            return _UNSAFE_STATE if any(v is _UNSAFE_STATE for v in parts) else (type(value), parts)
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            parts = tuple((k, _operator_state_key(v, visiting)) for k, v in sorted(value.items()))
            return _UNSAFE_STATE if any(v is _UNSAFE_STATE for _, v in parts) else (dict, parts)
        cls = type(value)
        if (isinstance(value, LinearOperator) and cls.__module__.startswith("ocs.operators.")
                and all(getattr(getattr(cls, method), "__module__", "").startswith("ocs.operators.")
                        for method in ("matvec", "rmatvec"))):
            state = _operator_state_key(vars(value), visiting)
            if state is not _UNSAFE_STATE:
                return ("operator", _IdentityKey(value), _IdentityKey(cls.matvec),
                        _IdentityKey(cls.rmatvec), state)
        return _UNSAFE_STATE
    finally:
        visiting.remove(id(value))


def _cached_smfe_jit(operators, settings):
    """Cache a program only for explicit, inspectable built-in operator state.

    Arbitrary matvec closures remain uncached: they may capture mutable data.
    This caches executables, not numerical equilibration results. The bounded
    cache retains identities so Python cannot recycle a live cache key.
    """
    state = _operator_state_key(operators) if operators is not None else _UNSAFE_STATE
    context = tuple(str(getattr(jax.config, name, None)) for name in (
        "jax_enable_x64", "jax_default_matmul_precision", "jax_default_device",
        "jax_default_prng_impl", "jax_threefry_partitionable",
        "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion"))
    key = (state, settings, context)
    try:
        hash(key)
    except TypeError:
        state = _UNSAFE_STATE

    def decorate(fn):
        with _SMFE_CACHE_LOCK:
            # A retained operator can have been mutated since its program was
            # cached. Drop obsolete entries even when this call will bypass;
            # otherwise an old small key could retain newly attached big data.
            for old_key in list(_SMFE_RUN_CACHE):
                old_state = old_key[0]
                old_ops = tuple(part[1].value for part in old_state[1])
                if _operator_state_key(old_ops) != old_state:
                    del _SMFE_RUN_CACHE[old_key]
            while _SMFE_RUN_CACHE and _retained_array_bytes(_SMFE_RUN_CACHE) > _SMFE_CACHE_ARRAY_BYTES:
                _SMFE_RUN_CACHE.popitem(last=False)
            if state is _UNSAFE_STATE or _retained_array_bytes((key,)) > _SMFE_CACHE_ARRAY_BYTES:
                _SMFE_CACHE_STATS["bypasses"] += 1
                return operand_jit(fn)
            if key in _SMFE_RUN_CACHE:
                _SMFE_CACHE_STATS["hits"] += 1
                _SMFE_RUN_CACHE.move_to_end(key)
                return _SMFE_RUN_CACHE[key]
            compiled = operand_jit(fn)
            _SMFE_CACHE_STATS["builds"] += 1
            _SMFE_RUN_CACHE[key] = compiled
            while (len(_SMFE_RUN_CACHE) > _SMFE_CACHE_LIMIT or
                   _retained_array_bytes(_SMFE_RUN_CACHE) > _SMFE_CACHE_ARRAY_BYTES):
                _SMFE_RUN_CACHE.popitem(last=False)
            return compiled
    return decorate


def _scalar_span_layout(m, spans):
    """Compact disjoint scalar spans; retain legacy handling of other layouts."""
    if not spans or not any(b - a > 1 for a, b in spans):
        return None, m
    layout, row, packed = [], 0, 0
    for a, b in spans:
        if not row <= a < b <= m:
            # Ordered spans are produced by compile(). Other callers may use
            # overlapping or unusual slices whose sequential coercion matters.
            return None, m
        if a > row:
            layout.append((row, a, packed, False))
            packed += a - row
        layout.append((a, b, packed, True))
        packed += 1
        row = b
    if row < m:
        layout.append((row, m, packed, False))
        packed += m - row
    return tuple(layout), packed


def smfe_equilibration(matvec, rmatvec, n, m, scalar_spans=(), *,
                       iters=80, probes=2, gamma=1.0, M=6.0, safety=0.9, seed=0,
                       diagnostics=False, schedule="sgd", tie_columns=False,
                       _cache_operators=None, _cache_tag=None):
    """probes : Rademacher probes averaged per step (variance reduction).  Note:
    antithetic pairs s,-s give identical |DAEs|^2 (the estimator is even in s), so
    they do NOT reduce variance here -- multi-probe averaging is the right lever.

    diagnostics : if True, also return a dict with Bradley's normalized-variance
    equilibration diagnostic nvar(x) = d * var(x) / ||x||^2 (0 = perfectly
    equilibrated) of the row and column squared-norms of the operator BEFORE
    (identity scaling) and AFTER equilibration.  All estimated matrix-free with
    the same Rademacher probes, so the operator-only contract is preserved.  The
    global rescale is a uniform factor and leaves nvar invariant, so the AFTER
    numbers use the averaged log-scalings directly.

    tie_columns : if True, constrain the column log-scalings to one common
    value (a scalar primal metric), solving the construction problem with that
    constraint in place rather than reducing a free solution afterwards. Same
    oracle budget: 2*iters*probes gradient products plus 30 for the norm.

    _cache_tag : an optional hashable value carried into the program cache key
    and nowhere else. The key already carries every restriction this program
    itself applies (``scalar_spans``, the row sharing structure, and
    ``tie_columns``, the column restriction); a caller that resolves a richer
    DECLARATION down to those passes the declaration itself here, so a cached
    executable can never be returned for a family whose declared metric
    restrictions or block/sharing structure differ from the one it was built
    under. It changes no arithmetic.
    """
    tie_columns = bool(tie_columns)
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)
    row_layout, row_state_size = _scalar_span_layout(m, spans)
    r_tgt = jnp.sqrt(jnp.float32(n) / jnp.float32(m))   # row squared-norm target
    s_tgt = jnp.sqrt(jnp.float32(m) / jnp.float32(n))   # col squared-norm target (m r = n s)

    def expand_rows(u):
        if row_layout is None:
            return u
        return jnp.concatenate([
            jnp.broadcast_to(u[p:p + 1], (b - a,)) if scalar
            else u[p:p + b - a]
            for a, b, p, scalar in row_layout])

    def project_rows(u, w, M):
        """Project onto the box intersected with the tied-coordinate subspace.

        Free coordinates keep the value the caller already clipped. For a tied
        block the Euclidean projection onto {a*1 : |a| <= M} is the mean of the
        UNCLIPPED tentative point w, clipped afterwards. Averaging the clipped
        values instead is a different map and is not that projection: for
        M = 1 and w = (2, -1) it returns 0 rather than 1/2.
        """
        if row_layout is None:
            for a, b in spans:
                u = u.at[a:b].set(jnp.clip(jnp.mean(w[a:b]), -M, M))
            return u
        return jnp.concatenate([
            jnp.clip(jnp.mean(w[a:b], keepdims=True), -M, M) if scalar
            else u[a:b]
            for a, b, _p, scalar in row_layout])

    def project_columns(w):
        """Projection onto {box} cap {constant}: clip the mean of the unclipped
        tentative column logs (the column analogue of ``project_rows``)."""
        return jnp.broadcast_to(jnp.clip(jnp.mean(w), -M, M), w.shape)

    def rade(k, d):
        return jnp.where(jax.random.bernoulli(k, 0.5, (d,)), 1.0, -1.0).astype(jnp.float32)

    def probe_sum(key, eu, ev, count):
        # Keep the probe bodies in a loop: unrolling can retain independent FFT
        # and row-sized temporaries for every probe, including diagnostics.
        # The RNG splits and accumulation order match the unrolled estimator.
        def one_probe(_q, carry):
            key, row_e, col_e = carry
            key, k1, k2 = jax.random.split(key, 3)
            row_e = row_e + (eu * matvec(ev * rade(k1, n))) ** 2
            col_e = col_e + (ev * rmatvec(eu * rade(k2, m))) ** 2
            return key, row_e, col_e

        return jax.lax.fori_loop(
            0, count, one_probe,
            (key, jnp.zeros(m, jnp.float32), jnp.zeros(n, jnp.float32)))

    def body(t, carry):
        key, u, v, ubar, vbar, wsum = carry
        u = expand_rows(u)
        eu, ev = jnp.exp(u), jnp.exp(v)
        key, row_e, col_e = probe_sum(key, eu, ev, probes)
        tf = t.astype(jnp.float32)
        gu = row_e / probes - r_tgt + gamma * u
        gv = col_e / probes - s_tgt + gamma * v
        eta = 2.0 / (gamma * (tf + 2.0))
        w_u = u - eta * gu                        # tentative point, unclipped
        u = jnp.clip(w_u, -M, M)
        if tie_columns:
            v = project_columns(v - eta * gv)
        else:
            v = jnp.clip(v - eta * gv, -M, M)
        u = project_rows(u, w_u, M)
        wt = tf + 1.0
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    def body_bradley(t, carry):
        # Bradley SNBIN (thesis Ch.2, Fig.2.1, eqs.2.25-2.27): one (or few) probes
        # and a damped log-space fixed-point step toward equal norms, with a
        # decaying running-average weight omega that halves at k=2,4,8,16 (floor
        # 1/16) -- move fast early, settle late.  The running estimate (u,v) itself
        # is the output; no separate iterate-average.
        key, u, v, ubar, vbar, wsum = carry
        u = expand_rows(u)
        eu, ev = jnp.exp(u), jnp.exp(v)
        key, row_e, col_e = probe_sum(key, eu, ev, probes)
        row_e = row_e / probes; col_e = col_e / probes
        k = t.astype(jnp.float32) + 1.0
        w = jnp.maximum(2.0 ** (-jnp.floor(jnp.log2(jnp.maximum(k, 1.0)))), 1.0 / 16.0)
        w_u = u + 0.5 * w * (jnp.log(r_tgt) - jnp.log(row_e + 1e-30))
        u = jnp.clip(w_u, -M, M)
        if tie_columns:
            v = project_columns(v + 0.5 * w * (jnp.log(s_tgt) - jnp.log(col_e + 1e-30)))
        else:
            v = jnp.clip(v + 0.5 * w * (jnp.log(s_tgt) - jnp.log(col_e + 1e-30)), -M, M)
        u = project_rows(u, w_u, M)
        wt = k
        return (key, u, v, ubar + wt * u, vbar + wt * v, wsum + wt)

    step_body = body_bradley if schedule == "bradley" else body

    def _rc_est(key, eu, ev, P):            # matrix-free row/col squared-norm estimate
        _key, row, col = probe_sum(key, eu, ev, P)
        return row / P, col / P

    def _nvar(x, d):                         # Bradley's normalized variance in [0,1)
        return float(d) * jnp.var(x) / (jnp.sum(x * x) + 1e-30)

    # The cache identity of this program. It carries the metric restrictions
    # and the block/sharing structure -- the tied row spans and the column tie
    # -- plus the caller's own declaration in `_cache_tag`, so no cached
    # executable can be reused under a different declaration.
    settings = (n, m, spans, iters, probes, gamma, M, safety, seed, diagnostics, schedule,
                tie_columns, _cache_tag)

    @_cached_smfe_jit(_cache_operators, settings)
    def run():                              # whole equilibration is ONE compiled program
        init = (jax.random.PRNGKey(seed),
                jnp.zeros(row_state_size, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.zeros(row_state_size, jnp.float32), jnp.zeros(n, jnp.float32),
                jnp.asarray(0.0, jnp.float32))
        _k, u, v, ubar, vbar, wsum = jax.lax.fori_loop(0, iters, step_body, init)
        # sgd: iterate-averaged log-scalings; bradley: the running estimate itself
        u_f = expand_rows(u if schedule == "bradley" else ubar / wsum)
        v_f = v if schedule == "bradley" else vbar / wsum
        tau = jnp.exp(2.0 * v_f)            # primal per-coordinate step
        sigma = jnp.exp(2.0 * u_f)          # dual per-coordinate step
        st, sg = jnp.sqrt(tau), jnp.sqrt(sigma)
        nrm = _power_norm(lambda x: sg * matvec(st * x),
                          lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        scale = safety / (nrm + 1e-30)
        if diagnostics:
            dk = jax.random.PRNGKey(seed + 99)
            r0, c0 = _rc_est(dk, jnp.ones(m, jnp.float32), jnp.ones(n, jnp.float32), 8)
            r1, c1 = _rc_est(jax.random.PRNGKey(seed + 131),
                             jnp.exp(u_f), jnp.exp(v_f), 8)
            # nu_row / nu_col are the imbalance factors of the block-scalar
            # bound (see the paper's block-sparsity theorem): the largest row
            # (column) 2-norm divided by the root-mean-square, which is 1 at
            # exact balance and degrades the bound linearly.
            _nu = lambda x: jnp.sqrt(jnp.max(x) / (jnp.mean(x) + 1e-30))
            diag = {"nvar_row_pre": _nvar(r0, m), "nvar_col_pre": _nvar(c0, n),
                    "nvar_row_post": _nvar(r1, m), "nvar_col_post": _nvar(c1, n),
                    "nu_row": _nu(r1), "nu_col": _nu(c1)}
            return tau * scale, sigma * scale, nrm, diag
        return tau * scale, sigma * scale, nrm

    if diagnostics:
        tau, sigma, nrm, diag = run()
        return tau, sigma, float(nrm), {k: float(x) for k, x in diag.items()}
    tau, sigma, nrm = run()
    return tau, sigma, float(nrm)
