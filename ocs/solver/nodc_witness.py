"""Equality-projected dual witness for nonnegative Fourier/TV imaging without DC.

Family:  minimize 1/2 ||F x - b||^2 + lam * sum_v ||(D x)_v||_2   s.t.  x >= 0,
with F = FFTMask whose sampling mask EXCLUDES the DC frequency and
D = Gradient3D (Neumann forward differences, three per voxel).

Dual witness (u, w): d(u, w) = -b^T u - 1/2 ||u||^2 is a lower bound on the
optimal value when ||w_v||_2 <= lam for every voxel and the coupling
r = F^T u + D^T w is nonnegative.  Because F 1 = 0 (no DC) and D 1 = 0,
1^T r = 0 in exact arithmetic, so r >= 0 forces r = 0: the coupling must hold
as an EQUALITY and no strictly positive interior anchor exists.  The general
witness routes of compile.py (ray search, violation correction, free-row
anchor) therefore cannot certify this family.

Representation: the equality-eliminated TV multiplier.  The route returns a
pair (u, w) in float64 and certifies the EXACT-ARITHMETIC witness (u, w~),

    w~ = w - D (D^T D)^+ r,      r = F^T u + D^T w,

which satisfies F^T u + D^T w~ = 0 exactly.  The correction exists because r
is mean-zero (F 1 = D 1 = 0; "0 not in the mask" is an exact integer fact
checked at compile time) and range(D^T) is the mean-zero space of a
connected grid.  Its size is bounded through the smallest nonzero singular
value of D^T: the Neumann Laplacian D^T D is the Kronecker sum of 1-D
Neumann Laplacians with smallest nonzero eigenvalue 4 sin^2(pi / 2N), so

    ||w~ - w||_2 <= ||r||_2 / sigma,     sigma = 2 sin(pi / (2 max(D, H, W))),
    max_v ||w~_v||_2 <= max_v ||w_v||_2 + ||r||_2 / sigma.

The balls hold for w~ whenever the right side is at most lam, and the bound
value d depends on u alone.  The residual norm, the peak and d are float64
reductions carrying declared roundoff envelopes (``eta`` for the FFT and
stencil actions, ``dmargin`` for the value).  They are numerical, not an
interval proof: the same epistemic status as every other OCS assessment.

Improvement (recover=True), from the solver's float32 iterate (x, u, w):
  1. ``admm_iters`` warm ADMM iterations on the witness problem
         minimize 1/2 ||u + b||^2  s.t.  F^T u + D^T w = 0,  ||w_v|| <= lam,
     whose equality multiplier is -x*: start at z = -x with scaled multiplier
     D x / rho.  Each (u, w) step is one CG solve with F^T F + D^T D / rho on
     the mean-zero space (u = -b - F z, w = v - D z / rho).
  2. One equality projection of (u, w_ball) through CG on F^T F + D^T D.
  3. A contraction into radius lam (1 - ball_margin) with a ray search.
  4. The certificate above.  An unresolved witness is REFUSED (valid=False);
     nothing is clipped and no localized bound is substituted.
CG uses a Fourier-diagonal preconditioner (the exact symbol of F^T F and the
periodic symbol of the Laplacian), which changes iteration counts only.
Operators are applied through their actions; nothing is formed.  This route
does not change the model, the iteration, the step sizes or the tolerance.
"""
from __future__ import annotations

import math
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .._operands import operand_jit

SOURCE = "equality_projected"
REPRESENTATION = "equality_eliminated_tv_multiplier"

DEFAULTS = {
    "enabled": True,
    "admm_iters": 10,        # warm ADMM iterations (each one CG solve)
    "rho": 10.0,             # ADMM penalty
    "cg_tol": 1e-12,         # relative CG residual
    "cg_maxit": 5000,        # CG iteration cap per solve
    "ball_margin": 1e-9,     # contraction target radius lam (1 - ball_margin)
    "precondition": True,
    # patch D: "tiled" runs run() through the memory-bounded engine in
    # ocs.solver.nodc_tiled (same semantics and report); "global" is the
    # single-program route below.
    "engine": "tiled",
    # p1 (B200 integration, 2026-10-23): default "legacy", the installed symbol.  The brief admits
    # the flat-index preconditioner only if its certificates are identical; patch D's own controls
    # show the improved certificates differ from the installed route by 1.3e-12 to 4.9e-12 relative
    # with "flat" and by about 4e-16 (identical CG counts) with "legacy".  "flat" stays selectable.
    "precond_symbol": "legacy",  # tiled engine only: "legacy" (default, the installed symbol) or "flat"
    "slab_bytes": 1 << 30,       # tiled engine: largest float64 (T, H, W, 3) slab
    "idx_chunk": 1 << 26,        # tiled engine: mask entries per gather/scatter chunk
}
_settings = dict(DEFAULTS)


def configure(**kwargs):
    """Set route parameters for subsequently COMPILED families; returns a copy."""
    for key, value in kwargs.items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown equality-projected route setting {key!r}")
        _settings[key] = type(DEFAULTS[key])(value)
    return dict(_settings)


def settings():
    return dict(_settings)


def _fft_envelope(n, eps):
    # Declared normwise roundoff envelope for one orthonormal FFT of length n
    # and its real-part/scatter companions: 8 log2(n) eps (a conservative
    # multiple of the classical (log2 n) eps FFT bound).
    return 8.0 * max(1.0, math.log2(n)) * eps


class EqualityProjectedRoute:
    """One compiled family's route.  ``eligible`` is decided from exact host
    facts (operator types, index set, grid shape); nothing numerical."""

    def __init__(self, fft_op, grad_op, lam, data_slice, tv_slice, n, m, params=None):
        self.params = dict(_settings if params is None else params)
        self.n, self.m = int(n), int(m)
        self.data_slice, self.tv_slice = tuple(data_slice), tuple(tv_slice)
        # The compiled float32 program evaluates the TV term with the REPRESENTED
        # float32 weight; the balls are certified against that value (never the
        # larger decimal), so the witness is valid for the problem actually solved.
        self.lam_declared = float(lam)
        self.lam = float(min(float(lam), float(np.float32(lam))))
        self.shape = (int(grad_op.D), int(grad_op.H), int(grad_op.W))
        self.idx = np.asarray(jax.device_get(fft_op.idx)).astype(np.int64)
        self.k = int(self.idx.size)
        self.eligible, self.reason = self._check(fft_op, grad_op)
        self.sigma = (2.0 * math.sin(math.pi / (2.0 * max(self.shape)))
                      if max(self.shape) > 1 else float("nan"))
        self._compiled = {}

    # ---- exact host eligibility -------------------------------------------
    def _check(self, fft_op, grad_op):
        D, H, W = self.shape
        if D * H * W != self.n or int(fft_op.n) != self.n:
            return False, "grid and transform sizes differ from the variable size"
        if grad_op.shape[0] != 3 * self.n:
            return False, "gradient output is not three components per voxel"
        if self.data_slice[1] - self.data_slice[0] != 2 * self.k or self.tv_slice[1] - self.tv_slice[0] != 3 * self.n:
            return False, "dual slices do not match the operators"
        if self.k == 0 or np.any(self.idx < 0) or np.any(self.idx >= self.n) or np.unique(self.idx).size != self.k:
            return False, "mask indices are not a unique subset of [0, n)"
        if 0 in set(self.idx.tolist()):
            return False, "mask contains the DC frequency; the free-row anchor route applies"
        if max(self.shape) < 2:
            return False, "degenerate grid"
        if not (math.isfinite(self.lam) and self.lam > 0):
            return False, "TV weight is not a positive finite number"
        return True, "mask excludes DC; F1 = D1 = 0, coupling must vanish exactly"

    def describe(self):
        return {"source": SOURCE, "representation": REPRESENTATION, "eligible": self.eligible,
                "reason": self.reason, "grid": self.shape, "mask_size": self.k,
                "tv_weight_declared": self.lam_declared, "tv_weight_represented_float32": self.lam,
                "sigma_min_gradient_adjoint": self.sigma, "assessment_dtype": "float64",
                "settings": dict(self.params)}

    # ---- float64 actions (traced inside jax.enable_x64) ---------------------
    def _functions(self):
        if "fns" in self._compiled:
            return self._compiled["fns"]
        n, k = self.n, self.k
        D, H, W = self.shape
        idx_j = jnp.asarray(self.idx, jnp.int32)
        c128 = jnp.complex128

        def F(x):
            c = jnp.fft.fft(x, norm="ortho")[idx_j]
            return jnp.concatenate([jnp.real(c), jnp.imag(c)])

        def FT(u):
            spec = jnp.zeros(n, c128).at[idx_j].add(u[:k] + 1j * u[k:])
            return jnp.real(jnp.fft.ifft(spec, norm="ortho"))

        def Dg(x):
            v = x.reshape(D, H, W)
            dx = jnp.pad(v[1:] - v[:-1], ((0, 1), (0, 0), (0, 0)))
            dy = jnp.pad(v[:, 1:] - v[:, :-1], ((0, 0), (0, 1), (0, 0)))
            dz = jnp.pad(v[:, :, 1:] - v[:, :, :-1], ((0, 0), (0, 0), (0, 1)))
            return jnp.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], 1).reshape(-1)

        def DT(g):
            g = g.reshape(-1, 3)
            dx = g[:, 0].reshape(D, H, W)[:-1]
            dy = g[:, 1].reshape(D, H, W)[:, :-1]
            dz = g[:, 2].reshape(D, H, W)[:, :, :-1]
            a = jnp.pad(dx, ((1, 0), (0, 0), (0, 0))) - jnp.pad(dx, ((0, 1), (0, 0), (0, 0)))
            a = a + jnp.pad(dy, ((0, 0), (1, 0), (0, 0))) - jnp.pad(dy, ((0, 0), (0, 1), (0, 0)))
            a = a + jnp.pad(dz, ((0, 0), (0, 0), (1, 0))) - jnp.pad(dz, ((0, 0), (0, 0), (0, 1)))
            return a.reshape(-1)

        # Fourier-diagonal preconditioner symbols (host, once).
        mask = np.zeros(n); mask[self.idx] = 1.0
        sym = 0.5 * (mask + mask[(-np.arange(n)) % n])
        kk = np.indices((D, H, W)).reshape(3, -1)
        lap = (4 * np.sin(np.pi * kk[0] / D) ** 2 + 4 * np.sin(np.pi * kk[1] / H) ** 2
               + 4 * np.sin(np.pi * kk[2] / W) ** 2)
        sym_j, lap_j = jnp.asarray(sym), jnp.asarray(lap)
        precond = bool(self.params["precondition"])
        tol, maxit = float(self.params["cg_tol"]), int(self.params["cg_maxit"])

        def make_cg(rho):
            symbol = sym_j + lap_j / rho
            inv = (1.0 / jnp.where(symbol > 0, symbol, 1.0)).at[0].set(0.0)

            def A(z):
                out = FT(F(z)) + DT(Dg(z)) / rho
                return out - out.mean()

            def Minv(r):
                if not precond:
                    return r
                out = jnp.real(jnp.fft.ifft(jnp.fft.fft(r) * inv))
                return out - out.mean()

            def cg(rhs, z0):
                rhs = rhs - rhs.mean()
                nb = jnp.linalg.norm(rhs)
                z = z0 - z0.mean()
                r = rhs - A(z); q = Minv(r); p = q; rq = r @ q

                def cond(s):
                    return (jnp.linalg.norm(s[1]) > tol * nb) & (s[5] < maxit)

                def body(s):
                    z, r, p, q, rq, it = s
                    Ap = A(p)
                    alpha = rq / (p @ Ap)
                    z = z + alpha * p
                    r = r - alpha * Ap; r = r - r.mean()
                    q = Minv(r); rqn = r @ q
                    p = q + (rqn / rq) * p
                    return z, r, p, q, rqn, it + 1

                z, r, p, q, rq, it = lax.while_loop(cond, body, (z, r, p, q, rq, jnp.int32(0)))
                return z - z.mean(), it, jnp.linalg.norm(r) / jnp.where(nb > 0, nb, 1.0)
            return cg

        def ball(w, radius):
            g = w.reshape(-1, 3)
            nrm = jnp.linalg.norm(g, axis=1)
            return (g * jnp.minimum(1.0, radius / jnp.maximum(nrm, 1e-300))[:, None]).reshape(-1)

        eps = float(np.finfo(np.float64).eps)
        lam, sigma = self.lam, self.sigma
        fft_env = _fft_envelope(n, eps)

        def certificate(b, u, w):
            r = FT(u) + DT(w)
            r2 = jnp.linalg.norm(r)
            eta = fft_env * jnp.linalg.norm(u) + 8.0 * eps * jnp.linalg.norm(w)
            delta = (r2 + eta) / sigma
            peak = jnp.linalg.norm(w.reshape(-1, 3), axis=1).max()
            d = -(b @ u) - 0.5 * (u @ u)
            dmargin = 4.0 * eps * (jnp.abs(b) @ jnp.abs(u) + u @ u) + 2.0 * eps * jnp.abs(d)
            valid = (peak * (1.0 + 8.0 * eps) + delta <= lam) & jnp.isfinite(d) & jnp.all(jnp.isfinite(u)) & jnp.all(jnp.isfinite(w))
            return {"d": d, "d_lower": d - dmargin, "valid": valid, "residual_2": r2, "residual_inf": jnp.abs(r).max(),
                    "eta": eta, "delta": delta, "peak": peak, "ball_slack": lam - peak - delta}

        rho = float(self.params["rho"])
        admm_iters = int(self.params["admm_iters"])
        margin = float(self.params["ball_margin"])
        cg_rho, cg_one = make_cg(rho), make_cg(1.0)

        @operand_jit
        def improve(x, u0, w0, b):
            FTb = FT(b)
            wp = ball(w0, lam); lamb = Dg(x) / rho; z = -x
            cg_total = jnp.int32(0)

            def step(_, s):
                wp, lamb, z, cg_total = s
                v = wp - lamb
                z, it, _res = cg_rho(DT(v) - FTb, z)
                u = -b - F(z); w = v - Dg(z) / rho
                wp = ball(w + lamb, lam); lamb = lamb + w - wp
                return wp, lamb, z, cg_total + it

            wp, lamb, z, cg_total = lax.fori_loop(0, admm_iters, step, (wp, lamb, z, cg_total))
            u = jnp.where(admm_iters > 0, -b - F(z), u0)
            wb = wp if admm_iters > 0 else ball(w0, lam)
            # one equality projection, then contraction with a ray search
            z1, it1, res1 = cg_one(FT(u) + DT(wb), jnp.zeros_like(x))
            up, wq = u - F(z1), wb - Dg(z1)
            peak = jnp.linalg.norm(wq.reshape(-1, 3), axis=1).max()
            smax = jnp.minimum(1.0, lam * (1.0 - margin) / jnp.where(peak > 0, peak, 1.0))
            uu = up @ up
            s = jnp.clip(-(b @ up) / jnp.where(uu > 0, uu, 1.0), 0.0, smax)
            s = jnp.where(uu > 0, s, 0.0)
            us, ws = s * up, s * wq
            cert = certificate(b, us, ws)
            cert.update({"scale": s, "cg_iterations_admm": cg_total, "cg_iterations_projection": it1,
                         "cg_relative_residual_projection": res1})
            return us, ws, cert

        @operand_jit
        def certify(b, u, w):
            return certificate(b, u, w)

        @operand_jit
        def localized(b, u, w, upper):
            # ROUTE B (separate named guarantee, never strict dual feasibility):
            # inf_{x>=0} P = inf_z H(z) with H shift-invariant, so a mean-zero
            # optimizer z* exists with lam * sigma * ||z*|| <= lam ||D z*||_{2,1}
            # <= p* <= U; hence ||z*||_2 <= U / (lam sigma).  For (u, w_ball)
            # Fenchel gives H(z) >= d0(u) + z^T r, so at z*:
            #     p* >= d0(u) - (U / (lam sigma)) ||Pi_{1 perp} r||_2.
            wb = ball(w, lam)
            r = FT(u) + DT(wb)
            r = r - r.mean()
            eta = fft_env * jnp.linalg.norm(u) + 8.0 * eps * jnp.linalg.norm(wb)
            radius = upper / (lam * sigma)
            d0 = -(b @ u) - 0.5 * (u @ u)
            dmargin = 4.0 * eps * (jnp.abs(b) @ jnp.abs(u) + u @ u) + 2.0 * eps * jnp.abs(d0)
            lower = d0 - dmargin - radius * (jnp.linalg.norm(r) + eta)
            return {"localized_lower_bound": lower, "d0": d0, "radius": radius,
                    "projected_residual_2": jnp.linalg.norm(r), "eta": eta,
                    "valid": jnp.isfinite(lower) & (upper >= 0)}

        fns = {"improve": improve, "certify": certify, "localized": localized}
        self._compiled["fns"] = fns
        return fns

    # ---- public ---------------------------------------------------------------
    def _split(self, y64):
        a, b = self.data_slice
        c, d = self.tv_slice
        return y64[a:b], y64[c:d]

    def _assemble(self, u, w):
        y = np.zeros(self.m, np.float64)
        a, b = self.data_slice; c, d = self.tv_slice
        y[a:b] = np.asarray(u, np.float64); y[c:d] = np.asarray(w, np.float64)
        return y

    def _data(self, l, c):
        l64 = np.asarray(jax.device_get(l), np.float64)
        c64 = np.asarray(jax.device_get(c), np.float64)
        a, b = self.data_slice; e, f = self.tv_slice
        if np.any(c64 != 0.0):
            return None, None, "a linear cost term is present; the zero-sum coupling argument needs c = 0"
        if np.any(l64[e:f] != 0.0):
            return None, None, "the TV block carries an offset; not the declared imaging family"
        return -l64[a:b], l64, None

    def run(self, x, y, l, c, *, recover, upper=None):
        """Returns a report dict; ``valid`` True only when the certificate holds.

        ``recover=True`` improves the supplied pair (ADMM, projection,
        contraction) and certifies the result; ``recover=False`` certifies the
        SUPPLIED pair under the same semantics with no change to it.  When a
        finite feasible upper bound ``upper`` (P at the returned x >= 0) is
        given, the separate ROUTE B localized lower bound is reported under
        ``localized`` for the supplied pair and for the improved pair; it is
        a named guarantee of its own and never awards strict dual feasibility."""
        if self.params.get("engine", "global") == "tiled":       # patch D
            from . import nodc_tiled
            return nodc_tiled.run(self, x, y, l, c, recover=recover, upper=upper)
        t0 = time.perf_counter()
        report = {"source": SOURCE, "representation": REPRESENTATION, "valid": False,
                  "improved": bool(recover), "settings": dict(self.params)}
        if not self.eligible:
            report["refused"] = self.reason
            return report
        b, _l64, why = self._data(l, c)
        if why is not None:
            report["refused"] = why
            return report
        y64 = np.asarray(jax.device_get(y), np.float64).reshape(-1)
        if y64.size != self.m:
            report["refused"] = "dual size mismatch"
            return report
        u0, w0 = self._split(y64)
        with jax.enable_x64():
            fns = self._functions()
            b_j = jnp.asarray(b, jnp.float64)
            if recover:
                x64 = jnp.asarray(np.asarray(jax.device_get(x), np.float64).reshape(-1))
                us, ws, cert = fns["improve"](x64, jnp.asarray(u0), jnp.asarray(w0), b_j)
                us, ws = np.asarray(jax.device_get(us), np.float64), np.asarray(jax.device_get(ws), np.float64)
            else:
                cert = fns["certify"](b_j, jnp.asarray(u0), jnp.asarray(w0))
                us, ws = u0, w0
            cert = {k: (bool(v) if v.dtype == jnp.bool_ else float(v)) for k, v in jax.device_get(cert).items()}
            if upper is not None and math.isfinite(float(upper)):
                xmin = float(np.min(np.asarray(jax.device_get(x), np.float64)))
                loc = {"bound_kind": "localized", "dual_feasible": False, "upper_bound": float(upper),
                       "upper_bound_feasible": bool(xmin >= 0.0), "sigma_min_gradient": self.sigma}
                if xmin >= 0.0:
                    for name, (uu, ww) in (("supplied", (u0, w0)), ("improved", (us, ws))):
                        rb = fns["localized"](b_j, jnp.asarray(uu), jnp.asarray(ww), jnp.asarray(float(upper), jnp.float64))
                        loc[name] = {k: (bool(v) if v.dtype == jnp.bool_ else float(v)) for k, v in jax.device_get(rb).items()}
                else:
                    loc["refused"] = "the returned point is not nonnegative, so P(x) is not an upper bound"
                report["localized"] = loc
        report.update(cert)
        report["valid"] = bool(cert["valid"]) and math.isfinite(cert["d_lower"])
        report["dual_float64"] = self._assemble(us, ws)
        admm = int(self.params["admm_iters"]) if recover else 0
        cg_total = int(cert.get("cg_iterations_admm", 0)) + int(cert.get("cg_iterations_projection", 0))
        # each CG iteration applies F, F^T, D, D^T once (plus the preconditioner FFT pair)
        report["operator_applications"] = 4 * cg_total + 4 * admm + 8 + (4 if recover else 2)
        report["cg_iterations_total"] = cg_total
        report["seconds"] = time.perf_counter() - t0
        if not report["valid"]:
            report["refused"] = ("the certificate does not hold: peak + delta exceeds lam, or the value is not finite"
                                 if self.eligible else self.reason)
        return report
