"""Memory-bounded float64 engine for the equality-projected (no-DC) witness.

Patch D, 2026-10-23.  Same witness semantics, same arithmetic sequence and
same report contract as ``EqualityProjectedRoute.run`` in
``ocs.solver.nodc_witness``: ``admm_iters`` warm ADMM iterations (one CG
solve each with F^T F + D^T D / rho on the mean-zero space), one joint
equality projection (CG with F^T F + D^T D), a contraction with a ray
search, and the certificate of the implicit witness
w~ = w - D (D^T D)^+ r with ball margin (||r||_2 + eta) / sigma.  Only the
residency of the arrays and the way the operators are applied change:

* The CG state (z, r, p and one transient) is the only n-sized float64 state
  on the device.  Every w-sized (3n) array (the ADMM pair wp, lamb, the ball
  projection, the returned multiplier) lives on the host as D-slabs and is
  streamed through the device slab by slab with a one-plane halo.
* The 2k-sized arrays (b, u) live on the host and are streamed in chunks for
  the masked gather and scatter of the spectrum.
* F^T F p = irfft(sym . rfft(p)) with the symmetrized half-spectrum mask
  (real-transform packing of the installed complex path), and the Neumann
  Laplacian D^T D is one stencil without the 3n gradient temporary.
* Large arrays (the half-spectrum mask, the index chunks) are runtime
  operands of the compiled programs, never captured constants, and they are
  released at the end of every call: between checks the engine holds host
  arrays only.

The preconditioner is the Fourier-diagonal one of the installed route.
``precond_symbol="legacy"`` is the installed symbol (the 3-D periodic
Laplacian indexed by unravel(f)), symmetrized exactly as the installed
complex path's real part symmetrizes it.  ``precond_symbol="flat"`` is the
exact symbol of the flat-periodic Laplacian in the DFT basis,
4 sin^2(pi f/n) + 4 sin^2(pi f W/n) + 4 sin^2(pi f H W/n).  The
preconditioner changes CG iteration counts only; every certified number is
a direct evaluation on the returned pair.

The module also carries the two host-residency helpers of the
phase-separated drive loop (``to_host``, ``to_device``).
"""
from __future__ import annotations

import math
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

EPS64 = float(np.finfo(np.float64).eps)
ENGINE = "tiled"


def _fft_envelope(n, eps=EPS64):
    # Same declared envelope as nodc_witness._fft_envelope.
    return 8.0 * max(1.0, math.log2(n)) * eps


def choose_slab(D, H, W, target_bytes):
    """Largest divisor T of D whose float64 (T, H, W, 3) slab fits target_bytes."""
    best = 1
    for T in range(1, D + 1):
        if D % T == 0 and T * H * W * 3 * 8 <= target_bytes:
            best = T
    return best


# ---- host residency helpers (phase-separated drive) --------------------------
def to_host(tree):
    """Device arrays -> host numpy arrays with the same bytes; other leaves kept."""
    return jax.tree_util.tree_map(
        lambda a: np.asarray(jax.device_get(a)) if isinstance(a, jax.Array) else a, tree)


def to_device(tree):
    """Host numpy arrays -> device arrays with the same bytes; other leaves kept."""
    return jax.tree_util.tree_map(
        lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, tree)


class TiledEngine:
    """One no-DC family's memory-bounded float64 assessment.

    Construction is host-only (index metadata).  Every public method must be
    called inside ``jax.enable_x64()``.
    """

    def __init__(self, shape, idx, lam, sigma, params):
        self.shape = tuple(int(s) for s in shape)
        self.D, self.H, self.W = self.shape
        self.n = self.D * self.H * self.W
        self.idx = np.asarray(idx, np.int64)
        self.k = int(self.idx.size)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.rho = float(params["rho"])
        self.cg_tol = float(params["cg_tol"])
        self.cg_maxit = int(params["cg_maxit"])
        self.margin = float(params["ball_margin"])
        self.precondition = bool(params["precondition"])
        self.symbol = str(params.get("precond_symbol", "legacy"))
        if self.symbol not in ("legacy", "flat"):
            raise ValueError("precond_symbol must be 'legacy' or 'flat'")
        self.T = choose_slab(self.D, self.H, self.W, int(params.get("slab_bytes", 1 << 30)))
        self.nslabs = self.D // self.T
        self.idx_chunk = max(1, int(params.get("idx_chunk", 1 << 26)))
        n = self.n
        self.nh = n // 2 + 1
        # 2 sym(f) = mask[f] + mask[n - f] on the half spectrum (uint8, host).
        mask = np.zeros(n, np.uint8)
        mask[self.idx] = 1
        mirror = np.empty(self.nh, np.uint8)
        mirror[0] = mask[0]
        if self.nh > 1:
            mirror[1:] = mask[n - 1:n - self.nh:-1]
        self.sym2_host = mask[:self.nh] + mirror
        del mask, mirror
        half = np.minimum(self.idx, n - self.idx)
        self.half_idx = half
        self.flip = self.idx > n // 2                  # conjugate for mirrored entries
        self.endpoint = (half == 0) | ((n % 2 == 0) & (half == n // 2))
        self.stats = {}
        self._build()

    # ------------------------------------------------------------------ kernels
    def _build(self):
        D, H, W, n, T, nh = self.D, self.H, self.W, self.n, self.T, self.nh
        rho, lam = self.rho, self.lam
        tol, maxit = self.cg_tol, self.cg_maxit
        precondition, symbol = self.precondition, self.symbol

        def spec_fwd(v):
            return jnp.fft.rfft(v, norm="ortho")

        def spec_inv(s):
            return jnp.fft.irfft(s, n=n, norm="ortho")

        def FTF(p, sym2):
            return spec_inv(spec_fwd(p) * (0.5 * sym2.astype(jnp.float64)))

        def lap(p):
            v = p.reshape(D, H, W)
            dx = v[1:] - v[:-1]; dy = v[:, 1:] - v[:, :-1]; dz = v[:, :, 1:] - v[:, :, :-1]
            a = jnp.pad(dx, ((1, 0), (0, 0), (0, 0))) - jnp.pad(dx, ((0, 1), (0, 0), (0, 0)))
            a = a + jnp.pad(dy, ((0, 0), (1, 0), (0, 0))) - jnp.pad(dy, ((0, 0), (0, 1), (0, 0)))
            a = a + jnp.pad(dz, ((0, 0), (0, 0), (1, 0))) - jnp.pad(dz, ((0, 0), (0, 0), (0, 1)))
            return a.reshape(-1)

        def lap_symbol(f):
            if symbol == "flat":
                fd = f.astype(jnp.float64)
                return (4.0 * jnp.sin(jnp.pi * fd / n) ** 2 + 4.0 * jnp.sin(jnp.pi * fd * W / n) ** 2
                        + 4.0 * jnp.sin(jnp.pi * fd * (H * W) / n) ** 2)
            k0 = f // (H * W); rem = f - k0 * (H * W); k1 = rem // W; k2 = rem - k1 * W
            return (4.0 * jnp.sin(jnp.pi * k0.astype(jnp.float64) / D) ** 2
                    + 4.0 * jnp.sin(jnp.pi * k1.astype(jnp.float64) / H) ** 2
                    + 4.0 * jnp.sin(jnp.pi * k2.astype(jnp.float64) / W) ** 2)

        def make_cg(rho_):
            def A(z, sym2):
                out = FTF(z, sym2) + lap(z) / rho_
                return out - out.mean()

            def inv_at(ff, sym2):
                # sym is symmetric under f <-> n - f, so sym2 at the half index serves both.
                value = 0.5 * sym2.astype(jnp.float64) + lap_symbol(ff) / rho_
                inv = jnp.where(value > 0, 1.0 / jnp.where(value > 0, value, 1.0), 1.0)
                return jnp.where(ff == 0, 0.0, inv)

            def Minv(r, sym2):
                if not precondition:
                    return r
                f = jnp.arange(nh, dtype=jnp.int64)
                # The installed complex path multiplies by the symbol and takes the
                # real part, which applies the symmetrized symbol; use exactly that.
                inv = 0.5 * (inv_at(f, sym2) + inv_at((n - f) % n, sym2))
                out = spec_inv(spec_fwd(r) * inv)
                return out - out.mean()

            def cg(rhs, z0, sym2):
                rhs = rhs - rhs.mean()
                nb = jnp.linalg.norm(rhs)
                z = z0 - z0.mean()
                r = rhs - A(z, sym2); q = Minv(r, sym2); p = q; rq = r @ q

                def cond(s):
                    return (jnp.linalg.norm(s[1]) > tol * nb) & (s[4] < maxit)

                def body(s):
                    z, r, p, rq, it = s
                    Ap = A(p, sym2)
                    alpha = rq / (p @ Ap)
                    z = z + alpha * p
                    r = r - alpha * Ap; r = r - r.mean()
                    q = Minv(r, sym2); rqn = r @ q
                    p = q + (rqn / rq) * p
                    return z, r, p, rqn, it + 1

                z, r, p, rq, it = lax.while_loop(cond, body, (z, r, p, rq, jnp.int32(0)))
                return z - z.mean(), it, jnp.linalg.norm(r) / jnp.where(nb > 0, nb, 1.0)
            return cg

        self.cg_rho = jax.jit(make_cg(rho), donate_argnums=(0, 1))
        self.cg_one = jax.jit(make_cg(1.0), donate_argnums=(0, 1))

        # ---- device buffer helpers (in place through donation) --------------
        slab_n = T * H * W
        self.put = jax.jit(lambda buf, start, val: lax.dynamic_update_slice(buf, val, (start,)),
                           donate_argnums=(0,))
        self.add = jax.jit(lambda buf, start, val: lax.dynamic_update_slice(
            buf, lax.dynamic_slice(buf, (start,), val.shape) + val, (start,)), donate_argnums=(0,))
        self.neg = jax.jit(lambda v: -v)
        self.zeros_n = jax.jit(lambda: jnp.zeros(n, jnp.float64))
        self.zeros_spec = jax.jit(lambda: jnp.zeros(nh, jnp.complex128))
        self.center = jax.jit(lambda v: v - v.mean())
        self.norm = jax.jit(jnp.linalg.norm)
        self.absmax = jax.jit(lambda v: jnp.abs(v).max())

        @jax.jit
        def get_halo(buf, start):
            return lax.dynamic_slice(buf, (start,), (slab_n + H * W,)).reshape(T + 1, H, W)

        @jax.jit
        def get_halo_last(buf, start):
            s = lax.dynamic_slice(buf, (start,), (slab_n,)).reshape(T, H, W)
            return jnp.concatenate([s, s[-1:]], 0)
        self.get_halo, self.get_halo_last = get_halo, get_halo_last

        # ---- slab stencils --------------------------------------------------
        def dg_slab(zs):
            """Forward differences on T planes given T + 1 (halo replicated on the last slab)."""
            dx = zs[1:] - zs[:-1]
            dy = jnp.pad(zs[:T, 1:] - zs[:T, :-1], ((0, 0), (0, 1), (0, 0)))
            dz = jnp.pad(zs[:T, :, 1:] - zs[:T, :, :-1], ((0, 0), (0, 0), (0, 1)))
            return jnp.stack([dx, dy, dz], -1)

        def dt_slab(v, vx_prev, last):
            """D^T on a (T, H, W, 3) slab; vx_prev is the x component on the plane before it."""
            vx = v[..., 0]
            if last:
                vx = vx.at[-1].set(0.0)                 # D^T ignores the far-boundary component
            vx_in = jnp.concatenate([vx_prev[None], vx[:-1]], 0)
            a = vx_in - vx
            dy = v[:, :-1, :, 1]; dz = v[:, :, :-1, 2]
            a = a + jnp.pad(dy, ((0, 0), (1, 0), (0, 0))) - jnp.pad(dy, ((0, 0), (0, 1), (0, 0)))
            a = a + jnp.pad(dz, ((0, 0), (0, 0), (1, 0))) - jnp.pad(dz, ((0, 0), (0, 0), (0, 1)))
            return a

        def ball(w, radius):
            nrm = jnp.linalg.norm(w, axis=-1)
            return w * jnp.minimum(1.0, radius / jnp.maximum(nrm, 1e-300))[..., None]

        def init_kernel(xs, w0s, ftb_s, vx_prev, last):
            """wp = ball(w0), lamb = D x / rho, and the first rhs slab D^T(wp - lamb) - F^T b."""
            wp = ball(w0s, lam); lamb = dg_slab(xs) / rho
            v = wp - lamb
            rhs = dt_slab(v, vx_prev, last) - ftb_s
            return wp, lamb, rhs, v[-1, :, :, 0], -xs[:T]

        def sweep_kernel(zs, wp, lamb, ftb_s, vx_prev, last):
            """One ADMM (w, wp, lamb) update on a slab, plus the next rhs slab."""
            v = wp - lamb
            w = v - dg_slab(zs) / rho
            wp2 = ball(w + lamb, lam); lamb2 = lamb + w - wp2
            v2 = wp2 - lamb2
            rhs = dt_slab(v2, vx_prev, last) - ftb_s
            return wp2, lamb2, rhs, v2[-1, :, :, 0]

        def dt_kernel(ws, vx_prev, last):
            return dt_slab(ws, vx_prev, last), ws[-1, :, :, 0]

        def wq_kernel(z1s, wbs):
            """Statistics of wq = wb - D z1 on a slab (peak, sum of squares, finite)."""
            wq = wbs - dg_slab(z1s)
            nrm = jnp.linalg.norm(wq, axis=-1)
            return nrm.max(), (wq * wq).sum(), jnp.all(jnp.isfinite(wq))

        def ws_kernel(z1s, wbs, s, vx_prev, last):
            """ws = s (wb - D z1) on a slab, its D^T and its statistics."""
            ws = s * (wbs - dg_slab(z1s))
            nrm = jnp.linalg.norm(ws, axis=-1)
            return (ws, dt_slab(ws, vx_prev, last), ws[-1, :, :, 0],
                    (nrm.max(), (ws * ws).sum(), jnp.all(jnp.isfinite(ws))))

        def cert_w_kernel(ws, vx_prev, last):
            nrm = jnp.linalg.norm(ws, axis=-1)
            return (dt_slab(ws, vx_prev, last), ws[-1, :, :, 0],
                    (nrm.max(), (ws * ws).sum(), jnp.all(jnp.isfinite(ws))))

        def ball_dt_kernel(ws, vx_prev, last):
            """Route B: wb = ball(w) on a slab, its D^T and its sum of squares."""
            wb = ball(ws, lam)
            return dt_slab(wb, vx_prev, last), wb[-1, :, :, 0], (wb * wb).sum()

        st = dict(static_argnames=("last",))
        self.init_kernel = jax.jit(init_kernel, **st)
        self.sweep_kernel = jax.jit(sweep_kernel, **st)
        self.dt_kernel = jax.jit(dt_kernel, **st)
        self.wq_kernel = jax.jit(wq_kernel)
        self.ws_kernel = jax.jit(ws_kernel, **st)
        self.cert_w_kernel = jax.jit(cert_w_kernel, **st)
        self.ball_dt_kernel = jax.jit(ball_dt_kernel, **st)
        self.ball_slab = jax.jit(lambda w: ball(w, lam))

        # ---- spectrum gather / scatter on the half spectrum --------------------
        @jax.jit
        def gather(spec, hidx, flip):
            c = spec[hidx]
            return jnp.real(c), jnp.where(flip, -jnp.imag(c), jnp.imag(c))

        def scatter(spec, hidx, flip, endpoint, re, im):
            im = jnp.where(flip, -im, im)
            wgt = jnp.where(endpoint, 1.0, 0.5)
            return spec.at[hidx].add((re + 1j * im) * wgt)
        self.gather = gather
        self.scatter = jax.jit(scatter, donate_argnums=(0,))
        self.spec_fwd = jax.jit(spec_fwd)
        self.spec_inv = jax.jit(spec_inv)

    # --------------------------------------------------------------- helpers
    def _start(self, s):
        return jnp.asarray(s * self.T * self.H * self.W, jnp.int64)

    def _chunks(self):
        for s in range(0, self.k, self.idx_chunk):
            yield s, min(s + self.idx_chunk, self.k)

    def _h2d(self, nbytes):
        self.stats["h2d_bytes"] = self.stats.get("h2d_bytes", 0) + int(nbytes)

    def _d2h(self, nbytes):
        self.stats["d2h_bytes"] = self.stats.get("d2h_bytes", 0) + int(nbytes)

    def _tick(self, name, t0):
        ph = self.stats.setdefault("phase_s", {})
        ph[name] = ph.get(name, 0.0) + time.perf_counter() - t0

    def _plane_zero(self):
        return jnp.zeros((self.H, self.W), jnp.float64)

    def _slabs(self, arr):
        """Views of a host (n,) or (3n,) array as D-slabs."""
        a = np.asarray(arr)
        if a.size == self.n:
            a = a.reshape(self.D, self.H, self.W)
        else:
            a = a.reshape(self.D, self.H, self.W, 3)
        return [a[s * self.T:(s + 1) * self.T] for s in range(self.nslabs)]

    def _halo(self, buf, s):
        last = s == self.nslabs - 1
        return (self.get_halo_last if last else self.get_halo)(buf, self._start(s))

    def FT_host(self, u):
        """F^T u for a host 2k-vector u, as a device n-vector."""
        spec = self.zeros_spec()
        k = self.k
        for s, e in self._chunks():
            re = jnp.asarray(np.asarray(u[s:e], np.float64))
            im = jnp.asarray(np.asarray(u[k + s:k + e], np.float64))
            spec = self.scatter(spec, jnp.asarray(self.half_idx[s:e]), jnp.asarray(self.flip[s:e]),
                                jnp.asarray(self.endpoint[s:e]), re, im)
            self._h2d(16 * (e - s) + 10 * (e - s))
        return self.spec_inv(spec)

    def F_host(self, z):
        """F z for a device n-vector z, as a host float64 2k-vector."""
        spec = self.spec_fwd(z)
        k = self.k
        out = np.empty(2 * k, np.float64)
        for s, e in self._chunks():
            re, im = self.gather(spec, jnp.asarray(self.half_idx[s:e]), jnp.asarray(self.flip[s:e]))
            out[s:e] = np.asarray(re); out[k + s:k + e] = np.asarray(im)
            self._h2d(9 * (e - s)); self._d2h(16 * (e - s))
        del spec
        return out

    # ------------------------------------------------------------- the route
    def improve(self, x, u0, w0, b, admm_iters, ws_out):
        """Improve the supplied pair and certify the result.

        x: host (n,), u0: host (2k,), w0: host (3n,), b: host (2k,) float64.
        ws_out: host float64 (3n,) buffer that receives the returned multiplier
        ws = s (wb - D z1).  Returns (us, cert) with cert keyed as the installed
        route's ``improve``.
        """
        self.stats = {}
        T, H, W, n, k = self.T, self.H, self.W, self.n, self.k
        slab_n = T * H * W
        b = np.asarray(b, np.float64)
        sym2 = jnp.asarray(self.sym2_host)
        self._h2d(self.sym2_host.nbytes)
        cg_per_solve = []
        try:
            # F^T b, held on the host as slabs
            t0 = time.perf_counter()
            ftb_dev = self.FT_host(b)
            ftb_slabs = [np.asarray(lax.dynamic_slice(ftb_dev, (self._start(s),), (slab_n,))).reshape(T, H, W)
                         for s in range(self.nslabs)]
            self._d2h(8 * n)
            del ftb_dev
            self._tick("ftb", t0)

            # init sweep: wp, lamb host slabs; z = -x and the first rhs on the device
            t0 = time.perf_counter()
            x_slabs = self._slabs(np.asarray(x, np.float64))
            w0_slabs = self._slabs(np.asarray(w0, np.float64))
            z = self.zeros_n(); rhs = self.zeros_n()
            wp_slabs, lamb_slabs = [None] * self.nslabs, [None] * self.nslabs
            vx_prev = self._plane_zero()
            for s in range(self.nslabs):
                last = s == self.nslabs - 1
                xs = np.asarray(x_slabs[s], np.float64)
                nxt = xs[-1:] if last else np.asarray(x_slabs[s + 1][:1], np.float64)
                xs_j = jnp.asarray(np.concatenate([xs, nxt], 0))
                w0_j = jnp.asarray(np.asarray(w0_slabs[s], np.float64))
                self._h2d(8 * (slab_n + H * W) + 24 * slab_n + 8 * slab_n)
                wp, lamb, rhs_s, vx_prev, negx = self.init_kernel(
                    xs_j, w0_j, jnp.asarray(ftb_slabs[s]), vx_prev, last=last)
                z = self.put(z, self._start(s), negx.reshape(-1))
                rhs = self.put(rhs, self._start(s), rhs_s.reshape(-1))
                wp_slabs[s] = np.asarray(wp); lamb_slabs[s] = np.asarray(lamb)
                self._d2h(48 * slab_n)
            del x_slabs, w0_slabs
            self._tick("init_sweep", t0)

            # ADMM: one CG solve per iteration, then one slab sweep
            cg_admm = 0
            for _ in range(int(admm_iters)):
                t0 = time.perf_counter()
                z, it, _res = self.cg_rho(rhs, z, sym2)
                it = int(it); cg_admm += it; cg_per_solve.append(it)
                self._tick("admm_cg", t0)
                t0 = time.perf_counter()
                rhs = self.zeros_n()
                vx_prev = self._plane_zero()
                for s in range(self.nslabs):
                    last = s == self.nslabs - 1
                    wp_j = jnp.asarray(wp_slabs[s]); lamb_j = jnp.asarray(lamb_slabs[s])
                    self._h2d(56 * slab_n)
                    wp, lamb, rhs_s, vx_prev = self.sweep_kernel(
                        self._halo(z, s), wp_j, lamb_j, jnp.asarray(ftb_slabs[s]), vx_prev, last=last)
                    rhs = self.put(rhs, self._start(s), rhs_s.reshape(-1))
                    wp_slabs[s] = np.asarray(wp); lamb_slabs[s] = np.asarray(lamb)
                    self._d2h(48 * slab_n)
                self._tick("admm_sweep", t0)
            del rhs, ftb_slabs, lamb_slabs

            # u = -b - F z, wb = wp (ball-projected by construction)
            t0 = time.perf_counter()
            if admm_iters > 0:
                u = -b - self.F_host(z)
                wb_slabs = wp_slabs
            else:
                u = np.asarray(u0, np.float64)
                wb_slabs = [np.asarray(self.ball_slab(jnp.asarray(ws_)))
                            for ws_ in self._slabs(np.asarray(w0, np.float64))]
            del z
            self._tick("u", t0)

            # projection: z1 = (F^T F + D^T D)^-1 (F^T u + D^T wb)
            t0 = time.perf_counter()
            rhs1 = self.FT_host(u)
            vx_prev = self._plane_zero()
            for s in range(self.nslabs):
                last = s == self.nslabs - 1
                wb_j = jnp.asarray(wb_slabs[s]); self._h2d(24 * slab_n)
                dts, vx_prev = self.dt_kernel(wb_j, vx_prev, last=last)
                rhs1 = self.add(rhs1, self._start(s), dts.reshape(-1))
            self._tick("proj_rhs", t0)
            t0 = time.perf_counter()
            z1, it1, res1 = self.cg_one(rhs1, self.zeros_n(), sym2)
            it1 = int(it1); res1 = float(res1)
            self._tick("proj_cg", t0)
            del sym2

            # contraction: peak of wq = wb - D z1, then the ray search on the host
            t0 = time.perf_counter()
            peak = 0.0
            for s in range(self.nslabs):
                pk, _sq, _fin = self.wq_kernel(self._halo(z1, s), jnp.asarray(wb_slabs[s]))
                self._h2d(24 * slab_n)
                peak = max(peak, float(pk))
            up = u - self.F_host(z1)
            del u
            uu = float(up @ up)
            smax = min(1.0, self.lam * (1.0 - self.margin) / (peak if peak > 0 else 1.0))
            s_ = min(max(-float(b @ up) / (uu if uu > 0 else 1.0), 0.0), smax)
            s_ = s_ if uu > 0 else 0.0
            self._tick("contract", t0)

            # certificate on (us, ws) = s (up, wq), by direct evaluation
            t0 = time.perf_counter()
            us = s_ * up
            del up
            r = self.FT_host(us)
            ws_view = np.asarray(ws_out).reshape(self.D, H, W, 3)
            vx_prev = self._plane_zero(); peak_s = 0.0; sumsq_s = 0.0; finite_s = True
            for s in range(self.nslabs):
                last = s == self.nslabs - 1
                wb_j = jnp.asarray(wb_slabs[s]); self._h2d(24 * slab_n)
                ws_j, dts, vx_prev, (pk, sq, fin) = self.ws_kernel(
                    self._halo(z1, s), wb_j, jnp.float64(s_), vx_prev, last=last)
                r = self.add(r, self._start(s), dts.reshape(-1))
                ws_view[s * T:(s + 1) * T] = np.asarray(ws_j); self._d2h(24 * slab_n)
                peak_s = max(peak_s, float(pk)); sumsq_s += float(sq); finite_s = finite_s and bool(fin)
            del z1, wb_slabs, wp_slabs
            cert = self._certificate(b, us, r, peak_s, sumsq_s, finite_s)
            del r
            self._tick("certificate", t0)
        finally:
            sym2 = None
        cert.update(scale=float(s_), cg_iterations_admm=float(cg_admm),
                    cg_iterations_projection=float(it1), cg_relative_residual_projection=res1)
        self.stats["cg_per_solve"] = cg_per_solve + [it1]
        return us, cert

    def _certificate(self, b, u, r_dev, peak, sumsq_w, finite_w):
        r2 = float(self.norm(r_dev)); rinf = float(self.absmax(r_dev))
        eta = _fft_envelope(self.n) * float(np.linalg.norm(u)) + 8.0 * EPS64 * math.sqrt(sumsq_w)
        delta = (r2 + eta) / self.sigma
        d = -float(b @ u) - 0.5 * float(u @ u)
        dmargin = 4.0 * EPS64 * (float(np.abs(b) @ np.abs(u)) + float(u @ u)) + 2.0 * EPS64 * abs(d)
        valid = (bool(peak * (1.0 + 8.0 * EPS64) + delta <= self.lam) and math.isfinite(d)
                 and bool(np.all(np.isfinite(u))) and bool(finite_w))
        return {"d": d, "d_lower": d - dmargin, "valid": valid, "residual_2": r2, "residual_inf": rinf,
                "eta": eta, "delta": delta, "peak": peak, "ball_slack": self.lam - peak - delta}

    def certify(self, u, w, b):
        """Certificate of a SUPPLIED host pair (u, w); nothing is moved."""
        self.stats = {}
        slab_n = self.T * self.H * self.W
        u = np.asarray(u, np.float64); b = np.asarray(b, np.float64)
        r = self.FT_host(u)
        vx_prev = self._plane_zero(); peak = 0.0; sumsq = 0.0; fin_all = True
        for s, ws_ in enumerate(self._slabs(np.asarray(w, np.float64))):
            last = s == self.nslabs - 1
            dts, vx_prev, (pk, sq, fin) = self.cert_w_kernel(jnp.asarray(ws_), vx_prev, last=last)
            self._h2d(24 * slab_n)
            r = self.add(r, self._start(s), dts.reshape(-1))
            peak = max(peak, float(pk)); sumsq += float(sq); fin_all = fin_all and bool(fin)
        return self._certificate(b, u, r, peak, sumsq, fin_all)

    def localized(self, b, u, w, upper):
        """ROUTE B of the installed route (a separate named guarantee), slab-wise."""
        slab_n = self.T * self.H * self.W
        u = np.asarray(u, np.float64); b = np.asarray(b, np.float64)
        r = self.FT_host(u)
        vx_prev = self._plane_zero(); sumsq = 0.0
        for s, ws_ in enumerate(self._slabs(np.asarray(w, np.float64))):
            last = s == self.nslabs - 1
            dts, vx_prev, sq = self.ball_dt_kernel(jnp.asarray(ws_), vx_prev, last=last)
            self._h2d(24 * slab_n)
            r = self.add(r, self._start(s), dts.reshape(-1))
            sumsq += float(sq)
        r = self.center(r)
        r2 = float(self.norm(r)); del r
        eta = _fft_envelope(self.n) * float(np.linalg.norm(u)) + 8.0 * EPS64 * math.sqrt(sumsq)
        radius = float(upper) / (self.lam * self.sigma)
        d0 = -float(b @ u) - 0.5 * float(u @ u)
        dmargin = 4.0 * EPS64 * (float(np.abs(b) @ np.abs(u)) + float(u @ u)) + 2.0 * EPS64 * abs(d0)
        lower = d0 - dmargin - radius * (r2 + eta)
        return {"localized_lower_bound": lower, "d0": d0, "radius": radius,
                "projected_residual_2": r2, "eta": eta,
                "valid": bool(math.isfinite(lower) and float(upper) >= 0)}

    def describe(self):
        return {"engine": ENGINE, "slab_planes": self.T, "slabs": self.nslabs,
                "idx_chunk": self.idx_chunk, "precond_symbol": self.symbol,
                "real_transform_packing": True,
                "device_state": "CG state and transients only; w-sized arrays as host slabs"}


def run(route, x, y, l, c, *, recover, upper=None):
    """``EqualityProjectedRoute.run`` through the tiled engine (same contract)."""
    from .nodc_witness import SOURCE, REPRESENTATION
    t0 = time.perf_counter()
    report = {"source": SOURCE, "representation": REPRESENTATION, "valid": False,
              "improved": bool(recover), "settings": dict(route.params)}
    if not route.eligible:
        report["refused"] = route.reason
        return report
    b, _l64, why = route._data(l, c)
    del _l64
    if why is not None:
        report["refused"] = why
        return report
    y64 = np.asarray(jax.device_get(y), np.float64).reshape(-1)
    if y64.size != route.m:
        report["refused"] = "dual size mismatch"
        return report
    u0, w0 = route._split(y64)
    da, db_ = route.data_slice
    ta, tb = route.tv_slice
    with jax.enable_x64():
        eng = route._compiled.get("tiled")
        if eng is None:
            eng = TiledEngine(route.shape, route.idx, route.lam, route.sigma, route.params)
            route._compiled["tiled"] = eng
        if recover:
            x64 = np.asarray(jax.device_get(x), np.float64).reshape(-1)
            dual = np.zeros(route.m, np.float64)
            us, cert = eng.improve(x64, u0, w0, b, int(route.params["admm_iters"]), dual[ta:tb])
            dual[da:db_] = us
            ws = dual[ta:tb]
            del x64
        else:
            cert = eng.certify(u0, w0, b)
            us, ws = u0, w0
            dual = route._assemble(u0, w0)
        engine_stats = dict(eng.stats)
        if upper is not None and math.isfinite(float(upper)):
            xmin = float(np.min(np.asarray(jax.device_get(x), np.float64)))
            loc = {"bound_kind": "localized", "dual_feasible": False, "upper_bound": float(upper),
                   "upper_bound_feasible": bool(xmin >= 0.0), "sigma_min_gradient": route.sigma}
            if xmin >= 0.0:
                for name, (uu, ww) in (("supplied", (u0, w0)), ("improved", (us, ws))):
                    loc[name] = eng.localized(b, uu, ww, float(upper))
            else:
                loc["refused"] = "the returned point is not nonnegative, so P(x) is not an upper bound"
            report["localized"] = loc
    report.update(cert)
    report["valid"] = bool(cert["valid"]) and math.isfinite(cert["d_lower"])
    report["dual_float64"] = dual
    admm = int(route.params["admm_iters"]) if recover else 0
    cg_total = int(cert.get("cg_iterations_admm", 0)) + int(cert.get("cg_iterations_projection", 0))
    # same logical count as the installed route
    report["operator_applications"] = 4 * cg_total + 4 * admm + 8 + (4 if recover else 2)
    report["cg_iterations_total"] = cg_total
    report["engine"] = dict(eng.describe(), **engine_stats)
    report["seconds"] = time.perf_counter() - t0
    if not report["valid"]:
        report["refused"] = ("the certificate does not hold: peak + delta exceeds lam, or the value is not finite"
                             if route.eligible else route.reason)
    return report
