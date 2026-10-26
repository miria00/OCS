"""Problem families for the block-metric pilot (examples/experimental/blockmetric_pilot.py).

Every family is  minimize_x  g(x) + sum_blocks f_i(K_i x + l_i)  with coupled
Euclidean-norm output blocks of size k, and is returned twice: once as an OCS
ProblemFamily (compiled only for the shipped engine arm and for the public
certificate), and once as a "Spec" that the research PDHG loop consumes
directly (operators, offsets, block layout, exact per-block Grams when the
family allows them).  No family forms a global matrix; the controlled family
applies its 4x4 mixing blocks as products of a scalar, a selection and a
rank-one term, exactly as the pilot brief prescribes.

Families
  F1  controlled correlation      blockmetric_f1(n, rho, k, variant, seed)
  F2  multichannel deblurring     blockmetric_f2(L, mixing, seed)
  F3  3-D TV imaging control      blockmetric_f3(L, seed)   (the killer_3d_imaging model)
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import ocs
from ocs.ir import Term
from ocs.operators.base import LinearOperator
from ocs.prox.library import SumSquares


class Spec:
    """What the research loop needs.

    n, mu            : variable size; ridge weight (0 means no ridge)
    primal           : "ridge" or "nonneg"
    dual_blocks      : list of dicts with keys
        op    : LinearOperator (m_i x n)
        kind  : "group_l2" (lam * sum of block norms, block size k) | "sum_squares"
        k     : block size (1 for sum_squares)
        lam   : scalar weight (group_l2)
        offset: (m_i,) array l_i  (Kx + l enters f)
        exact_gram : callable(t0_vec, s0) -> (B, k, k) exact Grams of
                     s0^{1/2} K_i T0^{1/2} per coupled block, or None
    data             : dict fed to the OCS compiled family
    exact_norm       : callable(sig_blocks, tau) -> exact ||Sigma^{1/2} K T^{1/2}||_2, or None
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ----------------------------------------------------------------------------- F1
class BlockMixSelect(LinearOperator):
    """K x = stack_i  w_i * C_i x_{S_i},   C_i = c0 [(1-rho) I + rho a_i a_i^T],
    a_i = s_i / sqrt(k) with s_i in {-1,+1}^k, so rows and columns of every
    C_i have equal norms, and c0 normalizes them to one.  eig(C_i) =
    c0 (1-rho) (multiplicity k-1) and c0 (along a_i), so kappa(C_i) = 1/(1-rho).
    w_i is an optional per-row magnitude (the imbalance variant).  Applied
    without forming any matrix: a gather, a scalar, and a rank-one term.
    """
    def __init__(self, perm, signs, rho, row_scale=None):
        B, k = signs.shape
        n = perm.shape[0]
        super().__init__((B * k, n))
        self.perm = jnp.asarray(perm, jnp.int32)
        self.inv_perm = jnp.asarray(np.argsort(perm), jnp.int32)
        self.a = jnp.asarray(signs, jnp.float32) / np.sqrt(k)
        self.rho = float(rho)
        # row norm^2 of (1-rho)I + rho a a^T with |a_j|^2 = 1/k:
        rn2 = (1 - rho) ** 2 + 2 * (1 - rho) * rho / k + rho ** 2 / k
        self.c0 = float(1.0 / np.sqrt(rn2))
        self.B, self.k = B, k
        self.w = None if row_scale is None else jnp.asarray(row_scale, jnp.float32)

    def _mix(self, z):                      # z (B, k) -> C z per block (C symmetric)
        proj = jnp.sum(self.a * z, axis=1, keepdims=True)
        return self.c0 * ((1.0 - self.rho) * z + self.rho * self.a * proj)

    def matvec(self, x):
        z = x[self.perm].reshape(self.B, self.k)
        out = self._mix(z)
        if self.w is not None:
            out = out * self.w
        return out.reshape(-1)

    def rmatvec(self, y):
        z = y.reshape(self.B, self.k)
        if self.w is not None:
            z = z * self.w
        z = self._mix(z).reshape(-1)
        return z[self.inv_perm]

    def dense_blocks(self):                 # (B, k, k) explicit C_i (diagnostics only)
        a = np.asarray(self.a, np.float64)
        k = self.k
        C = self.c0 * ((1 - self.rho) * np.eye(k)[None] + self.rho * a[:, :, None] * a[:, None, :])
        if self.w is not None:
            C = np.asarray(self.w, np.float64)[:, :, None] * C
        return C


class StackedBlockMix(LinearOperator):
    """K = [K_1; K_2], two BlockMixSelect operators on two INDEPENDENT random
    partitions of the coordinates: every coordinate appears in exactly two mixed
    blocks, so the blocks are coupled through shared coordinates and the problem
    does not decouple (the partition variants do)."""
    def __init__(self, K1, K2):
        super().__init__((K1.shape[0] + K2.shape[0], K1.shape[1]))
        self.K1, self.K2 = K1, K2
        self.B, self.k = K1.B + K2.B, K1.k

    def matvec(self, x):
        return jnp.concatenate([self.K1.matvec(x), self.K2.matvec(x)])

    def rmatvec(self, y):
        m1 = self.K1.shape[0]
        return self.K1.rmatvec(y[:m1]) + self.K2.rmatvec(y[m1:])

    def dense_blocks(self):
        return np.concatenate([self.K1.dense_blocks(), self.K2.dense_blocks()])


def blockmetric_f1(n=200_000, rho=0.99, k=4, variant="equal", seed=0, mu=1e-2, lam=1.0):
    """F1: controlled within-block correlation.
    variant: "equal"     one random partition, rows/columns of equal norm: K is permuted
                         block-diagonal and the ridge is separable, so the problem splits
                         into n/k independent k-dimensional problems -> SEPARABLE SANITY CHECK
             "imbalance" as "equal" with rows scaled by log-uniform factors in [1, 100]
                         (also separable)
             "overlap"   two independent random partitions stacked (m = 2n): every
                         coordinate is in two mixed blocks, blocks are coupled -> the
                         variant that decides whether off-diagonal information helps
    """
    rng = np.random.default_rng(seed)
    assert n % k == 0
    B = n // k
    perm = rng.permutation(n)
    signs = rng.choice([-1.0, 1.0], size=(B, k))
    row_scale = None
    if variant == "imbalance":
        row_scale = np.exp(rng.uniform(0.0, np.log(100.0), size=(B, k)))
    K = BlockMixSelect(perm, signs, rho, row_scale)
    if variant == "overlap":
        perm2 = rng.permutation(n)
        signs2 = rng.choice([-1.0, 1.0], size=(B, k))
        K = StackedBlockMix(K, BlockMixSelect(perm2, signs2, rho))
        perm = np.concatenate([perm, perm2])
        B = 2 * B
    m = K.shape[0]
    x_true = rng.normal(size=n)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(x_true, jnp.float32))) + 0.3 * rng.normal(size=m)
    b = b.astype(np.float32)

    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(mu)),
                         ocs.group_l2(K @ x - bp, lam=lam, group_size=k)])
    data = {"b": jnp.asarray(b)}

    C = K.dense_blocks()                    # (B,k,k) float64, diagnostics only
    perm_np = np.asarray(perm)

    def exact_gram(t0, s0):
        t = np.asarray(t0, np.float64)
        t = np.broadcast_to(t, (n,))[perm_np].reshape(B, k)
        return s0 * np.einsum("bij,bj,blj->bil", C, t, C)

    def block_norms(sig, tau):
        """largest singular value of every preconditioned block Sigma_i^{1/2} C_i T_{S_i}^{1/2}"""
        t = np.broadcast_to(np.asarray(tau, np.float64), (n,))[perm_np].reshape(B, k)
        Bm = C * np.sqrt(t)[:, None, :]
        if np.ndim(sig) == 0:
            Bm = np.sqrt(float(sig)) * Bm
        elif np.ndim(sig) == 2:
            Bm = np.sqrt(np.asarray(sig, np.float64))[:, :, None] * Bm
        else:
            w, Q = np.linalg.eigh(np.asarray(sig, np.float64))
            half = np.einsum("bij,bj,bkj->bik", Q, np.sqrt(w), Q)
            Bm = np.einsum("bij,bjk->bik", half, Bm)
        return np.linalg.svd(Bm, compute_uv=False)[:, 0]

    def exact_norm(sig, tau):
        """Partition variants: the operator is permuted block-diagonal, so the norm is
        exactly the largest block norm.  Overlap variant: no exact value (returns None)."""
        if variant == "overlap":
            return None
        return float(np.max(block_norms(sig, tau)))

    def norm_upper(sig, tau):
        """Overlap variant: ||[B_1; B_2]||_2 <= sqrt(||B_1||^2 + ||B_2||^2), each a block-diagonal norm."""
        s = block_norms(sig, tau)
        if variant != "overlap":
            return float(np.max(s))
        half_ = len(s) // 2
        return float(np.sqrt(np.max(s[:half_]) ** 2 + np.max(s[half_:]) ** 2))

    tag_mu = "" if mu == 1e-2 else f"-mu{mu:g}"
    spec = Spec(n=n, mu=mu, primal="ridge", data=data, name=f"F1-{variant}-n{n}-k{k}-rho{rho}{tag_mu}",
                norm_upper=norm_upper, note=("coupled (two stacked partitions)" if variant == "overlap"
                                             else "separable sanity check (one partition)"),
                dual_blocks=[dict(op=K, kind="group_l2", k=k, lam=lam,
                                  offset=jnp.asarray(-b), exact_gram=exact_gram)],
                exact_norm=exact_norm, family="F1", rho=rho, k=k, variant=variant)
    return prob, spec


# ----------------------------------------------------------------------------- F2
class ChannelBlur(LinearOperator):
    """c-channel circular Gaussian blur followed by a fixed c x c channel mix.
    x layout: (c, L, L) channel-major;  output layout: (L*L, c) pixel-major, so
    each pixel's c channels form one contiguous coupled block.  c = 3 is the
    family used to develop the predictor; c = 5, 6 are held-out block sizes."""
    def __init__(self, L, M, sigma_blur=0.7):
        N = L * L
        c = int(np.asarray(M).shape[0])
        super().__init__((c * N, c * N))
        self.L, self.N, self.c = L, N, c
        self.M = jnp.asarray(M, jnp.float32)
        yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        yy = np.minimum(yy, L - yy); xx = np.minimum(xx, L - xx)
        h = np.exp(-0.5 * (yy ** 2 + xx ** 2) / sigma_blur ** 2)
        h = h / h.sum()
        self.h = h                                        # spatial kernel (wrapped)
        self.H = jnp.asarray(np.fft.fft2(h))              # real & even -> real spectrum
        self.H2 = jnp.asarray(np.fft.fft2(h ** 2))        # spectrum of the squared kernel

    def blur(self, img):                                  # (c, L, L) -> (c, L, L), self-adjoint
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(img) * self.H[None])).astype(jnp.float32)

    def matvec(self, x):
        img = self.blur(x.reshape(self.c, self.L, self.L)).reshape(self.c, self.N)
        out = jnp.matmul(self.M, img, precision=jax.lax.Precision.HIGHEST)
        return out.T.reshape(-1)                                            # pixel-major

    def rmatvec(self, y):
        z = jnp.matmul(self.M.T, y.reshape(self.N, self.c).T, precision=jax.lax.Precision.HIGHEST)
        return self.blur(z.reshape(self.c, self.L, self.L)).reshape(-1)

    def row_energy(self, t):
        """e_c(p) = sum_j h_{pj}^2 t_{c,j}  exactly, via the squared kernel (c, N)."""
        timg = jnp.broadcast_to(jnp.asarray(t, jnp.float32), (self.c * self.N,)).reshape(self.c, self.L, self.L)
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(timg) * self.H2[None]))
        return np.asarray(e, np.float64).reshape(self.c, self.N)


def _mixing_matrix(kind, seed=0, c=3):
    """near / strong were used to develop the predictor.  The remaining kinds are
    HELD OUT: they are new mixing structures, drawn with a different seed offset,
    used only to score a prediction made in advance."""
    rng = np.random.default_rng(seed + 17)

    def _svals(target_kappa):
        """c singular values, log-spaced, spanning the requested condition number."""
        return np.logspace(0.0, -np.log10(target_kappa), c)

    def _rand(target_kappa, offset):
        r = np.random.default_rng(seed + offset)
        U, _ = np.linalg.qr(r.normal(size=(c, c)))
        V, _ = np.linalg.qr(r.normal(size=(c, c)))
        return U @ np.diag(_svals(target_kappa)) @ V.T

    if kind == "near":
        A = 0.08 * rng.normal(size=(c, c))
        M = np.eye(c) + 0.5 * (A + A.T)
    elif kind == "strong":
        if c == 3:
            U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            M = U @ np.diag([1.0, 0.25, 0.02]) @ V.T             # kappa = 50 (development family)
        else:
            M = _rand(50.0, 17)
    elif kind == "moderate":                                      # held out: kappa = 8
        M = _rand(8.0, 991)
    elif kind == "extreme":                                       # held out: kappa = 500
        M = _rand(500.0, 4242)
    elif kind == "circulant":                                     # held out: different STRUCTURE
        first = np.concatenate([[1.0], np.linspace(0.55, 0.30, c - 1)])
        M = np.stack([np.roll(first, i) for i in range(c)])
    else:
        raise ValueError(kind)
    M = M / np.sqrt(np.mean(np.sum(M ** 2, axis=1)))           # rms row norm 1
    return M


def _test_image(L, rng, nchan=3):
    yy, xx = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L), indexing="ij")
    chans = []
    for c in range(nchan):
        img = np.zeros((L, L))
        for _ in range(6):
            cy, cx, r = rng.uniform(-0.8, 0.8, 3) * [1, 1, 0.5]
            img += rng.uniform(0.3, 1.0) * (((yy - cy) ** 2 + (xx - cx) ** 2) < (0.15 + abs(r)) ** 2)
        img += 0.2 * np.sin(3 * (c + 1) * xx) * np.cos(2 * yy)
        chans.append(img)
    return np.stack(chans)


def blockmetric_f2(L=256, mixing="near", seed=0, mu=1e-2, lam=1.0, sigma_blur=0.7, channels=3):
    """F2: c-channel deblurring with correlated channels; coupled blocks of size c per
    pixel.  c = 3 is the development family; c = 5, 6 are held-out block sizes."""
    rng = np.random.default_rng(seed)
    c = int(channels)
    M = _mixing_matrix(mixing, seed, c)
    K = ChannelBlur(L, M, sigma_blur)
    n = c * L * L
    x_true = _test_image(L, rng, c).reshape(-1)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(x_true, jnp.float32))) + 0.05 * rng.normal(size=n)
    b = b.astype(np.float32)
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(mu)),
                         ocs.group_l2(K @ x - bp, lam=lam, group_size=c)])
    data = {"b": jnp.asarray(b)}
    N = L * L

    def exact_gram(t0, s0):
        e = K.row_energy(t0)                                  # (c, N)
        Mn = np.asarray(M, np.float64)
        return s0 * np.einsum("ic,cp,jc->pij", Mn, e, Mn)      # (N, c, c)

    def exact_norm(sig, tau):
        # exact only when tau is uniform and sig is one common 3x3 block; else None
        tau = np.asarray(tau, np.float64)
        if tau.ndim and np.ptp(tau) > 1e-6 * np.max(tau):
            return None
        t = float(np.mean(tau))
        Mn = np.asarray(M, np.float64)
        Hmax = float(np.max(np.abs(np.asarray(K.H))))
        if np.ndim(sig) == 0:
            return float(np.sqrt(float(sig) * t) * np.linalg.norm(Mn, 2) * Hmax)
        S = np.asarray(sig, np.float64)
        if S.ndim == 3 and np.allclose(S, S[:1], rtol=1e-5, atol=1e-7 * np.abs(S).max()):
            w, Q = np.linalg.eigh(S[0]); half = Q @ np.diag(np.sqrt(w)) @ Q.T
            return float(np.sqrt(t) * np.linalg.norm(half @ Mn, 2) * Hmax)
        return None

    tag_mu = "" if mu == 1e-2 else f"-mu{mu:g}"
    tag_c = "" if c == 3 else f"-c{c}"
    spec = Spec(n=n, mu=mu, primal="ridge", data=data, name=f"F2-{mixing}-L{L}{tag_c}{tag_mu}",
                dual_blocks=[dict(op=K, kind="group_l2", k=c, lam=lam,
                                  offset=jnp.asarray(-b), exact_gram=exact_gram)],
                exact_norm=exact_norm, family="F2", rho=float(np.linalg.cond(M)), k=c,
                variant=mixing, mixing_cond=float(np.linalg.cond(M)),
                note=f"{c}-channel" + ("" if c == 3 else " (held-out block size)"))
    return prob, spec


# ----------------------------------------------------------------------------- F3
def blockmetric_f3(L=64, lam=0.02, ratio=0.4, seed=0):
    """F3: the killer_3d_imaging / gated_weight_runner imaging model at L^3:
    nonneg(x) + 1/2||F x - y||^2 + lam * sum_voxel ||(grad3 x)_voxel||_2."""
    n = L ** 3
    vol = np.zeros((L, L, L), np.float32)
    vol[8:28, 10:34, 12:30] = 1.0
    vol[20:36, 6:20, 18:38] += 0.5
    phantom = vol.reshape(-1)
    F = ocs.FFTMask.random(n, ratio=ratio, seed=seed)
    G3 = ocs.Gradient3D(L, L, L)
    y = np.asarray(F.matvec(jnp.asarray(phantom)))
    x = ocs.Variable(n); yp = ocs.Parameter("y")
    prob = ocs.minimize([ocs.nonneg(x), ocs.least_squares(F @ x - yp),
                         ocs.group_l2(G3 @ x, lam=lam, group_size=3)])
    data = {"y": jnp.asarray(y)}

    def exact_gram(t0, s0):
        t = np.broadcast_to(np.asarray(t0, np.float64), (n,)).reshape(L, L, L)
        G = np.zeros((L, L, L, 3, 3))
        shifted = [np.concatenate([t[1:], np.zeros((1, L, L))], 0),
                   np.concatenate([t[:, 1:], np.zeros((L, 1, L))], 1),
                   np.concatenate([t[:, :, 1:], np.zeros((L, L, 1))], 2)]
        interior = [np.arange(L)[:, None, None] < L - 1,
                    np.arange(L)[None, :, None] < L - 1,
                    np.arange(L)[None, None, :] < L - 1]
        for a in range(3):
            G[..., a, a] = np.where(interior[a], t + shifted[a], 0.0)
            for bb in range(3):
                if a != bb:
                    G[..., a, bb] = np.where(interior[a] & interior[bb], t, 0.0)
        return s0 * G.reshape(n, 3, 3)

    spec = Spec(n=n, mu=0.0, primal="nonneg", data=data, name=f"F3-tv3d-L{L}",
                dual_blocks=[dict(op=F, kind="sum_squares", k=1, lam=1.0,
                                  offset=jnp.asarray(-y), exact_gram=None),
                             dict(op=G3, kind="group_l2", k=3, lam=lam,
                                  offset=jnp.zeros(3 * n, jnp.float32), exact_gram=exact_gram)],
                exact_norm=None, family="F3", rho=0.0, k=3, variant="tv3d", phantom=phantom)
    return prob, spec
