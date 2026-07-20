"""Fast structured transforms as matrix-free operators (Boyd's wishlist):
DCT, orthonormal Haar wavelet, fast Gauss transform (Gaussian blur), and a
Laplacian (Poisson) solve.  Each is applied through matvec/rmatvec only; the
dense transform is never formed.  FFTs use complex64 (float32 substrate).

  DCTOperator      : orthonormal DCT-II          (adjoint = inverse DCT)
  HaarWavelet      : orthonormal multi-level Haar (adjoint = inverse DWT)
  GaussianBlur     : Gaussian convolution via FFT (self-adjoint)
  InverseLaplacian : (alpha I - L)^{-1} via FFT   (self-adjoint) -- the Laplace solve
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.fft as jsf

from .base import LinearOperator

_SQRT2I = 0.7071067811865476


class DCTOperator(LinearOperator):
    """Orthonormal DCT-II of a length-n real vector.  Orthogonal, so the adjoint is
    the inverse DCT (DCT-III).  Never forms the n x n cosine matrix."""
    def __init__(self, n):
        super().__init__((n, n))

    def matvec(self, x):
        return jsf.dct(x, type=2, norm="ortho")

    def rmatvec(self, y):
        return jsf.idct(y, type=2, norm="ortho")


class HaarWavelet(LinearOperator):
    """Orthonormal multi-level Haar wavelet transform of a length-n vector
    (n = 2^L).  Output layout: [approx (coarsest) ; detail_L ; ... ; detail_1].
    Orthogonal, so rmatvec is the inverse transform.  Matrix-free (O(n))."""
    def __init__(self, n, levels=None):
        L = int(round(jnp.log2(n).item())) if levels is None else int(levels)
        assert (1 << L) <= n and n % (1 << L) == 0, "n must be a multiple of 2^levels"
        super().__init__((n, n))
        self.n, self.levels = n, L

    def matvec(self, x):
        a = x; details = []
        for _ in range(self.levels):
            e, o = a[0::2], a[1::2]
            a = (e + o) * _SQRT2I
            details.append((e - o) * _SQRT2I)
        return jnp.concatenate([a] + details[::-1])            # [approx, d_L, ..., d_1]

    def rmatvec(self, y):
        m = self.n >> self.levels                              # coarsest approx length
        a = y[:m]; off = m
        sizes = [self.n >> lvl for lvl in range(self.levels, 0, -1)]   # detail sizes, coarse->fine
        dets = []
        for s in sizes:
            dets.append(y[off:off + s]); off += s
        for d in dets:                                         # reconstruct coarse -> fine
            e = (a + d) * _SQRT2I; o = (a - d) * _SQRT2I
            a = jnp.stack([e, o], axis=1).reshape(-1)
        return a


class GaussianBlur(LinearOperator):
    """Fast Gauss transform: circular Gaussian convolution of a length-n signal via
    FFT.  The kernel is real and even, so the operator is SELF-ADJOINT."""
    def __init__(self, n, sigma=2.0):
        super().__init__((n, n))
        k = jnp.fft.fftfreq(n).astype(jnp.float32)
        self.filt = jnp.exp(-0.5 * (sigma * 2.0 * jnp.pi * k) ** 2).astype(jnp.complex64)

    def matvec(self, x):
        return jnp.real(jnp.fft.ifft(jnp.fft.fft(x.astype(jnp.complex64)) * self.filt)).astype(jnp.float32)

    def rmatvec(self, y):
        return self.matvec(y)                                  # self-adjoint


class InverseLaplacian(LinearOperator):
    """Laplacian (Poisson) solve on a periodic 1-D grid: apply (alpha I - L)^{-1},
    L the discrete Laplacian, diagonalized by the FFT with eigenvalues
    2 - 2 cos(2 pi k / n).  Real symmetric spectrum -> SELF-ADJOINT.  This is the
    matrix-free 'Laplacian solve' (a smoothing / Green's-function apply)."""
    def __init__(self, n, alpha=1.0):
        super().__init__((n, n))
        k = jnp.arange(n).astype(jnp.float32)
        self.inv_eig = (1.0 / (alpha + 2.0 - 2.0 * jnp.cos(2.0 * jnp.pi * k / n))).astype(jnp.complex64)

    def matvec(self, x):
        return jnp.real(jnp.fft.ifft(jnp.fft.fft(x.astype(jnp.complex64)) * self.inv_eig)).astype(jnp.float32)

    def rmatvec(self, y):
        return self.matvec(y)                                  # self-adjoint
