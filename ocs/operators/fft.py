"""Fast-transform operators.  The FFT has no sparse form, hence this is the
regime where a stored-matrix solver cannot even represent the problem, and OCS
applies K in O(n log n) time and O(n) memory.

FFTMask implements K = M F, a subsampled orthonormal FFT, as a REAL operator by
stacking the real and imaginary parts of the m kept frequencies: K : R^n -> R^2m.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .base import LinearOperator


class FFTMask(LinearOperator):
    """K x = [Re; Im] of (orthonormal FFT of x) restricted to `idx` frequencies.

    A subset of orthonormal-FFT rows: columns/rows already ~uniform norm, so
    equilibration is skipped (well_conditioned)."""
    well_conditioned = True

    def __init__(self, idx, n):
        idx = jnp.asarray(np.asarray(idx), jnp.int32)
        m = idx.shape[0]
        super().__init__((2 * m, n))
        self.idx = idx
        self.n = n
        self.m = m

    def matvec(self, x):
        c = jnp.fft.fft(x, norm="ortho")[self.idx]
        return jnp.concatenate([jnp.real(c), jnp.imag(c)])

    def rmatvec(self, u):
        cc = u[:self.m] + 1j * u[self.m:]
        z = jnp.zeros(self.n, jnp.complex64).at[self.idx].set(cc)
        return jnp.real(jnp.fft.ifft(z, norm="ortho"))

    @classmethod
    def random(cls, n, ratio=0.25, seed=0):
        m = max(1, int(n * ratio))
        idx = np.sort(np.random.default_rng(seed).choice(n, m, replace=False))
        return cls(idx, n)
