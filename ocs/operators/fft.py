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
    equilibration is skipped (well_conditioned).

    ``real_fft=None`` uses a real transform for lengths at least ``2**21``;
    smaller transforms retain the complex implementation. This performance
    heuristic is based on measured 128/256-cubed GPU workloads, not a guarantee
    for every transform length or device. ``True`` and ``False`` explicitly
    select either implementation. Complex inputs and indices outside the
    documented unique, nonnegative subset retain the complex implementation.
    Both implementations represent the same real operator and adjoint.
    """
    well_conditioned = True

    def __init__(self, idx, n, *, real_fft=None):
        if real_fft is not None and not isinstance(real_fft, (bool, np.bool_)):
            raise TypeError("real_fft must be None, True, or False")
        # The device indices are int32 (x64 off): refuse, loudly, a transform
        # length or an index that int32 cannot hold, instead of wrapping it.
        raw_idx = np.asarray(idx)
        if int(n) > np.iinfo(np.int32).max:
            raise ValueError(f"FFTMask: n = {int(n)} exceeds the int32 index range")
        if raw_idx.size and (int(raw_idx.max()) > np.iinfo(np.int32).max
                             or int(raw_idx.min()) < np.iinfo(np.int32).min):
            raise ValueError("FFTMask: an index exceeds the int32 range")
        host_idx = raw_idx.astype(np.int32, copy=False)
        use_real_fft = n >= 2**21 if real_fft is None else bool(real_fft)
        self._realfft_compatible = bool(
            use_real_fft and np.all((host_idx >= 0) & (host_idx < n))
            and len(np.unique(host_idx)) == host_idx.size)
        idx = jnp.asarray(host_idx, jnp.int32)
        m = idx.shape[0]
        super().__init__((2 * m, n))
        self.idx = idx
        self.n = n
        self.m = m

    def matvec(self, x):
        # Preserve the previous behavior for complex inputs and indices outside
        # the documented unique, nonnegative frequency-subset contract.
        if not self._realfft_compatible or jnp.iscomplexobj(x):
            c = jnp.fft.fft(x, norm="ortho")[self.idx]
            return jnp.concatenate([jnp.real(c), jnp.imag(c)])
        half_idx = jnp.minimum(self.idx, self.n - self.idx)
        c = jnp.fft.rfft(x, norm="ortho")[half_idx]
        imag = jnp.where(self.idx <= self.n // 2, jnp.imag(c), -jnp.imag(c))
        return jnp.concatenate([jnp.real(c), imag])

    def rmatvec(self, u):
        if not self._realfft_compatible or jnp.iscomplexobj(u):
            cc = u[:self.m] + 1j * u[self.m:]
            # The buffer follows the input dtype: complex64 for the float32
            # iterates (unchanged), complex128 for a float64 product under a
            # scoped jax.enable_x64 (the refinement routes).
            z = jnp.zeros(self.n, cc.dtype).at[self.idx].set(cc)
            return jnp.real(jnp.fft.ifft(z, norm="ortho"))
        half_idx = jnp.minimum(self.idx, self.n - self.idx)
        imag = jnp.where(self.idx <= self.n // 2, u[self.m:], -u[self.m:])
        endpoints = (half_idx == 0) | ((self.n % 2 == 0) & (half_idx == self.n // 2))
        cc = (u[:self.m] + 1j * imag) * jnp.where(endpoints, 1., .5)
        z = jnp.zeros(self.n // 2 + 1, cc.dtype).at[half_idx].add(cc)
        return jnp.fft.irfft(z, n=self.n, norm="ortho")

    @classmethod
    def random(cls, n, ratio=0.25, seed=0, *, real_fft=None):
        m = max(1, int(n * ratio))
        idx = np.sort(np.random.default_rng(seed).choice(n, m, replace=False))
        return cls(idx, n, real_fft=real_fft)
