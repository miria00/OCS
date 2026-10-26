"""Counted copy of the E-reference vectorized weighted SOC projection.

The arithmetic is the E-reference ``projx.proj_block_rows`` line for line
(Newton from the side of monotone convergence, PDHCG-CQP arXiv 2608.09159
App. C.2 and PDCS arXiv 2505.00311 for the one-root form). The only addition
is a counter: for every cone that enters the Newton loop, the number of Newton
iterations it took before it was marked done. fair_test_projx.py checks this
copy bitwise against projx and against the verified projectors.
"""
import numpy as np

NEWTON_MAX = 200
NEWTON_RTOL = 1e-12


def _norm_rows(W):
    return np.sqrt(np.einsum("ij,ij->i", W, W))


def proj_block_rows(T, Wr, a, b, fallback=None, stats=None, count=None):
    N, d = Wr.shape
    R = _norm_rows(Wr)
    feas = R <= T
    Bw = _norm_rows(b * Wr)
    polar = (~feas) & (-a * T >= Bw)
    act = (~feas) & (~polar)
    Tp = np.where(feas, T, 0.0)
    Zr = np.where(feas[:, None], Wr, 0.0)
    zero = act & (T == 0.0)
    if zero.any():
        i = np.where(zero)[0]
        z = b[i] * Wr[i] / (b[i] + a[i][:, None])
        Zr[i] = z
        Tp[i] = _norm_rows(z)
    bad = []
    for branch in ("pos", "neg"):
        i = np.where(act & ((T > 0.0) if branch == "pos" else (T < 0.0)))[0]
        if i.size == 0:
            continue
        A = a[i]
        B = b[i]
        w = Wr[i]
        tau = T[i]
        s = A.copy() if branch == "pos" else np.zeros_like(A)
        done = np.zeros(i.size, bool)
        iters = 0
        for it in range(NEWTON_MAX):
            iters += int(np.count_nonzero(~done))
            den = (B + (A - s)[:, None]) if branch == "pos" else (B + (A + s)[:, None])
            u = B * w / den
            phi = _norm_rows(u)
            if branch == "pos":
                h = s * phi - A * tau
                dphi = np.einsum("ij,ij->i", u, u / den) / np.where(phi > 0, phi, 1.0)
            else:
                h = s * phi + A * tau
                dphi = -np.einsum("ij,ij->i", u, u / den) / np.where(phi > 0, phi, 1.0)
            hp = phi + s * dphi
            side = (h <= 0.0) if branch == "pos" else (h >= 0.0)
            step = np.where(done | side, 0.0, h / np.where(hp > 0, hp, 1.0))
            s_new = s - step
            if branch == "pos":
                s_new = np.clip(s_new, 0.0, A)
            else:
                s_new = np.maximum(s_new, 0.0)
            newly = np.abs(s_new - s) <= NEWTON_RTOL * np.maximum(s_new, 1e-300)
            s = np.where(done, s, s_new)
            done |= newly | side
            if done.all():
                break
        if stats is not None:
            stats["newton_max_iter"] = max(stats.get("newton_max_iter", 0), it + 1)
        if count is not None:
            count["newton_iters"] = count.get("newton_iters", 0) + iters
            count["root_cones"] = count.get("root_cones", 0) + int(i.size)
        lam = (A - s) if branch == "pos" else (A + s)
        good = done & (s > 0)
        zr = B * w / (B + lam[:, None])
        t = A * np.abs(tau) / np.where(s > 0, s, 1.0)
        Zr[i[good]] = zr[good]
        Tp[i[good]] = t[good]
        for jj in np.where(~good)[0]:
            bad.append(i[jj])
    if bad:
        if fallback is None:
            raise RuntimeError(f"{len(bad)} cones did not converge and no fallback was given")
        for j in bad:
            t, zr = fallback(j, T[j], Wr[j])
            Tp[j] = t
            Zr[j] = zr
        if stats is not None:
            stats["fallbacks"] = stats.get("fallbacks", 0) + len(bad)
    return Tp, Zr
