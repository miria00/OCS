"""Advanced matrix-free equilibrators (experimental), for the equilibration bake-off.

All operator-pure: the operator is touched ONLY through matvec/rmatvec; nothing is
formed.  Two methods plus a shared norm estimator:

  Remedy A -- diagpp_equilibration:
      Estimate the Gram diagonals diag(K^T K) (column norms^2) and diag(K K^T)
      (row norms^2) with the low-rank-plus-Hutchinson estimator (Hutch++/Diag++:
      Meyer et al. SOSA'21; Epperly-Tropp-Webber SIMAX'24), which needs p << rank
      matvecs by capturing a low-rank sketch EXACTLY and estimating only the
      residual with a few random probes.  Then Livne-Golub binormalization sets the
      diagonal PDHG metric.

  Bonus -- sketch_norm:
      Estimate ||A||_2 from the same sketch (top singular value of A Q, Q the
      sketched top-right-singular subspace), replacing the 15-iteration power
      method (TwINEst++ lineage; arXiv:2508.04444, 2410.08297).

  Remedy B -- optdiag_equilibration:
      Simplified matrix-free approximation of optimal diagonal preconditioning
      (Qu-Gao-Hinder-Ye-Zhou, Oper. Res. 2024; Gao et al. arXiv:2312.15594):
      target the CONDITION NUMBER via extremal eigenpairs of the scaled Gram matrix
      (Lanczos, matrix-free) and descend the diagonal.  This needs lambda_min, which
      is the expensive part matrix-free -- the cost is the point of the comparison.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .equilibrate import _power_norm


def _rade(key, shape):
    return jnp.where(jax.random.bernoulli(key, 0.5, shape), 1.0, -1.0).astype(jnp.float32)


def _cols(fn, Z):                         # apply a vector op to each column of Z
    return jax.vmap(fn, in_axes=1, out_axes=1)(Z)


def hutchpp_gram_diag(matvec, rmatvec, n, m, which, k, kh, key):
    """Low-rank + Hutchinson estimate of the diagonal of the Gram operator.
    which='col' -> diag(K^T K) in R^n (column norms^2);
    which='row' -> diag(K K^T) in R^m (row norms^2).  Operator-pure.
    Matvec cost: 4k + 2kh operator products (matvec or rmatvec)."""
    if which == "col":
        d = n
        def B(Z):                          # (n,c) -> K^T K Z
            return _cols(rmatvec, _cols(matvec, Z))
    else:
        d = m
        def B(Z):                          # (m,c) -> K K^T Z
            return _cols(matvec, _cols(rmatvec, Z))
    k1, k2 = jax.random.split(key)
    Omega = _rade(k1, (d, k))
    Y = B(Omega)                           # (d,k)   sketch
    Q, _ = jnp.linalg.qr(Y)                # (d,k)   orthonormal top subspace
    BQ = B(Q)                              # (d,k)
    lowrank = jnp.sum(Q * BQ, axis=1)      # diag(Q Q^T B)  (exact for the sketch)
    Psi = _rade(k2, (d, kh))
    BPsi = B(Psi)                          # (d,kh)
    resid = BPsi - Q @ (Q.T @ BPsi)        # (I - Q Q^T) B Psi
    hutch = jnp.mean(Psi * resid, axis=1)  # Hutchinson diag of the residual
    return jnp.maximum(lowrank + hutch, 0.0)


def sketch_norm(mv, rmv, n, k, key):
    """||A||_2 estimate from a k-column sketch: sigma_max(A Q), Q = top-right subspace.
    Cost: ~3k products.  (Bonus: replaces the power-iteration norm.)"""
    Omega = _rade(key, (n, k))
    Q, _ = jnp.linalg.qr(_cols(rmv, _cols(mv, Omega)))   # qr(A^T A Omega): top-right subspace
    AQ = _cols(mv, Q)                                     # (m,k)
    return jnp.linalg.svd(AQ, compute_uv=False)[0]


def diagpp_equilibration(matvec, rmatvec, n, m, scalar_spans=(), *,
                         k=6, kh=4, safety=0.9, seed=0, diagnostics=False,
                         rescale="power"):
    """Remedy A: Hutch++ Gram-diagonal estimate -> Livne-Golub binormalization.
    rescale='power' uses the power method for the global step scale; rescale='sketch'
    uses the Bonus sketch norm (tests whether the sketch replaces power iteration)."""
    spans = tuple((int(a), int(b)) for a, b in scalar_spans)
    key = jax.random.PRNGKey(seed)
    ka, kb, kc, kd = jax.random.split(key, 4)

    @jax.jit
    def run():
        # Sinkhorn col->row pass with FULL equilibration (van der Sluis: equal column
        # norms is within sqrt(n) of optimal).  Floor each estimate away from zero.
        def _floor(x):
            return jnp.maximum(x, 1e-6 * jnp.max(x)) + 1e-30
        dcol = _floor(hutchpp_gram_diag(matvec, rmatvec, n, m, "col", k, kh, ka))
        e = dcol ** (-0.5)                     # full column equilibration: e_j = ||col_j||^{-1}
        mv_e = lambda x: matvec(e * x)         # operator K E (operator-pure)
        rmv_e = lambda y: e * rmatvec(y)
        drow = _floor(hutchpp_gram_diag(mv_e, rmv_e, n, m, "row", k, kh, kb))
        d = drow ** (-0.5)                     # full row equilibration of K E: d_i = ||row_i||^{-1}
        tau_s = e ** 2                         # primal per-col step
        sigma_s = d ** 2                       # dual per-row step
        for (a, b) in spans:                   # scalar-metric blocks: shared row scale
            sigma_s = sigma_s.at[a:b].set(jnp.exp(jnp.mean(jnp.log(sigma_s[a:b]))))
        st, sg = jnp.sqrt(tau_s), jnp.sqrt(sigma_s)
        nrm_pow = _power_norm(lambda x: sg * matvec(st * x),
                              lambda y: st * rmatvec(sg * y), n, iters=15, seed=seed + 7)
        nrm_skt = sketch_norm(lambda x: sg * matvec(st * x),
                              lambda y: st * rmatvec(sg * y), n, k, kc)
        nrm = nrm_skt if rescale == "sketch" else nrm_pow
        scale = safety / (nrm + 1e-30)
        tau, sigma = tau_s * scale, sigma_s * scale
        diag = {}
        if diagnostics:
            def nvar(x, dd):
                return dd * jnp.var(x) / (jnp.sum(x * x) + 1e-30)
            # nvar of the equilibrated operator's row/col norms (fresh probes)
            r1 = hutchpp_gram_diag(lambda z: sg * matvec(st * z),
                                   lambda w: st * rmatvec(sg * w), n, m, "row", k, kh, kd)
            c1 = hutchpp_gram_diag(lambda z: sg * matvec(st * z),
                                   lambda w: st * rmatvec(sg * w), n, m, "col", k, kh, kd)
            diag = {"nvar_row_post": nvar(r1, m), "nvar_col_post": nvar(c1, n),
                    "nrm_power": nrm_pow, "nrm_sketch": nrm_skt}
        return tau, sigma, nrm, diag

    tau, sigma, nrm, diag = run()
    mvp = 2 * (4 * k + 2 * kh) + 30          # products: two Gram diagonals + ~power/sketch
    out = {kk: float(v) for kk, v in diag.items()}
    out["matvec_products"] = mvp
    if diagnostics:
        return tau, sigma, float(nrm), out
    return tau, sigma, float(nrm)
