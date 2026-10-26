"""Mixed-class family for the unified block-metric equilibration study.

`unified_equil.py` reuses F1 / F2 / F3 from `blockmetric_models.py` unchanged so
that results are comparable with the 2026-09-12 block-metric pilot.  This module
adds the one family the pilot did not have: an instance whose blocks genuinely
want DIFFERENT metric classes, so that a single equilibration has to handle a
scalar declaration, a diagonal declaration and a full declaration at once.

F1M  "mixed" : two stacked random partitions of the coordinates (so the blocks
     are coupled through shared coordinates exactly as in F1-overlap), with the
     k x k mixing of every block drawn from one of three structures:

       group S  C_i = g_i * R_i,  R_i random orthogonal
                -> C_i C_i^T = g_i^2 I : the block Gram is ISOTROPIC when the
                   primal scaling is uniform over the block, so a SCALAR metric
                   is the right class and a diagonal one buys nothing.
       group D  C_i = g_i * diag(w_i) / rms(w_i),  w_i log-uniform in [1, 100]
                -> the block Gram is DIAGONAL and badly scaled: a diagonal
                   metric is the right class, a scalar one cannot fix it and a
                   full one has no off-diagonal work to do.
       group F  C_i = g_i * c0 [(1-rho) I + rho a_i a_i^T],  rho = 0.99
                -> strongly coupled block Gram, condition 1/(1-rho) = 100
                   within the block: only a FULL metric fixes it.

     g_i is log-uniform in [1, 10] across all blocks, so the blocks also differ
     in overall scale and the per-block scalar level has work to do everywhere.

The operator stores its per-block k x k mixing as one (B, k, k) array and
applies it with a batched einsum.  That is MODEL data, the same way the pilot's
F1 stores its rank-one mixing: the equilibration never sees it, it only calls
matvec / rmatvec.  `dense_blocks()` / `exact_gram()` exist only as a diagnostic
reference and are never used inside a reported operator-only arm.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import ocs
from ocs.ir import Term
from ocs.operators.base import LinearOperator
from ocs.prox.library import SumSquares

from blockmetric_models import Spec, StackedBlockMix


class BlockMatSelect(LinearOperator):
    """K x = stack_i C_i x_{S_i} over a partition S of the coordinates, with the
    per-block k x k matrices carried as one (B, k, k) array (model data)."""

    def __init__(self, perm, C):
        B, k, _ = C.shape
        n = perm.shape[0]
        super().__init__((B * k, n))
        self.perm = jnp.asarray(perm, jnp.int32)
        self.inv_perm = jnp.asarray(np.argsort(perm), jnp.int32)
        self.C = jnp.asarray(C, jnp.float32)
        self._C64 = np.asarray(C, np.float64)
        self.B, self.k = B, k

    def matvec(self, x):
        z = x[self.perm].reshape(self.B, self.k)
        return jnp.einsum("bij,bj->bi", self.C, z,
                          precision=jax.lax.Precision.HIGHEST).reshape(-1)

    def rmatvec(self, y):
        z = jnp.einsum("bji,bj->bi", self.C, y.reshape(self.B, self.k),
                       precision=jax.lax.Precision.HIGHEST).reshape(-1)
        return z[self.inv_perm]

    def dense_blocks(self):
        return self._C64


def _group_blocks(rng, B, k, rho=0.99):
    """(B,k,k) mixings and the (B,) class codes 0=scalar, 1=diag, 2=full."""
    codes = np.arange(B) % 3                      # 0 -> S, 1 -> D, 2 -> F
    g = np.exp(rng.uniform(0.0, np.log(10.0), size=B))          # per-block scale
    C = np.zeros((B, k, k))
    for i in range(B):
        if codes[i] == 0:                          # isotropic: random orthogonal
            Q, R = np.linalg.qr(rng.normal(size=(k, k)))
            Q = Q * np.sign(np.diag(R))
            C[i] = Q
        elif codes[i] == 1:                        # diagonal, badly scaled
            w = np.exp(rng.uniform(0.0, np.log(100.0), size=k))
            C[i] = np.diag(w / np.sqrt(np.mean(w ** 2)))
        else:                                      # strongly coupled
            s = rng.choice([-1.0, 1.0], size=k)
            a = s / np.sqrt(k)
            M = (1 - rho) * np.eye(k) + rho * np.outer(a, a)
            rn = np.sqrt(np.mean(np.sum(M ** 2, axis=1)))
            C[i] = M / rn
    return C * g[:, None, None], codes


def unified_f1_mixed(n=200_000, k=4, seed=0, mu=1e-2, lam=1.0, rho=0.99):
    """F1M: coupled two-partition instance whose blocks want three different
    metric classes.  Returns (ocs problem, Spec); the Spec carries
    `block_classes`, a list per dual term of a (B,) int array of class codes."""
    rng = np.random.default_rng(seed)
    assert n % k == 0
    B = n // k
    perm1 = rng.permutation(n)
    perm2 = rng.permutation(n)
    C1, code1 = _group_blocks(rng, B, k, rho)
    C2, code2 = _group_blocks(rng, B, k, rho)
    K1, K2 = BlockMatSelect(perm1, C1), BlockMatSelect(perm2, C2)
    K = StackedBlockMix(K1, K2)
    perm = np.concatenate([perm1, perm2])
    codes = np.concatenate([code1, code2])
    m = K.shape[0]

    x_true = rng.normal(size=n)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(x_true, jnp.float32))) + 0.3 * rng.normal(size=m)
    b = b.astype(np.float32)

    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([Term(ocs.Identity(n), SumSquares().scaled(mu)),
                         ocs.group_l2(K @ x - bp, lam=lam, group_size=k)])
    data = {"b": jnp.asarray(b)}

    Call = np.concatenate([C1, C2])
    perm_np = np.asarray(perm)

    def exact_gram(t0, s0):                       # diagnostic reference only
        t = np.asarray(t0, np.float64)
        t = np.broadcast_to(t, (n,))[perm_np].reshape(2 * B, k)
        return s0 * np.einsum("bij,bj,blj->bil", Call, t, Call)

    def exact_coldiag(sig_blocks):
        """diag(K^T Sigma K) exactly, for a (2B,k,k) dual metric.  Diagnostic
        reference for the column-norm probe estimator; never used in an arm."""
        S = np.asarray(sig_blocks, np.float64)
        # (C^T S C)_{pp} summed into the coordinate that block-position p gathers
        A = np.einsum("bji,bjl,blm->bim", Call, S, Call)
        d = np.einsum("bii->bi", A).reshape(-1)
        out = np.zeros(n)
        np.add.at(out, perm_np, d)
        return out

    spec = Spec(n=n, mu=mu, primal="ridge", data=data,
                name=f"F1M-mixed-n{n}-k{k}-rho{rho}",
                note="mixed-class: 1/3 isotropic (scalar), 1/3 diagonal, 1/3 coupled (full); two stacked partitions",
                dual_blocks=[dict(op=K, kind="group_l2", k=k, lam=lam,
                                  offset=jnp.asarray(-b), exact_gram=exact_gram)],
                exact_norm=None, family="F1M", rho=rho, k=k, variant="mixed",
                block_classes=[codes], exact_coldiag=exact_coldiag)
    return prob, spec
