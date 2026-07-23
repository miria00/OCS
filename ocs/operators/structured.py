"""Structured operators for the matrix-variable / fast-transform problems that are
  accessed only through matvec/rmatvec, never formed.

  MarginalOperator : X (N x M) |-> (row sums, col sums)   [optimal transport]
  Gradient2D       : image (H x W) |-> per-pixel (d_x, d_y) forward differences
                     [total variation / MRI / imaging]
"""
from __future__ import annotations

import jax.numpy as jnp

from .base import LinearOperator


class MarginalOperator(LinearOperator):
    """x = vec(X), X in R^{N x M}.  K x = [X 1_M ; X^T 1_N]  in R^{N+M}.
    Never forms the (N+M) x (NM) constraint matrix."""
    def __init__(self, N, Mc):
        super().__init__((N + Mc, N * Mc))
        self.N, self.Mc = N, Mc

    def matvec(self, x):
        X = x.reshape(self.N, self.Mc)
        return jnp.concatenate([X.sum(axis=1), X.sum(axis=0)])

    def rmatvec(self, y):
        u, v = y[:self.N], y[self.N:]
        return (u[:, None] + v[None, :]).reshape(-1)      # X_ij <- u_i + v_j


class Gradient3D(LinearOperator):
    """volume x (D x H x W), row-major vec.  K x = per-voxel (d_x, d_y, d_z)
    forward differences, laid out contiguously per voxel (group_size=3 gives
    isotropic 3-D TV).  Output dim = 3 D H W."""
    def __init__(self, D, H, W):
        super().__init__((3 * D * H * W, D * H * W))
        self.D, self.H, self.W = D, H, W

    def matvec(self, x):
        v = x.reshape(self.D, self.H, self.W)
        dx = jnp.zeros_like(v).at[:-1].set(v[1:] - v[:-1])
        dy = jnp.zeros_like(v).at[:, :-1].set(v[:, 1:] - v[:, :-1])
        dz = jnp.zeros_like(v).at[:, :, :-1].set(v[:, :, 1:] - v[:, :, :-1])
        return jnp.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], 1).reshape(-1)

    def rmatvec(self, g):
        g = g.reshape(-1, 3)
        D, H, W = self.D, self.H, self.W
        dx = g[:, 0].reshape(D, H, W); dy = g[:, 1].reshape(D, H, W); dz = g[:, 2].reshape(D, H, W)
        a = jnp.zeros((D, H, W), jnp.float32)
        a = a.at[1:].add(dx[:-1]).at[:-1].add(-dx[:-1])
        a = a.at[:, 1:].add(dy[:, :-1]).at[:, :-1].add(-dy[:, :-1])
        a = a.at[:, :, 1:].add(dz[:, :, :-1]).at[:, :, :-1].add(-dz[:, :, :-1])
        return a.reshape(-1)


class Kronecker(LinearOperator):
    """X (p x q) matrix variable, row-major vec.  K vec(X) = vec(A X B^T),
    A (m x p), B (n x q)  ==  (A (x) B) vec(X).  Never forms the (mn) x (pq)
    Kronecker matrix; applies A and B to the p x q block directly."""
    def __init__(self, A, B):
        A = jnp.asarray(A, jnp.float32); B = jnp.asarray(B, jnp.float32)
        self.A, self.B = A, B
        self.m, self.p = A.shape
        self.n, self.q = B.shape
        super().__init__((self.m * self.n, self.p * self.q))

    def matvec(self, x):
        return (self.A @ x.reshape(self.p, self.q) @ self.B.T).reshape(-1)

    def rmatvec(self, y):
        return (self.A.T @ y.reshape(self.m, self.n) @ self.B).reshape(-1)


class Sylvester(LinearOperator):
    """X (p x q) matrix variable, row-major vec.  K vec(X) = vec(A X + X B),
    A (p x p), B (q x q).  The Sylvester map, never formed (the explicit matrix
    would be I(x)A + B^T(x)I of size (pq) x (pq))."""
    def __init__(self, A, B):
        A = jnp.asarray(A, jnp.float32); B = jnp.asarray(B, jnp.float32)
        p, q = A.shape[0], B.shape[0]
        super().__init__((p * q, p * q))
        self.A, self.B, self.p, self.q = A, B, p, q

    def matvec(self, x):
        X = x.reshape(self.p, self.q)
        return (self.A @ X + X @ self.B).reshape(-1)

    def rmatvec(self, y):
        Y = y.reshape(self.p, self.q)
        return (self.A.T @ Y + Y @ self.B.T).reshape(-1)


class BinPack(LinearOperator):
    """Fractional bin-packing structure.  x = vec(X), X (I items x B bins).
    K x = [ X 1_B  (per-item assignment) ; vec(Rmat X) (per-resource, per-bin
    usage) ], with the fixed resource matrix Rmat (R x I).  Never formed."""
    def __init__(self, Rmat, B):
        Rmat = jnp.asarray(Rmat, jnp.float32)
        R, I = Rmat.shape
        super().__init__((I + R * B, I * B))
        self.Rmat, self.R, self.I, self.B = Rmat, R, I, B

    def matvec(self, x):
        X = x.reshape(self.I, self.B)
        return jnp.concatenate([X.sum(axis=1), (self.Rmat @ X).reshape(-1)])

    def rmatvec(self, y):
        u = y[:self.I]
        Wm = y[self.I:].reshape(self.R, self.B)
        return (u[:, None] + self.Rmat.T @ Wm).reshape(-1)


class LeftMatMul(LinearOperator):
    """Theta (F x T) matrix variable, row-major vec.  K vec(Theta) = vec(A Theta),
    A in R^{m x F}.  The matrix variable is never expanded into a giant (mF)-vector;
    the operator applies A to the F x T block directly.  (For feature selection:
    row-group l2 on Theta selects features.)"""
    def __init__(self, A, T):
        A = jnp.asarray(A, jnp.float32)
        m, F = A.shape
        super().__init__((m * T, F * T))
        self.A, self.m, self.F, self.T = A, m, F, T

    def matvec(self, x):
        return (self.A @ x.reshape(self.F, self.T)).reshape(-1)

    def rmatvec(self, y):
        return (self.A.T @ y.reshape(self.m, self.T)).reshape(-1)


class MultiCommodityFlow(LinearOperator):
    """x = vec(F), F in R^{E x K} (E arcs, K commodities).  matvec = the stacked
    per-commodity flow conservation vec(N F) in R^{V*K}, where N (V x E) is the
    node-arc incidence.  The block-diagonal incidence (one N per commodity) is
    never formed; N is applied to the E x K flow block directly.  The shared-arc
    capacity coupling sum_k f_k is the separate CommoditySum operator."""
    def __init__(self, N, K):
        N = jnp.asarray(N, jnp.float32)
        V, E = N.shape
        super().__init__((V * K, E * K))
        self.N, self.V, self.E, self.K = N, V, E, K

    def matvec(self, x):
        return (self.N @ x.reshape(self.E, self.K)).reshape(-1)

    def rmatvec(self, y):
        return (self.N.T @ y.reshape(self.V, self.K)).reshape(-1)


class CommoditySum(LinearOperator):
    """x = vec(F), F in R^{E x K}.  matvec = sum_k f_k in R^E (per-arc total usage
    across commodities) -- the coupling for a shared-capacity constraint."""
    def __init__(self, E, K):
        super().__init__((E, E * K))
        self.E, self.K = E, K

    def matvec(self, x):
        return x.reshape(self.E, self.K).sum(axis=1)

    def rmatvec(self, y):
        return jnp.broadcast_to(y[:, None], (self.E, self.K)).reshape(-1)


class Gradient2D(LinearOperator):
    """image x (H x W), row-major vec.  K x = per-pixel (d_x, d_y) forward
    differences (Neumann boundary), laid out contiguously per pixel so a
    group-l2 block of size 2 gives isotropic TV.  Output dim = 2 H W."""
    def __init__(self, H, W):
        super().__init__((2 * H * W, H * W))
        self.H, self.W = H, W

    def matvec(self, x):
        H, W = self.H, self.W
        img = x.reshape(H, W)
        dx = jnp.zeros_like(img).at[:-1, :].set(img[1:, :] - img[:-1, :])
        dy = jnp.zeros_like(img).at[:, :-1].set(img[:, 1:] - img[:, :-1])
        return jnp.stack([dx.reshape(-1), dy.reshape(-1)], axis=1).reshape(-1)

    def rmatvec(self, g):
        H, W = self.H, self.W
        g = g.reshape(H * W, 2)
        dx = g[:, 0].reshape(H, W); dy = g[:, 1].reshape(H, W)
        # adjoint of forward difference = negative backward difference
        ax = jnp.zeros_like(dx)
        ax = ax.at[1:, :].add(dx[:-1, :]).at[:-1, :].add(-dx[:-1, :])
        ay = jnp.zeros_like(dy)
        ay = ay.at[:, 1:].add(dy[:, :-1]).at[:, :-1].add(-dy[:, :-1])
        return (ax + ay).reshape(-1)
