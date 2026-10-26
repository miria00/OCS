"""Differentiable MoE routing as a regularized assignment layer (OCS).

Route N tokens to E experts through the quadratically regularized balanced
transportation program (the killer_ot formulation)

    minimize    <c, x> + (eps/2) ||x||^2
    subject to  x >= 0,   Kx = [1_N ; (N/E) 1_E],

with x = vec(X) in R^{N E}, K the marginal operator (rows and columns),
c = -vec(C) the negative router affinities.  The rows of X are the routing
gates (each sums to one).  Balance is EXACT by construction
(BASE-layer style); the quadratic term makes the layer differentiable with
informative gradients (an LP's argmin is piecewise constant in C) and keeps
the routing SPARSE, unlike entropic (Sinkhorn) routing; eps sets the
emergent experts-per-token.

Configs mimic Mixtral (E=8, top-2) and DeepSeek-MoE (E=64, top-6;
V3-shaped E=256, top-8).  Baselines: top-k routing and a Sinkhorn router.
Gradients through the solve (w.r.t. the logits) are checked against finite
differences.  Every solve is the OCS restart-PDHG engine.

Run:  PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python examples/moe_router.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time

import numpy as np
import jax
import jax.numpy as jnp

import ocs


# ---- the routing layer ---------------------------------------------------
def build_router(N, E, eps=0.1, **compile_kwargs):
    """Compile the balanced routing family once; logits enter only the cost."""
    K = ocs.MarginalOperator(N, E)
    x = ocs.Variable(N * E)
    c = ocs.Parameter("c")
    m = ocs.Parameter("m")
    se = jnp.full(N * E, np.sqrt(eps), jnp.float32)
    prob = ocs.minimize([ocs.nonneg(x),
                         ocs.equality(K @ x - m),
                         ocs.sum_squares(ocs.Diagonal(se) @ x)], cost=c)
    compiled = prob.compile(**compile_kwargs)
    mrg = np.concatenate([np.full(N, 1.0), np.full(E, N / E)]
                         ).astype(np.float32)

    def data_of(logits):
        return {"c": -jnp.asarray(logits, jnp.float32).reshape(-1), "m": mrg}

    return compiled, data_of, mrg


def gates_of(x, N, E):
    """Transport plan -> routing gates: clip and row-normalize (the layer's
    differentiable post-step; rows sum to one exactly)."""
    X = np.maximum(np.asarray(x).reshape(N, E), 0.0)
    return X / (X.sum(1, keepdims=True) + 1e-9)


def raw_row_err(x, N, E):
    X = np.asarray(x).reshape(N, E)
    return np.abs(X.sum(1) - 1.0).max()


# ---- structured logits (tokens with cluster affinity) --------------------
def make_logits(key, N, E, d=64, temp=2.0):
    """Two-center mixtures: tokens carry genuine multi-expert affinity."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    centers = jax.random.normal(k1, (E, d))
    a = jax.random.randint(k2, (N,), 0, E)
    b = jax.random.randint(k3, (N,), 0, E)
    w = jax.random.uniform(k4, (N, 1), minval=0.35, maxval=0.65)
    h = w * centers[a] + (1 - w) * centers[b] + 0.5 * jax.random.normal(k5, (N, d))
    logits = temp * (h @ centers.T) / np.sqrt(d)
    return np.asarray(logits, np.float32)


# ---- baselines -----------------------------------------------------------
def topk_route(C, k, cap):
    """Top-k with softmax gates; count capacity overflow (dropped tokens)."""
    N, E = C.shape
    idx = np.argsort(-C, axis=1)[:, :k]
    gates = np.exp(C[np.arange(N)[:, None], idx])
    gates = gates / gates.sum(1, keepdims=True)
    X = np.zeros((N, E), np.float32)
    X[np.arange(N)[:, None], idx] = gates
    load = X.sum(0)
    overflow = np.maximum(load - cap, 0.0).sum() / N
    return X, overflow


def sinkhorn_route(C, cap, eps=0.05, iters=200):
    """Entropic OT to the same balanced marginals (ends on the row update)."""
    N, E = C.shape
    logK = C / eps
    la = np.zeros(N)                                    # log 1
    lb = np.log(np.full(E, N / E))
    u = np.zeros(N); v = np.zeros(E)
    for _ in range(iters):
        v = lb - _lse(logK + u[:, None], axis=0)
        u = la - _lse(logK + v[None, :], axis=1)
    return np.exp(logK + u[:, None] + v[None, :]).astype(np.float32)


def _lse(A, axis):
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# ---- metrics -------------------------------------------------------------
def report(name, X, C, cap, thresh=1e-2):
    X = np.asarray(X); N, E = X.shape
    load = X.sum(0)
    row_err = np.abs(X.sum(1) - 1).max()
    cap_viol = max(load.max() - cap, 0.0) / cap
    nnz = (X > thresh).sum(1).mean()
    aff = float((C * X).sum())
    print(f"    {name:22s} affinity={aff:10.2f}  balance(max/mean)="
          f"{load.max()/load.mean():5.3f}  cap_viol={cap_viol:7.1e}  "
          f"row_err={row_err:7.1e}  avg_experts/token={nnz:5.2f}")
    return aff


# ---- main ----------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # eps sweep: the emergent experts/token knob (Mixtral-shaped, small)
    print("== eps sweep (N=512, E=8): emergent experts/token ==")
    N0, E0 = 512, 8
    C0 = make_logits(jax.random.PRNGKey(7), N0, E0)
    for eps in (0.05, 0.1, 0.3, 1.0, 3.0):
        comp, dof, _ = build_router(N0, E0, eps=eps, tol=1e-4, max_blocks=800)
        s = comp.solve_instance(dof(C0))
        G = gates_of(s.x, N0, E0)
        nnz = (G > 1e-2).sum(1).mean()
        print(f"    eps={eps:4.2f}: experts/token {nnz:5.2f}  "
              f"residual {float(s.residual):.1e}")

    configs = [
        ("Mixtral-shaped",  2048,   8, 2, 0.30, 1e-4, 1600),
        ("DeepSeek-shaped", 2048,  64, 6, 3.00, 1e-4, 3200),
        ("V3-shaped",       1024, 256, 8, 3.00, 1e-4, 3200),
    ]
    for name, N, E, k, eps, tol, mb in configs:
        print(f"\n== {name}: N={N} tokens, E={E} experts, top-{k} ==")
        C = make_logits(key, N, E)
        cap = N / E

        t0 = time.time()
        compiled, data_of, mrg = build_router(N, E, eps=eps, tol=tol,
                                              max_blocks=mb)
        t_compile = time.time() - t0
        sol = compiled.solve_instance(data_of(C))          # warm-up + jit
        t0 = time.time()
        sol = compiled.solve_instance(data_of(C))
        t_solve = time.time() - t0
        G_ocs = gates_of(sol.x, N, E)
        print(f"    compile {t_compile:.1f}s; steady-state solve "
              f"{t_solve*1e3:.1f} ms  (residual {float(sol.residual):.1e}, "
              f"raw row-err {raw_row_err(sol.x, N, E):.2f} -> normalized)")

        report("OCS regularized-OT", G_ocs, C, cap)
        Xt, overflow = topk_route(C, k, cap)
        report(f"top-{k} softmax", Xt, C, cap)
        print(f"      (top-{k} capacity overflow: {overflow*100:.1f}% of tokens)")
        Xs = sinkhorn_route(C, cap)
        report("Sinkhorn (eps=0.05)", Xs, C, cap)

        # batched: L=8 layers' logits in ONE call
        L = 8
        keys = jax.random.split(jax.random.PRNGKey(1), L)
        Cb = np.stack([make_logits(kk, N, E) for kk in keys])
        db = {"c": -Cb.reshape(L, -1), "m": np.tile(mrg, (L, 1))}
        out = compiled.solve_many(db)                      # warm-up
        t0 = time.time()
        out = compiled.solve_many(db)
        t_many = time.time() - t0
        res_b = np.asarray(out[-1])
        print(f"    solve_many: {L} layers in {t_many*1e3:.1f} ms "
              f"({t_many/L*1e3:.2f} ms/layer, max residual {res_b.max():.1e})")

    # ---- gradient check: d loss / d logits through the solve -------------
    print("\n== gradient through the layer (small config) ==")
    N, E = 64, 8
    C0 = make_logits(jax.random.PRNGKey(2), N, E)
    compiled, data_of, mrg = build_router(N, E, eps=0.1, tol=1e-5,
                                          max_blocks=1600)
    Gw = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (N * E,)),
                    np.float32)

    def loss(logits):
        data = {"c": -logits.reshape(-1), "m": jnp.asarray(mrg)}
        x = compiled.solve_diff(data)
        return jnp.dot(jnp.asarray(Gw), x)

    g = np.asarray(jax.grad(loss)(jnp.asarray(C0)))
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(6):
        i, j = rng.integers(N), rng.integers(E)
        h = 1e-2
        Cp = C0.copy(); Cp[i, j] += h
        Cm = C0.copy(); Cm[i, j] -= h
        fd = (float(loss(jnp.asarray(Cp))) - float(loss(jnp.asarray(Cm)))) / (2 * h)
        errs.append(abs(fd - g[i, j]) / (1 + abs(fd)))
    print(f"    jax.grad vs central differences on 6 random logits: "
          f"max rel err {max(errs):.2e}")
    print(f"    grad norm {np.linalg.norm(g):.3f} (nonzero: the quadratic term "
          "makes the layer differentiable; an LP would give 0 a.e.)")
