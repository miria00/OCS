"""Train a toy MoE end to end through three routers: OCS (gradients through
the regularized-assignment solve), Sinkhorn (entropic, unrolled), and top-k
with the standard auxiliary load-balancing loss.

Task: tokens h_i drawn from E clusters; target y_i = W*_{z_i} h_i, where z_i
is the ground-truth cluster's expert.  Model: router MLP -> logits ->
routing X -> output sum_j X_ij (W_j h_i).  Router and experts train jointly;
the router only learns *through the routing*, so this isolates whether
gradients through the assignment help.

Run:  PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python examples/moe_router_train.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time

import numpy as np
import jax
import jax.numpy as jnp

from examples.moe_router import build_router

# ---- task ----------------------------------------------------------------
D, E, N, STEPS, LR = 32, 8, 256, 200, 1e-2
key = jax.random.PRNGKey(0)
kc, kw, kd = jax.random.split(key, 3)
centers = jax.random.normal(kc, (E, D))
W_true = jax.random.normal(kw, (E, D, D)) / np.sqrt(D)


def batch(k):
    k1, k2 = jax.random.split(k)
    z = jax.random.randint(k1, (N,), 0, E)
    h = centers[z] + 0.5 * jax.random.normal(k2, (N, D))
    y = jnp.einsum("ndf,nf->nd", W_true[z], h)
    return h, y, z


# ---- model pieces --------------------------------------------------------
def init_params(k):
    k1, k2, k3 = jax.random.split(k, 3)
    return {
        "r1": jax.random.normal(k1, (D, 64)) / np.sqrt(D),
        "r2": jax.random.normal(k2, (64, E)) / 8.0,
        "W": jax.random.normal(k3, (E, D, D)) / np.sqrt(D),
    }


def logits_of(p, h):
    return jnp.tanh(h @ p["r1"]) @ p["r2"]


def output_of(p, h, X):
    per_expert = jnp.einsum("edf,nf->ned", p["W"], h)        # (N, E, D)
    return jnp.einsum("ne,ned->nd", X, per_expert)


# ---- routers -------------------------------------------------------------
compiled, _data_of, _mrg = build_router(N, E, eps=0.3, tol=1e-3, max_blocks=800)
_mrg_j = jnp.asarray(_mrg)


def route_ocs(logits):
    data = {"c": -logits.reshape(-1), "m": _mrg_j}
    x = compiled.solve_diff(data)
    X = jnp.maximum(x.reshape(N, E), 0.0)
    return X / (X.sum(1, keepdims=True) + 1e-9)


def route_sinkhorn(logits, eps=0.1, iters=30):
    la = jnp.zeros(N)
    lb = jnp.log(jnp.full(E, N / E))
    K = logits / eps
    u = jnp.zeros(N); v = jnp.zeros(E)
    for _ in range(iters):
        u = la - jax.scipy.special.logsumexp(K + v[None, :], axis=1)
        v = lb - jax.scipy.special.logsumexp(K + u[:, None], axis=0)
    return jnp.exp(K + u[:, None] + v[None, :])


def route_topk(logits, k=2):
    # straight-through top-k with softmax gates; grads via the gates only
    gates = jax.nn.softmax(logits, axis=1)
    thresh = jnp.sort(gates, axis=1)[:, -k][:, None]
    mask = jax.lax.stop_gradient((gates >= thresh).astype(jnp.float32))
    Xg = gates * mask
    return Xg / (Xg.sum(1, keepdims=True) + 1e-9)


def aux_balance_loss(logits, X):
    # Switch-transformer style: E * sum_e (fraction routed) * (mean prob)
    P = jax.nn.softmax(logits, axis=1).mean(0)
    f = (X > 1e-3).astype(jnp.float32).mean(0)
    return E * jnp.sum(f * P)


# ---- training ------------------------------------------------------------
def train(router, name, aux_w=0.0):
    p = init_params(jax.random.PRNGKey(1))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)     # Adam state
    v = jax.tree_util.tree_map(jnp.zeros_like, p)

    def loss_fn(p, h, y):
        lg = logits_of(p, h)
        X = router(lg)
        out = output_of(p, h, X)
        l = jnp.mean((out - y) ** 2)
        if aux_w > 0:
            l = l + aux_w * aux_balance_loss(lg, X)
        return l, X

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    t0 = time.time()
    hist = []
    for step in range(STEPS):
        h, y, z = batch(jax.random.PRNGKey(100 + step))
        (l, X), g = grad_fn(p, h, y)
        # Adam
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        t = step + 1
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - LR * (mm / (1 - b1 ** t))
            / (jnp.sqrt(vv / (1 - b2 ** t)) + eps), p, m, v)
        hist.append(float(l))
        if step % 50 == 0 or step == STEPS - 1:
            load = np.asarray(X.sum(0)); load = load / load.mean()
            acc = float((jnp.argmax(X, 1) == z).mean())
            print(f"    step {step:3d}  loss {float(l):8.4f}  "
                  f"balance(max/mean) {load.max():5.2f}  route-acc {acc:.2f}")
    dt = time.time() - t0
    h, y, z = batch(jax.random.PRNGKey(999))
    (l, X), _ = grad_fn(p, h, y)
    acc = float((jnp.argmax(X, 1) == z).mean())
    load = np.asarray(X.sum(0)); load = load / load.mean()
    print(f"    FINAL {name}: eval loss {float(l):.4f}, route-acc {acc:.2f}, "
          f"balance {load.max():.2f}, {dt/STEPS*1e3:.0f} ms/step")
    return float(l), acc, float(load.max())


if __name__ == "__main__":
    print(f"toy MoE: N={N} tokens/step, E={E} experts, d={D}, {STEPS} steps")
    print("\n== OCS regularized-assignment router (grads THROUGH the solve) ==")
    r_ocs = train(route_ocs, "OCS")
    print("\n== Sinkhorn router (entropic, unrolled 30 iters) ==")
    r_sh = train(route_sinkhorn, "Sinkhorn")
    print("\n== top-2 router + aux balance loss (standard practice) ==")
    r_tk = train(lambda lg: route_topk(lg, 2), "top-2+aux", aux_w=0.01)

    print("\n== summary (eval loss / route accuracy / balance max-mean) ==")
    for nm, r in [("OCS", r_ocs), ("Sinkhorn", r_sh), ("top-2+aux", r_tk)]:
        print(f"    {nm:10s} loss {r[0]:8.4f}   acc {r[1]:.2f}   balance {r[2]:.2f}")
