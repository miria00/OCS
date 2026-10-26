"""A/B: residual-targeted re-equilibration at a restart boundary (the CORGI
"one-armed VM", recast as prescribed-marginal SMFE) vs. plain SMFE.

Protocol (equal total iteration budget):
  BASELINE   compile with SMFE, run 2B blocks.
  TREATMENT  compile with SMFE, run B blocks; from the iterate compute the
             per-row primal violation Ry and per-column dual violation Rx;
             pick the blocking side (mean ratio > 5 -> one-sided, else both);
             re-equilibrate with prescribed marginals (rho ~ Ry, gam ~ Rx,
             uniform on the frozen side); recompile with the given steps and
             continue B blocks warm-started from the same (x, y).
             (Re-equilibration cost: ~160 operator products ~ 1.6 blocks.)

Testbeds: (A) ill-scaled LPs k=1..3 (currently failing 1e-2 agreement),
(B) the SOCP knife-edge instances, (C) the MoE E=256 dead-rows regime.

Run:  PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python examples/requil_ab.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import scipy.sparse as _sp
for _cls in (_sp.csc_matrix, _sp.csr_matrix, _sp.coo_matrix):
    if not hasattr(_cls, "A"):
        _cls.A = property(lambda self: self.toarray())
import jax
import jax.numpy as jnp

import ocs
from ocs.solver.equilibrate_prescribed import prescribed_smfe

TINY_TOL = 1e-7      # never reached: both arms run their full block budget


def targets(Ry, Rx, ratio=5.0):
    """One-sided residual marginals: (rho, gam, tag)."""
    pr, du = float(np.mean(Ry)), float(np.mean(Rx))
    m, n = Ry.shape[0], Rx.shape[0]
    if pr > ratio * du:
        return Ry, np.ones(n, np.float32), "rows(primal)"
    if du > ratio * pr:
        return np.ones(m, np.float32), Rx, "cols(dual)"
    return Ry, Rx, "both"


# ====================== A: ill-scaled LPs ================================
def lp_ab(k, seed, B=750):
    from scipy.optimize import linprog
    r = np.random.default_rng(seed)
    mA, n = 40, 80
    A = r.normal(size=(mA, n)).astype("f")
    d = 10.0 ** r.uniform(-k, k, size=mA).astype("f")
    e = 10.0 ** r.uniform(-k, k, size=n).astype("f")
    A = (A * e[None, :]) * d[:, None]
    xt = np.abs(r.normal(size=n)).astype("f"); b = A @ xt
    c = np.abs(r.normal(size=n)).astype("f") * e
    ref = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun

    def fam():
        x = ocs.Variable(n)
        return ocs.minimize([ocs.nonneg(x),
                             ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                            cost=ocs.Parameter("c"))
    data = {"b": b, "c": c}
    rel = lambda xo: abs(float(c @ xo) - ref) / (1 + abs(ref))

    # baseline: 2B blocks
    base = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=2 * B)
    sb = base.solve_instance(data)

    # treatment: B + re-equilibrate + B
    c1 = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=B)
    s1 = c1.solve_instance(data)
    x1, y1 = np.asarray(s1.x), np.asarray(s1.y)
    # nonneg(x) is the primal g-block: the dual stack is the equality rows only
    Ry = np.abs(A @ x1 - b).astype("f")                          # per-row primal viol
    g = c + A.T @ y1
    Rx = np.abs(x1 - np.maximum(x1 - g, 0.0)).astype("f")        # per-col dual viol
    rho, gam, tag = targets(Ry, Rx)
    Aj = jnp.asarray(A)
    mv = lambda v: Aj @ v
    rmv = lambda w: Aj.T @ w
    tau2, sig2, nrm2 = prescribed_smfe(mv, rmv, n, mA, rho, gam)
    c2 = fam().compile(equilibrate=("given", tau2, sig2), tol=TINY_TOL, max_blocks=B)
    s2 = c2.solve_instance(data, warm_start=(jnp.asarray(x1), jnp.asarray(y1)))
    return rel(np.asarray(sb.x)), rel(np.asarray(s2.x)), tag


# ====================== B: SOCP knife edges ==============================
def socp_ab(sz, seed, B=3000, mu=1e-2):
    import cvxpy as cp
    r = np.random.default_rng(seed); n = sz
    c = r.normal(size=n).astype("f"); A = r.normal(size=(n, n)).astype("f")
    Ksoc = np.vstack([np.zeros((1, n), "f"), A])
    e1 = np.concatenate([[1.0], np.zeros(n)]).astype("f")
    sq = np.sqrt(mu) * np.ones(n, "f")

    def fam():
        x = ocs.Variable(n)
        return ocs.minimize([ocs.sum_squares(ocs.Diagonal(sq) @ x),
                             ocs.soc(ocs.Dense(Ksoc) @ x + e1)],
                            cost=ocs.Parameter("c"))
    data = {"c": c}
    f = lambda z: float(c @ z) + 0.5 * mu * float(np.sum(z * z))
    xv = cp.Variable(n)
    cp.Problem(cp.Minimize(c @ xv + 0.5 * mu * cp.sum_squares(xv)),
               [cp.norm(A @ xv, 2) <= 1]).solve(solver=cp.CLARABEL)
    ref = f(xv.value)
    rel = lambda xo: abs(f(xo) - ref) / (1 + abs(ref))

    base = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=2 * B,
                         adaptive_omega=False)
    sb = base.solve_instance(data)

    c1 = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=B,
                       adaptive_omega=False)
    s1 = c1.solve_instance(data)
    x1, y1 = np.asarray(s1.x), np.asarray(s1.y)
    # SOC rows: distance to the cone, per coordinate
    z = Ksoc @ x1 + e1
    nz = np.linalg.norm(z[1:])
    if nz <= z[0]:
        proj = z
    elif nz <= -z[0]:
        proj = np.zeros_like(z)
    else:
        a = 0.5 * (z[0] + nz)
        proj = np.concatenate([[a], a * z[1:] / (nz + 1e-30)]).astype("f")
    Ry = np.concatenate([np.zeros(n, "f"),                       # sumsq rows: free
                         np.abs(z - proj)]).astype("f")
    Rx = np.abs(c + sq * y1[:n] + Ksoc.T @ y1[n:]).astype("f")
    rho, gam, tag = targets(Ry, Rx)
    mv = lambda v: jnp.concatenate([jnp.asarray(sq) * v, jnp.asarray(Ksoc) @ v])
    rmv = lambda w: jnp.asarray(sq) * w[:n] + jnp.asarray(Ksoc).T @ w[n:]
    spans = [(n, n + n + 1)]                                     # SOC block is scalar-class
    tau2, sig2, _ = prescribed_smfe(mv, rmv, n, 2 * n + 1, rho, gam,
                                    scalar_spans=spans)
    c2 = fam().compile(equilibrate=("given", tau2, sig2), tol=TINY_TOL,
                       max_blocks=B, adaptive_omega=False)
    s2 = c2.solve_instance(data, warm_start=(jnp.asarray(x1), jnp.asarray(y1)))
    return rel(np.asarray(sb.x)), rel(np.asarray(s2.x)), tag


# ====================== C: MoE E=256 dead rows ===========================
def moe_ab(N=1024, E=256, eps=0.3, B=1600):
    from examples.moe_router import build_router, make_logits
    C = make_logits(jax.random.PRNGKey(0), N, E)
    n = N * E

    comp_b, dof, mrg = build_router(N, E, eps=eps, tol=TINY_TOL, max_blocks=2 * B)
    sb = comp_b.solve_instance(dof(C))

    comp1, dof1, _ = build_router(N, E, eps=eps, tol=TINY_TOL, max_blocks=B)
    s1 = comp1.solve_instance(dof1(C))
    x1, y1 = np.asarray(s1.x), np.asarray(s1.y)
    X1 = x1.reshape(N, E)
    Kx = np.concatenate([X1.sum(1), X1.sum(0)])
    # nonneg is the primal g-block: dual stack = [marginals (N+E); ridge (n)]
    Ry = np.concatenate([np.abs(Kx - mrg),
                         np.zeros(n, "f")]).astype("f")
    ym, yr = y1[:N + E], y1[N + E:]
    MargT = (ym[:N, None] + ym[None, N:]).reshape(-1)
    g = -C.reshape(-1) + MargT + np.sqrt(eps) * yr
    Rx = np.abs(x1 - np.maximum(x1 - g, 0.0)).astype("f")
    rho, gam, tag = targets(Ry, Rx)
    K = ocs.MarginalOperator(N, E)
    se = np.sqrt(eps)
    mv = lambda v: jnp.concatenate([K.matvec(v), se * v])
    rmv = lambda w: K.rmatvec(w[:N + E]) + se * w[N + E:]
    tau2, sig2, _ = prescribed_smfe(mv, rmv, n, n + N + E, rho, gam)
    comp2, dof2, _ = build_router(N, E, eps=eps, tol=TINY_TOL, max_blocks=B)
    # recompile with given steps: rebuild the same family through compile
    import ocs as _o
    xv = _o.Variable(n); cP = _o.Parameter("c"); mP = _o.Parameter("m")
    sev = jnp.full(n, np.sqrt(eps), jnp.float32)
    fam = _o.minimize([_o.nonneg(xv), _o.equality(K @ xv - mP),
                       _o.sum_squares(_o.Diagonal(sev) @ xv)], cost=cP)
    comp2 = fam.compile(equilibrate=("given", tau2, sig2), tol=TINY_TOL,
                        max_blocks=B)
    s2 = comp2.solve_instance(dof(C), warm_start=(jnp.asarray(x1), jnp.asarray(y1)))

    def stats(x):
        X = np.asarray(x).reshape(N, E)
        rowsum = X.sum(1)
        dead = int((rowsum < 0.5).sum())
        rerr = float(np.abs(rowsum - 1.0).max())
        G = np.maximum(X, 0.0); G = G / (G.sum(1, keepdims=True) + 1e-9)
        load = G.sum(0)
        return dead, rerr, float(load.max() / load.mean())
    return stats(sb.x), stats(s2.x), tag


if __name__ == "__main__":
    print("== A: ill-scaled LPs (obj rel err vs HiGHS; equal 2B-block budget) ==")
    for k in (1, 2, 3):
        for sd in range(5):
            try:
                rb, rt, tag = lp_ab(k, 10 + sd)
                win = "TREAT" if rt < rb / 1.5 else ("base " if rb < rt / 1.5 else "  ~  ")
                print(f"  k={k} seed={10+sd}: base {rb:9.2e}  treat {rt:9.2e}  [{win}] ({tag})")
            except Exception as ex:
                print(f"  k={k} seed={10+sd}: ERR {type(ex).__name__}: {ex}")

    print("\n== B: SOCP knife edges ==")
    for sz, sd in ((12, 3), (24, 0), (24, 4), (48, 5)):
        try:
            rb, rt, tag = socp_ab(sz, sd)
            win = "TREAT" if rt < rb / 1.5 else ("base " if rb < rt / 1.5 else "  ~  ")
            print(f"  sz={sz} seed={sd}: base {rb:9.2e}  treat {rt:9.2e}  [{win}] ({tag})")
        except Exception as ex:
            print(f"  sz={sz} seed={sd}: ERR {type(ex).__name__}: {ex}")

    print("\n== C: MoE E=256 (dead rows / raw row-err / balance; eps=0.3) ==")
    try:
        (db, rb, bb), (dt, rt, bt), tag = moe_ab()
        print(f"  baseline : dead={db:4d}  row_err={rb:.2f}  balance={bb:.3f}")
        print(f"  treatment: dead={dt:4d}  row_err={rt:.2f}  balance={bt:.3f}  ({tag})")
    except Exception as ex:
        print(f"  ERR {type(ex).__name__}: {ex}")
