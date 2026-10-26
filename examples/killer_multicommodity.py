"""Killer example: multicommodity flow with PIECEWISE-LINEAR convex costs.

    minimize_F   sum_a  phi( (sum_k f_k)_a )
    subject to   N f_k = d_k        (per-commodity flow conservation)
                 f_k >= 0,

with F = [f_1 ... f_K] in R^{E x K} (E arcs, K commodities) and phi a convex,
increasing, piecewise-linear congestion cost on each arc's total flow (equivalently,
maximizing a concave piecewise-linear utility of spare capacity -- the canonical
network-utility / traffic-equilibrium problem).  The variable is a MATRIX: any
matrix-forming solver expands F to a giant vector and forms a
(V*K + ...) x (E*K) constraint matrix.  OCS applies the block-diagonal incidence
(MultiCommodityFlow) and the capacity coupling (CommoditySum) matrix-free, and the
PWL cost via the PiecewiseLinearConvex prox block.  Validated against CVXPY.

Run:  python examples/killer_multicommodity.py
"""
import numpy as np
import scipy.sparse as _sp
# newer scipy dropped the deprecated .A attribute that older CVXPY still uses
for _c in (_sp.csc_matrix, _sp.csr_matrix, _sp.coo_matrix):
    if not hasattr(_c, "A"):
        _c.A = property(lambda self: self.toarray())
import jax.numpy as jnp
import ocs

# convex, increasing, piecewise-linear congestion cost phi(y) = max_j(s_j y + r_j),
# through the origin with increasing marginal cost (slopes 0.5 -> 4).
SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
BREAKS = np.array([0.5, 1.0, 1.5], np.float32)
INTER = [0.0]
for j in range(len(BREAKS)):
    INTER.append(INTER[-1] + (SLOPES[j] - SLOPES[j + 1]) * BREAKS[j])       # continuity
INTER = np.array(INTER, np.float32)
phi = lambda y: np.max(SLOPES[None, :] * np.asarray(y)[:, None] + INTER[None, :], axis=1)

# Paper scale: V=1000 nodes, E=2e4 arcs, K=1000 commodities -> 2e7 flow vars.
# OCS_MCF_SMALL=1 gives the quick smoke size (and a CVXPY cross-check).
import os as _os
if _os.environ.get("OCS_MCF_SMALL"):
    V, E, K, mu = 12, 40, 4, 1e-3
else:
    V, E, K, mu = 1000, 20000, 1000, 1e-3
r = np.random.default_rng(1)
N = np.zeros((V, E), np.float32)
for e in range(E):
    a, b = r.choice(V, 2, replace=False); N[a, e] = 1.0; N[b, e] = -1.0
Ftrue = r.uniform(0.0, 0.6, (E, K)).astype("f")
d = (N @ Ftrue).reshape(-1)                                # feasible demand d_k = N f_k^true

x = ocs.Variable(E * K)
Ncons = ocs.MultiCommodityFlow(N, K)                       # x -> stacked N f_k
Csum = ocs.CommoditySum(E, K)                              # x -> sum_k f_k (arc totals)
dp = ocs.Parameter("d")
prob = ocs.minimize([
    ocs.nonneg(x),                                         # f >= 0                    (g)
    ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(E * K, "f")) @ x),  # small ridge
    ocs.equality(Ncons @ x - dp),                          # N f_k = d_k
    ocs.pwl(Csum @ x, SLOPES, INTER),                      # PWL congestion on arc totals
])
comp = prob.compile(max_blocks=int(_os.environ.get('OCS_MCF_BLOCKS', 300)), tol=1e-4)
print(comp, f"  ({E} arcs x {K} commodities = {E*K} flow vars)")

sol = comp.solve_instance({"d": d})
F = np.asarray(sol.x).reshape(E, K)
obj = float(phi(F.sum(1)).sum())
print(f"residual={sol.residual:.1e}  congestion objective={obj:.4f}  "
      f"max conservation viol={np.max(np.abs(N @ F - d.reshape(V,K))):.1e}")

try:
    import cvxpy as cp
    Fv = cp.Variable((E, K), nonneg=True)
    y = cp.sum(Fv, axis=1)
    cong = cp.sum(cp.maximum(*[SLOPES[j] * y + INTER[j] for j in range(len(SLOPES))]))
    cons = [N @ Fv[:, k] == d.reshape(V, K)[:, k] for k in range(K)]
    # SCS (not Clarabel): the incidence rows are rank-deficient (sum to 0), so the
    # float32 demand is only consistent to ~1e-7 -- SCS tolerates it, Clarabel rejects.
    cp.Problem(cp.Minimize(cong + 0.5 * mu * cp.sum_squares(Fv)), cons).solve(solver=cp.SCS)
    oo = obj + 0.5 * mu * float(np.sum(F ** 2))
    ro = float(cong.value) + 0.5 * mu * float(np.sum(Fv.value ** 2))
    print(f"OCS obj (with ridge)={oo:.4f}  vs CVXPY={ro:.4f}  rel-err={abs(oo-ro)/(1+abs(ro)):.2e}")
except ImportError:
    print("(install cvxpy for the reference comparison)")
