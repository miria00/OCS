"""Exact a posteriori certificate on optimal transport, matrix-free.

OCS's reported `gap` is a LEGACY Fenchel surrogate, not a duality gap and not a
bound: its dual value can sit above the primal value at the same point.  On the
transportation family both defects are repairable in closed form, giving an
UNCONDITIONAL exact certificate:

  primal: round the plan to the exact marginals (Altschuler-Weed-Rigollet
          rounding: scale rows, then columns, then spread the residual mass) --
          operations on the VARIABLE only, the operator is untouched;
  dual  : shift the potentials to exact dual feasibility,
          v_j <- min_i (C_ij - u_i)  (a data operation: C is the cost data);
  then  cost(rounded plan) - dualvalue(shifted potentials)  brackets the
        optimum from both sides: a true certified gap, no assumption.

Proposition p-network supplies the companion distance bound: the rounding
moves the plan by at most n * ||marginal violation||_inf.

Run:  PYTHONPATH=. python examples/ot_exact_cert.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp

import ocs

N = 64
EPS = 0.1
rng = np.random.default_rng(0)

K = ocs.MarginalOperator(N, N)
x = ocs.Variable(N * N)
prob = ocs.minimize([ocs.nonneg(x),
                     ocs.equality(K @ x - ocs.Parameter("m")),
                     ocs.sum_squares(ocs.Diagonal(
                         jnp.full(N * N, np.sqrt(EPS), jnp.float32)) @ x)],
                    cost=ocs.Parameter("c"))
comp = prob.compile(max_blocks=4000, tol=1e-6)

C = rng.uniform(0, 1, (N, N)).astype("f")
p = rng.uniform(0.2, 1, N).astype("f"); p /= p.sum()
q = rng.uniform(0.2, 1, N).astype("f"); q /= q.sum()
data = {"c": C.reshape(-1), "m": np.concatenate([p, q]).astype("f")}
sol = comp.solve_instance(data)

X = np.maximum(np.asarray(sol.x).reshape(N, N), 0.0)
viol0 = max(np.abs(X.sum(1) - p).max(), np.abs(X.sum(0) - q).max())

# ---- AWR rounding to exact marginals (variable-side only) ----------------
r_scale = np.minimum(1.0, p / (X.sum(1) + 1e-30))
X1 = X * r_scale[:, None]
c_scale = np.minimum(1.0, q / (X1.sum(0) + 1e-30))
X2 = X1 * c_scale[None, :]
dp = p - X2.sum(1); dq = q - X2.sum(0)              # both nonnegative
X_feas = X2 + np.outer(dp, dq) / max(dp.sum(), 1e-30)
assert np.abs(X_feas.sum(1) - p).max() < 1e-6
assert np.abs(X_feas.sum(0) - q).max() < 1e-6

def primal_val(Xm):
    return float((C * Xm).sum() + 0.5 * EPS * (Xm ** 2).sum())

# ---- dual: exact feasibility by shifting the potentials ------------------
# For the LP part (dropping the small ridge for a valid LP lower bound on the
# ridge-free objective; the ridge only ADDS to the primal, keeping the bracket
# valid): u_i + v_j <= C_ij. Take u from the solver's equality duals, then
# v_j = min_i (C_ij - u_i) is the tightest feasible completion.
# dual stack: nonneg is the primal g-block, so y = [marginals (2N); ridge].
y = np.asarray(sol.y)
best = None
for sgn in (+1.0, -1.0):
    u = sgn * y[:N].astype("f")
    # c-transform sweeps: alternating tightest completions are monotone dual
    # coordinate ascent over FEASIBLE potentials (each keeps u_i + v_j <= C_ij)
    for _ in range(5):
        v = (C - u[:, None]).min(axis=0)
        u = (C - v[None, :]).min(axis=1)
    v = (C - u[:, None]).min(axis=0)
    val = float(p @ u + q @ v)
    if best is None or val > best[0]:
        best = (val, u, v)
lower, u, v = best                                  # valid LP lower bound
upper_lp = float((C * X_feas).sum())                # feasible LP value

print(f"OT N={N}: solver residual {float(sol.residual):.1e}, "
      f"marginal violation before rounding {viol0:.2e}")
print(f"  rounded plan: exact marginals; plan moved "
      f"{np.abs(X_feas - X).max():.2e} (max entry)")
print(f"  CERTIFIED bracket (LP part):  {lower:.6f}  <=  opt  <=  {upper_lp:.6f}")
print(f"  certified exact gap: {upper_lp - lower:.2e}  "
      f"(unconditional: feasible primal vs feasible dual)")
cert = comp.certificate(sol.x, sol.y, data)
print(f"  reported LEGACY surrogate gap (NOT a bound): {cert['gap']:.2e}   "
      f"gap_exact flag: {bool(cert['gap_exact'])}   "
      f"surrogate dual above primal: {bool(cert['gap_surrogate_above_primal'])}")
print(f"  strict objective lower bound available: "
      f"{bool(cert['objective_lower_bound_available'])}   "
      f"({ocs.OBJECTIVE_BOUND_STATUS[int(cert['objective_bound_status'])]})")

# ---- refinement upgrades the unconditional certificate -------------------
xr, yr = comp.refine(sol.x, sol.y, data, steps=5, cg_iters=200)
Xr = np.maximum(np.asarray(xr).reshape(N, N), 0.0)
rs = np.minimum(1.0, p / (Xr.sum(1) + 1e-30)); X1r = Xr * rs[:, None]
cs = np.minimum(1.0, q / (X1r.sum(0) + 1e-30)); X2r = X1r * cs[None, :]
dpr = p - X2r.sum(1); dqr = q - X2r.sum(0)
Xf_r = X2r + np.outer(dpr, dqr) / max(dpr.sum(), 1e-30)
yr_np = np.asarray(yr)
best_r = None
for sgn in (+1.0, -1.0):
    ur = sgn * yr_np[:N].astype("f")
    for _ in range(5):
        vr = (C - ur[:, None]).min(axis=0)
        ur = (C - vr[None, :]).min(axis=1)
    vr = (C - ur[:, None]).min(axis=0)
    valr = float(p @ ur + q @ vr)
    if best_r is None or valr > best_r:
        best_r = valr
upper_r = float((C * Xf_r).sum())
print(f"  after refine: certified bracket  {best_r:.6f} <= opt <= {upper_r:.6f}"
      f"   exact gap {upper_r - best_r:.2e}")

# ---- certify the RIDGE problem itself (removes the LP/ridge mismatch) ----
# Fenchel dual of  min <C,X> + (eps/2)||X||^2  s.t. X1=p, X^T1=q, X>=0:
#   D(u,v) = p'u + q'v - (1/2 eps) sum_ij [(u_i + v_j - C_ij)_+]^2,
# valid for ANY (u,v) -- no dual feasibility constraint. Bracket against the
# feasible rounded plan's ridge objective.
def ridge_lower(u, v):
    slack = np.maximum(u[:, None] + v[None, :] - C, 0.0)
    return float(p @ u + q @ v - (slack ** 2).sum() / (2 * EPS))
best_ridge = None
for yy in (np.asarray(sol.y), np.asarray(yr)):
    for sgn in (+1.0, -1.0):
        uu = sgn * yy[:N].astype("f"); vv = sgn * yy[N:2 * N].astype("f")
        val = ridge_lower(uu, vv)
        if best_ridge is None or val > best_ridge:
            best_ridge = val
upper_ridge = primal_val(Xf_r)
print(f"  RIDGE-problem certificate: {best_ridge:.6f} <= opt <= "
      f"{upper_ridge:.6f}   exact gap {upper_ridge - best_ridge:.2e}  "
      f"(any-(u,v)-valid dual; unconditional)")

# ---- polish the dual by ascent on the smooth concave ridge dual ----------
# grad_u D = p - X(u,v) 1,  grad_v D = q - X(u,v)^T 1,
# with X(u,v) = (u \oplus v - C)_+ / eps: plain data-side gradient ascent.
uu = np.asarray(sol.y)[:N].astype("f").copy()
vv = np.asarray(sol.y)[N:2 * N].astype("f").copy()
if ridge_lower(-uu, -vv) > ridge_lower(uu, vv):
    uu, vv = -uu, -vv
step = EPS / N
for _ in range(3000):
    Xuv = np.maximum(uu[:, None] + vv[None, :] - C, 0.0) / EPS
    uu = uu + step * (p - Xuv.sum(1))
    vv = vv + step * (q - Xuv.sum(0))
lower_polished = ridge_lower(uu, vv)
print(f"  + 3000 dual-ascent steps (data-side): {lower_polished:.6f} <= opt <= "
      f"{upper_ridge:.6f}   exact gap {upper_ridge - lower_polished:.2e}")
