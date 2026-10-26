"""Two invariance checks on the termination criterion.

(A) ALGORITHM INVARIANCE.  The stopping residual is defined from the optimality
    conditions (a prox fixed point), not from iterate progress, so it is a
    property of the POINT.  We verify this the only way that means anything:
    take the same least-squares family, solve it two different ways (the
    restarted-PDHG engine and the matrix-free CGLS path), and evaluate BOTH
    certificates at BOTH points.  A method-independent criterion must rank the
    two points identically regardless of which formula reports it.

(B) SCALE INVARIANCE.  The solver itself rescales the problem (equilibration is
    a D K E transformation), so the tolerance is only meaningful if it survives
    such a rescale.  We solve the same LP under objective scaling, right-hand-
    side scaling, and a random row/column rescale of the operator, each to the
    same reported tolerance, and compare the RELATIVE OBJECTIVE ERROR actually
    achieved.  If the criterion is scale invariant, a fixed reported residual
    buys the same true accuracy in every variant.

Run:  PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python examples/termination_invariance.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax.numpy as jnp

import ocs

# ===================== (A) algorithm invariance ==========================
print("== (A) algorithm invariance: same point, two solvers, two formulas ==")
rng = np.random.default_rng(0)
m, n = 120, 60
A = rng.normal(size=(m, n)).astype("f")
b = (A @ rng.normal(size=n)).astype("f") + 0.05 * rng.normal(size=m).astype("f")


def ls_family():
    x = ocs.Variable(n)
    return ocs.minimize([ocs.least_squares(ocs.Dense(A) @ x - ocs.Parameter("b"))])


data = {"b": b}
pdhg = ls_family().compile(method="pdhg", tol=1e-8, max_blocks=4000)
obj = lambda z: 0.5 * float(np.sum((A @ np.asarray(z) - b) ** 2))

# a path of increasingly converged points, produced by CGLS
pts = []
for it in (5, 20, 80, 300):
    cg = ls_family().compile(method="cg", cg_iters=it, rho=0.0, tol=1e-12)
    s = cg.solve_instance(data)
    pts.append((f"cgls({it})", np.asarray(s.x), cg))
sp = pdhg.solve_instance(data)
pts.append(("pdhg", np.asarray(sp.x), None))

opt = min(obj(x) for _, x, _ in pts)
print(f"   {'point':>12} | {'PDHG cert':>11} {'CGLS cert':>11} {'obj - opt':>11}")
for name, x, cg in pts:
    y = jnp.asarray(A @ x - b)                 # the implied dual for 1/2||.||^2
    cp = pdhg.certificate(jnp.asarray(x), y, data)
    cpv = max(cp["r_primal"], cp["r_dual"])
    ref = cg if cg is not None else pts[0][2]
    cgv = ref.certificate(jnp.asarray(x), None, data)["r_primal"]
    print(f"   {name:>12} | {cpv:11.2e} {cgv:11.2e} {obj(x) - opt:11.2e}")
print("   (both formulas must order the points identically: the criterion is a"
      "\n    property of the point, not of the method that produced it)")

# ===================== (B) scale invariance ==============================
print("\n== (B) scale invariance: same tolerance, four rescalings of one LP ==")
from scipy.optimize import linprog

r2 = np.random.default_rng(1)
mA, nn = 40, 80
A0 = r2.normal(size=(mA, nn)).astype("f")
xt = np.abs(r2.normal(size=nn)).astype("f")
b0 = A0 @ xt
c0 = np.abs(r2.normal(size=nn)).astype("f")


def solve_lp(A, b, c, tol=1e-4):
    x = ocs.Variable(A.shape[1])
    comp = ocs.minimize([ocs.nonneg(x),
                         ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                        cost=ocs.Parameter("c")).compile(tol=tol, max_blocks=20000)
    s = comp.solve_instance({"b": b, "c": c})
    ref = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun
    xo = np.asarray(s.x)
    rel = abs(float(c @ xo) - ref) / (1 + abs(ref))
    return float(s.residual), rel, float(np.linalg.norm(xo))


variants = [
    ("baseline",              A0,                         b0,        c0),
    ("objective x 1e3",       A0,                         b0,        c0 * 1e3),
    ("rhs x 1e3",             A0,                         b0 * 1e3,  c0),
    ("row/col rescale",       None,                       None,      None),
]
D = 10.0 ** r2.uniform(-2, 2, mA).astype("f")
E = 10.0 ** r2.uniform(-2, 2, nn).astype("f")
variants[3] = ("row/col rescale", (A0 * E[None, :]) * D[:, None], b0 * D, c0 * E)

print(f"   {'variant':>18} | {'reported resid':>14} {'true rel obj err':>16} {'||x||':>10}")
for name, A, b, c in variants:
    res, rel, nx = solve_lp(A, b, c)
    print(f"   {name:>18} | {res:14.2e} {rel:16.2e} {nx:10.2e}")
print("   (a scale-invariant criterion buys the same true accuracy in every row)")
