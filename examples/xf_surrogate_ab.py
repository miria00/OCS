"""Seam B A/B: Xiong-Freund-surrogate re-equilibration vs plain SMFE.

Xiong & Freund (arXiv 2406.01942) prove the diagonal rescaling that improves
restarted-PDHG level-set geometry is D = sqrt(eta) X(eta) at a CENTRAL-PATH
point: column scales proportional to the iterate magnitudes (big steps for
coordinates far from their bounds), not to data row norms.  This experiment
tests the matvec-only surrogate: at a restart boundary, set the COLUMN
squared-norm targets proportional to the smoothed iterate magnitudes squared,
re-equilibrate with prescribed-marginal SMFE, and continue warm-started, with
a trial-chunk revert guard (the safeguard whose absence blew up the
residual-targeted variant).

Protocol (equal total budget 2B blocks):
  BASELINE   SMFE, 2B blocks.
  TREATMENT  SMFE, B blocks -> gamma_j ~ (|x_j| + delta)^2, rho uniform ->
             prescribed SMFE -> trial B/4 under the new metric; if the
             residual worsened by >1.2x, revert to the old metric for the
             remaining 3B/4, else continue under the new one.

Run:  PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python examples/xf_surrogate_ab.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax.numpy as jnp

import ocs
from ocs.solver.equilibrate_prescribed import prescribed_smfe

TINY_TOL = 1e-7


def lp_xf(k, seed, B=750):
    from scipy.optimize import linprog
    r = np.random.default_rng(seed)
    mA, n = 40, 80
    A = r.normal(size=(mA, n)).astype("f")
    d = 10.0 ** r.uniform(-k, k, size=mA).astype("f")
    e = 10.0 ** r.uniform(-k, k, size=n).astype("f")
    A = (A * e[None, :]) * d[:, None]
    b = A @ np.abs(r.normal(size=n)).astype("f")
    c = np.abs(r.normal(size=n)).astype("f") * e
    ref = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun

    def fam():
        x = ocs.Variable(n)
        return ocs.minimize([ocs.nonneg(x),
                             ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                            cost=ocs.Parameter("c"))
    data = {"b": b, "c": c}
    rel = lambda xo: abs(float(c @ xo) - ref) / (1 + abs(ref))

    base = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=2 * B)
    sb = base.solve_instance(data)

    c1 = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=B)
    s1 = c1.solve_instance(data)
    x1, y1 = np.asarray(s1.x), np.asarray(s1.y)

    # XF surrogate: column targets ~ smoothed iterate magnitudes squared
    xa = np.abs(x1)
    delta = 1e-2 * (xa.mean() + 1e-12)
    gam = ((xa + delta) ** 2).astype("f")
    rho = np.ones(mA, np.float32)
    Aj = jnp.asarray(A)
    tau2, sig2, _ = prescribed_smfe(lambda v: Aj @ v, lambda w: Aj.T @ w,
                                    n, mA, rho, gam)
    c2 = fam().compile(equilibrate=("given", tau2, sig2), tol=TINY_TOL,
                       max_blocks=B // 4)
    s2a = c2.solve_instance(data, warm_start=(jnp.asarray(x1), jnp.asarray(y1)))

    rest = B - B // 4
    if float(s2a.residual) > 1.2 * float(s1.residual):     # revert guard
        c3 = fam().compile(equilibrate="sgd", tol=TINY_TOL, max_blocks=rest)
        s2 = c3.solve_instance(data, warm_start=(jnp.asarray(x1), jnp.asarray(y1)))
        guard = "REVERTED"
    else:
        c3 = fam().compile(equilibrate=("given", tau2, sig2), tol=TINY_TOL,
                           max_blocks=rest)
        s2 = c3.solve_instance(data, warm_start=(s2a.x, s2a.y))
        guard = "kept"
    return rel(np.asarray(sb.x)), rel(np.asarray(s2.x)), guard


if __name__ == "__main__":
    print("== Xiong-Freund surrogate (column targets = iterate magnitudes) ==")
    print("   equal 2B budget; guard reverts a metric that worsens the residual")
    wins = losses = ties = 0
    for k in (0, 1, 2):
        for sd in range(5):
            try:
                rb, rt, guard = lp_xf(k, 10 + sd)
                tag = ("TREAT" if rt < rb / 1.5 else
                       ("base " if rb < rt / 1.5 else "  ~  "))
                wins += tag == "TREAT"; losses += tag == "base "
                ties += tag == "  ~  "
                print(f"  k={k} seed={10+sd}: base {rb:9.2e}  treat {rt:9.2e}  "
                      f"[{tag}] ({guard})")
            except Exception as ex:
                print(f"  k={k} seed={10+sd}: ERR {type(ex).__name__}: {ex}")
    print(f"\ntreatment wins {wins}, losses {losses}, ties {ties}")
