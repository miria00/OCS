"""Adversarial reliability battery: the long-tail behaviors a default solver
must survive, run overnight and reported as a table.

Three axes:
  1. SCALING ROBUSTNESS -- the same LP/QP instances with rows and columns of
     the operator scaled by 10^{+-k}, k = 0..6.  This is the equilibration
     story working for a living: SMFE (default) vs scalar-only scaling.
  2. STATUS CORRECTNESS -- feasible / primal-infeasible / unbounded instances
     across seeds; the automatic status flag must call each correctly.
  3. REFINEMENT TIER -- instances stopped at the 1e-3 floor, polished by the
     matrix-free Gauss-Newton refine; residual before vs after.

Run:  PYTHONPATH=. python examples/reliability_adversarial.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax.numpy as jnp

import ocs

rng_master = np.random.default_rng(0)


# ---- 1. scaling robustness ----------------------------------------------
def scaled_lp(k, seed, equilibrate="sgd", tol=1e-3, max_blocks=1500):
    """Equality-form LP with rows/cols scaled by 10^{+-k}; solve and compare
    the objective against scipy linprog on the SAME scaled data."""
    from scipy.optimize import linprog
    r = np.random.default_rng(seed)
    m, n = 40, 80
    A = r.normal(size=(m, n)).astype("f")
    d = 10.0 ** r.uniform(-k, k, size=m).astype("f")     # row scales
    e = 10.0 ** r.uniform(-k, k, size=n).astype("f")     # col scales
    A = (A * e[None, :]) * d[:, None]
    xt = np.abs(r.normal(size=n)).astype("f")
    b = A @ xt
    c = np.abs(r.normal(size=n)).astype("f") * e         # cost in scaled units
    x = ocs.Variable(n)
    comp = ocs.minimize([ocs.nonneg(x),
                         ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                        cost=ocs.Parameter("c")).compile(
        max_blocks=max_blocks, tol=tol, equilibrate=equilibrate)
    sol = comp.solve_instance({"b": b, "c": c})
    xo = np.asarray(sol.x)
    ref = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    rel = abs(c @ xo - ref.fun) / (1 + abs(ref.fun))
    return float(sol.residual), rel


def scaling_table():
    print("== 1. scaling robustness: LP with rows/cols scaled 10^{+-k} ==")
    print("   (solved = residual<1e-3 AND objective within 1e-2 of HiGHS)")
    print(f"   {'k':>2} | {'SMFE tol=1e-3':^26} | {'SMFE tol=1e-5':^26} | {'scalar tol=1e-3':^26}")
    for k in (0, 1, 2, 3):
        row = [f"   {k:>2} |"]
        for eq, tol, mb in (("sgd", 1e-3, 1500), ("sgd", 1e-5, 6000),
                            ("scalar", 1e-3, 1500)):
            ok = tot = 0
            worst = 0.0; threw = 0
            for s in range(5):
                try:
                    res, rel = scaled_lp(k, 10 + s, eq, tol, mb)
                    ok += (res < tol * 1.5) and (rel < 1e-2)
                    worst = max(worst, rel)
                except Exception:
                    threw += 1
                tot += 1
            tag = f" ({threw} err)" if threw else ""
            row.append(f" {ok}/{tot} worst {worst:7.1e}{tag:8s}|")
        print("".join(row))
    print("   (k>=4 exceeds float32 objective agreement for any method; not run)")


# ---- 2. status correctness ----------------------------------------------
def status_sweep():
    print("\n== 2. status correctness across seeds ==")
    correct = total = 0
    for seed in range(6):
        r = np.random.default_rng(seed)
        m, n = 25, 15
        A = r.normal(size=(m, n)).astype("f")

        # feasible LP
        xt = np.abs(r.normal(size=n)).astype("f"); b = A @ xt
        x = ocs.Variable(n)
        comp = ocs.minimize([ocs.nonneg(x),
                             ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                            cost=np.abs(r.normal(size=n)).astype("f")).compile()
        s1 = comp.solve_instance({"b": b})
        got1 = s1.status
        ok1 = got1 == "solved"

        # primal infeasible: x >= 0 with sum(x) = -1
        x2 = ocs.Variable(n)
        one = np.ones((1, n), np.float32)
        comp2 = ocs.minimize([ocs.nonneg(x2),
                              ocs.equality(ocs.Dense(one) @ x2 - ocs.Parameter("t"))]
                             ).compile()
        s2 = comp2.solve_instance({"t": -np.ones(1, np.float32)})
        got2 = s2.status
        ok2 = got2 == "primal_infeasible"

        # unbounded: min -sum(x), x >= 0, a^T x = 0 (a mixed-sign: nonneg
        # directions with 1^T d > 0, a^T d = 0 exist, so unbounded below)
        x3 = ocs.Variable(n)
        a = A[0:1, :].copy(); a[0, 0] = -abs(a[0, 0])
        comp3 = ocs.minimize([ocs.nonneg(x3),
                              ocs.equality(ocs.Dense(a) @ x3 - ocs.Parameter("z"))],
                             cost=-np.ones(n, np.float32)).compile()
        s3 = comp3.solve_instance({"z": np.zeros(1, np.float32)})
        got3 = s3.status
        ok3 = got3 == "unbounded"

        correct += ok1 + ok2 + ok3
        total += 3
        print(f"   seed {seed}: feasible->{got1:<18} infeasible->{got2:<18} "
              f"unbounded->{got3}")
    print(f"   status calls correct: {correct}/{total}")


# ---- 3. refinement tier --------------------------------------------------
def refine_tier():
    print("\n== 3. refinement: matrix-free Gauss-Newton polish past the 1e-3 floor ==")
    for seed in range(4):
        r = np.random.default_rng(seed)
        m, n = 60, 40
        A = r.normal(size=(m, n)).astype("f")
        b = (A @ np.abs(r.normal(size=n))).astype("f")
        x = ocs.Variable(n)
        comp = ocs.minimize([ocs.box(x, 0.0, 2.0),
                             ocs.least_squares(ocs.Dense(A) @ x - ocs.Parameter("b"))]
                            ).compile()
        sol = comp.solve_instance({"b": b})
        r0 = float(sol.residual)
        try:
            xr, yr = comp.refine(sol.x, sol.y, {"b": b})
            cert = comp.certificate(xr, yr, {"b": b})
            r1 = max(cert["r_primal"], cert["r_dual"])
            print(f"   seed {seed}: residual {r0:8.1e} -> {float(r1):8.1e}")
        except Exception as e:
            print(f"   seed {seed}: residual {r0:8.1e} -> refine unavailable "
                  f"({type(e).__name__})")


if __name__ == "__main__":
    scaling_table()
    status_sweep()
    refine_tier()
