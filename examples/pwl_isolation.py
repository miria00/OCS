"""Is the piecewise-linear block itself the problem, or only PWL coupled to
equality constraints?

Stephen's requirement is a PWL utility ("linear with a max value"), so this
block has to work.  The multicommodity example plateaued at a primal residual of
2.5e-3 and its recorded ablation showed that removing the PWL term, keeping
conservation, nonnegativity and the ridge, dropped r_p to 6.2e-7.  That pins the
blame on PWL but does not say whether PWL alone is broken.

This walks up the structure one term at a time, checking each against CVXPY on
the identical problem:

  A  PWL alone                       min sum_i phi(x_i - b_i) + ridge
  B  PWL through an operator         min sum_i phi((Kx - b)_i) + ridge
  C  PWL + nonnegativity             ... with x >= 0
  D  PWL + equality                  ... with Nx = d
  E  PWL + nonneg + equality         the multicommodity structure, small

Where the residual first blows up is the answer.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
import ocs

SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
_I = [0.0]
for _j, _b in enumerate([0.5, 1.0, 1.5]):
    _I.append(_I[-1] + (SLOPES[_j] - SLOPES[_j + 1]) * _b)
INTER = np.array(_I, np.float32)
MU = 1e-3
BLK = 100
MB = int(os.environ.get("PWL_MAX_BLOCKS", 400))
TOL = 1e-8


def phi(y):
    y = np.asarray(y, np.float64)
    return np.max(SLOPES[None, :].astype(np.float64) * y[:, None]
                  + INTER[None, :].astype(np.float64), axis=1)


def cvxpy_ref(n, K, b, N, d, nonneg, equality):
    import cvxpy as cp
    x = cp.Variable(n)
    y = (K @ x - b) if K is not None else (x - b)
    cost = cp.sum(cp.maximum(*[float(SLOPES[j]) * y + float(INTER[j])
                               for j in range(len(SLOPES))]))
    cons = []
    if nonneg:
        cons.append(x >= 0)
    if equality:
        cons.append(N @ x == d)
    p = cp.Problem(cp.Minimize(cost + 0.5 * MU * cp.sum_squares(x)), cons)
    p.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    return float(p.value), (np.asarray(x.value) if x.value is not None else None), p.status


def ocs_solve(n, Kmat, b, N, d, nonneg, equality):
    x = ocs.Variable(n)
    bp = ocs.Parameter("b")
    terms = [ocs.sum_squares(ocs.Diagonal(np.sqrt(MU) * np.ones(n, "f")) @ x)]
    op = ocs.Dense(Kmat) if Kmat is not None else ocs.Identity(n)
    terms.append(ocs.pwl(op @ x - bp, SLOPES, INTER))
    if nonneg:
        terms.append(ocs.nonneg(x))
    data = {"b": jnp.asarray(b.astype("f"))}
    if equality:
        dp = ocs.Parameter("d")
        terms.append(ocs.equality(ocs.Dense(N) @ x - dp))
        data["d"] = jnp.asarray(d.astype("f"))
    comp = ocs.minimize(terms).compile(block=BLK, max_blocks=MB, tol=TOL,
                                       equilibrate="sgd")
    s = comp.solve_instance(data); jax.block_until_ready(s.x)
    t0 = time.perf_counter()
    s = comp.solve_instance(data); jax.block_until_ready(s.x)
    return time.perf_counter() - t0, np.asarray(s.x), float(s.residual), s.status, s.iters


def true_obj(x, Kmat, b, n):
    y = (Kmat @ x - b) if Kmat is not None else (x - b)
    return float(phi(y).sum() + 0.5 * MU * float(np.asarray(x) @ np.asarray(x)))


if __name__ == "__main__":
    n, m, v = 200, 150, 40
    rng = np.random.default_rng(0)
    Kmat = (rng.normal(size=(m, n)) / np.sqrt(n)).astype("f")
    N = np.zeros((v, n), np.float32)
    for e in range(n):
        a, c = rng.choice(v, 2, replace=False)
        N[a, e] = 1.0; N[c, e] = -1.0
    xf = rng.uniform(0, 0.6, n).astype("f")
    d = (N @ xf).astype("f")

    cases = [
        ("A  PWL alone",              None,  False, False),
        ("B  PWL + operator",         Kmat,  False, False),
        ("C  PWL + nonneg",           Kmat,  True,  False),
        ("D  PWL + equality",         Kmat,  False, True),
        ("E  PWL + nonneg + equality", Kmat, True,  True),
    ]
    print(f"n={n}, operator {m}x{n}, {v} equality rows, tol={TOL:g}, "
          f"budget={BLK*MB} iterations\n")
    print(f"{'case':28s} {'OCS resid':>11s} {'status':>12s} {'iters':>7s} "
          f"{'rel obj err':>12s}  {'ref':>10s}")
    for name, Km, nn, eq in cases:
        bb = (rng.normal(size=(m if Km is not None else n))).astype("f")
        try:
            t, xo, res, st, it = ocs_solve(n, Km, bb, N, d, nn, eq)
            ref, xr, rst = cvxpy_ref(n, Km, bb.astype(np.float64), N.astype(np.float64),
                                     d.astype(np.float64), nn, eq)
            jo = true_obj(xo.astype(np.float64), Km.astype(np.float64) if Km is not None else None,
                          bb.astype(np.float64), n)
            rel = abs(jo - ref) / max(1.0, abs(ref))
            flag = "" if (res < 1e-5 and rel < 1e-4) else "   <-- FAILS"
            print(f"{name:28s} {res:11.3e} {st:>12s} {it:7d} {rel:12.3e}  {rst:>10s}{flag}")
        except Exception as exc:
            print(f"{name:28s} EXCEPTION {type(exc).__name__}: {str(exc)[:60]}")
