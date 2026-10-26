"""Does the achievable residual floor scale like eps * ||x|| on every family,
or was that a piecewise-linear coincidence?

The PWL study showed the smallest attainable residual tracking
eps32 * ||x||_inf across three orders of magnitude.  Before any solver-wide
status rule is built on that, it has to hold on more than one family.

Each family is solved with an unreachable tolerance and a large budget, so the
solver always returns its best point.  We report the residual it reaches, the
size of the solution, and the ratio to eps32 * ||x||_inf.  A predictor that
generalizes gives ratios of order one everywhere.
"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
import ocs

EPS32 = float(np.finfo(np.float32).eps)
BLK = 100
MB = int(os.environ.get("FLOOR_MAX_BLOCKS", 300))
TOL = 1e-12                      # deliberately unreachable


def solve(terms, data, cost=None):
    prob = ocs.minimize(terms) if cost is None else ocs.minimize(terms, cost=cost)
    comp = prob.compile(block=BLK, max_blocks=MB, tol=TOL, equilibrate="sgd")
    s = comp.solve_instance(data); jax.block_until_ready(s.x)
    return s


def f_pwl(scale):
    """Control: the family the predictor was derived from."""
    n = 200
    S = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
    I = [0.0]
    for j, b in enumerate([0.5, 1.0, 1.5]):
        I.append(I[-1] + (S[j] - S[j + 1]) * b)
    I = np.array(I, np.float32)
    mu = 1e-1 / scale                       # smaller mu inflates ||x||
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    b = np.random.default_rng(0).normal(size=n).astype("f")
    return solve([ocs.sum_squares(ocs.Diagonal(np.sqrt(mu) * np.ones(n, "f")) @ x),
                  ocs.pwl(ocs.Identity(n) @ x - bp, S, I)], {"b": jnp.asarray(b)})


def f_lasso(scale):
    n, m = 400, 200
    rng = np.random.default_rng(1)
    A = (rng.normal(size=(m, n)) / np.sqrt(m)).astype("f")
    b = (rng.normal(size=m) * scale).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    return solve([ocs.least_squares(ocs.Dense(A) @ x - bp),
                  ocs.l1(ocs.Identity(n) @ x, 0.01)], {"b": jnp.asarray(b)})


def f_imaging(scale):
    L = 32; n = L * L
    F = ocs.FFTMask.random(n, 0.35, seed=0); Gr = ocs.Gradient2D(L, L)
    y = (np.random.default_rng(0).normal(size=F.shape[0]) * scale).astype("f")
    x = ocs.Variable(n); yp = ocs.Parameter("y")
    return solve([ocs.least_squares(F @ x - yp), ocs.group_l2(Gr @ x, 0.03, 2)],
                 {"y": jnp.asarray(y)})


def f_ot(scale):
    N = 32; n = N * N
    K = ocs.MarginalOperator(N, N)
    rng = np.random.default_rng(0)
    C = (rng.uniform(0, 1, (N, N)) * scale).astype("f")
    p = rng.uniform(.2, 1, N).astype("f"); p /= p.sum()
    q = rng.uniform(.2, 1, N).astype("f"); q /= q.sum()
    se = jnp.full(n, np.sqrt(0.1), jnp.float32)
    x = ocs.Variable(n); c = ocs.Parameter("c"); m = ocs.Parameter("m")
    return solve([ocs.nonneg(x), ocs.equality(K @ x - m),
                  ocs.sum_squares(ocs.Diagonal(se) @ x)],
                 {"c": jnp.asarray(C.reshape(-1)),
                  "m": jnp.asarray(np.concatenate([p, q]))}, cost=c)


def f_nnls(scale):
    """Nonnegative least squares. Needs a genuine operator: the compiler
    rejects a model whose every term goes through the identity."""
    n, m = 300, 200
    rng = np.random.default_rng(2)
    A = (rng.normal(size=(m, n)) / np.sqrt(m)).astype("f")
    b = (rng.normal(size=m) * scale).astype("f")
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    return solve([ocs.nonneg(x), ocs.least_squares(ocs.Dense(A) @ x - bp)],
                 {"b": jnp.asarray(b)})


FAMILIES = [("pwl (control)", f_pwl), ("lasso", f_lasso),
            ("imaging fft+tv", f_imaging), ("optimal transport", f_ot),
            ("nnls", f_nnls)]

if __name__ == "__main__":
    print(f"eps32 = {EPS32:.3e}; tol requested {TOL:g} (unreachable), "
          f"budget {BLK*MB} iterations\n")
    print(f"{'family':20s} {'scale':>7s} {'|x|_inf':>10s} {'residual':>11s} "
          f"{'eps*|x|':>10s} {'ratio':>8s} {'status':>11s}")
    for name, fn in FAMILIES:
        for scale in (1.0, 10.0, 100.0):
            try:
                s = fn(scale)
                xm = float(np.abs(np.asarray(s.x)).max())
                pred = EPS32 * max(xm, 1.0)
                print(f"{name:20s} {scale:7.0f} {xm:10.3f} {s.residual:11.3e} "
                      f"{pred:10.3e} {s.residual/pred:8.2f} {s.status:>11s}")
            except Exception as exc:
                print(f"{name:20s} {scale:7.0f} EXCEPTION {type(exc).__name__}: "
                      f"{str(exc)[:45]}")
