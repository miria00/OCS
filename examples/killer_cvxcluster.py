"""Killer example: cluster resource allocation (CvxCluster Stage-1 relaxation),
operator-native.  Complementary solver for the relaxation of Nnorom, Boyd & Levis;
NOT a competitor to their scheduler.

Stage-1 places task types on server shapes by a continuous relaxation:

    maximize_A  sum_ij p_j A_ij         (place high-priority work)
    s.t.        A >= 0
                A T <= S                 (per-shape resource capacity)
                sum_i A_ij <= n_j        (each task type placed at most n_j times)

A in R^{s x t} is a MATRIX variable (s shapes, t task types).  The only linear
maps are the capacity map A -> A T (a Kronecker/right-multiply by the FIXED
task-type demand profiles T in R^{t x r}) and the assignment map A -> sum over
shapes (column sums), both applied matrix-free -- the s*t-column constraint matrix
of the vectorized LP is never formed.  Because the cluster and the task-type
profiles T are fixed infrastructure while the priorities p, capacities S, and
counts n vary each scheduling round, the operator is shared and only the data
changes: the parametric regime (compile once, solve many rounds batched).

Run:  PYTHONPATH=/home/miria/OCSolver/OCS /home/miria/jaxenv2/bin/python examples/killer_cvxcluster.py
"""
import time
import numpy as np
import jax.numpy as jnp
import ocs

s, t, r = 8, 40, 3          # shapes, task types, resources


def cap_op(T):              # A -> -(A T), matrix-free (fixed T); minus for the <= form
    return ocs.LinearOperator.from_forward(
        lambda x: -(x.reshape(s, t) @ jnp.asarray(T)).reshape(-1), n=s * t, m=s * r)


def asgn_op():             # A -> -column sums (each task type's total placement)
    return ocs.LinearOperator.from_forward(
        lambda x: -x.reshape(s, t).sum(axis=0), n=s * t, m=t)


def build(T):
    x = ocs.Variable(s * t)
    negS = ocs.Parameter("negS"); negn = ocs.Parameter("negn"); c = ocs.Parameter("c")
    prob = ocs.minimize([ocs.nonneg(x),                       # A >= 0
                         ocs.nonneg(cap_op(T) @ x - negS),    # -(A T) + S >= 0  <=>  A T <= S
                         ocs.nonneg(asgn_op() @ x - negn)],   # -colsum + n >= 0  <=>  colsum <= n
                        cost=c)
    return prob.compile(max_blocks=2000, tol=1e-3)


def round_data(seed):
    g = np.random.default_rng(seed)
    S = g.uniform(3.0, 6.0, size=(s, r)).astype("f")         # per-shape capacities
    p = g.uniform(0.0, 1.0, size=t).astype("f")             # task-type priorities
    n = g.integers(1, 5, size=t).astype("f")                # counts per task type
    c = -np.tile(p, s).astype("f")                          # minimize -sum p_j A_ij
    return {"negS": -S.reshape(-1), "negn": -n, "c": c}, S, p, n


def ref_lp(T, S, p, n):
    from scipy.optimize import linprog
    # variables vec(A) (s*t); maximize sum p_j A_ij  ->  min -c
    cobj = -np.tile(p, s)
    rows = []; b = []
    # capacity A T <= S : for shape i, resource k: sum_j A_ij T_jk <= S_ik
    for i in range(s):
        for k in range(r):
            row = np.zeros(s * t); row[i * t:(i + 1) * t] = T[:, k]; rows.append(row); b.append(S[i, k])
    # assignment sum_i A_ij <= n_j
    for j in range(t):
        row = np.zeros(s * t); row[j::t] = 1.0; rows.append(row); b.append(n[j])
    res = linprog(cobj, A_ub=np.array(rows), b_ub=np.array(b), bounds=[(0, None)] * (s * t))
    return -float(res.fun)


if __name__ == "__main__":
    T = np.random.default_rng(0).uniform(0.2, 1.0, size=(t, r)).astype("f")   # FIXED task-type demands
    comp = build(T)

    print("== validation vs scipy linprog (per-round LP) ==")
    for seed in (1, 2, 3):
        data, S, p, n = round_data(seed)
        sol = comp.solve_instance(data)
        ocs_val = float(np.tile(p, s) @ np.asarray(sol.x))    # sum p_j A_ij
        ref = ref_lp(T, S, p, n)
        print(f"round {seed}: OCS={ocs_val:8.3f}  linprog={ref:8.3f}  "
              f"relerr={abs(ocs_val-ref)/(1+abs(ref)):.2e}  res={sol.residual:.1e} iters={sol.iters}")

    print("\n== parametric regime: same operator, many rounds, one batched solve ==")
    B = 64
    batch = {k: np.stack([round_data(1000 + i)[0][k] for i in range(B)]) for k in ("negS", "negn", "c")}
    comp.solve_many(batch)                                     # warm
    t0 = time.perf_counter(); comp.solve_many(batch); dt = time.perf_counter() - t0
    print(f"{B} scheduling rounds (A is {s}x{t}, {s*t} vars each), one batched call: {dt:.3f}s "
          f"({1000*dt/B:.2f} ms/round)")
