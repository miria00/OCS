"""Matrix-free equilibration -- scalar vs Diamond-Boyd SMFE, on TWO ill-conditioned
operator families, to show the win is not specific to one problem.

  (A) column-scaled dense NNLS:  min 1/2||A x - b||^2 s.t. x>=0, A = G diag(s),
      column norms spanning a range R  (primal/column ill-conditioning).
  (B) row-weighted FFT NNLS:     min 1/2||W F x - c||^2 s.t. x>=0, K = diag(w) F a
      subsampled FFT with per-row weights spanning R  (dual/row ill-conditioning,
      operator-native -- K is a fast transform, never formed).

Both equilibrators use only matvec/rmatvec.  We report the residual at a fixed
6000-iteration budget (geo-mean over trials).  SMFE (projected SGD over row/column
log-scalings) stays flat as R grows in BOTH; the scalar step degrades.

Run:  python examples/equilibration_ablation.py
"""
import os
import numpy as np
from scipy.stats import gmean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ocs

BUDGET, TRIALS = 60, 5
Rs = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def solve_resid(terms_fn, R, kind):
    """Average residual over TRIALS random instances for one R and equilibrator."""
    out = []
    for trial in range(TRIALS):
        r = np.random.default_rng(500 + trial)
        prob, data = terms_fn(R, r)
        comp = prob.compile(max_blocks=BUDGET, tol=1e-12, equilibrate=kind)
        out.append(comp.solve_instance(data).residual)
    return gmean(out)


# ---- Family A: column-scaled dense NNLS ----
mA, nA = 240, 180
GA = np.random.default_rng(0).normal(size=(mA, nA)).astype(np.float32)
def familyA(R, r):
    s = (R ** r.uniform(0.0, 1.0, size=nA)).astype(np.float32)
    A = GA * s[None, :]
    b = (A @ np.abs(r.normal(size=nA)).astype(np.float32)).astype(np.float32)
    x = ocs.Variable(nA); bp = ocs.Parameter("b")
    return ocs.minimize([ocs.nonneg(x), ocs.least_squares(ocs.Dense(A) @ x - bp)]), {"b": b}


# ---- Family B: row-weighted subsampled-FFT NNLS (operator-native) ----
nB = 1024
FB = ocs.FFTMask.random(nB, ratio=0.5, seed=0)
mB = FB.shape[0]
def familyB(R, r):
    w = (R ** r.uniform(0.0, 1.0, size=mB)).astype(np.float32)
    K = ocs.Weighted(FB, w)
    xt = np.zeros(nB); xt[r.choice(nB, nB // 20, replace=False)] = np.abs(r.normal(size=nB // 20))
    c = np.asarray(K.matvec(np.asarray(xt, np.float32)))
    x = ocs.Variable(nB); cp = ocs.Parameter("c")
    return ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - cp)]), {"c": c}


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (name, fam) in zip(axes, [("(A) column-scaled dense NNLS", familyA),
                                   ("(B) row-weighted FFT NNLS (operator-native)", familyB)]):
    sc = [solve_resid(fam, R, "scalar") for R in Rs]
    sg = [solve_resid(fam, R, "sgd") for R in Rs]
    for R, a, b in zip(Rs, sc, sg):
        print(f"  {name[:3]} R={R:6.0f}   scalar {a:.2e}   SMFE {b:.2e}   {a/b:.0f}x")
    ax.loglog(Rs, sc, "s--", lw=2.4, ms=8, color="#d62728", label="scalar")
    ax.loglog(Rs, sg, "o-", lw=2.6, ms=8, color="#1f77b4", label="SMFE (Diamond-Boyd)")
    ax.axhline(1e-3, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("ill-conditioning range  R"); ax.set_ylabel("residual after 6000 iters")
    ax.set_title(name, fontsize=11); ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()

fig.suptitle("Matrix-free stochastic equilibration (Diamond-Boyd) stays flat as K worsens, "
             "on two operator families", fontsize=12.5)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "output", "equilibration_ablation.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("-> wrote", out)
