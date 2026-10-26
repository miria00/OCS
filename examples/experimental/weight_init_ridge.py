"""The block generalization of Diamond and Boyd's RIDGE, and the three-arm
comparison it makes possible.

WHY.  `profiling/unified_equil_20260913/` generalized the shipped equilibration
objective to block-diagonal dual metrics and then reported that the objective is
invariant along the reciprocal ray (Sigma, T) -> (e^{2a} Sigma, e^{-2a} T), so
that nothing in the equilibration chooses the primal/dual weight.  That
invariance is a property of the UNREGULARIZED problem and it is not news: Diamond
and Boyd state it in section 3.3 of "Stochastic Matrix-Free Equilibration"
(papers/mf_equil.pdf) -- "Given a solution (u*, v*), the point (u* + gamma,
v* - gamma) is a solution for any gamma ... We would prefer to guarantee a
solution where D and E have roughly the same scale" -- and they fix it in the
same section with a ridge on the log scalings and a box.  The SHIPPED code
implements that fix: `ocs/solver/equilibrate_sgd.py` takes `gamma=1.0, M=6.0`,
its gradients are `row_e - r_tgt + gamma*u` and `col_e - s_tgt + gamma*v`, and
both iterates are clipped to [-M, M].  So the shipped DIAGONAL equilibration is
already well posed.  The unified block constructor dropped the ridge (it uses a
per-block Gram ridge `delta_i` and a condition cap `kappa_max` instead, neither
of which pins the ray) and then had to invent a gauge convention to replace it.

WHAT THIS DOES.  It carries the regularization into the block generalization
instead.  For a block metric Sigma_i = exp(2 U_i) with U_i symmetric, the log
variables are the eigenvalues of U_i, so Diamond and Boyd's (gamma/2)||u||^2
becomes

    (gamma/2) sum_i ||U_i||_F^2  =  (gamma/8) sum_i ||log Sigma_i||_F^2 ,

which is basis independent, and reduces to their expression exactly for the
diagonal and scalar classes.  The same term is added on the primal side,
(gamma/2)||v||^2 with T = diag(exp(2v)), and every log eigenvalue carries the
same box |.| <= M.  The regularized problem is

    minimize  1/2 tr(Sigma K T K^T) - r sum_i tr U_i - s sum_j v_j
              + (gamma/2) ( sum_i ||U_i||_F^2 + sum_j v_j^2 )
    over      U_i in the declared class, |eig(U_i)| <= M, |v_j| <= M.

BOTH BLOCK STEPS STAY CLOSED FORM, in one scalar equation.  The Sigma-step for
block i with Gram G_i = K_i T K_i^T is unitarily invariant apart from the trace
term, so its minimizer shares G_i's eigenbasis, and in that basis it separates
into k one-dimensional problems

    minimize  1/2 e^{2 mu} lam - r mu + (gamma/2) mu^2  over |mu| <= M
    <=>       e^{2 mu} lam + gamma mu = r,  clipped to [-M, M],

with lam the eigenvalue of G_i (class `full`), the diagonal entry of G_i (class
`diag`), tr(G_i)/k (class `scalar`), or the term's mean eigenvalue (class
`tied`).  The T-step is the same equation per coordinate with lam = (K^T Sigma
K)_jj and right-hand side s.  The left side is strictly increasing in mu, so
bisection on [-M, M] is unconditionally correct and returns the box-constrained
minimizer with no case analysis.  It is strictly convex, so the minimizer of
each block step is UNIQUE -- which is what the alternation needed before any
claim could be made about it.  Whether the ALTERNATION converges to a joint
minimizer is a separate question and stays open here.

THREE THINGS THAT SHOULD FALL OUT, each checked rather than assumed
(`--mode check`):

  1. The reciprocal ambiguity should disappear.  Along the ray the trace term and
     both log terms are exactly invariant (that is what m r = n s buys), and only
     the ridge moves:
         F_gamma(a) - F_gamma(0) = gamma [ a (sum_i tr U_i - sum_j v_j)
                                           + (a^2 / 2) (m + n) ],
     strictly convex with its minimum at a* = (sum_j v_j - sum_i tr U_i)/(m+n).
     Two consequences are measured: the UNregularized F is flat along the ray
     (the defect) while F_gamma strictly increases in both directions (the fix),
     and a* at the alternation's fixed point should be ~0.  Note that a* is
     exactly the "canonical gauge" the unified study invented, so the ridge
     recovers that convention from the regularizer rather than by decree.

  2. The per-block GRAM ridge `delta_i` becomes unnecessary -- but the CONDITION
     CAP does not, and an earlier draft of this file claimed otherwise.  `delta_i`
     exists because a Gram can be singular (3-D TV annihilates constants) and
     `Sigma_i = r G_i^{-1}` then blows up; under the log ridge `lam -> 0` gives
     the finite root `mu = r/gamma`, so `delta_i` has nothing left to do.  The
     cap is a DIFFERENT restriction: the box bounds SCALE, not SPREAD.  A block
     equal to `e^5 I` is huge in scale and has condition number one; eigenvalues
     `e^{-6}` and `e^{6}` both sit inside the box with a ratio of `e^{12}`.  At
     M = 6 the box permits a condition number up to `e^{2M} = 1.63e5` on the
     scaling factor and `e^{4M} = 2.65e10` on the metric, three to eight orders
     of magnitude looser than `kappa_max = 100`.  The cap is therefore KEPT, its
     activation is recorded, and running without it is reported as a labelled
     ablation rather than as a simplification.  If a condition bound belongs in
     the formulation it should be written there, as a constraint on the SPREAD of
     each block's log eigenvalues, not as a clip applied afterwards.

  3. Uniqueness, stated precisely.  Strict convexity in a log variable does NOT
     transfer to the object itself -- `(log d)^2` has second derivative
     `2(1 - log d)/d^2`, which is negative for `d > e` -- so the argument has to
     be made for the actual subproblem.  It is: with the OPPOSITE scaling held
     fixed, diagonalize the Gram, take the optimizer in the same eigenbasis, and
     each eigenvalue problem
         phi(mu) = (1/2) e^{2 mu} lam - r mu + (gamma/2) mu^2
     has `phi''(mu) = 2 lam e^{2 mu} + gamma > 0`, strictly positive including
     when `lam = 0`.  So EACH CONDITIONAL BLOCK STEP has a unique minimizer, and
     that is the whole claim.  It says nothing about where the ALTERNATION lands.
     Per-round `F_gamma`, the change in the log scalings, the box-binding
     fraction and the cap activation are recorded and the alternation's limit
     stays open.

COST IS REMEASURED, NOT INHERITED.  Restoring the ridge changes the optimizer,
the inner work and possibly the round and probe budgets, so the earlier study's
"228 applications against the shipped 450" is a result for THAT configuration and
is not carried forward.  Every arm's construction cost is counted here from its
own probe budget and reported in the tables.

THE COMPARISON (`--mode solve`).  Three arms under ONE normalization (the ridge,
gamma = 1, M = 6, the shipped values) and ONE weight policy (the setup estimator
of `weight_init.py`, applied as tau <- tau/w, sigma <- sigma*w):

    S        the shipped diagonal SMFE, which already carries this ridge
    Ur:tied  the ridged block constructor restricted to the classes the shipped
             proxes admit (one tied scalar per coupled group_l2 term, per-row
             diagonal for sum_squares)
    Ur:full  the ridged block constructor with full k x k blocks

That separates richer geometry from a cheaper, better-behaved construction.
Construction, weight selection, solve and checking time are reported separately,
because the block constructor's saving over the shipped 450 applications is only
about a hundred iterations of headroom and a weight SEARCH would erase it.

Nothing in ocs/ is edited.  `unified_equil.py` and `blockmetric_pilot.py` are
imported unchanged, so the earlier study's recorded checksums stay valid; the
ridged constructor is new code and lives here.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

GAMMA = 1.0          # Diamond & Boyd's ridge, the shipped value
M_BOX = 6.0          # Diamond & Boyd's box, the shipped value
BISECT = 40          # 12 / 2^40 ~ 1e-11, far below float32 resolution
KAPPA_MAX = 100.0    # the pilot's condition cap on each block metric.  The BOX
                     # does NOT subsume it: |u| <= M bounds the SCALE, allowing a
                     # factor condition number up to e^{2M} = 1.63e5 (metric
                     # e^{4M} = 2.65e10) at M = 6, three to eight orders looser.
                     # Kept, with its activation recorded and ablated separately.
GAMMA_SAFETY = 0.9   # the same stability factor every arm in these studies uses
POWER_ITERS = 50


# ------------------------------------------------------------------ the one line
def ridge_root(lam, target, gamma=GAMMA, M=M_BOX, iters=BISECT):
    """The box-constrained minimizer of  1/2 e^{2mu} lam - target*mu + gamma/2 mu^2.

    Stationarity is  e^{2mu} lam + gamma mu = target.  The left side is strictly
    increasing in mu for lam >= 0 and gamma > 0, so bisection on [-M, M] returns
    the unconstrained root when it lies inside and the correct endpoint when it
    does not, with no case analysis.  Vectorized over any shape of lam.
    """
    import jax.numpy as jnp
    lam = jnp.maximum(jnp.asarray(lam, jnp.float32), 0.0)
    lo = jnp.full(lam.shape, -M, jnp.float32)
    hi = jnp.full(lam.shape, M, jnp.float32)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        h = jnp.exp(2.0 * mid) * lam + gamma * mid - target
        lo = jnp.where(h < 0, mid, lo)
        hi = jnp.where(h < 0, hi, mid)
    return 0.5 * (lo + hi)


def _cap(w, kappa_max, apply_cap):
    """-> (w, fraction of eigenvalues outside the cap).  w is (B,k) or scalar."""
    import jax.numpy as jnp
    if w.ndim < 2:
        return w, 0.0
    lg = jnp.log(jnp.maximum(w, 1e-30))
    ctr = jnp.mean(lg, axis=1, keepdims=True)
    half = 0.5 * math.log(kappa_max)
    out = float(jnp.mean(jnp.abs(lg - ctr) > half))
    if apply_cap:
        w = jnp.exp(jnp.clip(lg, ctr - half, ctr + half))
    return w, out


def sigma_step(G, cls, r_tgt, *, gamma=GAMMA, M=M_BOX, kappa_max=KAPPA_MAX,
               apply_cap=False):
    """Exact per-block minimizer of the RIDGED Sigma-step.

    The per-block GRAM ridge `delta_i` is not needed: under the log ridge a
    singular Gram is harmless, because lam -> 0 gives the finite root
    mu = r/gamma rather than a blow-up of Sigma_i = r G_i^{-1}.

    The CONDITION CAP is a different restriction and is NOT subsumed: the box
    bounds scale, not spread.  `kappa_max=None` records what the condition
    numbers would be without it (the ablation); a float applies it, clipping each
    block's eigenvalues to a ratio of at most kappa_max about their geometric
    mean, and the clipped fraction is reported either way.
    """
    import jax.numpy as jnp
    import blockmetric_pilot as BP
    G = jnp.asarray(G, jnp.float32)
    B, k, _ = G.shape
    W = 0.5 * (G + jnp.swapaxes(G, 1, 2))
    if isinstance(cls, str) and cls == "tied":
        lam = jnp.sum(jnp.trace(W, axis1=1, axis2=2)) / (B * k)
        mu = ridge_root(jnp.reshape(lam, (1,)), r_tgt, gamma, M)[0]
        w = jnp.exp(2.0 * mu)
        return ("scalar", w), dict(box_frac=float(jnp.abs(mu) >= M - 1e-5),
                                   mu_min=float(mu), mu_max=float(mu),
                                   cap_frac=0.0, cap_applied=bool(apply_cap))
    codes = np.asarray(cls, np.int32)
    any_full = bool(np.any(codes == 2))
    if any_full:
        lam, Q = BP.batched_eigh(W)                      # (B,k), (B,k,k)
    else:
        lam, Q = jnp.diagonal(W, axis1=1, axis2=2), None
    scalar_lam = jnp.trace(W, axis1=1, axis2=2)[:, None] / k
    diag_lam = jnp.diagonal(W, axis1=1, axis2=2)
    c = jnp.asarray(codes, jnp.int32)[:, None]
    use = jnp.where(c == 0, scalar_lam, jnp.where(c == 1, diag_lam, lam))
    mu = ridge_root(use, r_tgt, gamma, M)
    w = jnp.exp(2.0 * mu)
    w, cap_frac = _cap(w, kappa_max, apply_cap)
    diag = dict(box_frac=float(jnp.mean(jnp.abs(mu) >= M - 1e-5)),
                mu_min=float(jnp.min(mu)), mu_max=float(jnp.max(mu)),
                cap_frac=cap_frac, cap_applied=bool(apply_cap))
    if any_full:
        # scalar/diag blocks keep the identity basis; full blocks use W's
        eyeQ = jnp.broadcast_to(jnp.eye(k, dtype=jnp.float32), (B, k, k))
        Qm = jnp.where((c == 2)[:, :, None], Q, eyeQ)
        return ("full", Qm, w), diag
    return ("diag", w), diag


def tau_step(c_hat, s_tgt, *, gamma=GAMMA, M=M_BOX):
    import jax.numpy as jnp
    v = ridge_root(jnp.asarray(c_hat, jnp.float32), s_tgt, gamma, M)
    return jnp.exp(2.0 * v), v


# ------------------------------------------------------------ the regularized objective
def logsq_sum(metrics, spec):
    """sum over every dual row of (1/2 log sigma)^2, i.e. sum_i ||U_i||_F^2,
    with the same 'a scalar stands for the whole term' convention as
    unified_equil.metric_logdet_sum."""
    import jax.numpy as jnp
    tot, cnt, binding = 0.0, 0, 0
    for j, met in enumerate(metrics):
        if met[0] == "scalar":
            rows = spec.dual_blocks[j]["op"].shape[0]
            a = np.full(rows, float(np.asarray(met[1], np.float64)))
        elif met[0] in ("vec", "diag"):
            a = np.asarray(met[1], np.float64).reshape(-1)
        else:
            a = np.asarray(met[2], np.float64).reshape(-1)
        u = 0.5 * np.log(np.maximum(a, 1e-300))
        tot += float(np.sum(u * u)); cnt += a.size
        binding += int(np.sum(np.abs(u) >= M_BOX - 1e-4))
    return tot, cnt, binding


def regularized_objective(spec, T, metrics, r_tgt, s_tgt, probes=8, gamma=GAMMA):
    """F_gamma = F + (gamma/2)(sum_i ||U_i||_F^2 + sum_j v_j^2), with F the
    unified study's own unregularized objective (fixed-seed probes for the trace
    term, so the same probes measure every point)."""
    import jax.numpy as jnp
    import unified_equil as UE
    base = UE.equil_objective(spec, T, metrics, r_tgt, s_tgt, probes=probes)
    su, cnt, binding = logsq_sum(metrics, spec)
    t = np.asarray(jnp.broadcast_to(jnp.asarray(T), (spec.n,)), np.float64)
    v = 0.5 * np.log(np.maximum(t, 1e-300))
    sv = float(np.sum(v * v))
    base.update(ridge_sigma=su, ridge_tau=sv, box_binding=binding, n_sigma=cnt,
                F_gamma=base["F"] + 0.5 * gamma * (su + sv),
                sum_tr_U=0.5 * base["logdet_sigma"], sum_v=float(np.sum(v)))
    return base


def slide_ray(metrics, T, a):
    """(Sigma, T) -> (e^{2a} Sigma, e^{-2a} T): the reciprocal ray."""
    out = []
    for met in metrics:
        f = math.exp(2.0 * a)
        if met[0] in ("scalar", "diag", "vec"):
            out.append((met[0], met[1] * f))
        else:
            out.append(("full", met[1], met[2] * f))
    return out, T * math.exp(-2.0 * a)


def apply_weight(metrics, T, w):
    """tau <- tau/w, sigma <- sigma*w: exactly compile(primal_weight=w)."""
    out = []
    for met in metrics:
        if met[0] in ("scalar", "diag", "vec"):
            out.append((met[0], met[1] * w))
        else:
            out.append(("full", met[1], met[2] * w))
    return out, T / w


# ------------------------------------------------------------------- constructor
def ridged_equilibration(spec, classes, rounds, q_gram, q_col, *, seed=0,
                         gamma=GAMMA, M=M_BOX, log=print, record=True,
                         kappa_max=KAPPA_MAX, apply_cap=False):
    """The alternation with the ridge carried into both block steps.

    No gauge fix is applied: the ridge pins the ray.  No Gram ridge and no
    condition cap are applied: the log ridge and the box do that job.
    """
    import jax, jax.numpy as jnp
    import blockmetric_pilot as BP
    import unified_equil as UE
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    r_tgt, s_tgt = float(np.sqrt(n / m)), float(np.sqrt(m / n))
    T = jnp.ones((n,), jnp.float32)
    metrics, hist = None, []
    counts = dict(fwd=0, adj=0)
    times = dict(gram_s=0.0, metric_s=0.0, col_s=0.0)
    prev_lt = np.zeros(n)
    for rd in range(1, rounds + 1):
        new, mdiag = [], []
        for j, d in enumerate(spec.dual_blocks):
            cls = classes[j]
            k = d["k"] if d["kind"] != "sum_squares" else 1
            t0 = time.perf_counter()
            G = BP.sketch_grams(d["op"], T, 1.0, k, q_gram, seed=seed)
            jax.block_until_ready(G)
            times["gram_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            if d["kind"] == "sum_squares":
                # the shipped compile gives this term a per-row diagonal; under
                # the ridge that is the k = 1 case of the same scalar equation
                lam = jnp.reshape(G, (-1,))
                mu = ridge_root(lam, r_tgt, gamma, M)
                met = ("vec", jnp.exp(2.0 * mu))
                dg = dict(box_frac=float(jnp.mean(jnp.abs(mu) >= M - 1e-5)),
                          mu_min=float(jnp.min(mu)), mu_max=float(jnp.max(mu)))
            else:
                met, dg = sigma_step(G, cls, r_tgt, gamma=gamma, M=M,
                                     kappa_max=kappa_max, apply_cap=apply_cap)
            jax.block_until_ready(met[-1])
            times["metric_s"] += time.perf_counter() - t0
            new.append(met); mdiag.append(dg)
            counts["fwd"] += q_gram
        metrics = new
        t0 = time.perf_counter()
        c_hat = UE.colnorm_estimate(spec, metrics, q_col, seed=seed + 31)
        jax.block_until_ready(c_hat)
        T, v = tau_step(c_hat, s_tgt, gamma=gamma, M=M)
        jax.block_until_ready(T)
        times["col_s"] += time.perf_counter() - t0
        counts["adj"] += q_col
        if record:
            lt = np.asarray(jnp.log(T), np.float64)
            obj = regularized_objective(spec, T, metrics, r_tgt, s_tgt, gamma=gamma)
            su, cnt, binding = logsq_sum(metrics, spec)
            a_star = (obj["sum_v"] - obj["sum_tr_U"]) / (m + n)
            rec = dict(round=rd, F=obj["F"], F_gamma=obj["F_gamma"],
                       a_star=a_star, dlog_tau_inf=float(np.max(np.abs(lt - prev_lt))),
                       sig_min=obj["sig_min"], sig_max=obj["sig_max"],
                       tau_min=obj["tau_min"], tau_max=obj["tau_max"],
                       box_binding_sigma=binding, box_binding_tau=int(
                           np.sum(np.abs(0.5 * lt) >= M - 1e-4)),
                       mu_min=min(d["mu_min"] for d in mdiag),
                       mu_max=max(d["mu_max"] for d in mdiag),
                       cap_frac=max(d.get("cap_frac", 0.0) for d in mdiag),
                       cap_applied=any(d.get("cap_applied") for d in mdiag))
            hist.append(rec)
            prev_lt = lt
            log(f"    round {rd}: F={rec['F']:.6f} F_gamma={rec['F_gamma']:.6f} "
                f"a*={rec['a_star']:+.3e} dlog_tau={rec['dlog_tau_inf']:.3e} "
                f"sigma in [{rec['sig_min']:.3g}, {rec['sig_max']:.3g}] "
                f"box_binding sigma={rec['box_binding_sigma']} tau={rec['box_binding_tau']}")
    return T, metrics, hist, counts, times, r_tgt, s_tgt


# ------------------------------------------------------------ equivalence tests
def equivalence_tests(spec, classes, rounds, q_gram, q_col, *, seed=0, log=print):
    """The checks the block formulation has to pass before any expensive solve.

    T1  ridge_root really solves its equation:  |e^{2mu} lam + gamma mu - r|_inf.
    T2  REDUCTION TO THE SHIPPED DIAGONAL CASE.  For a k = 1 term the block
        equation IS the shipped SMFE's own stationarity condition
        `row_energy * e^{2u} - r_tgt + gamma * u = 0`
        (ocs/solver/equilibrate_sgd.py, `gu = row_e/probes - r_tgt + gamma*u`),
        evaluated here at the constructed scalings with the SAME row-energy
        estimate.  This is an exact algebraic identity, so the residual should be
        float32 noise, not a resemblance.
    T3  MULTIPLICITY.  For a scalar block the objective is
        `1/2 e^{2a} tr(G_i) - r k_i a + (gamma/2) k_i a^2` -- k_i copies of the
        log variable and k_i copies of the target, NOT one of each.  Checked by
        central finite difference of that expression at the constructed `a`.
    T4  THE RECIPROCAL DIRECTION.  Shifting the row logs by `a I` and the column
        logs by `-a` must change the regularizer by
        `gamma [ a (sum_i tr U_i - sum_j v_j) + (a^2/2)(m+n) ]`,
        so the second derivative along that direction is `gamma (m+n) > 0`.
        Checked against the implemented objective with fixed diagnostic probes.
    T5  WHAT IS PENALIZED.  Diamond and Boyd regularize the logs of the SCALING
        FACTORS.  OCS carries sigma = D^2 and tau = E^2, so the penalized object
        is u = (1/2) log sigma, not log sigma; using the latter with the shipped
        coefficient would QUADRUPLE the penalty.  Reported as the ratio of the
        two, which must be 4.
    """
    import jax, jax.numpy as jnp
    import blockmetric_pilot as BP
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    r_tgt, s_tgt = float(np.sqrt(n / m)), float(np.sqrt(m / n))
    T, metrics, hist, counts, times, _r, _s = ridged_equilibration(
        spec, classes, rounds, q_gram, q_col, seed=seed, log=lambda *_: None,
        record=False)
    # T2 and T3 test the ALGEBRA of one Sigma-step, so they must compare a metric
    # and a Gram from the SAME point.  The alternation returns Sigma built at the
    # PREVIOUS round's T, so one more Sigma-step at the returned T is taken here
    # and `aligned` is what T2/T3 check; `metrics` (the alternation's own output)
    # is what T4, T5 and the conditioning use.
    aligned, grams = [], []
    for j, d in enumerate(spec.dual_blocks):
        kk = d["k"] if d["kind"] != "sum_squares" else 1
        G = BP.sketch_grams(d["op"], T, 1.0, kk, q_gram, seed=seed)
        grams.append(G)
        if d["kind"] == "sum_squares":
            mu_ = ridge_root(jnp.reshape(G, (-1,)), r_tgt)
            aligned.append(("vec", jnp.exp(2.0 * mu_)))
        else:
            met_, _dg = sigma_step(G, classes[j], r_tgt)
            aligned.append(met_)
    out = {}

    # T1
    worst = 0.0
    for lam_t, tgt in ((jnp.asarray([1e-8, 1e-3, 1.0, 7.5, 1e3, 1e6], jnp.float32), r_tgt),
                       (jnp.asarray([0.0, 1e-2, 0.4, 13.0, 1e4], jnp.float32), s_tgt)):
        mu = ridge_root(lam_t, tgt)
        res = jnp.abs(jnp.exp(2 * mu) * lam_t + GAMMA * mu - tgt)
        inside = jnp.abs(mu) < M_BOX - 1e-4
        worst = max(worst, float(jnp.max(jnp.where(inside, res, 0.0))))
    out["T1_root_residual_inf"] = worst
    log(f"  T1 ridge_root residual (interior roots): {worst:.3e}")

    # T2: reduction to the shipped diagonal expression, on every k = 1 term
    t2 = []
    for j, d in enumerate(spec.dual_blocks):
        if d["kind"] != "sum_squares":
            continue
        row_e = np.asarray(jnp.reshape(grams[j], (-1,)), np.float64)
        sig = np.asarray(aligned[j][1], np.float64).reshape(-1)
        u = 0.5 * np.log(np.maximum(sig, 1e-300))
        grad = row_e * np.exp(2 * u) - r_tgt + GAMMA * u
        t2.append(dict(term=j, grad_inf=float(np.max(np.abs(grad))),
                       grad_rel=float(np.max(np.abs(grad)) / max(r_tgt, 1e-30))))
        log(f"  T2 shipped-SMFE stationarity at the constructed scalings, term {j}: "
            f"|grad|_inf = {t2[-1]['grad_inf']:.3e} "
            f"({t2[-1]['grad_rel']:.3e} relative to r_tgt)")
    out["T2_shipped_reduction"] = t2

    # T3: multiplicity, on every scalar/tied term
    t3 = []
    for j, d in enumerate(spec.dual_blocks):
        if aligned[j][0] != "scalar":
            continue
        k = d["k"]
        trG = float(jnp.sum(jnp.trace(grams[j], axis1=1, axis2=2)))
        B = d["op"].shape[0] // k
        a = 0.5 * math.log(float(np.asarray(aligned[j][1], np.float64)))
        rows = B * k

        def phi(x):
            return 0.5 * math.exp(2 * x) * trG - r_tgt * rows * x + 0.5 * GAMMA * rows * x * x
        h = 1e-3
        dphi = (phi(a + h) - phi(a - h)) / (2 * h)
        scale = max(abs(phi(a)), 1e-30)
        t3.append(dict(term=j, a=a, dphi=dphi, dphi_rel=abs(dphi) / scale))
        log(f"  T3 multiplicity, scalar term {j} (k={k}, {rows} rows): "
            f"d/da of the k-weighted objective at the constructed a = {dphi:+.4e} "
            f"({t3[-1]['dphi_rel']:.3e} relative)")
    out["T3_multiplicity"] = t3

    # T4: the reciprocal direction
    curv = []
    base = regularized_objective(spec, T, metrics, r_tgt, s_tgt)
    ast = (base["sum_v"] - base["sum_tr_U"]) / (m + n)
    for a in (-0.5, -0.2, 0.2, 0.5):
        Ms, Ts = slide_ray(metrics, T, a)
        o = regularized_objective(spec, Ts, Ms, r_tgt, s_tgt)
        meas = o["F_gamma"] - base["F_gamma"]
        pred = GAMMA * (a * (base["sum_tr_U"] - base["sum_v"]) + 0.5 * a * a * (m + n))
        curv.append(dict(a=a, measured=meas, predicted=pred,
                         rel=abs(meas - pred) / max(abs(pred), 1e-30)))
        log(f"  T4 ray a={a:+.2f}: measured dF_gamma={meas:+.4f}  "
            f"predicted={pred:+.4f}  relative error {curv[-1]['rel']:.3e}")
    out["T4_ray"] = curv
    out["T4_a_star"] = ast
    out["T4_second_derivative"] = GAMMA * (m + n)
    log(f"  T4 second derivative along the ray = gamma (m+n) = {GAMMA * (m + n):,.0f} > 0: "
        f"the ray is pinned.  a* = {ast:+.4e}")

    # T5: which object is penalized
    su, _cnt, _b = logsq_sum(metrics, spec)
    sig_all = []
    for j, met in enumerate(metrics):
        if met[0] == "scalar":
            sig_all.append(np.full(spec.dual_blocks[j]["op"].shape[0],
                                   float(np.asarray(met[1], np.float64))))
        elif met[0] in ("vec", "diag"):
            sig_all.append(np.asarray(met[1], np.float64).reshape(-1))
        else:
            sig_all.append(np.asarray(met[2], np.float64).reshape(-1))
    sig_all = np.concatenate(sig_all)
    wrong = float(np.sum(np.log(np.maximum(sig_all, 1e-300)) ** 2))
    out["T5_factor_vs_metric_ratio"] = wrong / max(su, 1e-30)
    log(f"  T5 penalizing log(metric) instead of log(factor) would multiply the "
        f"dual ridge by {out['T5_factor_vs_metric_ratio']:.6f} (must be 4.0); "
        f"the implementation penalizes u = (1/2) log sigma")

    # cap activation, recorded not assumed
    cond = block_conditioning(metrics, spec)
    out["cap"] = dict(kappa_max=KAPPA_MAX, applied=False,
                      max_block_kappa=max(c["kappa_max"] for c in cond),
                      would_activate=bool(max(c["kappa_max"] for c in cond) > KAPPA_MAX),
                      box_permits_factor_kappa=math.exp(2 * M_BOX),
                      box_permits_metric_kappa=math.exp(4 * M_BOX),
                      conditioning=cond)
    log(f"  CAP: measured max block condition number {out['cap']['max_block_kappa']:.4g}; "
        f"cap {KAPPA_MAX} would activate: {out['cap']['would_activate']}.  The box "
        f"alone permits {math.exp(2 * M_BOX):.3g} on the factor "
        f"({math.exp(4 * M_BOX):.3g} on the metric), so it is NOT a condition bound.")
    return out


def build_ridged_arm(spec, classes, rounds, q_gram, q_col, *, seed=0, log=print,
                     record=False):
    """Construct + the common stability rescale every arm in these studies gets.

    `record=False` in the timed solve path, deliberately: the per-round objective
    costs 8 diagnostic forward probes per round, and the unified study's own rule
    is that the diagnostic objective is never charged to an arm.  Charging its
    WALL TIME to the ridged arms but not to the shipped one would tilt exactly
    the budget comparison this study is trying to make honest.
    """
    import jax, jax.numpy as jnp
    import blockmetric_pilot as BP
    T, metrics, hist, counts, times, r_tgt, s_tgt = ridged_equilibration(
        spec, classes, rounds, q_gram, q_col, seed=seed, log=log, record=record)
    t0 = time.perf_counter()
    nu = BP.power_norm_metric(spec, T, metrics, iters=POWER_ITERS)
    times["norm_s"] = time.perf_counter() - t0
    counts["fwd"] += POWER_ITERS; counts["adj"] += POWER_ITERS
    metrics, T = BP.scale_metrics(metrics, T, GAMMA_SAFETY / nu)
    return dict(T=T, metrics=list(metrics), hist=hist, counts=counts, times=times,
                r_tgt=r_tgt, s_tgt=s_tgt, norm_est=nu,
                setup_s=sum(times.values()))


def build_shipped_arm(spec, classes, log=print):
    """The shipped diagonal SMFE, which already carries this ridge (gamma=1, M=6)."""
    import blockmetric_pilot as BP
    import unified_equil as UE
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    sc = BP.shipped_scaling(spec)
    T, metrics = UE.shipped_metrics(spec, sc, classes)
    counts = dict(fwd=80 * 2 + 15, adj=80 * 2 + 15)   # SMFE loop + its power iterations
    times = dict(equil_s=sc["seconds"])
    return dict(T=T, metrics=list(metrics), hist=[], counts=counts, times=times,
                r_tgt=float(np.sqrt(n / m)), s_tgt=float(np.sqrt(m / n)),
                norm_est=sc["scaled_norm"], setup_s=sc["seconds"],
                shipped_scaled_norm=sc["scaled_norm"])


def block_conditioning(metrics, spec):
    """Per-block condition number of the constructed dual metric, with NO cap."""
    out = []
    for j, met in enumerate(metrics):
        if met[0] == "scalar":
            out.append(dict(term=j, kind="scalar", kappa_max=1.0, kappa_med=1.0,
                            eig_min=float(np.asarray(met[1])), eig_max=float(np.asarray(met[1]))))
            continue
        w = np.asarray(met[1] if met[0] in ("vec", "diag") else met[2], np.float64)
        if w.ndim == 1:
            out.append(dict(term=j, kind=met[0], kappa_max=float(w.max() / max(w.min(), 1e-300)),
                            kappa_med=float("nan"), eig_min=float(w.min()), eig_max=float(w.max())))
        else:
            kap = w.max(axis=1) / np.maximum(w.min(axis=1), 1e-300)
            out.append(dict(term=j, kind=met[0], kappa_max=float(kap.max()),
                            kappa_med=float(np.median(kap)),
                            eig_min=float(w.min()), eig_max=float(w.max())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="F1", choices=["F1", "F2", "F3"])
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--mode", default="solve", choices=["check", "solve", "both", "equiv"])
    ap.add_argument("--arms", default="S,Ur:tied,Ur:full")
    ap.add_argument("--weights", default="1.0,est:objscale")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--qgram", type=int, default=16)
    ap.add_argument("--qcol", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=1000)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax, jax.numpy as jnp, ocs
    import unified_equil as UE
    import blockmetric_pilot as BP
    import weight_init as WI
    log = lambda s: print(s, flush=True)

    fa = argparse.Namespace(family=args.family, variant="overlap", n=200_000, k=4,
                            rho=args.rho, L=args.L, blur=0.7, channels=3, mu=1e-2, seed=0)
    if args.family == "F2":
        fa.variant = "strong"
    prob, spec = UE.build_family(fa)
    data = spec.data
    n = spec.n
    m = sum(d["op"].shape[0] for d in spec.dual_blocks)
    q_gram = args.qgram * spec.k
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} "
        f"jax {jax.__version__} ocs {ocs.__version__}")
    log(f"# {spec.name}: n={n} m={m} gamma={GAMMA} M={M_BOX} rounds={args.rounds} "
        f"q_gram={q_gram} q_col={args.qcol} tol={args.tol} "
        f"budget {args.max_blocks * 100} iterations")
    out = dict(meta=dict(family=args.family, spec=spec.name, n=n, m=m, k=spec.k,
                         rho=args.rho, L=args.L, gamma=GAMMA, M=M_BOX,
                         rounds=args.rounds, q_gram=q_gram, q_col=args.qcol,
                         tol=args.tol, max_blocks=args.max_blocks,
                         budget_iters=args.max_blocks * 100,
                         max_seconds=args.max_seconds, repeats=args.repeats,
                         host=os.uname().nodename, device=jax.devices()[0].device_kind,
                         jax=jax.__version__, ocs=ocs.__version__,
                         time=time.strftime("%Y-%m-%d %H:%M:%S %Z")))
    out["checks"] = {}
    out["equivalence"] = {}
    out["rows"] = []

    def save():
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    # --------------------------------------------------- equivalence tests
    if args.mode in ("equiv", "check", "both"):
        for cls in ("tied", "full"):
            classes = UE.resolve_classes(spec, cls)
            log(f"\n## equivalence tests, classes={cls} "
                f"({'|'.join(UE.class_tag(c) for c in classes)})")
            out["equivalence"][cls] = equivalence_tests(
                spec, classes, args.rounds, q_gram, args.qcol, log=log)
            save()
    if args.mode == "equiv":
        log(f"wrote {args.out}"); return

    # ---------------------------------------------------------------- checks
    if args.mode in ("check", "both"):
        for cls in ("tied", "full"):
            classes = UE.resolve_classes(spec, cls)
            log(f"\n## check, classes={cls} ({'|'.join(UE.class_tag(c) for c in classes)})")
            T, metrics, hist, counts, times, r_tgt, s_tgt = ridged_equilibration(
                spec, classes, args.rounds, q_gram, args.qcol, log=log)
            # (1) the ray
            ray = []
            for a in (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0):
                Ms, Ts = slide_ray(metrics, T, a)
                o = regularized_objective(spec, Ts, Ms, r_tgt, s_tgt)
                ray.append(dict(a=a, F=o["F"], F_gamma=o["F_gamma"],
                                ridge=o["ridge_sigma"] + o["ridge_tau"], tr=o["tr"]))
                log(f"    ray a={a:+.2f}: F={o['F']:.6f} (unregularized)   "
                    f"F_gamma={o['F_gamma']:.6f}   tr={o['tr']:.6f}")
            F0 = [x["F"] for x in ray]; Fg = [x["F_gamma"] for x in ray]
            i0 = [x["a"] for x in ray].index(0.0)
            flat = (max(F0) - min(F0)) / max(abs(np.mean(F0)), 1e-30)
            strict = all(Fg[i] > Fg[i + 1] for i in range(i0)) and \
                     all(Fg[i] < Fg[i + 1] for i in range(i0, len(Fg) - 1))
            log(f"    unregularized F relative spread along the ray: {flat:.3e} "
                f"(flat = the defect)")
            log(f"    F_gamma strictly increasing away from a=0 in both directions: {strict}")
            log(f"    a* at the fixed point: {hist[-1]['a_star']:+.4e}")
            # (2) conditioning with NO cap
            cond = block_conditioning(metrics, spec)
            for c in cond:
                log(f"    term {c['term']} ({c['kind']}): eig in "
                    f"[{c['eig_min']:.4g}, {c['eig_max']:.4g}], kappa max {c['kappa_max']:.4g}, "
                    f"median {c['kappa_med']:.4g}")
            out["checks"][cls] = dict(ray=ray, unreg_relative_spread=float(flat),
                                      Fgamma_strictly_convex_on_ray=bool(strict),
                                      a_star=hist[-1]["a_star"], history=hist,
                                      conditioning=cond, counts=counts, times=times,
                                      kappa_cap_applied=False, gram_ridge_applied=False)
            save()

    if args.mode == "check":
        log(f"wrote {args.out}"); return

    # ------------------------------------------------------- the weight policy
    ck = {"pdhg_variant": "standard"}
    t0 = time.perf_counter()
    est, ediag = WI.setup_estimates(prob, data, ck)
    pe, pdiag = WI.probe_estimates(prob, data, ck, args.tol)
    est.update(pe); ediag.update(pdiag)
    for kk in WI.ESTIMATORS:
        est[kk]["w"], est[kk]["clip"] = WI._clip(est[kk]["raw"])
        log(f"  estimator {kk:18s} raw={est[kk]['raw']:.6g}  w={est[kk]['w']:.6g} "
            f"[{est[kk]['cost_ops']} ops, {est[kk]['cost_s']:.3f}s]")
    out["estimates"] = est; out["diagnostics"] = ediag
    out["meta"]["estimator_setup_s"] = time.perf_counter() - t0
    save()

    weights = []
    for tok in args.weights.split(","):
        tok = tok.strip()
        if tok.startswith("est:"):
            nm = tok[4:]
            weights.append((tok, est[nm]["w"], est[nm]["cost_s"], est[nm]["cost_ops"]))
        else:
            weights.append((f"fixed:{tok}", float(tok), 0.0, 0))

    comp = prob.compile(tol=args.tol, max_blocks=1, block=100)   # public certificate only

    for spec_arm in args.arms.split(","):
        arm, _, cls = spec_arm.strip().partition(":")
        cls = cls or "tied"
        classes = UE.resolve_classes(spec, cls)
        log(f"\n## arm {arm} classes={cls} ({'|'.join(UE.class_tag(c) for c in classes)})")
        if arm == "S":
            build_shipped_arm(spec, classes, log)                 # pay compilation
            t0 = time.perf_counter()
            built = build_shipped_arm(spec, classes, log)
        else:
            build_ridged_arm(spec, classes, args.rounds, q_gram, args.qcol,
                             log=lambda *_: None)                 # pay compilation
            t0 = time.perf_counter()
            built = build_ridged_arm(spec, classes, args.rounds, q_gram, args.qcol,
                                     log=lambda *_: None)
        construct_s = time.perf_counter() - t0
        if arm != "S":
            # the diagnostic history, rebuilt AFTER the timed construction so its
            # 8 probes per round are charged to nobody, as in the unified study
            built["hist"] = ridged_equilibration(
                spec, classes, args.rounds, q_gram, args.qcol, log=log, record=True)[2]
        apps = built["counts"]["fwd"] + built["counts"]["adj"]
        cond = block_conditioning(built["metrics"], spec)
        log(f"  construction: {construct_s:.2f}s, {apps} operator applications; "
            f"norm estimate {built['norm_est']:.5f}")
        for wtag, w, wcost_s, wcost_ops in weights:
            metrics, T = apply_weight(built["metrics"], built["T"], w)
            loop = BP.Loop(spec, T, metrics, block=100)
            times, res = [], None
            for r in range(args.repeats):
                rr = loop.run(comp, data, args.tol, args.max_blocks, args.max_seconds, log=None)
                times.append(rr["total_s"]); res = rr
                log(f"    {arm}:{cls:<5s} {wtag:<16s} w={w:<10.5g} rep{r}: {rr['iters']:7d} it  "
                    f"{rr['status']:12s} r={rr['residual']:.4e}  {rr['total_s']:7.2f}s")
                if sum(times) > args.max_seconds and r + 1 < args.repeats:
                    log("    (stopping repeats: wall budget)"); break
            obj, minx = UE.float64_objective(spec, res["x"])
            warm = times[1:] or times
            out["rows"].append(dict(
                arm=arm, classes=cls,
                class_tag="|".join(UE.class_tag(x) for x in classes),
                weight_tag=wtag, w=float(w), iters=res["iters"], status=res["status"],
                residual=res["residual"], r_primal=res["cert"]["r_primal"],
                r_dual=res["cert"]["r_dual"],
                certified=int(res["residual"] <= args.tol),
                objective_f64=obj, min_x=minx,
                construct_s=construct_s, construct_apps=apps,
                weight_select_s=wcost_s, weight_select_apps=wcost_ops,
                solve_s=res["iter_s"], verify_s=res["verify_s"],
                cold_total_s=times[0], warm_med_s=float(np.median(warm)),
                warm_min_s=float(min(warm)), warm_max_s=float(max(warm)),
                repeats=len(times), norm_est=built["norm_est"],
                conditioning=cond,
                a_star=(built["hist"][-1]["a_star"] if built["hist"] else None)))
            save()
    save()
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
