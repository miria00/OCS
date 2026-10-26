"""Can the PDHG primal weight be chosen automatically, once, from problem data?

Two independent results of 2026-09-13 say the primal weight dominates and that
nothing currently selects it:

  profiling/mcf_restart_20260913/SUMMARY.md   -- the shipped default w = 1 turns
      a multicommodity-flow solve into a budget-exhausted failure; the weight
      that works is 32 to 256 depending on size and bar.
  profiling/unified_equil_20260913/README.md  -- the equilibration objective is
      exactly INVARIANT along the primal/dual weight ray while PDHG is not, so
      no equilibration, diagonal or block, can choose the parameter.
  profiling/gate_campaign_20260911/GATE_COUNTEREXAMPLES.md -- the adaptive gate
      moves in quantized factor-of-two steps and cannot close a 64-fold global
      misestimate inside a practical budget.

This module implements FIVE one-time, setup-only, operator-pure estimators of
the primal weight and measures them against the default of 1 and against the
best swept weight, on families where the answer is known and on families where
it is not.

CONVENTION.  OCS `compile(primal_weight=w)` sets tau = tau0 / w (primal) and
sigma = sigma0 * w (dual), where (tau0, sigma0) come from the equilibration.
Larger w means a larger dual step.  This is exactly PDLP's omega.

THE COMMON PRINCIPLE.  After equilibration the preconditioned operator has norm
about one, so the primal and dual iterates have the scales of their own data.
Every estimator below is an estimate of the SAME quantity, the ratio of dual
solution scale to primal solution scale,

    w* ~ ||y*||_2 / ||x*||_2 ,

and they differ only in how that ratio is guessed without solving.  PDLP's own
initialization ||c|| / ||b|| and PDLP's own weight update ||dy|| / ||dx|| are
both instances of it.

ESTIMATORS (one rule, one set of constants, applied to every family):

  E0 default      w = 1.  The shipped default.  Not an estimator; the baseline.

  E1 pdlp         w = ||c'||_2 / ||l'||_2, fallback 1 when either is zero.
                  Faithful transcription of PDLP's `select_initial_primal_weight`
                  (Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy,
                  "Practical Large-Scale Linear Programming using Primal-Dual
                  Hybrid Gradient", NeurIPS 2021; FirstOrderLp.jl, with
                  primal_importance = 1), applied like PDLP's to the RESCALED
                  problem: c' = T^{1/2} c and l' = Sigma^{1/2} l, where
                  T = diag(tau0) and Sigma = diag(sigma0) are the shipped
                  equilibration's own steps.  Cost: zero extra operator work.

  E2 objscale     w = ||T^{1/2}(c + K^T yref)||_2 / ||Sigma^{1/2} l||_2.
                  PDLP's ratio with the missing cost supplied by the proxes:
                  OCS families often have no linear cost at all (multicommodity
                  flow does not), in which case E1 degenerates to 1 by its own
                  fallback.  yref is the stacked Moreau subgradient of each dual
                  block AT ITS OWN OFFSET at unit step,
                      yref_i = l_i - prox_{f_i}(l_i, 1)  in  d f_i(prox(l_i)),
                  which is the same anchor the shipped stopping rule uses to set
                  its stationarity scale (ocs/solver/stopping.py).  It is
                  reached only through the public prox interface, and K only
                  through ONE adjoint product.  Cost: 1 adjoint application.

  E3 probe_move   P = 100 iterations of the shipped engine at w = 1 from a cold
                  start, then w = ||y_P - y_0|| / ||x_P - x_0|| = ||y_P||/||x_P||.
                  This is PDLP's primal-weight update formula applied ONCE, at
                  the first restart, with smoothing theta = 1.  Cost: 100
                  iterations, about 202 operator applications, plus one compile.

  E3L/E4L probe_move_long / probe_resid_long
                  The SAME two rules with P = 1000 instead of P = 100.  A longer
                  probe is a different rule with a different cost, not a tuned
                  version of the same one, so it is reported as its own arm.  It
                  shares the compiled executable with the short probe and is
                  exactly the short probe continued.  Cost: 1000 iterations,
                  about 2002 operator applications, plus one compile.

  E4 probe_resid  The same probe, read differently:
                      w = sqrt( r_dual / r_primal )
                  from the shipped certificate at the probe point.  In OCS's
                  relative contract r_dual is the dual-block graph error (the
                  feasibility side) and r_primal is the stationarity error, so a
                  large r_dual/r_primal is exactly the "primal update has
                  nothing left to do while the dual step is too weak" signature
                  diagnosed on multicommodity flow.  The square root is the
                  natural exponent: w enters tau and sigma reciprocally, so the
                  residual ratio moves as w^2.  Cost: shared with E3, free.

  E5 equil        Reuse of quantities the setup already computed, with NO extra
                  operator work at all: the minimum-log-norm point of the weight
                  ray, the "canonical gauge" of the unified equilibration study,
                      w = exp( ( sum_j log tau0_j - sum_i log sigma0_i ) / (m+n) ).
                  It is the unique w minimizing ||log(tau0/w)||^2 + ||log(sigma0 w)||^2,
                  i.e. the point that makes the primal and dual metrics equally
                  far from unit scale.  Cost: zero.

CONSTANTS, fixed for every family: probe length P = 100 iterations; clip of
every estimate to [1e-3, 1e3]; no quantization (the engine takes any positive
float).  Nothing here is chosen per family.  A per-family choice would be a
different experiment and would have to be labelled as one.

OPERATOR PURITY.  K is reached only through matvec/rmatvec, and the prox blocks
only through the public `prox(v, t)`.  `--purity` re-runs every estimator with
each operator wrapped in a proxy that exposes ONLY matvec/rmatvec, counts the
applications, and raises on any other attribute access; the estimates must come
out identical.

Nothing in ocs/ is edited.  The shipped engine is compiled through its public
`ProblemFamily.compile(primal_weight=w)` and driven through `solve_instance`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the OCS tree
sys.path.insert(0, os.path.dirname(HERE))                    # examples/
sys.path.insert(0, HERE)                                     # examples/experimental/

PROBE_ITERS = 100
PROBE_ITERS_LONG = 1000
W_LO, W_HI = 1e-3, 1e3
ESTIMATORS = ("pdlp", "objscale", "probe_move", "probe_resid",
              "probe_move_long", "probe_resid_long", "equil")


def _clip(w):
    try:
        w = float(w)
    except (TypeError, ValueError):
        return 1.0, "nonfinite->1"
    if not math.isfinite(w) or w <= 0:
        return 1.0, "nonfinite->1"
    if w < W_LO:
        return W_LO, "clipped_lo"
    if w > W_HI:
        return W_HI, "clipped_hi"
    return float(w), ""


# --------------------------------------------------------------------- purity
class _PureOp:
    """Exposes ONLY matvec/rmatvec/shape and counts applications.

    Any other attribute access raises, so a code path that reaches for
    `dense_blocks`, `row_energy`, `sq_matvec`, an exact Gram or an entry of K
    fails loudly instead of silently succeeding.
    """
    _ALLOWED = ("matvec", "rmatvec", "shape", "fwd", "adj")

    def __init__(self, op, tally):
        object.__setattr__(self, "_op", op)
        object.__setattr__(self, "_tally", tally)
        object.__setattr__(self, "shape", op.shape)

    def matvec(self, x):
        self._tally["fwd"] += 1
        return self._op.matvec(x)

    def rmatvec(self, y):
        self._tally["adj"] += 1
        return self._op.rmatvec(y)

    def __getattr__(self, name):
        raise AttributeError(
            f"operator purity violation: estimator reached for {name!r} on {type(self._op).__name__}")


# --------------------------------------------------------------------- routing
def route(prob):
    """Mirror compile_family's routing: first offset-free Identity term is the
    primal block g, everything else is a dual block.  No engine edit; this is
    the same rule read off ocs/model/compile.py."""
    from ocs.operators import Identity
    g_block, dual = None, []
    for t in prob.terms:
        if g_block is None and isinstance(t.op, Identity) and t.offset_fn is None:
            g_block = t.block
        else:
            dual.append(t)
    if not dual:
        raise ValueError("need at least one term through a non-identity operator")
    ops = [t.op for t in dual]
    blocks = [t.block for t in dual]
    dims = [op.shape[0] for op in ops]
    slices, a = [], 0
    for d in dims:
        slices.append((a, a + d)); a += d
    return g_block, dual, ops, blocks, slices, sum(dims)


def stacked(ops, slices):
    import jax.numpy as jnp

    def matvec(x):
        return jnp.concatenate([op.matvec(x) for op in ops])

    def rmatvec(y):
        return sum(op.rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))
    return matvec, rmatvec


def equilibrate_like_compile(prob, data, ck, tally=None):
    """Reproduce the steps (tau0, sigma0) the shipped compile would install,
    by calling the SHIPPED equilibration on the SAME stacked operator.  The
    scaled-norm is returned so the caller can check it against the compiled
    family's own meta: agreement to float32 is the evidence that the routing
    and the equilibration matched."""
    import jax.numpy as jnp
    from ocs.solver.equilibrate import scalar_steps
    from ocs.solver.equilibrate_sgd import smfe_equilibration

    g_block, dual, ops, blocks, slices, m = route(prob)
    n = prob.n
    if tally is not None:
        ops = [_PureOp(o, tally) for o in ops]
    matvec, rmatvec = stacked(ops, slices)
    equil = ck.get("equilibrate", "sgd")
    scalar_spans = [sl for blk, sl in zip(blocks, slices) if blk.scaling_class == "scalar"]
    if equil == "sgd" and all(getattr(t.op, "well_conditioned", False) for t in dual):
        equil = "scalar"                     # the engine's own operator-aware skip
    if equil == "scalar":
        t0, s0, nrm = scalar_steps(matvec, rmatvec, n)
        tau = jnp.full((n,), t0, jnp.float32)
        sigma = jnp.full((m,), s0, jnp.float32)
    elif equil == "sgd":
        tau, sigma, nrm, _d = smfe_equilibration(
            matvec, rmatvec, n, m, scalar_spans, diagnostics=True,
            probes=ck.get("smfe_probes", 2), iters=ck.get("smfe_iters", 80),
            schedule=ck.get("smfe_schedule", "sgd"))
        tau = jnp.broadcast_to(jnp.asarray(tau, jnp.float32), (n,))
        sigma = jnp.broadcast_to(jnp.asarray(sigma, jnp.float32), (m,))
    else:
        raise NotImplementedError(f"equilibrate={equil!r} is outside this study")
    if g_block is not None and g_block.scaling_class == "scalar":
        tau = jnp.full((n,), jnp.min(tau), jnp.float32)
    return dict(tau=tau, sigma=sigma, scaled_norm=float(nrm), n=n, m=m,
                ops=ops, blocks=blocks, slices=slices, dual=dual, g_block=g_block,
                matvec=matvec, rmatvec=rmatvec)


def problem_data(prob, dual, slices, m, data):
    """The offsets l (stacked, as the engine stacks them) and the cost c."""
    import jax.numpy as jnp
    l = np.zeros(m, np.float32)
    for t, (s, e) in zip(dual, slices):
        if t.offset_fn is not None:
            l[s:e] = np.asarray(t.offset_fn(data), np.float32).reshape(-1)
    c = (np.zeros(prob.n, np.float32) if prob.cost_fn is None
         else np.asarray(prob.cost_fn(data), np.float32).reshape(-1))
    return jnp.asarray(l), jnp.asarray(c)


# ----------------------------------------------------------------- estimators
def setup_estimates(prob, data, ck, tally=None):
    """E1, E2, E5: everything computable at setup with no solve.

    Returns raw values and a record of the operator applications charged.
    """
    import jax.numpy as jnp
    t0 = time.perf_counter()
    eq = equilibrate_like_compile(prob, data, ck, tally=tally)
    equil_s = time.perf_counter() - t0
    equil_fwd = dict(tally)["fwd"] if tally is not None else None
    tau, sigma, n, m = eq["tau"], eq["sigma"], eq["n"], eq["m"]
    l, c = problem_data(prob, eq["dual"], eq["slices"], m, data)

    t1 = time.perf_counter()
    ls = jnp.sqrt(sigma) * l
    cs = jnp.sqrt(tau) * c
    nl, nc = float(jnp.linalg.norm(ls)), float(jnp.linalg.norm(cs))
    w_pdlp = (nc / nl) if (nc > 0.0 and nl > 0.0) else 1.0

    # E2: prox-anchor subgradient at the offset, one adjoint application
    one = jnp.asarray(1.0, jnp.float32)
    parts = []
    for blk, (s, e) in zip(eq["blocks"], eq["slices"]):
        li = l[s:e]
        parts.append(li - blk.prox(li, one))
    yref = jnp.concatenate(parts)
    q = c + eq["rmatvec"](yref)
    nq = float(jnp.linalg.norm(jnp.sqrt(tau) * q))
    w_obj = (nq / nl) if (nq > 0.0 and nl > 0.0) else 1.0

    # E5: canonical gauge, zero operator work
    lw = (float(jnp.sum(jnp.log(tau))) - float(jnp.sum(jnp.log(sigma)))) / (m + n)
    w_equil = math.exp(lw)
    setup_s = time.perf_counter() - t1

    est = {}
    est["pdlp"] = dict(raw=w_pdlp, cost_ops=0, cost_s=setup_s, duplicated_equil_s=equil_s,
                       note="marginal cost 0: tau0, sigma0 are computed by compile() anyway. "
                            "This prototype recomputes them outside the engine, which is the "
                            "duplicated_equil_s / 382 applications reported separately.")
    est["objscale"] = dict(raw=w_obj, cost_ops=1, cost_s=setup_s, duplicated_equil_s=equil_s,
                           note="marginal cost 1 adjoint application + 1 prox per dual block")
    est["equil"] = dict(raw=w_equil, cost_ops=0, cost_s=setup_s, duplicated_equil_s=equil_s,
                        note="marginal cost 0: reuses tau0, sigma0 only")
    diag = dict(n=n, m=m, scaled_norm=eq["scaled_norm"], equil_s=equil_s,
                norm_l_scaled=nl, norm_c_scaled=nc, norm_q_scaled=nq,
                norm_l_raw=float(jnp.linalg.norm(l)), norm_c_raw=float(jnp.linalg.norm(c)),
                geomean_tau=math.exp(float(jnp.mean(jnp.log(tau)))),
                geomean_sigma=math.exp(float(jnp.mean(jnp.log(sigma)))),
                equil_traced_fwd=equil_fwd,
                equil_ops_analytic=382,
                equil_ops_note=("the shipped SMFE is ONE jitted program, so a Python-level "
                                "counter sees traces, not executions; the runtime count is "
                                "80 iters x 2 probes x (fwd+adj) = 320, + 15 power iterations "
                                "(30), + 2 x 8 diagnostic probes (32) = 382 applications, "
                                "which compile() pays anyway"))
    return est, diag


def probe_estimates(prob, data, ck, tol):
    """E3/E4 at P = 100 and the same two rules at P = 1000.

    One compiled executable at w = 1 is driven in blocks of PROBE_ITERS with warm
    starts, so the short and the long probe share a compile and the long probe is
    exactly the short one continued.  Reported separately because the probe length
    is a constant of the rule, and a rule with a longer probe is a DIFFERENT rule
    with a different cost, not a tuned version of the same one.
    """
    import jax, jax.numpy as jnp
    t0 = time.perf_counter()
    comp = prob.compile(block=PROBE_ITERS, max_blocks=1, tol=1e-30,
                        primal_weight=1.0, adaptive_omega=False, **ck)
    compile_s = time.perf_counter() - t0

    def read(x, y, iters, solve_s, suffix):
        nx = float(jnp.linalg.norm(x)); ny = float(jnp.linalg.norm(y))
        cert = {k: float(v) for k, v in comp.certificate(x, y, data).items()
                if np.ndim(v) == 0}
        w_move = (ny / nx) if nx > 0 else 1.0
        rp, rd = cert.get("r_primal", float("nan")), cert.get("r_dual", float("nan"))
        w_res = (math.sqrt(rd / rp) if (rp > 0 and math.isfinite(rp)
                                        and rd > 0 and math.isfinite(rd)) else 1.0)
        est = {f"probe_move{suffix}": dict(raw=w_move, cost_ops=2 * iters + 2, cost_s=solve_s,
                                           note=f"{iters} iterations at w=1, from a cold start"),
               f"probe_resid{suffix}": dict(raw=w_res, cost_ops=0, cost_s=0.0,
                                            note=f"shares the {iters}-iteration probe")}
        diag = {f"probe{suffix}_iters": iters, f"probe{suffix}_solve_s": solve_s,
                f"probe{suffix}_norm_x": nx, f"probe{suffix}_norm_y": ny}
        diag.update({f"probe{suffix}_{k}": v for k, v in cert.items()})
        return est, diag

    t1 = time.perf_counter()
    sol = comp.solve_instance(data)
    jax.block_until_ready((sol.x, sol.y))
    short_s = time.perf_counter() - t1
    est, diag = read(sol.x, sol.y, int(sol.iters), short_s, "")

    x, y, iters, elapsed = sol.x, sol.y, int(sol.iters), short_s
    while iters < PROBE_ITERS_LONG:
        t2 = time.perf_counter()
        sol = comp.solve_instance(data, warm_start=(x, y))
        jax.block_until_ready((sol.x, sol.y))
        elapsed += time.perf_counter() - t2
        x, y, iters = sol.x, sol.y, iters + int(sol.iters)
    e2, d2 = read(x, y, iters, elapsed, "_long")
    est.update(e2); diag.update(d2)
    diag["probe_compile_s"] = compile_s
    diag["probe_scaled_norm"] = float(comp.meta["scaled_norm"])
    del comp
    return est, diag


# ---------------------------------------------------------------------- arms
def solve_at(prob, data, ck, w, *, tol, block, max_blocks, max_seconds, repeats=1,
             log=print):
    """One arm: compile at fixed weight w and solve cold, `repeats` times."""
    import jax
    t0 = time.perf_counter()
    comp = prob.compile(block=block, max_blocks=max_blocks, tol=tol,
                        primal_weight=float(w), adaptive_omega=False, **ck)
    compile_s = time.perf_counter() - t0
    rows = []
    for r in range(repeats):
        t1 = time.perf_counter()
        sol = comp.solve_instance(data)
        jax.block_until_ready((sol.x, sol.y))
        dt = time.perf_counter() - t1
        rows.append(dict(rep=r, iters=int(sol.iters), status=str(sol.status),
                         residual=float(sol.residual), solve_s=dt))
        log(f"    w={w:<12.6g} rep{r}: {int(sol.iters):8d} it  {str(sol.status):12s} "
            f"r={float(sol.residual):.4e}  {dt:8.2f}s")
        if sum(x["solve_s"] for x in rows) > max_seconds and r + 1 < repeats:
            log(f"    (stopping repeats at {r + 1}: wall budget {max_seconds}s)")
            break
    cert = {k: float(v) for k, v in comp.certificate(sol.x, sol.y, data).items()
            if np.ndim(v) == 0}
    warm = [x["solve_s"] for x in rows[1:]] or [rows[0]["solve_s"]]
    out = dict(w=float(w), compile_s=compile_s, iters=rows[-1]["iters"],
               status=rows[-1]["status"], residual=rows[-1]["residual"],
               first_solve_s=rows[0]["solve_s"], warm_med_s=float(np.median(warm)),
               warm_min_s=float(min(warm)), warm_max_s=float(max(warm)),
               repeats=len(rows), reps=rows, certificate=cert,
               x_norm=float(np.linalg.norm(np.asarray(sol.x, np.float64))),
               y_norm=float(np.linalg.norm(np.asarray(sol.y, np.float64))))
    del comp
    return out


# ------------------------------------------------------------------- families
def get_family(name):
    """(prob, data, compile_kwargs, note).  Compile kwargs are each family's own
    recorded ones, except that pdhg_variant is forced to 'standard' for every
    family so the weight is measured in one drive mode (the mode the
    multicommodity sweep used, and the mode the unified study calls A0std)."""
    import ocs, jax.numpy as jnp
    mcf_dir = os.environ.get("WI_MCF200K", os.path.expanduser("~/mcf_restart"))
    mcf20m_dir = os.environ.get("WI_MCF20M", os.path.expanduser("~/mcf_restart/inputs20m"))
    if name.startswith("mcf"):
        d = mcf_dir if name == "mcf200k" else mcf20m_dir
        N = np.load(os.path.join(d, "N.npy")); dd = np.load(os.path.join(d, "d.npy"))
        slopes = np.load(os.path.join(d, "slopes.npy"))
        ints = np.load(os.path.join(d, "intercepts.npy"))
        sqrt_mu = np.load(os.path.join(d, "sqrt_mu.npy"))
        V, E = N.shape; K = dd.shape[1]; n = E * K
        x = ocs.Variable(n); rhs = ocs.Parameter("rhs")
        prob = ocs.minimize([
            ocs.nonneg(x),
            ocs.sum_squares(ocs.Diagonal(sqrt_mu * np.ones(n, np.float32)) @ x),
            ocs.equality(ocs.MultiCommodityFlow(N, K) @ x - rhs),
            ocs.pwl(ocs.CommoditySum(E, K) @ x, slopes, ints)])
        return prob, {"rhs": jnp.asarray(dd.reshape(-1))}, {}, f"{name} n={n}"
    if name.startswith("f1"):
        from blockmetric_models import blockmetric_f1
        rho = float(name.split("rho")[1])
        prob, spec = blockmetric_f1(n=200_000, rho=rho, k=4, variant="overlap", seed=0)
        return prob, spec.data, {}, spec.name
    if name.startswith("f2"):
        from blockmetric_models import blockmetric_f2
        L = int(name.split("L")[1])
        prob, spec = blockmetric_f2(L=L, mixing="strong", seed=0)
        return prob, spec.data, {}, spec.name
    if name.startswith("f3"):
        from blockmetric_models import blockmetric_f3
        L = int(name.split("L")[1])
        prob, spec = blockmetric_f3(L=L, seed=0)
        return prob, spec.data, {}, spec.name
    if name.startswith("ot"):
        from gate_models import ot_problem
        N = int(name.split("-")[1])
        prob, d, _check, tag, ck, w_hint = ot_problem(N)
        ck = dict(ck); ck.pop("pdhg_variant", None)
        return prob, d, ck, f"{tag} (family's own recorded weight hint {w_hint})"
    if name.startswith("featsel"):
        from gate_models import featsel_problem
        F = int(name.split("F")[1].split("T")[0]); T = int(name.split("T")[1])
        prob, d, _check, tag, ck, _w = featsel_problem(m=600, F=F, T=T, k_active=12)
        ck = dict(ck); ck.pop("pdhg_variant", None)
        return prob, d, ck, tag
    raise ValueError(f"unknown family {name!r}")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--mode", default="all", choices=["estimate", "sweep", "arms", "all"])
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--max-blocks", type=int, default=1000)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sweep-repeats", type=int, default=1)
    ap.add_argument("--sweep", default="", help="comma list of weights for --mode sweep")
    ap.add_argument("--arms", default="", help="extra comma list of weights for --mode arms")
    ap.add_argument("--purity", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax, ocs
    log = lambda s: print(s, flush=True)
    prob, data, ck, note = get_family(args.family)
    ck = dict(ck); ck.setdefault("pdhg_variant", "standard")
    log(f"# host {os.uname().nodename} device {jax.devices()[0].device_kind} "
        f"jax {jax.__version__} ocs {ocs.__version__} at {ocs.__file__}")
    log(f"# family {args.family}: {note}  n={prob.n} compile_kwargs={ck}")
    log(f"# tol {args.tol} block {args.block} max_blocks {args.max_blocks} "
        f"budget {args.block * args.max_blocks} iterations / {args.max_seconds}s per arm")
    out = dict(meta=dict(family=args.family, note=note, n=prob.n, compile_kwargs=ck,
                         tol=args.tol, block=args.block, max_blocks=args.max_blocks,
                         budget_iters=args.block * args.max_blocks,
                         max_seconds=args.max_seconds, repeats=args.repeats,
                         probe_iters=PROBE_ITERS, clip=[W_LO, W_HI],
                         host=os.uname().nodename, device=jax.devices()[0].device_kind,
                         jax=jax.__version__, ocs=ocs.__version__, ocs_file=ocs.__file__,
                         time=time.strftime("%Y-%m-%d %H:%M:%S %Z")))

    if args.mode in ("estimate", "all"):
        t0 = time.perf_counter()
        est, diag = setup_estimates(prob, data, ck)
        pe, pdiag = probe_estimates(prob, data, ck, args.tol)
        est.update(pe); diag.update(pdiag)
        diag["estimate_total_s"] = time.perf_counter() - t0
        for k in ESTIMATORS:
            w, flag = _clip(est[k]["raw"])
            est[k]["w"] = w; est[k]["clip"] = flag
            log(f"  estimator {k:12s} raw={est[k]['raw']:.6g}  w={w:.6g} {flag} "
                f"[{est[k]['cost_ops']} ops, {est[k]['cost_s']:.3f}s]")
        if args.purity:
            tally = {"fwd": 0, "adj": 0}
            est_p, diag_p = setup_estimates(prob, data, ck, tally=tally)
            ok = all(abs(est_p[k]["raw"] - est[k]["raw"]) <=
                     1e-4 * max(1e-30, abs(est[k]["raw"])) for k in ("pdlp", "objscale", "equil"))
            log(f"  purity: fwd={tally['fwd']} adj={tally['adj']} identical={ok}")
            diag["purity"] = dict(fwd=tally["fwd"], adj=tally["adj"], identical=bool(ok),
                                  raw={k: est_p[k]["raw"] for k in ("pdlp", "objscale", "equil")})
        out["estimates"] = est; out["diagnostics"] = diag
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    weights = []
    if args.mode in ("sweep", "all") and args.sweep:
        weights += [float(w) for w in args.sweep.split(",")]
    if args.mode in ("arms", "all"):
        if args.arms:
            weights += [float(w) for w in args.arms.split(",")]
        if "estimates" in out:
            weights += [out["estimates"][k]["w"] for k in ESTIMATORS]
    sweep_only = {round(math.log(float(w)), 6) for w in
                  ([float(x) for x in args.sweep.split(",") if x]
                   if args.mode in ("sweep", "all") else [])}
    claim = set()
    if args.mode in ("arms", "all"):
        claim |= {round(math.log(float(x)), 6) for x in args.arms.split(",") if x}
        if "estimates" in out:
            claim |= {round(math.log(out["estimates"][k]["w"]), 6) for k in ESTIMATORS}
    seen, uniq = set(), []
    for w in weights:
        key = round(math.log(float(w)), 6)
        if key not in seen:
            seen.add(key); uniq.append(float(w))
    if uniq:
        out["arms"] = {}
        for w in uniq:
            log(f"  arm w={w:.6g}")
            key = round(math.log(w), 6)
            reps = args.repeats if key in claim else args.sweep_repeats
            try:
                out["arms"][f"{w:.6g}"] = solve_at(
                    prob, data, ck, w, tol=args.tol, block=args.block,
                    max_blocks=args.max_blocks, max_seconds=args.max_seconds,
                    repeats=reps, log=log)
            except Exception as exc:           # a failed arm is a reported failure
                log(f"    FAILED: {type(exc).__name__}: {exc}")
                out["arms"][f"{w:.6g}"] = dict(w=float(w), status="error",
                                               error=f"{type(exc).__name__}: {exc}")
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1, default=str)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
