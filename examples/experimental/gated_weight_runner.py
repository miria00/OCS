"""Restart-gated primal weight, as a runner outside the package.

The shipped engine is untouched.  It is compiled at a fixed primal weight
(``compile(primal_weight=w)``: tau = tau0 / w, sigma = sigma0 * w, fixed steps)
and driven in epochs of ``epoch_blocks`` blocks with warm starts.  Between
epochs the runner reads the returned point, its certificate, and the distance
the iterates moved, and applies the PDLP restart gate to the KKT residual:

    sufficient decay    r_k <= beta_suff * r_restart
    necessary decay     r_k <= beta_nec  * r_restart   and   r_k > r_prev
    artificial restart  epochs since the last restart >= artificial * total epochs

At a gated restart the weight moves to PDLP's log-smoothed movement ratio

    w <- exp( theta log(dy / dx) + (1 - theta) log w ),
    dx = ||x_k - x_restart||_2,   dy = ||y_k - y_restart||_2,

(dual movement over primal movement, the PDLP primal weight with tau = eta / w
and sigma = eta w, which is OCS's ``primal_weight`` convention),

quantized to a factor-of-two grid and clipped to [w0 / 100, w0 * 100].  The
compiled solver is cached per grid value, so the number of recompiles is
bounded by the number of distinct weights visited.  Everything the gate
consumes is norms of iterates and the problem-based certificate: no entry of
any operator is read.  The baseline is the same epoch loop with the gate off.

Optional summable budget (``--budget B``, in log2 units).  The j-th accepted
change may move the weight by at most a factor 2**b_j with b_j = B * decay**j,
so sum_j b_j = B / (1 - decay) is finite: the total multiplicative movement of
the metric is bounded and the weight converges.  This is the variable-metric
condition of Combettes and Vu (U_{n+1} <= (1 + eta_n) U_n with summable eta_n)
specialized to a scalar weight, and the bounded-deterioration form of Bonnans,
Gilbert, Lemarechal and Sagastizabal (1995).  On the factor-of-two grid the
allowed exponent is floored; once it drops below 1 the weight is frozen.
PDLP's primal weight and OSQP's adaptive rho have no such budget.

Run (GPU box):
    python examples/experimental/gated_weight_runner.py --problem turf --mode server \\
        --tasks 10000 --servers 250 --tol 1e-5 --gate on --out gate_server_10k.csv
    python examples/experimental/gated_weight_runner.py --problem imaging --size 128 --gate off
"""
import os, sys, time, math, csv, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # examples/ for the model builders


# ----------------------------------------------------------------------------- models
def turf_problem(tasks, servers, mode, seed=0, load=1.35, utility="linear"):
    """The CvxCluster Stage-1 LP of examples/cvxcluster_turf.py, as (prob, data)."""
    import ocs
    from cvxcluster_turf import instance, lp_arrays, independent
    inst = instance(tasks, servers, mode, load, seed)
    t, B = inst["tasks"], inst["bins"]
    rhs, cost = lp_arrays(inst)
    x = ocs.Variable(t * B); bp = ocs.Parameter("rhs"); cp_ = ocs.Parameter("c")
    if utility != "linear":
        raise NotImplementedError("the runner covers the linear utility; see cvxcluster_turf.py for PWL")
    K = ocs.BinPack(inst["Rmat"], B)
    prob = ocs.minimize([ocs.nonneg(x), ocs.nonneg(ocs.Weighted(K, -1.) @ x + bp)], cost=cp_)
    data = {"rhs": rhs, "c": cost}

    def check(xv):
        X = np.asarray(xv, np.float64).reshape(t, B)
        try:
            return independent(inst, X, utility)
        except TypeError:                          # older cvxcluster_turf: independent(inst, X)
            return independent(inst, X)
    return prob, data, check, f"turf-{mode}-{tasks}x{servers}"


def imaging_problem(L, lam=0.02, ratio=0.4, seed=0):
    """The 3-D imaging model of examples/killer_3d_imaging.py, as (prob, data)."""
    import ocs, jax.numpy as jnp
    n = L ** 3
    vol = np.zeros((L, L, L), np.float32)
    vol[8:28, 10:34, 12:30] = 1.0
    vol[20:36, 6:20, 18:38] += 0.5
    phantom = vol.reshape(-1)
    F = ocs.FFTMask.random(n, ratio=ratio, seed=seed)
    G3 = ocs.Gradient3D(L, L, L)
    y = np.asarray(F.matvec(jnp.asarray(phantom)))
    x = ocs.Variable(n); yp = ocs.Parameter("y")
    prob = ocs.minimize([ocs.nonneg(x), ocs.least_squares(F @ x - yp),
                         ocs.group_l2(G3 @ x, lam=lam, group_size=3)])
    data = {"y": jnp.asarray(y)}

    def check(xv):
        xv = np.asarray(xv, np.float64)
        return {"rel_error_vs_phantom": float(np.linalg.norm(xv - phantom) / np.linalg.norm(phantom)),
                "min_x": float(xv.min())}
    return prob, data, check, f"imaging-{L}"


# ----------------------------------------------------------------------------- runner
class GatedWeightRunner:
    def __init__(self, prob, data, *, w0=1.0, epoch_blocks=10, block=100, tol=1e-5,
                 gate=True, beta_suff=0.2, beta_nec=0.8, artificial=0.36, theta=0.5,
                 grid_base=2.0, clip=100.0, budget=None, budget_decay=0.5,
                 compile_kwargs=None, log=print, epoch_extra=None):
        self.budget, self.budget_decay = (None if budget is None else float(budget)), float(budget_decay)
        self.prob, self.data = prob, data
        self.epoch_extra = epoch_extra                # optional host diagnostic per epoch, outside the timed region
        self.w0, self.epoch_blocks, self.block, self.tol = float(w0), int(epoch_blocks), int(block), float(tol)
        self.gate = bool(gate)
        self.beta_suff, self.beta_nec, self.artificial, self.theta = beta_suff, beta_nec, artificial, theta
        self.grid_base, self.clip = float(grid_base), float(clip)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.log = log
        self._cache = {}
        self._warm = set()                            # weights whose compiled solver has already run once
        self.compile_s = 0.0

    def _quantize(self, w):
        k = round(math.log(w / self.w0, self.grid_base))
        w = self.w0 * self.grid_base ** k
        return float(min(max(w, self.w0 / self.clip), self.w0 * self.clip))

    def compiled(self, w):
        """The shipped engine at fixed weight w, cached per grid value."""
        if w not in self._cache:
            t0 = time.perf_counter()
            self._cache[w] = self.prob.compile(block=self.block, max_blocks=self.epoch_blocks,
                                               tol=self.tol, primal_weight=w,
                                               adaptive_omega=False, **self.compile_kwargs)
            self.compile_s += time.perf_counter() - t0
            self.log(f"   compiled w={w:g} ({time.perf_counter() - t0:.1f} s)")
        return self._cache[w]

    def run(self, max_epochs=200, max_seconds=3600.0):
        import jax
        rows, w = [], self._quantize(self.w0)
        x = y = None
        r_restart = r_prev = math.inf
        x_restart = y_restart = None
        last_restart_epoch = 0
        changes = 0                                   # accepted weight changes so far (budget schedule index)
        iters_total, t_start = 0, time.perf_counter()
        epochs_s = 0.0                                # solve-only clock: the sum of the timed epochs
        status = "budget"
        for epoch in range(1, max_epochs + 1):
            first_call = w not in self._warm            # the first solve at a weight carries its XLA compile
            comp = self.compiled(w)
            t0 = time.perf_counter()
            sol = (comp.solve_instance(self.data) if x is None
                   else comp.solve_instance(self.data, warm_start=(x, y)))
            jax.block_until_ready((sol.x, sol.y))
            epoch_s = time.perf_counter() - t0
            epochs_s += epoch_s
            self._warm.add(w)
            iters_total += int(sol.iters)
            r_k = float(sol.residual)
            xn, yn = sol.x, sol.y
            if x_restart is None:
                x_restart, y_restart, r_restart = xn, yn, r_k
            dx = float(jax.numpy.linalg.norm(xn - x_restart))
            dy = float(jax.numpy.linalg.norm(yn - y_restart))
            decision = ""
            if self.gate and epoch > 1:
                since = epoch - last_restart_epoch
                if r_k <= self.beta_suff * r_restart:
                    decision = "sufficient"
                elif r_k <= self.beta_nec * r_restart and r_k > r_prev:
                    decision = "necessary"
                elif since >= max(2, int(self.artificial * epoch)):
                    decision = "artificial"
                if decision:
                    if dx > 0 and dy > 0 and math.isfinite(dx) and math.isfinite(dy):
                        w_raw = math.exp(self.theta * math.log(dy / dx) + (1 - self.theta) * math.log(w))
                        w_new = self._quantize(w_raw)
                    else:
                        w_raw, w_new = w, w
                    if self.budget is not None and w_new != w:
                        allowed = self.budget * self.budget_decay ** changes      # log2 units for this change
                        steps = math.floor(allowed + 1e-12)
                        if steps < 1:
                            decision += " frozen (budget spent)"
                            w_new = w
                        else:
                            want = round(math.log(w_new / w, self.grid_base))
                            capped = max(-steps, min(steps, want))
                            if capped != want:
                                decision += f" capped {want:+d}->{capped:+d} grid steps (allowed {allowed:.2f})"
                            w_new = self._quantize(w * self.grid_base ** capped)
                    if w_new != w:
                        changes += 1
                    decision += f" w {w:g}->{w_new:g} (raw {w_raw:.3g})"
                    w = w_new
                    x_restart, y_restart, r_restart = xn, yn, r_k
                    last_restart_epoch = epoch
            rows.append(dict(epoch=epoch, w=w, iters=iters_total, residual=r_k, dx=dx, dy=dy,
                             epoch_s=round(epoch_s, 3), epochs_s=round(epochs_s, 3), first_call=first_call,
                             elapsed_s=round(time.perf_counter() - t_start, 2),
                             decision=decision, status=sol.status))
            if self.epoch_extra is not None:          # host diagnostic, not timed, never fed back to the gate
                try:
                    rows[-1].update(self.epoch_extra(xn, yn))
                except Exception as exc:
                    rows[-1]["extra_error"] = f"{type(exc).__name__}: {exc}"[:120]
            self.log(f"epoch {epoch:4d} w={w:<8g} it={iters_total:8d} r={r_k:.3e} dx={dx:.3e} dy={dy:.3e} {decision}")
            x, y, r_prev = xn, yn, r_k
            if sol.status == "solved" and r_k <= self.tol:
                status = "solved"; break
            if sol.status in ("primal_infeasible", "unbounded", "numerical_error"):
                status = sol.status; break
            if time.perf_counter() - t_start > max_seconds:
                status = "time cap"; break
        return dict(status=status, iterations=iters_total, residual=r_k, w_final=w,
                    weight_changes=changes, budget=self.budget,
                    weights_visited=sorted(self._cache), compile_s=round(self.compile_s, 2),
                    solve_s=round(time.perf_counter() - t_start, 2), epochs_s=round(epochs_s, 3),
                    epochs=len(rows), x=x, y=y, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True,
                    choices=["turf", "imaging", "ot", "fft", "mri", "kron", "sylvester", "rof", "featsel"])
    ap.add_argument("--mode", default="server", choices=["shape", "global", "server"])
    ap.add_argument("--tasks", type=int, default=10000); ap.add_argument("--servers", type=int, default=250)
    ap.add_argument("--size", type=int, default=128, help="imaging L, mri L, fft n, ot N, kron T")
    ap.add_argument("--fs-m", type=int, default=600); ap.add_argument("--fs-F", type=int, default=10000)
    ap.add_argument("--fs-T", type=int, default=10000)
    ap.add_argument("--p", type=int, default=60); ap.add_argument("--q", type=int, default=50)
    ap.add_argument("--rof-data", default=None, help="npy of the noisy image (built through odl when absent)")
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--epoch-blocks", type=int, default=10); ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--gate", choices=["on", "off"], default="on")
    ap.add_argument("--budget", type=float, default=None, help="total log2 weight movement B; b_j = B*decay^j per accepted change")
    ap.add_argument("--budget-decay", type=float, default=0.5)
    ap.add_argument("--max-epochs", type=int, default=300); ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--epoch-check", action="store_true", help="run the independent check after every epoch (host, untimed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import jax
    if args.problem == "turf":
        prob, data, check, name = turf_problem(args.tasks, args.servers, args.mode)
        ck = {"equilibrate": "sgd", "residual_norm": "relative"}
        if args.compact:
            ck["data_layout"] = "compact"
    elif args.problem == "imaging":
        prob, data, check, name = imaging_problem(args.size)
        ck = {}
    else:
        import gate_models as gm
        build = {"ot": lambda: gm.ot_problem(args.size),
                 "fft": lambda: gm.fft_problem(args.size),
                 "mri": lambda: gm.mri_problem(args.size),
                 "kron": lambda: gm.kron_problem(args.size),
                 "sylvester": lambda: gm.sylvester_problem(args.p, args.q),
                 "rof": lambda: gm.rof_problem(args.rof_data),
                 "featsel": lambda: gm.featsel_problem(args.fs_m, args.fs_F, args.fs_T)}[args.problem]
        prob, data, check, name, ck, _paper_w = build()
        if args.compact:
            ck["data_layout"] = "compact"
    import ocs
    print(f"{name}: gate {args.gate}, budget {args.budget} (decay {args.budget_decay}), w0 {args.w0}, "
          f"epoch {args.epoch_blocks} blocks of {args.block}, tol {args.tol}, compile {ck}, "
          f"device {jax.devices()[0].device_kind}, jax {jax.__version__}, ocs {ocs.__version__} at {ocs.__file__}", flush=True)

    def run_check(xv, yv):
        try:
            return check(xv, yv)
        except TypeError:                          # older builders: check(x)
            return check(xv)
    extra = (lambda xv, yv: run_check(xv, yv)) if args.epoch_check else None
    runner = GatedWeightRunner(prob, data, w0=args.w0, epoch_blocks=args.epoch_blocks, block=args.block,
                               tol=args.tol, gate=(args.gate == "on"), budget=args.budget,
                               budget_decay=args.budget_decay, compile_kwargs=ck, epoch_extra=extra)
    res = runner.run(max_epochs=args.max_epochs, max_seconds=args.max_seconds)
    summary = {k: v for k, v in res.items() if k not in ("x", "y", "rows")}
    if args.out:
        fields = []
        for r in res["rows"]:
            fields += [k for k in r if k not in fields]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(res["rows"])
        print(f"wrote {args.out}", flush=True)
    try:
        summary.update(run_check(res["x"], res["y"]) if res["x"] is not None else {})
    except Exception as exc:                       # the independent check is diagnostic
        summary["check_error"] = f"{type(exc).__name__}: {exc}"[:200]
    print("RESULT", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()}, flush=True)
    if args.out:
        import json
        with open(args.out.replace(".csv", "") + "_summary.txt", "w") as fh:
            fh.write(f"{name} gate={args.gate}\n{summary}\n")
        meta = dict(name=name, problem=args.problem, gate=args.gate, w0=args.w0, tol=args.tol, budget=args.budget,
                    epoch_blocks=args.epoch_blocks, block=args.block, compile_kwargs=ck,
                    max_epochs=args.max_epochs, max_seconds=args.max_seconds,
                    device=jax.devices()[0].device_kind, jax=jax.__version__, ocs=ocs.__version__,
                    ocs_file=ocs.__file__, size=args.size, tasks=args.tasks, servers=args.servers, mode=args.mode,
                    time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        with open(args.out.replace(".csv", "") + "_summary.json", "w") as fh:
            json.dump({"meta": meta, "summary": summary}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
