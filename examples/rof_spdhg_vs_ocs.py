"""Regime-3 head-to-head: OCS against the authors' own SPDHG on ROF denoising.

The specialized method is SPDHG [CERS18], run from the reference implementation
in odl.contrib.solvers.spdhg, with the problem, the image, the noise seed and
every step-size constant taken from Ehrhardt's own example_rof.py
(github.com/mehrhardt/spdhg).  OCS solves the identical problem through the
same operator, accessed only by matvec and rmatvec.

Common objective, on ODL's scale:
    J(x) = ||D_r x||_1 + ||D_c x||_1 + (1/(2 alpha)) ||x - b||_2^2

ODL's PartialDerivative(method='forward', pad_mode='symmetric') on a unit-cell
grid is exactly OCS's Gradient2D: forward differences with Neumann boundary.
The two solvers therefore see the same problem, not two similar ones.

Emits plots/rof_spdhg_vs_ocs.csv with schema
    method,unit,work,wall_s,objective,rel_gap
"""
import os, sys, time, csv
import numpy as np

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.path.join(PLOTS, "rof_spdhg_vs_ocs.csv")

ALPHA = 0.12          # Ehrhardt example_rof.py
GAMMA = 0.99          # gamma^2 bounds the step-size product
SEED = 1807
STDDEV = 0.1


def to_numpy(v):
    """Unwrap an ODL 1.0 space element, a JAX array or a plain array."""
    if isinstance(v, (list, tuple)):
        v = v[0]
    v = getattr(v, "data", v)
    if hasattr(v, "detach"):                      # torch tensor, possibly on GPU
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def objective(x, b, H, W, alpha=ALPHA):
    """Evaluated by one piece of code for both solvers, so the number compared
    is the same number."""
    img = to_numpy(x).reshape(H, W)
    d_r = np.zeros_like(img); d_r[:-1, :] = img[1:, :] - img[:-1, :]
    d_c = np.zeros_like(img); d_c[:, :-1] = img[:, 1:] - img[:, :-1]
    fid = np.sum((img - to_numpy(b).reshape(H, W)) ** 2)
    return float(np.abs(d_r).sum() + np.abs(d_c).sum() + fid / (2.0 * alpha))


def build_problem(impl="numpy", device=None):
    import odl, odl.contrib.datasets.images as images
    gt = images.building(gray=True)
    H, W = gt.shape
    kw = {"impl": impl}
    if device:
        kw["device"] = device
    X = odl.uniform_discr([0, 0], [H, W], [H, W], **kw)
    truth = X.element(gt)
    # odl.core.phantom.white_noise trips over numpy 2's __array__ signature for a
    # plain space element.  Its body is exactly the two lines below, so this
    # reproduces Ehrhardt's data bit for bit.
    np.random.seed(SEED)
    b_np = np.random.normal(loc=to_numpy(gt),
                            scale=STDDEV, size=(H, W))
    data = X.element(b_np)
    return X, data, b_np, H, W


def run_spdhg(X, data, b_np, H, W, nepoch, sync=None):
    """Ehrhardt's reference SPDHG, his parameters, unmodified."""
    import odl, odl.contrib.solvers.spdhg as spdhg
    A = odl.BroadcastOperator(*[odl.PartialDerivative(X, d, pad_mode="symmetric")
                                for d in (0, 1)])
    f = odl.functionals.SeparableSum(*[odl.functionals.L1Norm(Yi) for Yi in A.range])
    g = 1 / (2 * ALPHA) * odl.functionals.L2NormSquared(X).translated(data)

    n = 2                                    # two subsets, one per derivative
    prob = [1 / n] * n
    rng = np.random.default_rng(0)
    Ai_norm = [2, 2]                         # ||D||<=2 per direction
    sigma = [GAMMA / a for a in Ai_norm]
    tau = GAMMA / (n * max(Ai_norm))

    trace, clock = [], {"t": 0.0, "last": None}

    class Rec:
        """Records (work, wall, objective) at epoch boundaries and keeps its own
        cost out of the measured time."""
        def __init__(self): self.k = 0
        def __call__(self, xy):
            self.k += 1
            if self.k % n:
                return
            if sync:
                sync()
            clock["t"] += time.perf_counter() - clock["last"]
            trace.append((self.k // n, clock["t"], objective(xy, b_np, H, W)))
            clock["last"] = time.perf_counter()

    x = X.zero()
    clock["last"] = time.perf_counter()
    spdhg.spdhg(x, f, g, A, tau, sigma, n * nepoch, prob=prob,
                fun_select=lambda k: [int(rng.choice(n, p=prob))], callback=Rec())
    if sync:
        sync()
    clock["t"] += time.perf_counter() - clock["last"]
    return trace


def run_split_bregman(b_np, H, W, budgets):
    """Split Bregman [Goldstein-Osher 2009].  The authors' own release is not
    retrievable (their UMD links are dead and the Wayback Machine never captured
    them), so this is scikit-image's maintained implementation of their method.

    scikit-image's docstring says weight equals lambda/2, but a measured scan
    over the weight shows its objective is  sum_i |grad u_i| + (weight/2)(f_i -
    u_i)^2, so weight = 1/alpha is the setting that matches J; 1/(2 alpha) lands
    4.6 percent high.  isotropic=False gives the anisotropic TV that J uses.  Its discretization is its own, so we check
    empirically whether it reaches the same optimum rather than assuming it.
    """
    from skimage.restoration import denoise_tv_bregman
    out = []
    for nit in budgets:
        t0 = time.perf_counter()
        u = denoise_tv_bregman(b_np.reshape(H, W), weight=1.0 / ALPHA,
                               max_num_iter=int(nit), eps=1e-12, isotropic=False)
        wall = time.perf_counter() - t0
        J = objective(u, b_np, H, W)
        out.append((int(nit), wall, J))
        print(f"  SplitBregman {nit:6d} it  {wall:8.3f}s  J={J:.6f}", flush=True)
    return out


def run_ocs(b_np, H, W, budgets, block=100):
    """One independent OCS run per budget: compile, warm up, then time.
    Compilation is reported separately, never folded into the solve time."""
    import jax, jax.numpy as jnp, ocs
    n = H * W
    out = []
    for nit in budgets:
        mb = max(1, nit // block)
        x = ocs.Variable(n); bp = ocs.Parameter("b"); Gr = ocs.Gradient2D(H, W)
        # OCS minimizes alpha*J, an identical minimizer; J is recomputed for reporting.
        comp = ocs.minimize([ocs.least_squares(ocs.Identity(n) @ x - bp),
                             ocs.l1(Gr @ x, ALPHA)]).compile(
            block=block, max_blocks=mb, tol=1e-12, equilibrate="sgd")
        bj = jnp.asarray(b_np.astype("f").reshape(-1))
        t0 = time.perf_counter()
        s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        s = comp.solve_instance({"b": bj}); jax.block_until_ready(s.x)
        wall = time.perf_counter() - t0
        out.append((s.iters, wall, objective(s.x, b_np, H, W),
                    compile_s, float(s.residual), s.status))
        print(f"  OCS  {s.iters:6d} it  {wall:8.3f}s  J={out[-1][2]:.6f}  "
              f"(compile {compile_s:.1f}s, res {s.residual:.2e}, {s.status})", flush=True)
    return out


def _write(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "unit", "work", "wall_s", "objective",
                    "compile_s", "residual", "status"])
        w.writerows(rows)
    print(f"wrote {path}")


def verdict_lines(series, best):
    """A floor above `best` means one of two different things: the method
    converged to its own fixed point, or it ran out of budget. Only a stalled
    tail separates them, so test that rather than the gap."""
    for name, pts in series.items():
        seq = [J for _, J in sorted(pts)]
        floor = min(seq)
        rel = (floor - best) / abs(best)
        tail = (seq[-2] - seq[-1]) / abs(best) if len(seq) > 1 else float("inf")
        if rel < 1e-5:
            v = "reaches the same optimum"
        elif tail < 1e-9:
            v = f"plateaus {rel:.2%} above: converged to its own discretization"
        else:
            v = (f"{rel:.2%} above, still descending ({tail:.2e}/step at the "
                 f"budget end): not converged, give it more")
        print(f"  {name:14s} best J={floor:.6f}  {v}")


if __name__ == "__main__":
    # SPDHG and OCS live in different environments on one machine, so each side
    # runs separately and the merge computes the common reference objective.
    args = sys.argv[1:]
    side = args[args.index("--side") + 1] if "--side" in args else "all"
    nepoch = int(args[0]) if args and args[0].isdigit() else 5000
    SIDE_CSV = {k: os.path.join(PLOTS, f"rof_side_{k}.csv") for k in ("spdhg", "ocs")}

    if side == "merge":
        rows, series = [], {}
        for f in SIDE_CSV.values():
            if not os.path.exists(f):
                print(f"missing {f}"); sys.exit(1)
            for r in csv.DictReader(open(f)):
                rows.append(r)
                series.setdefault(r["method"], []).append(
                    (float(r["wall_s"]), float(r["objective"])))
        best = min(J for v in series.values() for _, J in v)
        with open(OUT, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "unit", "work", "wall_s", "objective", "rel_gap",
                        "compile_s", "residual", "status"])
            for r in rows:
                J = float(r["objective"])
                w.writerow([r["method"], r["unit"], r["work"], r["wall_s"],
                            r["objective"], f"{(J - best) / abs(best):.6e}",
                            r["compile_s"], r["residual"], r["status"]])
        print(f"best objective seen: {best:.6f}")
        verdict_lines(series, best)
        print(f"wrote {OUT}")
        sys.exit(0)

    X, data, b_np, H, W = build_problem()
    print(f"ROF on Ehrhardt's 'building' image: {H}x{W} = {H*W:,} pixels, "
          f"alpha={ALPHA}, stddev={STDDEV}, seed={SEED}", flush=True)
    rows = []

    if side in ("spdhg", "all"):
        print(f"\nSPDHG-CPU (reference implementation, {nepoch} epochs)...", flush=True)
        for e, t, J in run_spdhg(X, data, b_np, H, W, nepoch):
            rows.append(["SPDHG-CPU", "epoch", e, round(t, 5), round(J, 6), "", "", ""])

        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device visible to torch")
            cap = torch.cuda.get_device_capability(0)
            print(f"\nSPDHG-GPU (impl='pytorch', device='cuda:0', as in "
                  f"get_started.py); torch {torch.__version__}, "
                  f"{torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}", flush=True)
            probe = torch.zeros(8, device="cuda:0") + 1.0     # fail fast if unsupported
            torch.cuda.synchronize(); assert float(probe.sum()) == 8.0
            Xg, datag, b_g, Hg, Wg = build_problem(impl="pytorch", device="cuda:0")
            assert np.allclose(b_g, b_np), "GPU problem data differs from CPU"
            for e, t, J in run_spdhg(Xg, datag, b_np, H, W, nepoch,
                                     sync=torch.cuda.synchronize):
                rows.append(["SPDHG-GPU", "epoch", e, round(t, 5), round(J, 6), "", "", ""])
        except Exception as exc:
            print(f"  SPDHG-GPU unavailable: {type(exc).__name__}: {exc}", flush=True)

        print(f"\nSplit Bregman (scikit-image implementation of [GO09])...", flush=True)
        sb = [10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
        try:
            for n_, t, J in run_split_bregman(b_np, H, W, sb):
                rows.append(["SplitBregman", "iteration", n_, round(t, 5),
                             round(J, 6), "", "", ""])
        except Exception as exc:
            print(f"  SplitBregman unavailable: {type(exc).__name__}: {exc}", flush=True)
        _write(rows, SIDE_CSV["spdhg"])

    if side in ("ocs", "all"):
        budgets = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
        print(f"\nOCS ({len(budgets)} independent runs)...", flush=True)
        orows = [["OCS", "iteration", it, round(t, 5), round(J, 6),
                  round(c, 3), f"{r:.3e}", st]
                 for it, t, J, c, r, st in run_ocs(b_np, H, W, budgets)]
        _write(orows, SIDE_CSV["ocs"])
