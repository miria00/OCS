"""Problem builders for the gate campaign: every experiment class of the paper,
constructed exactly as the paper's example scripts and figure code construct
them (same instance generators, seeds, sizes, regularization and compile
keywords), returned uncompiled so that gated_weight_runner.py can drive the
shipped engine at a fixed weight per epoch.

Each builder returns (prob, data, check, name, compile_kwargs, paper_weight).
`check(x, y)` is the example's own independent host-side diagnostic (float64,
never an OCS residual).  Nothing here forms a matrix the example does not
already form: every operator is the same ocs operator the example uses.

Sources:
    ot        examples/_quadratic_ot.py family() + ot_sweep.py instance   (figure H)
    fft       examples/make_plots.py _fft_runtime                          (figure C)
    mri       examples/make_plots.py D_mri_scale                           (figure D)
    kron      examples/kron_wedge.py run_ocs                               (figure F)
    sylvester examples/killer_sylvester.py                                 (section text)
    rof       examples/rof_spdhg_vs_ocs.py run_ocs                         (figure I)
    featsel   examples/killer_feature_selection.py                         (section text)
    turf, imaging: in gated_weight_runner.py                               (figures J, K)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)


def _f64(v):
    return np.asarray(v, np.float64)


# ----------------------------------------------------------------------------- OT (figure H)
def ot_problem(N, seed=0):
    """examples/_quadratic_ot.family(N) uncompiled, instance as ot_sweep.py draws it
    (rng(0), uniform cost).  Compile: tol 2e-6, primal_weight 30, standard PDHG,
    loop_unroll 4 (the family's keywords)."""
    import ocs, jax.numpy as jnp
    from ocs.ir import Term
    from _quadratic_ot import instance, data, diagnostics, EPS
    rng = np.random.default_rng(seed)
    c, p, q = instance(N, rng)
    x = ocs.Variable(N * N)
    marginal = ocs.Parameter("m"); cost = ocs.Parameter("c")
    nonnegative_quadratic = ocs.NonNeg().with_linear_quadratic(e=EPS)
    prob = ocs.minimize([Term(ocs.Identity(N * N), nonnegative_quadratic),
                         ocs.equality(ocs.MarginalOperator(N, N) @ x - marginal)], cost=cost)
    d = data(c, p, q)

    def check(xv, yv=None):
        if yv is None:
            return {}
        out = diagnostics(c, p, q, xv, yv)
        return {k: float(v) for k, v in out.items()}
    ck = {"pdhg_variant": "standard", "loop_unroll": 4}
    return prob, d, check, f"ot-{N}", ck, 30.0


# ----------------------------------------------------------------------------- FFT LASSO (figure C)
def fft_problem(n, ratio=0.25, lam=1e-3, seed=0):
    """make_plots._fft_runtime: subsampled-FFT LASSO, equilibrate='scalar'."""
    import ocs, jax, jax.numpy as jnp
    K = ocs.FFTMask.random(n, ratio, seed=seed)
    x = ocs.Variable(n); y = ocs.Parameter("y")
    prob = ocs.minimize([ocs.l1(x, lam), ocs.least_squares(K @ x - y)])
    ym = np.random.default_rng(0).normal(size=K.shape[0]).astype("f")
    d = {"y": jnp.asarray(ym)}
    mv = jax.jit(K.matvec)

    def check(xv, yv=None):
        xv = np.asarray(xv, np.float32)
        r = _f64(mv(jnp.asarray(xv))) - _f64(ym)
        return {"objective": float(0.5 * r @ r + lam * np.abs(_f64(xv)).sum()),
                "nnz_1e-6": int((np.abs(xv) > 1e-6).sum())}
    return prob, d, check, f"fft-{n}", {"equilibrate": "scalar"}, 1.0


# ----------------------------------------------------------------------------- MRI TV SOCP (figure D)
def mri_problem(L, ratio=0.35, lam=0.03, seed=0):
    """make_plots.D_mri_scale: masked FFT + isotropic TV (group_l2, size 2), equilibrate='sgd'."""
    import ocs, jax, jax.numpy as jnp
    n = L * L
    F = ocs.FFTMask.random(n, ratio, seed=seed); Gr = ocs.Gradient2D(L, L)
    x = ocs.Variable(n); y = ocs.Parameter("y")
    prob = ocs.minimize([ocs.least_squares(F @ x - y), ocs.group_l2(Gr @ x, lam, 2)])
    ym = np.random.default_rng(0).normal(size=F.shape[0]).astype("f")
    d = {"y": jnp.asarray(ym)}
    fmv, gmv = jax.jit(F.matvec), jax.jit(Gr.matvec)

    def check(xv, yv=None):
        xj = jnp.asarray(np.asarray(xv, np.float32))
        r = _f64(fmv(xj)) - _f64(ym)
        g = _f64(gmv(xj)).reshape(-1, 2)
        return {"objective": float(0.5 * r @ r + lam * np.linalg.norm(g, axis=1).sum())}
    return prob, d, check, f"mri-{L}", {"equilibrate": "sgd"}, 1.0


# ----------------------------------------------------------------------------- Kronecker lasso (figure F)
def kron_problem(T, seed=0):
    """kron_wedge.run_ocs: S = 64, lam 0.05, tol 1e-6, equilibrate='sgd'."""
    import ocs, jax, jax.numpy as jnp
    import kron_wedge as kw
    A, B, Xt = kw.make(T, seed)
    S, LAM = kw.S, kw.LAM
    n = S * T
    K = ocs.Kronecker(A, B)
    b = np.asarray(jax.jit(K.matvec)(jnp.asarray(Xt.reshape(-1))))
    x = ocs.Variable(n); bp = ocs.Parameter("b")
    prob = ocs.minimize([ocs.least_squares(ocs.Kronecker(A, B) @ x - bp),
                         ocs.l1(ocs.Identity(n) @ x, LAM)])
    d = {"b": jnp.asarray(b)}
    mv = jax.jit(K.matvec)

    def check(xv, yv=None):
        xv = np.asarray(xv, np.float32)
        r = _f64(mv(jnp.asarray(xv))) - _f64(b)
        return {"objective": float(0.5 * r @ r + LAM * np.abs(_f64(xv)).sum()),
                "rel_error_vs_truth": float(np.linalg.norm(_f64(xv) - Xt.reshape(-1)) / np.linalg.norm(Xt))}
    return prob, d, check, f"kron-T{T}", {"equilibrate": "sgd"}, 1.0


# ----------------------------------------------------------------------------- Sylvester LS (section text)
def sylvester_problem(p=60, q=50, seed=0):
    """killer_sylvester.py: pure least squares, PDHG at tol 1e-8 with adaptive_omega=False;
    the independent reference is scipy.linalg.solve_sylvester."""
    import ocs, jax.numpy as jnp
    from scipy.linalg import solve_sylvester
    rng = np.random.default_rng(seed)
    shift_p, shift_q = 0.5 * np.sqrt(p) + 5, 0.5 * np.sqrt(q) + 5
    A = rng.normal(size=(p, p)).astype(np.float32) * 0.5 + shift_p * np.eye(p, dtype=np.float32)
    B = rng.normal(size=(q, q)).astype(np.float32) * 0.5 + shift_q * np.eye(q, dtype=np.float32)
    Xt = rng.normal(size=(p, q)).astype(np.float32)
    C = (A @ Xt + Xt @ B).astype(np.float32)
    Xr = solve_sylvester(A.astype(np.float64), B.astype(np.float64), C.astype(np.float64))
    S = ocs.Sylvester(A, B); x = ocs.Variable(p * q); c = ocs.Parameter("c")
    prob = ocs.minimize([ocs.least_squares(S @ x - c)])
    d = {"c": jnp.asarray(C.reshape(-1))}

    def check(xv, yv=None):
        X = _f64(xv).reshape(p, q)
        return {"rel_error_vs_direct": float(np.linalg.norm(X - Xr) / np.linalg.norm(Xr))}
    return prob, d, check, f"sylvester-{p}x{q}", {}, 1.0


# ----------------------------------------------------------------------------- ROF denoising (figure I)
def rof_problem(data_path=None):
    """rof_spdhg_vs_ocs.run_ocs: Identity data term + anisotropic TV through Gradient2D,
    OCS minimizes alpha*J, equilibrate='sgd'.  The noisy image b is Ehrhardt's
    (odl building image, seed 1807, stddev 0.1); it is built through odl when
    available, or loaded from `data_path` (saved by a previous odl build)."""
    import ocs, jax.numpy as jnp
    import rof_spdhg_vs_ocs as rof
    if data_path and os.path.exists(data_path):
        b_np = np.load(data_path); H, W = b_np.shape
    else:
        _, _, b_np, H, W = rof.build_problem()
        if data_path:
            np.save(data_path, b_np)
    n = H * W
    x = ocs.Variable(n); bp = ocs.Parameter("b"); Gr = ocs.Gradient2D(H, W)
    prob = ocs.minimize([ocs.least_squares(ocs.Identity(n) @ x - bp), ocs.l1(Gr @ x, rof.ALPHA)])
    d = {"b": jnp.asarray(b_np.astype("f").reshape(-1))}

    def check(xv, yv=None):
        return {"objective_J": rof.objective(xv, b_np, H, W)}
    return prob, d, check, f"rof-{H}x{W}", {"equilibrate": "sgd"}, 1.0


# ----------------------------------------------------------------------------- feature selection (section text)
def featsel_problem(m=600, F=10000, T=10000, k_active=12, lam=0.5, seed=0):
    """killer_feature_selection.py: matrix variable Theta (F x T), LeftMatMul, row-group lasso."""
    import ocs, jax.numpy as jnp
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, F)).astype(np.float32)
    active = rng.choice(F, k_active, replace=False)
    Theta_true = np.zeros((F, T), np.float32)
    Theta_true[active] = rng.normal(size=(k_active, T))
    B = (A @ Theta_true + 0.01 * rng.normal(size=(m, T))).astype(np.float32)
    K = ocs.LeftMatMul(A, T)
    th = ocs.Variable(F * T); b = ocs.Parameter("b")
    prob = ocs.minimize([ocs.group_l2(th, lam=lam, group_size=T), ocs.least_squares(K @ th - b)])
    d = {"b": jnp.asarray(B.reshape(-1))}
    active_set = set(active.tolist())

    def check(xv, yv=None):
        Th = np.asarray(xv, np.float32).reshape(F, T)
        row_norm = np.linalg.norm(Th, axis=1)
        sel = set(np.argsort(row_norm)[-k_active:].tolist())
        inactive = np.array([i for i in range(F) if i not in active_set])
        return {"true_positives": len(sel & active_set), "k_active": k_active,
                "median_inactive_row_norm": float(np.median(row_norm[inactive])),
                "min_active_row_norm": float(row_norm[list(active_set)].min())}
    return prob, d, check, f"featsel-m{m}-F{F}-T{T}", {}, 1.0
