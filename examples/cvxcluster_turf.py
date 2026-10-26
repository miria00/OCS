"""CvxCluster on its own turf.

The Stage-1 pricing program of CvxCluster (Nnorom, Boyd and Levis, arXiv
2605.01614) is the relaxation of their placement MILP (their equations 1 to 3):

    maximize  p^T A^T 1        priority-weighted placement
    s.t.      A^T 1 <= 1       each task placed at most once
              A T   <= S       resource capacity
              A >= 0           (A <= 1 is implied by the first constraint)

with A the (bins x tasks) assignment.  Their bins are the m server *shapes*
after aggregation (m = 5 in their base cluster; m = 1 is their "Global"
variant).  Without aggregation the bins are the s individual servers, the
program they describe as O(s^2 t) and never run at scale.

OCS solves the same LP with the variable X = A^T (tasks x bins) and the
operator ocs.BinPack(Rmat, B): K x = [X 1_B ; vec(Rmat X)], Rmat = T^T (r x
tasks).  Nothing is formed.  The explicit solvers (HiGHS, cuOpt, SCS) receive
the identical LP as a CSR matrix and run on the same machine.  Their Gurobi
timings are quoted from their Table 1 in the figure, not reproduced here
(no license on the measurement machine).

Instance, following their Table 1 and Section 5: t tasks, s = t/40 servers,
five shapes spanning 96 to 192 vCPU and 256 to 768 GB, r = 2 resources, four
priority levels 1, 2, 4, 8 with task counts inversely proportional to the
level.  Per-task demands are synthetic (their Azure-trace instances and
simulator are not distributed with the paper); the demand scale is set so
that total demand is `load` times total capacity, which puts the LP
placement rate near their reported 88 to 89 percent (load 1.35).

Every solver runs in a guarded child (timeout, optional address-space cap),
timed inside the child.  One CSV row per (mode, tasks, servers, solver):

    plots/cvxcluster_turf.csv

Run (5090):  cd examples && ~/jaxenv/bin/python cvxcluster_turf.py --tasks 100000 --mode shape --solvers OCS
             /home/miria/optenv2/bin/python cvxcluster_turf.py --tasks 100000 --mode shape --solvers HIGHS,CUOPT,SCS
"""
import os, sys, time, csv, argparse, json, subprocess
import numpy as np
from guarded import run_guarded

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
os.makedirs(PLOTS, exist_ok=True)
OUT = os.path.join(PLOTS, "cvxcluster_turf.csv")
FIELDS = ["mode", "tasks", "servers", "bins", "variables", "nnz_explicit", "solver",
          "method", "status", "build_s", "solve_s", "warm_s", "objective",
          "placement_rate", "asgn_violation", "cap_violation_rel", "residual",
          "iterations", "tol", "device", "note"]

SHAPES = np.array([[96, 256], [128, 384], [160, 512], [176, 640], [192, 768]], np.float64)
LEVELS = np.array([1., 2., 4., 8.])
LEVEL_FRAC = (1 / LEVELS) / (1 / LEVELS).sum()          # 8/15, 4/15, 2/15, 1/15


def instance(tasks, servers, mode, load, seed):
    """Synthetic CvxCluster round.  Returns small arrays only: the demand table
    Rmat (r x tasks), the capacities C (r x bins) and the priorities p."""
    g = np.random.default_rng(seed)
    shape_of_server = g.integers(0, len(SHAPES), size=servers)
    level = g.choice(len(LEVELS), size=tasks, p=LEVEL_FRAC)
    p = LEVELS[level]
    cpu = 2.0 ** g.integers(0, 5, size=tasks)              # 1 to 16 vCPU before scaling
    mem = cpu * g.uniform(2., 8., size=tasks)               # GB per vCPU
    cap_total = SHAPES[shape_of_server].sum(0)              # [vCPU, GB] of the cluster
    cpu *= load * cap_total[0] / cpu.sum()
    mem *= load * cap_total[1] / mem.sum()
    Rmat = np.stack([cpu, mem])
    if mode == "server":
        C = SHAPES[shape_of_server].T
    elif mode == "shape":
        counts = np.bincount(shape_of_server, minlength=len(SHAPES))
        C = (SHAPES * counts[:, None]).T
    elif mode == "global":
        C = cap_total[:, None]
    else:
        raise ValueError(mode)
    return {"Rmat": Rmat.astype("f"), "C": np.ascontiguousarray(C, "f"), "p": p.astype("f"),
            "tasks": tasks, "servers": servers, "bins": C.shape[1], "mode": mode}


def lp_arrays(inst):
    t, B = inst["tasks"], inst["bins"]
    rhs = np.concatenate([np.ones(t, "f"), inst["C"].ravel()])
    cost = -np.repeat(inst["p"], B)                           # x = vec(X) row-major, X is tasks x bins
    return rhs, cost


def explicit_csr(inst, dtype=np.float64):
    """The LP matrix the explicit solvers cannot avoid: t assignment rows then
    r*B capacity rows, over the t*B variables."""
    from scipy import sparse
    t, B = inst["tasks"], inst["bins"]
    R = inst["Rmat"].astype(dtype)
    r = R.shape[0]
    asgn = sparse.csr_matrix((np.ones(t * B, dtype), np.arange(t * B, dtype=np.int32),
                              np.arange(0, t * B + 1, B, dtype=np.int64)), shape=(t, t * B))
    cols = (np.arange(B, dtype=np.int64)[:, None] + B * np.arange(t, dtype=np.int64)[None, :]).ravel()
    cap = sparse.csr_matrix((np.concatenate([np.tile(R[k], B) for k in range(r)]),
                             np.tile(cols, r).astype(np.int32),
                             np.arange(0, r * B * t + 1, t, dtype=np.int64)), shape=(r * B, t * B))
    return sparse.vstack([asgn, cap], format="csr")


def utility_value(p, placed, utility):
    if utility == "linear":
        return float(p @ placed)
    if utility == "pwl1":
        return float(p @ np.minimum(placed, 1.0))
    return float(p @ (np.minimum(placed, 1.0) + 0.25 * np.clip(placed - 1.0, 0.0, 1.0)))


def independent(inst, X, utility="linear"):
    """Numpy float64 checks of a returned point, independent of every solver."""
    R = inst["Rmat"].astype(np.float64); C = inst["C"].astype(np.float64); p = inst["p"].astype(np.float64)
    X = np.asarray(X, np.float64)
    placed = X.sum(1)
    return {"objective": utility_value(p, placed, utility),
            "placement_rate": float(np.minimum(placed, 1.0).mean()),
            "asgn_violation": float(max(0.0, (placed - 1.0).max())),
            "cap_violation_rel": float(max(0.0, ((R @ X - C) / C).max())),
            "min_x": float(X.min())}


# ---------------------------------------------------------------- solvers (child processes)

def solve_ocs(inst, tol, blocks, weight, variant, compact=False, adaptive=False, utility="linear", fixed=False):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax, ocs
    t, B = inst["tasks"], inst["bins"]
    rhs, cost = lp_arrays(inst)
    x = ocs.Variable(t * B); bp = ocs.Parameter("rhs"); cp = ocs.Parameter("c")
    if utility == "linear":
        K = ocs.BinPack(inst["Rmat"], B)
        prob = ocs.minimize([ocs.nonneg(x), ocs.nonneg(ocs.Weighted(K, -1.) @ x + bp)], cost=cp)
    else:
        # PWL utility per task, "linear with a max value": u_j(z) = p_j min(z, 1)
        # (and, for the two-piece variant, a quarter slope from 1 to 2, flat after).
        # The cost is the convex PWL  phi(z) = max_k (s_k z + r_k), applied to the
        # placement z = X 1 through the assignment operator; the priority p_j
        # scales the term by putting p_j into the operator row. Capacity rows
        # stay a nonnegativity cone on -Rmat X + C. No assignment cap row: the
        # utility saturates instead.
        import jax.numpy as jnp
        r = inst["Rmat"].shape[0]; Rm = jnp.asarray(inst["Rmat"])
        cap = ocs.LinearOperator.from_forward(lambda v: -(Rm @ v.reshape(t, B)).reshape(-1), n=t * B, m=r * B)
        # One PWL term per priority level (the block shares its pieces across
        # coordinates): cost p_k * max(-z, -1) = max(-p_k z, -p_k) on the placements
        # z = X 1 of the tasks at level k; pwl2 adds a quarter-slope second replica.
        terms = [ocs.nonneg(x), ocs.nonneg(cap @ x + ocs.Parameter("cap"))]
        levels = np.unique(inst["p"])
        for pk in levels:
            idx = jnp.asarray(np.flatnonzero(inst["p"] == pk)); pk = float(pk)
            sel = ocs.LinearOperator.from_forward(lambda v, idx=idx: v.reshape(t, B).sum(axis=1)[idx], n=t * B, m=int(idx.size))
            if utility == "pwl1":
                slopes, inter = [-pk, 0.0], [0.0, -pk]
            else:
                slopes, inter = [-pk, -0.25 * pk, 0.0], [0.0, -0.75 * pk, -1.25 * pk]
            terms.append(ocs.pwl(sel @ x, slopes, inter))
        prob = ocs.minimize(terms)
    t0 = time.perf_counter()
    comp = prob.compile(block=100, max_blocks=blocks, tol=tol, equilibrate="sgd",
                        residual_norm="relative", pdhg_variant=variant, primal_weight=weight,
                        data_layout="compact" if compact else "dense",
                        adaptive_omega=True if adaptive else (False if fixed else None))   # None: the package default
    build_s = time.perf_counter() - t0
    data = {"rhs": rhs, "c": cost} if utility == "linear" else {"cap": inst["C"].ravel()}
    t0 = time.perf_counter()
    sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = comp.solve_instance(data); jax.block_until_ready((sol.x, sol.y))
    warm_s = time.perf_counter() - t0
    X = np.asarray(sol.x, np.float64).reshape(t, B)
    y = np.asarray(sol.y, np.float64)
    out = independent(inst, X, utility)
    out.update(solver="OCS", method=f"pdhg-{variant}" + ("-compact" if compact else "") + ("-adaptive" if adaptive else "") + ("-fixed" if fixed else "") + f"-auto{comp.meta.get('adaptive_omega')}{comp.meta.get('adaptive_blocks')}", status=str(sol.status),
               residual=float(sol.residual), iterations=int(sol.iters),
               build_s=build_s, solve_s=first_s, warm_s=warm_s,
               prices=(-y[t:]).tolist() if y.size - t <= 200000 else [],
               device=jax.devices()[0].device_kind, note="build=compile+equilibrate; solve=first call incl. JIT; warm=second call")
    return out


def epigraph_lp(inst, utility):
    """Explicit LP for the PWL utility: variables (vec X, u) with u_j <= each affine
    piece of the concave utility; maximize sum u_j. Rows: capacity (r*B), then the
    pieces (t per piece)."""
    from scipy import sparse
    t, B = inst["tasks"], inst["bins"]
    A = explicit_csr(inst)                                  # [assignment rows (t); capacity rows (r*B)]
    asg, cap = A[:t], A[t:]
    p = inst["p"].astype(np.float64)
    pieces = [(1.0, 0.0), (0.0, 1.0)] if utility == "pwl1" else [(1.0, 0.0), (0.25, 0.75), (0.0, 1.25)]
    rows = [sparse.hstack([cap, sparse.csr_matrix((cap.shape[0], t))], format="csr")]
    rhs = [inst["C"].astype(np.float64).ravel()]
    for a, b in pieces:                                     # u_j - a p_j z_j <= b p_j
        rows.append(sparse.hstack([-a * sparse.diags(p) @ asg, sparse.identity(t, format="csr")], format="csr"))
        rhs.append(b * p)
    Aub = sparse.vstack(rows, format="csr"); bub = np.concatenate(rhs)
    cost = np.concatenate([np.zeros(t * B), -np.ones(t)])
    return Aub, bub, cost


def solve_highs(inst, tol, timeout_s, hmethod="highs-ipm", utility="linear"):
    from scipy.optimize import linprog
    t, B = inst["tasks"], inst["bins"]
    t0 = time.perf_counter()
    if utility == "linear":
        rhs, cost = lp_arrays(inst); A = explicit_csr(inst); rhs = rhs.astype(np.float64)
    else:
        A, rhs, cost = epigraph_lp(inst, utility)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = linprog(cost.astype(np.float64), A_ub=A, b_ub=rhs, bounds=(0, None),
                  method=hmethod, options={"time_limit": max(10.0, timeout_s - 30.0), "disp": False})
    solve_s = time.perf_counter() - t0
    out = {"solver": "HIGHS", "method": hmethod, "status": res.message.split(".")[0],
           "build_s": build_s, "solve_s": solve_s, "warm_s": "", "residual": "", "iterations": int(res.nit),
           "nnz_explicit": int(A.nnz), "device": _cpu_name(), "note": "scipy.optimize.linprog method=highs, float64"}
    if res.x is not None:
        out.update(independent(inst, res.x[:t * B].reshape(t, B), utility))
        if res.ineqlin is not None:
            out["prices"] = (-res.ineqlin.marginals[t:]).tolist() if res.ineqlin.marginals.size - t <= 200000 else []
    return out


def solve_cuopt(inst, tol, timeout_s, method, utility="linear"):
    from cuopt.linear_programming import DataModel, SolverSettings, SolverMethod, Solve
    t, B = inst["tasks"], inst["bins"]
    t0 = time.perf_counter()
    if utility == "linear":
        rhs, cost = lp_arrays(inst); A = explicit_csr(inst); rhs = rhs.astype(np.float64)
    else:
        A, rhs, cost = epigraph_lp(inst, utility)
    build_s = time.perf_counter() - t0
    nvar = A.shape[1]
    dm = DataModel()
    dm.set_csr_constraint_matrix(A.data.astype(np.float64), A.indices.astype(np.int32), A.indptr.astype(np.int32))
    dm.set_constraint_upper_bounds(rhs.astype(np.float64))
    dm.set_constraint_lower_bounds(np.full(rhs.size, -np.inf))
    dm.set_objective_coefficients(cost.astype(np.float64))
    dm.set_variable_lower_bounds(np.zeros(nvar))
    dm.set_variable_upper_bounds(np.full(nvar, np.inf))
    st = SolverSettings()
    st.set_parameter("method", getattr(SolverMethod, method))
    st.set_parameter("time_limit", max(10.0, timeout_s - 30.0))
    st.set_optimality_tolerance(tol)
    try:
        st.set_parameter("log_to_console", False)
    except Exception:
        pass
    t0 = time.perf_counter(); sol = Solve(dm, st); solve_s = time.perf_counter() - t0
    x = np.asarray(sol.get_primal_solution(), np.float64)
    status = str(sol.get_termination_status())
    try:
        status = str(sol.get_termination_reason())
    except Exception:
        pass
    out = {"solver": "CUOPT", "method": f"cuopt-{method}", "status": status,
           "build_s": build_s, "solve_s": solve_s, "warm_s": "", "residual": "", "iterations": "",
           "nnz_explicit": int(A.nnz), "device": _gpu_name(), "note": "cuOpt DataModel CSR float64; solve_s is wall around Solve()"}
    try:
        out["note"] += f"; cuopt solve_time={float(sol.get_solve_time()):.3f}"
    except Exception:
        pass
    if x.size >= t * B:
        out.update(independent(inst, x[:t * B].reshape(t, B), utility))
        try:
            y = np.asarray(sol.get_dual_solution(), np.float64)
            out["prices"] = (-y[t:]).tolist() if y.size - t <= 200000 else []
        except Exception:
            pass
    return out


def solve_scs(inst, tol, timeout_s, indirect=False):
    import scs
    from scipy import sparse
    t, B = inst["tasks"], inst["bins"]
    n = t * B
    rhs, cost = lp_arrays(inst)
    t0 = time.perf_counter()
    K = explicit_csr(inst)
    A = sparse.vstack([K, -sparse.identity(n, format="csr")], format="csc")
    b = np.concatenate([rhs.astype(np.float64), np.zeros(n)])
    build_s = time.perf_counter() - t0
    solver = scs.SCS({"A": A, "b": b, "c": cost.astype(np.float64)}, {"l": int(A.shape[0])},
                     eps_abs=tol, eps_rel=tol, verbose=False, time_limit_secs=max(10.0, timeout_s - 30.0),
                     **({"linear_solver": "cpu_indirect"} if indirect else {}))
    t0 = time.perf_counter(); res = solver.solve(); solve_s = time.perf_counter() - t0
    out = {"solver": "SCS", "method": "scs-indirect" if indirect else "scs-direct", "status": res["info"]["status"],
           "build_s": build_s, "solve_s": solve_s, "warm_s": "", "residual": "",
           "iterations": int(res["info"]["iter"]), "nnz_explicit": int(K.nnz),
           "device": _cpu_name(), "note": "scs 3.x direct, x>=0 as -I rows; nnz_explicit counts K only"}
    x = np.asarray(res["x"], np.float64)
    if x.size == n and np.isfinite(x).all():
        out.update(independent(inst, x.reshape(t, B)))
        y = np.asarray(res["y"], np.float64)
        out["prices"] = (y[t:t + rhs.size - t]).tolist() if rhs.size - t <= 200000 else []
    return out


def _cpu_name():
    try:
        line = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        return line.split(":", 1)[1].strip() + f" x{os.cpu_count()}"
    except Exception:
        return "cpu"


def _gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).split("\n")[0].strip()
    except Exception:
        return "gpu"


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=1000)
    ap.add_argument("--servers", type=int, default=0, help="0 means tasks/40 as in their Table 1")
    ap.add_argument("--mode", choices=["shape", "global", "server"], default="shape")
    ap.add_argument("--solvers", default="OCS,HIGHS,CUOPT,SCS")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--mem-gb", type=float, default=0.0, help="RLIMIT_AS for CPU solvers; 0 = none")
    ap.add_argument("--load", type=float, default=1.35, help="total demand over total capacity; 1.35 gives their 89 percent placement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--variant", default="halpern")
    ap.add_argument("--compact", action="store_true", help="OCS data_layout=compact (smaller device footprint)")
    ap.add_argument("--adaptive-omega", action="store_true", help="OCS adaptive primal weight, continuous")
    ap.add_argument("--fixed-omega", action="store_true", help="OCS fixed primal weight (adaptive off)")
    ap.add_argument("--utility", default="linear", choices=["linear", "pwl1", "pwl2"],
                    help="linear: their LP with the assignment cap; pwl1: PWL utility p*min(z,1) (linear with a max value), no cap row; pwl2: two-piece PWL utility")
    ap.add_argument("--cuopt-method", default="Concurrent")
    ap.add_argument("--highs-method", default="highs-ipm", choices=["highs", "highs-ipm", "highs-ds"])
    ap.add_argument("--scs-indirect", action="store_true", help="SCS conjugate-gradient linear solver instead of the direct factorization")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    servers = args.servers or max(1, args.tasks // 40)
    inst = instance(args.tasks, servers, args.mode, args.load, args.seed)
    t, B = inst["tasks"], inst["bins"]
    nnz = t * B * (1 + inst["Rmat"].shape[0])
    print(f"CvxCluster turf: mode={args.mode} utility={args.utility} tasks={t:,} servers={servers:,} bins={B:,} "
          f"variables={t*B:,} explicit nnz={nnz:,} ({nnz*12/1e9:.2f} GB CSR float64) tol={args.tol}", flush=True)

    rows = []
    for name in [s.strip().upper() for s in args.solvers.split(",") if s.strip()]:
        if name == "OCS":
            g = run_guarded(solve_ocs, args=(inst, args.tol, args.blocks, args.weight, args.variant, args.compact, args.adaptive_omega, args.utility, args.fixed_omega),
                            label="OCS", timeout_s=args.timeout)
        elif name == "HIGHS":
            g = run_guarded(solve_highs, args=(inst, args.tol, args.timeout, args.highs_method, args.utility), label="HIGHS",
                            timeout_s=args.timeout, mem_gb=args.mem_gb or None)
        elif name == "CUOPT":
            g = run_guarded(solve_cuopt, args=(inst, args.tol, args.timeout, args.cuopt_method, args.utility),
                            label="CUOPT", timeout_s=args.timeout)
        elif name == "SCS":
            g = run_guarded(solve_scs, args=(inst, args.tol, args.timeout, args.scs_indirect), label="SCS",
                            timeout_s=args.timeout, mem_gb=args.mem_gb or None)
        else:
            print(f"  unknown solver {name}", flush=True); continue
        row = {k: "" for k in FIELDS}
        row.update(mode=args.mode + ("" if args.utility == "linear" else "-" + args.utility), tasks=t, servers=servers, bins=B, variables=t * B,
                   nnz_explicit=nnz, solver=name, tol=args.tol, note=args.tag)
        if g["status"] == "ok":
            v = g["value"]
            prices = v.pop("prices", [])
            if prices:
                np.save(os.path.join(PLOTS, f"cvxcluster_turf_prices_{args.mode}_t{t}_s{servers}_{name}.npy"),
                        np.asarray(prices))
            row.update({k: v[k] for k in v if k in FIELDS})
            row["note"] = (args.tag + "; " if args.tag else "") + v.get("note", "")
            warm = "" if v["warm_s"] == "" else f"{float(v['warm_s']):.3f}s"
            print(f"  {name:6s} {v['status']:>14s} build {float(v['build_s']):8.2f}s solve {float(v['solve_s']):9.2f}s "
                  f"warm {warm} "
                  f"obj {v.get('objective', float('nan')):.4f} placed {v.get('placement_rate', float('nan')):.4f} "
                  f"viol {v.get('asgn_violation', float('nan')):.1e}/{v.get('cap_violation_rel', float('nan')):.1e}"
                  + (f" res {v['residual']:.1e} it {v['iterations']}" if name == "OCS" else ""), flush=True)
        else:
            row.update(status=g["status"], solve_s=round(g["seconds"], 2))
            print(f"  {name:6s} {g['status']} after {g['seconds']:.1f}s", flush=True)
        rows.append(row)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    objs = [float(r["objective"]) for r in rows if r["objective"] != ""]
    if len(objs) > 1:
        spread = (max(objs) - min(objs)) / max(1.0, abs(np.median(objs)))
        print(f"objective spread across solvers: {spread:.2e} relative", flush=True)
    print(f"appended {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
