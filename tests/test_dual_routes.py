"""The dual-bound routes of the objective-gap controller (0.3.6).

``compile(dual_routes=...)`` adds three candidates, each for one structure
decided on exact host facts: the recession shift, the packing-LP assignment
dual and the compound ridge.  These tests check:

1. the option: default, ``()``, subsets, refusals, and ``meta["dual_routes"]``;
2. eligibility on the intended structures and refusal on near misses;
3. validity: every route witness passes the production gate with
   ``recover=False`` AND an independent float64 membership test, and its
   float64 dual value is a lower bound on an independent optimum;
4. directed rounding under jit and jit+vmap with ``optimization_barrier``:
   every rounded-up float32 entry is at or above its float64 target;
5. batched equals single: the route under the package's batched pattern
   (vmap inside ``jax.enable_x64()``) is bitwise equal to single calls;
6. the controller: a route is consulted only where the evaluator's packet
   does not accept; with routes on, the run's history up to its stopping check
   is the routes-off history; ``dual_routes=()`` records no route work; the
   returned bound is the ``recover=False`` bound of the returned pair; and
   the batch interface is unchanged by the option.

The instances are the paper's two LP-structured families at small size: the
cluster-pricing packing LP (``examples/cvxcluster_turf.py``, shape form, 1000
tasks) and the multicommodity flow ladder (``examples/mcf_competitors.py``,
K = 4), generated here by copies of those generators.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import ocs
from ocs.model.compile import (DUAL_ROUTES, DUAL_ROUTES_DEFAULT, DUAL_ROUTE_SOURCES,
                               DUAL_ROUTE_OPERATOR_APPLICATIONS, _check_dual_routes)

F32 = np.float32
EPS = float(np.finfo(F32).eps)
TINY = float(np.finfo(F32).tiny)


# ---------------------------------------------------------------------------
# Instances (copies of the example generators)
# ---------------------------------------------------------------------------
SHAPES = np.array([[96, 256], [128, 384], [160, 512], [176, 640], [192, 768]], np.float64)
LEVELS = np.array([1., 2., 4., 8.])
LEVEL_FRAC = (1 / LEVELS) / (1 / LEVELS).sum()


def pricing_instance(tasks=1000, servers=25, load=1.35, seed=0):
    """examples/cvxcluster_turf.py instance(..., mode="shape") and lp_arrays()."""
    g = np.random.default_rng(seed)
    shape_of_server = g.integers(0, len(SHAPES), size=servers)
    level = g.choice(len(LEVELS), size=tasks, p=LEVEL_FRAC)
    p = LEVELS[level]
    cpu = 2.0 ** g.integers(0, 5, size=tasks)
    mem = cpu * g.uniform(2., 8., size=tasks)
    cap_total = SHAPES[shape_of_server].sum(0)
    cpu *= load * cap_total[0] / cpu.sum()
    mem *= load * cap_total[1] / mem.sum()
    Rmat = np.stack([cpu, mem]).astype(F32)
    counts = np.bincount(shape_of_server, minlength=len(SHAPES))
    C = np.ascontiguousarray((SHAPES * counts[:, None]).T, F32)
    B = C.shape[1]
    rhs = np.concatenate([np.ones(tasks, F32), C.ravel()])
    cost = -np.repeat(p.astype(F32), B)
    return {"Rmat": Rmat, "B": B, "t": tasks, "rhs": rhs, "c": cost}


def pricing_problem(inst, weight=-1.0, extra_block=False):
    t, B = inst["t"], inst["B"]
    x = ocs.Variable(t * B)
    bp, cp = ocs.Parameter("rhs"), ocs.Parameter("c")
    terms = [ocs.nonneg(x), ocs.nonneg(ocs.Weighted(ocs.BinPack(inst["Rmat"], B), weight) @ x + bp)]
    if extra_block:
        terms.append(ocs.sum_squares(ocs.Diagonal(np.full(t * B, 1e-3, F32)) @ x))
    return ocs.minimize(terms, cost=cp), {"rhs": inst["rhs"], "c": inst["c"]}


PRICING_KW = dict(block=100, max_blocks=200, tol=1e-3, equilibrate="sgd",
                  pdhg_variant="halpern", primal_weight=10.0)

MCF_V, MCF_E, MCF_MU = 50, 200, 1e-3
MCF_SLOPES = np.array([0.5, 1.0, 2.0, 4.0], np.float64)
_inter = [0.0]
for _j, _b in enumerate([0.5, 1.0, 1.5]):
    _inter.append(_inter[-1] + (MCF_SLOPES[_j] - MCF_SLOPES[_j + 1]) * _b)
MCF_INTER = np.array(_inter, np.float64)


def mcf_instance(K=4, seed=1):
    """examples/mcf_competitors.py network() and instance(N, K)."""
    r = np.random.default_rng(seed)
    N = np.zeros((MCF_V, MCF_E), F32)
    for e in range(MCF_E):
        a, b = r.choice(MCF_V, 2, replace=False)
        N[a, e] = 1.0
        N[b, e] = -1.0
    r = np.random.default_rng(seed)
    Ftrue = r.uniform(0.0, 0.6, (MCF_E, K)).astype(F32)
    d = (N @ Ftrue).reshape(-1).astype(np.float64)
    return {"N": N, "K": K, "d": d.astype(F32), "sqrt_mu": F32(np.sqrt(MCF_MU))}


def mcf_problem(inst, ridge_op=None, second_ridge=False):
    N, K = inst["N"], inst["K"]
    V, E = N.shape
    n = E * K
    x = ocs.Variable(n)
    rhs = ocs.Parameter("rhs")
    ridge = ridge_op if ridge_op is not None else ocs.Diagonal(inst["sqrt_mu"] * np.ones(n, F32))
    terms = [ocs.nonneg(x), ocs.sum_squares(ridge @ x),
             ocs.equality(ocs.MultiCommodityFlow(N, K) @ x - rhs),
             ocs.pwl(ocs.CommoditySum(E, K) @ x, MCF_SLOPES.astype(F32), MCF_INTER.astype(F32))]
    if second_ridge:
        terms.append(ocs.sum_squares(ocs.Diagonal(np.full(n, 0.01, F32)) @ x))
    return ocs.minimize(terms), {"rhs": jnp.asarray(inst["d"])}


MCF_KW = dict(block=100, max_blocks=200, tol=1e-5, equilibrate="sgd",
              pdhg_variant="standard", primal_weight=128.0)


def lp_problem(m=40, n=60, seed=0):
    """minimize c^T x  s.t.  A x <= b, x >= 0 (Dense; the shift is the only route)."""
    g = np.random.default_rng(seed)
    A = g.uniform(0.0, 1.0, (m, n)).astype(F32)
    b = g.uniform(1.0, 2.0, m).astype(F32)
    c = -g.uniform(0.1, 1.0, n).astype(F32)
    x = ocs.Variable(n)
    bp, cp = ocs.Parameter("b"), ocs.Parameter("c")
    prob = ocs.minimize([ocs.nonneg(x), ocs.nonneg(ocs.Dense(-A) @ x + bp)], cost=cp)
    return prob, {"b": b, "c": c}, (A, b, c)


# ---------------------------------------------------------------------------
# Independent float64 algebra (numpy only)
# ---------------------------------------------------------------------------
def pricing_float64(inst, y):
    """Exact membership and dual value of y = -(alpha; beta) for the packing LP."""
    t, B = inst["t"], inst["B"]
    R = inst["Rmat"].astype(np.float64)
    y = np.asarray(y, np.float64)
    KTy = (y[:t, None] + R.T @ y[t:].reshape(R.shape[0], B)).ravel()   # (-K)^T y = -K^T y
    z = KTy - inst["c"].astype(np.float64)                             # z = -(-K)^T y - c
    member = bool(np.all(y <= 0.0) and np.all(z <= 0.0))
    return member, float(inst["rhs"].astype(np.float64) @ y)


def pricing_optimum(inst):
    scipy = pytest.importorskip("scipy")
    from scipy import sparse
    from scipy.optimize import linprog
    t, B = inst["t"], inst["B"]
    R = inst["Rmat"].astype(np.float64)
    asgn = sparse.kron(sparse.eye(t), np.ones((1, B)))
    cap = sparse.vstack([sparse.kron(sparse.csr_matrix(R[k:k + 1]), sparse.eye(B)) for k in range(R.shape[0])])
    A = sparse.vstack([asgn, cap]).tocsr()
    res = linprog(inst["c"].astype(np.float64), A_ub=A, b_ub=inst["rhs"].astype(np.float64),
                  bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def mcf_split(inst, y):
    E, K = inst["N"].shape[1], inst["K"]
    n, VK = E * K, inst["N"].shape[0] * K
    y = np.asarray(y, np.float64)
    return y[:n], y[n:n + VK], y[n + VK:]


def mcf_float64(inst, y):
    """Exact membership and dual value for the MCF layout [y_s | y_e | y_p]."""
    N64 = inst["N"].astype(np.float64)
    E, K = inst["N"].shape[1], inst["K"]
    ys, ye, yp = mcf_split(inst, y)
    sm = float(inst["sqrt_mu"])
    KTy = (sm * ys.reshape(E, K) + N64.T @ ye.reshape(-1, K) + yp[:, None]).ravel()
    s32, r32 = MCF_SLOPES.astype(F32).astype(np.float64), MCF_INTER.astype(F32).astype(np.float64)
    in_dom = bool(np.all((yp >= s32.min()) & (yp <= s32.max())))
    member = bool(in_dom and np.all(-KTy <= 0.0))
    if not in_dom:
        return member, float("-inf")
    # every piece is active for these slopes, so g^* interpolates (s_j, -r_j)
    conj = np.interp(yp, s32, -r32)
    d64 = inst["d"].astype(np.float64)
    return member, float(-d64 @ ye - 0.5 * ys @ ys - conj.sum())


def mcf_optimum_consistent(inst):
    """The optimum of the consistent instance (each commodity's demand minus its
    node mean), by an independent conic solver; the stored float32 demands are
    inconsistent by about 1e-8 relative, as in the dualfix record."""
    cp = pytest.importorskip("cvxpy")
    pytest.importorskip("clarabel")
    N64 = inst["N"].astype(np.float64)
    V, E = N64.shape
    K = inst["K"]
    D = inst["d"].astype(np.float64).reshape(V, K)
    D = D - D.mean(axis=0, keepdims=True)
    F = cp.Variable((E, K), nonneg=True)
    S = cp.sum(F, axis=1)
    s32, r32 = MCF_SLOPES.astype(F32).astype(np.float64), MCF_INTER.astype(F32).astype(np.float64)
    phi = cp.max(cp.vstack([s32[j] * S + r32[j] for j in range(len(s32))]), axis=0)
    mu = float(np.float64(inst["sqrt_mu"]) ** 2)
    prob = cp.Problem(cp.Minimize(cp.sum(phi) + 0.5 * mu * cp.sum_squares(F)), [N64 @ F == D])
    prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert prob.status == "optimal", prob.status
    return float(prob.value)


# ---------------------------------------------------------------------------
# Shared compiled families and iterates (computed once per module)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pricing():
    inst = pricing_instance()
    prob, data = pricing_problem(inst)
    on = prob.compile(**PRICING_KW)
    off = prob.compile(**PRICING_KW, dual_routes=())
    s_off = off.solve_instance(data)
    s_on = on.solve_instance(data)
    return {"inst": inst, "data": data, "on": on, "off": off, "s_on": s_on, "s_off": s_off}


@pytest.fixture(scope="module")
def mcf():
    inst = mcf_instance()
    prob, data = mcf_problem(inst)
    on = prob.compile(**MCF_KW)
    off = prob.compile(**MCF_KW, dual_routes=())
    s_off = off.solve_instance(data)
    s_on = on.solve_instance(data)
    return {"inst": inst, "data": data, "on": on, "off": off, "s_on": s_on, "s_off": s_off}


@pytest.fixture(scope="module")
def lp():
    prob, data, arrays = lp_problem()
    on = prob.compile(block=50, max_blocks=200, tol=1e-4)
    return {"prob": prob, "data": data, "arrays": arrays, "on": on}


def _route(comp, name):
    fn, n_ops = comp._dual_routes[name]
    return fn, n_ops


# ---------------------------------------------------------------------------
# 1. The option
# ---------------------------------------------------------------------------
def test_option_values_and_refusals():
    assert DUAL_ROUTES == ("shift", "packing", "ridge")
    assert DUAL_ROUTES_DEFAULT == DUAL_ROUTES
    assert _check_dual_routes(None) == DUAL_ROUTES
    assert _check_dual_routes(()) == ()
    assert _check_dual_routes(["ridge", "shift"]) == ("shift", "ridge")
    assert _check_dual_routes(("packing",)) == ("packing",)
    for bad in ("shift", ("interval",), ("shift", "shift"), True, False, 3, ("Shift",)):
        with pytest.raises(ValueError):
            _check_dual_routes(bad)
    assert set(DUAL_ROUTE_SOURCES) == set(DUAL_ROUTES) == set(DUAL_ROUTE_OPERATOR_APPLICATIONS)
    assert DUAL_ROUTE_OPERATOR_APPLICATIONS == {"shift": 4, "packing": 3, "ridge": 3}


def test_compile_refuses_a_bad_option_before_any_work():
    prob, _, _ = lp_problem(m=4, n=5)
    with pytest.raises(ValueError, match="dual_routes"):
        prob.compile(dual_routes=("interval",))
    with pytest.raises(ValueError, match="dual_routes"):
        prob.compile(dual_routes="shift")


def test_meta_records_requested_eligible_and_enabled(lp):
    meta = lp["on"].meta["dual_routes"]
    assert meta["requested"] == list(DUAL_ROUTES)
    assert meta["eligible"] == {"shift": True, "packing": False, "ridge": False}
    assert meta["enabled"] == ["shift"]
    assert meta["operator_applications"] == DUAL_ROUTE_OPERATOR_APPLICATIONS
    assert meta["sources"] == DUAL_ROUTE_SOURCES
    off = lp["prob"].compile(block=50, max_blocks=200, tol=1e-4, dual_routes=())
    assert off.meta["dual_routes"]["requested"] == []
    assert off.meta["dual_routes"]["eligible"]["shift"] is True
    assert off.meta["dual_routes"]["enabled"] == [] and off._dual_routes == {}
    explicit_none = lp["prob"].compile(block=50, max_blocks=200, tol=1e-4, dual_routes=None)
    assert explicit_none.meta["dual_routes"]["enabled"] == ["shift"]


# ---------------------------------------------------------------------------
# 2. Eligibility
# ---------------------------------------------------------------------------
def test_eligibility_on_the_intended_structures(pricing, mcf):
    assert pricing["on"].meta["dual_routes"]["eligible"] == {"shift": True, "packing": True, "ridge": False}
    assert pricing["on"].meta["dual_routes"]["enabled"] == ["packing", "shift"]
    assert mcf["on"].meta["dual_routes"]["eligible"] == {"shift": True, "packing": False, "ridge": True}
    assert mcf["on"].meta["dual_routes"]["enabled"] == ["ridge", "shift"]
    assert pricing["off"].meta["dual_routes"]["enabled"] == []
    assert mcf["off"].meta["dual_routes"]["enabled"] == []


def test_a_subset_builds_only_what_is_requested():
    inst = mcf_instance()
    prob, _ = mcf_problem(inst)
    comp = prob.compile(**MCF_KW, dual_routes=("ridge",))
    assert comp.meta["dual_routes"]["enabled"] == ["ridge"]
    assert sorted(comp._dual_routes) == ["ridge"]
    comp = prob.compile(**MCF_KW, dual_routes=("packing",))      # requested, not eligible
    assert comp.meta["dual_routes"]["enabled"] == [] and comp._dual_routes == {}


def test_near_misses_are_not_eligible():
    small = pricing_instance(tasks=80, servers=4)
    # packing needs the weight -1 exactly and a single dual block
    prob, _ = pricing_problem(small, weight=-2.0)
    assert prob.compile(**PRICING_KW).meta["dual_routes"]["eligible"]["packing"] is False
    prob, _ = pricing_problem(small, extra_block=True)
    assert prob.compile(**PRICING_KW).meta["dual_routes"]["eligible"]["packing"] is False
    inst = mcf_instance()
    n = inst["N"].shape[1] * inst["K"]
    # ridge needs Diagonal(d > 0) on all n columns, and exactly one such block
    d = inst["sqrt_mu"] * np.ones(n, F32); d[3] = 0.0
    prob, _ = mcf_problem(inst, ridge_op=ocs.Diagonal(d))
    assert prob.compile(**MCF_KW).meta["dual_routes"]["eligible"]["ridge"] is False
    prob, _ = mcf_problem(inst, second_ridge=True)
    assert prob.compile(**MCF_KW).meta["dual_routes"]["eligible"]["ridge"] is False
    # no NonNeg primal block: nothing is eligible
    g = np.random.default_rng(0)
    A = g.normal(size=(20, 10)).astype(F32)
    x = ocs.Variable(10)
    bp = ocs.Parameter("b")
    prob = ocs.minimize([ocs.sum_squares(ocs.Dense(A) @ x - bp), ocs.l1(x)])
    meta = prob.compile(tol=1e-4).meta["dual_routes"]
    assert meta["eligible"] == {"shift": False, "packing": False, "ridge": False}
    assert meta["enabled"] == []


# ---------------------------------------------------------------------------
# 3. Validity of every route witness
# ---------------------------------------------------------------------------
def mcf_primal64(inst, x):
    E, K = inst["N"].shape[1], inst["K"]
    F = np.asarray(x, np.float64).reshape(E, K)
    S = F.sum(1)
    s32, r32 = MCF_SLOPES.astype(F32).astype(np.float64), MCF_INTER.astype(F32).astype(np.float64)
    phi = np.max(s32[None, :] * S[:, None] + r32[None, :], axis=1)
    return float(phi.sum() + 0.5 * float(np.float64(inst["sqrt_mu"]) ** 2) * np.sum(F * F))


def test_pricing_routes_are_valid_against_independent_float64(pricing):
    inst, comp, data = pricing["inst"], pricing["on"], pricing["data"]
    dense = comp._resolve_dense(data)
    x, y = pricing["s_off"].x, pricing["s_off"].y_iterate
    p_star = pricing_optimum(inst)
    for name in ("packing", "shift"):
        fn, _ = _route(comp, name)
        res = fn(x, y, dense)
        assert bool(res["valid"]), (name, int(res["status"]))
        w = np.asarray(res["dual"])
        member, d64 = pricing_float64(inst, w)
        assert member, name
        ob = comp.objective_bound(x, jnp.asarray(w), data, recover=False)
        assert ob["objective_lower_bound_available"], name
        assert ob["source"] == "supplied_witness"
        lower = float(res["bound"])
        assert lower == float(ob["objective_lower_bound"])
        assert lower <= d64 + 1e-6 * abs(d64), (name, lower, d64)
        assert d64 <= p_star + 1e-9 * abs(p_star), (name, d64, p_star)


def test_mcf_routes_are_valid_against_independent_float64(mcf):
    inst, comp, data = mcf["inst"], mcf["on"], mcf["data"]
    dense = comp._resolve_dense(data)
    x, y = mcf["s_off"].x, mcf["s_off"].y_iterate
    p_star = mcf_optimum_consistent(inst)
    for name in ("ridge", "shift"):
        fn, _ = _route(comp, name)
        res = fn(x, y, dense)
        if not bool(res["valid"]):
            assert name == "shift"          # the shift may be unreachable; it never awards then
            continue
        w = np.asarray(res["dual"])
        member, d64 = mcf_float64(inst, w)
        assert member, name
        ob = comp.objective_bound(x, jnp.asarray(w), data, recover=False)
        assert ob["objective_lower_bound_available"], name
        lower = float(res["bound"])
        assert lower == float(ob["objective_lower_bound"])
        assert lower <= d64 + 1e-6 * abs(d64), (name, lower, d64)
        # weak duality against the consistent instance (data inconsistent by ~1e-8)
        assert d64 <= p_star * (1 + 1e-6), (name, d64, p_star)


def test_ridge_bound_is_useful_where_the_evaluator_packet_is_not(mcf):
    """The MCF failure the route repairs: on the converged iterate the
    evaluator's packet holds only a trivial bound, the ridge witness meets (7)."""
    comp, data = mcf["on"], mcf["data"]
    dense = comp._resolve_dense(data)
    x, y = mcf["s_off"].x, mcf["s_off"].y_iterate
    res = _route(comp, "ridge")[0](x, y, dense)
    assert bool(res["valid"])
    J = mcf_primal64(mcf["inst"], x)
    lower = float(res["bound"])
    assert abs(J - lower) / max(abs(J), abs(lower)) <= 1e-5
    held = float(mcf["s_off"].objective_lower_bound)
    assert abs(J - held) / max(abs(J), abs(held)) > 1e-5


# ---------------------------------------------------------------------------
# 4. Directed rounding under jit and jit+vmap (optimization_barrier)
# ---------------------------------------------------------------------------
def test_barrier_keeps_the_upward_step_under_jit_and_vmap():
    def up(v32):
        with jax.enable_x64():
            v = v32.astype(jnp.float64) * (1.0 + 2.0 ** -40) + 2.0 ** -60
            r = v.astype(jnp.float32)
            back = jax.lax.optimization_barrier(r).astype(jnp.float64)
            r = jnp.where(back < v, jnp.nextafter(r, jnp.asarray(jnp.inf, jnp.float32)), r)
            return jnp.sum(r.astype(jnp.float64) < v).astype(jnp.int32)
    xs = jnp.asarray(np.random.default_rng(0).uniform(0.1, 10.0, 4096).astype(F32))
    assert int(up(xs)) == 0
    assert int(jax.jit(up)(xs)) == 0
    assert int(jnp.sum(jax.jit(jax.vmap(up))(xs.reshape(8, 512)))) == 0


def _packing_target(inst, y):
    y64 = np.asarray(y, np.float64)
    y64 = np.where(np.abs(y64) < TINY, 0.0, y64)
    t, B = inst["t"], inst["B"]
    R = inst["Rmat"].astype(np.float64)
    beta = np.maximum(-y64[t:], 0.0).reshape(R.shape[0], B)
    A = R.T @ beta
    cc = inst["c"].astype(np.float64).reshape(t, B)
    need = -cc - A
    return np.maximum(0.0, (need + 4 * EPS * (np.abs(cc) + A + np.abs(need)) + 4 * TINY).max(1))


def _ridge_target(inst, y):
    ys, ye, yp = mcf_split(inst, np.asarray(y, np.float64))
    yp = np.clip(yp, MCF_SLOPES.min(), MCF_SLOPES.max())
    K = inst["K"]
    te = (inst["N"].T @ ye.astype(F32).reshape(-1, K)).astype(np.float64).ravel()
    tp = np.repeat(yp, K)
    Q = te + tp
    mag = np.abs(te) + np.abs(tp)
    return np.maximum(0.0, (-Q + 3 * EPS * (mag + np.abs(Q)) + 4 * TINY) / float(inst["sqrt_mu"]))


def _batched(fn, xs, ys, dense):
    db = jax.tree_util.tree_map(lambda a: jnp.stack([a] * len(xs)), dense)
    with jax.enable_x64():                 # the package's batched-evaluator pattern
        return jax.jit(jax.vmap(fn))(jnp.stack(xs), jnp.stack(ys), db)


def _three_iterates(fam):
    """The returned iterate of the routes-off run and two perturbed copies."""
    x, y = fam["s_off"].x, fam["s_off"].y_iterate
    g = np.random.default_rng(1)
    ys = [y] + [y * (1 + 1e-3 * jnp.asarray(g.standard_normal(y.shape).astype(F32))) for _ in range(2)]
    return [x] * 3, ys


def test_packing_rounds_up_single_and_batched(pricing):
    inst, comp = pricing["inst"], pricing["on"]
    dense = comp._resolve_dense(pricing["data"])
    fn, _ = _route(comp, "packing")
    xs, ys = _three_iterates(pricing)
    t = inst["t"]
    batched = _batched(fn, xs, ys, dense)
    for i, (x, y) in enumerate(zip(xs, ys)):
        single = fn(x, y, dense)
        target = _packing_target(inst, y)
        got = -np.asarray(single["dual"][:t], np.float64)
        assert np.all(got >= target), int(np.sum(got < target))
        # batched equals single, bit for bit, in the witness, bound and verdict
        assert np.array_equal(np.asarray(batched["dual"][i]).view(np.uint32),
                              np.asarray(single["dual"]).view(np.uint32))
        assert bool(batched["valid"][i]) == bool(single["valid"])
        assert np.array_equal(np.asarray(batched["bound"][i]), np.asarray(single["bound"]), equal_nan=True)


def test_ridge_rounds_up_single_and_batched(mcf):
    inst, comp = mcf["inst"], mcf["on"]
    dense = comp._resolve_dense(mcf["data"])
    fn, _ = _route(comp, "ridge")
    xs, ys = _three_iterates(mcf)
    n = inst["N"].shape[1] * inst["K"]
    batched = _batched(fn, xs, ys, dense)
    # An entry whose float64 numerator cancels to the allowance's floor may be
    # formed with a different contraction on the device (measured: 0 against
    # 1.5e-36); that floor, 4 tiny / d, is excused, nothing above it.
    floor = 8 * TINY / float(inst["sqrt_mu"])
    for i, (x, y) in enumerate(zip(xs, ys)):
        single = fn(x, y, dense)
        target = _ridge_target(inst, y)
        got = np.asarray(single["dual"][:n], np.float64)
        assert np.all(got >= target - floor), int(np.sum(got < target - floor))
        assert np.array_equal(np.asarray(batched["dual"][i]).view(np.uint32),
                              np.asarray(single["dual"]).view(np.uint32))
        assert bool(batched["valid"][i]) == bool(single["valid"])


@pytest.mark.parametrize("family", ["pricing", "mcf"])
def test_shift_batched_equals_single(family, request):
    fam = request.getfixturevalue(family)
    comp = fam["on"]
    dense = comp._resolve_dense(fam["data"])
    fn, _ = _route(comp, "shift")
    xs, ys = _three_iterates(fam)
    batched = _batched(fn, xs, ys, dense)
    for i, (x, y) in enumerate(zip(xs, ys)):
        single = fn(x, y, dense)
        assert np.array_equal(np.asarray(batched["dual"][i]).view(np.uint32),
                              np.asarray(single["dual"]).view(np.uint32))
        assert bool(batched["valid"][i]) == bool(single["valid"])
        assert np.array_equal(np.asarray(batched["bound"][i]), np.asarray(single["bound"]), equal_nan=True)


def test_shift_on_a_dense_operator_batched_lanes_pass_the_gates(lp):
    """Under vmap the Dense float32 adjoint becomes a matrix product with another
    accumulation order, so on a Dense operator the batched shift is not bitwise
    the single one (measured 2026-10-23: the step differed by 8e-5 relative).
    The routes run only in the scalar controller.  Every batched witness the
    route calls valid must still pass the production gate and the float64
    membership test."""
    comp, data = lp["on"], lp["data"]
    s = comp.solve_instance(data)
    dense = comp._resolve_dense(data)
    fn, _ = _route(comp, "shift")
    g = np.random.default_rng(2)
    ys = [s.y_iterate] + [s.y_iterate + 1e-4 * jnp.asarray(g.standard_normal(s.y_iterate.shape).astype(F32))
                          for _ in range(2)]
    xs = [s.x] * 3
    batched = _batched(fn, xs, ys, dense)
    A, b, c = lp["arrays"]
    for i, (x, y) in enumerate(zip(xs, ys)):
        for res in (fn(x, y, dense), {k: v[i] for k, v in batched.items()}):
            if not bool(res["valid"]):
                continue
            w = np.asarray(res["dual"], np.float64)
            z = -(-A.astype(np.float64)).T @ w - c.astype(np.float64)
            assert np.all(w <= 0.0) and np.all(z <= 0.0)
            ob = comp.objective_bound(x, jnp.asarray(res["dual"]), data, recover=False)
            assert ob["objective_lower_bound_available"]
            assert float(ob["objective_lower_bound"]) <= float(b.astype(np.float64) @ w) + 1e-6


# ---------------------------------------------------------------------------
# 6. The controller
# ---------------------------------------------------------------------------
_LEGACY = ("check", "segment_blocks", "iterations", "legacy_residual", "bound_available",
           "objective_at_returned_point", "objective_lower_bound", "retained_previous_lower",
           "relative_gap", "relative_primal_infeasibility")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


@pytest.mark.parametrize("family", ["pricing", "mcf"])
def test_routes_change_a_run_only_by_an_earlier_accepted_check(family, request):
    fam = request.getfixturevalue(family)
    on, off = fam["s_on"], fam["s_off"]
    h_on, h_off = on.gap_history, off.gap_history
    # The routes never feed the iteration or the evaluator's chain: the
    # routes-on history is a prefix of the routes-off history in every
    # evaluator field, bit for bit.
    assert len(h_on) <= len(h_off)
    for a, b in zip(h_on, h_off):
        for key in _LEGACY:
            assert _same(a[key], b[key]), (key, a[key], b[key])
    assert on.iters <= off.iters
    # A route certifies these (the evaluator alone did not, at the shipped
    # budget, on 0.3.5); a retained route witness reports "retained_witness".
    assert on.status == "solved"
    assert on.objective_bound_source in set(DUAL_ROUTE_SOURCES.values()) | {"retained_witness"}
    assert any(h.get("dual_route_taken") for h in h_on)
    if off.status == "solved":
        assert on.iters < off.iters
    # Routes ran only at checks the evaluator's packet did not accept, and
    # every consultation was charged.
    ops = DUAL_ROUTE_OPERATOR_APPLICATIONS
    enabled = fam["on"].meta["dual_routes"]["enabled"]
    for h in h_on:
        assert "dual_routes" in h and sorted(h["dual_routes"]) == enabled
        assert h["dual_route_operator_applications"] == sum(ops[r] for r in enabled)
        assert h["dual_route_seconds"] > 0.0
        for r in enabled:
            assert h["dual_routes"][r]["operator_applications"] == ops[r]
    # The returned witness is the certifying one and re-evaluates bit for bit.
    assert on.dual_source == "certifying"
    ob = fam["on"].objective_bound(on.x, on.y, fam["data"], recover=False)
    assert ob["objective_lower_bound_available"]
    assert float(ob["objective_lower_bound"]) == float(on.objective_lower_bound)


@pytest.mark.parametrize("family", ["pricing", "mcf"])
def test_routes_off_record_no_route_work(family, request):
    fam = request.getfixturevalue(family)
    for h in fam["s_off"].gap_history:
        assert not any(k.startswith("dual_route") or k.startswith("extended_") for k in h)
    assert fam["s_off"].objective_bound_source not in DUAL_ROUTE_SOURCES.values()


def test_a_check_the_packet_accepts_runs_no_route(lp):
    """An instance the evaluator certifies at the start point (c >= 0, so x = 0
    with y = 0 is optimal): identical pair, status and history fields with and
    without the routes, and no route ran."""
    comp_on = lp["on"]
    comp_off = lp["prob"].compile(block=50, max_blocks=200, tol=1e-4, dual_routes=())
    data = {"b": lp["data"]["b"], "c": np.abs(lp["data"]["c"])}
    a, b = comp_on.solve_instance(data), comp_off.solve_instance(data)
    assert b.status == a.status == "solved"
    assert a.objective_bound_source not in DUAL_ROUTE_SOURCES.values()
    assert a.iters == b.iters
    for u, v in ((a.x, b.x), (a.y, b.y), (a.y_iterate, b.y_iterate)):
        assert np.array_equal(np.asarray(u).view(np.uint32), np.asarray(v).view(np.uint32))
    assert a.objective_lower_bound == b.objective_lower_bound
    assert "dual_routes" not in a.gap_history[-1]
    for u, v in zip(a.gap_history, b.gap_history):
        for key in _LEGACY:
            assert _same(u[key], v[key])


def test_the_batch_interface_is_unchanged_by_the_option(pricing):
    data = pricing["data"]
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([jnp.asarray(a)] * 2), data)
    on = pricing["on"].solve_many_results(batch)
    off = pricing["off"].solve_many_results(batch)
    for a, b in zip(on, off):
        assert a.status == b.status and a.iters == b.iters
        assert np.array_equal(np.asarray(a.x).view(np.uint32), np.asarray(b.x).view(np.uint32))
        assert np.array_equal(np.asarray(a.y).view(np.uint32), np.asarray(b.y).view(np.uint32))
        assert a.objective_bound_source not in DUAL_ROUTE_SOURCES.values()
