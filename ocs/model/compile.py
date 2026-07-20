"""compile(): lower a ProblemFamily to a CompiledFamily.

Once per family: route the primal term g (its operator is the identity), stack
the dual operators K_i, run matrix-free DIAGONAL equilibration (respecting each
block's scaling class), group blocks by type for GPU dispatch, and assemble the
engine.  Then solve many instances with changing data (offsets l and cost c) on
the GPU via solve_instance / solve_many, with optional warm starts.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..operators.common import Identity
from ..prox.base import ProxBlock
from ..prox.groups import build_groups, grouped_moreau_dual
from ..solver.composite_pdhg import make_solver
from ..solver.equilibrate import diagonal_equilibration, scalar_steps
from ..solver.equilibrate_sgd import smfe_equilibration
from ..solver.stopping import make_residual


@dataclass
class Solution:
    x: object
    y: object
    iters: int
    residual: float


def _overrides(block, name):
    return getattr(type(block), name) is not getattr(ProxBlock, name)


class CompiledFamily:
    def __init__(self, solver, resolve_data, report, n, m, meta):
        self._solver = solver
        self._resolve = resolve_data
        self._report = report
        self.n, self.m = n, m
        self.meta = meta

    def _run(self, data_b, x0, y0):
        return self._solver(data_b, x0, y0)

    def solve_instance(self, data=None, warm_start=None):
        d = self._resolve(data or {})
        db = jax.tree_util.tree_map(lambda a: a[None], d)
        x0 = (warm_start[0][None] if warm_start else jnp.zeros((1, self.n), jnp.float32))
        y0 = (warm_start[1][None] if warm_start else jnp.zeros((1, self.m), jnp.float32))
        x, y, blocks, res = self._run(db, x0, y0)
        return Solution(x[0], y[0], int(blocks[0]) * self.meta["block"], float(res[0]))

    def solve_many(self, data_batch, warm_start=None):
        d = jax.vmap(self._resolve)(data_batch)
        N = jax.tree_util.tree_leaves(d)[0].shape[0]
        x0 = warm_start[0] if warm_start else jnp.zeros((N, self.n), jnp.float32)
        y0 = warm_start[1] if warm_start else jnp.zeros((N, self.m), jnp.float32)
        return self._run(d, x0, y0)           # (x, y, blocks, res)

    def certificate(self, x, y, data=None):
        """Report the matrix-free certificate for a single (x, y): the primal and
        dual fixed-point residuals, and (when every block exposes a conjugate)
        the relative duality gap."""
        return self._report(x, y, self._resolve(data or {}))

    def __repr__(self):
        return (f"CompiledFamily(n={self.n}, m={self.m}, "
                f"dual_blocks={self.meta['n_dual']}, "
                f"prox_kernels={self.meta['n_groups']}, "
                f"equilibration={self.meta['equilibrate']}, "
                f"||K_scaled||~{self.meta['scaled_norm']:.3g})")


class CGCompiledFamily:
    """Compiled pure-least-squares family solved by matrix-free CGLS."""
    def __init__(self, solver, resolve_b, matvec, rmatvec, n, m, meta):
        self._solver = solver
        self._resolve = resolve_b
        self._matvec, self._rmatvec = matvec, rmatvec
        self.n, self.m, self.meta = n, m, meta

    def solve_instance(self, data=None, warm_start=None):
        b = self._resolve(data or {})
        x0 = (warm_start[0][None] if warm_start else jnp.zeros((1, self.n), jnp.float32))
        x, k, rel = self._solver(b[None], x0)
        return Solution(x[0], None, int(k[0]), float(rel[0]))

    def solve_many(self, data_batch, warm_start=None):
        b = jax.vmap(self._resolve)(data_batch)
        N = b.shape[0]
        x0 = warm_start[0] if warm_start else jnp.zeros((N, self.n), jnp.float32)
        return self._solver(b, x0)                 # (x, iters, rel_resid)

    def certificate(self, x, y=None, data=None):
        b = self._resolve(data or {})
        rho = self.meta["rho"]
        s = self._rmatvec(self._matvec(x) - b) + rho * x
        rel = float(jnp.linalg.norm(s) / (jnp.linalg.norm(self._rmatvec(b)) + 1e-30))
        return {"r_primal": rel, "r_dual": 0.0, "gap": float("nan")}

    def __repr__(self):
        return (f"CGCompiledFamily(n={self.n}, m={self.m}, method=cgls, "
                f"cg_iters={self.meta['cg_iters']}, rho={self.meta['rho']:g})")


def _compile_cg(family, n, *, cg_iters, rho, tol):
    from ..prox.library import SumSquares
    from ..solver.cg import make_cgls_solver
    terms = family.terms
    if len(terms) != 1 or not isinstance(terms[0].block, SumSquares) or family.cost_fn is not None:
        raise ValueError("method='cg' needs exactly one least-squares term "
                         "1/2||K x - b||^2 and no linear cost")
    t = terms[0]; op = t.op; m = op.shape[0]
    solver = make_cgls_solver(op.matvec, op.rmatvec, n, m, iters=cg_iters, rho=rho, tol=tol)

    def resolve_b(raw):
        l = t.offset_fn(raw) if t.offset_fn is not None else 0.0
        return -jnp.broadcast_to(jnp.asarray(l, jnp.float32), (m,))   # min 1/2||Kx+l||^2

    meta = {"method": "cg", "cg_iters": cg_iters, "rho": rho, "n_dual": 1, "n_groups": 1,
            "scaled_norm": float("nan"), "equilibrate": "none", "block": 1,
            "gap_available": False, "scaling_classes": ["matrix"], "eval_kinds": ["closed_form"]}
    return CGCompiledFamily(solver, resolve_b, op.matvec, op.rmatvec, n, m, meta)


def compile_family(family, *, block=100, max_blocks=200, tol=1e-3,
                   probes=32, equilibrate="sgd", adaptive_omega=True,
                   method="pdhg", cg_iters=300, rho=0.0):
    n = family.n

    # ---- pure least-squares: conjugate-gradient path --------------------
    # min 1/2||K x - b||^2 (one SumSquares term, no g, no cost) is the CG/LSQR
    # regime, where restart-PDHG plateaus.  Route it to matrix-free CGLS.
    if method == "cg":
        return _compile_cg(family, n, cg_iters=cg_iters, rho=rho, tol=tol)

    # ---- route primal g vs dual blocks ----------------------------------
    g_block, dual = None, []
    for t in family.terms:
        if g_block is None and isinstance(t.op, Identity) and t.offset_fn is None:
            g_block = t.block
        else:
            dual.append(t)
    if not dual:
        raise ValueError("need at least one term through a non-identity operator")

    ops = [t.op for t in dual]
    blocks = [t.block for t in dual]
    dims = [op.shape[0] for op in ops]
    m = sum(dims)
    slices, a = [], 0
    for d in dims:
        slices.append((a, a + d)); a += d
    scalar_spans = [sl for blk, sl in zip(blocks, slices)
                    if blk.scaling_class == "scalar"]

    # ---- stacked operator (matrix-free) ---------------------------------
    def matvec(x):
        return jnp.concatenate([op.matvec(x) for op in ops])

    def rmatvec(y):
        return sum(op.rmatvec(y[s:e]) for op, (s, e) in zip(ops, slices))

    # ---- matrix-free equilibration (once) -------------------------------
    # operator-aware skip: if every dual operator is already well conditioned
    # (e.g. an orthonormal transform), equilibration has nothing to do.
    if equilibrate == "sgd" and all(getattr(op, "well_conditioned", False) for op in ops):
        equilibrate = "scalar"

    if equilibrate == "sgd":                     # Diamond-Boyd matrix-free SMFE
        tau, sigma, scaled_norm = smfe_equilibration(
            matvec, rmatvec, n, m, scalar_spans)
    elif equilibrate == "diagonal":              # one-shot Hutchinson (experimental)
        tau, sigma, scaled_norm = diagonal_equilibration(
            matvec, rmatvec, n, m, scalar_spans, probes=probes)
    elif equilibrate == "scalar":
        t0, s0, scaled_norm = scalar_steps(matvec, rmatvec, n)
        tau = jnp.full((n,), t0, jnp.float32)
        sigma = jnp.full((m,), s0, jnp.float32)
    else:
        raise ValueError("equilibrate must be 'scalar', 'diagonal', or 'sgd'")

    # ---- proxes (dual prox uses type-grouped GPU dispatch) --------------
    groups = build_groups(blocks, slices)
    dual_apply = grouped_moreau_dual(groups, m)

    def prox_primal(xt, tw, data):
        xt = xt - tw * data[1]                       # fold in cost c
        return g_block.prox(xt, tw) if g_block is not None else xt

    def prox_dual(yt, sw, data):
        return dual_apply(yt + sw * data[0], sw)     # fold in offset l

    # smooth dual blocks: (slice, grad f_i) -- used for exact termination
    smooth_terms = [(s, e, blk.smooth_grad) for blk, (s, e) in zip(blocks, slices)
                    if getattr(blk, "smooth_grad", None) is not None]
    residual = make_residual(matvec, rmatvec, prox_primal, prox_dual, tau, sigma,
                             smooth_terms=smooth_terms, offset_index=0)
    solver = make_solver(matvec, rmatvec, tau, sigma, prox_primal, prox_dual,
                         residual, n, m, block=block, max_blocks=max_blocks, tol=tol,
                         adaptive_omega=adaptive_omega)

    # ---- per-instance data:  (l, c) -------------------------------------
    def resolve_data(raw):
        parts = []
        for t, (s, e) in zip(dual, slices):
            off = t.offset_fn(raw) if t.offset_fn is not None else 0.0
            parts.append(jnp.broadcast_to(jnp.asarray(off, jnp.float32), (e - s,)))
        l = jnp.concatenate(parts)
        c = (family.cost_fn(raw) if family.cost_fn is not None
             else jnp.zeros(n, jnp.float32))
        return (l, c)

    # ---- certificate reporting (primal/dual residual + optional gap) ----
    gap_ok = all(_overrides(b, "conj_value") for b in blocks) and \
        all(_overrides(b, "value") for b in blocks) and \
        (g_block is None or _overrides(g_block, "conj_value"))

    @jax.jit
    def report(x, y, data):
        l, c = data
        Kx = matvec(x); s = Kx + l
        y_eff = y
        for (a, b, grad) in smooth_terms:                  # exact dual for smooth blocks
            y_eff = y_eff.at[a:b].set(grad(s[a:b]))
        r_p = jnp.linalg.norm(x - prox_primal(x - tau * rmatvec(y_eff), tau, data)) \
            / (1.0 + jnp.linalg.norm(x))
        dres = y - prox_dual(y + sigma * Kx, sigma, data)
        for (a, b, _g) in smooth_terms:
            dres = dres.at[a:b].set(0.0)
        r_d = jnp.linalg.norm(dres) / (1.0 + jnp.linalg.norm(y))
        gap = jnp.asarray(jnp.nan, jnp.float32)
        if gap_ok:
            # Fenchel dual of  min c^T x + g(x) + sum_i f_i(K_i x + l_i):
            #   D(y) = <l, y> - g^*(-K^T y - c) - sum_i f_i^*(y_i).
            # Indicators contribute 0 (their +inf is feasibility, certified by the
            # residuals), so we take the finite part of every value / conjugate.
            fin = lambda z: jnp.nan_to_num(z, posinf=0.0, neginf=0.0)
            s = matvec(x) + l
            primal = jnp.dot(c, x)
            primal += fin(g_block.value(x)) if g_block is not None else 0.0
            for blk, (a, b) in zip(blocks, slices):
                primal += fin(blk.value(s[a:b]))
            dual = jnp.dot(l, y)
            if g_block is not None:
                dual -= fin(g_block.conj_value(-rmatvec(y) - c))
            for blk, (a, b) in zip(blocks, slices):
                dual -= fin(blk.conj_value(y[a:b]))
            gap = jnp.abs(primal - dual) / (1.0 + jnp.abs(primal) + jnp.abs(dual))
        return {"r_primal": r_p, "r_dual": r_d, "gap": gap}

    def report_py(x, y, data):
        out = report(x, y, data)
        return {k: float(v) for k, v in out.items()}

    meta = {"block": block, "n_dual": len(dual), "n_groups": len(groups),
            "scaled_norm": scaled_norm, "gap_available": gap_ok,
            "equilibrate": equilibrate,
            "scaling_classes": [b.scaling_class for b in blocks],
            "eval_kinds": [b.eval_kind for b in blocks]}
    return CompiledFamily(solver, resolve_data, report_py, n, m, meta)
