# OCS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

OCS is an operator-native convex solver. It solves

    minimize  sum_i f_i(K_i x + l_i)

where each `K_i` is reached only through its forward and adjoint products and
each `f_i` is a proximal block evaluated on the GPU. The constraint matrix is
never formed. A problem family is compiled once and then solved for many
instances with changing data, one at a time or as a batch.

This is version **0.3.8**. It decides second-order cone and PSD dual-domain
membership, and PSD primal feasibility, in scoped float64 on the represented
float32 values, and its PSD cone never passes an exactly zero matrix to the
eigensolver (see `CHANGES.md`). It retains everything in 0.3.7: recovered
bounds validated on the actual returned witness, bounded opt-in refinement,
reduced large-array storage and compilation overhead, the objective-gap stall
fix, stable opt-in SOC/PSD metric construction, and the prepared quadratic
transport solver.
Before publication, install the qualified wheel file directly; the index
commands below apply once that version is published.

## Installation

For repeated dense quadratic OT on an NVIDIA GPU, see
[prepared quadratic transport](docs/quadratic_transport.md). Its optional
`transport` extra and `ocs.transport.PreparedQuadraticOT` API use a specialized
first-order implementation and an explicit numerical objective-bracket test.
This path keeps the general solver defaults unchanged. Warm comparison results
exclude preparation and do not establish a capacity or full-metric advantage.

This successor targets Linux and Python 3.12 with `jax==0.11.1` and
`numpy>=1.24`. The large-array lowering implementation uses JAX's compiler
internals, so this package pins the JAX version it qualifies. Older environment
pins in `requirements.txt` are historical records, not this installation route.
The GPU procedure that this release was qualified on is Python 3.12 with
JAX 0.11.1 and CUDA 12 on an NVIDIA RTX 5090, with the compiler settings below.
These settings avoid the GPU compiler failures encountered during qualification;
they change compilation scheduling, not OCS's accuracy criterion.

GPU, the qualified procedure:

```bash
python3.12 -m venv ocs-env
source ocs-env/bin/activate
python -m pip install "jax[cuda12]==0.11.1"
python -m pip install "ocsolver==0.3.8"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=1"
python -c "import jax; print(jax.devices())"     # expect a CudaDevice
```

Pin the version to reproduce the qualified installation. CPU only, for small problems:

```bash
python -m pip install "jax==0.11.1" "ocsolver==0.3.8"
```

Confirm the installed identity before reporting a result:

```bash
python -c "import ocs, importlib.metadata as m; print(ocs.__version__, m.version('ocsolver'))"
```

Both must print `0.3.8`.

## First solve

Nonnegative least squares with the operator supplied as a function. The
solution is approximately `[0, 1, 1]`. The same code is in
`examples/hello_world.py`.

```python
import jax.numpy as jnp
import ocs

weights = jnp.array([1.0, 2.0, 3.0])
K = ocs.LinearOperator.from_forward(lambda v: weights * v, n=3, m=3)
x = ocs.Variable(3)
b = ocs.Parameter("b")
family = ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - b)])
compiled = family.compile(tol=1e-3)
sol = compiled.solve_instance({"b": jnp.array([-1.0, 2.0, 3.0])})
print(sol.status, sol.accuracy_criterion, sol.objective_gap, sol.x)
```

`from_forward` builds the adjoint of a JAX-linear function; a
`LinearOperator` subclass can supply `matvec` and `rmatvec` directly. The
library operators include subsampled FFT, 2-D and 3-D gradients, marginal
maps, Kronecker and Sylvester products, and multicommodity-flow incidence.

## Accuracy: what a status means

`sol.status` is one of `solved`, `inaccurate`, `unverified`,
`primal_infeasible`, `unbounded` and `numerical_error`.
`sol.accuracy_criterion` names the test that awarded it. Under the default,
`objective_gap`, the policy is `returned-point-three-condition-scaleinv-v2`:

- **`solved`** means that at the returned point `x`, with an exactly
  dual-feasible witness `y`, the cost `J(x)` is finite, the relative primal
  infeasibility is at most `tol`, and `|J(x) - D(y)| <= tol * max(|J(x)|, |D(y)|)`.
  This is an approximate-KKT acceptance of the kind SCS and OSQP use, and it
  is invariant to rescaling the problem data. It is **not** a bound on the
  optimal value: the cost at an approximately feasible point bounds nothing.
- **`inaccurate`** means the test was evaluated and failed, or the iteration
  budget ran out first; `sol.accuracy_detail` says which.
- **`unverified`** means nothing could be decided: no block contract exists
  for this family, no witness was found, or the computed gap and `tol` differ
  by less than float32 can resolve at this objective scale. An `unverified`
  result is never reported as optimal.

The witness `y` is a candidate built from the current dual iterate: the
iterate itself, a violation-corrected dual, the dual scaled along its own ray,
or a blend toward a feasible anchor. A candidate counts only when it passes the
exact conjugate-domain and coupling tests, and the stronger valid bound is kept
from check to check. Sensitive tests use scoped float64 with a rounding
envelope: PSD cone membership, coupling of an l1 primal over subsampled Fourier
operators, and the marginal LP's represented price/cost slack. A witness whose
sign is unresolved is refused. When none of these meets the criterion at a check, the
single-instance controller (`solve_instance`) also tries the dual-bound routes
that fit the family's structure: the recession shift, the packing-LP assignment
dual and the compound ridge. They are selected by
`compile(dual_routes=("shift", "packing", "ridge"))`, the default, and turned
off with `dual_routes=()`. Their candidates pass the same tests, their time and
operator applications are recorded in `sol.gap_history`, and
`sol.objective_bound_source` names the candidate that certified.

`sol.objective_gap`, `sol.objective_lower_bound`, `sol.objective_primal_value`
and `sol.relative_primal_infeasibility` carry the quantities tested.
`compiled.acceptance(report)` applies the same decision to any bound report
and also answers the stricter **bracket** question, `target_met_bracket`,
which carries weak duality and is claimed only when the upper bound is
justified by an exactly feasible point. `sol.residual` is this algorithm's
relative prox-graph residual; it is reported and decides nothing. Pass
`accuracy="residual"` to `solve_instance` to accept on it instead, as
releases before 0.2.0 did.

`sol.termination` distinguishes meeting the criterion from an iteration limit,
numerical failure, or `"stalled"`. The objective-gap controller can stop after
three check intervals without one-percent relative improvement in the gap,
residual or primal feasibility. A stall does not award success, relax `tol`,
or prove that further progress is impossible. Scalar and batch result interfaces
preserve the pair and report at each lane's own stopping check. The batch driver
can still evaluate terminal lanes while other lanes continue; those evaluations
do not replace the stored result. The raw `solve_many` interface retains its
existing residual-based behavior.

## Equilibration

The default, `equilibrate="sgd"`, is the stochastic matrix-free equilibration
of Diamond and Boyd, from forward and adjoint products only. It uses diagonal
metrics where the proximal block supports them and scalar metrics on blocks
whose default proximal interface accepts only a scalar. This is what every
example above uses.

A richer metric is declared, never selected automatically, through one call:

```python
compiled = family.compile(
    tol=1e-5,
    equilibrate="construct",
    metric_classes={0: "full"},        # {dual_term_index: class}
)
print(compiled.meta["metric_construction"])   # what was built, from how many products
```

The classes are `"scalar"`, `"diagonal"`, `"full"`, `"spatial_full"` and `"automorphism"`;
`(class, "per_block")` or `(class, "term")` says which mathematical blocks
share one metric. Dual terms are indexed in the order they appear, skipping
the primal term. Every construction uses products only and records its
product count. Supported structures in this release:

| class | supported on | notes |
|---|---|---|
| `scalar` | every supported proximal block | one scalar per mathematical block, or shared across a term |
| `diagonal` | blocks with a coordinatewise metric prox; supported quadratic, `L2` and `GroupL2` structures | availability and sharing follow the block's metric contract; not arbitrary weights on every cone |
| `full` | separable quadratic blocks, `L2` and `GroupL2` blocks | one rotated `k x k` shape, shared across the identical blocks of a term |
| `spatial_full` | one `SecondOrderCone` term | `diag(a, Q diag(b) Q^T)`, block size at most 1024 |
| `automorphism` | `SecondOrderCone`, `SOCBatch`, `PSDCone` | cone-preserving full metric, per block or shared by a term |

In particular, the constructor **rejects `diagonal` for `SecondOrderCone`,
`SOCBatch` and `PSDCone`**. Use `scalar`, or request `automorphism` for a
cone-preserving full metric. For a PSD block, a diagonal congruence
`Z -> D Z D` belongs to the automorphism class; it is not an arbitrary diagonal
weighting of the flattened matrix entries. Other coupled blocks whose proximal
interface supports only scalar metrics also reject unsupported diagonal
declarations. These are implementation support limits, not claims that other
weighted projections are mathematically impossible.

Anything else is refused by name with a `ValueError` that says why:
`spatial_full` on a batched cone, a PSD cone or a non-cone block;
`automorphism` on exponential, power or nonsymmetric cones; a block size above
the constructor's limit.
Whether a richer class pays is measured, not assumed: on the families it was
qualified on it wins at strong within-block coupling with a reused operator,
and on second-order cones of block size 128 and above, and it loses or ties
elsewhere. `equilibrate="scalar"` and `"exact"` remain available; `"exact"`
needs squared-operator oracles and is outside the two-product interface.

In 0.3.2, `equilibrate="construct", metric_classes={i: "automorphism"}`
opts term `i` into stable automatic symmetric-cone construction. SOC updates
use scaled exponentials and a log-determinant identity, clipping spectral logs
before exponentiating. PSD updates bound each carrier's step by
`eta_eff = min(eta, 0.5 / ||Y||_F)`. This changes the old PSD step schedule;
the construction report records the effective steps. Invalid arithmetic fails
explicitly, and the existing metric-domain and measured-norm safeguards remain.
Scalar/diagonal defaults are unchanged. A full metric is not automatically
faster: qualification of the arithmetic is separate from a performance benefit.

## Batching

The specialized quadratic OT solver also supports prepared GPU batches through
`PreparedQuadraticOT(..., batch_size=B).solve_many_results(data)`, returning one
`QuadraticOTResult` per problem. It vectorizes the fused OT update and checks
each actual returned pair. Its lockstep policy and finite output ownership are
documented in [quadratic transport](docs/quadratic_transport.md#batches).

`compiled.solve_many_results(data)` solves a batch in one GPU call and returns
one `Solution` per row, each held to the **same** acceptance test as
`solve_instance`, through the same code, so no row is accepted on a test the
single solve would refuse:

```python
rows = compiled.solve_many_results({"b": jnp.array([[-1.0, 2.0, 3.0], [2.0, -2.0, 6.0]])})
for r in rows:
    print(r.status, r.objective_gap, r.iters)
```

`compiled.solve_many(data)` is the **legacy** raw interface: it returns device
arrays `(x, y, blocks, residuals)` from the residual-driven loop and does not
apply the default accuracy contract. Use it only when you score the points
yourself.

## Limits

- The method assumes a saddle point. Infeasibility and unboundedness are
  detected only for the supported block domains; an absent certificate is not
  proof of feasibility.
- Arithmetic is float32. A tolerance finer than float32 can resolve at the
  problem's objective scale is refused as `unverified`, not granted.
- Exponential and power cones have bounded validation; general cone-program
  coverage is experimental.
- `ocs.cvxpy_mf` solves a CVXPY problem matrix-free: CVXPY lowers the problem
  to cones and OCS stops the chain before matrix stuffing, so the constraint
  matrix is never formed. It uses private CVXPY reductions and is tested
  against CVXPY 1.6 to 1.9. Sparse matrix constants use ordinary JAX
  gather/scatter-add operations, without JAX's experimental sparse extension.
  `ocs.cvxpy_harness` is a **validation harness**,
  used explicitly and never by the solver: it materializes the stuffed matrix
  to compare small problems against a reference solver and is for bounded
  validation only. Both require the `validation` extra:
  `pip install "ocsolver[validation]==0.3.8"`.
- Research-only paths (the online and `diagpp`
  equilibration variants, `ocs.experimental`) are reachable by name and change
  nothing unless asked for.
- No performance figure in this README is a benchmark claim; the measured
  campaigns are reported in the paper that accompanies the release.

## Source and reporting

The source tree, tests and examples ship in the source distribution
(`ocsolver-0.3.8.tar.gz`). Run the tests from an installed package with
`python -m pytest tests` from the unpacked source distribution. Report a
problem with the output of the identity check above, the `Solution` fields
`status`, `accuracy_criterion`, `accuracy_detail` and `gap_history`, and a
problem small enough to attach. A public repository URL is not advertised until
its location is confirmed.
