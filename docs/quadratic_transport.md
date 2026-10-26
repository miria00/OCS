# Prepared quadratic transport

`ocs.transport.PreparedQuadraticOT` is an opt-in NVIDIA GPU solver for

\[
\min_{X\geq0,\ X\mathbf1=p,\ X^T\mathbf1=q}
\langle C,X\rangle+\frac{\mu}{2}\|X\|_F^2.
\]

It uses strongly convex Chambolle–Pock, input-derived algebraic dual centering
and a fused Pallas update. It does not call a competing OT solver or change the
general OCS modeling interface. It requires dense square cost and plan storage;
there is no capacity-wedge or full-matrix-equilibration claim for this path.

Install GPU JAX 0.11.1 for your CUDA installation and `ocsolver[transport]`.
The transport extra pins JAX 0.11.1 and CuPy CUDA 12.x 13.6.0. A compatible
NVIDIA driver and CUDA 12 runtime remain system requirements for CuPy. The
tested B200 configuration uses JAX CUDA 13 alongside CUDA 12.8 for CuPy.
Plain `import ocs.transport` does not import CuPy or allocate GPU storage.

On an RTX 5090 reporting CUDA compute capability 12.0, construction supplies
the device metadata missing from JAX 0.11.1's Pallas/Triton registry. No external
shim is required. This compatibility handling is limited to that device and
pinned JAX version; it preserves existing registry entries and does not infer
support for other unknown GPUs. Optimization and assessment are unchanged.

```python
import numpy as np
from ocs.transport import PreparedQuadraticOT

n = 256
points = np.random.default_rng(7).normal(size=(2, n, 2)).astype(np.float32)
C = np.sum((points[0, :, None] - points[1, None, :]) ** 2, axis=-1)
C = np.ascontiguousarray(C / C.max(), dtype=np.float32)
p = np.full(n, 1 / n, dtype=np.float32)
with PreparedQuadraticOT(n, regularization=2*n*.01, return_capacity=1) as solver:
    result = solver.solve(C, p, p.copy())
assert result.solved
print(result.report['relative_width'], result.statistics['complete_warm_s'])
X = result.plan  # remains valid after the prepared solver closes
```

Inputs must be C-contiguous float32 NumPy arrays with finite, nonnegative
entries. Marginal totals must be positive and exactly equal when summed in
float64. No normalization or cost/regularization rescaling is performed.
The scalar regularization is represented in float32. Callers must leave inputs
unchanged for the duration of a call. One session handles calls serially.
Sessions cannot be copied or serialized; construct a new session instead.

Construction compiles dummy-shaped operations and allocates workspace plus
one to three distinct pinned host output slots. Each solve starts from a fresh
zero primal and centering derived from the current input. No prior solution is
reused. An admitted call consumes a slot even if it fails; allocate a new
session after exhaustion. `close()` releases internal references and preserves
all previously issued output arrays. Outputs are initially read-only NumPy
arrays, not a tamper-proof security boundary. Reports and statistics are immutable.

Every accepted return passes relative marginal, nonnegativity and objective
bracket tests at **1e-3**. Optimization uses float32; assessment uses numerical
float64, not rigorous interval bounds. The upper objective is evaluated on a
feasible rounding of the raw plan; the raw returned plan must itself pass the
marginal and nonnegativity tolerances. Maximizing potentials `u, v` give lower

\[
p^Tu+q^Tv-\frac{1}{2\mu}\|[u\mathbf1^T+\mathbf1v^T-C]_+\|_F^2.
\]

Screening occurs every 100 updates and at the final cap. The default cap is
20,000 updates. A complete check alone awards accuracy. Failure raises
`OTConvergenceError` with its last report; it does not return a solved result.
Invalid input, nonfinite arithmetic and concurrent use also fail explicitly.

For warm benchmarking, time the whole `solve` call with current host inputs.
Charge staging, validation, transfers, fresh initialization, optimization,
all checks and complete host returns. Record construction separately: it pays
compilation, workspace and all output allocations. File loading, post-return
independent validation and publication are separate costs. This contract is
for a finite set of prepared outputs, not unlimited output storage or cold-start
performance. Experimental source timings do not replace installed-wheel results.

## Batches

Prepare the same public solver with a fixed `batch_size`, then use the same
`solve_many_results(data)` naming as compiled OCS families:

```python
b = 16
with PreparedQuadraticOT(n, regularization=2*n*.01,
                        batch_size=b, return_capacity=1) as solver:
    rows = solver.solve_many_results({
        'cost': [C.copy() for _ in range(b)],
        'p': [p.copy() for _ in range(b)],
        'q': [p.copy() for _ in range(b)],
    })
assert all(row.solved for row in rows)
plans = [row.plan for row in rows]
```

Each field accepts either a stacked C-contiguous float32 array with a leading
batch dimension or a list/tuple of individually contiguous arrays. A sequence
avoids a separate dense packing copy before the call. Each problem may have its
own costs and marginals; shape and regularization are shared. `return_capacity`
counts whole batches, so one slot owns B independent, initially read-only plans
and their potentials. Repeated calls start fresh and use distinct output storage.
Each result remains valid after subsequent calls and session closure. Calls on
one session are serial; vectorization runs the problems within a call together.

This path uses JAX `vmap` over the same fused OT updates and complete numerical
assessment. All lanes advance in lockstep. A screen triggers assessment when
all lanes are eligible, or at the exact iteration cap. Only a complete passing
report for **every current lane** ends the batch early. An easier lane can do
extra iterations while others finish. Returned arrays and reports always refer
to the final common endpoint; no earlier report is reused to accept a later
point. At the cap, the method returns each actual result with `solved` and
`statistics['termination']` (`criterion_met` or `iteration_limit`). Check each
row; a batch containing a miss is not an all-success result. Invalid input or
nonfinite/invalid arithmetic refuses the whole call and never awards success.

The existing general `compiled.solve_many_results` controller is unchanged:
it preserves each lane's individual stopping pair. The specialized OT lockstep
controller avoids dense lane-selection copies and amortizes launch/check costs.
It is an explicit specialization, not automatic dispatch from an arbitrary
compiled cone problem. The scalar OT method continues to raise
`OTConvergenceError` on a miss; the batch method returns per-lane outcomes.

Time the complete `solve_many_results` call for warm batch throughput. All
staging, initialization, work on easy lanes, assessment, and full host outputs
are included. Preparation and its B-times output allocation must be reported
separately. Batched execution uses more GPU memory and is not automatically
faster for all problem sizes or batch sizes. The measured QDROT comparison uses
the authors' serial interface on the same GPU, not a bespoke batched QDROT kernel.
