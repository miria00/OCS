# OCS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

OCS is an operator-native convex solver for large-scale problems on GPUs. It
solves `minimize sum_i f_i(K_i x + l_i)` where `f` is block separable and each
linear operator `K_i` is accessed **only** through its forward and adjoint
products (never formed, factored, or indexed), and each `f_i` through a
GPU-native proximal block. You compile a problem family once and then solve
many instances with changing data in a single batched GPU call.

## Installation

OCS is a GPU solver, so the recommended path installs JAX with CUDA. Known-good
versions are pinned in [`requirements.txt`](requirements.txt) (tested on Python
3.10).

### From source (to try it now)

```bash
git clone https://github.com/miria00/OCS.git
cd OCS
python -m venv .venv && source .venv/bin/activate

# GPU run (recommended): install JAX with CUDA 12 first, then everything else
pip install "jax[cuda12]==0.4.30"
pip install -r requirements.txt

# then install OCS itself (editable)
pip install -e .
```

For a **CPU-only trial** (small problems, slower), skip the `jax[cuda12]` line and
just run `pip install -r requirements.txt` (the plain `jax` pin is CPU). The
competitor-overlay figures also need NVIDIA cuOpt, which is GPU only and not on
PyPI; see the note at the bottom of `requirements.txt`.

### From PyPI (once released)

```bash
pip install ocs                       # core solver
pip install "ocs[validation,dev]"     # plus the CVXPY harness and figure scripts
```

Check the install:

```bash
python examples/hello_world.py        # a compressed-sensing LASSO solve
pytest -q                             # the test suite
```

## Background

OCS targets the problem

$$
\begin{array}{ll}
\text{minimize} & \sum_{i=1}^N f_i(K_i x + l_i),
\end{array}
$$

with variable $x \in \mathbf{R}^n$, block-separable closed convex $f_i$, offsets
$l_i \in \mathbf{R}^{m_i}$, and linear operators $K_i : \mathbf{R}^n \to \mathbf{R}^{m_i}$.
This form covers linear, quadratic, and second-order-cone programs, as well as
imaging, total-variation, optimal-transport, and matrix-variable problems. A
linear cost or a simple separable term is itself a block (with $K_i = I$).

What makes OCS operator-native is the contract on $K_i$: it is supplied only as
a pair of functions,

$$
v \mapsto K_i v \quad (\text{matvec}), \qquad w \mapsto K_i^T w \quad (\text{rmatvec}),
$$

so a subsampled FFT, a DCT, a Haar wavelet, a Gaussian blur, a Laplacian solve, a
gradient, a marginal (optimal transport), or a Kronecker product $A \otimes B$ is
applied directly to its structured argument and the $m_i \times n$ matrix is never
materialized. Everything downstream respects this: equilibration, the PDHG
iteration, and termination all run from a constant number of `matvec`/`rmatvec`
and prox calls. Each $f_i$ is a proximal block that declares how finely it can be
metric-scaled (scalar / diagonal / matrix); see
[`docs/scaling_contract.md`](docs/scaling_contract.md).

The whole solve runs on one adaptive-restart PDHG engine in float32, to a 1e-3
relative residual, and is `jax.vmap`-batched over instances.

## Usage: two entry points

OCS has exactly two ways in:

1. **Solve at scale** (the real solver). Define a family `min f(Kx+l)` with `K`
   matrix-free, compile once, and solve it, including the "one fixed `K`, many
   data" regime, on the GPU, at sizes where a formed-matrix solver runs out of
   memory.
2. **Sanity-check against CVXPY** (small problems): a separate harness that
   solves the same small problem with OCS and with SCS/Clarabel and compares. Use
   it to validate before scaling up. It is not part of the core solver.

### Entry point 1: solve at scale

Compressed-sensing LASSO with a subsampled-FFT operator,

$$
\begin{array}{ll}
\text{minimize} & \lambda \lVert x \rVert_1 + \tfrac{1}{2}\lVert M F x - y \rVert_2^2,
\end{array}
$$

where $K = M F$ (mask times orthonormal FFT) is never formed. Compile the family
once, then solve one instance, or a whole batch of instances with the same `K`
and different data `y`, in a single GPU call.

```python
import numpy as np
import jax.numpy as jnp
import ocs

n = 4096
K = ocs.FFTMask.random(n, ratio=0.25, seed=0)   # K: R^n -> R^{2m}, matrix-free
x = ocs.Variable(n)
y = ocs.Parameter("y")                           # per-instance data (varies per instance)

problem  = ocs.minimize([
    ocs.l1(x, lam=1e-3),                         # an l1 block (K = I)
    ocs.least_squares(K @ x - y),                # f(Kx + l),  l = -y
])
compiled = problem.compile()                     # compile the family + scaling ONCE

# one instance
sol = compiled.solve_instance({"y": y_meas})

# MANY instances, same K, different y: one batched GPU solve (the parametric regime)
xb, yb, iters, res = compiled.solve_many({"y": Y_batch})

# matrix-free certificate for a solution: primal / dual residual + surrogate gap
cert = compiled.certificate(sol.x, sol.y, {"y": y_meas})
```

Run it with `python examples/hello_world.py`. This is the regime where a
formed-matrix GPU solver would need the explicit `M F` (impossible at scale); see
the wedge figures in [`plots/`](plots/).

### Entry point 2: sanity-check against CVXPY

```python
from ocs.cvxpy_harness import validate_against_cvxpy
report = validate_against_cvxpy(cvxpy_problem)   # small LP / QP / SOCP
print(report["rel_err"])                         # OCS vs SCS objective agreement
```

For a whole sweep of classes and sizes, `python examples/reliability_battery.py`
reports how often OCS reaches tolerance and agrees with CVXPY (SCS / Clarabel).

## Experiments

The `examples/` directory holds the killer problems: optimal transport at scale
(`killer_ot_scale.py`, **10^4 x 10^4 = 10^8 variables**), MRI-SOCP, feature selection,
bin-packing, 3D imaging, Sylvester (with the CG path), Kronecker, multicommodity
flow with piecewise-linear congestion costs, and sparse deconvolution via the fast
Gauss transform, plus the figure scripts. To regenerate the figure deck and run
the checks:

```bash
python examples/make_plots.py              # writes plots/*.png
python examples/reliability_battery.py     # solve-rate table across classes -> plots/A2_*.png
python examples/matrix_metric_demo.py      # the matrix scaling tier (39 vs 588 iters)
python examples/run_competitors.py         # optional: cuOpt / HiGHS / SCS overlays
```

The reliability battery sweeps LP / QP / SOCP / SDP / exp / power / quantile /
logistic across sizes and seeds and reports, per class, how often OCS reaches its
matrix-free tolerance and how often it agrees with CVXPY (SCS / Clarabel). Figures
land in [`plots/`](plots/); [`plots/README.md`](plots/README.md) captions each one.

## Design notes

- **One solver.** The problem always runs on the operator-pure restart-PDHG
  engine in `ocs/solver/composite_pdhg.py`. The one exception is the pure
  least-squares regime (a lone quadratic through an operator, e.g. the Sylvester
  problem), where `compile(method="cg")` uses matrix-free CGLS, reaching the same accuracy in ~1500× fewer iterations.
- **Three scaling tiers, all live.** scalar / diagonal / matrix, chosen per block
  by structure (`docs/scaling_contract.md`). The matrix tier (`prox_matrix`, a
  dense `(Q+D)⁻¹` solve for quadratic blocks) preconditions ill-conditioned QPs.
- **Matrix-free termination.** Primal and dual fixed-point residuals, plus a
  surrogate duality gap, all from a constant number of oracle calls; for smooth
  blocks the exact gradient is substituted into the primal residual for tight
  termination. Validated against SCS.
- **Matrix-free equilibration.** The default (`equilibrate="sgd"`) adapts Diamond
  & Boyd's stochastic matrix-free equilibration (JOTA 2017) to the operator-native
  PDHG setting: projected SGD over row/column log-scalings from `matvec`/`rmatvec`
  only. Robust orders-of-magnitude win on ill-conditioned operators, harmless on
  well-conditioned ones.
- **The prox library** is organized as a system: a POGS-style calculus, a declared
  scaling tier per block, the LP/QP/SOCP cone set plus PSD / exp / power (public
  term builders `soc`/`psd`/`exp_cone`/`pow_cone`), the regression losses
  (`pinball`/`quantile`, `logistic`), piecewise-linear convex costs (`pwl`, for
  concave-utility / congestion problems), and type-grouped GPU dispatch: one
  batched `vmap` dispatch per block type, flat in the number of blocks. See
  [`docs/prox_inventory.md`](docs/prox_inventory.md).
- **Float32.** Required for fast GPU cuFFT on transform operators; the 1e-3
  relative tolerance sits far above f32 epsilon.
- **CVXPY is validation only**, a separate harness, not part of the core.

## Package layout

```
ocs/
  ir.py            the IR:  {x, (K_i, l_i, f_i)}
  operators/       LinearOperator (matvec/rmatvec): identity, diagonal, dense,
                   FFTMask, Marginal, Gradient2D/3D, Kronecker, Sylvester, ...
  prox/            base + POGS calculus + scaling contract; library, cones,
                   groups (type-grouped GPU dispatch)
  model/           surface API (Variable, Parameter, terms) + compile()
  solver/          restart-PDHG engine, matrix-free equilibration + termination
  cvxpy_harness/   thin validation-only bridge 
docs/ · examples/ · plots/ · tests/
```
