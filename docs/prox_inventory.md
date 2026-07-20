# Prox libraries — inventory, GPU status, and how OCS organizes its own

Stephen's ask: *don't just list prox functions.* First, what already exists and
which are GPU; then organize the library as a system with a clean statement of
what each block can compute.

## What already exists

| library | language | GPU? | autodiff? | cone coverage |
|---|---|---|---|---|
| ProximalOperators.jl | Julia | no | no | broad (norms, cones, indicators) |
| pyproximal (PyLops) | Python | partial (CuPy) | no | moderate |
| TFOCS | MATLAB | no | no | moderate |
| Proximity Operator Repository (Chierchia) | MATLAB | no | no | catalog, broad |
| Parikh–Boyd reference code | MATLAB/Python | no | no | teaching set |
| **POGS** (Fougner–Boyd) | C++/CUDA | **yes** | no | separable funcs + calculus |
| **ProxImaL** (Heide … Boyd, 2016) | Python + **Halide** | **yes** | no | imaging proxes/operators |
| cone projections inside SCS / COSMO / Clarabel | C/Rust | SCS/cuClarabel yes | no | full cone set, bundled in a solver |
| torch/JAX ad-hoc (`relu`, soft-threshold, …) | Python | yes | yes | orthant / box / norm balls only |

**Reading of the landscape.** Coverage and GPU support are negatively correlated.
The *simple* projections (nonnegative orthant = a ReLU, box, norm balls) are
already GPU-native and differentiable in array frameworks. The *harder* cone
projections (second-order, semidefinite, exponential, power) exist only inside
CPU solvers or, on GPU, bundled inside a complete solver — not exposed as
standalone, composable, GPU-native, differentiable operators.

**Halide / ProxImaL** is the GPU prox work Stephen remembers: Heide, Diamond,
Nießner, Ragan-Kelley, Heidrich, Wetzstein, *ProxImaL* (SIGGRAPH 2016) compiled
proxes + linear operators to fast CPU/GPU code via Halide. It is imaging-focused,
not a general cone library, and Halide codegen is heavy to maintain.

**POGS** (Fougner & Boyd) is the model to follow for *organization*: every block
is stored as a transform of a small base prox, so a linear and quadratic term is
added to any prox automatically.

**The gap OCS fills:** a modern GPU-native, autodiff-friendly, *composable* prox
library that covers the LP/QP/SOCP cone set (plus PSD / exp / power), organized by
a calculus and a scaling contract, with type-grouped GPU dispatch.

## How OCS organizes its library

Not a list — a small set of base blocks, a **calculus** that augments any block,
a **scaling contract** that states exactly what metric each block admits, and a
**grouped GPU dispatch** so a heterogeneous list of blocks evaluates as one
batched dispatch per type.

| block | base prox | auto lin/quad aug. | scaling class | eval kind | grouped dispatch key |
|---|---|---|---|---|---|
| `SumSquares` ½‖·‖² | `v/(1+t)`, or matrix `(I+D)⁻¹Dv` | yes | matrix | closed_form | `SumSquares` |
| `Quadratic` ½·ᵀP·+qᵀ· | `(P+D)⁻¹(Dv−q)` | yes | matrix | linear_solve | `Quadratic` |
| `L1` λ‖·‖₁ | soft-threshold | yes | diagonal | closed_form | `(L1, λ)` |
| `L2` λ‖·‖₂ | block shrinkage | yes | scalar | closed_form | `(L2, λ)` |
| `Pinball` ρτ (quantile) | asymmetric soft-threshold | yes | diagonal | closed_form | `(Pinball, τ)` |
| `Logistic` Σlog(1+eᵘ) | safeguarded Newton | yes | diagonal | newton | `Logistic` |
| `PiecewiseLinearConvex` Σmaxⱼ(sⱼu+rⱼ) | monotone bisection | yes | diagonal | bisection | `(PWL, P, …)` |
| `NonNeg` | `max(v,0)` | yes | diagonal | closed_form | `NonNeg` |
| `Box` | `clip(v,lo,hi)` | yes | diagonal | closed_form | `(Box,lo,hi)` |
| `Equality` {·=0} | `0` | yes | diagonal | closed_form | `Equality` |
| `Zero` (free) | `v` | yes | matrix | closed_form | `Zero` |
| `SecondOrderCone` | cone projection | yes | scalar | closed_form | `SecondOrderCone` |
| `PSDCone` | eigenvalue clip | yes | scalar | spectral | `(PSDCone, k)` |
| `ExpCone` | Newton projection | yes | scalar | newton | `ExpCone` |
| `PowerCone` | Newton projection | yes | scalar | newton | `(PowerCone, α)` |

- **auto lin/quad augmentation** = the POGS-style calculus (`scaled`,
  `precomposed`, `with_linear_quadratic` in `prox/base.py`).
- **scaling class** = the richest metric `D` the block admits; see
  [`scaling_contract.md`](scaling_contract.md).
- **grouped dispatch key** = blocks with equal `(key, size)` run in one `vmap`
  dispatch (`prox/groups.py`); see `examples/bench_grouped_prox.py` for the
  measured payoff (flat in the number of blocks).

## Public vs experimental (v1)

- **Headline / validated:** affine, box, nonnegative, equality, `L1`, `L2`,
  least-squares, **SOCP** — plus the LP/QP/SOCP path validated against SCS.
- **Present, experimental:** `PSDCone`, `ExpCone`, `PowerCone`, and the full
  *matrix* scaling tier — implemented and unit-tested (feasible + idempotent),
  not yet benchmarked at scale or wired into the CVXPY translation.
