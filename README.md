# OCS: Operator-Native Convex Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

OCS is a numerical optimization package for solving large-scale convex
problems on GPUs, in which each linear operator is given only as a function.
The current version is `0.3.8`.

OCS solves problems of the form

```
minimize    f_1(K_1 x + l_1) + ... + f_B(K_B x + l_B)
```

where `x in R^n` is the variable. Each `K_i` is a linear operator, which OCS
accesses only through the products `K_i x` and `K_i^T y`, so its matrix is
never formed. Each `f_i` is a closed convex function from the library of
proximal blocks below, and each `l_i` is a data vector. A problem family,
with fixed operators and changing data, is compiled once and then solved for
many instances, one at a time or as a batch.

OCS uses the primal-dual hybrid gradient method, with a matrix-free
equilibration. It stops when the returned point, paired with a dual feasible
point built from operator products, meets a relative feasibility and objective
test (see [Accuracy](#accuracy)).

## Installation

OCS requires Python 3.12 and JAX 0.11.1. On an NVIDIA GPU, run

```
git clone https://github.com/miria00/OCS.git
pip install "jax[cuda12]==0.11.1"
pip install ./OCS
```

For small problems on a CPU, install `jax==0.11.1` in place of
`jax[cuda12]==0.11.1`. This release was tested on an NVIDIA RTX 5090 with

```
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=1"
```

## Quick start: Bounded least squares

The following code solves a least-squares problem in which the variable is
constrained by lower and upper bounds.

```python
import numpy as np
import jax.numpy as jnp
import ocs

# Problem data.
m, n = 30, 20
np.random.seed(1)
A = jnp.asarray(np.random.randn(m, n), dtype=jnp.float32)
b = jnp.asarray(np.random.randn(m), dtype=jnp.float32)

# The operator, given only as a function that computes A @ v.
K = ocs.LinearOperator.from_forward(lambda v: A @ v, n=n, m=m)

# Construct the problem family: minimize (1/2)||Kx - b||_2^2 subject to 0 <= x <= 1.
x = ocs.Variable(n)
family = ocs.minimize([ocs.box(x, 0.0, 1.0), ocs.least_squares(K @ x - ocs.Parameter("b"))])

# Compile once, then solve an instance.
compiled = family.compile(tol=1e-3)
sol = compiled.solve_instance({"b": b})
print(sol.status)
print(sol.x)
```

`from_forward` builds the adjoint of a JAX-linear function by automatic
differentiation. A subclass of `ocs.LinearOperator` can instead supply
`matvec` and `rmatvec` directly. Another small example, with checks, is in
[examples/hello_world.py](examples/hello_world.py).

## Batches

A compiled family also solves a batch of instances in one GPU call,
`compiled.solve_many_results(data)`, where each row of the data is one
instance. Each returned `Solution` is held to the same test as
`solve_instance`.

For repeated dense quadratic optimal transport, a specialized solver is
described in [docs/quadratic_transport.md](docs/quadratic_transport.md).

## Operators

Built-in operators include the subsampled Fourier transform (`FFTMask`), the
discrete cosine and Haar wavelet transforms (`DCTOperator`, `HaarWavelet`),
Gaussian blur (`GaussianBlur`), 2-D and 3-D discrete gradients (`Gradient2D`,
`Gradient3D`), the row and column sums of optimal transport
(`MarginalOperator`), Kronecker and Sylvester products (`Kronecker`,
`Sylvester`), and multicommodity flow incidence (`MultiCommodityFlow`).
Operators are composed with `@`, `+`, and scalar multiplication.

## Proximal blocks

Each term passed to `ocs.minimize` applies one block function to an affine
expression `z = K x + l`. The metric is the most general scaling the block's
proximal operator accepts.

| Term | Block function of `z` | Metric |
|---|---|---|
| `equality(e)` | indicator of `z = 0` | diagonal |
| `nonneg(e)` | indicator of `z >= 0` | diagonal |
| `box(e, lo, hi)` | indicator of `lo <= z <= hi` | diagonal |
| `l1(e, lam)` | `lam ||z||_1` | diagonal |
| `least_squares(e)` | `(1/2) ||z||_2^2` | full |
| `l2(e, lam)` | `lam ||z||_2` | full |
| `group_l2(e, lam, group_size)` | `lam sum_g ||z_g||_2` | full |
| `pinball(e, tau)` | `sum_i max(tau z_i, (tau - 1) z_i)` | diagonal |
| `logistic(e)` | `sum_i log(1 + exp(z_i))` | diagonal |
| `pwl(e, ...)` | separable convex piecewise-linear | diagonal |
| `soc(e)` | indicator of `||w||_2 <= s`, where `z = (s, w)` | full |
| `psd(e, k)` | indicator of `z`, as a `k x k` matrix, being PSD | full |
| `exp_cone(e)`, `pow_cone(e)` | exponential and power cones | scalar |

Each block is also a `ProxBlock` object, such as `ocs.NonNeg()`, whose method
`with_linear_quadratic(d, e)` returns the block
`f(z) + d^T z + (1/2) z^T diag(e) z`, with its proximal operator evaluated
from that of `f`. Exponential and power cone support is experimental.

## Accuracy

`sol.status` is one of `solved`, `inaccurate`, `unverified`,
`primal_infeasible`, `unbounded`, and `numerical_error`. The status `solved`
means that at the returned point `x`, with an exactly dual feasible point `y`,
the relative primal infeasibility is at most `tol`, and the objective value
`J(x)` and dual value `D(y)` satisfy

```
|J(x) - D(y)| <= tol * max(|J(x)|, |D(y)|).
```

The test does not depend on the algorithm, and it does not change when the
data are scaled. The status `inaccurate` means the test failed or the
iteration limit came first, and `unverified` means the test could not be
decided, for example because no dual feasible point was found. The fields
`sol.objective_gap`, `sol.objective_lower_bound`, `sol.objective_primal_value`,
and `sol.relative_primal_infeasibility` hold the quantities tested. The
default tolerance is `tol=1e-3`, and the iterates are in single precision.

## Scaling

By default, OCS equilibrates with the stochastic matrix-free method of Diamond
and Boyd, from operator products alone. It uses a diagonal metric where a
block supports one, and a scalar metric otherwise. A richer metric is chosen
per term,

```python
compiled = family.compile(equilibrate="construct", metric_classes={0: "full"})
```

with classes `scalar`, `diagonal`, `full`, `spatial_full` (one second-order
cone term), and `automorphism` (second-order and semidefinite cones). The key
is the index of a dual term, counting the terms in order and skipping the term
applied to `x` itself. A full
metric pays when the ill conditioning lies inside a block and the operator is
reused, or on second-order cones of dimension 128 and above. It can be slower
on a single solve.

## CVXPY

`ocs.cvxpy_mf` solves a CVXPY problem without forming its constraint matrix.
`ocs.cvxpy_harness` does form the matrix, and is used only to check small
problems against a reference solver. Both need the `validation` extra,
`pip install "./OCS[validation]"`.

## Tests and bug reports

Run the tests from the cloned repository with `python -m pytest tests`.
Please report issues via the GitHub issue tracker. Include the output of
`python -c "import ocs; print(ocs.__version__)"`, the fields `status`,
`accuracy_criterion`, `accuracy_detail`, and `gap_history` of the returned
`Solution`, and a small problem that shows the issue.

## Citing

A paper describing OCS, by Miria Feng and Stephen Boyd, is in preparation.

## License

Apache 2.0. See [LICENSE](LICENSE).
