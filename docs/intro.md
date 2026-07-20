# OCS in five minutes

OCS solves `minimize_x  g(x) + Σ_i f_i(K_i x + l_i)` on the GPU. You supply three
things and nothing else:

1. an **operator** `K` as two functions — `matvec` (apply `K`) and `rmatvec`
   (apply `Kᵀ`). The matrix is never formed.
2. **prox blocks** `f_i` from the library (`l1`, `least_squares`, `nonneg`,
   `box`, `SecondOrderCone`, …).
3. per-instance **data** as named `Parameter`s.

## Hello world

`minimize_x  λ‖x‖₁ + ½‖MFx − y‖²` — compressed sensing with a subsampled FFT:

```python
import numpy as np, jax.numpy as jnp, ocs

n = 4096
K = ocs.FFTMask.random(n, ratio=0.25)      # K = M F : R^n -> R^{2m}, matrix-free
x = ocs.Variable(n)
y = ocs.Parameter("y")

problem = ocs.minimize([
    ocs.l1(x, lam=1e-3),                    # g(x)  — a primal block
    ocs.least_squares(K @ x - y),           # f(Kx + l), with l = -y
])

compiled = problem.compile()               # equilibrate once, build the engine
sol = compiled.solve_instance({"y": y_meas})
print(sol.residual, sol.iters)
```

## Same K, many instances

The point of `compile()` is the parametric regime — one operator, many data
vectors, solved in a single batched GPU call:

```python
xs, ys, iters, res = compiled.solve_many({"y": y_batch})   # y_batch: (N, 2m)
```

## What you are actually giving OCS

| you write | OCS receives |
|---|---|
| `K @ x` | an operator `(matvec, rmatvec)` — never a matrix |
| `l1(x, λ)`, `least_squares(...)` | prox blocks `f_i`, each with a scaling class |
| `Parameter("y")` | the per-instance offset `l_i(data)` |

That is the whole interface. Everything downstream — matrix-free equilibration,
the restart-PDHG solve, matrix-free termination — is automatic.
