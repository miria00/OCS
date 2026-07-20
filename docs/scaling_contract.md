# The scaling contract — what OCS can compute

Every block is accessed through its **scaled proximal operator**

```
prox_f^D(v) = argmin_u  f(u) + (1/2)(u - v)^T D (u - v),   D > 0.
```

`D` is the metric (the step-size / preconditioner). There are exactly three
tiers, in increasing power and *decreasing* availability:

| tier | `D` | cost | admissible when |
|---|---|---|---|
| **scalar** | `(1/t) I` | O(m) | **always** (whenever `f` has a prox) |
| **diagonal** | `diag(d), d>0` | O(m) | **iff `f` is coordinatewise separable**, `f(u)=Σ_j f_j(u_j)` |
| **matrix** | any `D ≻ 0` | a linear solve | **iff `f` is quadratic / affine**, `f(u)=½uᵀQu+qᵀu` |

**Why these are the exact boundaries.** The prox couples coordinates through
*both* `f` and the metric. A scalar `D` stays isotropic and never adds coupling.
A diagonal `D` keeps a closed form only if `f` doesn't couple coordinates — then
the problem splits into `m` independent 1-D proxes; a coupled `f` (a norm, a
cone) loses its closed form under an elliptical metric. A full `D` gives a closed
form only when the prox is itself a linear solve, i.e. `f` quadratic:
`prox_f^D(v) = (Q + D)^{-1}(Dv - q)`.

## Per block

| block | `f` | scaling tier | eval kind |
|---|---|---|---|
| `SumSquares` ½‖·‖² | quadratic (+ separable) | matrix | closed_form |
| `L1` λ‖·‖₁ | separable | diagonal | closed_form |
| `Box`, `NonNeg`, `Equality` | separable indicator | diagonal | closed_form |
| `L2` λ‖·‖₂ (not squared) | coupled norm | scalar | closed_form |
| `SecondOrderCone` | coupled cone | scalar | closed_form |
| `PSDCone` | coupled spectral | scalar | spectral |
| `ExpCone`, `PowerCone` | coupled cone | scalar | newton |
| `Zero` / affine | affine | matrix | closed_form |

**Rule of thumb:** *separable → diagonal · coupled cone/norm → scalar ·
quadratic/affine → full matrix.*

## The orthogonal axis: calculus

Independent of the metric tier, the calculus (`prox/base.py`) folds three things
into **any** block automatically, by closed-form parameter transforms:

- scaling `c·f`  (`.scaled(c)`),
- affine pre-composition `f(a·−b)`  (`.precomposed(a, b)`),
- a linear + separable-quadratic term `f(·)+dᵀ·+½·ᵀdiag(e)·`  (`.with_linear_quadratic(d, e)`).

These preserve the tier. So what OCS can compute is
`{scalar | diagonal | matrix}` metric × `{scale, affine, linear, quadratic}`
calculus.

## How the metric is chosen (you never set it)

The user picks `f` (the block) — that fixes the admissible tier. OCS chooses the
actual `D` from **matrix-free equilibration** (`solver/equilibrate.py`): Hutchinson
probes estimate the column energies `diag(KᵀK)` and row energies `diag(KKᵀ)` from
`matvec`/`rmatvec` only, giving per-coordinate steps `τ_j ~ 1/‖K_{:,j}‖`,
`σ_i ~ 1/‖K_{i,:}‖`. Rows belonging to a coupled (scalar-class) block are forced
to share one uniform `σ`, so the metric handed to each block stays inside its
admissible tier. A final power iteration on the scaled operator rescales both
steps to satisfy the PDHG stability bound.

**Status.** All three tiers are live. Every block *declares* its tier
(`scaling_class`) and *implements* a prox that accepts a per-coordinate step, and
the diagonal equilibrator produces one — so the scalar and diagonal tiers drive
the default solver. The **matrix tier** is implemented as `prox_matrix(v, D)` on
the quadratic blocks (`SumSquares`, `Quadratic`): `prox_f^D(v) = (Q+D)^{-1}(Dv-q)`,
a dense linear solve. It is *consumed* by a preconditioned Chambolle–Pock solve
that uses a matrix primal metric `M ≈ P` for a quadratic `g` — on an
ill-conditioned QP (cond `P` ≈ 1e3) this cuts the iteration count from ~590
(scalar metric) to ~40, insensitive to conditioning
(`examples/matrix_metric_demo.py`). The full matrix metric is for moderate /
structured blocks (the dense solve is `O(k³)`), by design; the operator itself is
still only touched through `matvec`/`rmatvec`.
