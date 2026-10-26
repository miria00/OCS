# Changes

## 0.3.7 (2026-10-23; unpublished candidate)

- Prepared scalar and batched quadratic OT register the RTX 5090 metadata
  missing from the pinned JAX 0.11.1 Pallas backend during construction.
  The reported architecture is checked; existing registry entries are kept.
  Users no longer need the qualification-only device shim. Plain imports
  remain lazy; kernels, numerical policy and acceptance are unchanged.
- Every recovered ray is assessed after materializing its actual float32
  witness, using that witness's own adjoint. The marginal LP domain also uses
  a scoped float64 check of represented prices and costs with an outward
  arithmetic envelope. A cached-product success cannot accompany a different
  returned array. New and additional assessment actions are reported.
  Refinement reserves that full structural work bound before dispatching a
  proposal assessment, including the sensitive arithmetic checks.
- `CompiledFamily.solve_with_refinement(data)` explicitly requests one bounded
  correction after an eligible unsuccessful ordinary solve. Default refinement
  stays off. Failure retains the original assessed return; acceptance uses the
  unchanged original request. This incorporates the previously isolated legacy
  quadratic-OT refinement.
- Integrates the admitted large-array operand lowering, static slice windows,
  private-buffer donation and phase-separated no-DC imaging assessment. The
  equality-eliminated witness, global FFT and legacy preconditioner symbol are
  preserved. The dependency is now pinned to JAX 0.11.1 (Python >=3.12), whose
  compiler interfaces this lowering uses. Final combined qualification is
  recorded separately; the import alone is not billion-scale qualification.
- CG retains its explicit primal strong-convexity exception: `y=None`,
  `dual_source="none"`, and no new dual-witness requirement.

## 0.3.6 (2026-10-23)

The 0.3.5 release with three dual-bound routes for the objective-gap
controller, an exact domain test for `PiecewiseLinearConvex`, and stricter
acceptance at float32 sign and domain boundaries: an unresolved sign is no
longer accepted.

- Stricter acceptance at float32 sign and domain boundaries.  Two dual-domain
  decisions that float32 made at a boundary without an error bound are now
  decided in scoped float64 with a rounding envelope: the PSD cone's sign
  (both as a dual block and as the primal block's coupling) and the coupling
  of an l1 primal block over subsampled Fourier operators.  An unresolved
  sign is no longer accepted.  A witness whose sign or margin float32 cannot
  resolve is refused, never certified, so a result that relied on one is
  `inaccurate` or `unverified` rather than `solved`, unless another candidate
  certifies it.  Each closes float32 boundary admissions of witnesses that
  lie outside the domain in exact arithmetic.  Neither relaxes any test, and
  families that trace neither decision run the same computation as before.
  Details in the next two items.
- FFT/l1 coupling decided in float64.  For a primal block `L1` whose dual
  operators are all `FFTMask` with unique indices (decided on host at
  compile time), the coupling `||-K^T y - c||_inf <= lam` of every assessed
  witness is also decided on a scoped float64 adjoint of the same operator,
  with a componentwise float64 envelope:
  `(max|-K^T y - c| + env64)(1 + 4 eps64) <= lam`.  The float64 adjoint
  scatters the real and imaginary parts as separate float64 arrays, because
  XLA serializes complex128 scatters on the GPU.  The witness passes through
  `jax.lax.optimization_barrier`, and ratios are rounded up.  The float32
  test stays, and both must admit the witness.  The ray search's float32
  trials remain proposals, and its winner is re-decided in float64.  When
  float32 accepted the winner and float64 puts it outside by no more than the
  a priori float32 adjoint allowance, the winner is scaled into the domain,
  `a' = a (1 - 64 eps32) / ratio`.  It is then re-evaluated by the same gates,
  assessed in float64, and returned.  Otherwise it is refused.  The domain is
  never relaxed.  This closes a float32 admission, observed on a lasso with
  Gaussian data, of a witness with `max|F^T y| / lam = 1.0000044`.  Recorded in
  the bound packet (`coupling_arithmetic_guard =
  "fftmask_l1_float64_adjoint_envelope"`, `fft_l1_*` keys).  The dual
  routes pass the cost `c` to the strict-bound gates, and no route traces
  this guard (all three need a NonNeg primal).
  (`tests/test_fft_l1_coupling.py`)
- PSD witnesses need a resolved sign (stricter acceptance, never looser).
  On every PSD-dual block and on a PSD primal block's coupling, the
  objective-bound evaluator now decides membership with
  `PSDCone.conj_in_domain_resolved`.  It admits a dual only if the float32
  test `conj_in_domain` admits it AND, on the represented float32 values, the
  largest eigenvalue of the symmetric part, computed in scoped float64,
  satisfies `lambda_max <= -c k eps32 ||sym||_2` with
  `c = ocs.prox.cones.PSD_RESOLVED_SIGN_C = 2`.  The f32 -> f64 conversion
  passes through `jax.lax.optimization_barrier`.  A sign inside that envelope
  is unresolved at the scale on which the data were formed.  It is not
  admitted, so such a result stays unverified (`inaccurate`), never `solved`.
  The ray search's chosen point is re-validated as materialized: the float32
  witness `a * y`, its own adjoint (not the trials' cached `a K^T y`), and the
  final gates.  This is the computation a recovery-off recheck performs.
  Trials use the same envelope; only the final, materialized check decides.  The supplied witness was already
  validated through its own adjoint; the retained witness is the certificate
  so validated at its check.  This closes float32 boundary admissions of PSD
  witnesses whose coupling or block is outside the cone in exact arithmetic,
  and the host dependence of the float32 spectral decision at sub-resolution
  signs.  It costs one adjoint per ray search on PSD families
  (`operator_applications_breakdown["ray_search_materialized_revalidation"]`)
  and float64 eigenvalues of the PSD blocks per trial.  Recorded in
  `meta["psd_resolved_sign"]`.  Non-PSD families are unchanged: their
  traced computation is the same.  `PSDCone.conj_in_domain` itself (used
  outside the evaluator) is unchanged.

- Dual-bound routes, `compile(..., dual_routes=("shift", "packing",
  "ridge"))`, default all three on, `dual_routes=()` to turn them off
  (`ocs.model.compile.DUAL_ROUTES`; recorded in `meta["dual_routes"]` as
  requested, eligible and enabled).  Each route builds one more dual
  candidate from the current iterate for one structure, decided on exact
  host facts at compile time:
  - `"shift"`, the recession shift: the block-wise exact projection of the
    iterate moved along a recession direction of every block domain (+1 on
    unrestricted dual blocks, -1 on NonNeg dual blocks) by the smallest step
    that clears the coupling with a rounding allowance.  Eligible when the
    primal block is NonNeg and some dual block is NonNeg or unrestricted.
  - `"packing"`, the packing-LP assignment dual: the iterate's resource dual
    projected onto >= 0, with the smallest admissible assignment dual for it.
    Eligible when the primal block is NonNeg and the only dual block is NonNeg
    with operator `Weighted(BinPack, -1)`.
  - `"ridge"`, the compound ridge: the multiplier of the one SumSquares dual
    block with operator `Diagonal(d > 0)` on all columns, reconstructed from
    the other blocks projected into their own domains.  Eligible when the
    primal block is NonNeg and that block is the only such one among at least
    two dual blocks.

  The packing and ridge witnesses are formed in scoped float64 with explicit
  rounding allowances and rounded up to float32 through
  `jax.lax.optimization_barrier`, so the upward step survives jit and
  jit+vmap.  A route is consulted only at a check where the evaluator's own
  packet does not meet the criterion; its candidate counts only when the
  evaluator's strict-bound gates validate it and it is strictly better than
  the bound held, and it is kept on its own retention chain, which never feeds
  the evaluator's chain or the stall rule.  A run therefore differs from
  `dual_routes=()` only by stopping at an earlier check on a route's
  candidate, or by reporting a stronger valid bound at a budget or stall exit
  with the request unmet.  The routes change neither criterion (7), the
  acceptance predicate nor any validity gate.  With `dual_routes=()` the
  controller is the 0.3.5 controller with the two boundary fixes above.
  A certifying route is named in `Solution.objective_bound_source`
  (`recession_shift`, `packing_assignment_dual`, `ridge_nonneg_compound`).
  The routes run in the single-instance controller (`solve_instance`); the
  batch interfaces keep the 0.3.5 candidates.
- Cost: each consultation is charged in `Solution.gap_history`, in wall
  seconds and operator applications per route
  (`DUAL_ROUTE_OPERATOR_APPLICATIONS`, counted at the call sites as the
  evaluator counts its own): 4 for the shift, 3 for packing and 3 for the
  ridge.  That is 4 or 7 operator applications per consulted check: 4 where
  only the shift is eligible, and 7 on the packing and ridge families, which
  also have the shift eligible.  Nothing is charged at a check the
  evaluator's packet accepts.
- PiecewiseLinearConvex domain test: `conj_in_domain` decided `y >= s_min`
  with a float comparison, and XLA:CPU flushes subnormal operands, so with
  s_min = 0 it admitted y = -1.4e-45 (one subnormal below the domain) eagerly
  and returned the conjugate value -0.  The order is now decided on the bits
  (integer keys of the float32 values) in the domain test, in the choice of
  candidate inside `conj_value`, and in a correction after the clip in
  `conj_domain_project`; a wider dtype refuses a y below float32's smallest
  normal when a slope is subnormal.  When an input holds a subnormal, the
  point candidate (y equal to a slope) now carries the absolute flush
  allowance the interpolation candidates already had.  A backend that
  compares exactly gives the same decisions as before, and on data without
  subnormals the same values (`tests/test_pwl_conjugate.py`).

## 0.3.5 (2026-10-23)

The frozen 0.3.4 release with two changes.

- patch B: integer overflow in an equilibration diagnostic at m > 2^31.
  `smfe_equilibration`'s normalized-variance diagnostic `_nvar` multiplied the
  Python int `m` into a traced float32 expression; with x64 off JAX parses it
  as int32 and raised `OverflowError` at m = 3.8e9.  It now multiplies
  `float(m)`, which gives the same float32 value for every m below 2^31.  The
  diagnostic is reported only and is not part of the scaling.
- PiecewiseLinearConvex supplies the conjugate and domain test, so
  piecewise-linear families become eligible for criterion (7) certification.
  `conj_value` is the lower convex
  envelope of the points (s_j, -r_j) on dom g^* = [s_min, s_max] (+inf off it),
  evaluated as the smallest two-piece convex combination in the scoped float64
  assessment with a rounding envelope and rounded up, so it never understates
  the conjugate of the represented data; `conj_in_domain` is the exact
  closed-interval test, `conj_domain_radius` the largest admissible scaling
  (0 when none exists; the interval need not contain 0), `conj_domain_project`
  the clip to the interval, and `bound_capable = True`
  (`tests/test_pwl_conjugate.py`).  Before, a family with this block was
  refused under the default accuracy (`unverified`, termination
  `unsupported`).

## 0.3.4 (2026-10-23)

The 0.3.3 successor candidate (below) with the qualified repair streams of
2026-10-22/23 merged, in this order: no-DC route with patch A, SOC witness
policy, bounded refinement.  Where an item has a detailed entry further down
in this file, the bullet says so.

- Routing contract: the fold rule is `ocs.model.compile.FOLD_POLICY`, every
  compiled family records `meta["routing"]` (one entry per declared term:
  role, dual term index, dual rows, step and metric source, witnessing rows
  of `sol.y`), `equilibrate=("given", tau, sigma)` also accepts `sigma` as a
  dict bound to the dual terms by position, and a layout mismatch is refused
  by name with the compiled layout and the fold spelled out (entry "The
  routing is explicit" under 0.3.3; `tests/test_given_routing_contract.py`).
- Compact-batch dtype boundary: the batched certificate route
  (`certificate_many`, `jax.jit(jax.vmap(report_device))`) now batches the
  un-jitted report body on both data layouts, so the scoped
  `jax.enable_x64()` assessment is entered during the batched trace exactly
  as in the scalar route; before, the already-jitted report was re-traced
  under `vmap` outside the scope and the guarded float64 adjoint product was
  truncated to float32 (`TypeError` in `lax.add`).  Iterates stay float32,
  the assessment stays the scoped float64 route, no acceptance rule changes
  (`tests/test_certificate_many_scoped_assessment.py`).
- Source distribution manifest: `MANIFEST.in` grafts `tests/` (so
  `tests/conftest.py` and `tests/reference/` ship) and excludes caches and
  bytecode; the wheel is unchanged by it.
- SOC dual-witness contraction policy `compile(..., witness_contraction=
  "once" | "compound")`, default `"once"`: the objective-bound evaluator's
  dual-correction rounds build the `violation_corrected` candidate with the
  exact block projections and apply a second-order cone block's interior
  adjustment `(4 k + 16) eps32` one time, on the final candidate, instead of
  once per round (32 rounds compounded the pull to about `32 (4 k + 16)
  eps32 lambda / |p*|` below the optimum).  The final conservative
  membership test (`_conservative_norm_leq`, envelope `(2 k + 8) eps32`), the
  coupling gate, the acceptance decision and criterion (7) are unchanged;
  recovery-off evaluations are identical under both policies.  `"compound"`
  restores the 0.3.3 route per compile.  The regression test on the preserved
  0.3.2 packet is restated for this default (detailed entry below).
- No-DC imaging witness route `equality_projected` (`ocs.solver.nodc_witness`):
  for `nonneg(x) + least_squares(FFTMask x - b) + group_l2(Gradient3D x, 3)`
  with a sampling mask that excludes the DC frequency, the evaluator certifies
  an equality-eliminated float64 TV multiplier (`bound_kind="strict_dual"`,
  `witness_representation="equality_eliminated_tv_multiplier"`, arithmetic
  "mixed", certifying dual in float64) with an explicit correction bound
  through `s_D = 2 sin(pi / (2 max(D, H, W)))`; an unresolved witness is
  refused by name, nothing is clipped, and the localized Route B number is
  reported beside it as its own guarantee, never substituted.  Eligibility is
  an exact host fact (operator types, index set, grid); a DC mask keeps the
  free-row anchor route.  Settings in `meta["equality_projected_route"]`.
- Patch A: in that route's `F^T` action the complex128 spectrum is assembled
  with `.at[idx].add` instead of `.at[idx].set` (bitwise equal on the unique
  mask indices; the scatter-set path was serialized by XLA GPU in this JAX,
  about 10^4 times slower).
- Bounded refinement policy `compile(..., refine=False | "auto")`, default
  OFF; `refine="auto"` is the documented opt-in (at most one bounded stage
  on the stall guard, two supported structures, assessed by the unchanged
  criterion, recorded in `Solution.refinement`; detailed entry below).

## 0.3.3 (2026-10-22, successor candidate)

Bounded refinement policy (refine_auto candidate, 2026-10-22):

- `compile(refine=...)`, default `False`, with `"auto"` the one other value
  (also per call: `solve_instance(..., refine=)`,
  `solve_many_results(..., refine=)`). Under `"auto"` a run of the default
  objective-gap path that ends on the existing stall guard with the request
  unmet gets AT MOST ONE bounded refinement stage, on two structures only:
  a separable-penalty-plus-least-squares family with a fixed support (the
  FFT lasso shape; route `primal_dual_refinement`, fixed-support
  Jacobi-PCG on `K_S^T K_S` through operator products) and a
  nonnegative-quadratic-plus-equality family (the quadratic OT shape; route
  `active_support_solver`, Newton correction of the multipliers on the active
  support through `K M_S K^T` products, with the `MarginalOperator` gauge
  projected out). The refined pair is assessed by the UNCHANGED criterion (7)
  exactly as a check point is and is returned only when it passes; otherwise
  the ordinary packet is returned unchanged. A pair that already passes gets
  no refinement work; a family without an exact conjugate-domain contract
  gets the missing capability reported (`refinement["capability_missing"]`);
  a run that ended by the iteration budget is not refined (the trigger is
  the stall event, fixed, not per benchmark); a workspace that does not fit
  the admitted memory budget (`refine_memory_bytes`, else derived from the
  device or host) is skipped with the reason. `Solution.refinement` carries
  the record: outcome, route, the assessed state the stage started from, and
  the work (operator applications with correction and assessment parts, CG
  iterations, assessments, trials, wall seconds, arithmetic), so a row can be
  charged end to end; `iters` counts PDHG iterations only. The record keeps
  witness recovery (the assessment's SOC / no-DC routes, unchanged, never
  performed by the policy), primal-dual refinement and the active-support
  solver route distinct by name. Batched lanes refine one at a time after
  the batched iteration, each reading and replacing only its own row.
  `refine_assessed(x, y, data)` runs the stage on a supplied pair on request.
  Nothing in criterion (7), its normalizations, the conjugate-domain checks or
  the acceptance decision changed (`ocs/solver/refine_auto.py`,
  `tests/test_refine_auto.py`).
- `CompiledFamily.fixed_point_residual` exposes the fixed-point residual map
  of `refine` read-only (two operator applications); `refine` is unchanged.
  `ocs.solver.refine_safeguarded` (not imported by default) wraps proposals
  in a criterion-(7) acceptance transaction for explicit use
  (`tests/test_refine_safeguarded.py`).
- `FFTMask.rmatvec` allocates its complex buffer in the input's dtype:
  complex64 for the float32 iterates (unchanged bytes), complex128 for a
  float64 product under a scoped `jax.enable_x64`, which the refinement route
  uses.


Two pending change sets, reconciled from separate 0.3.2 development trees
(see the candidate's RECONCILIATION.md). Neither adds a feature.

Single-evaluation residual path (from the 2026-10-19 candidate):

- The opt-in residual path evaluates its stopping residual once per check and
  reports that evaluation. When the iterate the loop holds passes, the loop
  returns it with the repaired dual the residual was measured at and the
  residual that was tested, so a run that stops on the criterion is `solved`.
  Before, the loop stopped on one float32 evaluation and the residual was
  recomputed after the loop in a separately compiled context; at the threshold
  the two could differ (measured: an SDP stopped at 9.99e-5 and reported
  1.000084e-4, status `inaccurate` with `termination="criterion"`). The
  acceptance test is unchanged: `solved` still means the reported residual is
  at most `tol`. A budget exit is checked once at the returned point, as
  before. The objective-gap driving loops, the experimental online engines and
  a contract that stops the loop keep their former wrappers.

Criterion-(7) repair (from the 2026-10-21 candidate, v23):

- The guarded orthant assessment also handles declared weighted BinPack and
  diagonal operators with scoped float64 structured adjoints. It promotes
  stored coefficients and assessment buffers, never forms the full allocation
  operator, and leaves float32 iterations unchanged. An independently observed
  batch coupling violation cannot be accepted as a rounded zero. Structured
  reference/absolute/recheck actions have their own counters.
- Fresh-witness rechecks charge the measured adjoint and cost-addition error;
  the additional materialization envelope is charged only for ray proposals.
  This avoids rejecting an independently valid returned witness merely by
  charging for a second scaling that never happened. Interior proposals clear
  the guarded coupling envelope and recheck their actual multiplier bytes.

- Complete strict conjugate-domain contracts for pinball, logistic (including
  continuous entropy endpoints), PSD polar membership and supported affine /
  linear-quadratic wrappers. A positive-curvature smooth compound includes its
  inner prox error bound. Partial-zero unsupported curvature stays refused.
- SOC membership assessment now scales before squaring and withholds membership
  inside a conservative reduction uncertainty region. Exactly decidable axis
  and small-integer boundary cases remain accepted; subnormal ambiguity is
  refused. Interior recovery proposals clear that margin. Iteration vectors
  and operator/prox steps remain float32; no global X64 switch is introduced.
- Route an explicit nonsingular diagonal least-squares term into the primal
  block when no primal or explicit metric mapping is supplied. This preserves
  its represented objective while removing an unnecessary auxiliary equality;
  it changes that splitting and requires affected application qualification.
- The routing is explicit (repair candidate, 2026-10-22). The fold rule above
  is `ocs.model.compile.FOLD_POLICY`; every compiled family records
  `meta["routing"]`, one entry per declared term (role `primal`,
  `primal_folded` or `dual`, dual term index, dual rows, step source, metric
  source, and which rows of `sol.y` witness it). Because a plain compile may
  fold while a compile with supplied steps or metrics never does, the same
  family can have different dual row counts on the two paths;
  `equilibrate=("given", tau, sigma)` therefore also accepts `sigma` as a
  dict `{dual_term_index: scalar or (rows,) array}` bound to the dual terms
  by position (the `block_metrics` indices), and a wrong-length vector, a
  wrong key, a missing term or a wrong-length entry is refused by name with
  the compiled layout and the fold spelled out. The one-step block-metric
  test requests `accuracy="residual"` explicitly: its family has no dual-
  witness contract, so the default objective-gap route refuses it before
  optimizer work (below), which the earlier fall-through to the residual
  loop had hidden. Numerics unchanged: no step, prox, normalization,
  stopping test or conjugate-domain check was touched
  (tests/test_given_routing_contract.py).
- Feed the zero dual into pure-feasibility bound checking. Add a free-data-row
  interior witness proposal for nonnegative least-squares plus grouped TV.
  Its adjoint must be strictly positive and the actual full pair must pass
  ordinary domain checks. Missing DC does not gain a feasibility claim.
- Expose `objective_bound(..., recover=False)`, report its actual route counts,
  and align comparator acceptance with the controller's unresolved float32
  boundary. Normalize the returned-point discrepancy consistently and record
  bound provenance for scalar and batched results. Unsupported objective-gap
  requests return `unverified` / `unsupported` with zero iterations and name
  missing block contracts. The opt-in residual route remains available.
- `solve_many` now honors the compiled accuracy mode, like
  `solve_many_results`. Its former unconditional device-only residual loop is
  available with `accuracy="residual"`. Objective-gap batching includes its
  real checking costs and returns the per-lane certifying duals; a tuple alone
  does not establish that all lanes passed. Call this mode outside enclosing
  `jax.jit`; its kernels compile internally. Explicit residual batching retains
  its device-only/JIT contract and cannot stand in for objective-gap evidence.
- These candidate changes do not promote localized imaging bounds into strict
  dual witnesses, change criterion (7)'s tolerance, or replace the separate
  CG strong-convexity exception described below.
- Keep the rejected/non-improving doubled point as the right bracket of a
  dual-ray search. The previous bracket ended at the last feasible point and
  could lose nearly half the LP lower bound after a single halving. Every
  selected candidate still passes the same domain/value checks; no optimizer
  iteration, tolerance, recovery-round or operator-action budget increases.
- A constrained family's valid lower is no longer rejected merely for lying
  above the raw cost at an approximately feasible point. Toleranced indicator
  values do not certify a primal upper bound; the controller still checks
  feasibility and the absolute discrepancy. Feasible brackets remain separate.
- Dense LP adjoint coupling is checked with a scoped float64 product and a
  conservative error envelope, including the represented scaled witness. An unresolved
  slack cannot become feasible merely because float32 rounds it to zero.
  Additional absolute-adjoint assessment actions are counted separately;
  solver actions and storage remain float32; this assessment temporarily promotes
  the explicit matrix and reports mixed arithmetic. This guard applies to explicit
  Dense operators with a NonNeg primal block, not arbitrary opaque callbacks.
- Scalar and batched objective-gap controllers retain the strongest previously
  checked lower and its matching witness within each fixed-data solve. New
  costs and feasibility always belong to the current primal point. Reports name
  retained witnesses; no bound is cached across solves or changed parameters.
- For a PSD primal block and unrestricted dual blocks, a positive-definite
  symmetrized adjoint of the all-ones multiplier supplies an interior witness
  direction. A spectral shift proposes a feasible dual; the ordinary complete
  gate must still accept it. No operator matrix is extracted, and a missing
  positive-definite direction supplies no claimed anchor.

- Objective-gap PDHG results return the dual selected by the final bound
  check, including available witnesses on unsuccessful accuracy outcomes.
  `Solution.y_iterate` preserves the repaired iteration dual and `dual_source`
  names which dual is returned. Residual and contract diagnostics are evaluated
  at the returned pair. This adds a terminal reporting evaluation; it never
  restarts a segment from the recovered witness or changes acceptance,
  tolerance, check scheduling, stall decisions or recovery rounds.
- Retain supplied-witness and feasible-anchor candidates correctly in
  `objective_bound(..., return_dual=True)` and expose the same selected witness
  in batched checking. Freeze it at each lane's own terminal check.
- Preserve certificate directions, the opt-in residual path and the batch
  fallback that performs no bound check. CGLS retains its documented
  strong-convexity exception: `y`/`y_iterate` are `None`, with
  `dual_source="none"`. For `rho > 0`, objective-gap results identify their
  certified primal suboptimality bound as
  `objective_bound_source="strong_convexity"`; this is separate from the
  primal-dual paths' dual-witness criterion. With `rho = 0`, objective-gap
  accuracy remains unverified. CG iteration counts and outcomes are unchanged.

Dual-witness contraction policy (2026-10-23, named option; default `"once"` since 0.3.4):

- `compile(..., witness_contraction=None | "compound" | "once")` selects how
  the objective-bound evaluator's dual-correction rounds apply a block's
  interior safety adjustment while they build the `violation_corrected`
  candidate. `"compound"` (the behaviour through 0.3.3; selectable per compile)
  applies each block's `conj_domain_project` in
  every round, so a second-order cone block is pulled `rounds` times
  `(4 k + 16) eps32` inside its polar cone whether or not the round corrected a
  coupling violation; with the default 32 rounds the certified bound then sits
  about `32 (4 k + 16) eps32 lambda / |p*|` below the optimum on SOC families
  (4.0e-4 relative on the n = 24 ball fixture, 1e-3 to 2e-3 on the showcase
  family), independent of the iteration count. `"once"` runs the rounds with the
  new `ProxBlock.conj_domain_project_exact` (the projection with no
  adjustment; `SecondOrderCone` and `SOCBatch` override it, every other block
  inherits `conj_domain_project`) and applies the adjustment one time, by the
  final round, on the materialized candidate. The candidate is then validated
  as a whole by the unchanged exact conjugate-domain test
  (`_conservative_norm_leq`, envelope `(2 k + 8) eps32`) and the coupling gate;
  recovery-off evaluations, the acceptance decision and criterion (7) are
  identical under both policies. Recorded in `meta["witness_contraction"]` and
  in every `objective_bound` report.

## 0.3.2 (2026-10-18)

- Remove the optional CVXPY bridge's dependency on JAX's experimental sparse
  extension. Sparse matrix constants use ordinary JAX gather/scatter-add
  operations for forward products and automatic adjoints, without expanding
  the matrix. The prepared OT and batching implementation is unchanged.

- Add `ocs.transport.PreparedQuadraticOT` with optional CUDA dependencies,
  strongly convex Chambolle–Pock, fused Pallas updates, fresh input-derived
  initialization and the qualified numerical 1e-3 quadratic-OT accuracy gate.
  Its finite prepared output slots are independent and survive session close.
  Input validation, immutable reports and explicit cap failures are public.
  This addition does not change the generic solver or full-cone metric policy.

- Enable opt-in automatic SOC and PSD automorphism construction. SOC steps use
  scaled exponentials and log-determinant reconstruction, clipping eigenvalue
  logs before exponentiation. PSD steps bound each carrier's exponent norm and
  report the effective step length; this explicitly changes the former unstable
  unrestricted PSD step. Invalid construction steps retain the previous state
  with failure status. Scalar and diagonal defaults remain unchanged.
- Extend existing strict-interior zero-normal recovery to SOC and SOCBatch.
  Each cone uses its own data/iterate scale; the same coupled KKT evaluator
  selects and checks the actual candidate. Original tolerances remain unchanged.
- Supply the analytic conjugate contract for the exact positive scalar
  quadratic-plus-nonnegative wrapper, allowing its existing fixed-step proximal
  specialization to be recognized without dropping validation.
- Retain the 0.3.1 stall correction. No stall awards success. Omit the unverified
  repository URL from new package metadata and document the tested compiler
  settings explicitly.

## 0.3.1 (2026-10-18)

- Objective-gap solves can stop on insufficient relative progress even when
  the residual plateaus above 1e-7. Gap, residual and feasibility progress are
  considered; missing/nonfinite observations do not establish a plateau.
- Scalar and batch results report `termination="stalled"` at the preserved
  stopping point. A stall never awards success or relaxes the requested accuracy,
  and its explanation makes no claim that future progress is impossible.
  Batch lanes retain their own terminal pair, report and check history while
  unfinished lanes continue. The existing vectorized driver can still evaluate
  inactive lanes; their later values do not replace a terminal result.
- The acceptance predicate, solver recurrences, precision and metric-constructor
  refusal are unchanged. Full cone-metric construction remains separate work.

## 0.3.0rc3 (2026-10-16)

Release candidate; not a stable release.

- The live controller now stops successfully exactly when the acceptance
  award would mark the same assessed pair `solved`.  One predicate,
  `ocs.model.compile._gap_accepts`, serves the award, the scalar drive loop
  and the batched drive loop.  Before, the loops stopped on `gap <= tol`
  while the award refused a gap within the float32 resolution of `tol` as
  `unverified`; such a lane now continues to its next check.  The criterion,
  its normalization and its resolution guard are unchanged.
- The award tests primal feasibility before the sign of the objective
  discrepancy, as `CompiledFamily.acceptance` always did.  A negative
  discrepancy at a point infeasible beyond `tol` is `inaccurate`; before it
  could be awarded `solved`.  This only ever refuses more.
- One source of truth for the version: `ocs.__version__` and the
  distribution metadata agree, and `tests/test_release_identity.py` checks
  it on the installed package.
- README rewritten for the supported scope of this release.
- Stopping points, run times and accepted-lane counts of measurements made
  on 0.3.0rc2 may differ on this release; those results stay labelled rc2.

## 0.3.0rc2 (2026-10-16)

- One `equilibrate="construct"` call delivers the declared classes
  `scalar`, `diagonal`, `full` and `spatial_full`; unsupported declarations
  are refused by name.  SMFE remains the default.
- `_Scaled` blocks forward the exact conjugate-domain contract; the
  second-order cone blocks carry the exact polar contract.
- The objective-gap check schedule is configurable (`gap_checks`,
  `gap_check_growth`, `gap_check_first`).

## 0.3.0rc1 (2026-10-16)

- Acceptance policy `returned-point-three-condition-scaleinv-v2`: `solved`
  is a property of the returned point, invariant to rescaling the data.
# Prepared quadratic OT batching in 0.3.2

- `PreparedQuadraticOT(..., batch_size=B).solve_many_results(data)` runs a fixed
  batch through the installed vectorized OT update and complete assessment.
  It returns individually assessed, independently owned host results, including
  per-lane outcomes at the iteration limit. Preparation reserves whole-batch
  output slots; calls always initialize fresh numerical states.
- The measured lockstep policy amortizes GPU launches and uses more storage.
  Existing generic batching, scalar OT, stall detection and repaired full-cone
  metrics are preserved. See the transport guide for stopping/timing scope.

