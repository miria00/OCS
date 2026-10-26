# The reference projector used by the spatial-full SOC prox tests

`fair_projx.py` is a byte-identical copy of

    candidates/positive_cone_metric_20260922/fair/scripts/fair_projx.py
    sha256 a784549e21fde17d8df031c3ea221972526f841099d95174233d10e51a6ee89c

which is itself a counted copy of the E-reference vectorized weighted SOC
projection, `projx.proj_block_rows`, and whose own archived test
(`fair_test_projx.py`, results in `fair/results/test_projx.json`) checked it
bitwise against that E-reference and against the independent
`projectors.proj_block`.

It is copied here so the test travels with the tree. It is NOT edited: it is a
checked object, and a reference that gets adjusted to agree is not a reference.

## What is no longer on disk

The E-reference modules themselves, `projx.py` and `projectors.py`, lived under
`~/poscone_fair_20260922/scripts` and are gone. So the comparison available
today is against this counted copy plus the archived record of its bitwise
agreement, not against the E-reference directly. That is weaker evidence than
the original chain and it is stated rather than implied.

## The deliberate difference in the returned point

The reference returns the radius as `r(theta) = a t / (a - theta)`. This module
returns `||u(theta)||`. The two are the SAME number at the exact root, and they
differ by the root error off it. The difference is intentional:

* with `r(theta)` the returned point can sit outside the cone by the root error,
* with `||u(theta)||` it is on the cone boundary by construction, the residual
  `M (v - z)` is in `-Q` exactly whenever the kept bracket end has `G >= 0`, and
  the entire root error lands in `|z^T n|`, which bounds `||z - z*||_M` by its
  square root.

So agreement with the reference is expected only to root accuracy, never
bitwise, and the tests check it that way.

# The preserved 0.3.2 packet used by `test_spatial_full_construct_class.py`

`socp_ball_n24_s3_032_endpoint.npz` holds the pair `(x, y)` that the 0.3.2
release wheel (sha256 afbc0eb71650529cea179e5209a4cab908ca300f5c2fbaef9120024f4f627235)
returned as `solved` on the instance `ball(n=24, seed=3)` of that test file
(`equilibrate="construct"`, `metric_classes={0: "spatial_full"}`, `tol=1e-4`,
`max_blocks=2000`, `block=100`; 3100 iterations, termination `criterion`,
reported gap -5.823875e-05), together with the instance data `c`, `A`, `mu`
and `tol` so the test can assert it is scoring the same problem.  Captured
2026-10-22 on an RTX 5090 (pod f83f9df148d5) by
`candidates/release_034_repair_20261022/integrated/soc/soc_032_endpoints.py`,
record `endpoint_032_t1_construct_tol1e-4.json` beside it.  It is a checked
object and is not edited.  The test asserts that the CURRENT policy rejects
this packet (recovery off: the dual is refused by the exact conjugate-domain
test; recovery on: the recovered bound misses the tolerance) and records why.

# The PSD resolved-sign controls (`psd_signfix_controls.npz`)

Float32 arrays copied byte for byte from saved packets of the PSD projection
lane (rented A100, 2026-10-23; `candidates/psd_proj_20261023`), used by
`tests/test_psd_resolved_sign.py`. Exact rational arithmetic on these bytes
gives the following.

- `b300_*`: representation B `[psd(x, 3), sum_squares(x)]`, seed 300. Returned
  by OCS 0.3.4 on GPU as `solved` (source `residual_scaled`). The coupling
  `V - y` is outside the polar cone: lambda_max(sym) >= 3.6e-8 > 0, proved.
- `a301_check2_*`: representation A `[sum_squares(x), psd(x, 3)]`, seed 301.
  The supplied iterate at check 2, admitted by the float32 GPU gate, has
  lambda_max(sym y) = +2.77e-8 in float64. It is outside the polar cone.
- `a302_cpu_*`: representation A, seed 302. The retained witness returned on
  CPU has lambda_max >= 9.19e-9 > 0, proved. The reported objective_gap
  rested on it.
- `maxcut1002_*`: the v17 MaxCut packet `stage8_sdp_10_1002` (x, y_iterate,
  cost c). On JAX CPU, 0.3.4 certified a recovered dual whose slack
  C + diag(y) has minimum eigenvalue -8.75e-8, a production pass with a failed
  reference.
- `b800_*`: the representation-B boundary packet (k = 8, seed 800). Its coupling
  is negative definite, proved (lambda_max = -1.3058e-7), but that is 0.044 of
  the float32 eigenvalue resolution. The float32 gate refused it on one host
  and admitted it on another.

The source packets and their sha256 are recorded in
`candidates/psd_signfix_20261023/receipts/controls_provenance.txt`.
