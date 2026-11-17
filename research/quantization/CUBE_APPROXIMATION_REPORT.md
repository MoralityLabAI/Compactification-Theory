# Intrinsic closed-cube regularity and spatial anchor coverage

58 authored roots pass frozen Lean compilation, native kernel replay and OxiLean.
The campaign now has 374 research roots, including the earlier 12 attributed
upstream Taylor roots. Hodge and all previously accepted proof sources are unchanged.

## Closed-cube regularity

`CubeTaylor.cube d` is the coordinate cube [0,1]^d in the supremum norm.
`cube_convex`, `cube_uniqueDiff`, and `cube_segment` prove convexity, unique
differentiability and containment of every interpolation segment, including faces
and corners. The new derivative tensors are `iteratedFDerivWithin` on this cube.
No values or regularity of f outside the cube are required.

`dir_smooth`, `dir_tensor`, `dir_ofFn` and `line_derivatives` identify recursively
defined within-directional derivatives with genuine within-Frechet tensors and
with scalar derivatives on the closed Taylor interval. `holder_remainder` derives
the multivariate Holder Taylor estimate from intrinsic C^k regularity and a Holder
bound on the original highest within-derivative tensor. Boundary points are
handled by within-interval derivatives; a global smooth extension is not constructed
or assumed.

`values` uses the preceding ordered-word register layout, J=sum_(r=0)^k d^r,
with z_w(a)=mu^r/r! D^r_cube f(a)[e_w]. `row_compatibility` proves that the same
integer motion matrix represents its genuine Taylor polynomial. `axes_difference`,
`axis_line_holder`, `axis_remainder`, and `matrix_residual` derive the common
normalized residual from the highest within-tensor Holder bound. The top rank,
degree-zero transport, zero motion and clipped repeated boundary anchors are covered.

## Concrete spatial anchors and motion stream

`CubeAnchors.unrank` is an explicit inverse of the existing scalar serpentine rank;
`unrank_rank` proves the inverse on every valid grid digit vector. `unrank_bound`
and `CubeWalk.unrank_step` prove bounded digits and unit-coordinate adjacency.

For integer cap, D>=1, integer shifts t_j, coarse block c, and mu*cap=1, the
physical grid anchors are

`a(g)_j = mu * clamp_[0,cap](D*(c_j*K+g_j)+t_j-D)`.

`point_in_cube` keeps every anchor in [0,1]^d. `point_anchor` identifies this
integer lattice formula with the clipped real anchor

`clip_[0,1](h*(c_j*K+g_j+theta_j-1)), h=mu*D, theta_j=t_j/D`.

`lattice_step` and `adjacent_motion` prove that adjacent anchors differ by
`mu*q*e_axis` for a signed integer q with |q|<=D. `delta_motion` represents this
step in the existing finite motion-tag alphabet. `schedule` constructs the tags
for all K^d anchors, and `schedule_anchors` proves that their actual recursive
physical orbit equals the serpentine grid enumeration. Boundary clipping can
shorten a step to zero; these steps remain valid tags. The tags, anchor orbit,
true jets and correction packets depend on the coarse block and fixed parameters,
and are independent of the query's fine address.

For a query x in the cube whose phase coordinates satisfy the explicit safe-cell
inequalities, `safe_coverage` proves ||x-a||<=h. `selected_coverage` and
`schedule_selected` identify that anchor with the one selected by the existing
spatial rank circuit. No assumed selected-anchor distance, schedule alignment,
unit-step oracle or cube-membership condition remains in `CubeSpatial.approximation`.

## Connection to decoded approximation

`CubeCapture.targets_residuals` discharges the actual within-jet transport residuals.
The preceding Hamming selector, signed correction quantizer, base-4 sentinel
decoder and clipped reference recurrence prove tau/2 coefficient accuracy. The
finite correction alphabet retains the common bound

`N >= 2 H D^(k+alpha) + A_D/2 + 1/2`, `A_D=2^(k+1) D^k`.

The tag and correction bit widths satisfy `tagCount(d,D)<2^p` and `2N<2^a`.
With k=n+1>=1, intrinsic C^k cube regularity, highest-tensor Holder constant H,
alpha>=0, derivative-tensor bound M on the cube, mu>0, tau>0,
`mu^(k+alpha)<=2 tau`, and the stated safe-cell and finite-parameter inequalities,
`CubeSpatial.approximation` proves for the actual decoded and spatially captured jet

`|f(x)-evaluate(decodedJet,(x-a)/mu)| <= H*(D*mu)^(k+alpha)/k! + J*D^k*tau/2`.

The factor D^k is necessary in this interface: geometric coverage is h=D*mu,
so normalized query coordinates have norm at most D. The older global-regularity
theorem assumed distance rho<=mu and obtained J*tau/2; its source is preserved.
The new `evaluation_error` supplies the correct coefficient sensitivity at radius D.

The closed-cube theorem assumes neither a global extension nor matrix compatibility,
transport residuals, coefficient accuracy, nor anchor coverage. It proves these
from regularity and the concrete finite construction. It remains a safe-phase
analytic decoded-polynomial theorem. Coverage by enough safe phases and median
aggregation, approximate multiplication, and dense quantized ReLU network resource
realization remain open. Exact real polynomial multiplication is used. The older
grouped-multiindex layout is separate from the ordered-word realization.

## Verification

* New roots: 58; frozen source compiled and native LeanChecker replay accepted.
* OxiLean: 28954 verified; 0 unsupported; 0 rejected.
* Native/export: 30436 constants matched in type, value and
  universe parameters. Comparison erases only metadata and the let optimization
  flag; expression alpha equivalence is retained.
* Closed, unambiguous dependency graph; no unsafe declarations or admissions.
* Per-root native/export axiom sets agree and use only propext, Quot.sound and
  Classical.choice.
* 19 acceptance tamper controls and 7 structural auditor controls passed.
* All prior accepted source hashes, including WordTransport, SharpTransport,
  the original spatial interface, and Hodge, remain unchanged.
* The newly imported original SpatialInterface artifact was also rebuilt from
  its unchanged source; the complete olean is byte-identical to the cached artifact.

Lean is pinned at 4.32.0-rc1 (b4812ae53eea93439ad5dce5a5c26591c31cb697).
Mathlib remains at 360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56 with all retained
artifacts validated before and after replay. No new software or mathlib modules
were installed. Native checking uses the unmodified pinned
LeanChecker.replayFromImports API. Runtime, exporter/parser, previous module and
mathlib artifacts are checked byte-for-byte in Linux-local mirrors; a source
mirror receipt binds frozen compiled input to the committed proof bytes. A full
source rebuild of mathlib is not claimed. Main Lean 4.35 work is preserved.

Proof SHA-256: `ac12e290700873a1cd1d62dbcaebc77b9ee6d3596aee0fac75cef727db4541f1`.

Export SHA-256: `285242bb16a5260d1b7bb0721530c1de0c6ff2db20e8e9b5f79673a889ee6fa9`.

## Reproduction in the preserved workspace

Recreate the earlier pinned runtime/mathlib mirrors if WSL has restarted. Run
`prepare_cube_replay.py` on Windows to freeze the source and mirror the earlier
accepted SharpTransport, WordTransport and SpatialInterface artifacts. Validate
the original mathlib tree into `replay432/Cube/mathlib_before.json`; in WSL run
`bash /dev/shm/oxilean-sharp432-20261117/run_cube432.sh`. Validate mathlib again
into `mathlib_after.json`, then run `promote_cube_results.py`, `accept_cube432.py`
and `test_cube_acceptance.py` on Windows. Native binaries and oleans are local
proof dependencies and are not included in the delivery bundle.
