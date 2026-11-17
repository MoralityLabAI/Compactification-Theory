# General normalized multivariate jet matrix compatibility

34 authored roots pass frozen Lean compilation, native kernel replay and OxiLean.
The campaign now has 316 roots, including the earlier 12 attributed upstream
Taylor roots; Hodge remains separate and unchanged.

## Representation and compatibility

The new concrete realization stores ordered derivative words, with
`J=sum_(r=0)^k d^r` registers (`size_formula`). A word w of length r stores
`z_w(a)=mu^r/r! D^r f(a)[e_w]`. Directional order is retained in the registers.
This is a different finite layout from the earlier grouped multiindex layout;
its width depends only on fixed dimension and degree.

For an axis j and signed step q the row is explicitly

`(T_(j,q) z)_w = sum_(t=0)^(k-r) choose(t+r,r) q^t z_(j^t ++ w)`.

`matrix_application` identifies this formula with multiplication by the defined
J-by-J matrix; `matrix_integral` proves integer entries for integer q.
`directional_component` and `directional_ofFn` identify recursively defined
directional derivatives with genuine Frechet derivative tensor components.
`factorial_transport` supplies the normalization identity. Consequently
`matrix_compatibility` proves, for every dimension, order, coordinate, anchor,
mu and q, that the matrix predictor is the degree-(k-r) analytic Taylor
polynomial of the rank-r component at displacement `mu*q*e_j`, scaled by
`mu^r/r!`. There is no compatibility or mixed-partial symmetry premise.

The matrix has the existing common bound `A_D=2^(k+1) D^k` for `|q|<=D`,
D>=1. `matrix_residual` derives the common normalized residual
`|z_w(a+mu*q*e_j)-(T_(j,q)z(a))_w| <= 2 H D^(k+alpha) tau`
from C^k regularity, a Holder bound H on the original highest derivative,
mu>=0, alpha>=0, tau>0 and `mu^(k+alpha)<=2 tau`. All ranks, including the
degree-zero top component, and zero displacements are covered.

## Decoding and approximation

Motion tags encode axis, sign and bounded integer magnitude into a finite
alphabet. The actual Hamming selector reads these tags, and each packet also
contains the rounded signed corrections. All packets are concatenated into
the existing base-4 sentinel word. The decoded clipped recurrence and spatial
capture are proved to reproduce the synthesized reference trajectory.

`targets_residuals` discharges the true-jet residual contract for the recursively
constructed physical anchors. Corrections are bounded by

`N >= 2 H D^(k+alpha) + A_D/2 + 1/2`.

Thus N is independent of mu, tau and stream length. The motion and correction
bit widths obey `tagCount(d,D)<2^p` and `2N<2^a`; `finite_parameters` proves
finite choices exist. With mu<=1 and derivative tensors bounded by M at the
anchors, every true word coefficient is in [-M,M]. Rounding and clipping then
prove tau/2 accuracy for every reference state and for the selected decoded jet.

`evaluation_values` proves that evaluating the normalized true word jet at
`(x-a)/mu` gives the genuine multivariate Taylor polynomial. For k>=1, mu>0,
query distance at most rho<=mu, and the explicit regularity, boundedness,
safe-site and schedule-length conditions, `WordCapture.decoded_approximation`
proves

`|f(x)-evaluate(decodedJet,(x-a)/mu)| <= H rho^(k+alpha)/k! + J tau/2`.

The final theorem assumes neither matrix compatibility nor coefficient
accuracy nor a desired transport residual: these are derived. Coefficients,
packets and anchors are independent of the query. Regularity is global ContDiff
and the highest-derivative Holder bound is global; lower tensor bounds are only
required at the finite anchors. The new proof realizes the general decoder
through ordered words. It does not identify the earlier grouped-coordinate
matrix with the new matrix. Closed-cube extension and geometric specialization,
safe-phase aggregation, multiplication approximation, and dense ReLU/quantized
network resource realization remain open. Exact real polynomial multiplication
is used in this analytic decoded approximation theorem.

## Verification

* New roots: 34; frozen source compiled and native LeanChecker replay accepted.
* OxiLean: 29072 verified; 0 unsupported; 0 rejected.
* Native/export: 30560 constants matched, including types,
  values and universe parameters. Normalization only erases metadata and the
  let optimization flag; Lean alpha equivalence is used for expressions.
* Closed, unambiguous dependency graph with no unsafe declarations/admissions.
* Per-root native/export axiom sets agree and use only propext, Quot.sound and
  Classical.choice.
* 18 acceptance tamper controls and 7 structural auditor controls passed.
* All earlier accepted proof source hashes, including SharpTransport and Hodge,
  remain unchanged. Main Lean 4.35 work is preserved; new checks use 4.32.0-rc1.

Mathlib remains at 360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56, with the existing
cached artifacts validated before and after replay. No software or additional
mathlib modules were installed. Windows-mount startup stalls are excluded from
successful checks. Frozen proof input, runtime artifacts and existing pinned
executables use byte-identical Linux-local copies, checked against their source
hashes. The source mirror receipt ties the compiled input to the repository
proof bytes. Native checking uses the unmodified pinned
LeanChecker.replayFromImports API. A full source rebuild of mathlib is not claimed.

Export SHA-256: `8a219dee6229187be4f641651369ea94036dc9c9847679e17fc3e99f998eb5fb`.

## Reproduction in the preserved workspace

Recreate the earlier pinned runtime/mathlib mirrors if WSL has restarted. Run
`prepare_matrix_replay.py` on Windows; it freezes the proof, records the prior
accepted module and creates Linux-local runners and executable copies. Validate
the original mathlib tree into `replay432/Matrix/mathlib_before.json`. In WSL run
`bash /dev/shm/oxilean-sharp432-20261117/run_matrix432.sh`. Validate mathlib again
into `mathlib_after.json`, then run `promote_matrix_results.py`,
`accept_matrix432.py` and `test_matrix_acceptance.py` on Windows. Pinned workspace
dependencies and the preceding accepted SharpTransport olean are required;
binaries and oleans are not in the delivery bundle.
