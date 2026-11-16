# Common gate bound and genuine scalar Taylor remainders

2026-11-16, Asia/Taipei. This continuation adds **40 authored theorem roots**
and replays **12 attributed Mathlib foundation roots** under new names. Together
with the preceding 158 roots, there are **210 research roots**; the unchanged
Hodge root remains separate. The full multivariate network approximation theorem
remains open.

## Mathematical results

The 23 core roots in CommonGateBound/CommonGateCapture prove
A_D=2^(k+1)*D^k for D>=1 and an injective layout of coordinate degrees at most k.
Each compatible fibre has at most one entry per axis degree. Filtering the row,
then applying the hockey-stick identity, proves the bound without a matrix-norm
assumption. All candidates with input bound C use the same gate constant C*A_D;
C=2 gives 2*A_D, independently of the number of streamed spatial sites.
The actual Hamming-selected update, packed scheduled recurrence, and exact/robust
spatial capture are connected using that common constant.

The isolated Mathlib project proves 17 authored analytic roots:

* C^(n+1) regularity and |D^(n+1)f|<=M on the closed interval give
  |f(x)-T_n(a,x)| <= M*|x-a|^(n+1)/(n+1)!.
* For k=n+1>=1, C^k regularity and a highest-derivative Hoelder bound with
  constant H and exponent alpha>=0 give
  |f(x)-T_k(a,x)| <= H*|x-a|^(k+alpha)/k!.
  Degree-zero Hoelder functions and coincident endpoints are also covered.
* With s=k+alpha, mu>=0, tau>0, |q|<=D and mu^s<=2*tau, the normalized
  remainder at a+mu*q is bounded by 2*H*D^s/k!.
* The core Horner coefficient list is proved equal to the analytic Taylor
  polynomial. Coordinate errors eta_i contribute at most sum eta_i*|h|^i;
  uniform errors epsilon cost at most J*epsilon when |h|<=1.
* decoded_holder_uniform_approximation substitutes the actual common-bound
  Hamming/sentinel/spatial capture vector into the approximation estimate,
  giving H*rho^s/k! + J*epsilon for rho<=1.

The analytic premises are actual ContDiffOn, within-interval derivatives and
Hoelder inequalities, rather than an assumed Taylor-remainder conclusion.
The final capture theorem retains an explicit coefficient-accuracy contract for
the concrete reference recurrence. Derivatives are taken within the pointwise
anchor/query interval. Identifying these coefficients with fixed-domain and
multivariate jets, constructing valid correction frames with the required accuracy,
deriving the geometric schedule/canonical layout, and proving the dense ReLU
arithmetic/resource compiler remain required for the full approximation theorem.
The scalar evaluator still uses exact real multiplication; its realization by an
approximate ReLU arithmetic circuit is not proved in this stage.

## Independent verification

| Corpus | Roots | Verified declarations | Unsupported | Rejected | Native constants matched |
|---|---:|---:|---:|---:|---:|
| Common bound/capture | 23 | 3537 | 0 | 0 | 3858 |
| Analytic Taylor/capture, including 12 upstream roots | 29 | 22410 | 0 | 0 | 23783 |

Every declaration verdict is checked, not just the process exit code. Every
exported type, proof/definition value and universe-parameter list matches the
native imported environment; only metadata and the nondependent-let flag are
normalized. Dependency graphs are closed and name-unambiguous. Native and exported
axiom inventories agree root by root and contain only propext, Quot.sound and
Classical.choice. There are no admitted proofs or added mathematical axioms.
Eight common-bound and ten analytic acceptance tamper controls pass. The concrete
real-number diagnostic separately verifies 4,524 declarations with no unsupported
or rejected declarations.

Lean 4.32.0-rc1 compilation and native kernel replay pass for all five new modules.
For the three analytic modules, the driver calls the unmodified pinned
replayFromImports API from the pinned LeanChecker module directly, omitting only the stock CLI search-path
scan. The stock analytic CLI run was interrupted by a WSL restart and is not
counted as successful. DirectKernel.lean and direct_kernel.log record the resumed
checks. The two core modules use the stock leanchecker CLI.
The two core modules also pass preserved Lean 4.35 and leanchecker. The analytic
modules use the isolated 4.32 Mathlib project; no 4.35 analytic replay is claimed.
All preceding proof sources retain their accepted hashes.

Common export SHA-256: `afd5dee654f3a6dbd99050f6ce3efbd2f247a2f9189872834c4dc0c19257365b`.
Analytic export SHA-256: `e6023490bcaffb9112c4ebb6d258c678107eb6c8b1d1686f933103787850041e`.

Mathlib is pinned to 360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56
(v4.32.0-rc1). The selective installation has 1,906 Git-blob-verified Mathlib/helper
sources, eight manifest-pinned packages, and 2,145 cached build modules with 19,305
retained artifact hashes checked before and after replay. Official cache artifacts
are used; the full Mathlib library is not rebuilt from source. The dependency-closed
exported proof terms are independently checked. Regenerable C/editor-index files
are omitted locally, and retained files are compressed without changing content.
The earlier full-disk extraction failed and was discarded as installation evidence;
every required archive was subsequently freshly extracted and hash-recorded.

TaylorFoundation is a namespaced, attributed subset of the pinned upstream Taylor
module, with unchanged proof bodies, direct tactic imports and no interval-integral
imports. It is covered by the included Apache-2.0 license and source provenance.

## Reproduction and remaining program

Restore the included helper scripts into CAMPAIGN_WORK. Use the pinned local
Lean/exporter/OxiLean builds, restore the recorded Mathlib dependency, then run
run_common_replay432.sh, check_common435.sh, run_analytic_replay432.sh and the two
acceptance-control suites. Large analytic NDJSON exports are stored as deterministic
gzip files; decompress into replay432/Analytic/Lemmas.ndjson for offline audit.
The receipts bind all current proof sources, dependency artifacts and tool binaries.

For looped-layer experiments, the common gate and normalized remainder constants
supply concrete register ranges and correction budgets to test or enforce during
training. These are invariants of the specified decoder, not empirical evidence
that a trained transformer implements it. No new transformer performance claim is
made by this formalization.

The review whitespace check reports trailing spaces in the frozen coefficient proof source and CRLF endings in a few source/configuration and raw log files. These bytes are retained to preserve the exact verified source and evidence hashes; this is a formatting finding, separate from the passed mathematical and acceptance checks.
