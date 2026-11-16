# Multivariate Taylor synthesis and constructed coefficient accuracy

This stage adds **47 authored theorem roots**, bringing the research campaign
to **257 roots**, including the earlier 12 attributed upstream Taylor roots.
The unchanged Hodge theorem remains separate. All 47 new roots compile from
frozen sources and pass native Lean kernel replay and independent OxiLean replay.

## Mathematical result

`MultivariateTaylor` constructs coefficients from Frechet derivative tensors at
a fixed anchor. Multilinear expansion over coordinate words and grouping by
multiindex prove that evaluating these query-independent coefficients gives
the actual multivariate Taylor polynomial. Dimension and degree are arbitrary.
The canonical rectangular index set has `(k+1)^d` elements, and the coefficient
of a multiindex with total degree above k is zero by its defining finite sum.

`MultivariateBounds` bounds those coefficients using operator norms, proves
coefficient perturbation estimates and derives the degree-n integer remainder.
`MultivariateHolder` derives the degree-k fractional remainder, for k>=1 and
alpha>=0, from a Holder estimate on the highest derivative tensor. Regularity
is actual global ContDiff on R^d; segment derivative bounds are explicit.

`JetSynthesis` and `FrameSynthesis` construct integer corrections, finite frames
and clipped reference states using the existing concrete Taylor transport
predictor. Rounding contributes at most tau/2. Clipping is nonexpansive toward
targets in [-C,C]. Consequently every reference state is within tau/2 of its
target, independently of stream length. The initial state is constructed by
the same rounding and clipping operation.

`SynthesisCapture` connects valid synthesized frames to the existing base-4
sentinel decoder, Hamming selector, common gate bound and decoded spatial
capture. `MultivariateCapture.jet_accuracy` instantiates its targets with the
derivative-tensor coefficients at the anchor sites. **The coefficient-accuracy
contract is proved, rather than assumed, in the final approximation theorems.**

For the selected anchor a and ||x-a||<=rho<=1, the actual decoded coefficient
vector q therefore satisfies:

* degree n, bounded derivative of order n+1:
  `|f(x)-evaluate(q,x-a)| <= M*rho^(n+1)/(n+1)! + (n+1)^d*tau/2`;
* degree k>=1, Holder exponent alpha>=0 on derivative k:
  `|f(x)-evaluate(q,x-a)| <= H*rho^(k+alpha)/k! + (k+1)^d*tau/2`.

Only the selected anchor must be close to the query. The coefficients, frames
and reference trajectory do not depend on that query. Valid sites, schedule
length, bounded step metadata, anchor derivative bounds and finite bit-width
inequalities remain explicit hypotheses.

The correction alphabet uses the proved general predictor bound:
`N >= (C+C*A_D)/tau+1/2`, `D<2^p`, `2*N<2^a`, with
`A_D=2^(k+1)*D^k` and `C=(sum_{j=0}^k d^j)*M`. A separate lemma proves that
finite parameters satisfying these inequalities exist. This bound grows as tau
decreases; it does not give the sharper small alphabet needed for the target
network parameter rate. The corresponding sharp transport residual estimate,
closed-cube/domain and geometric specialization, and dense ReLU arithmetic and
resource realization remain open. Polynomial evaluation uses exact real
multiplication. This stage does not finish the full quantized network theorem.

## Verification

| Check | Result |
|---|---|
| New authored theorem roots | 47 |
| Frozen-source Lean compilation | All seven modules passed |
| Native kernel replay | All seven modules passed |
| OxiLean dependency declarations | 28931 verified |
| Unsupported / rejected declarations | 0 / 0 |
| Native constants compared with export | 30423 matched |
| Acceptance tamper controls | 12 passed |
| Structural auditor control/equivalence tests | 7 passed |

Every exported dependency receives a checked per-declaration verdict. The export
graph is closed, has unambiguous names, and contains no unsafe declarations.
Every exported type, proof/definition value and universe-parameter list matches
the native environment; normalization only removes metadata and normalizes the
nondependent-let optimization flag.
Expression equality uses Lean's alpha equivalence for binder names/annotations.
Native and exported axiom inventories agree for each of the 47 roots and contain
only propext, Quot.sound and Classical.choice.
There are no admitted proofs or added mathematical axioms in the new modules.

Native checking calls the unmodified pinned LeanChecker.replayFromImports API
from its pinned module, omitting the stock CLI recursive search-path discovery.
DirectKernel.lean and direct_kernel.log record every module. The IO harness is
not part of the exported mathematical declarations. No Lean 4.35 replay is
claimed for these Mathlib modules; the main MathCan 4.35 toolchain is preserved.
All previously accepted proof source hashes and Hodge source bytes remain intact.

Mathlib stays at 360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56 (v4.32.0-rc1).
The 1,906 Mathlib/helper source files, eight pinned packages and 2,145 cached
artifact modules (19,305 retained artifacts) are validated before the frozen
rebuild and again for final acceptance after native replay. Official cached artifacts
are used; a full source rebuild of Mathlib
is not claimed. The dependency-closed exported proof terms are independently
checked. No extra Mathlib modules or software installs were required this stage.
The final export/source comparison uses a Linux-local copy of 8,580 runtime
artifacts (1,582,920,792 bytes), checked against the same pinned artifact receipt.
Source/copy hashes agree and every mirror file is rechecked around the replay.

Export SHA-256: `435ed39aa9be01548ecafc84c236f7a03bc0e635982f2ca25dd3adbf3e3679af`.
The BUILD_RECEIPT and acceptance files record source, artifact and binary hashes.
Startup failures before the frozen rebuild are infrastructure failures and are
not included as successful checks. Earlier pending checkpoints are historical.
The additional stock CLI path-discovery probe timed out (exit 124) and is not
counted. The accepted native result is the completed seven-module direct API
replay. A slow Windows-mounted export was interrupted and excluded; the complete
Linux-local replacement was checked and copied back with byte/hash equality.
The structural auditor's missing-name query was changed from repeated table
scans to direct dictionary membership. The original auditor remains intact;
all existing fixtures produce exactly equal results with the optimized version,
including target-scoped closure. The new graph is computed by audit_export_fast.py.

## Reproduction

Use the existing pinned campaign workspace. Restore the helper scripts into
CAMPAIGN_WORK and provision the previously verified analytic/core libraries and
recorded Mathlib artifacts. Run mirror_synthesis_mathlib432.py and set
SYNTHESIS_MATHLIB_MIRROR to the target recorded in mathlib_mirror.json before
using the replay helpers. Run the source admission scan and Mathlib validator,
then freeze_synthesis432.sh and build_synthesis432.sh. After the build finishes,
kernel_synthesis432.sh and export_synthesis432.sh may run concurrently: they read
the frozen artifacts and write separate logs. Run compare_synthesis432.sh after
the export/OxiLean stage. Once every replay and comparison has passed, validate
Mathlib again, recheck the mirror with mirror_synthesis_mathlib432.py --verify-only,
run accept_synthesis432.py and test_synthesis_acceptance.py, and
retain all logs. Publication requires every stage to finish successfully.
For slow Windows-mounted I/O, fast_synthesis_replay432.sh writes the export,
OxiLean results and source comparison to RAM-backed Linux temporary storage.
finish_fast_synthesis432.sh can resume the audit and source comparison in
parallel after a complete, per-declaration-verified export already exists.
After all checks finish, promote_fast_synthesis432.py verifies and bulk-copies
those results into the canonical evidence directory. The optional runtime cache
helper copies only recorded artifacts, checks source/copy SHA-256 equality and
uses available-memory and filesystem-capacity guards; it downloads nothing.
The accepted raw export is stored as deterministic gzip in pinned432_synthesis;
decompress it into replay432/Synthesis/Lemmas.ndjson to rerun offline acceptance.
The preceding CompareLemmas.lean supplies the independent parser comparison
template used by prepare_synthesis_replay.py. Runtime binaries and oleans are
not committed or bundled.

These invariants can inform register bounds and correction-accuracy objectives
for looped-layer transformer experiments. The Lean results establish properties
of the specified mathematical decoder and evaluator; no new empirical claim
about trained transformer implementations is made here. No external message or
repository comment was sent in this continuation.
