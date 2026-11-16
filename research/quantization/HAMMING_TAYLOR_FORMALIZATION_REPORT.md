# Hamming-selected Taylor transport and concrete spatial capture

2026-11-16, Asia/Taipei. **63 new theorem roots are proved and independently
replayed.** There are now 158 research theorem roots across the four corpora;
the unchanged Hodge theorem remains separate. The exact finite polynomial,
frame-decoder and spatial-capture semantics are proved. The full smooth-function
approximation theorem and dense compiler/resource theorem remain open.

## Proven modules

| Module | New lemmas | Main result |
|---|---:|---|
| `HammingSelector` | 11 | Little-endian amplitude words recover their integer; the ReLU Hamming selector is exactly one-hot; bounded gates select the candidate |
| `TaylorPolynomial` | 17 | Pascal coefficients and explicit finite polynomial operations; exact affine substitution and its coefficient formula at every degree |
| `TaylorTransport` | 19 | Concrete multivariate axis matrix, polynomial-fibre meaning, integer entries, and computable finite row bounds |
| `TaylorFrame` | 11 | Fixed-width amplitude/correction windows and the actual Hamming-gated Taylor update |
| `ConcreteJetCapture` | 5 | Layer-scheduled concrete recurrence equals packed decoding, then equals the spatially captured jet, including initial code perturbations |

All statements use Lean core's standard ordered-field classes. There is no new
real-number implementation, admitted proof, custom axiom, arbitrary candidate
matrix, or update-function oracle in the concrete capture theorem.

## Hamming selector and frame format

`digits p q` computes p little-endian Boolean digits by division and remainder.
For q<2^p, `unsignedNat (digits p q)=q`; hence the encoding is injective on
the represented range. The selector is ReLU(1-distance), where distance sums
the affine mismatch terms b for a zero pattern bit and 1-b for a one pattern
bit. At Boolean inputs of equal width it is one exactly at the matching pattern
and zero elsewhere. For D<2^p this selects precisely one candidate among 0..D.
The power-of-two width lemma covers D=2^r with p=r+1.

A frame is the p amplitude bits followed by J correction words of a bits each.
Its length is exactly p+a*J. Every correction window at offset p+a*beta is
proved to recover its own little-endian integer; the signed correction is that
integer minus the fixed centre. Empty bit windows and the D=0,p=0 case are
included when the stated encoding bounds hold.

## Concrete Taylor transport

For an axis j, step h and multiindices beta,gamma, the implemented matrix entry
is zero unless all other coordinates agree and beta_j<=gamma_j. Otherwise it is

    choose(gamma_j,beta_j) * h^(gamma_j-beta_j).

`choose` is defined by Pascal recursion and proved zero above the diagonal and
one on the diagonal. The integer-entry theorem embeds exactly the corresponding
Int matrix when h is integral. Axis locality and zeros below the diagonal are
explicitly proved.

The polynomial semantics is constructed independently of this formula.
`affineMul h` multiplies a coefficient list by X+h; repeated multiplication
constructs (X+h)^n. Its coefficient at beta is then proved to be
choose(n,beta)*h^(n-beta). Arbitrary finite sums of monomials satisfy exact
affine substitution. For every row of the multivariate matrix, the transported
value is proved equal to the normalized coefficient of the translated polynomial
fibre with the other coordinates fixed. This holds at arbitrary finite degree.

The index map is an explicit finite layout. The paper specializes it to its
distinct total-degree-bounded multiindices. Formal construction/cardinality of
that complete layout, and identification with normalized derivatives of a
smooth function plus a remainder, are not supplied by these polynomial identities.

## Gate budgets and the scheduled capture theorem

For every row beta define the explicit finite budget

    R_beta(D) = sum_gamma choose(gamma_j,beta_j) * D^(gamma_j-beta_j).

Terms above the binomial diagonal are zero. This conservative budget includes
entries excluded by the other-coordinate test. If |h|<=D and each state
coordinate lies in [-C,C], the candidate is proved to lie in
[-C*R_beta(D), C*R_beta(D)]. The Hamming circuit uses this computed gate bound;
it does not assume a bound on the desired candidate as a correctness oracle.
For a fixed index layout and D the budget is independent of the number of
spatial blocks and streamed sites.

This is a valid row-dependent gate construction. The sharper common paper
constant A_D=2^(k+1)*D^k, and the original compiler's uniform 2*A_D gates, have
not yet been bounded by a Lean proof in this phase. No complete dense-layer
parameter, depth, weight or precision accounting is claimed here.

`selected_update_frame` proves that the actual gate sum computes the signed-step
Taylor matrix times the current jet plus tau times the decoded correction.
Axis and orientation are fixed metadata of each recurrent layer; amplitudes
and corrections come from the sentinel word. They can change at every frame.
Coordinatewise clipping supplies the state invariant for the next update.

The main roots are:

- `MathCan.ConcreteJetCapture.concrete_spatial_capture`
- `MathCan.ConcreteJetCapture.robust_concrete_spatial_capture`

For a d-coordinate safe spatial record and B=K^d, B-1 valid frames provide all
B jet states, including the initial state. The selected serpentine rank is
proved to be within this actual stream. Packed decoding and bounded capture
return exactly the concrete reference jet at that rank. The robust version
allows the prior sentinel input-error budget for the entire payload. It does
not introduce fresh arithmetic error at every update.

The rank list uses highest-coordinate-first order. To match the Python snake's
physical low-coordinate-first tuples, reverse that tuple when constructing
the rank list; the implementation controls verify this convention explicitly.
The geometric derivation of the layer schedule from clipped anchors remains
a separate obligation.

## Verification evidence

- All five modules compile and pass `leanchecker` in Lean 4.32.0-rc1.
- All five also compile and pass `leanchecker` in the preserved Lean 4.35.0
  environment, whose Lean and leanchecker binary hashes remain unchanged.
- OxiLean v0.1.5: **3,580 verified, 0 unsupported, 0 rejected** records.
- Every one of the 63 new named roots and its complete dependency closure verifies.
- All **3,901 exported constants** match the native environment's
  names, kinds, universes, types and proof/value terms. Only expression metadata
  and let nondep hints are normalized; no proof substitution is used.
- Export SHA-256: `9c4f4270978238015613cdde938247549c35bbb4099c2be0b319e6fde85fed15`.
- Per-root native axiom inventories agree with the exported closure: only
  `propext`, `Quot.sound` and `Classical.choice` occur.
- Eight acceptance tamper controls pass. All five source hashes are checked
  before/after replay and previously accepted imported sources are unchanged.
- Exporter source: `3de59f10bc4b4a0f2de698597aeb1246caa0df0a`.
- Checker source: `3adb983d2aefc6a596bdc8b63438f54feecd1f55`.

The exact manifest, raw NDJSON, declaration verdicts, per-root dependency IDs,
native logs and reference-parser comparison are in MathCan's
`research/oxilean_v015/pinned432_transport/`.

The implementation controls pass **1,028 exact assertions**, including
168 polynomial-substitution cases through degree 3 in
dimensions 1..3, positive/negative/zero steps, integer entries, 48
decoded frame updates across 16 streams, empty windows, compiled signed-pair
agreement, frame suffixes, and every selected spatial state. These are finite
implementation checks, separate from the arbitrary-degree Lean proofs.

## Remaining full approximation obligations

1. Construct and count the complete total-degree multiindex layout; prove the
   sharp common row budget needed for the paper's stated constants.
2. Derive safe spatial records, layer axes/orientations and actual clipped
   displacement magnitudes from every relevant point of the cube.
3. Establish normalized smooth-function jets, Taylor/Hölder remainder and
   correction-size bounds for the concrete transport, using the existing
   quantized recurrence lemmas where applicable.
4. Prove packed-word/initial-jet coarse lookup, polynomial-network error,
   safe-phase coverage, median stitching and dense compiler/resource budgets.

## Reproduction and publication

Set CAMPAIGN_WORK to the original WSL work/ path, then run

```bash
bash "$CAMPAIGN_WORK/run_transport_replay432.sh"
python3 "$CAMPAIGN_WORK/test_transport_acceptance.py"
bash "$CAMPAIGN_WORK/check_transport435.sh"
```

The user authorized pushing the verified research branches. Exact remote heads
are checked in the output `PUSH_RECEIPT.json`. The previously prepared follow-up
to Tetsuya was sent after the user's separate approval; no additional message
is sent as part of this proof continuation.
