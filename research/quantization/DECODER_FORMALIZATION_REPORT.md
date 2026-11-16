# Sentinel decoder and spatial jet capture: Lean continuation

2026-11-16, Asia/Taipei. **39 new theorem roots are proved and independently
replayed.** There are now 95 research theorem roots across the three corpora;
Hodge remains a separate unchanged theorem. The full analytic approximation
theorem is still open.

## What is now proved

`SentinelDecoder.word` encodes a Boolean payload followed by a trailing digit-one
sentinel: word([])=1/4, word(b::tail)=(1+bit(b)+word(tail))/4. The scalar field
is arbitrary subject to Lean core's standard ordered-field classes, as in the
previous spatial proofs. These statements do not construct a concrete real type.

1. Every suffix lies in [1/4,2/3]. The exact two-hinge reader
   4*(ReLU(4*z-7/4)-ReLU(4*z-2)) returns the leading payload bit. Clipping the
   suffix update 4*z-1-readBit(z) into [0,1] preserves the encoded tail.
2. All payload bits, concatenated prefixes, and frame boundaries decode exactly.
   After all S payload reads, the sentinel remains at 1/4. The theorem does not
   treat an additional sentinel read as a payload read. Empty payloads and empty
   frames are covered.
3. At each valid read, a two-sided perturbation of magnitude at most 1/48 leaves
   the bit exact and changes the suffix error by exactly a factor of four.
   `noiseBudget(0)=1/48`, `noiseBudget(S+1)=noiseBudget(S)/4` gives a sufficient
   full-payload input budget. This is 1/(48*4^S); it is a conservative budget.
   Prefix and frame theorems propagate the remaining budget explicitly.
4. The terminal suffix error equals 4^S times the initial code error. This is a
   sensitivity result, not length-independent numerical stability. No theorem
   here covers arbitrary fresh arithmetic noise at every recurrent update.
5. The encoded word times 2^(2*(S+1)) is an integer numerator, which lies below
   4^(S+1). Thus 2*(S+1) binary fractional slots suffice for the stored word.
   This statement concerns representation of the parameter, not noise tolerance
   of activations or the total precision budget of the network.
6. Little-endian payload integers and signed corrections decode exactly.
   A finite matrix-plus-correction update constructor is supplied. More generally,
   every deterministic finite-dimensional update receives exactly the same frame
   bits in the reference and decoded recurrences. Coordinatewise clipping bounds
   every resulting jet state and preserves states already in the allowed range.

## Connection to spatial capture

The main roots are `MathCan.DecodedJetCapture.decoded_spatial_capture` and
`MathCan.DecodedJetCapture.robust_decoded_spatial_capture`.

For valid safe-cell records, the earlier spatial circuit extracts fine indices,
reads parity, and computes the clipped normalized serpentine rank. If the number
of payload frames is B-1 for B=K^d sites, that rank selects an actual jet state in
the frame stream, including the initial state. The sentinel decoder supplies
each frame directly from the one packed word. The captured vector is exactly
the corresponding reference clipped jet state, including under the stated
initial code perturbation budget. No decoder-correctness premise or external
jet-bound premise is substituted for this composition: bounds follow from the
initial-state bound and the clipped updates.

The update function is a parameter of the semantic theorem. The provided matrix
constructor makes finite matrix transport plus correction an instance; proving
that the particular Hamming amplitude selector and Taylor transport matrix
implement the paper's quantized jet remains a separate obligation. This does
not establish the full Taylor approximation or a dense ReLU resource bound.

## Verification

- Lean 4.32.0-rc1: both modules compile and pass `leanchecker`.
- Preserved Lean 4.35.0: both modules compile and pass `leanchecker`; original
  Lean and leanchecker binary hashes are unchanged.
- OxiLean v0.1.5: **3,243 verified, 0 unsupported, 0 rejected** declarations.
- All 39 named new roots and their complete dependency closures verify.
- All **3,560 exported constants** match native names, kinds,
  universe parameters, types and proof/value terms. Only metadata and let nondep
  hints are normalized, with no proof substitution or reduction.
- Export SHA-256: `3c3db534bde131f43b75903408cd66a8a4cddd94fa78d789bac4fb351843519b`.
- Native and exported root axiom inventories agree: only `propext`, `Quot.sound`,
  and `Classical.choice`. There are no admitted declarations or custom axioms.
- All eight acceptance tamper controls pass. Source hashes are checked before
  and after the pipeline, and the previous four spatial sources remain unchanged.
- Exporter source: `3de59f10bc4b4a0f2de698597aeb1246caa0df0a`.
- Checker source: `3adb983d2aefc6a596bdc8b63438f54feecd1f55`.

The complete manifest, per-declaration verdicts, root closures, native logs,
source comparison and acceptance data are in `pinned432_decoder/` in MathCan.
The prior Hodge, 14-lemma and 42-lemma artifacts remain separate historical runs.

## Contribution to looped-transformer mechinterp and guided training

The direct contribution is now a checked specification and concrete reference
data: **95 generic research theorem roots**, plus **144 positive
decoder trace fixtures**, one no-sentinel negative control and **1,896
exact decoder steps**, with payload lengths through 64.
The register and signed-pair ReLU evaluators agree at every fixture step. The
fixtures include split labels for training-pipeline development, calibration,
and longer-length checks. They are public fixtures, not a sealed evaluation set.
The serializer/data loader itself has not been Lean verified.

The shared-transition and state-factorization framing matches the program's
[looped/TRM/LDT taxonomy](https://github.com/MoralityLabAI/HybridTRMLDT/blob/e412e84131548e5b0c7534bd6f440a9089962b73/docs/looped_transformer_trm_ldt_taxonomy.md).
The current [looped-mechinterp paper](https://github.com/MoralityLabAI/LoopedTransformers/blob/88b99b97f1e8e6d1d5117f5eeaafc841d3639025/papers/rmp_looped_mechinterp/main.tex)
already frames interpretation in terms of repeated invocations and exact task
traces. The proposed connection is to use named code, suffix, correction, jet,
and capture registers as intermediate supervision and causal-intervention
targets, indexed by recurrent visit and computational phase.

No transformer training job, improved accuracy, recovered neural mechanism or
transfer effect was demonstrated in this continuation. A useful next experiment
compares output-only supervision with transition/invariant guidance at matched
compute, freezes an independent length-generalization set, and checks whether
interventions transfer the named state variables. A learned mechanism must be
tested separately from correctness of its reference algorithm.

## Remaining approximation obligations

- Derive safe-cell records from all safe shifted-grid points and formalize
  multidimensional coarse lookup of the packed word and initial jets.
- Specialize frame parsing to displacement/amplitude and all J correction windows;
  prove the Hamming selector, actual Taylor transport and correction-size bounds.
- Supply real-field/analytic foundations, Taylor remainder and polynomial-network
  error estimates, safe-phase counting and median stitching.
- Prove dense compiler correctness, staged selector amplification and all weight,
  depth, parameter and precision budgets.

## Reproduction and publication

Set CAMPAIGN_WORK to the original WSL work/ path and run

```bash
bash "$CAMPAIGN_WORK/run_decoder_replay432.sh"
python3 "$CAMPAIGN_WORK/test_decoder_acceptance.py"
bash "$CAMPAIGN_WORK/check_decoder435.sh"
```

The user authorized publication of the existing research branches on 2026-11-16.
Exact pushed heads are verified separately in the output `PUSH_RECEIPT.json`.
The follow-up to Tetsuya is an unsent draft; no email or repository comment was sent.
