# SMC-12: Exact angular cones and trained decoder self-selection

**Completed November 15, 2026.** Source-locked mathematical experiment using frozen SAE tensors from the verified SMC-11C recurrence campaign. The mathematical companion is [MathCan OLM-12](https://github.com/patrickdugan/MathCan/pull/84).

Both jobs of [SMC-12 hosted Actions #43724511859](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43724511859) succeeded. The [archived exact certificates and dictionary-wide controls](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43724511859/artifacts/14165469079) contain 48 fully rational cone certificates and 576 additional exact trained-versus-initialization dictionary classifications. The complete unmodified archive SHA-256 is d76b811d2781c0b4e07cce0a76afbcbcc99279440fad025c7580c49939723a7d; its two member files have SHA-256 d63d3e96e9d4b39fce0ab372545800dacaf2b9178c16d99fa51ca0b436efa5d2 and 6331746aec8a6112b54b11ff7bf6fb707720a7e12ee035da63f6f49917404440, respectively.

## Exact source provenance

The teacher is an externally pretrained 44,001-parameter recurrent Sudoku model at upstream commit f0406195fece4d0db29ced28f3a69d1993d95b1e, checkpoint SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1, with the source-locked 64-puzzle roster SHA-256 eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.

The [original SMC-11C run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721281348) froze three top-four SAE fits at seeds 71,89,113 after 300 training updates, each on 32 training puzzles with 16 validation puzzle IDs and 16 *previously inspected* heldout puzzle IDs. All 81 tokens in each puzzle remained in a single split.

SMC-12 performs **no retraining**. It loads and hashes the original joint-full.json source and all three saved weights-only SAE snapshots, cross-checks the source teacher and roster and verifies the original canonical FP32 tensor fingerprints. The selected feature IDs and physical decoder directions come from those same SAE fits; the source weight/normalization conventions are identical to SMC-09/11.

## OLM-12: Exact all-radius cone theorem

A selected highest-positive SAE neuron t has physical affine score normals v_i = W_i/scale. Its positive-ray self-selection cone comprises every nonzero physical direction d satisfying

\[
v_t\cdot d\ge0,\qquad (v_t-v_j)\cdot d\ge0
\quad (j\ne t).
\]

If the source anchor strictly selects t, all positive decoder-ray amplitudes preserve its identity exactly when d belongs to that cone. The cone is polyhedral, with 96 nonzero constraints for each of the 48 source cases.

The exact-rational verifier computes the sharp normalized squared margin

\[
q_*^2=\min_{n\ne0}
\frac{(n\cdot d)^2}{\|n\|^2\|d\|^2},
\]

using the *exact dyadic FP64 decoder direction after the causal evaluator's normalization* and exactly stored FP32 encoder/scaling tensors. From q_*² follow three related robustness thresholds: relative physical raw decoder error \(q_*\), angular margin \(\arcsin(q_*)\), and unit-vector Euclidean chord margin \(\sqrt{2-2\sqrt{1-q_*^2}}\). Membership guards compare rational squares directly, avoiding rounding of inverse trig or square root functions.

### Sharp 48-case results

All **48 out of 48** selected decoder rays have strictly positive slopes for all 96 score constraints, and all 48 have a rigorously computed nearest cone boundary and a corresponding *exact rational adversarial witness*. Every witness reaches a zero score-gap slope at its sharp perturbation radius; overshooting the witness turns the slope negative.

| Margin across 48 selected goal-token/SAE cases | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Relative raw physical decoder error | **0.21679178** | 0.46391353 | 0.56146047 |
| Angular radius | **12.52067°** | 27.63993° | 34.15686° |
| Unit-vector Euclidean chord radius | **0.21809233** | 0.47774368 | 0.58736095 |

The minimum occurs at SAE seed 113, puzzle 7, selected feature 93, with competitor 25. The exact squared normalized cone margin is included as an unreduced-in-display rational in the evidence JSON.

| Relative physical raw decoder uncertainty | Certified interior decoder rays /48 |
|---:|---:|
| 10% | 48 |
| **20%** | **48** |
| 21% | 48 |
| 21.5% | 48 |
| 22% | 47 |
| 30% | 44 |
| 40% | 34 |
| 50% | 18 |

A unit-vector L2 chord perturbation of 0.2 is likewise certified for all 48. The cone condition guarantees that the highest positive feature remains t along **every finite nonnegative amplitude** in each admitted direction. It makes no claim of downstream causal effect ordering for all amplitudes.

The sufficient **joint constraint-normal and decoder error** condition from OLM-12D also passes for all 48 selected rays at 10% relative error in each score-constraint normal and 10% relative error in the physical raw decoder, *conditional on preserving the original selected source feature after changing the encoder*.

### Full top-four support is more fragile

**Zero of the 48 selected positive decoder rays preserve their entire top-four support to arbitrarily large positive amplitudes.** Each top-four cone contains at least one constraint with negative ray slope and therefore eventually reaches a finite selected/unselected crossing. This is consistent with SMC-11's exact finite-ray results: 47/48 preserve their four-feature support through amplitude 0.2, while 43/48 preserve it through 0.8.

The independent strongest-feature result is therefore strictly about top-one self-selection, rather than global invariance of the complete sparse code.

## OLM-12E: Cross-talk matrix mechanism and matched initialization control

Let W be the fitted encoder's linear weights and D the decoder matrix. In the ideal exact-real raw physical direction d_t=scale × D_t, the score change for neuron i is exactly

\[
v_i\cdot d_t=(WD)_{it}.
\]

Writing K=WD, the tth decoder atom reinforces its own highest feature at every positive amplitude precisely when its column is diagonally dominant:

\[
K_{tt}\ge0,\qquad K_{tt}\ge K_{jt}\quad(j\ne t).
\]

The [dictionary control script](smc12_dictionary_controls.py) applies that exact rational criterion to *all 96 atoms per SAE*, independently from evaluating actual FP32/F64-normalized physical decoder cones. All 576 matched trained/initialization classifications agreed with the ideal K criterion; there were **zero rounding-related classification discrepancies**.

### All dictionary atoms, trained versus original initialization

| Training seed | Trained SAE strict self-selection interior /96 | Original initialization /96 | Distinct selected heldout goal features |
|---:|---:|---:|---:|
| 71 | 54 | 1 | 12 |
| 89 | 53 | 0 | 9 |
| 113 | 50 | 0 | 11 |
| **Combined** | **157/288** | **1/288** | 32 distinct per-seed feature identities |

The random baseline uses the original TopKSAE architecture, original training seeds and the exact source-frozen training scales. Its three initialized decoder weight hashes are verified against the SMC-07 training receipts before computing any cone property. The definition is conditional: if a dictionary atom is strictly highest-positive at a source state, its self-selection ray is all-amplitude stable when its column passes the criterion. An atom can pass the matrix test without ever being selected in data.

The large change in the internal encoder-decoder cross-talk geometry is consistent with the SAE's reconstruction training encouraging self-aligned atoms. This three-seed finite result does not establish population-level generalization, feature semantics, causal necessity or an improvement in Sudoku accuracy. The downstream teacher's gradient sensitivity is measured by the separate causal Gram \(D^\top H D\), and cannot be inferred solely from the internal K=WD matrix.

## Rigorous limitations and next experiment

The source teacher, puzzle roster and SAE training recipes all come from earlier research. The 16 heldout puzzle IDs have been examined since SMC-07, so this is a **source-locked explanatory proof campaign**, not independent new generalization evidence.

The cone result assumes a frozen affine SAE encoder, originally strict selected feature and a physical decoder ray within the computed angular margin. It does not certify hardware rounding across arbitrary intervention amplitudes, decoder retraining, changed teacher weights, uncertainty in remote tokens or generalization to other recurrent models. The joint constraint-normal condition requires a separate start-of-ray feature-selection check under encoder changes.

A suitable follow-up is to directly train on a genuinely independent puzzle family or a separate model checkpoint, and compare how the WD self-selection columns transfer under SAE retraining and cross-model dictionary alignment. Such a test would distinguish reconstructed-code geometry from stable semantic and causal features.
