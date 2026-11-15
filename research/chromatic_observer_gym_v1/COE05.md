# COE-05: does a recurrent one-bit head encode the order of its visits?

**Executed:** 2026-11-15. **Scope:** six small CPU-trained frozen synthetic causal-attention models, exact phase-code control, deterministic reproducibility. **Primary status:** exploratory observational probe experiment and constructive mathematical positive control. No pretrained TRM, Huginn, Nanbeige, or Qwen weights were loaded.

Companion formal note: [MathCan PHASE_GAUGE_CAPACITY.md](https://github.com/patrickdugan/MathCan/blob/feat/coe01-chromatic-observer-20261115/research/compactification/PHASE_GAUGE_CAPACITY.md).

## Scientific question

COE-04 showed that the four-tick transcript of a shared, frozen one-bit threshold head improved evaluator separation over a one-bit final-only head. Yet the selected transcripts were no more informative than their **positive-bit count**. COE-05 asks whether selecting for an order-specific gain helps on unseen probe contexts. An independent exact cyclic-register model is used as a test of whether the proposed phase mechanism is mathematically possible at all.

For a chosen head `h=(w,theta)`, let `c(a,b)` denote its four-bit transcript across tied visits for source label `a` and distractor context `b`. Define

`order_only(u,v,b) := [c(u,b) != c(v,b)] AND [popcount(c(u,b)) = popcount(c(v,b))]`.

This predicate measures exactly those evaluator pairs distinguished by the **sequence** but merged by a count-only observer. Ordinary temporal separation is the disjoint union of count-separated pairs and order-only pairs.

## Provenance and experimental design

The base model implementation is `run_frozen_loop.py` from COE-04: a shared pre-norm causal multihead-attention block, four visits, two tokens, residual dimension 24 and three heads. It was adapted from the tied-visit equations in `MoralityLabAI/LoopedTransformers/research_gym/neural/looped_decoder.py` and is **not** a load of the external LoopedTransformers model weights.

- COE-04 model seeds `2406,2407,2408`: frozen and evaluated in the previous study. **Retrospective diagnostic cohort**, because their prior heldout contexts had already been inspected.
- New independent initialization seeds `2411,2412,2413`: each trained for the same `160` deterministic, full-batch AdamW steps on `64` synthetic `(source,distractor)` inputs, then frozen. All `64/64` first-token copy predictions succeeded before probe fitting. The model and probe tasks still share those 64 base inputs; the context holdout is for the *probe*, not the base model.
- Probe fitting receives contexts `b=0..3`, evaluation receives contexts `b=4..7`; the complete `K8` evaluator has `4*28=112` edge-tests per heldout model. This split, along with the fitter's deterministic candidates, is inherited from COE-04. The earlier negative result helped motivate the new head-selection objective, so this is **exploratory**, not a preregistered confirmatory result.
- Candidate generation is exactly the earlier `candidate_projections` list: fit-only between-class mean directions, fixed-seed random directions, thresholds from fit-only projection quantiles and per-label means. The selected head maximizes the lexicographically ordered tuple `(number of order-only edges present in at least two fit contexts, total order-only fit hits, all fit hits, -candidate index)`. Ties are resolved solely by the fit data and original candidate position.
- The `comparative_best_total_head` in each record is the same-pool head maximizing `(fit_all_hits, fit_order_only_hits, -candidate_index)`.

`phase_order_search.py` reruns exhaustive fit-only selection with NumPy for speed. `phase_order_replay.py` uses **only the Python standard library** to recompute every reported fit and heldout metric from the immutable selected head and frozen activation values. The input weights and activations are separately committed as gzip files for the new cohort; weight SHA-256s are verified against the stored model state dictionaries, without requiring Torch for replay. All original float32 weight arrays and all 64×4×24 activations per new model are retained.

## Heldout outcomes

| Cohort | Seed | Ordered edge hits /112 | Count-only hits /112 | Order-only gain /112 |
| --- | ---: | ---: | ---: | ---: |
| Previous (retrospective) | 2406 | 37 | 37 | 0 |
| Previous (retrospective) | 2407 | 69 | 69 | 0 |
| Previous (retrospective) | 2408 | 84 | 83 | 1 |
| **New model initialization** | **2411** | **82** | **78** | **4** |
| **New model initialization** | **2412** | **70** | **69** | **1** |
| **New model initialization** | **2413** | **91** | **86** | **5** |
| New cohort pooled | | **243/336** | **233/336** | **10/336** |

A phase-sensitive signal appears in the new cohort, but the gain from ordered history is **2.98 percentage points** of the total evaluated pairs. The selected head commonly loses evaluator distinctions compared with the head selected to maximize *total* fit coverage. No model-level statistical significance is assigned from three seeds or from correlated evaluator edges.

**Falsification boundary.** The new synthetic result does not demonstrate that larger *pretrained* recurrent transformers exploit noncommutative visit order. The external pretrained model weights were absent from this sandbox. Earlier LoopedTransformers summaries contain measurements and model tags but do not supply the exact raw residual sequences needed for a new probe fit. A native-pretrained execution will require pinned checkpoint and dataset hashes, pre-registered evaluator edges, per-visit latent traces, and exact group-disjoint probe splits; see the final section.

## Exact constructive positive control: an eight-visit affine-plane clock

The companion `phase_register.py` implements the time-independent map `R(word)=rotate_right_8(word)` and the tied readout `p(word)=word&1`. Each of the fourteen initial eight-bit registers is the indicator of one affine hyperplane in `F2^3`. This is a classical `S(3,4,8)` code. Each register contains exactly four one-bits, and every two distinct registers differ in at least four positions.

The exact oracle verifies:

- **91/91** K14 evaluator edges separated by the eight-bit *ordered* history, and **0/91** separated by the count-only observation (every count is four).
- **112/112** independently injected single-bit flips recovered uniquely by nearest-codeword decoding. These are all 14 labels × all eight flip positions.
- All **56** possible three-element subsets of the eight positions occur exactly once among the four-element supports. This meets the `14`-code upper bound from triple counting, while a seven-visit constant-count one-flip code can carry at most **seven** labels. The eight-visit construction is therefore sharp for 14 labels.
- Under an unknown uniformly random cyclic phase *per example*, the codebook falls into only **four** rotation orbits of sizes `[2,4,4,4]`. Its optimal noiseless uniform-prior label decoder falls from `14/14` to **`4/14`** accuracy. For count-only decoding the accuracy is `1/14`.

This phase-orbit result is a **clock-calibration obstruction**: a capable ordered observer can depend on knowing which repeated application was the first. The cyclic-register system is deliberately constructed; its results are *capacity witnesses*, not measurements on the trained tied-attention models.

## Reproduce

From the Compactification-Theory repository root:

```bash
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_*.py' -v
python research/chromatic_observer_gym_v1/phase_order_replay.py \
  --activation research/chromatic_observer_gym_v1/replication_activations.json.gz \
  --receipt research/chromatic_observer_gym_v1/new_fit_phase_receipt.json
python research/chromatic_observer_gym_v1/phase_register.py
```

Re-run the exhaustive fit-only head selection (requires NumPy 2.x; CI pins it):

```bash
python research/chromatic_observer_gym_v1/phase_order_search.py \
  --input research/chromatic_observer_gym_v1/replication_activations.json.gz \
  --out /tmp/coe05_selection.json
```

Regenerate the independent model seeds and their frozen weights (optional; requires CPU PyTorch as in COE-04):

```bash
python research/chromatic_observer_gym_v1/freeze_replication.py
```

The provided artifact hashes lock the exact data used for the reported result:

- `replication_activations.json.gz`: SHA256 `4a5c72532881c7b0f6d9d20bd006b5c50bed5ee65e815d3650746c2fed97cdc0`.
- `replication_weights.json.gz`: SHA256 `5c68678b33e9fb5a57890d4d43e56273573416470f085a6d7fa4095440a10097`.

## Next native-pretrained gate

Use the existing `MoralityLabAI/LoopedTransformers` LoopScope checkpoint capture machinery. Freeze one checkpoint and a group-disjoint board/task split before probing. Extract the **same residual site** for eight or more visit indices, with input identity, evaluator-pair label, final behavioral correctness, model checkpoint SHA and provenance recorded. Compare ordered-bit separation, count-only separation, unknown-phase control, and matched eight-bit final readouts on heldout groups. The evaluator should use independent causal or task-output evidence rather than the labels of a newly fitted probe. Keep unsolved or base-model-failed items in a separately reported stratum; a probe reading without task behavior has a different interpretation.

The proof manuscript and Lean exact-code certificates live in [MathCan PR #47](https://github.com/patrickdugan/MathCan/pull/47). The synthetic experiment and immutable witnesses live in [Compactification-Theory PR #143](https://github.com/MoralityLabAI/Compactification-Theory/pull/143).
