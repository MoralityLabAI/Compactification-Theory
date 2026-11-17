# HC-01: byte-to-abstraction evaluation protocol

Date: 2026-10-29. Status: **prospective** except for the exact finite reference in this directory. Neural experiments, compute budgets, model assets, source hashes and decision thresholds have not been frozen. No H-Net or RTX 5080 run is claimed.

## 0. Local reference gate

From the repository root, on Python 3.10+ with no external packages:

```bash
python -m unittest discover -s research/hnet_causal_chunking_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/hnet_causal_chunking_v1 -p 'test_*.py' -v
python research/hnet_causal_chunking_v1/reference.py
```

The exact checker validates total finite transition tables, computes the minimum future-observation partition, distinguishes observational sufficiency from quotient-transition factorization, and returns the shortest action suffix exposing a same-code behavioral collision. Tests compare partition refinement against an independent greatest-fixed-point oracle on 100 seeded finite systems, with negative controls for missing transitions and an intervention-induced collision. This is a synthetic proof-checking exercise, separate from trained-model evidence.

## 1. Freeze a training/evaluation manifest before seeing scored results

Record the dataset source, original byte length, normalization/UTF-8 policy, executable grammar and generator seed, train/validation/test family split, model and tokenizer/checkpoint SHA, routing implementation SHA, frozen hyperparameter grids, GPU/driver/framework revisions, actual training and inference FLOPs, measured wall-clock, power if available, VRAM and host-memory peaks, transfer bytes and live/cache state. Define the reachable domain, intervention grammar, target readouts, chunk-phase alignment, test horizons, held-out units, and statistical method.

Keep independent source bytes distinct from token equivalents. Freeze the compression-rate band and measure bytes per emitted chunk. Match full-model parameters and optimizer searches as closely as feasible; report mismatches. For every student or teacher, include the learned boundary generator and any adaptation or retraining cost in the resource ledger.

## 2. Two distinct comparisons

**Learning efficiency:** compare (a) matched total pretraining compute to reach a fixed held-out byte-normalized loss, (b) matched original training bytes to reach a fixed transfer-task target, and (c) final loss at a common compute cap. Use training seeds 0, 1 and 2 as proposed pilot seeds, pending a source-hashed manifest and full budget. Equalize the data mixture or document differences.

**Preservation of causal distinctions:** freeze an encoding from a given trained model, construct same/near-code state pairs under a declared precision policy, replay common future action words and measure output/decision disagreement. For a finite exact simulator, enumerate all reachable state-action pairs. For a neural model, distinguish held-out collision *rates* from any claimed uniform certificate. Evaluate horizons 0, 1, 2, 4 and 8 and report action margins before treating readout closeness as decision agreement.

## 3. Model and boundary controls

- BPE-tokenized transformer and a byte-level nonhierarchical baseline, separately tuned under the same published search budget.
- Fixed stride at the learned chunker's observed average compression rate; a fixed semantic/whitespace heuristic only as a separately labelled prior.
- Learned online boundaries jointly trained with the model, alongside frozen pretrained boundaries at similar chunk count.
- Shuffled boundaries matched in number and approximate length distribution, plus a boundary-only auxiliary loss ablation when the source implementation permits it.
- Where feasible, a BLT-like entropy boundary policy, with its own amortization and inference overhead reported.

Compare both full end-to-end systems and a frozen-backbone/router-specific ablation. Full-model improvements alone do not locate a mechanism in the router. When reusing external H-Net code, pin a checked commit; the released 2025 implementation is not presumed identical to the newly announced 2026 model.

## 4. Small evaluation families

**HC-SYM (symbolic syntax).** Generate synthetic grammars with held-out operator names, escaped bytes, Unicode encodings and unseen delimiter conventions. Include executable parser and equivalence transformations. Score byte-normalized prediction and exact parsed-output agreement. A renamed-symbol variant tests whether learned structure survives surface renaming.

**HC-MDP (Storyworld strategic rules).** Generate finite state/action games with an exact dynamic-programming oracle and probability-bearing transitions. In the held-out split, mutate rule schemas, action names, transition weights and nesting depth without leaking the new rules into training. Compare the policy produced from compact states to the exact oracle, measuring regret, probability calibration, rule-adaptation examples and total retraining/teacher costs. Keep the larger-model game-designer loop as a separate, fully charged experimental factor.

**HC-INT (causal boundary adversaries).** Construct matched prefixes whose proposed chunk encodings collide, then intervene by inserting, deleting or reordering bytes/actions through an explicitly specified state transition. Test whether a route flip changes later action predictions. Include the 2-bit shift fixture from the exact reference as the minimal negative control, with hidden buffer and alignment state in the concrete machine.

All three families require independent, immutable test seeds and a reproducible generator/manifest before numerical claims. Existing IC-SW-01, IC-01 and E10 assets retain their owners and definitions; this document does not claim that fixtures, H-Net checkpoints or GPU traces already exist.

## 5. Primary metrics and decision rules

Report held-out bits-per-byte with normalization made explicit, source-byte learning curves, downstream success and regret, intervention-conditioned error and collision counts, encoder/code length in **bits**, bounded-precision perturbation flips, chunk-rate distribution, train/inference FLOPs, per-byte latency, memory and cost. Publish raw predictions with seeds and hashes, including failures.

An **exact** quotient fails when two states with identical encoded codes produce different outputs under an admitted suffix. Transition factorization also fails when same-code states yield different successor codes under an action. A finite minimum-class count gives a lower bound under that exact declared operation alphabet.

An **approximate** uniform guarantee fails on a verified example above the claimed tolerance, including the near-fiber lower-bound condition. Numerical inequalities require independent rounding bounds and a justified Lipschitz constant. Held-out average accuracy supports a distributional conclusion only.

An **incremental learned-boundary** claim needs a preregistered minimum effect, uncertainty interval and matched-control gate. If the controls reach the same target within that gate, treat the specific benefit hypothesis as unsupported. Untested alternatives, missing checkpoint assets and absent matched-construct controls are inconclusive. No significance threshold or GPU measurement is asserted at this stage.

## 6. Execution boundary

The included standard-library fixture runs quickly on CPU. A neural pilot would first pin code and inputs, allocate separate compute budgets, secure reproducible H-Net source and model artifacts, and then run the RTX 5080 benchmarks under their existing workstation owner. This PR does not trigger those jobs or change production inference paths, model checkpoints, CI or E10 selectors.
