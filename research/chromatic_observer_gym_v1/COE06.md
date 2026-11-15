# COE-06: phase-blind observers, clock gauge, and a sharp 21-label certificate

**Research date:** 2026-11-15. The frozen-model study and the mathematical positive control are independent components. The first is a post-hoc read-only diagnostic on six context groups from COE-05; the second gives a fully explicit finite coding theorem under an unknown cyclic starting phase.

## Applied measurement

On held-out contexts 4–7 for the frozen synthetic tied-attention models 2411, 2412 and 2413, the selected fixed heads distinguish 243/336 ordered evaluator pairs. Canonicalizing each four-bit history under cyclic rotation reduces this to 233/336, identical to positive-count-only distinction. All ten additional order-sensitive hits were phase-origin-sensitive. The source activation and fit-only head receipts remain immutable and are validated by `phase_gauge_replay.py`.

The new mathematical positive control uses a **14-bit cyclic register**, one fixed low-bit readout and fourteen iterations. Its codebook has **21 labels**, constant Hamming weight six, and minimum inter-orbit Hamming distance four. Every cyclicly shifted input with zero or one corrupted bit is uniquely decoded: 3,885 of 3,885 exact cases. The count-only observer separates none of 210 label pairs.

## Sharp width certificate

The [pure math proof in MathCan](https://github.com/patrickdugan/MathCan/blob/feat/coe01-chromatic-observer-20261115/research/compactification/PHASE_BLIND_OBSERVER_CAPACITY.md) uses disjoint necklace classes of deleted `(w−1)`-subsets to bound the maximum phase-invariant code size. Exact exhaustive counting across `T=1..13` gives an upper bound of **17** labels for any length at most thirteen. Consequently fourteen visits are necessary and sufficient for 21 constant-weight labels robust to one flip with unknown phase. This is a theorem in the specified abstract coding model, not an empirical measure of transformer capacity.

The all-width finite counting proof uses only integer masks and combinatorics; the 14-visit witness is checked by independent brute-force distance, rotation and corruption loops. Lean checks the specific codewords and arithmetic budget consequences, while the written proof covers the general deletion-resource theorem. Source papers OpenAI Math #098 and #332 motivate task-relative compactification and stable observation; neither is assumed as an axiom.

## Reproduction

From the repo root:

```bash
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_phase_blind.py' -v
python research/chromatic_observer_gym_v1/gauge_orbit_codes.py
python research/chromatic_observer_gym_v1/resource_bound.py
python research/chromatic_observer_gym_v1/phase_gauge_replay.py \
  --activation research/chromatic_observer_gym_v1/replication_activations.json.gz \
  --receipt research/chromatic_observer_gym_v1/new_fit_phase_receipt.json
```

CI uses Python 3.12 with standard-library-only tests. Frozen weights and activation files were already committed in COE-05 and are reused without modification. No pretrained transformer weights were loaded or downloaded in COE-06.

## Scope

The cycle-register witness is a deterministic mechanism whose information survives unknown phase. The trained four-step attention models do not demonstrate that mechanism, and the phase-invariant quotient removes their observed order-only gains. The correct next experimental test uses native pinned LoopedTransformers or LoopScope checkpoints with eight or more stored visits, preregistered evaluator labels, and frozen group-disjoint probe splits.
