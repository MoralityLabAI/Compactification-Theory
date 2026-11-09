# RRC-07: learned recurrent self-refinement and charged exact feedback

**Executed 2026-11-09.** Single CPU-only training seed, frozen evaluation. RRC-06's raw-coefficient base checkpoint is held fixed. The new code adds a shared-weight two-pass recurrent corrector with 4,227 weights, a 4,297-weight feedforward comparator, a label-shuffled recurrent negative control, and an exact first-fault feedback interface with explicit query counts. NumPy accelerated inference was checked against the independent scalar implementations to within `1e-10`; NumPy does not alter the model architecture or supply algebraic information.

## Primary held-out result: new instances, already-known grammar families

The main scored set contains 72 newly generated periodic-part product examples with unseen-to-RRC-07-training moduli 5, 7, 11, 13, 17 and 19. Two of their prefixes alias examples in the **previous** RRC-06 training roster; the remaining 70 are distinct from both RRC-07 and RRC-06 training prefixes. Previously studied mathematical families are therefore used as fresh instance-level controls rather than claimed as a novel blind source family.

| Architecture | FP32 parameters, RRC-07 add-on | Correct Euler exponents /4,608 | Mean degree of first wrong exponent | Exactly verified programs /72 |
|---|---:|---:|---:|---:|
| Frozen RRC-06 base, two GRU scans | 0 | 2,894 (62.80%) | 4.7500 | 0 |
| RRC-07 recurrent, two refinement scans | 4,227 | 2,894 (62.80%) | **6.7639** | 0 |
| RRC-07 feedforward, one pass | 4,297 | **2,975 (64.56%)** | 5.8056 | 0 |
| RRC-07 shuffled-supervision recurrence | 4,227 | 1,938 (42.06%) | 1.8611 | 0 |
| Exact symbolic Euler/grammar reference | shared algorithm | 4,608 | complete | **72** |

Matched by example, recurrent refinement advances the first-error index on **25** cases and worsens it on **one** relative to the frozen baseline, leaving **46** unchanged. Against the feedforward control it wins **23**, loses **11** and ties **38**, with a descriptive two-sided sign-test `p≈0.0576`. The mean paired shift is **+2.0139 degrees** over frozen RRC-06 and **+0.9583** over feedforward. These fixed-roster comparisons test the direction of the effect, with no multiple-seed or corrected family-level significance claim.

The recurrent first-fault location moves roughly **2.01 degrees** beyond the frozen base and **0.96 degrees** beyond feedforward at trained depth. It gives up global exact-coordinate accuracy relative to feedforward, and it does not produce a fully verifiable program. Each coordinate is rounded to an integer; any nonzero exact series residual rejects the whole proposal. Retaining this as a mixed partial-positive/complete-program-negative result avoids overstating accuracy.

### Recurrence-depth ablation on the same 72 cases

| Extra learned scans | Correct coordinates /4,608 | First-error degree | Accepted programs |
|---:|---:|---:|---:|
| 0 | 2,894 | 4.7500 | 0 |
| 1 | **2,942** | 5.9028 | 0 |
| 2 (trained) | 2,894 | **6.7639** | 0 |
| 4 | 2,884 | 6.7222 | 0 |
| 8 | 2,881 | 6.5278 | 0 |

The frozen two-pass checkpoint has no monotonic gain with more iterations, and the one-pass setting has more correct exponents overall. Repeated forward passes cannot be claimed as a self-correcting proof mechanism on the basis of these numbers.

## Exact-residual assistance, separately identified

The verifier supplies the **entire** 64-coordinate coefficient-residual vector. At the earliest nonzero index `m`, the exact correction is `R[m]/m`, so a policy with this interface can fix one index on each pass. This is analytic algorithmic information beyond the raw neural model's inputs.

| Initial proposal | First error before feedback | Fully accepted after 32 edits maximum /72 | Mean edits to full exact vector | Mean residual-vector queries to full exact vector |
|---|---:|---:|---:|---:|
| Frozen RRC-06 | 4.7500 | 55 | 23.8056 | 24.8056 |
| Recurrent RRC-07 | **6.7639** | **61** | 23.8056 | 24.8056 |
| Feedforward RRC-07 | 5.8056 | **61** | **22.6806** | **23.6806** |

With 64 corrections allowed, all three initializer classes recover **72/72** verified truncated programs and future predictions. The exact grammar synthesizer reaches **72/72** with one formal-series inversion and bounded grammar search. The RRC-07 recurrent student therefore improves the *position* of the first error and the 32-query acceptance rate relative to the frozen base, yet fails to reduce the total number of required corrections. Feedforward saves approximately 1.125 exact coordinate edits per case and uses far fewer neural multiply-accumulates.

A full residual query is charged as an `O(T²)` exact-integer computation at `T=64`. The entire repeated-feedback trajectory can cost `O(E T²)` for `E` incorrect coordinates and is slower in arithmetic-operation terms than a one-shot `O(T²)` inversion. A query here is an entire residual vector, not one cheap local scalar signal.

## Other evaluation families

| Source group | Number of cases | Exact symbolic accepted | Recurrent direct accepted | Feedforward direct accepted |
|---|---:|---:|---:|---:|
| Development periodic products | 96 | 96 | 0 | 1 |
| Unseen modulus, caps ∞/1 | 72 | 72 | 0 | 0 |
| Unseen cap 2 | 30 | 30 | 0 | 0 |
| Gap-constrained partitions | 30 | 3 | 0 | 0 |
| Irregular allowed parts | 12 | 0 | 0 | 0 |
| Late-defect collisions | 10 | 10 | 0 | 0 |

Five paired future mutants share every coefficient through 64 with a canonical product base and diverge at degrees 67,79,91,107,123. The exact synthesizer makes **five** future false acceptances among these ten candidates. The neural direct outputs abstain on all ten. Every feedback arm returns the same exact truncated state for each member of a pair, so the future ambiguity survives successful prefix certification.

## Leakage and resource ledger

RRC-07's train and test periodic-product prefixes have no duplicates. Two `local_gap` prefixes alias RRC-07 training, and two `unseen_moduli` prefixes alias *RRC-06* training. Development has 15 RRC-06 training aliases, and cap-2 has one. All are listed in the source-bound receipt. They exclude any interpretation of the entire roster as completely raw-input-blind across the research sequence.

Bare RRC-07 weights: recurrent **135,264 FP32 bits**, feedforward **137,504 FP32 bits**; the frozen RRC-06 component adds **64,896 FP32 bits** to both stacks. Recurrent extra hidden state is 32 FP32 values, plus the frozen 23-value GRU state and cached intermediate outputs. Approximate dense multiply-accumulates per case, excluding all nonlinearities, data features, verification, big-integer arithmetic, and model training: **245,180** for the frozen base, **536,640** for two new recurrent passes, and **4,095** for the feedforward add-on. The feedforward stack consequently uses about 249,275 MACs against the recurrent stack's 781,820. These are analytic counts, not wall-clock hardware profiles.

Training used **240 Adam updates per arm**, batch 64, a single CPU thread, and a fixed seed. The same trained weights were evaluated at all recurrence depths. The reproducible receipt records source and checkpoint hashes, exact roster hashes, alias counts, all split metrics, correction budgets, and per-case trace SHA-256. The scalar and NumPy implementations share the same raw inputs and stored weights, with checked numerical parity. The pilot uses no GPU or large-model fine-tuning.

## Reproduction

From the repository root:

```bash
python -m pip install 'numpy>=2,<3'
python -m unittest discover -s research/rrc_compactification -p 'test_rrc07.py' -v
python research/rrc_compactification/bench07.py --check
```

The frozen checkpoint can be retrained with an optional CPU PyTorch environment:

```bash
python research/rrc_compactification/train07.py --steps 240 --batch-size 64
python research/rrc_compactification/bench07.py --check
```

**Research judgment:** The first-fault metric improved through recurrent refinement, but neural exact program discovery remains unsolved. A strong next experiment would withhold complete mathematical *operator families* and test models able to construct their own algebraic correction circuits. Any exact residual access must remain a separately budgeted comparator.
