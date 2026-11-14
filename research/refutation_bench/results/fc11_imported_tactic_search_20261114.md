# FC-11: original MathCan intermediate Lean states and corrected tactic-search evaluation

Date: November 14, 2026. **The source, scratch verifier, model training and corrected frozen-checkpoint replay passed CI.**

## Source and frozen model provenance

FC-11 preregistration was committed first at b57e2500e829b12fdfc257077b8e410171c93b98. MathCan source is pinned to c84e2025334e9a967c1297425a490f98d5b3d189 and its actual Lean 4.35.0 toolchain. No original MathCan proof bodies were edited.

The train-only theorem selector was frozen at 2e7e749d4c7f8610a4083f4f8551d084102ea55a, before a held-out selector existed. [Train source CI #43540113159](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540113159) compiled 16 original research proofs and extracted **31 genuine intermediate Lean tactic states**. Goal-state JSONL SHA256: 47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d.

The held-out selector was frozen separately at b821e0d85423606cf971b1f80db8092ede693d81. [Test source CI #43540299693](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540299693) compiled 14 original proofs from Intervention, HookState and GraphAuthorization, extracting **29 actual intermediate states**. Goal-state SHA256: a8e7bee63f80c03c640ccdbb9bada1f8d943e33fb403e37ad1fc7dbf536e11ad.

Model training retained the original TypedLoop shared cell and a 62,401-parameter tactic scorer, four tied passes, seeds 17 and 29. It trained on original train-only source effects for eight registered tactics. The frozen checkpoints from [run #43541360928](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43541360928) were independently SHA256-checked before the corrected replay.

## Every registered tactic was actually tried during training

[Train-only probe CI #43541101044](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43541101044) passed 248 genuine Lean4.35 scratch tactic applications across 31 train states and eight scripts. A second independent model-training run reproduced every one of the 248 Boolean action outcomes and the exact candidate-source SHA256s; only wall-time fields differed.

| Tactic | Train states probed | Valid/progressive scratch outcomes | Closed scratch goals |
| --- | ---: | ---: | ---: |
| omega | 31 | 9 | 9 |
| decide | 31 | 2 | 2 |
| rfl | 31 | 1 | 1 |
| simp_all | 31 | 16 | 5 |
| aesop | 31 | 0 | 0 |
| assumption | 31 | 0 | 0 |
| intro | 31 | 3 | 0 |
| constructor | 31 | 2 | 1 |
| **Total** | **248** | **33** | **18** |

Eleven of the 31 original training states had no progressive action in the eight-script registry. Scratch preview runs may contain sorry and are NEVER formal certificates. A positive theorem claim requires separate sorry-free Lean compilation.

## First model run and invalid native baseline

[First complete model CI #43541890610](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43541890610) trained on source-only data, compiled **4/14** genuine held-out theorems for seed 17 and **1/14** for seed 29 from original model-selected actions, and verified that they contained no sorry. All five model-selected successes were GraphAuthorization arithmetic lemmas. The models failed all Intervention and HookState test theorems.

That first test incorrectly reported 0/14 for Lean-native search due an invalid generated tactic-search source. The generated source put a macro declaration before a subsequent import. This is an implementation defect, so the first run's apparent learned-over-symbolic victory **must not be interpreted as valid**. The original model proof successes and their frozen checkpoint hashes remain valid.

## Corrected post-hoc replay with unchanged model checkpoints

A new [independent CI job #43543404745](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543404745) downloaded the original immutable trained checkpoint artifact from run #43541360928. It preserved the same 14 held-out source theorems, the same Lean version and the same eight scripts, then evaluated the corrected Lean-native tactic search on identical model-specific time allowances.

| Seed | Neural accepted /14 | Corrected CPU-budgeted native accepted /14 | Native timeout cases | Native total elapsed ms | Model total elapsed ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | **4** | 3 | 2 | 14,475.85 | 22,920.84 |
| 29 | 1 | **4** | 1 | 14,662.53 | 32,006.77 |

All five model positives were separately compiled from the model's original tactic lists. For every theorem, the symbolic time allowance equals the matching model's measured total inference/compiler time; source and model hashes match. The seed-17 one-proof difference depends on two symbolic timeout cases whose measured overrun total was only 115.70 ms. Seed 29's native search exceeded the learner by three proofs. The measured neural CPU cost per accepted proof was higher in both seeds.

The corrected run is explicitly **post-hoc**, because the first preregistered comparison used invalid generated Lean syntax. Accordingly, it does not establish a confirmatory neural learning advantage.

## Relaxed native tactic-search ceiling

A separate [original-source native CI #43543846501](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43543846501) tested all 14 fixed held-out theorems with 20 seconds allowed per theorem. The corrected registered tactic search compiled **5/14**, consisting of all four GraphAuthorization lemmas and Intervention.obsEqOn_refl. It had zero timeouts and zero deadline overruns. These five formal proofs exceed both trained model counts. The remaining nine theorems need tactics beyond the limited search grammar and/or a more effective tactic sequencing strategy.

## Independent cross-artifact audit

The post-run local archive audit checked:

- All 248 train action/progress/closure outcomes and their generated source SHA256s against two separately compiled training runs.
- Original train and test goal-state SHA256s, with 31 and 29 states.
- Both original frozen checkpoint hashes and identical model action sources.
- All 28 model-case results and the 28 matching CPU-limited native records.
- Every accepted model proof's explicit separate sorry-free compiler flag, all case identities and the three-tactic bound.
- Exact case-by-case equality of symbolic allowed milliseconds to the corresponding model's measured elapsed milliseconds.

No certificate-status inconsistency was found. The audit records the model's original valid proof claims and the repaired native comparison separately. Source compilation under Lean 4.35.0 remains the formal authority.

## Research interpretation

This pilot establishes a working pipeline that captures actual intermediate tactic states from existing research theorems, trains the original recurrent cell from genuine Lean action effects, and tests closed-loop candidate proofs with a sorry-free kernel release gate. Learned proof-selection performance remained weak: it mostly selected arithmetic tactics and solved at most four of fourteen source-held-out theorems.

A subsequent preregistered benchmark should include positive training examples for each intended tactic class and expand the action vocabulary to source-relevant rewrites, exact hypothesis application, induction and case splitting. It should also measure a native Lean-native multi-tactic solver with the same source and CPU budget from the outset. Neither this result nor the original MathCan source proves unformalized mathematical conjectures.
