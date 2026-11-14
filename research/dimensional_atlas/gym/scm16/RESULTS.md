# SCM-16: Executed local CPU study

**Execution date:** 2026-11-14. **Backend:** synthetic finite-ring algebra (distinct from EvalMath), PyTorch 2.10.0+cpu, 32 hidden, four heads, four visits, rank-8 final readout, SAE 64/TopK6, VPD rank 4. All recurrent cores physically tied. **Source:** `results/scm16_toy_3seed_receipt.json`.

Each of seeds `160`, `1157`, `2154` ran 120 tied-base, 60 SAE, 180 AT-VPD and 60 frontier-replay optimizer steps, 420 per seed. The three models share 96 held-out synthetic periodic-generator tasks. 64 fixed public-DSL programs and 12 paid exact candidate checks per task. The independently scored new challenge set contains 16 tasks from four discovery first-fault local maxima.

| Arm | Seed 160 | Seed 1157 | Seed 2154 | Total correct / 288 |
| --- | ---: | ---: | ---: | ---: |
| Tied base | 53 | 60 | 46 | **159** |
| Joint SAE/fiber AT-VPD | 54 | 62 | 47 | **163** |
| Visible only | 54 | 61 | 44 | **159** |
| Fiber only | 55 | 60 | 47 | **162** |
| Shuffled visit gates | 55 | 62 | 46 | **163** |
| Zero SAE code | 53 | 60 | 46 | **159** |

`joint - tied = +4/288` is descriptive and small. The shuffled-visit arm achieves exactly the same total. Therefore **no evidence of specialized visit-conditioned reasoning or general capability gain** has been established. These seed-level observations share the same test tasks and generated families, so 288 is not a count of independent generator families.

### Exact symbolic controls

| Exact syntax-order control | Correct / 96 | Mean charged verifier queries |
| --- | ---: | ---: |
| First 12 programs | 45 | 9.04 |
| Exhaustive 64-program search | 95 | 17.31 |

The exhaustive control's one remaining failure arises because finite-prefix agreement does not uniquely determine the hidden future. Its larger exact-query budget is separately charged, and its CPU symbolic computation is outside neural MAC counts.

### Discovery local maxima and new evaluations

Of the first 24 discovery cases, the generator chose four *strict first-fault* local maxima of the exact prefix-agreement score among nine single-operator edits, with local-peak first-fault scores `12`, `8`, `6`, and `5` out of 13. It executed 1,536 additional exact verifier queries for landscape discovery. The four source cells produced 16 fresh evaluation cases and 16 replay-training cases, all with unique public signatures and separate seed ranges. Source cell selection is learned from discovery feedback; these challenges are explicitly *adaptive, exploratory* evaluations.

Fresh challenge accuracy: tied `11/16,11/16,10/16`; joint `11/16,11/16,10/16` in the three seeds. The joint adapter supplies no measured new-challenge advantage here.

### Causal dimensional readout

The final learned linear readout's kernel projector has a maximum observed direct readout residual below `1e-6`. A matched-norm probe of the kernel fiber on a discovery example changes the final logits by only `0.8–1.6e-6` when inserted after the final recurrent block. Earlier injections cause nonzero logit changes, establishing a toy *causal pathway* from head-orthogonal state into the later visible readout. The same experiment uses visible-direction patches as its comparison. This is a local, one-task, float32 diagnostic and has no topological certificate.

### Boundaries

All claims concern a trained **synthetic task-specific** 32-hidden model and the chosen readout. The finite-difference atlas cannot establish intrinsic neural topology; the original SCM-14 source Hessian certificates concern separate frozen synthetic models. Hosted EvalMath results are recorded separately below; evaluations on pretrained looped-model checkpoints remain pending. Private challenge gold exists only outside the committed source package.

## Hosted pinned EvalMath integration (completed)

**GitHub Actions:** [SCM-16 workflow run #6](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43565693173), completed 2026-11-14, exact oracle checkout at `88bb6cf8d2fe172e37d25964dfb2c285d9033841`. Both hosted jobs completed successfully. The run's public-evidence artifact is [ID 14101849183](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43565693173/artifacts/14101849183). Bytewise original public receipt SHA-256: `e4bbe8d322db87b6c3d7556b3c0e2fd0b02a2a2c7230c5737d19c30e282a3555`.

The exact EvalMath roster contains 168 positives from seven source families, twelve deliberately inconsistent negatives and twelve finite-prefix collision cases (six duplicate-public-signature pairs), yielding **192 rows but 186 distinct visible signatures**. It is an existing *public* benchmark, accessed only after these models had frozen. The trained model is still a small synthetic task-specific transformer, not a pretrained Qwen/LOOP-T checkpoint.

| Arm | Seed 160 | Seed 1157 | Seed 2154 | Future-correct / 576 |
| --- | ---: | ---: | ---: | ---: |
| Tied baseline | 79 | 80 | 76 | **235** |
| Joint SAE/fiber AT-VPD | 76 | 81 | 80 | **237** |
| Visible-only correction | 76 | 83 | 77 | **236** |
| Fiber-only correction | 77 | 82 | 75 | **234** |
| Shuffled visit gates | 76 | 82 | 77 | **235** |
| Zero SAE codes | 79 | 80 | 76 | **235** |

The joint advantage is **2/576 = 0.35 percentage points**, and the seed-160 model loses three problems. Pairwise wins/losses are `13/16`, `11/10`, `9/5` (joint-only/tied-only), a mixed outcome. All three models correctly abstained on **12/12 deliberately inconsistent negatives**; the `future_correct` metric assigns these no positive credit, so report abstention separately. Across the 168 positive examples only, tied future-correct is `75,75,73` and joint future-correct is `71,77,77`. All sixteen newly generated EvalMath challenges are disjoint from training and official test by public signature; tied vs joint gets `8/16 vs 9/16`, `12/16 vs 10/16`, `11/16 vs 12/16`, totalling **31/48 for each**. The four discovery local maxima had first-fault scores `12,8,8,6`, all from discovery tasks.

The exact symbolic controls, evaluated at different paid verifier budgets, achieved:

| Exact source-DSL control | Correct / 192 | False future acceptances | Mean paid queries |
| --- | ---: | ---: | ---: |
| First 12 lexicographic programs | 86 | 12 | 8.87 |
| Exhaustive 64-program syntax search | 163 | 17 | 19.23 |

The exhaustive solver carries stronger observable-prefix coverage than any learned arm but still fails future-continuation ambiguities. Its paid-query budget differs from the neural top-12 budget, and no policy has compute-matched base training costs. No statistically established capability or virtual-depth advantage follows from these results. The exact last-visit readout kernel and earlier causal response were verified by finite-float interventions on each trained toy instance; they establish a mechanism candidate within this synthetic architecture.
