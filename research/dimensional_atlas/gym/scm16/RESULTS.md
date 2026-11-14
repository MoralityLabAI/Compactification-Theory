# SCM-16: Executed local CPU study

**Execution date:** 2026-11-14. **Backend:** synthetic finite-ring algebra (distinct from EvalMath), PyTorch 2.10.0+cpu, 32 hidden, four heads, four visits, rank-8 final readout, SAE 64/TopK6, VPD rank 4. All recurrent cores physically tied. **Source:** `results/scm16_toy_3seed_receipt.json`.

Each of seeds `160`, `1157`, `2154` ran 120 tied-base, 60 SAE, 180 AT-VPD and 60 frontier-replay optimizer steps, 420 per seed. The three models share 96 held-out synthetic periodic-generator tasks. 64 fixed public-DSL programs and 12 paid exact candidate checks per task. The independently scored new challenge set contains 16 tasks from four discovery first-fault local maxima.

| Arm | Seed 160 | Seed 1157 | Seed 2154 | Total correct / 288 |
| --- | ---: | ---: | ---: | ---: |
| Tied base | 53 | 60 | 46 | **159** |
| Joint SAE/fiber AT-VPD | 54 | 62 | 47 | **163** |
| Visible only | 53 | 60 | 44 | **157** |
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

All claims concern a trained **synthetic task-specific** 32-hidden model and the chosen readout. The finite-difference atlas cannot establish intrinsic neural topology; the original SCM-14 source Hessian certificates concern separate frozen synthetic models. The EvalMath pinned-backend run and any fresh-pretrained-model evaluations must be reported separately when their hosted jobs complete. Private challenge gold exists only outside the committed source package.
