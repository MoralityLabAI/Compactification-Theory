# SMC-14: Calibration-frozen SAE feature matching predicts heldout causal digit-margin responses

**Executed November 16, 2026.** The independent mathematical companion [OLM-14 in MathCan](https://github.com/patrickdugan/MathCan/pull/91) is merged. [SMC-14 registered the causal task, matched controls, model pairs and acceptance gate before reading heldout response scores](SMC14_PROTOCOL.md).

The final [source-locked GitHub Actions run #43789275273](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43789275273) passed both the pilot regression and the complete seven-pair experiment plus its independent audit. The [full artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43789275273/artifacts/14190804468) includes 4,664,699-byte full.json and a separate audit.json. Its downloaded archive SHA-256 is 37c1f9b824ae252d099b519aa75a007902d4665ce05940c997598ee76d17190d.

## Source/model/data boundary

- Externally pretrained source A: third-party 44,001-parameter, 4-loop Sudoku recurrent teacher, source commit f0406195fece4d0db29ced28f3a69d1993d95b1e, weights SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1.
- Independently trained new teachers B501 and B701, using disjoint noncanonical Sudoku teacher-training and validation puzzles; their binary checkpoints and each of four 48→96 top-four SAE models are source-locked in SMC-13A. Original pretrained source SAE seed 71 is the exact frozen SMC-11C fit.
- The original SMC-13C 64-puzzle position-and-visible-clue-conditioned Hungarian assignments are independently recomputed with **all seven matched feature permutation SHA-256s and all seven clue-shuffled control assignment SHA-256s verified before loading any heldout hidden states**.
- Separate test cohorts: 64 novel-template 30-hole puzzles, 64 novel-template 34-hole puzzles, and 64 canonical-orbit 30-hole puzzles. All complete solutions are distinct, and the novel-template cohorts have a positive intercalate invariant proving exclusion from the original canonical source generator orbit.

For every puzzle, the visible blank pattern chooses eight evenly spaced missing cells. A positive and negative perturbation of physical L2 radius 0.2 is applied to exactly one frozen stage-one goal token per episode, then the teacher runs three FP64 recurrent steps while the other 80 token states remain untouched at injection. The task outcome is the central two-sided response of the **correct Sudoku digit minus the teacher's strongest clean competing digit**. This is numerical functional intervention evidence, with no Arb interval enclosure for the cross-checkpoint transfer.

The score is the correlation across the eight goal-token causal responses after subtracting each puzzle's mean response. A single sign orientation is fitted on the 64 independent calibration puzzles for each assignment arm and then frozen. Target solution digits never fit the SAE, teacher checkpoints or Hungarian matching. Calibration labels enter the calibration-only orientation and final test labels enter evaluation.

## Primary source-A to independent-B causal transfer

All **four preregistered source→new teacher/SAE comparisons passed** the test on both novel 30-hole and novel 34-hole puzzle families. Each cell is an average across 64 independent puzzle clusters, not across 512 independent token examples.

| Frozen matched source and target SAEs | Novel30 matched causal cosine | Novel30 clue-shuffled mapping | Novel34 matched causal cosine | Novel34 clue-shuffled mapping |
|---|---:|---:|---:|---:|
| Source A SAE71 → B501 SAE71 | 0.4531 | 0.0520 | 0.4747 | 0.0723 |
| Source A SAE71 → B501 SAE89 | 0.5438 | 0.1164 | 0.5404 | 0.1608 |
| Source A SAE71 → B701 SAE71 | 0.4476 | 0.1383 | 0.4734 | -0.0036 |
| Source A SAE71 → B701 SAE89 | 0.4636 | 0.0988 | 0.5514 | 0.1010 |

The **paired matched-minus-clue-shuffled** 95% descriptive puzzle-bootstrap intervals were:

| Cross-checkpoint pair | Novel 30-hole interval | Novel 34-hole interval |
|---|---|---|
| A→B501 SAE71 | [0.2992, 0.5011] | [0.3024, 0.5049] |
| A→B501 SAE89 | [0.3144, 0.5339] | [0.2907, 0.4688] |
| A→B701 SAE71 | [0.1733, 0.4375] | [0.3727, 0.5866] |
| A→B701 SAE89 | [0.2444, 0.4820] | [0.3371, 0.5649] |

The source-matched assignments also outperform the seeded random-feature permutations, with all eight corresponding paired 95% descriptive bootstrap lower bounds above zero. The third preregistered within-puzzle goal-response shuffle produces matched causal correlations near zero and smaller than matched in every source→new pair and test family. The canonical-orbit cohort is a separate regression control, giving matched causal cosine about 0.47–0.57 across these four comparisons, with the same direction of advantage over their controls.

This passes the registered gate **as a predictive causal-response association**, conditional on the chosen task margin, calibration sign and controls. It establishes a meaningful next target for mechanistic investigation, while leaving the exact features' causal semantics unresolved.

## Independent model and SAE comparisons

The other three controls were retained under the original preregistered decision rule:

| Control pairing | Novel30 matched cosine | Novel34 matched cosine | Gate on both novel test families |
|---|---:|---:|---|
| Two SAE seeds, same B501 recurrent teacher | 0.3096 | 0.2693 | **Passed** |
| Two SAE seeds, same B701 recurrent teacher | 0.0545 | 0.0916 | **Failed** |
| B501 SAE71 → independent B701 SAE71 | 0.1182 | 0.0712 | **Failed** |

The failures are important: sharing the same teacher weights or a common recurrent architecture does not guarantee similar causal feature response patterns. The source-A→B effects can coexist with weak B-to-B causal transfer and pronounced source/new teacher capability differences.

## Capability and feature-activation caveats

On the selected eight heldout blank cells per puzzle, source A correctly predicts 94.5% of novel30 goal digits and 90.4% on novel34; B501 correctly predicts about 47.5% and 40.8%, while B701 correctly predicts about 46.3% and 43.2%. These values differ slightly from SMC-13's aggregate all-hole accuracy because this experiment selects eight fixed missing positions per puzzle.

The **matched target feature appears in the target SAE's active top-four set on only about 8.6–16%** of source→new selected test-goal tokens. Thus this experiment measures the output influence of transferred *decoder directions*, including atoms inactive in the receiving model. It does not show that the receiving teacher actively uses the matched feature as part of its normal reasoning trajectory.

The original B501→B701 matched feature response is weak and passes only the random mapping control on one of the novel cohorts. Training capability, task response heads and token representation conventions remain different among the teachers.

## Independent standard-library replay and harder target-label null

The [independent receipt auditor](audit_smc14.py) recomputes from the raw intervention records the seven exact feature-assignment hashes, all calibration-only orientation signs, 1,344 pair/family/puzzle score cases, 64-puzzle-cluster bootstrap confidence bounds and the registered pass/fail gates. Both the full numerical experiment and the auditor passed source-pinned hosted CI.

As a stronger **exploratory** control, the auditor shuffles target-model matched causal response values only within test-cohort groups sharing a correct Sudoku digit and the row's three-row band. It keeps the task-label/row-band-conditioned distribution of target response effects unchanged, disrupts individual source/target intervention correspondence and recomputes centered per-puzzle causal agreement over 16 seeded null replications. Each cohort has 27 nonempty conditioning groups, with no singleton groups.

The matched-minus-digit/row-band-null causal cosine advantages are:

| Cross-checkpoint pair | Novel30, paired mean [95% bootstrap] | Novel34, paired mean [95% bootstrap] |
|---|---|---|
| A→B501 SAE71 | 0.4199 [0.3454, 0.4950] | 0.4504 [0.3824, 0.5165] |
| A→B501 SAE89 | 0.4981 [0.4231, 0.5735] | 0.5209 [0.4530, 0.5872] |
| A→B701 SAE71 | 0.4191 [0.3358, 0.4973] | 0.4360 [0.3651, 0.5053] |
| A→B701 SAE89 | 0.3920 [0.3102, 0.4733] | 0.4903 [0.4153, 0.5616] |

This additional control **was constructed after inspecting the original experimental result**, so its results are correctly labeled posthoc. It cannot be counted as a preregistered independent replication. It narrows one plausible confound while leaving residual shared puzzle structure and teacher head geometry as possible explanations.

## Source-locked receipts and reproducibility

The final successful full run is [Actions #43789275273](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43789275273). Full experiment JSON SHA-256 is 2e52d66c15b21bbd5d9be7b27c9763f6b14312f0c2fc7e16e3e715ce0caf5f14. Independent audit JSON SHA-256 is c51174bcd9614eb3dff7935a44d838826d195526744172c2abf7f994c65ebd89. The complete ZIP SHA-256 is 37c1f9b824ae252d099b519aa75a007902d4665ce05940c997598ee76d17190d.

The [source protocol](SMC14_PROTOCOL.md), [intervention evaluator](smc14_causal_transfer.py), [receipt auditor](audit_smc14.py) and their respective unit tests reproduce the data from SHA-pinned SMC-11C/SMC-13A/SMC-13C artifacts. All source model and SAE snapshots remain untouched.

### Interpretation and next independent test

The study demonstrates that calibration-only SAE activation matching can predict **similar signed task-output responses to finite hidden-state decoder interventions** across independently trained recurrent Sudoku teachers. It passes controls for clue-conditioned calibration shuffling, random feature assignments, within-puzzle goal shuffling and, exploratorily, correct-digit/row-band shuffling.

The result does not yet establish causal *necessity* of naturally active features or correspondence of their internal reasoning algorithms. A decisive follow-up would directly ablate **features the receiving SAE actively encodes**, compare downstream task accuracy and goal-margin effects, and repeat on capability-matched independently trained teachers or a different model family. Source-coded tasks and empirical FP64 metrics remain distinct from the earlier exact-real Arb certificates.
