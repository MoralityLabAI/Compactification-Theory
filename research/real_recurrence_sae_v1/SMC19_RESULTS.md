# SMC-19: Independent SAE refits preserve feature coactivation more consistently than causal erasure

**Executed November 16, 2026.** The first standalone [SMC-19 preregistration commit fd96967](https://github.com/MoralityLabAI/Compactification-Theory/commit/fd96967fbf251c884a8a705d6d8a68c0c2cddd2e) fixed all seeds, two distinct experimental arms, calibration scheme, nulls, puzzle cohorts, intervention dose and gate **before** the four new SAE fits and new heldout state generation.

[GitHub Actions run #43810248525](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525) passed **all three separate source-locked jobs**: new SAE training+calibration freeze, four-puzzle pilot+independent audit, and complete fresh 128-puzzle × four-refit causal evaluation+independent standard-library receipt audit. The source-trained models/SAEs and original seven SMC-13C Hungarian assignments were checked against their unchanged provenance. No original checkpoint or earlier SMC-14–18 experiment was modified.

**Scientific registered result: two of four independent SAE refit comparisons passed the strict two-family causal-stability gate.** The exact joint orthogonal-basis transport control passed to approximately 10⁻¹⁴, as required by its algebraic invariance construction.

## Source-trained refits and the calibration/test split

Use only the original independent B501 and B701 recurrent Sudoku teachers (each fixed 44,001 parameters, four recurrent loops), and original SAE71 as the per-teacher reference. Each teacher receives two new **independently initialized** top-four 48→96→48 SAE fits, seeds 131 and 167. All four use the original unchanged 192-puzzle SMC-13 teacher training states and 64-puzzle validation states, 300 AdamW updates with training-only normalization and validation-loss-only checkpoint selection. Their checkpoint file bytes, FP32 tensor fingerprints and initial-decoder SHA hashes are preserved in the separate frozen artifact. All four selected step 300 by original validation normalized reconstruction loss:

| Frozen teacher | New SAE seed | Validation normalized MSE |
|---|---:|---:|
| B501 | 131 | 0.15397 |
| B501 | 167 | 0.13261 |
| B701 | 131 | 0.13113 |
| B701 | 167 | 0.13642 |

After the four snapshots were frozen, the original 64 **SMC-13 feature_calibration** Sudoku puzzles alone determined each new source-original-SAE71→refit-SAE131/167 **visible-position+digit-conditioned** Hungarian 96-feature assignment and its within-clue shuffled and random-permutation nulls. Global matched and other-active causal orientation signs used only this same calibration cohort, with both arms receiving the same single-sign budget. All eight arm-specific calibration signs across the four pairs were identified, and their assignments and orientation receipts were SHA-archived before any SMC-19 test states were generated.

Two **new** structurally noncanonical, exact-unique 64-puzzle Sudoku families use full-grid seed bases **202619030000** (30 holes) and **202619040000** (34 holes), each with a separate complete solution and visible input. They are disjoint from every SMC-13, 15, 16, 17 and 18 train/calibration/test puzzle and target. Eight goal tokens per puzzle are selected only from its visible missing-cell pattern. The two 64-puzzle test-family SHA-256 manifests are recorded in the machine-readable summary.

## Registered causal-refit outcome

At a goal with a strict positive reference SAE71 top-one atom \(i\), use the original calibration-frozen matched new SAE atom \(\pi(i)\). It must be naturally active in the new top-four dictionary, with another distinct active feature available. The same frozen teacher's clean FP32 stage-one goal state undergoes three independent single-token counterfactuals: removal of the original reference SAE atom's native physical decoder contribution, removal of the **refit matched** atom's native contribution, and a dose-matched removal along another **refit active** decoder direction. All edits continue the same teacher for three FP64 recurrence steps, with true-digit-minus-frozen-clean-rival margin loss. Separate calibration-only global signs orient the refit matched and other-active effect responses against the reference effect.

The preregistered primary score is the eligible-token mean **matched-minus-other-active signed causal concordance** with a 2,000-resample *64-puzzle-cluster* descriptive 95% interval; the score is a difference of two \([-1,1]\) concordances and can range from \(-2\) to \(+2\). Each pair must pass **both** fresh puzzle families: ≥24 naturally active eligible tokens across ≥20 distinct puzzles, identified calibration signs, 2,000 valid puzzle-cluster draws and a paired 95% lower confidence endpoint strictly above zero.

| Receiving independently trained SAE | Fresh 30-hole paired concordance [95% cluster interval] | Eligible goals / puzzles | Fresh 34-hole paired concordance [95% cluster interval] | Eligible goals / puzzles | Both-family gate |
|---|---|---:|---|---:|---|
| B501 SAE131 | +0.3681 [0.1796, 0.5525] | 163 / 61 | +0.1489 **[−0.0430, 0.3261]** | 188 / 63 | **FAIL** |
| B501 SAE167 | +0.4174 [0.2336, 0.6220] | 230 / 63 | +0.3750 [0.2244, 0.5297] | 224 / 64 | **PASS** |
| B701 SAE131 | +0.3478 [0.1463, 0.5481] | 161 / 61 | +0.4400 [0.2179, 0.6617] | 150 / 60 | **PASS** |
| B701 SAE167 | +0.1912 **[−0.0310, 0.4462]** | 136 / 61 | +0.1513 **[−0.1250, 0.4091]** | 119 / 56 | **FAIL** |

Both failed refit pairs met coverage/identified-sign requirements: **B501 SAE131 failed the 34-hole confidence bound**, and **B701 SAE167 failed the bound in both novel cohorts**. All raw observations and original frozen gates are retained. No retrospective seed preference, threshold change or cohort selection was made.

The receiving matched feature was naturally active on **23.2–44.9%** of the 512 visible-mask-selected goal tokens per cohort. The primary result is conditional on this activity subset. Missing matched activity was explicitly excluded rather than assigned a zero causal effect.

### Observational SAE alignment is stronger than causal consistency

The same new refit dictionaries showed high *heldout binary-support overlap* under the calibration-frozen matching: across the eight pair/family conditions, **matched overlap 0.345–0.417**, compared with **0.053–0.118** after an original visible-position/digit-conditioned calibration shuffle. The random feature-mapping control likewise showed lower support overlap (approximately 0.020–0.102). The coactivation result holds for **all four refit pairs**, including both pairs that failed the causal-transfer confidence criterion.

The outcome here is sharper than reconstruction or activation alignment alone: a calibration-matched sparse atom can overlap strongly at new Sudoku positions while its **native physical decoded causal contribution** remains refit-dependent. A secondary clean-correct→wrong prediction flip check makes this caution concrete. In the B501 SAE167 pair, the mapped refit feature's erasure caused **5.7–6.3%** of all eligible goals to move from clean-correct to wrong, while its equal-dose other-active control caused **8.7–10.7%**. The positive preregistered score concerns signed correspondence with a reference SAE feature; it cannot be promoted to superior necessity for Sudoku correctness or a shared intrinsic semantic atom.

The two receiving teachers themselves are exactly unchanged for every within-teacher comparison. This isolates changes introduced by SAE training/feature decomposition, although repeated SAE fits, the two training teachers and seeded puzzle generators are correlated rather than independent population replications.

## Exact orthogonal transport control

The separate engineering arm creates one deterministic dense 48×48 FP64 orthogonal matrix \(Q\) from seed **20261919**. It transports original B501 stage-one hidden states and SAE71 physical decoder contributions by \(h\mapsto hQ\), \(\delta\mapsto\delta Q\), and reuses the same unchanged recurrent teacher through the inverse-coordinate adapter \(v\mapsto vQ^\top\). This is an **identical model and SAE under a coordinate change**, never an independently refitted dictionary. On four new30 puzzle prefixes and 32 selected goal tokens it achieved:

| Transport invariant | Maximum absolute numerical discrepancy |
|---|---:|
| \(Q^\top Q-I\) | \(8.88\times10^{-16}\) |
| SAE feature code | \(5.33\times10^{-15}\) |
| Clean goal logits | \(1.07\times10^{-14}\) |
| Native-erasure intervened logits | \(1.11\times10^{-14}\) |

All are below the preregistered engineering threshold **2×10⁻⁹**. The result verifies exact-coordinate gauge invariance to floating-point precision, as predicted by companion OLM-19, and serves as a control for the independent SAE refit test.

## Source receipts and independent replay

- Original [SMC-19 three-stage hosted workflow #43810248525](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525), experiment code commit \`fc819a7ec273027aff20ff49db3b539b46b3687b\`.
- Four new **weights-only SAE checkpoints + training/validation selection and calibration**: [artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904); ZIP SHA-256 \`43630f8b9e6e7fbd07f8c668dc607f8b53c3a6b78cd47ce43aefe33bc005cb9e\`; frozen \`frozen.json\` SHA-256 \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`.
- [Full heldout \`full.json\` and independent \`audit.json\` artifact #14200232278](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14200232278); ZIP SHA-256 \`8fb7509fb7844456e433f3c4fde35963d8191666551b2b57d9b32c191cc253e8\`; source \`full.json\` SHA-256 \`49318e017a381079864c96fdcda47f6cc7ec81ceefc3fd4d7859687c70734f62\`; independent \`audit.json\` SHA-256 \`4f2eb9c146e0d15047138603b4038f79ba10fa42cab0b1b608dcaa135d9256e1\`.
- [Permanent machine-readable source-locked summary](SMC19_SUMMARY.json) preserves all four independent model/SAE fingerprints, eight registered paired causal effects and intervals, support nulls, activity coverage, gauge errors, cohort manifests, original source SHA checks and independent audit evidence.

The independent **stdlib-only** auditor SHA-checked all four archived refit weight files, original seven SMC-13C feature assignments and four new calibration mappings; recomputed eight calibration-only arm signs across the four pairs, per-puzzle native erasure eligibility/dose controls, 16,000 paired puzzle bootstrap draws across eight conditions, and the original strict scientific gate. The audit checked complete 64-puzzle fresh solution rosters, new manifest hashes, and the gauge transport error receipt. It **replays saved numerical observations** rather than independently training four new SAEs or recomputing the neural teacher's forward passes.

## Pure math and next discriminating step

The [MathCan OLM-19 companion](https://github.com/patrickdugan/MathCan/pull/105) proves exact orthogonal conjugacy for an arbitrary nonlinear recurrent teacher and finite physical SAE interventions; it also constructs **two ReLU SAEs that exactly reconstruct every input in a positive 2D cone and have a perfectly matched feature activation on every input, yet have opposite signed native erasure effects for the same unchanged clean teacher**. A sharp Lipschitz bound shows that sufficiently close *physical decoded contributions*, rather than coactivation labels alone, suffice for stable task-margin effects.

SMC-19 establishes partial operational feature stability across independently refitted dictionaries under a fixed trained teacher. A next campaign should register **physical decoded-contribution distance and task-gradient alignment as predictors** of which calibrated feature matches survive refitting, against hard heldout source/target controls, and separately test whether allowing a changed teacher capability level affects those predictors. The same model's high coactivation transfer and variable erasure responses already show why representation-level agreement needs a task-functional causal score.
