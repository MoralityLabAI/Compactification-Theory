# SMC-23: Target-native SAE erasure does not establish beneficial feature necessity in the independent teachers

**Executed 17 November 2026. Registered program verdict: FAIL (0/4 B-teacher SAE pairs, both families).**

The original protocol was committed before model data at commit 1f4c602. The [complete source-locked Actions run #44097734385](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44097734385) finished calibration, fresh test and an independent standard-library audit in separate jobs, all green. The scientific gate failed in every B501/B701 teacher/SAE pair. This is a full-powered negative result under the registered thresholds, **not** an engineering/source or sample-count failure.

## Scientific motivation and relation to earlier work

[SMC-14](SMC14_RESULTS.md) found that source-A SAE directions mapped through calibration-only coactivation can predict similar signed native correct-digit margin responses in two independently trained teachers, despite the mapped atom being naturally active in only about 8.6–16% of those goal tokens. The later [SMC-15](SMC15_RESULTS.md) identified rare low-condition-number native output frames, with a preregistered low-sigma group underpowered in all ten teacher/SAE/family cells. Earlier unmerged, separately registered [SMC-16 #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177) found 3/4 positive source-matched selective-kill comparisons against a random tangent sham; the harder [SMC-17 #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179) retained 1/4 using stronger inactive-preactivation controls.

SMC-23 explicitly chooses the **target teacher's own top-active SAE coordinate**. Source-A feature matching is audited for provenance without influencing eligibility. The active-feature edit is the minimum-norm physical subtraction reducing that encoder coordinate to -0.02 preactivation while preserving the other three originally active encodings. The matched sham is selected by inactive-preactivation discrepancy from 33 outcome-blind tangent candidates, all with identical input L2 dose. This tests a different operational hypothesis from SMC-16/17 and cannot be used to revise their frozen results.

## Frozen sources, eligibility and output

The native model cohort is unchanged: source pretrained A (44,001 parameters) and independent teachers B501/B701, with five total 48→96 Top4 SAE fits (A SAE71; B501 and B701 each SAE71/89). Source checkpoint, original SAE, independent-teacher training, SMC-13C source-conditioned correspondence file hashes and four matched/clue-shuffled mapping replays passed. The **calibration-only** 64-puzzle job finished and uploaded its artifact before the separate **fresh-test** job started.

Two new 64-puzzle, exactly unique, noncanonical Sudoku cohorts were generated using the registered seed bases **202633030000** (30 holes) and **202633040000** (34 holes). Their 128 full solutions and puzzle inputs are distinct from the 768 enumerated prior SMC-13, SMC-15 and SMC-16 complete targets. Each of five model/SAE components contributes eight input-only selected missing-cell goals per puzzle, for **5,120 preregistered candidate goal interventions**. All full native forwards used the source FP32 stage-one state followed by exactly three FP64 recurrent steps. Original top-four encoder-cell preservation and task-blind controls chose admitted edits before consulting Sudoku labels.

On the complete evaluation, **2,990/5,120 candidate goals** passed the strict pre-output selective-kill and matched-sham admission. B501/B701 accounted for **2,096/4,096** eligible goal measurements (51.2%); A for 894/1,024 (87.3%). The remaining abstentions arise from target-recode failure or the absence of a recode-stable hard sham; every reason appears in the raw case receipts.

## Registered primary result

The primary per-goal difference is the *true-digit minus strongest clean rival* margin under the hard tangent sham minus that under the exact feature kill:

\[
D = [m_{\rm clean}-m_{\rm kill}] - [m_{\rm clean}-m_{\rm sham}]
  = m_{\rm sham}-m_{\rm kill}.
\]

Positive D means that selectively erasing the native SAE coordinate damages the model's true-digit margin more than the matched control. Each SAE/cohort cell separately required at least 60 admitted goals from 30 puzzles, 2,000 puzzle-cluster bootstrap draws, a **Bonferroni-corrected 99.375% lower bound strictly above +0.02 logits** and more clean-correct-to-wrong transitions under kill than sham. Both new families had to pass for the same SAE; program success also required one successful SAE per independent teacher.

All eight B teacher/SAE/cohort cells passed the coverage requirement. Every B cell failed the corrected margin gate and had fewer correct→wrong flips under kill than sham.

| Teacher/SAE | Fresh 30-hole: eligible / puzzles | Mean D, 95% puzzle CI | Kill vs sham flips | Fresh 34-hole: eligible / puzzles | Mean D, 95% puzzle CI | Kill vs sham flips |
|---|---|---|---|---|---|---|
| B501 SAE71 | 213 / 63 | +0.0084 [−0.2677, +0.2750] | 12 / 26 | 208 / 63 | −0.1914 [−0.4049, +0.0188] | 20 / 24 |
| B501 SAE89 | 251 / 63 | +0.0503 [−0.1618, +0.2592] | 13 / 22 | 231 / 64 | −0.1290 [−0.3567, +0.1065] | 13 / 25 |
| B701 SAE71 | 307 / 64 | +0.0994 [−0.1489, +0.3376] | 17 / 26 | 319 / 64 | −0.2329 [−0.4601, −0.0117] | 18 / 39 |
| B701 SAE89 | 288 / 64 | −0.0881 [−0.3267, +0.1524] | 9 / 19 | 279 / 63 | −0.1083 [−0.3435, +0.1182] | 9 / 20 |

The strict **99.375% adjusted interval** in every B cell includes zero or negative effects; its lower endpoint ranges from approximately −0.230 to −0.546 for the four 30/34 configurations and both SAE seeds. Among all B cells pooled descriptively by eligible goal, the mean D is **−0.07373 logits**, with shared teachers, SAE fits, and puzzles making a naive pooled independent-error model inappropriate.

The one B701 SAE71 fresh34 cell has a 95% *negative* interval [−0.4601, −0.0117], meaning the target kill tended to damage the correct margin **less** than its matched sham there. This single descriptive effect is outside the registered positive gate and does not establish a stable reverse mechanism under the familywise uncertainty accounting.

## Pretrained source A capability comparator

The same code, target-native feature choice and sham construction applied to the more capable original pretrained teacher A SAE71.

| A SAE71 | Admitted / puzzles | Mean D [95% puzzle-cluster interval] | Kill / sham clean-correct→wrong |
|---|---:|---:|---:|
| Fresh 30-hole | 445 / 64 | **+2.1304 [1.8572, 2.4029]** | 97 / 10 |
| Fresh 34-hole | 449 / 64 | **+1.9038 [1.5018, 2.3069]** | 103 / 17 |

This comparison establishes that the source-trained SAE and teacher can produce large task-margin and answer-flip effects under the exact intervention and sham harness. A differs from B in task capability, checkpoint training and physical intervention dose: mean admitted dose about **4.26/4.19** for A versus **3.09–3.49** across B cells. These comparisons therefore do not isolate one mechanism variable and are descriptive, not randomized between teachers. The strong A results do not override the B program failure.

## Pre-registered calibration-beneficial subset

Calibration independently selected at most eight feature IDs for each SAE, requiring at least four admitted calibration goals from three different puzzles and a positive mean selective-kill effect >0.02. Exactly eight IDs were locked for each of the five SAEs before the fresh run. These lists are **secondary**; they did not alter the primary all-active denominator.

Across the B models, the secondary selected-feature mean effects on fresh 30/34 cohorts were:

| Target SAE | Fresh30 admitted, mean D | Fresh34 admitted, mean D |
|---|---|---|
| B501 SAE71 | 61, +0.0953 | 56, −0.4338 |
| B501 SAE89 | 66, −0.0622 | 58, −0.0848 |
| B701 SAE71 | 83, +0.3928 | 86, −0.2338 |
| B701 SAE89 | 92, −0.2929 | 86, −0.2191 |

No calibrated selected-feature subset showed a consistently positive mean across both fresh families. The chosen top-eight IDs, the original calibration effects, puzzle-level variance and admission receipts are archived. This supports the failure of calibration-selected feature usefulness to generalize under the present intervention protocol. A’s calibration-selected subset retained positive means of +3.1209 and +2.5907, respectively, across the two novel families.

## Causal interpretation and limitations

Selective encoding is an operational constraint within one frozen SAE's original top-four affine cell. The intervention is an **actual 48D physical hidden-state edit** followed by a full native three-block recurrence. The 33-candidate sham is chosen by input geometry alone and preserves all four originally active encoder values, while the target edit suppresses the top coordinate and preserves the three others. The 92 inactive encoder covectors generally span all 48 hidden-state dimensions, so target and sham can still have irreducible differences outside the active cell. Mean inactive-preactivation mismatch on the admitted goals was about **3.60–3.99** for B and **4.10–4.22** for A, in the protocol's physical affine units.

The negative B results constrain a strong claim that the teacher's highest-active SAE coordinate is uniformly *beneficial* for solving its Sudoku task. They leave open alternative notions of usefulness, such as informative prediction, a harmful feature, redundant distributed encoding, nonlinear latent manifolds, and local causal response directions that do not optimize the true-digit margin. SMC-14's source-matched directional transfer and the older SMC-16/17 source-conditioned eligibility criteria address separate questions and retain their historical outcomes. No experiment here measures endogenous inference-time semantic identity or global dynamical quotient.

A precise next direction is capability-conditioned target-native erasure across several **capability-matched independently trained** teachers, retaining the current 33-candidate null and a new calibration-selected feature criterion with independent examples. The two B teachers are materially weaker than source A and thus restrict mechanistic generalization. A second follow-up could compare the direct task-gradient predictor of SMC-20 to feature-selective erasure across these same task regimes, with a fresh prospective cohort rather than reusing this test.

## Source receipts and reproducibility

- Protocol: [SMC23_PROTOCOL.md](SMC23_PROTOCOL.md), initially committed in 1f4c602 before any SMC-23 measurement.
- Exact source-matched native implementation: [smc23_native_first.py](smc23_native_first.py).
- Six source-independent CPU tests: [test_smc23_native_first.py](test_smc23_native_first.py).
- Independent standard-library auditor: [audit_smc23.py](audit_smc23.py). The hosted audit replays chronology/source hashes, all 5,120 candidate goal statuses, every admitted margin identity, target-native selections, original selected goal positions, 2,000 puzzle-cluster bootstrap draws per comparison and all eight adjusted gates. It checks saved numerical receipts, not independent model forward passes.
- [Hosted three-job run #44097734385](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44097734385), successful in all three stages.
- [Calibration-only archive #14334651704](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44097734385/artifacts/14334651704): ZIP SHA-256 3cfed0fd370a6613070913a8e0a4b8156c93cc2c2f3897c33173e812ba47f294, contained calibration.json SHA-256 93c0f30d5cc3cb8a966e8f75be35a0d30ac28bf81768a2af7435e7d69e10df38.
- [Fresh 128-puzzle archive #14334612445](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44097734385/artifacts/14334612445): ZIP SHA-256 f13029c55b556efcb54dccf5fe7e987164db0b654a380cff686c9df636103040, contained full.json SHA-256 dd1e1339060e398e14b63fddc1b90d85f1871b9e15d6f3c0672b4dff2288149a.
- [Independent source-receipt audit #14334128910](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44097734385/artifacts/14334128910) uploaded and the auditor printed SMC23_INDEPENDENT_AUDIT_PASS, program_success false.
- New fresh roster SHA-256: novel30 2f7c199c332eb4a8e4e551e9e084ebac16abb8dc1e26b99552b339a6e7b3d0c7; novel34 9c48469a93ef4286295592d4aeca89c04311f12adab453a24d564f0aff51ce43. The original 64 calibration source puzzle roster hash, full test puzzle inputs/targets and per-goal numerical effects are available in the corresponding JSON artifacts.

The original ARG-03-N 168-case sealed test set and every earlier protocol remain unchanged.
