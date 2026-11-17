# NOR-41: original first source-frozen CAL scaling and input-family OOD results

**Registered scientific result: FAIL, 4/5 original gates PASS.** This document describes the **first original** 64-source score from [GitHub Actions run #44124795440](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44124795440), preserving the frozen thresholds and first-grade raw JSON exactly. The [source-only freeze report](FIRST_SOURCE_FREEZE.md), [first registration](REGISTRATION.md), and [pre-data amendment](AMENDMENT_01.md) document chronology and the CAL duplicate-support selection rule. The separate-process full source/CAL/EVAL replay is tracked as a separate CI job; its final completion should be checked independently of the scorer's source-integrity Boolean.

## Original immutable evidence

| Evidence | First exact bytes | SHA-256 |
| --- | ---: | --- |
| Public-only 64-source fixture, original source generation in run #44124006976, uploaded byte-identically in run #44124795440 | 9,618,649 | `4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d` |
| Complete first original CAL4096/1024-EVAL graded receipt, [artifact 14349983984](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44124795440/artifacts/14349983984) | 5,007,355 | `4efa6bdb1dd3656d0fbb6639065a8f2d934bd4a1acb952c128748cea9918de58` |
| Permanent deterministic [original source+grade ZIP](results/nor41_original_first_hosted.zip) | 3,388,660 | `0752c8c736ebe4460d02d655d7513662093d5c9dd2ea6e632a0ea76705469d12` |

The [immutable first-run JSON receipt](results/nor41_first_frozen_receipt.json) and read-only [original_archive_audit.py](original_archive_audit.py) pin the original source, grader bytes, SHA hashes, four CAL budgets and scientific FAIL gates. Original prospective registration commit: `87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f`; original pre-data attack-support selection amendment commit: `e5a95cad1ffdcd019f153cd09dd4d48b40848ef9`.

The first source-generation process produced the complete public-only 64-source fixture and printed its exact SHA, then stopped in a separate admission checker due to a stale inherited 48-count assertion. **No new-cohort CAL, EVAL, private target or grade ran in that stopped process.** The corrected source-first workflow required byte-for-byte reproduction of the first SHA, attested disjointness from all 324 earlier original sources, and uploaded that fixture before the first grading process began. The original source grammar, seeds, auditor/attacker code, test thresholds and challenge streams remained frozen. Every source's original public RRC25 mask had cardinality one.

## Exactly five prospective science gates

| Gate | Original registered result | Registered criterion | First scorer |
| --- | --- | --- | --- |
| 1. Degree-sensitive CAL-to-EVAL transfer | **D512 minus D1024 = −0.1491pp**; **5/16** replicate-block contrasts positive | ≥+0.1000pp and ≥11/16 positive blocks | **FAIL** |
| 2. Novel input-family TOP-curvature signal | **+3.5501pp** Global7 vs atom-random; **28/32** new-family sources positive; retains **21.8591%** of deterministic advantage | ≥+2pp, ≥24/32 positive, ≥20% retained | PASS |
| 3. CAL4096 attack transfer under independent EVAL | **+0.2013pp** reduction in Global7 detection vs CAL256; **45/64** sources improved | ≥+0.1000pp and ≥32/64 wins, CAL dominance | PASS |
| 4. Global7 robustness under CAL4096 attacks | Global7 minus deterministic strong self-play **+2.3063pp**, **64/64** positive; worst-four-strong-class Global7 minus original mixture **−0.5322pp**, **56/64** within −1pp; absolute Global7 worst-four-strong-class detection **3.0776%** | +1pp self-play and ≥48/64 positive; worst-class gap ≥−1pp and ≥48/64 within −1pp; absolute ≥2.0% | PASS |
| 5. Source-first, CAL brackets, native integer witness and replay contract | **TRUE** in original scorer, all 64 sources, original exact-check and first-frozen chronology flags | Exact source and certificate checks, original immutable evidence, independent replay | PASS as scored; independent job separately required |

The scientific **FAIL** is substantive. NOR40's 48-source exploratory gradient suggested stronger attack learning at future length 769 than 1537; NOR41 preregistered that scale direction. The new 64-source cohort produced the opposite mean direction. The finite 16-block interval for the difference includes zero, limiting any claim of a reliably reversed scaling law.

## The scale result and source-family shift

NOR41 has **16 independent replicate blocks**, each containing four source identities formed from D512/D1024 and the historical dense/new block-regime input families. The first 64-source fixture is completely disjoint from the earlier 324 original sources. All output witnesses use the unchanged RRC25 six-parameter native program grammar.

| Group | Sources | CAL256 minus CAL4096 Global7 heldout detection, percentage points |
| --- | ---: | ---: |
| D512, future length **769** | 32 | **+0.1267** |
| D1024, future length **1537** | 32 | **+0.2758** |
| Historical `wide_dense_small` input | 32 | +0.1515 |
| New `wide_regime_blocks` input | 32 | +0.2511 |
| Overall equal-source mean | 64 | **+0.2013** |

The 16 within-replicate degree contrasts have mean **−0.1491pp**, descriptive t95 interval **[−0.3482, +0.0500]pp**, and just **5/16** are positive. The interval's upper bound remains below the preregistered minimum effect of +0.1000pp. The overall CAL256→4096 heldout gain has 16-block descriptive t95 interval **[+0.1068,+0.2958]pp**, with 13/16 positive block means and 45/64 positive source differences.

The new 64-coefficient input family had **30.857%** nonzero query coefficient density, compared with 100% for the historical dense anchor. The four deterministic input regimes produced source-only mean nonzero rates of approximately 90.446%, 4.150%, 4.004%, and 24.829%. The frozen 64-source fixture contains 15 of the 16 native program classes across sources, and **every individual** source has a unique RRC25-admissible class. This is **input distribution OOD only**. It neither tests uncertainty over the true program grammar nor authenticates a target-output witness by itself.

## Equal-query four-budget attacker curves

All entries are equal-source means of heldout EVAL detection by each frozen auditor against the eight best CAL-only attacks constructed for that defender and calibration budget. Every audit still checks **k=16 exact target-output coefficients** and the attacked support radius is **r=4**. The four attacker budgets are nested prefixes of the same 4096-CAL schedule, with sixteen greedy starts, two CAL-only swaps and an exact C(16,4)=1,820 candidate-pool optimization at each budget.

| Auditor | CAL256 self-play | CAL512 self-play | CAL2048 self-play | CAL4096 self-play | Worst of four CAL4096-calibrated attacker classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-output uniform16 | 6.1136% | 5.9269% | 5.8788% | 5.8817% | **5.2767%** |
| Original mixture | 3.9051% | 3.9641% | 3.9444% | 3.9538% | 3.6098% |
| Deterministic curvature mixture | 1.5058% | 1.3353% | 1.0633% | 1.0837% | 1.0163% |
| **Global7 uniform-curvature reserve** | **3.5912%** | **3.5815%** | **3.4603%** | **3.3899%** | **3.0776%** |

The absolute CAL4096 strong self-play detection gap of Global7 above deterministic curvature is **+2.3063pp**, with 16/16 positive replicate-block means and descriptive 95% block interval **[+2.1957,+2.4168]pp**. Global7's sourcewise worst-four-calibrated-class detection is lower than the original mixture by **−0.5322pp** on average, with descriptive 95% interval **[−0.6303,−0.4340]pp**. Whole-output uniform16 retains the largest observed worst-class coverage among these four fixed defender laws.

## Exact curvature support detection

Within the new 32-source block-regime input family, the exact conditional Global7 TOP-curvature advantage over `atom_random_h50` is **+3.5501pp**, with **28/32** positive source differences. The deterministic curvature mixture has a **+16.2409pp** TOP advantage over atom-random in that same family, so the registered retained fraction is **21.8591%**. The historical dense-anchor source family shows a +4.1182pp Global7 TOP gain. Across all 64 sources the mean gain is **+3.8341pp**, with 16-block descriptive 95% interval **[+3.2746,+4.3937]pp**.

TOP is a fixed 2+2 high-curvature support sample inside/outside the original risk union and its exact conditional rational expectation under the frozen policy law. It is not the observed hit rate against the optimized radius-four malicious CAL attacker. These are different attack distributions.

## CAL optimization certificates and model limitations

Each of four trained auditors at each CAL budget has a 16-coordinate CAL-only candidate pool and enumerates **exactly 1,820 distinct radius-four supports** in that pool. The reported optimum is certified within the selected pool only. The fourth-smallest **full-domain singleton** weighted exposure gives a sound, usually loose lower bound on the unrestricted CAL optimum, as proved in [MathCan NOR40SingletonBracket](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR40SingletonBracket.lean). The best tested feasible support provides its upper bound. The first scorer's sourcewise CAL bracket and original greedy-eight dominance assertions all passed.

All numeric scores assume an independently authenticated future target coefficient witness and an audit draw hidden from the adversary. Public SHA challenge streams are deterministic scientific replay fixtures. The stronger attacker gains reflect finite optimization of a fixed precommitted four-coordinate change, rather than a proof of optimal adversarial support under all possible optimization budgets. A fully schedule-informed attacker remains a separate evasion scenario.

## Interpretation and next experiment

NOR41 **rejects its preregistered positive scale-direction criterion**, while the new input-family TOP mechanism and general CAL4096-to-EVAL attacker transfer survive. The different source grammar and larger CAL budget may explain the difference from NOR40; the source-block interval does not establish the mechanism of the inversion. Repeating NOR41's selected source groups or retuning its practical threshold would require a new prospectively registered study with fresh seeds.

A future scale analysis could explicitly register the interaction among output length, input family, CAL search budget and attack-set uniqueness. A more important generalization frontier is the independently registered **program-grammar OOD** proposal in [NOR-42 issue #229](https://github.com/MoralityLabAI/Compactification-Theory/issues/229). [NOR42's historical smoke results](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/nor42_paired_program_ood/RESULTS.md) and the [Lean prefix-identifiability theorem](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR42HiddenSuffix.lean) already demonstrate how a hidden late suffix can produce identical public outputs while changing future truth. They do not constitute a new prospective NOR42 experiment.

**Reproducibility status:** first original NOR41 grade SHA-locked and archived; independent second-implementation full source and 22×4 cross-play replay must be read separately. The source/five-gate scorer record alone cannot certify its own independent verification job.
