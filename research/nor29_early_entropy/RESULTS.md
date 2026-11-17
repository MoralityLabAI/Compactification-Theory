# NOR-29: first prospective results under early disclosure of audit entropy

**Registered scientific outcome: PASS**, following the frozen first-commit [REGISTRATION.md](REGISTRATION.md) at 2447248fa8c192991da695ed06af2d17d642dc5f. First original source-locked experiment and independent replay: [GitHub Actions #44025451594](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44025451594). All five original CI jobs passed, including a separate-process replayer of exactly **2,359,296** fixed-batch attack/policy/challenge decisions and **36,864** fully informed per-schedule controls. A second independently rooted CI run, [#44025482938](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44025482938), also passed. The original byte receipts below belong solely to the earlier-initiated #44025451594.

## Registered primary result

The 24 public source identities were fresh replicates 12–15, at 512/1024 observation degrees and all three NOR-21 RRC25 source families. Each source had **32 independently selected, randomly salted blind attack sets committed before fresh OS root generation**. After seeing the complete 512-schedule batch from that root, a separately designated adversary constructed 32 new greedy batch-fixed attacks per source/auditor, each with four altered coordinates and its own new SHA256 commitment. Both classes were evaluated on the very same 512 realized schedules.

| Audit policy, k=16 | Original blind precommit | Post-root batch-fixed attack | Detection decrease |
| --- | ---: | ---: | ---: |
| **Hybrid_50, risk8 plus uniform8** | **5.4871%** | **0.1673%** | **5.3197 pp** |
| Uniform16 | 6.0626% | 1.0986% | 4.9639 pp |
| Static trained-risk16 | 6.1198% | 0.0000% | 6.1198 pp |

Hybrid blind-minus-root-aware detection was positive in **24/24 independent source pairs**, and the equal-source mean difference was **+0.053197224934895836** absolute, exceeding the frozen >=20/24 and +0.010 gates. All five registered requirements **PASSED**. The source-paired median difference was **4.5837 pp**. A retrospective source-block percentile bootstrap with 10,000 resamples (seed 202611290111) gave a descriptive 95% interval of approximately **3.97 to 6.96 pp**. Audit/attack repeated draws inside a source were not counted as independent sources.

Degree-group means:

| Output length / observation degree | Hybrid blind precommit | Hybrid root-aware batch |
| --- | ---: | ---: |
| 769 / D=512 (12 sources) | 7.6111% | 0.3347% |
| 1537 / D=1024 (12 sources) | 3.3630% | 0.0000% |

For every larger-output source, the root-informed batch attacker found four altered coefficients that were never sampled in the 512 hybrid audits, so its fixed batch attack escaped the entire set. At m=769, some optimized fixed attack sets still intersected an audit schedule, giving nonzero 0.17% overall detection. The root-aware greedy algorithm is an approximate best response; it has no global optimum guarantee.

## Source-level paired evidence, k=16, radius-four

Each percentage averages 32 altered sets times 512 audit challenges within the same source identity. The final column counts the number of distinct hybrid late-root attack choices among the 32 independently tie-randomized attempts.

| Rep | D | Family | Blind hybrid | Root-aware hybrid | Paired decrease | Distinct late /32 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 12 | 512 | dense | 4.169% | 0.586% | +3.583 pp | 31 |
| 12 | 512 | sparse | 4.193% | 0.195% | +3.998 pp | 16 |
| 12 | 512 | rational | 6.683% | 0.195% | +6.488 pp | 17 |
| 12 | 1024 | dense | 8.240% | 0% | +8.240 pp | 32 |
| 12 | 1024 | sparse | 2.063% | 0% | +2.063 pp | 32 |
| 12 | 1024 | rational | 5.261% | 0% | +5.261 pp | 32 |
| 13 | 512 | dense | 4.108% | 0.500% | +3.607 pp | 30 |
| 13 | 512 | sparse | 7.367% | 0.391% | +6.976 pp | 30 |
| 13 | 512 | rational | 7.300% | 0% | +7.300 pp | 5 |
| 13 | 1024 | dense | 2.008% | 0% | +2.008 pp | 32 |
| 13 | 1024 | sparse | 5.170% | 0% | +5.170 pp | 32 |
| 13 | 1024 | rational | 2.136% | 0% | +2.136 pp | 32 |
| 14 | 512 | dense | 6.970% | 0% | +6.970 pp | 5 |
| 14 | 512 | sparse | 19.196% | 0.391% | +18.805 pp | 30 |
| 14 | 512 | rational | 4.059% | 0.586% | +3.473 pp | 31 |
| 14 | 1024 | dense | 1.953% | 0% | +1.953 pp | 32 |
| 14 | 1024 | sparse | 1.971% | 0% | +1.971 pp | 32 |
| 14 | 1024 | rational | 2.161% | 0% | +2.161 pp | 32 |
| 15 | 512 | dense | 10.059% | 0.586% | +9.473 pp | 31 |
| 15 | 512 | sparse | 10.217% | 0.586% | +9.631 pp | 32 |
| 15 | 512 | rational | 7.013% | 0% | +7.013 pp | 5 |
| 15 | 1024 | dense | 2.081% | 0% | +2.081 pp | 32 |
| 15 | 1024 | sparse | 5.170% | 0% | +5.170 pp | 32 |
| 15 | 1024 | rational | 2.142% | 0% | +2.142 pp | 32 |

The original random precommit arm occasionally samples risk-targeted coordinates, which creates large detection jumps on an entire fixed attack set. The large +18.805 pp source result reflects this fixed-attack sampling variability. The median, stratified degree means and all 24 paired directions are reported alongside the mean.

## Commitment chronology and informed controls

PREP emitted **768 pre-root salted SHA256 commitments**, and all verified against their opening preimages. Deliberately changing either the nonce or one altered coordinate failed to open the original. The subsequent CHALLENGE job sampled one fresh 256-bit OS root using secrets.token_bytes(32) after the PREP artifact had been finalized, then domain-separated that root through HMAC-SHA256 before generating 512 unbiased without-replacement audits per source/policy. The entire schedule list was SHA-frozen before the late-root attacker read it.

The compromised attacker created **2,304 NEW post-root commitments**, one per source/auditor/attack replica. None of the new attacks could open the prior blind commitment. The exact authenticated witness comparison was invariant under nonzero shifts +1 and +4294049777 across all 2,359,296 decisions. Honest unaltered targets passed and original RRC25 native acceptance masks were preserved.

For a separate stronger adversary allowed to choose a new four-coordinate set after each individual schedule, all **36,864** attacks evaded inspection. Each late replacement required a new commitment and failed original binding. This control has stronger editing freedom than the fixed-batch early-root attacker.

The prior NOR-25 blind hypergeometric floor is approximately 4.147% for m=769 and 2.079% for m=1537. It assumes altered coordinates are fixed independently of the secret uniform audit draws. The root-aware arm intentionally chooses its altered set with knowledge of those same draws, so the floor does not apply.

## Critical causal limitation

**The registered blind arm uses randomly chosen attack positions, while the early-root arm uses policy-aware greedy optimization on the realized schedule batch.** The measured 5.32 pp difference contains an advantage from knowing the public audit distribution as well as the additional advantage from knowing the actual root. It cannot by itself identify the incremental effect of leaking entropy.

The static ranking control illustrates the issue: its inspected positions are already public and deterministic, so an attacker can evade it without needing the root. The 6.12 pp static comparison is therefore a timing-and-optimization stressor, not evidence that static detection depends on root secrecy.

### Executed retrospective matched-budget diagnostic

The posthoc [matched-control script](retrospective_matched_control.py) has completed successfully and its **exact 29,223-byte JSON result** is source-controlled at [results/nor29_retrospective_matched_control.json](results/nor29_retrospective_matched_control.json), SHA-256 db2ec77170067868b81fd2a69576f8f0e7d7889ec3874b802fbd2725a4a21aa2. The original source, challenge and outcome SHA-256 are verified against the registered NOR-29 first-run archive. The calculation was designed and executed **after the registered first outcome**, and has no authority to change the registered PASS.

For each source and auditor, this control constructs the same 32 greedy attack replicas and randomized tie orders using exactly **512** simulated audit schedules. The pre-root attacker sees a **separate, independently derived public calibration root**, not the actual OS-rooted evaluation schedules. The exposed-root attacker retains access to the actual evaluation batch. Both choose one fixed four-coordinate set per 512-schedule batch and use the same greedy algorithm.

| Audit policy | 512-schedule independent calibration | 512-schedule actual leaked root | Added detection reduction |
| --- | ---: | ---: | ---: |
| **Hybrid_50** | **3.3033%** | **0.1673%** | **3.1359 pp** |
| Uniform16 | 6.1867% | 1.0986% | 5.0880 pp |
| Static trained-risk16 | 0% | 0% | 0 pp |

Hybrid and uniform each showed a positive matched-calibration-minus-root-aware difference on **24/24 sources**. The hybrid paired median was **2.5848 pp**, and a descriptive source-block percentile bootstrap (10,000 resamples, seed 202611290121) gave approximately **2.705 to 3.595 pp** for the source-mean difference. The static control remained zero in both regimes because its schedule is public and deterministic.

This comparison holds fixed the number of greedy attack candidates, tie streams, source inputs, policy knowledge, calibration batch size and evaluation trial count. Its remaining major difference is knowledge of the **actual** realized schedule batch. The diagnostic therefore provides better evidence of an incremental early-root disclosure effect, subject to posthoc analysis and correlation among source-world experiments. A fresh prospective study using this matched comparator as a registered primary arm remains necessary for confirmatory causal attribution.

The original preregistered random-attack baseline of 5.4871% for hybrid remains unchanged, and the matched-calibration comparator of 3.3033% is a **distinct retrospective control**.

## Permanent first original evidence and replay

All seven original raw files are preserved in [results/nor29_original_hosted.zip](results/nor29_original_hosted.zip), a **3,319,719-byte source-controlled archive**, and rechecked against the immutable [first original SHA receipt](results/nor29_first_original_receipt.json) by [the read-only archive auditor](original_archive_audit.py). The original independently replayed run is Actions #44025451594 with artifacts 14301869762, 14302189068 and 14302398666.

| Original raw file | Bytes | SHA-256 |
| --- | ---: | --- |
| Public-only 24-source fixture | 5,786,533 | 27a38ace5359329555b3a42696308dc5c4d7cd2ed16eced7265ec7b7fba041df |
| Sealed 768 early blind attacks | 258,995 | 7ed26a2aaab2193ed9f6ba772f87b9f34ec68a465a6c8352b2f6e5ddb6c9ffa5 |
| Public early commitment manifest | 198,231 | 00802911f45909b92a36e98f040961909e5428027d5f572114b1f71208b5dee1 |
| Frozen original root and audit schedules | 11,081,083 | ca62e44967c796b5445f5d1e592da7206f5fe237d26dbf55ff3273017bd93410 |
| Sealed 2,304 late root-aware attacks | 935,751 | 5b51046b08b27eca398b20ab09fbe4546228d6323ca38b70b7ef46f6bf23f5a5 |
| Public late commitment manifest | 575,748 | ce83db8b133d840c088ec214bc539024dc76b0d8421c0329491c070d04fe7703 |
| Complete first outcome | 1,586,040 | 9d13345da6c8e1292d06b2c25a40459f7b2f33251399a56b3da425073dd6832a |

First original root commitment: 1a4050013073d3cc34ca7d2b5be08d60782ccaa5a4054aa8f0015a8ab37f1c2f.

The independent replay imports **no NOR-29 evaluator, sampler or greedy attacker**. It independently reconstructs source-native program acceptance, SHA256 openings, all HMAC schedules through a sparse swap-map sampler, batch attack selection using trial-ID sets, and every exact detection bitset. It passed all 2,359,296 original fixed-batch decisions, 36,864 informed evasion controls and the registered source-level verdict.

**External dependencies:** trust in the GitHub Actions job chronology and original artifacts, SHA256 binding, HMAC PRF properties, entropy-root secrecy until leak, and a genuinely authenticated exact private target witness. The first-run root is publicly disclosed only after the committed experiment. The finite synthetic RRC25 setting provides a controlled causal benchmark rather than a deployed protocol-security certificate.

The companion [MathCan PR #119](https://github.com/patrickdugan/MathCan/pull/119) kernel-proves the finite root-aware expected-loss inequality, a strict two-root counterexample and the exact batch-fixed evasion condition, conditional on abstract binding assumptions.
