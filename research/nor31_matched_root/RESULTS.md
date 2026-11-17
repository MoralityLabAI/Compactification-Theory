# NOR-31: first prospective, equal-budget independent-calibration versus disclosed-root result

**Registered outcome: PASS.** The exact pre-data registration is [REGISTRATION.md](REGISTRATION.md), first committed as dd0e9f60cdab90ed2b2fb482dedc7496f7f50870. The first initiated full study was [GitHub Actions #44032053843](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44032053843). All five CI jobs passed, including the independently implemented source/commitment/HMAC replay of **2,359,296 fixed-batch decisions**. An additional concurrent run [#44032085437](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44032085437) independently minted its own OS calibration and evaluation roots, completed all five jobs, and reproduced the positive direction. This report and the [first original SHA-256 receipt](results/nor31_first_original_receipt.json) use the earlier initiated #44032053843, preserving the first original data unchanged.

## Registered result on 24 new source identities

Every policy and attack condition used exactly **32 greedy attack restarts**, with the **same SHA256 tie seeds**, **512 training schedules**, one **fixed radius-four attack per evaluation batch**, and the exact same **512 evaluation audit schedules**. Each original precommit attack was trained against an independently sampled calibration root C and bound before root E existed. Each late attack was trained directly against E's already-realized schedules and given its own new commitment. The source-only public proposals and frozen NOR-24 prior were unchanged.

| Equal-source detection mean | Independent calibration C, precommitted | Actual evaluation root E leaked before attack choice | Difference |
| --- | ---: | ---: | ---: |
| **Hybrid_50: risk8 + uniform8** | **3.1125%** | **0.0895%** | **+3.0230 pp** |
| Uniform16 | 5.9252% | 1.0272% | +4.8981 pp |
| Static trained-risk16 | 0.0000% | 0.0000% | 0.0000 pp |

The predeclared primary gate required hybrid detection to decrease by at least **+1.5 percentage points** and show improvement on at least **20 of 24** independently seeded source identities. The first run showed **+3.0230204 pp and 24/24 positive source pairs**. All five registered gates passed, including commitments, root chronology, authentic exact witness decisions, independent schedule replay and per-schedule informed evasion.

The median source-level hybrid decrease was **+2.4933 pp**, with source differences between **+1.8188 pp and +5.3162 pp**. A descriptive source-block bootstrap with 10,000 replacement draws and frozen analysis seed 202611310117 gives a central 95% interval of **+2.5981 to +3.4693 pp**. This uncertainty analysis was performed after the registered first result and does not change its decision.

## Original individual-source primary paired rates

Each cell averages 32 fixed attack sets × 512 realized audit schedules on one public source. These 16,384 decisions are correlated repeated observations from that source.

| Replicate | Observation degree | Family | Independent-calibration detect | Leaked-root detect | Paired gain |
| ---: | ---: | --- | ---: | ---: | ---: |
| 22 | 512 | dense | 4.358% | 0.000% | +4.358 pp |
| 22 | 512 | sparse | 4.211% | 0.000% | +4.211 pp |
| 22 | 512 | rational | 5.316% | 0.000% | +5.316 pp |
| 22 | 1024 | dense | 2.002% | 0.000% | +2.002 pp |
| 22 | 1024 | sparse | 2.338% | 0.000% | +2.338 pp |
| 22 | 1024 | rational | 2.216% | 0.000% | +2.216 pp |
| 23 | 512 | dense | 4.309% | 0.781% | +3.528 pp |
| 23 | 512 | sparse | 3.461% | 0.000% | +3.461 pp |
| 23 | 512 | rational | 2.844% | 0.195% | +2.649 pp |
| 23 | 1024 | dense | 2.124% | 0.000% | +2.124 pp |
| 23 | 1024 | sparse | 1.819% | 0.000% | +1.819 pp |
| 23 | 1024 | rational | 2.069% | 0.000% | +2.069 pp |
| 24 | 512 | dense | 3.088% | 0.195% | +2.893 pp |
| 24 | 512 | sparse | 4.016% | 0.391% | +3.625 pp |
| 24 | 512 | rational | 4.767% | 0.195% | +4.572 pp |
| 24 | 1024 | dense | 2.173% | 0.000% | +2.173 pp |
| 24 | 1024 | sparse | 1.868% | 0.000% | +1.868 pp |
| 24 | 1024 | rational | 2.216% | 0.000% | +2.216 pp |
| 25 | 512 | dense | 4.559% | 0.391% | +4.169 pp |
| 25 | 512 | sparse | 4.102% | 0.000% | +4.102 pp |
| 25 | 512 | rational | 4.883% | 0.000% | +4.883 pp |
| 25 | 1024 | dense | 2.057% | 0.000% | +2.057 pp |
| 25 | 1024 | sparse | 2.032% | 0.000% | +2.032 pp |
| 25 | 1024 | rational | 1.874% | 0.000% | +1.874 pp |

The degree-512 source group (m=769 coefficients) showed a mean decrease of **+3.9805 pp**. Degree-1024 sources (m=1537) showed **+2.0655 pp**, with **zero** leaked-root hybrid detection across those twelve long-output source identities. Observed independent-calibration hybrid detection by output length was 4.1595% versus 2.0655%, near the fixed-blind ideal-random lower floors of approximately 4.1472% and 2.0785%, respectively. The theoretical floor is a probability under independently uniform hidden sampling and does not impose a deterministic minimum on finite observed frequencies.

## Why the controls matter

**Uniform16** retained a higher independent-calibration detection rate than hybrid_50 and also experienced an even larger root-leak reduction. Its observed 5.9252% to 1.0272% change demonstrates that this is an information boundary involving sampling randomness, rather than an effect restricted to the hybrid risk prior.

**Static trained-risk16** was entirely deterministic. The greedy attacker can find coordinates outside its fixed sixteen inspections under either root, so both conditions scored zero. Root disclosure adds no advantage for a schedule that was already public.

There were exactly **2,304** attacks in each condition, and **1,536 attack supports changed** between matched source/policy/restart pairs. The deterministic static arm accounts for all 768 unchanged supports, whereas hybrid and uniform arms each changed on every matched pair. The optimization used identical tie seeds and exactly 512 input schedules per arm. Root-aware attacks were trained **on the evaluation schedules themselves**, which creates intentional in-sample selection, so their detection results are not an estimate of effectiveness on a future secret root.

Every one of 4,608 salted source-bound preimages opened its own original commitment and all deliberately altered nonce/index openings failed. Attempts to use late attack preimages against the earlier commitments failed. The separate full-schedule-informed attacker selected four unchecked positions on each of **36,864** realized audit schedules, giving zero detection and a new commitment incompatible with the earlier one. Replaying both nonzero coefficient shifts (+1 and +4,294,049,777) changed no exact witness decisions. Every legitimate public prediction matched the genuine reconstructed RRC25 generating program, and all native public acceptance masks agreed.

## Independent-root concurrent confirmation

A second CI run on exactly the same 24 registered public source identities, with **different calibration and evaluation OS roots**, produced **24/24 positive hybrid source pairs** and **+2.7901 pp** equal-source detection reduction: independent calibration **2.9752%** versus leaked-root **0.1851%**. All five CI jobs completed successfully, including its independent 2,359,296-decision replay.

The two runs use **the same 24 source identities and model family**, so they are separate randomness realizations rather than 48 independent sources. The first-run data and primary pass threshold remain unchanged.

## Mathematical companion

[MathCan NOR-31 PR #122](https://github.com/patrickdugan/MathCan/pull/122) formalized the finite *matched-restart* inequality. For calibration root c, evaluation root e, restart t and batch loss L, a **globally optimal** root-aware attack satisfies

    Σ_c Σ_t Σ_e L(Aware(e,t),e) ≤ Σ_c Σ_t Σ_e L(Calibrated(c,t),e).

It proves a strict two-root example where any independent calibration strategy loses to an informed attacker, and a separate counterexample where a deliberately suboptimal root-aware response loses to a fixed attack. This is important because NOR-31's measured root-aware attacker is a **greedy approximation**, and the Lean theorem does not claim the heuristic finds global optima. The proof also inherits the abstract commitment-binding boundary without proving SHA collision resistance or HMAC security.


## Retrospective diagnostic: the coupon-collector explanation

This analysis was developed **after** the original result and is deliberately separate from NOR-31's registered primary criteria. The original frozen evaluation schedules and full outcomes were read unchanged from the eight-file SHA-locked ZIP by [coverage_diagnostic.py](coverage_diagnostic.py).

Across 512 audits, a root-aware attacker can choose four changed coordinates that were **never inspected in the entire batch** whenever at least four such positions remain. For a hybrid auditor with t=8 targeted positions and u=8 random reserve positions on each audit, an ideal independently sampled reserve on the N=m-8 remaining positions yields exactly

    E[never-inspected coordinates after n audits] =
        N * (1 - 8/N)^n.

This follows by summing N indicator variables, each remaining uninspected with probability (1-8/N)^n. For uniform16, replace N by m and 8 by 16. The expression is an expected count, rather than a guarantee about any particular seeded batch.

| Original output length | Auditor | Ideal expected never inspected /512 audits | Observed mean | Observed range | Sources with at least four untouched |
| ---: | --- | ---: | ---: | ---: | ---: |
| 769 | Hybrid_50 | 3.400 | **3.417** | 0–6 | **6/12** |
| 1537 | Hybrid_50 | 104.216 | **104.583** | 89–122 | **12/12** |
| 769 | Uniform16 | 0.016 | 0 | 0 | 0/12 |
| 1537 | Uniform16 | 7.241 | **6.250** | 2–11 | **9/12** |
| 769 | Static risk16 | 753 | 753 | 753 | 12/12 |
| 1537 | Static risk16 | 1521 | 1521 | 1521 | 12/12 |

For all 24 hybrid source identities, **zero leaked-root detection occurred exactly when at least four never-inspected coordinates existed**; the registered greedy algorithm always selected four such coordinates when available. This directly explains the 0% long-output hybrid detection in the first result. The [existing Lean NOR-29 batch-evasion iff theorem](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR29EarlyEntropyGame.lean) gives the finite logical boundary: a batch-fixed attack evades every audit precisely when none of its changed positions lies in any inspected set.

At output length 1,537, the expected number of never-inspected hybrid-reserve coordinates would fall below four after approximately **1,134 independent audits**, compared with NOR-31's 512. At 769 positions the analogous threshold is **497**. These expected-count thresholds are descriptive and do **not** provide high-probability audit guarantees. Under ideal independent uniform sampling, Markov's inequality gives Pr(at least four never inspected) ≤ E[uninspected]/4. Reaching the **5% upper bound** from that argument would require at least **1,705** hybrid audits at m=1537 or **781** at m=769. Even after every coordinate has been sampled once, a four-coordinate attacker may still reduce the *fraction* of audit batches that detect its changes by choosing rarely inspected positions.

This diagnostic shows how finite batch coverage amplifies early schedule disclosure in a high-dimensional output space. It identifies **audit count relative to output length** as a strong prospective NOR-32 control. It neither changes NOR-31's data or PASS/FAIL decision nor certifies HMAC cryptographic security.

## Immutable original first-run data and reproduction

The [source-controlled eight-file original evidence archive](results/nor31_original_hosted.zip) preserves **31,714,166 original raw bytes** in a 3,705,746-byte ZIP. Its members are SHA-pinned by [results/nor31_first_original_receipt.json](results/nor31_first_original_receipt.json) and verified by a separate [read-only archive auditor](original_archive_audit.py) in CI.

| Original first-run file | Bytes | SHA-256 |
| --- | ---: | --- |
| Public-only fixture | 4,903,051 | 534c8729c12c542eebeb9f7faafb2dc8e1f118acff2990ced1198d11ec4e5802 |
| Calibration OS-root and 24×3×512 schedules | 11,085,703 | 5682975ac484298f7bf86431c42d2efebbeb4afb3d836c4497b22b158bdd7ab5 |
| 2,304 calibrated precommit openings | 961,146 | 0c42be81044caf2fdd92733932da37bf8c9532cc271433818fabfcd002a46cb3 |
| Public precommit manifest | 575,889 | a5d0c7b4ca88792d9f744aab81561dca958487e297ba2fa1345acda61ef3d4a2 |
| Evaluation OS-root and 24×3×512 schedules | 11,085,804 | a9d62bd119d3b22293e94b137be472491745165fe198a8d2e25e636e1621fb5f |
| 2,304 postroot openings | 958,920 | a3c9d76101a85c384bb5715e16ec30ea2c52dda76d30c2695036ed3adcbd6847 |
| Public postroot manifest | 575,952 | 401becc1e480e5c1d380c591d72290893b40a6edc052558e82f6540551cf904f |
| Full first-run exact outcome and source-level rows | 1,567,701 | c51ad4cf053fd4d44a7e36f47357b49b58f89f8c4fdfcfb62aecd773dd8893f5 |

The independent auditor reads the original precommit source, replays its native public-mask computation, restores the original generator only after the attack choices are frozen, reconstructs both HMAC-SHA256 root-derived schedules with a separately implemented sparse Fisher-Yates permutation, chooses both groups' 4-coordinate attacks using an independently implemented schedule-ID-set greedy optimizer, and verifies each of the **2,359,296 exact detected/undetected decision bits** and every source-level primary gate. It does not import the newly implemented NOR-31 sampler, optimizer, grader or source generator.

The experiment's OS roots and commitment nonces are released **after the completed commitment stages** for reproducibility. Their origin and the GitHub Actions stage chronology are operational observations. SHA256 binding, HMAC PRF pseudorandomness, OS entropy quality, CI runner integrity, and the authentic exact target-coefficient witness remain exogenous assumptions. The experiment measures a controlled synthetic information leak and must not be extrapolated to a deployed secure verifier without those assumptions.
