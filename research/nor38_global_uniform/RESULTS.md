# NOR-38: original scorer results and procedural integrity audit

**Original registered scoring output: 5/5 gated booleans TRUE.** The original first run [#44098045627](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44098045627) generated the 48-source fixture and scored all registered policies without modifying thresholds or selecting the treatment using these held-out data. The scorer's result is preserved as emitted. A separate audit of the preregistration sequence identifies a procedural breach affecting the claim of a fully untouched held-out cohort. Treat the first cohort's grade as **first-run evidence with a disclosed prospective-protocol deviation**, rather than a clean five-gate scientific replication.

## Original frozen provenance

- First prospective registration commit: \`37f9ed96b122ea16fb681dd5df638cbc4285d1e4\`, Nov 17, 2026.
- First unaltered public-only 48-source fixture: **9,319,507 bytes**, SHA-256 \`37fa2aeedeb5ab54bc1082ec2d50f8a084c4f4a7d6279acd5d122396d760e504\`.
- First original full JSON grade: **929,756 bytes**, SHA-256 \`9c69d663d4d1511e664e930c0f8ed42d8d6f59c4d10557323fb5ee54478c036d\`.
- Original 48 sources: replicates 60–67 × degree 512/1024 × original three NOR21 families; 180 historical SHA identities excluded.
- 256 CAL and 512 EVAL audit schedules per source, 8 radius-four greedy restarts per each of 7 auditor policies; 5 blind attack classes. Each auditor charges exactly 16 authentic target-output integer coefficient witnesses.
- Authenticated external true-target witness and actual deployment-hidden entropy are assumptions; SHA streams are deterministic public replay fixtures.
- Independent receipt replayer: \`research/nor38_global_uniform/receipt_audit.py\`, which separately reconstructs original native sources, 7 policy-aware greedy attack classes, 5 blind classes, component schedules, and complete held-out crossplay. Its completion state is reported separately in the CI run.

## Registered scorer results

| Frozen first-grade measure | Result |
| --- | --- |
| PRIMARY \`global7_mix\` own greedy self-play minus deterministic_mix own greedy self-play | **+1.98883 percentage points**, 48/48 positive |
| PRIMARY sourcewise worst of seven calibrated attacker classes vs original_mix sourcewise worst | **−0.28877 percentage points**, 46/48 sources no worse than −1pp |
| PRIMARY exact TOP-curvature conditional detection vs atom_random_h50 | **+3.21997 percentage points**, 44/48 positive |
| Deterministic_mix exact TOP advantage vs atom_random_h50 | +15.87231 percentage points |
| Retained deterministic TOP advantage | **20.28669%**, original minimum ≥20% |
| Per-trial original source, equal budget, exact witness, CAL/EVAL separation and native agreement | Evaluator recorded zero violations |
| Scientific scoring booleans | **5/5 TRUE**; original \`registered_primary_pass: true\` retained byte-for-byte |

The source-level worst-class gate requires a within-1pp result in at least 36/48 sources, equal-source mean gap ≥−1pp, and \`global7_mix\` worst-class mean ≥2.5%. All numerical thresholds passed. The exact TOP gain retained only **0.28669 percentage points of the ratio threshold's 20% minimum** beyond the registered boundary (20.28669% vs 20.0%); this is a narrow threshold crossing.

### Equal-budget auditor rates (48-source means)

Each auditor's self-play attacks were independently learned for it. The worst-class column takes each source's minimum over all seven calibrated attacker classes and then averages 48 source minima. TOP is an exact analytical two-support 2+2 conditional probability.

| Auditor | Own optimized self-play | Sourcewise worst class | Exact TOP support |
| --- | ---: | ---: | ---: |
| Pure uniform16 | **6.0038%** | **5.0593%** | 6.1048% |
| Original NOR31 mixture | 3.7869% | 3.3926% | 15.3591% |
| Deterministic curvature mixture | 1.5881% | 1.2972% | **35.0245%** |
| Stochastic50 mixture | 2.8748% | 2.5518% | 21.1814% |
| Atom-random hybrid50 | 3.2181% | 2.4633% | 19.1522% |
| **Global7 mixture** | **3.5769%** | **3.1039%** | **22.3721%** |
| Global8 mixture (sensitivity) | 3.8753% | 3.3338% | 20.5646% |

The 8/16 uniform-reserve sensitivity arm produces higher adversarial self-play and worst-class rates; its exact TOP advantage relative to atom-random is only +1.4125pp. That misses the primary 2pp curvature gain criterion if it were applied to this sensitivity arm. The primary 7/16 arm is the preregistered compromise.

Pure uniform16 remains highest for worst calibrated attacker coverage. The 7/16 reserve's exact unconditional fixed-support floor is **3.5357% at m=769** and **1.7952% at m=1537**, given ideal fresh independent private random draws. These are analytic policy floors, not required finite 512-schedule empirical lower limits. A realized-schedule-informed adversary still chooses unchecked coordinates.

## Source subgroup and uncertainty diagnostics (exploratory)

Source-paired descriptive normal-approximate intervals across eight replicate-block means (48 source rows, six per replicate) give:

| Source-level contrast | Mean difference | Approximate 95% replicate-block interval |
| --- | ---: | ---: |
| global7 vs deterministic own self-play | +1.9888pp | [+1.9204, +2.0573]pp |
| global7 vs original_mix worst-class | −0.2888pp | [−0.4665, −0.1111]pp |
| global7 vs atom-random exact TOP | +3.2200pp | [+2.1437, +4.2962]pp |

These intervals summarize the eight finite RRC25 replicate blocks and are not population coverage claims for unrelated programs or deployment. The 4/48 negative TOP cases all have family \`wide_sparse_small\`: replicate 64 at D1024, rep 65 at D512, and rep 66 at D512 and D1024. Two worst-class >1pp tolerance misses occur at replicate 67, D512, families \`wide_dense_small\` and \`wide_rational_short\`. Subgroup concentration matters for the next blinded source cohort.

## Prospective process-deviation finding: source preview before fixture freeze

The originally committed pre-data unit test at [first test commit f04e28a](https://github.com/MoralityLabAI/Compactification-Theory/commit/f04e28af30b200e3bf509eb28a4b197e69697e27) constructed \`new_source(60,512,FAMILIES[0],...)\` and called \`restore_native_target\` on it. This was after the NOR-38 prospective registration but **before** the full public-only 48-source fixture was frozen in [run #44098045627](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44098045627). The source generator is deterministic; that previewed source is a member of the original registered 48-source roster. The early test also exercised an ordinary CAL scheduling function on the same source.

The test was corrected in [commit e5d46ef](https://github.com/MoralityLabAI/Compactification-Theory/commit/e5d46ef7369e4cfb16f9188ef8a94a67ffc117b) to use archived **NOR-37 historical** sources, and later CI tests passed. This repair happened prior to the first complete NOR-38 fixture and grader. Nevertheless, the preregistration required the complete public-only fixture to be frozen before restoring any private native source program. The original grader's gate checks source/grade SHA parity and independent replay; they do not detect this earlier development-test information exposure.

**Procedural integrity assessment: FAIL for the strict first-frozen-source-before-private-truth condition, even if the original scorer's five booleans all show PASS.** The earlier preview contains no recorded defender-specific greedy calibration outcome or held-out EVAL result, and there is no evidence of numerical threshold retuning based on the preview. The strict chronology remains violated. Preserve the original grade and signatures without retroactively changing the frozen result or respecifying the primary gates.

A clean confirmatory follow-up must preregister a completely new source cohort with no source generator or native target restoration applied to its identities during pre-data unit tests. Test fixture code exclusively with archived historical sources or synthetic sentinel identities. Then archive the full public-only roster before any private target reconstruction, calibration or EVAL plans for the new cohort; rerun the seven-policy symmetric greedy comparison unchanged and separately independently replay every decision.

## Mathematical scope

[The merged MathCan NOR-38 proof](https://github.com/patrickdugan/MathCan/pull/135) gives the exact finite attack-incidence row count, weighted double counting and the global uniform reserve lower bound. The regular-game theorem's incidence double-count premise is explicit. An empirical eight-restart greedy attack is an approximate learned response, not a global minimax witness. Whole-output uniform16 is the exact unrestricted fixed-r-support minimax policy under independent, unannounced audit coins.

The complete original 12×7 attacker-defender cross-play is machine-readable in the SHA-locked graded JSON receipt and can be reconstructed in \`receipt_audit.py\`.
