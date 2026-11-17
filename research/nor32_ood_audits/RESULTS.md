# NOR-32: six unseen attack-support mechanisms, first prospective results

**Registered scientific verdict: FAIL.** The preregistered blind OOD average gate failed, while all other four original gates passed in the first hosted result. The frozen experiment [REGISTRATION.md](REGISTRATION.md) is the first NOR-32 commit **07f0f912**, made before source/attack/schedule data. This report is retrospective narration of the first prospective test and does not change any thresholds, sample sizes or support families.

## Locked source and audit design

The fixed NOR-31 six-policy mixture [2,0,1,1,4,8]/16 from merged PR #200 and its original SHA-locked evidence was reused with **zero new fitting**. There were 24 new RRC25 source-distinct public-only inputs at replicates 32/33/34/35, demonstration degree 512/1024 and each of the original three wide source families. All fresh complete-public hashes were disjoint from the **36** historical NOR-26, NOR-30 and NOR-31 source identities.

The six new support generators, each holding exactly 96 output-coordinate positions, were **central96, stride17_96, low_abs96, curvature96, anti_prior96, hash96**. Their *generation mechanisms* were absent from historical training, although their actual coordinate sets may intersect old risk supports. Every support contributed eight fixed random radius-four attacks from its 96 coordinates, drawn before new audit challenge schedules. Three further policy-aware greedy cohorts were trained against 256 source/calibration schedules for the mixture, hybrid_50 and pure uniform16, with four restarts each. Those attacks were fixed before 512 separately salted heldout audit schedules per source and each of six auditors. Every policy used exactly **k=16** authentic target coefficient queries per trial. Old tail/magnitude/lag/uniform template controls were kept separate.

## First registered outcome

The first completed evaluation [GitHub Actions #44042332472](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44042332472) produced source-only fixture **3,892,735 bytes**, SHA-256
\`129dba8ef9b1ae64edf58f9b71d6de761c3f4ca4fe5be4224e2495e3353518b7\`, and first full 24-source outcome **1,142,260 bytes**, SHA-256
\`4b475c62c7289574c9ff74b67e2634f9094b9c60f4d296269588cbe591bed0e9\`.

The complete outcome passed exact source, public native mask, in-family true target, and live exact-value sampled-oracle checks. It **failed the registered primary gate** despite four other individually passing gates.

| Equal-source mean; k=16, r=4 | Frozen NOR-31 mix | Hybrid_50 | Uniform16 |
| --- | ---: | ---: | ---: |
| **Six new OOD families, blind random attacks** | 7.3713% | **7.8996%** | 6.0591% |
| Worst OOD blind support per source | **3.8397%** | 2.8249% | source-dependent |
| Policy-calibrated OOD attacks against each policy's own optimizer | **5.0852%** | 4.5675% | recorded in raw full result |
| Original 4:3:2:1 weighted template controls | 17.7387% | **22.4973%** | recorded in raw full result |

Registered source-paired blind OOD comparison: **4/24 mixture source wins**, versus required at least **18/24**; mean difference **−0.5283 percentage points** versus required at least **+0.2 points**. This is the defining registered FAIL.

Other preregistered gates passed: the mixture had higher minimum OOD-family detection on **24/24** sources; its blind OOD overall mean was 121.7% of pure uniform's equal-source mean, above the 60% required floor; mixture-aware calibrated self-play won **19/24** source pairs with mean **+0.5177 percentage points** over hybrid_50's own calibrated attackers; and old-template detection exceeded 15%, with **22/24** sources retaining at least 75% of hybrid_50's old-template detection.

### Which unfamiliar supports drove the failure?

| New OOD attack-support generator | NOR-31 mixture | Hybrid_50 | Pure uniform |
| --- | ---: | ---: | ---: |
| Central contiguous 96 | **4.0221%** | 3.0874% | 6.0313% |
| Modular stride-17 positions | **6.1120%** | 5.9865% | 6.1167% |
| 96 smallest absolute forecast values | **3.9989%** | 3.2247% | 6.0903% |
| 96 largest cyclic second differences | 20.2017% | **26.1749%** | 6.0455% |
| 96 lowest old-prior risk positions | **3.9785%** | 3.1596% | 6.1239% |
| Public source-hash-selected 96 positions | **5.9147%** | 5.7648% | 5.9469% |

The curvature family had the largest differential, approximately **−5.97 percentage points** for NOR-31 relative to hybrid_50. Although curvature is a new *algorithm* for constructing an attack region, its selected positions were strongly aligned with historic supports. On average **74.75/96** curvature indices lay in the original high-magnitude support, while **54.96/96** lay in the old tail support. That cross-family overlap explains why the more aggressively targeted hybrid_50 recovered its advantage. Conversely, central, low-absolute-value and anti-prior supports had on average **91.67**, **91.25** and **96** positions respectively inside the old off_template complement, where the mixture's larger randomized reserve improved detection.

The two degree groups exhibited different source-level rates: at degree 512, OOD blind mixture 9.205% versus hybrid_50 9.609%; at degree 1024, mixture 5.538% versus hybrid_50 6.191%. Across source identities the signs of blind gains varied, including four positive cases in 24.

## Exact-control and statistical scope

The registered finite mixture floor, assuming four changed positions fixed before hidden independent audit coins, was positive for both output lengths. All 24 original public masks and native future predictions agreed, all honest original targets passed, and fully schedule-informed attackers evaded by choosing unchecked coordinates.

At the first evaluation, **87,552** explicitly checked source/attack/schedule pairs compared actual target integers at both nonzero deltas 1 and 4,294,049,777 and matched the equivalent exact finite set-intersection decision. The complete count grid also uses this algebraically equivalent integer-bitset intersection, which the separate independent replayer reconstructs through a different trial/coordinate representation. The exact coefficient witness is an **assumed external authentic oracle** and the deterministic public SHA-derived challenge streams are evidence receipts, not private production randomness.

The scientific independent unit is **24 source cases**. Attack restarts and challenge draws are paired repeated observations within each source. The sample supports six particular generation mechanisms; success or failure on this source roster does not establish distribution-free coverage of all conceivable support families.

### Permanent original evidence

The exact original source fixture and full first-run outcome are retained in [nor32_original_hosted.zip](results/nor32_original_hosted.zip), with [first-run SHA receipt](results/nor32_first_frozen_receipt.json). The original GitHub-hosted two-file artifact was produced before any posthoc correction or interpretation. An independent audit recomputes original generating programs, public acceptance masks, all six support definitions, every fixed/random/calibrated attack, all auditor plans, exact oracle controls and every source-level PASS/FAIL gate, and checks the immutable first-run hashes before publication.

The formal companion [MathCan NOR-32 PR #123](https://github.com/patrickdugan/MathCan/pull/123) proves the exact dual count of uniform altered r-subsets avoiding a fixed k-coordinate audit. It also proves that under **uniformly sampled attacks over all output coordinates**, every same-budget inspection policy and every finite policy mixture has the same average detection rate. Thus a nonuniform audit obtains gains only through a different allocation of inspection probability across support families.
