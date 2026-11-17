# NOR-30 first prospective source-heldout minimax audit outcome

**Registered primary verdict: PASS. Scientific interpretation: the selected mixed policy is distributionally a uniform auditor, and trades away the structured support coverage demonstrated by NOR-25.**

First registration commit [118dc142](https://github.com/MoralityLabAI/Compactification-Theory/commit/118dc142b1bd9143218f9868fea248c53ddc62dc) froze the training source, 16-quantum grid, test source roster, policy-aware attacks and four primary gates before new source generation. The first complete original GitHub Actions run [#44027909426](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44027909426) passed all four stages, including an **independently implemented 7×7 historical optimizer and 4,816,896 prospective held-out cross-play decisions**.

## Source separation and exact frozen training solution

The only training evidence was the **original NOR-26 first-run 7×7 integer cross-play matrix**, re-read directly from the 12 historical source records at original result SHA-256 \`a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a\`.

All 74,613 integer 16-quantum seven-policy mixtures were enumerated retrospectively in the independent replay. The constrained experiment searched 38,676 vectors with uniform weight 2–12 quanta. Its frozen maximizer, identical to the unrestricted grid maximizer, was:

| Auditor | Quanta / 16 |
| --- | ---: |
| Original uniform16 | 3 |
| Shuffled risk rank | 13 |
| Static, posterior, hybrid_25, hybrid_50, hybrid_75 | 0 |

The training matrix SHA-256 was \`9dbffb5d3015f705e3b5b09464260ca5bd7f1dc5750047c35f63f2462dff1192\`, and the pre-test TRAIN artifact was source-locked before the new test fixture job began.

**A post-result structural observation:** NOR-24's \`shuffled_prior\` first picks a fixed static ranking then applies a uniform random permutation of all m output coordinates before reading the ranked positions. Conditional on any static ranking independent of this permutation, its first k selected coordinates are uniformly distributed among distinct k-coordinate tuples. Accordingly a mixture of 13/16 shuffled_prior and 3/16 uniform is *exactly uniform in the ideal sampling distribution*. This is an admissibility loophole in the registered constraint: the search prevents the explicit uniform arm from getting all 16 quanta while allowing a second arm with the same distribution. The result cannot establish that combining substantively different audit strategies improves worst-case coverage. A new exhaustive finite-permutation control was added **retrospectively, after the first result**, to verify the identity for small m; it does not change the original frozen source/thresholds.

## First 12-source TEST outcomes

12 fresh public-only RRC25 problems were generated from new replicate seeds 20/21 and were SHA-disjoint from historical NOR-26 sources. Complete native masks, public query futures, and sealed postchoice native truth all agreed. Every attacker picked four changed coordinates using 256 calibration schedules; eight greedy restarts per policy and one greedy mixture-aware attack family were fixed before 512 evaluation schedules. The evaluation also included eight fresh blind attacks and 8 attacks each in five source-support families.

At k=16 and r=4, the registered two fixed-threat-model gates and exact witness controls passed:

| Equal-source mean | Frozen 3/16 uniform + 13/16 shuffled | Original hybrid_50 |
| --- | ---: | ---: |
| Self-play detection against each policy's own optimized attack | **6.4973%** | 3.0619% |
| Minimum over 8 policy-aware attacker classes | **5.4839%** | 2.6754% |
| In-template attacks, registered .40/.30/.20/.10 family mixture | 6.1026% | **22.7828%** |
| Off-template attacks outside all 3 structured supports | **5.9432%** | 2.9663% |

The source-level paired self-play gain is **+3.4353892 percentage points**, clearing the preregistered +0.5 percentage point threshold. The mixed policy beat the fixed hybrid50 self-play and worst-case cross-play measures on **12/12 independent source identities** (both gates required at least 9/12).

### Source-level paired self-play (percentage points)

| Source replicate | D | Family | Mixed detect | Hybrid50 detect | Paired gain |
| ---: | ---: | --- | ---: | ---: | ---: |
| 20 | 512 | dense | 9.692% | 3.882% | +5.811 |
| 20 | 512 | sparse | 9.190% | 3.735% | +5.455 |
| 20 | 512 | rational | 7.825% | 4.272% | +3.552 |
| 20 | 1024 | dense | 4.456% | 1.855% | +2.600 |
| 20 | 1024 | sparse | 4.079% | 2.441% | +1.637 |
| 20 | 1024 | rational | 4.298% | 2.319% | +1.979 |
| 21 | 512 | dense | 8.789% | 3.955% | +4.834 |
| 21 | 512 | sparse | 7.761% | 4.248% | +3.513 |
| 21 | 512 | rational | 9.338% | 3.784% | +5.554 |
| 21 | 1024 | dense | 4.202% | 1.880% | +2.322 |
| 21 | 1024 | sparse | 4.453% | 2.002% | +2.451 |
| 21 | 1024 | rational | 3.885% | 2.368% | +1.517 |

The degree-512 mean advantage is +4.786 percentage points and the degree-1024 advantage +2.084 points. The source-level estimates include eight attacks and 512 repeated schedule draws on the same source, and should not be treated as thousands of independent source experiments.

## Formal and security scope

The companion [MathCan NOR-30 #120](https://github.com/patrickdugan/MathCan/pull/120) passed Lean 4.35 kernel and axiom checks. It proves that any finite mixture inherits a weighted sum of valid per-arm detection floors, along with a **separate toy game** where mixing genuinely improves minimax protection. This finite-game existence result does not imply that the *fitted NOR-30 policy* achieved a genuine policy-diversity advantage.

The preregistered formal floor treated shuffled_prior conservatively as an arm with zero guaranteed reserve coverage. On the actual selected weights, the registered weighted bound was 1.5153% at output length 769 and 0.7694% at length 1537. The post-result shuffle-equivalence observation explains why the ideal uniform law permits a tighter floor. The original certified conservative bounds remain preserved.

Against attacks observing the realized audit coordinates, every radius-four attack could choose unchecked positions and evade. Exact honest witness comparisons were equivalent under the +1 and +4,294,049,777 target perturbations. The public SHA-derived experimental streams permit deterministic replay but are **not** secret or cryptographically secure. A deployed audit would require genuine independent private randomness and an authenticated external truth witness.

## Original immutable evidence

The original source-locked first-run evidence is preserved in [nor30_original_hosted.zip](results/nor30_original_hosted.zip) with [frozen SHA receipt](results/nor30_first_frozen_receipt.json). Original byte identities:

- NOR-26-only frozen train receipt: **3,113 bytes**, SHA-256 \`df9414c3ee3132a90fe8e6cb5fd231332e480fe9e87ea5b3c3bef86045330c84\`.
- First public-only 12-source TEST fixture: **2,284,577 bytes**, SHA-256 \`9961afd81cac78b32fc5280fe54291f7db4dca0cf85cb6624560ddac420e1878\`.
- Original complete TEST outcome: **337,563 bytes**, SHA-256 \`4189235f7cb5c14f3c7d1028abda264d893c7b5e5d2eda0c4c1de091f052b9c8\`.

The independent auditor imports original RRC25 interpreter semantics and historical NOR-25 schedule definitions, but does not import the new NOR-30 evaluator or train solver. It recomputes all 74,613 grid options, the 12 original planted programs, the 256/512 per-source calibration/evaluation plans, greedy attacks using a different trial-ID set representation, and every recorded cross-play cell. A read-only permanent archive auditor validates those SHA identities and reruns the whole independent replay in CI.

## Reproduction

    python -m unittest research.nor30_minimax_audits.test_experiment -v
    python research/nor30_minimax_audits/train.py --out /tmp/nor30_train_receipt.json
    python research/nor30_minimax_audits/experiment.py --training /tmp/nor30_train_receipt.json --make-fixture /tmp/nor30_fixture.json
    python research/nor30_minimax_audits/experiment.py --training /tmp/nor30_train_receipt.json --fixture /tmp/nor30_fixture.json --out /tmp/nor30_full.json
    python research/nor30_minimax_audits/receipt_audit.py --check --training /tmp/nor30_train_receipt.json --fixture /tmp/nor30_fixture.json --full /tmp/nor30_full.json
    python research/nor30_minimax_audits/original_archive_audit.py
