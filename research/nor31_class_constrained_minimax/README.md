# NOR-31: class-collapsed minimax with a historical template floor

**Prospectively registered 2026-11-17 at commit 7e37e8c**, before any new NOR-31 source data. Builds directly on merged NOR-30 [PR #198](https://github.com/MoralityLabAI/Compactification-Theory/pull/198) and [MathCan #120](https://github.com/patrickdugan/MathCan/pull/120).

## The specific NOR-30 defect corrected

The previous 13/16 shuffled_prior plus 3/16 uniform mixture is mathematically **uniform in distribution**. NOR-31 removes the duplicate shuffled_prior policy from its six eligible defender classes, while retaining shuffled-calibrated attacks. It adds a historical source-structured detection constraint so any frozen mixture must retain useful risk-targeted coverage.

TRAIN imports two **previously executed, SHA-pinned** datasets. NOR-26's original 12-source 7×7 policy-aware cross-play establishes the attack-game payoff matrix (seven attack classes, six nonalias defender columns). NOR-30's original 12-source four-family template tests establish **independent historical structured-detection** per defender, with fixed 4:3:2:1 family weights. The exact optimizer searches all six-class 16-quantum mixtures, subject to uniform weight 2–12 quanta and weighted historical in-template detection at least 18%. It chooses maximum worst-row payoff, then highest total payoff, then lexicographically smallest weight vector.

The frozen TRAIN receipt is uploaded before a dependent job generates twelve genuinely fresh public-only RRC25 source problems (reps 24/25, degree 512/1024 and three original wide families). Sources are disjoint from both historical sets. The original NOR-21 16-template public checker, true-future isolation, external exact-coefficient witness and NOR-25 audit policies remain fixed.

The evaluator registers 8 policy-aware four-coordinate attack classes: seven historical policy-specific greedy attackers and a new **mixture-aware** approximate best response. Each uses eight attack restarts optimized against 256 calibration schedules, fixed before 512 independent held-out schedules. New public source tests also include uniform blind attacks and tail/magnitude/lag/uniform/off-template controls. An attacker that sees the final realized challenge schedule is a separately labeled, stronger zero-detection control.

## Precommitted primary gates

At k=16, r=4, the constrained mix must beat hybrid_50 against each policy's own best response on at least 9/12 new source pairs with at least a +0.5 percentage point average gain, improve worst-case eight-attacker cross-play on at least 9/12, and retain at least 17% mean in-template detection plus 75% of hybrid_50's template detection on at least 9/12 source pairs. Exact native-mask, valid-output, nonzero-shift agreement, positive theoretical blind floor, fully informed evasion and independent complete replay also must pass. Any failed gate remains scientific FAIL.

The mathematical companion [MathCan NOR-31](https://github.com/patrickdugan/MathCan/pull/121) proves permutation-ranking class equivalence, duplicate payoff column collapse, abstract necessity of genuine risk-targeted weight under a strict template floor, and inherited conditional exact detection bounds.

## First hosted outcome

**PASS on all five frozen gates** in the first original complete [Actions run #44032550237](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44032550237), including separate-process reconstruction of both historical cohorts, all 12 original new public programs, and **4,128,768 policy-world decisions**.

The frozen trained weights are **[2,0,1,1,4,8]/16**, with historical in-template rate 18.0236%. On 12 fresh sources, the mixture caught **4.03%** of its own policy-aware best responses versus hybrid_50's **3.19%**, and achieved **3.75% worst-case** against eight attacker classes versus hybrid_50's **2.70%**. All 12 sources favored NOR-31 on both preregistered adversarial comparisons. The mixture retained **17.36% mean in-template detection**, compared with hybrid_50's **22.37%**, with **10/12 source pairs** meeting the 75% relative-retention criterion. It caught **3.91%** of off-template attacks versus hybrid50's **3.17%**. The registered mean self-response gain was **+0.842 percentage points**.

A pure uniform16 auditor still had stronger raw minimax detection against these policy-aware attacks (5.98% self-response, 5.41% worst-case) while detecting only 6.05% of the structured template attacks. NOR-31 is a **constrained balance**, not an unrestricted minimax solution. Two degree-1024 sources fell below the retained relative template-coverage gate, while 10 passed. The exact conditional ideal blind lower bounds are 5.2484% and 2.6467% at m=769/1537 output coordinates.

The original TRAIN receipt (5,300 bytes, SHA256 ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0), new public-only TEST fixture (2,245,544 bytes, SHA256 b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85), and complete first outcome (317,452 bytes, SHA256 f978a2c388715056e7999e01d4ddc30f6c24a778b0e6e3e741a070902223430c) are preserved in [the permanent first-run ZIP](results/nor31_original_hosted.zip), with [frozen receipt](results/nor31_first_frozen_receipt.json). Permanent read-only CI checks all original hashes and independently replays the original source/attack grid.

See the detailed [RESULTS.md](RESULTS.md) for the source-level table, original experimental assumptions and the exact mixed-versus-pure-uniform tradeoff.

The source interpreter is exact finite-series RRC25. Authenticated target coefficient witnesses and independently concealed audit randomness remain external assumptions. Public SHA-derived experimental streams are deterministic replay devices and carry no cryptographic unpredictability claim.
