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

This heading is registered before any fresh NOR-31 data. Append raw source and result SHA-256 receipts and source-level outcomes only after the original CI and independent replay finish. Avoid retroactive threshold changes and treat 12 source identities, rather than repeated attack restarts/challenges, as the independent sample size.

The source interpreter is exact finite-series RRC25. Authenticated target coefficient witnesses and independently concealed audit randomness remain external assumptions. Public SHA-derived experimental streams are deterministic replay devices and carry no cryptographic unpredictability claim.
