# NOR-38: full-output exploration against public curvature-optimized attacks

**Prospective protocol:** [REGISTRATION.md](REGISTRATION.md) was frozen at commit `37f9ed96b122ea16fb681dd5df638cbc4285d1e4` on November 17, 2026, before any NOR-38 source, attack calibration, EVAL schedule or graded result.

[Experimental design issue #213](https://github.com/MoralityLabAI/Compactification-Theory/issues/213) follows the complete independent NOR-37 original grade and replay in [PR #212](https://github.com/MoralityLabAI/Compactification-Theory/pull/212). The original NOR-37 SHA-locked source and grade archive is retained in `research/nor37_stochastic_curvature/results/nor37_original_hosted.zip`.

This prospective experiment uses **48 wholly new RRC25 public source identities**, replicates 60–67 at observed degrees 512/1024 across three wide input families, excluding all 180 archived earlier sources. It compares seven calibrated auditor distributions including original NOR31 mixture, deterministic curvature-transformed mix, stochastic50 mix, full-output uniform16, atom-random hybrid50, and two new whole-output reserves with probability **7/16 (primary)** or **8/16 (sensitivity)**.

Every auditor pays exactly 16 authenticated true-output coefficient queries. An unrestricted radius-four adversary learns each policy, receives 256 CAL schedules, optimizes four changed coordinates with eight seeded greedy restarts, and commits before 512 independently salted EVAL schedules. Four familiar blind attack classes and a new CAL-only low-exposure class are frozen before EVAL. Seven attacker classes plus five controls yield full 12×7 cross-play. Exact top-curvature detection is an integrated 2+2 support probability, distinct from eight finite blind attack draws.

The whole-output reserve branch is sampled uniformly from all m output coordinates, including positions outside the old top-risk union. Within the curvature branch, the original six-component NOR31 weights and eight per-trial atom quotas are preserved. The mixture is evaluated as a proper probabilistic distribution with denominator 256; diagnostic realizations are recorded separately and never reused for attack training.

For every **fixed precommitted** four-position attack A, independent hidden uniform exploration with probability α gives an exact detection floor
`Pr[detection | A] >= α[1-C(m-16,4)/C(m,4)]`.
Pure uniform16 reaches the exact unrestricted fixed-support maximin value. Reproducible SHA seeds are **public research fixtures**, not a claim about deployed private entropy. An authenticated true-target coefficient witness is assumed externally, and a schedule-informed attacker can still evade.

The evaluator writes a complete public-only fixture first, then privately reconstitutes native generating programs after CAL attacks and EVAL schedules are frozen. The separate `receipt_audit.py` independently regenerates original source inputs, all policy plans, all greedy CAL attack sets, all blind controls, the exact TOP fractions and **every held-out cross-play decision**, without importing the NOR38 evaluator. Its independently implemented trial-ID SETs are cross-validated against the scored evaluator using a historical source in the pre-data test suite.

The five registered gates include self-play improvement over deterministic_mix, worst-calibrated-class competitiveness with original_mix, retained feature-aligned top-curvature benefit, exact witness and whole-output support checks, and source/grade replay integrity. The preregistered result can be **PASS or FAIL**. Once the first source fixture and grader finish, their original SHA-256 receipts must be archived before any scientific follow-up.

Companion formal work: [MathCan issue #133](https://github.com/patrickdugan/MathCan/issues/133) and the NOR-38 finite regular-game theorem module.
