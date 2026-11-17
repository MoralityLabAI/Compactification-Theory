# NOR-32: nested 512-to-2048 audit scaling under full-root disclosure

**Frozen prospective registration:** [REGISTRATION.md](REGISTRATION.md), first committed as 44d8c506de2b2079c167cb0ce5f3a2eb620028b5, before any source or audit data.

The experiment isolates **finite-batch coverage** from per-challenge detection probability. In NOR31-M, known-root attackers could find four coefficient positions that remained untouched across the entire 512-audit batch of long RRC25 outputs. NOR-32 asks whether a larger but still completely disclosed batch removes that exact zero-detection opportunity, with precommitted independent-calibration and informed attackers both using equal greedy search budgets.

## Prospective design

- Fresh 24 public-only RRC25 sources (replicates 26–29, observation degrees 512/1024, three original NOR-21 families). The original native exact public-mask verifier and frozen NOR-24 TRAIN-only prior are unchanged.
- Two separately minted OS-root values, one in calibration PREP and another in the dependent CHALLENGE job. Each root produces ONE 2048-audit schedule sequence per source and policy. The four batch sizes are **nested prefixes** n=512, 1024, 1536, 2048 of that identical sequence.
- Three k=16 auditors: hybrid_50 (risk8 plus 8 uniform), fully uniform16, and deterministic static_prior16.
- For EACH depth/policy/source, 32 greedy restarts choose a fixed four-coordinate alteration using exactly the first n schedules and the same tie-seed stream as NOR31. The independent-calibration attack has its full set **bound before the evaluation OS root exists**. The disclosed-root attack sees the complete evaluation batch and chooses an in-sample fixed response only after its root has been revealed, with a new salted commitment that cannot open the original.
- Exactly **9216** pre-root salted commitments, 9216 postroot commitments and **23,592,960** source/policy/attack/audit decisions. All 147,456 complete-schedule-informed per-trial controls must evade their audits and fail to reopen preexisting commitments.

The primary feasibility score counts sources at m=1537 where **at least four coordinates were NEVER inspected anywhere** in the nested audit batch. This count is guaranteed monotone under batch extension for fixed schedules. The registered empirical gate requires at least 10/12 long sources to be feasible at n=512 and at most 2/12 at n=2048, a decrease of at least 8. A second primary gate requires independent-calibration hybrid detection to exceed the root-aware hybrid at n=2048 on at least 20/24 source pairs with +0.5 pp source-mean gain.

The prospective criterion includes full authentic exact target-witness parity, no original commitment substitution, two OS roots/CI stage ordering, exact independent replay and all nonzero delta checks. Any registered FAIL must be preserved without changing the original thresholds.

## Ideal-random benchmark

For m output positions and hybrid risk8/uniform8, the ideal expected untouched coordinates after n audits are

    (m-8) * (1-8/(m-8))**n.

The full-uniform16 control has expected untouched m*(1-16/m)**n; static16 leaves m−16 untouched forever. Under ideal independent sampling, Pr(at least four untouched) is bounded by expected untouched /4 using Markov. These benchmark probabilities concern ideal uniform randomness. The root and HMAC sampler in the experiment represent an operational computational model under explicit SHA, HMAC and OS entropy assumptions.

## Reproduction

Python 3.12, standard library, staged GitHub Actions jobs for source/calibration precommit, evaluation-root freeze, late-root attack grading and independent source-locked replay. All paths below are example local work directories:

    python -m unittest research.nor32_nested_audit_depth.test_experiment -v
    python research/nor32_nested_audit_depth/source.py --make-fixture /tmp/nor32_fixture.json
    python research/nor32_nested_audit_depth/sampler.py --phase calibration --fixture /tmp/nor32_fixture.json --out /tmp/nor32_calibration.json
    python research/nor32_nested_audit_depth/attacks.py --class independent_calibration --make --fixture /tmp/nor32_fixture.json --phase /tmp/nor32_calibration.json --sealed /tmp/nor32_pre_sealed.json --manifest /tmp/nor32_pre_manifest.json
    python research/nor32_nested_audit_depth/sampler.py --phase evaluation --fixture /tmp/nor32_fixture.json --calibration /tmp/nor32_calibration.json --manifest /tmp/nor32_pre_manifest.json --out /tmp/nor32_evaluation.json
    python research/nor32_nested_audit_depth/attacks.py --class leaked_evaluation_root --make --fixture /tmp/nor32_fixture.json --phase /tmp/nor32_evaluation.json --calibration /tmp/nor32_calibration.json --pre-manifest /tmp/nor32_pre_manifest.json --sealed /tmp/nor32_late_sealed.json --manifest /tmp/nor32_late_manifest.json
    python research/nor32_nested_audit_depth/grade.py --fixture /tmp/nor32_fixture.json --calibration /tmp/nor32_calibration.json --pre-sealed /tmp/nor32_pre_sealed.json --pre-manifest /tmp/nor32_pre_manifest.json --evaluation /tmp/nor32_evaluation.json --late-sealed /tmp/nor32_late_sealed.json --late-manifest /tmp/nor32_late_manifest.json --out /tmp/nor32_full.json

The receipt_audit.py program independently reconstructs all source-native masks, HMAC schedule outputs and 18,432 greedy attacks with schedule-ID sets. It does not import NOR32's source/sampler/optimizer/grader code and checks every frozen decision mask and source-level criterion.

## Formal boundary

[MathCan NOR-32](https://github.com/patrickdugan/MathCan/blob/research/nor32-nested-batch-coverage-20261117/MathCan/NOR32NestedAuditCoverage.lean) proves finite monotonicity of batch evasion feasibility and total hit counts, with an explicit counterexample showing fractional detection can decline under extra audits. The companion note derives ideal coupon-collector coverage expectations.

An attacker can still choose rarely inspected coefficients after the batch is revealed even when four totally untouched coordinates cease to exist. This experiment does not certify cryptographic entropy, true target-witness authenticity or optimized fractional detection monotonicity.
