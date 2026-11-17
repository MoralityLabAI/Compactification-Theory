# NOR-31: prospectively matched calibration-root versus disclosed-root attacks

[Frozen initial registration](REGISTRATION.md) at commit dd0e9f60cdab90ed2b2fb482dedc7496f7f50870. Follows merged [NOR-29](https://github.com/MoralityLabAI/Compactification-Theory/pull/197) and the separate [NOR-30 minimax mixture](https://github.com/MoralityLabAI/Compactification-Theory/pull/198).

## Experiment

The earlier NOR-29 study registered a comparison of 32 blind uniformly chosen attacks against an attacker who knew the realized root. The retrospective matched-control analysis suggested an advantage from the actual root even when the baseline attacker was given a separate 512-schedule root to optimize against. NOR-31 makes that matched search the **registered primary comparison**.

Across 24 new public-only source identities (replicates 22–25, observation degrees 512/1024, three original RRC25/NOR-21 source families), there are two attack conditions for each of three 16-probe auditors:

- **Independent calibration:** 32 greedy radius-four attacks selected from 512 audit schedules sampled using a new 256-bit calibration OS root C. Their original source/policy/replica-bound salted SHA256 commitments are frozen before a separate evaluation root E exists.
- **Leaked evaluation root:** another 32 greedy radius-four attacks chosen using the same algorithm, 32 identical tie-order seeds and the same 512-schedule search budget, after E and its entire realized schedule batch are revealed. These late attacks get new salted commitments, and their altered preimages cannot reopen a prior digest under the SHA binding assumption. Each remains fixed across its full 512-schedule batch.

**Both sets are tested on exactly the same 512 evaluation schedules**, which come from a new OS root independently minted in a later GitHub Actions job. The late attacks are deliberately in-sample optimizers. The baseline attacks have independent training data and no early access to E. This matches search resources and tests the information difference between training on another root and knowing the actual evaluation root. It does not evaluate transfer to a third unseen root.

The equal-budget auditors are hybrid_50 (eight frozen trained-risk coordinates and eight HMAC-uniform reserve probes), uniform16, and static_prior16. The original NOR-24 trained prior and public RRC25 native verifier are unchanged. Every challenge checks four exactly altered coefficients against an assumed authentic external exact-value witness. Two nonzero shifts (+1 and +4,294,049,777) must give identical decisions.

## Predeclared criterion

**Registered PASS iff all five frozen gates pass**, including strict hybrid reduction on at least 20/24 source pairs, a source-mean reduction of at least +1.5 percentage points, 4,608 valid source-bound salted commitments, independent reconstruction of the two OS roots, every HMAC audit schedule and **2,359,296** fixed-batch decisions, zero changes to already committed decisions after root disclosure, and 36,864 zero-detection per-schedule informed attack controls whose new preimages fail original binding. Every original RRC25 public mask and honest sealed generating program must agree.

The 24 source identities are the independent units. The 32 restarts and 512 trials in each phase are correlated within source. An early leaked-root greedy attack is not guaranteed to be a global set-cover minimizer; any mathematical optimum result is a separate conditional statement. Real-world OS entropy integrity, HMAC PRF pseudorandomness, SHA256 computational binding, GitHub runner isolation, and the external witness remain assumptions.

## Independent protocol stages

From repository root under Python 3.12, standard library only:

    python -m unittest research.nor31_matched_root.test_experiment -v

Source-only PREP, before either root exists:

    python research/nor31_matched_root/source.py --make-fixture /tmp/nor31_fixture.json

Calibration stage, minting OS root C and freezing its 24×3×512 schedules:

    python research/nor31_matched_root/sampler.py --phase calibration --fixture /tmp/nor31_fixture.json --out /tmp/nor31_calibration.json

Precommit stage, using C's schedules, with 2,304 salted original attack openings:

    python research/nor31_matched_root/attacks.py --class independent_calibration --make --fixture /tmp/nor31_fixture.json --phase /tmp/nor31_calibration.json --sealed /tmp/nor31_pre_sealed.json --manifest /tmp/nor31_pre_manifest.json

Evaluation CHALLENGE in a separate dependent CI job, minting E only after precommitment:

    python research/nor31_matched_root/sampler.py --phase evaluation --fixture /tmp/nor31_fixture.json --manifest /tmp/nor31_pre_manifest.json --calibration /tmp/nor31_calibration.json --out /tmp/nor31_evaluation.json

Root-leak attack from E's actual 512 challenge schedules:

    python research/nor31_matched_root/attacks.py --class leaked_evaluation_root --make --fixture /tmp/nor31_fixture.json --phase /tmp/nor31_evaluation.json --calibration /tmp/nor31_calibration.json --pre-manifest /tmp/nor31_pre_manifest.json --sealed /tmp/nor31_late_sealed.json --manifest /tmp/nor31_late_manifest.json

Exact result and independent replay:

    python research/nor31_matched_root/grade.py --fixture /tmp/nor31_fixture.json --calibration /tmp/nor31_calibration.json --pre-sealed /tmp/nor31_pre_sealed.json --pre-manifest /tmp/nor31_pre_manifest.json --evaluation /tmp/nor31_evaluation.json --late-sealed /tmp/nor31_late_sealed.json --late-manifest /tmp/nor31_late_manifest.json --out /tmp/nor31_full.json
    python research/nor31_matched_root/receipt_audit.py --fixture /tmp/nor31_fixture.json --calibration /tmp/nor31_calibration.json --pre-sealed /tmp/nor31_pre_sealed.json --pre-manifest /tmp/nor31_pre_manifest.json --evaluation /tmp/nor31_evaluation.json --late-sealed /tmp/nor31_late_sealed.json --late-manifest /tmp/nor31_late_manifest.json --outcome /tmp/nor31_full.json

The receipt auditor contains its own sparse HMAC sampling and schedule-ID-set greedy optimization code, and does not import any new NOR-31 evaluator, source generator, attack optimizer or sampler. It derives the old native generating programs only after checking the original public source proposals.

## Original evidence

Once the first complete prospective run and independent replay pass, freeze the original eight raw files: public source fixture, OS calibration root/schedules, salted precommit openings and public manifest, OS evaluation root/schedules, leaked-root openings and new manifest, and complete per-source result. Permanently commit the unaltered original bytes, with SHA256 checks in the repository CI.

A source-block bootstrap and any retrospective analysis will be labeled as descriptive and cannot alter the preregistered thresholds or first-run outcome.
