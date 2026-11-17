# NOR-29: early audit entropy disclosure and source-fixed batch attacks

**Prospective registration:** [REGISTRATION.md](REGISTRATION.md), first commit 2447248fa8c192991da695ed06af2d17d642dc5f. No NOR-29 fixture, attacks or OS-root evaluation preceded that registration. Continues [NOR-28](https://github.com/MoralityLabAI/Compactification-Theory/pull/196) and its [MathCan binding proof](https://github.com/patrickdugan/MathCan/pull/118).

NOR-28 demonstrated that, after an attack has become binding, revealing arbitrary bits of the audit root does not change the already-fixed detection decision. NOR-29 changes the **commitment time**. An opponent given the full audit root and all derived schedule positions *before* committing its alteration can choose a fixed set that avoids as many sampled schedules as possible.

## Three genuinely distinct attack powers

The first treatment **blind_uniform** binds 32 independently drawn radius-four alteration sets per source before fresh OS entropy exists. Its early source-bound SHA256 commitments use independent 32-byte salts. All challenge policies test the same 32 sets on matched audit schedules.

The second treatment **early_root_aware** sees the root and every schedule for the source's 512-audit batch before choosing its altered set. It greedily minimizes the number of schedules touched by its four positions, with 32 independent randomized tie breaks per source and auditor. The attacker makes a new post-root commitment. Each chosen set remains fixed across the whole 512-schedule batch. This is deliberately an *in-sample* best response, not held-out generalization.

The third treatment **fully_schedule_aware** chooses a different altered set after each particular audit schedule is disclosed. It can always choose four uninspected positions, but it has to create a new commitment outside the original protocol. This is a stronger attacker than the batch-fixed early-root treatment.

The data contain 24 fresh and source-distinct RRC25 algebraic public inputs, replicates 12/13/14/15 across observation degrees 512/1024 and the three original NOR-21 source families. Each produces 32 precommitted blind and 32 post-root attacks per auditor. At k=16 each auditor performs 512 newly rooted challenge schedules. The auditors are public deterministic risk16, risk8+uniform8, and full uniform16. All use the frozen NOR24 TRAIN-only prior and the original native exact public verifier.

## Exact blind probability and commitment boundary

For an independently fixed attack of r=4 positions, hybrid_50 has the ideal-random minimax detection floor

    1 - C(m-12,8) / C(m-8,8),

where m is 769 or 1537. This bound applies only if the attack is fixed **before** the unobserved reserve is drawn. A root-aware attacker who chooses the attack *after* seeing the realized challenge draws violates independence. The quantitative comparison therefore measures the causal importance of commitment ordering.

The root is sampled once in a new GitHub Actions CHALLENGE job with secrets.token_bytes(32) and domain-separated through HMAC-SHA256. This study assumes secure HMAC and SHA256 binding. The root is revealed in the original outcome to support replay; a real deployment would require a protected source of entropy until the attack was bound.

## Predeclared decision

At 24 independent source units, the hybrid blind precommit detection rate must exceed the same auditor's early-root-aware batch-fixed rate on at least 20/24 source pairs, with equal-source mean difference at least +1.0 percentage point. Additionally, all 768 first commitments must open exactly and all attempted one-coordinate mutations must fail; 2304 new root-aware attacks must fail to reopen prior commitments; an independent process must reconstruct the schedules and all 2,359,296 fixed-batch decisions; two nonzero exact shifts and postbinding root disclosure views must give unchanged decisions; the fully informed per-schedule attacker must evade every audit while failing original binding. All five gates are fixed before data. A scientific FAIL is kept when any threshold is missed.

## Reproduction

Python 3.12, standard library only. There are distinct programs for each stage; the GitHub Actions pipeline uses separate jobs and immutable named artifacts for ordered handoff.

PREP:
    python research/nor29_early_entropy/prepare.py --make-fixture /tmp/nor29_fixture.json
    python research/nor29_early_entropy/prepare.py --prepare --fixture /tmp/nor29_fixture.json --sealed /tmp/nor29_early_sealed.json --manifest /tmp/nor29_early_manifest.json

CHALLENGE:
    python research/nor29_early_entropy/challenge.py --fixture /tmp/nor29_fixture.json --manifest /tmp/nor29_early_manifest.json --out /tmp/nor29_challenges.json

LATE ROOT DISCLOSURE:
    python research/nor29_early_entropy/leaked_root_attacker.py --make --fixture /tmp/nor29_fixture.json --early-manifest /tmp/nor29_early_manifest.json --challenge /tmp/nor29_challenges.json --sealed /tmp/nor29_late_sealed.json --public /tmp/nor29_late_manifest.json
    python research/nor29_early_entropy/grade.py --fixture /tmp/nor29_fixture.json --early-sealed /tmp/nor29_early_sealed.json --early-manifest /tmp/nor29_early_manifest.json --challenge /tmp/nor29_challenges.json --late-sealed /tmp/nor29_late_sealed.json --late-manifest /tmp/nor29_late_manifest.json --out /tmp/nor29_full.json

INDEPENDENT REPLAY:
    python research/nor29_early_entropy/receipt_audit.py --fixture /tmp/nor29_fixture.json --early-sealed /tmp/nor29_early_sealed.json --early-manifest /tmp/nor29_early_manifest.json --challenge /tmp/nor29_challenges.json --late-sealed /tmp/nor29_late_sealed.json --late-manifest /tmp/nor29_late_manifest.json --outcome /tmp/nor29_full.json

The independent replayer **never imports any NOR-29 evaluator, sampler or attack-selection code**. It implements its own sparse Fisher-Yates HMAC schedule generator and explicit schedule-ID sets for the greedy adversary, verifies the earlier public source native masks and reconstructs source-level rates.

## Scope and evidence

The 24 public source identities are independent units. The 32 attacks, 512 audits and root-optimized responses are correlated within each source. A source-level paired difference is the principal readout, and a source-block bootstrap is descriptive. Two nonzero exact shifts can be scored on the same fundamental decision and are never counted as new samples. Experimental SHA256 salts and root disclosure after attack grading enable reproducibility and do not prove any cryptographic primitive secure.

All stage artifacts and the registered criterion will be preserved with immutable raw SHA256 receipts after the first successful independent replay. A real verifier would still need an authenticated external witness to the true coefficients, plus a trusted audit RNG and commitment-verification boundary.
