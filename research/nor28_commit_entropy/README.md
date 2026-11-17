# NOR-28: precommit the alteration before sampling OS audit entropy

**Prospective registration:** [REGISTRATION.md](REGISTRATION.md), first branch commit da296c8, before new NOR-28 sources or attack and challenge data. Continues [merged NOR-27](https://github.com/MoralityLabAI/Compactification-Theory/pull/195).

The experiment uses an explicit ordering protocol. A PREP job commits to eight hidden, source-bound, salted four-coordinate attack sets per one of 12 fresh synthetic RRC25 public input sources. The commitment hashes are placed in a public manifest, and their preimages are put in a separate frozen artifact. Only after PREP has completed can CHALLENGE sample a new 256-bit OS root with Python secrets.token_bytes(32). This root keys an HMAC-SHA256 counter-mode partial Fisher-Yates sampler with rejection sampling, giving exactly 512 16-coordinate audits per source and policy. The complete schedules and root commitment are written and SHA-256 hashed before a distinct grader opens the attacks. Original root bytes are disclosed in the retained result so an independent process can reproduce the full audit.

**Policies:** trained-risk static16, trained-risk8 plus uniform8, and full uniform16. Frozen NOR-24 TRAIN-only prior weights 295:246:153:78/772. Calibrated attacks minimize exposure over 256 public, source-distinct training schedules. Four additional uniform attack sets control for source-specific localization.

## Postcommit disclosure invariant

Reveal 0, 8, 16 or all 256 high bits of the root *after the attacker has committed*. Every already committed attack and already fixed audit schedule remains the same across these views. Thus detection must be exactly identical for all 147,456 underlying source/policy/attack/trial decisions; recording the four leak views never multiplies the independent sample size. The invariant formalizes a binding/timing property, rather than claiming that revealing raw bits has no effect on an attacker who can act before committing.

A counterfactual attacker who sees the complete inspection set chooses four unchecked coordinates and attains zero detection; it must provide a fresh commitment that fails the earlier commitment check. The transcript proves why **commitment timing matters**. SHA-256 collision resistance and fresh runner entropy remain assumptions about the cryptographic and operational implementation.

The exact ideal-random blind radius-four detection floor under the eight-uniform reserve is

    1 - C(m-12,8)/C(m-8,8),

where m is the output length. The result is approximately 4.15% at m=769 and 2.08% at m=1537. The OS root and HMAC sampler are a computational model for ideal independent uniform draws. Full output correctness still depends on an external honest exact coefficient witness.

## Frozen PASS criterion

The precommitted [protocol](REGISTRATION.md) requires valid openings of all 96 source-bound attack commitments and rejection of their deliberate mutations; a new 256-bit OS root first drawn after PREP has completed and schedules frozen before grading; exact equality of all postcommit disclosure views and both nonzero shift magnitudes; hybrid calibrated attack detection better than static in 10 of 12 source pairs and at least 2% mean; and a completely schedule-informed adversary evading every partial audit while failing the original commitment. Legitimate unchanged target acceptance and original native public verifier parity are required.

Source-level means are over 12 new source identities (replicates 10/11, degrees 512/1024, the original three NOR-21 families). Trials and attack replicas are correlated within source.

## Reproduction

Run from repository root under Python 3.12, standard library only:

    python -m unittest research.nor28_commit_entropy.test_experiment -v

Stage PREP:

    python research/nor28_commit_entropy/experiment.py --make-fixture /tmp/nor28_fixture.json
    python research/nor28_commit_entropy/experiment.py --commit-attacks \
      --fixture /tmp/nor28_fixture.json --sealed /tmp/nor28_sealed.json \
      --manifest /tmp/nor28_manifest.json
    python research/nor28_commit_entropy/experiment.py --verify-openings \
      --fixture /tmp/nor28_fixture.json --sealed /tmp/nor28_sealed.json \
      --manifest /tmp/nor28_manifest.json

Stage CHALLENGE (new process, after PREP receipt has been frozen):

    python research/nor28_commit_entropy/challenge_sampler.py \
      --fixture /tmp/nor28_fixture.json --manifest /tmp/nor28_manifest.json \
      --out /tmp/nor28_challenges.json
    python research/nor28_commit_entropy/grade.py \
      --fixture /tmp/nor28_fixture.json --sealed /tmp/nor28_sealed.json \
      --manifest /tmp/nor28_manifest.json \
      --challenge /tmp/nor28_challenges.json --out /tmp/nor28_full.json

Independent process:

    python research/nor28_commit_entropy/receipt_audit.py \
      --fixture /tmp/nor28_fixture.json --sealed /tmp/nor28_sealed.json \
      --manifest /tmp/nor28_manifest.json \
      --challenge /tmp/nor28_challenges.json --out /tmp/nor28_full.json

**First prospective result:** [RESULTS.md](RESULTS.md), registered PASS in [Actions #44019242345](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44019242345), independently replayed in a separate process. Calibrated hybrid detected 3.015% versus 6.250% static and 5.941% uniform on the 12-source mean; the registered paired gate passed 10/12, with two static outlier wins reversing the source-mean comparison. All 96 original commitments opened, every deliberate change failed, all postcommit root disclosure views agreed pathwise, and full-schedule-informed attacks evaded but failed prior binding.

The five original first-run source, sealed-attack, public manifest, OS challenge and outcome byte streams are permanently committed in [results/nor28_original_hosted.zip](results/nor28_original_hosted.zip). Their exact first-run SHA-256 values are pinned in [the source-locked receipt](results/nor28_first_original_receipt.json) and verified by [the read-only auditor](original_archive_audit.py) in CI. The source-only fixture excludes every generating program, future target, attack position and challenge entropy.

## Boundaries

The runner and its authenticated CI artifacts are trusted to preserve preparation order and prevent postcommit substitution. Opening the attack preimages after the audit is expected, while a private OS root must be sampled only once the attack commitments are fixed. The experimental root is later public for replay. Lean proofs establish finite information-flow/counting statements under abstract binding assumptions; they cannot certify OS entropy, SHA-256 security, HMAC unpredictability or the external exact witness.
