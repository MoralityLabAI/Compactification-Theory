# NOR-26: policy-aware, schedule-blind attacks against robust coefficient audits

**Precommitted registration:** [REGISTRATION.md](REGISTRATION.md). The initial registration preceded source generation and any NOR-26 challenge sampling. This experiment follows [merged NOR-25](https://github.com/MoralityLabAI/Compactification-Theory/pull/193), retaining its frozen prior, seven original unprivileged audit policies, exact trusted-value witness and 16-coordinate budget.

The policy-aware attacker observes public source structure, the frozen policy and **256 simulated calibration schedules**. For each of seven auditor policies, it chooses four distinct altered coordinates with minimum additional empirical calibration-schedule exposure at each greedy step. A random per-replica tie order yields **eight** precommitted attack candidates per source and policy. Those attack sets remain fixed as 512 independently salted evaluation schedules are generated. Evaluation produces a full 7 × 7 attacker-policy-by-auditor-policy cross-play matrix. An additional eight fresh uniformly chosen attacks per source are controls for NOR-25's single fixed attack-set dependence.

There are **12 new source-distinct public-only RRC25 algebraic problems** (replicates 6/7 and degrees 512/1024, three NOR-21 source families). The private true generating program is only regenerated after all public proposals, calibration attacks and held-out audit plans have been fixed. The original native verifier and the prior trained on NOR-24's TRAIN split are held constant.

## Primary outcome

The [preregistered rule](REGISTRATION.md) scores the **diagonal policy-aware self-play detection**, averaging eight attacker replicas times 512 held-out challenges per source. Hybrid_50 must beat static_prior on at least ten of twelve sources and achieve at least 2.0% source-mean detection. The exact combinatorial minimax floor must agree with the independent finite oracle, and an attacker observing the full schedule must be able to escape all partial audits. Every gate must pass; scientific FAIL is reported rather than altered.

Report cross-policy transfer, greedy calibration versus evaluation differences, fresh blind controls and the number of distinct greedy attacks. Eight attacks and 512 challenges within one source do not constitute 4,096 independent source units.

## Theoretical robustness

For any fixed attack set A with r altered coordinates, a policy may choose its public-risk targeted prefix T of length t using its own randomization. The final u coordinates are then selected uniformly and independently from the complement of T. Conditioned on T, either a target probe hits A and detection is certain, or the remaining uniform reserve detects with probability exactly

    1 - C(m-t-r,u)/C(m-t,u).

The same bound holds after averaging over any target-prefix distribution that is independent of the attack. It applies to a policy-aware attacker choosing A **before** the actual audit coins. The schedule-informed adversary has a constructive zero-detection strategy whenever r unchecked positions remain.

These are exact mathematical probabilities under ideal independent uniform sampling. The SHA256-based experimental streams are public reproducibility tools and are not a deployment security design. An exact target-dependent witness is an assumed external primitive.

## Commands

From the repo root on Python 3.12, using only the standard library:

    python -m unittest research.nor26_policy_aware_audits.test_experiment -v
    python research/nor26_policy_aware_audits/experiment.py --make-fixture /tmp/nor26_fixture.json
    python research/nor26_policy_aware_audits/experiment.py --fixture /tmp/nor26_fixture.json --out /tmp/nor26_full.json
    python research/nor26_policy_aware_audits/receipt_audit.py --fixture /tmp/nor26_fixture.json --full /tmp/nor26_full.json

The path-scoped CI first checks the registration ancestry and small-universe exact controls. It then freezes public-only source identities, runs the registered test, and uploads the original fixture and raw full outcome, regardless of the scientific PASS/FAIL decision. A separate GitHub Actions job replays all 12 sources, greedy attacks and cross-play decisions without importing NOR-26's evaluator.

The result section and original raw SHA256 receipts are added only after successful source-locked CI and independent replay.
