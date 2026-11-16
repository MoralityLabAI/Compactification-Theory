# NOR-25: a uniform reserve for source-risk coordinate audits

**Registered prior to data:** [REGISTRATION.md](REGISTRATION.md), frozen in the first commit of the branch. Continues the exact finite RRC25 verifier and NOR-24 source-conditional audit study ([merged PR #192](https://github.com/MoralityLabAI/Compactification-Theory/pull/192)).

## Scientific question

NOR-24's trained static risk ranking and sequential Bayesian auditor differed by approximately **0.10 percentage points** on the in-template primary readout. Both detected **0%** of its deliberately off-template attacks with 16 inspections. The core question for NOR-25 is whether allocating eight of sixteen authenticated coefficient probes to uniformly randomized unseen coordinates can achieve nonzero worst-case blind-attack coverage while retaining useful template localization.

The prospective primary policy, **hybrid_50**, selects the first eight coordinates of NOR-24's public trained-risk ordering, then samples eight positions uniformly without replacement from the complement of those selected eight. Risk ordering uses NOR-24's **TRAIN-only** prior 295:246:153:78 over four disclosed attack templates. No NOR-25 labels update that prior.

The alternate allocation policies hybrid_25 and hybrid_75 reserve respectively four and twelve of sixteen probes for uniform exploration. Controls preserve NOR-24's source-only uniform, static-risk, shuffled-prior, posterior and privileged true-support oracle policies. An informed adversary seeing the realized schedule is scored as a separate zero-detection control.

## Exact finite blind-attack guarantee

Write m for the number of coefficient positions, k for total probes, u for uniform reserve, t=k-u for the public-risk fixed prefix, and r for the size of the fixed blind attack set. Conditional on the attack set avoiding the prefix, the number of exploration subsets missing every altered coordinate is

    C(m-t-r,u)

out of the total

    C(m-t,u).

If the attack intersects the targeted prefix, detection is certain. Therefore for any blind radius-r attack selected before audit randomization, its chance of detection is at least

    1 - C(m-t-r,u)/C(m-t,u).

The minimum is attained by attack sets disjoint from the targeted prefix, assuming r<=m-t. This is an exact combinatorial floor, not a confidence interval. The implementation stores it as exact reduced integer numerator/denominator per source, radius, budget and hybrid policy. All feasible small universes are exhaustively tested.

The probability model assumes independent idealized secret audit randomness and an attack fixed before sampling. The reproducible SHA256-based test stream is a controlled scientific experiment, not a claimed production secrecy mechanism. An adversary selecting altered coordinates *after* seeing the realized probes avoids all inspected positions whenever r unchecked positions exist.

## Prospective protocol

Twelve new source-distinct RRC25 public inputs (replicates 4 and 5; degrees 512 and 1024; each of the three NOR-21 source families), frozen observed demos, public query, header and native acceptance mask. Source generation attempts are deterministic under the new NOR-25 source offset 99325000. Each public source has m=769 or 1537 true query coefficients.

The blind attack sets are held fixed per source, attack kind, and radius before generating any challenge schedule. 1024 random challenge ordinals per source, budgets 4/16/64, altered-coordinate radii 1/4/16, both exact nonzero shift magnitudes. Independent runs compare all policies on the same attack sets, while random audit schedules are separately salted. Budget levels for hybrid policies are sampled separately to maintain the exact two-stage uniform-reserve rule, rather than artificially imposing a nested schedule.

**Predeclared primary criterion, k=16, r=4:** hybrid_50 must beat uniform in at least nine of twelve source pairs and yield at least +0.10 absolute source-mean in-template weighted detection gain. It must also detect more off-template attacks than static ranking on at least ten of twelve sources, and reach at least 2% source-mean off-template detection. We will retain and report a FAIL outcome if any threshold is missed. In-template mixture weights are tail .4, magnitude .3, lag .2, background .1, preserving NOR-24's prior test mixture.

## Reproduce

From repo root on Python 3.12 (no third-party packages):

    python -m unittest research.nor25_robust_audits.test_experiment -v
    python research/nor25_robust_audits/experiment.py --make-fixture /tmp/nor25_fixture.json
    python research/nor25_robust_audits/experiment.py --fixture /tmp/nor25_fixture.json --out /tmp/nor25_full.json
    python research/nor25_robust_audits/receipt_audit.py --fixture /tmp/nor25_fixture.json --full /tmp/nor25_full.json

The first CI job verifies source-only isolation and exhaustive finite combinatorics. A second job generates the 12 new source identities and freezes the public-only JSON bytes before any attack or audit grading. An independent third job restores the CI artifacts and separately reconstructs every source, blind attack, hybrid schedule, observation count, theoretical floor and primary pass/fail computation, without importing NOR-25's evaluator.

## Scope and receipt status

The NOR-25 registration and source are committed prior to the first held-out outcome. The first CI run and exact original output byte hashes belong in an appended, dated result section after completion. No numerical result is claimed in advance.

The honest coefficient witness remains an external target-dependent oracle. The evaluation is restricted to finite synthetic algebraic programs, declared attack families, reproducible blind challenges and the stated support-shift stressor. It does not establish an unconditional full-output verifier or robustness to an attacker with schedule visibility.
