# NOR-27: audit seed-prefix leakage versus revealed reserve-coordinate leakage

Registered prospectively before any new source data in [REGISTRATION.md](REGISTRATION.md), first commit bcc49dfb. Continues the merged [NOR-26 applied policy-aware blind adversary](https://github.com/MoralityLabAI/Compactification-Theory/pull/194) and [MathCan NOR-26 random-prefix theorem](https://github.com/patrickdugan/MathCan/pull/116).

The causal distinction is information about the **randomized audit schedule**. An attacker can learn some bits of a seed without learning the resulting inspected coordinates, depending on the seed expansion and leakage channel. Revealing actual audit coordinates directly shrinks the randomness remaining after the attack is committed.

## Game A: previewing the actual uniform reserve

A public deterministic eight-coordinate trained-risk prefix is followed by eight random reserve coordinates, sampled without replacement. The attacker observes the first ell coordinates of the realized reserve, for ell in 0,2,4,6,8, before choosing four changed positions outside all known inspected indices.

Once the prefix of ell random reserve coordinates has been revealed, the remaining u-ell positions are uniform among m-t-ell coordinates. The exact ideal-random detection probability is

    P(hit | ell disclosed, any legal radius-r fixed attack) =
      1 - C(m-t-ell-r,u-ell)/C(m-t-ell,u-ell).

The full disclosure endpoint ell=u yields zero detection whenever enough unchecked positions exist. The study compares 1024 paired simulated audits per new source across all five leak levels and verifies the exact source-level combinatorial expressions. The choice function receives the disclosed prefix and source digest, never the undisclosed reserve suffix or its trial seed.

This experiment measures what happens when the attacker can predict some concrete sample outputs; it is not an assertion that ell disclosed coordinates correspond to ell leaked raw seed bits.

## Game B: high bits of the raw 64-bit sample seed

Independently, an attacker receives the top b=0,2,4 bits of a 64-bit sample seed, together with the public auditor algorithm and source. A source-specific seed and a policy label are domain-separated through SHA256 before Python Random chooses audit coordinates. The attacker must choose a fixed radius-four alteration after learning b and the corresponding bucket, and before the held-out audit seed is drawn from that bucket.

For each b-bit bucket, construct 128 independent calibration schedules and 256 separate evaluation schedules. Attack selection greedily minimizes exposure in the calibration schedule distribution, using four tie-randomized attacks per bucket and auditor. Auditors are the fixed eight-risk/eight-uniform hybrid and a 16-query fully uniform comparator. The seed prefix b leaves 64-b unobserved seed bits under the ideal uniform-seed game. The same b-bit bucket is known during both training and testing, while their lower-bit draws and salts remain disjoint.

The 12 independently generated RRC25 public-only inputs consist of replicates 8 and 9 at observation degrees 512 and 1024, over all three original NOR-21 source families. The frozen training prior comes from NOR-24, and the RRC25 verifier is unchanged. Public proposals and audit plans are fixed before any private original generating program is reconstructed.

The calibration and evaluation seed sets are source-and-phase-separated by SHA256 domain labels. These research streams are reproducible and public, so this protocol is a simulation of privately held randomness and its leakage, not cryptographic evidence for the specific Python pseudorandom generator.

## Predeclared outcome

The frozen PASS/FAIL criterion uses source-level averages: detection must degrade materially from reserve ell=0 to ell=4, reach exactly zero at ell=8, and remain within 1.5 percentage points of the theoretical conditional detection probabilities by output length. Game B measures whether b=2 or b=4 changes policy-specific source-mean detection by more than 1.5 percentage points relative to b=0. Native source parity, honest target acceptance, and nonzero shift invariance are required. See [REGISTRATION.md](REGISTRATION.md) for thresholds and streams.

Twelve source units are the independent observations. Attack tie replicas, buckets and trial ordinals all share source information and must be treated as repeated challenge data.

## Reproduce

From the repository root on Python 3.12 with standard-library dependencies:

    python -m unittest research.nor27_seed_leakage.test_experiment -v
    python research/nor27_seed_leakage/experiment.py --make-fixture /tmp/nor27_fixture.json
    python research/nor27_seed_leakage/experiment.py --fixture /tmp/nor27_fixture.json --out /tmp/nor27_full.json
    python research/nor27_seed_leakage/receipt_audit.py --fixture /tmp/nor27_fixture.json --full /tmp/nor27_full.json

CI has a first job for protocol ancestry and exact finite controls, a separate first source-only fixture freeze and prospective experiment, and an independent second-process replayer implemented without importing the NOR-27 evaluator.

The first [prospective result](RESULTS.md) PASSED all four registered criteria in [Actions #44015771328](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44015771328), with an independent second-process replay of 577,536 attack decisions. Both unaltered original byte streams are source-controlled in [the permanent first-run archive](results/nor27_original_hosted.zip), pinned by [their SHA-256 receipt](results/nor27_first_original_receipt.json), and checked by [the read-only auditor](original_archive_audit.py) in both applied CI verification jobs.

## Interpretation

All threat-model guarantees require an honest external exact-coefficient witness. A real adversary with code, public evaluation streams and a trial ordinal can reconstruct deterministic research audit schedules; practical protection therefore requires fresh, genuinely unpredictable private audit entropy. Any promise that leaking b raw bits preserves detection would require a separate conditional sampling/security argument for the actual production CSPRNG.
