# NOR-23: exact coordinate audit frontier

**Prospective registration:** [REGISTRATION.md](REGISTRATION.md), committed as ed760cf before any NOR-23 source generation, challenge sampling or sealed grading. This follows [NOR-21](https://github.com/MoralityLabAI/Compactification-Theory/pull/185) (conditional exact public verification) and [NOR-22](https://github.com/MoralityLabAI/Compactification-Theory/pull/187) (target-dependent modular fingerprints and colliding hidden futures). Companion [MathCan NOR-23](https://github.com/patrickdugan/MathCan/pull/112) proves the exact finite-sample counting and adaptive blind-spot results in Lean 4.35.

## Research question

Given a correct, frozen public-only candidate prediction from the original sixteen-program RRC25 grammar, what happens if an **assumed honest external witness** reveals exactly k true target-output coefficients? Target witness data are new information; an audit pass on k<m coordinates does not prove full future correctness. The original public-only indistinguishability boundary continues to hold without such additional information.

The experiment uses 12 genuinely distinct source observations (2 independent replicate seeds × degrees 512/1024 × three NOR-21 source families) and four independent, frozen input-generating programs per family/degree grouping. Each source gets the same public prediction across all attacks and audit budgets. Two nonzero shift magnitudes, +1 and +4,294,049,777 (the NOR-22 product-modulus collision), change r=1/4/16 output positions. Target words are sealed until after the public proposal is fixed.

For each distinct source, each k∈{0,1,4,16,64} has **4,096** independent deterministic challenge draws uniformly sampling k distinct output indices without replacement. The blind adversary has committed to r changed indices before challenge sampling; the adaptive attacker sees which positions will be checked and alters only unchecked indices. The latter threat deliberately violates the blind attack's independence premise.

## First prospective hosted outcome

[First original green Ubuntu 24.04 CPython 3.12 CI run #43828806440](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43828806440). All three jobs passed, including exhaustive small-population Python controls, exact source-locked first execution, and a **separate-process independent replay of the original artifacts**.

- First public-only source fixture: 2,364,428 bytes; SHA-256 **f8b293966a7332d5f0ddb572aab57eababe35a1afba08718bb488ab9a071094b**.
- Original full per-source/budget grade: 153,608 bytes; SHA-256 **ede96e99f3eba9d168020e4080f1db304770c600f17ba897ef8eda7cf19f6ba0**.
- [Preserved original ZIP](results/nor23_original_hosted.zip), with those exact two files, and [frozen first-run receipt](results/nor23_first_frozen_receipt.json). The permanent independent auditor checks each archive member's SHA against the receipt before rederiving results.

All **60** degree × k × r × magnitude cells passed the registered five-standard-error-plus-0.015 check against the exact hypergeometric detection probability, \`1 - C(m-r,k)/C(m,k)\`, conditional on an independent uniform k-subset challenge. At each degree and each (k,r,magnitude), the aggregate comprises six distinct sources × 4,096 challenge draws = **24,576 paired audit draws**. These are repeated challenges on the same source cohort, rather than 24,576 independent source inputs.

### Representative blind detection counts, unit shift

| Observed public degree | Future output length m | Audit k | Altered r | Detected / 24,576 | Observed | Exact expectation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 769 | 16 | 1 | 456 | 1.855% | 2.081% |
| 512 | 769 | 16 | 16 | 7,072 | 28.776% | 28.807% |
| 512 | 769 | 64 | 1 | 1,984 | 8.073% | 8.322% |
| 512 | 769 | 64 | 4 | 7,240 | 29.460% | 29.410% |
| 512 | 769 | 64 | 16 | 18,608 | 75.716% | 75.455% |
| 1024 | 1537 | 16 | 1 | 291 | 1.184% | 1.041% |
| 1024 | 1537 | 16 | 16 | 3,671 | 14.937% | 15.486% |
| 1024 | 1537 | 64 | 1 | 979 | 3.984% | 4.164% |
| 1024 | 1537 | 64 | 4 | 3,886 | 15.812% | 15.658% |
| 1024 | 1537 | 64 | 16 | 12,170 | 49.520% | 49.536% |

The full per-source results include k=0/1/4 and all radii. Changing the shift from +1 to +4,294,049,777 gives the **same detection counts**, as predicted for exact integer coefficient witnesses; unlike the fixed small modular fingerprints in NOR-22, coordinate checks depend on attack location rather than residue magnitude. Valid targets passed every challenge. For the **adaptive attacker**, every altered world passed every k≤64 challenge because at least r uninspected indices remained.

### CPU cost

The experiment charges exact challenge generation and comparison separately on Ubuntu 24.04 CPython 3.12. Median per-draw challenge generation times across the six degree-512 sources were approximately **8.8 μs** for k=0, **17.0 μs** for k=16, and **39.3 μs** for k=64. Across degree-1024 sources the corresponding medians were approximately **8.8, 17.0, and 38.9 μs**. These measurements include SHA-256-seeded Python PRNG initialization and sampling; they should not be interpreted as hardware-efficient cryptographic verification costs.

The native witness comparison covers a fixed thirteen-case batch per draw: one valid and twelve altered-world/radius/magnitude/threat checks. Median comparison CPU per one of those checks grows from approximately 0.6 μs at k=0 to 6.3–6.7 μs at k=64 in this synthetic Python implementation. Times are descriptive; the preregistration imposed no CPU success threshold.

## Reproduce and audit

    python -m unittest research.nor23_coordinate_audit.test_experiment -v
    python research/nor23_coordinate_audit/experiment.py --make-fixture /tmp/nor23_fixture.json
    python research/nor23_coordinate_audit/experiment.py --check --fixture /tmp/nor23_fixture.json --out /tmp/nor23_full.json
    python research/nor23_coordinate_audit/receipt_audit.py --check --fixture /tmp/nor23_fixture.json --full /tmp/nor23_full.json
    python research/nor23_coordinate_audit/receipt_audit.py --check --archive research/nor23_coordinate_audit/results/nor23_original_hosted.zip

The independent auditor reconstructs the planted programs from the preregistered fresh generator stream, computes exact native 16-template public masks, checks full future predictions against the independently reconstructed truth, and replays **245,760 challenge sets and 2,949,120 paired altered-world audit decisions**. The first frozen source fixture contains neither private planted programs nor future targets. The evidence supports this finite synthetic attack/witness model under the explicitly assumed honest publisher, with no cryptographic unpredictability or external theorem-proving guarantee.
