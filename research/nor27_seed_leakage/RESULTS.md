# NOR-27 first prospective outcome: registered PASS with different leakage channels

**First prospective completed study:** [GitHub Actions #44015771328](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44015771328). All four registered gates passed across **12 genuinely fresh public-only RRC25 sources**, generated from replicate seeds 8/9, degrees 512/1024 and three original NOR-21 families. The original pre-data protocol was committed first at bcc49dfb96acd9a9aae51925630b2f04f69e2315. The full original source and outcome SHA-256 are [pinned](results/nor27_first_original_receipt.json).

The independent separate-process replay reconstructed the native public verifier masks, 1,024 per-source preview worlds, all four greedy seed-prefix attack replicas across 21 calibration buckets and both auditor policies, exact rational theoretical floors and all four source-level gates. It reproduced **577,536 exact inspection/attack decisions** and both raw-file byte hashes in [the first separate-process replay run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44015771328).

## Game A: revealing the actual reserve coordinates

The auditor inspects eight public trained-risk locations and eight privately randomized reserve positions. The attacker observes the first ell sampled reserve locations, chooses four changes outside those disclosed coordinates and the risk prefix, and must commit before the remaining reserve is disclosed. Each source uses 1,024 paired schedule draws across all five leakage levels, yielding 12 source-level detection-rate observations per ell.

| Actual reserve positions revealed (ell) | Source-mean detect | Theoretical, m=769 | Observed, m=769 | Theoretical, m=1537 | Observed, m=1537 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 of 8 | **2.946%** | 4.147% | 3.841% | 2.079% | 2.051% |
| 2 of 8 | **2.360%** | 3.131% | 2.979% | 1.564% | 1.742% |
| 4 of 8 | **1.489%** | 2.101% | 1.807% | 1.046% | 1.172% |
| 6 of 8 | **0.716%** | 1.057% | 0.863% | 0.525% | 0.570% |
| 8 of 8 | **0.000%** | 0% | 0% | 0% | 0% |

The ell=0 minus ell=4 difference was **+1.4567 percentage points**, exceeding the registered +0.8 pp gate. Every degree-and-leakage cell was within **1.5 percentage points** of its exact ideal-random conditional probability, as preregistered. The complete leak ell=8 admitted four unchecked alteration coordinates in every evaluation, and all such attacks escaped.

The conditional formula is

    P(detect | disclosed reserve prefix ell) =
       1 - C(m-8-ell-4, 8-ell) / C(m-8-ell, 8-ell),

for ell in {0,2,4,6,8}. It conditions on the attacker avoiding the disclosed coordinates and is exact under uniform hidden continuation draws. The simulation uses SHA256-expanded Python Random streams to approximate that sampling model, rather than asserting a cryptographic proof.

## Game B: revealing only high bits of the 64-bit seed

The attacker learns the b=0,2,4 highest seed bits and chooses changes based on separate 128-schedule calibration sets within that observed seed bucket. Four tie-randomized greedy attacks are then evaluated on 256 independently sampled **held-out** seeds from the same bucket. Equal-bucket averages and then equal-source averages produce these results.

| Raw seed high bits revealed (b) | Hybrid_50 self-play | Uniform16 self-play |
| ---: | ---: | ---: |
| 0 | 2.954% | 6.356% |
| 2 | 3.225% | 6.081% |
| 4 | 3.077% | 6.051% |

The maximum deviation from the b=0 hybrid mean was **0.271 percentage points** and from the b=0 uniform mean **0.305 pp**, both comfortably within the preregistered ±1.5 pp sensitivity criterion. The study shows that exposing two or four high bits of the exact simulated seed had little measured effect on this *particular* SHA256-expanded Python Random construction under the registered calibration attack.

This result says nothing about the robustness of an arbitrary RNG. A generator that directly reveals a probe location through one seed bit could fail completely. The code fixes calibration and evaluation phase salts independently; public reproducibility of the full script and all seed streams precludes interpreting these experiments as deployment cryptographic security.

## First per-source Game A evidence

All entries are detected altered outputs as a percentage of 1,024 challenge ordinals per source. Families abbreviate wide_dense_small, wide_sparse_small and wide_rational_short. Output length is 769 when D=512 and 1537 when D=1024.

| Replicate | D | Source family | ell=0 | ell=2 | ell=4 | ell=6 | ell=8 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 512 | Dense | 4.395 | 3.711 | 1.074 | 0.977 | 0 |
| 8 | 512 | Sparse | 3.418 | 3.027 | 1.953 | 0.586 | 0 |
| 8 | 512 | Rational | 4.199 | 2.734 | 1.367 | 0.684 | 0 |
| 8 | 1024 | Dense | 2.246 | 1.465 | 1.172 | 0.293 | 0 |
| 8 | 1024 | Sparse | 1.660 | 2.832 | 1.074 | 0.586 | 0 |
| 8 | 1024 | Rational | 1.953 | 1.270 | 0.977 | 0.195 | 0 |
| 9 | 512 | Dense | 3.418 | 3.027 | 2.930 | 1.074 | 0 |
| 9 | 512 | Sparse | 3.613 | 2.637 | 1.660 | 1.074 | 0 |
| 9 | 512 | Rational | 4.004 | 2.734 | 1.855 | 0.781 | 0 |
| 9 | 1024 | Dense | 1.953 | 1.074 | 1.074 | 0.879 | 0 |
| 9 | 1024 | Sparse | 1.953 | 1.855 | 1.562 | 0.879 | 0 |
| 9 | 1024 | Rational | 2.539 | 1.953 | 1.172 | 0.586 | 0 |

The aggregate decline follows the exact forecast, while individual source rates sometimes increase between consecutive ell values because the finite stochastic trials and source-specific changed sets are correlated. No claim of per-source sample monotonicity is made.

## Original immutable evidence

- **Source-only fixture:** 2,328,063 original bytes; SHA-256 2d660516122efee0211555fdb321b347da91a667bbb688cff6435b4267f3e4b6.
- **Complete results:** 472,774 original bytes; SHA-256 5c3597a9436cc33d6d9be81b1e53ad456b3b2b775872ab2d17c536972d88d428.
- The original files are now permanently source-controlled in [results/nor27_original_hosted.zip](results/nor27_original_hosted.zip), a **874,004-byte** ZIP whose two members match their first-run byte hashes. The read-only [original_archive_audit.py](original_archive_audit.py) is included in both CI verification jobs.
- First original GitHub Actions #44015771328, artifact **14298756859**, with both files as originally emitted. [Original immutable receipt](results/nor27_first_original_receipt.json) fixes all source-count, leakage, statistical and file-hash fields.
- The separate-process replayer is independent of the NOR-27 evaluator and directly reconstructs all public algebra inputs and the original seeded attack/schedule decisions.
- Every original source's honest exact target passed; all tested nonzero shifts +1 and +4294049777 generated the same exact acceptance decision.

The results are synthetic and depend on an external authenticated exact-value witness. The intended production security invariant is that attack locations become irrevocable before the unpredictable audit seed and remaining reserve coordinates are released. The raw-seed-bit experiment supplies empirical sensitivity under one deterministic research sampler; its bounds cannot certify a deployed PRNG or hardware entropy source.
