# NOR-35: exact within-atom curvature ranking adds detection on constructed high-curvature attacks

**Registered prospective verdict: PASS on all five frozen gates.** The first NOR35 registration was committed [3773c1b](https://github.com/MoralityLabAI/Compactification-Theory/commit/3773c1b1522f419e54f34252b0c743018105bc79) before NOR35 source, attack or outcome data. The first **valid complete independent replay** is [GitHub Actions #44064368383](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44064368383), with original source/output byte hashes pinned below. This report was authored after original data generation and does not change any registered budget, family or success threshold.

## Why this tests a different question from NOR-34

NOR34's eight tail/top-magnitude/lag atom categories accounted for almost the entire curvature-versus-hybrid detection difference across the ORIGINAL NOR25 auditors. NOR35 modifies the auditor architecture to use a curvature score *within* each atom, while holding exactly the SAME atom inspection counts t_j on every source/trial and using the same 16 exact coefficient witness queries.

There are **24 new source-distinct complete-public RRC25 problems**, replicates 48–51 at D512/1024 with three wide original input families, all full-public SHA disjoint from the **108 previously archived historical sources**. The public-only source fixture omits native generating program, private query output, attacked positions, auditor plans and honest target witness answers. All candidate query predictions freeze before private native reconstruction.

Four NOR25-plan transformations are assessed: original, atom-quota-matched random, atom_curvature50 (ceil(t_j/2) highest public cyclic second-difference positions inside atom j, with random remaining slots), and atom_inverse50 (lowest public curvature with equal slot allocation). This preserves *every original per-atom inspection count* and total k=16 for all six original NOR25 auditor components. The frozen NOR31 mixture remains [2,0,1,1,4,8]/16.

Original attack supports are public candidate-output functions, with no hidden true-output or target oracle access: 48 highest-curvature positions inside the old union and 48 outside; the analogous 48+48 LOWEST curvature positions; exact eight-atom-quota matched redraws induced from the high-curvature attack; and whole-output uniform radius-four attacks. Every family has 16 fixed four-coordinate attacks/source. All attacks freeze before 512 new audit schedules per source and arm. Curvature and inverse-curvature attacks condition on their own public score ranking; this is *source-structured* rather than a game-theoretic best response to the new auditor.

## Prospectively registered primary outcome

The PRIMARY calculates exact rational conditional detection under each attack law for each original audit plan and averages across 512 fixed plans, then across 24 source identities. It does not count 16 sampled attack draws or repeated challenges as independent statistical sources.

| Registered contrast | Equal-source mean gain | Source pairs in registered direction | Frozen threshold | Gate |
| --- | ---: | ---: | --- | --- |
| C: curv50 versus within-atom random, TOP curvature attacks | **+16.7277 pp** | **24/24** | ≥18/24 and ≥+1.0 pp | PASS |
| B: curv50 versus ORIGINAL hybrid_50, TOP attacks | **+16.7206 pp** | **24/24** | ≥18/24 and ≥+0.5 pp | PASS |
| I: curv50 versus atom random, BOTTOM curvature attacks | **−2.1357 pp** | **24/24** | ≥18/24 negative and ≤−0.2 pp | PASS |
| M: curvature-transformed NOR31 mix versus original NOR31 mix, TOP attacks | **+19.6156 pp** | **24/24** | ≥18/24 and ≥+0.5 pp | PASS |
| Exact quota/clean-target/independent replay | Exact parity | 24/24 | All checks pass | PASS |

### Mean detection rates under each source-conditional attack distribution

The entries are **percentage probabilities**, analytically integrated over fixed plans and ideal uniform attack sets within the registered supports, with each of 24 sources equally weighted.

| Attack law | Hybrid50 original | Hybrid50 atom random | Hybrid50 curv50 | Hybrid50 inverse50 | NOR31 original mix | NOR31 curv50 mix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TOP curvature 48+48 | 19.1482 | 19.1410 | **35.8687** | 12.7161 | 15.1957 | **34.8113** |
| BOTTOM curvature 48+48 | 3.1801 | 3.2453 | **1.1097** | 18.0437 | 4.0682 | 1.5831 |
| Eight-atom quota-matched | **19.1626** | **19.1626** | **19.1626** | **19.1626** | **15.2050** | **15.2050** |
| Whole-output uniform | **6.0925** | **6.0925** | **6.0925** | **6.0925** | **6.0925** | **6.0925** |

The exact eight-atom quota-matched result is **identical for all four transformations of each original policy**, down to the original Fraction numerator/denominator. Every full-output uniform attack law also gives the same exact mean for every k=16 auditor, as expected from the NOR32 uniform-attack duality. The new ranking concentrates inspections inside high-curvature regions by design, producing a large targeted gain and a measurable inverse-curvature cost.

### All 24 prospectively sampled source-level contrasts

Each value below is an exact conditional-attack source-mean rate DIFFERENCE in percentage points. The original registered C/B/I/M labels are retained.

| Rep | Degree | Source family | C: curvature vs random | B: curvature vs original | I: bottom-curvature cost | M: mix transfer |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 48 | 512 | dense | +13.811 | +13.885 | −2.456 | +18.073 |
| 48 | 512 | sparse | +12.861 | +12.644 | −3.069 | +17.038 |
| 48 | 512 | rational | +17.245 | +17.377 | −2.345 | +18.916 |
| 48 | 1024 | dense | +18.266 | +17.878 | −1.791 | +21.429 |
| 48 | 1024 | sparse | +12.232 | +12.108 | −1.396 | +17.930 |
| 48 | 1024 | rational | +18.835 | +18.398 | −1.708 | +21.052 |
| 49 | 512 | dense | +14.928 | +14.850 | −2.561 | +19.925 |
| 49 | 512 | sparse | +15.261 | +15.232 | −2.830 | +17.333 |
| 49 | 512 | rational | +17.144 | +17.470 | −2.974 | +18.880 |
| 49 | 1024 | dense | +18.465 | +18.992 | −1.466 | +21.047 |
| 49 | 1024 | sparse | +18.745 | +18.792 | −1.568 | +21.266 |
| 49 | 1024 | rational | +18.243 | +18.263 | −1.631 | +21.048 |
| 50 | 512 | dense | +19.625 | +19.177 | −2.501 | +21.782 |
| 50 | 512 | sparse | +12.553 | +12.667 | −2.292 | +16.230 |
| 50 | 512 | rational | +16.812 | +16.915 | −2.888 | +18.388 |
| 50 | 1024 | dense | +19.911 | +20.231 | −2.855 | +22.277 |
| 50 | 1024 | sparse | +12.965 | +12.866 | −1.507 | +18.761 |
| 50 | 1024 | rational | +18.820 | +18.551 | −1.349 | +21.047 |
| 51 | 512 | dense | +16.608 | +16.649 | −2.711 | +18.344 |
| 51 | 512 | sparse | +14.665 | +14.794 | −2.488 | +17.404 |
| 51 | 512 | rational | +17.006 | +16.836 | −2.505 | +18.577 |
| 51 | 1024 | dense | +18.761 | +19.078 | −1.637 | +21.271 |
| 51 | 1024 | sparse | +18.863 | +19.020 | −1.451 | +21.444 |
| 51 | 1024 | rational | +18.841 | +18.618 | −1.277 | +21.314 |

Descriptive source-block bootstrap 95% intervals over 20,000 draws (one source sampled as the block, deterministic seed 202611350099): C **[+15.72,+17.66] pp**, B **[+15.72,+17.65]**, I **[−2.37,−1.90]**, M **[+18.90,+20.31]**. These intervals concern variation across the 24 source identities under this specific synthetic family, rather than arbitrary policy-aware attackers. D512 and D1024 both have all source signs in the registered direction.

## Kernel mathematics, authentic witness and scientific boundary

[MathCan NOR35 PR #127](https://github.com/patrickdugan/MathCan/pull/127) passed Lean 4.35 kernel checks without admitted axioms, proving exact per-atom uniform-reserve miss counts under independent post-prefix sampling and exact equal conditional attack count pairs for equal atom inspection budgets. The new empirical study also checks the latter property on every original source/trial/arm transformed plan.

The authenticated target-output integer witness is external assumed information, reconstructed for verification only AFTER every public source/policy/attack decision freezes. All original sixteen-template accepted-program masks and in-family native query futures matched the independent generator. The fully informed attacker can still alter unchecked coordinates after seeing the realized schedule, escaping exact partial inspection. Public reproducible SHA-derived challenge streams are NOT cryptographic secrets.

The TOP attack support is explicitly based on the SAME observable curvature feature that the new auditor uses. Its gain demonstrates source-feature-aligned localization rather than robust protection against a policy-aware adversary who deliberately selects unseen or inverse-curvature coordinates. The −2.14pp bottom-support loss measures that tradeoff. Source-level inference uses 24 public cases, with the 16 attacks and 512 trials paired and correlated within each.

## First original source-locked receipts

First public-only 24-source fixture: **4,539,616 bytes**, SHA-256:

    7a517c8a9f87b15e12362da2dd99d188933a018a233fe2d043ea4aca9c4b02ef

First VALID complete original grade (with preregistered ascending-index LOW-curvature tie break): **706,150 bytes**, SHA-256:

    194a0709ed5672dbb8b6706b4d40d6f8f3a28e28164dc3275005d8421ccd05a6

The first earlier workflow snapshot [#44064001927](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44064001927) used a pre-result bottom-curvature tie ordering inconsistent with the preregistration, and its independent replay script had not yet been committed. Its numerical outputs are **not** the archived registered finding. The corrected ordering was committed before the first VALID result #44064368383, and the user-visible conclusion draws only from the independently replayed corrected source/grade bytes. First source fixture identity remained the same in both runs.

The exact corrected original source and grade are preserved in [results/nor35_original_hosted.zip](results/nor35_original_hosted.zip) with [first SHA-locked receipt](results/nor35_first_frozen_receipt.json). The original green [GitHub Actions #44064368383](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44064368383) independently reconstructed **24 native source programs**, **1,536 fixed attack sets**, **294,912 fully transformed audit plans**, **18,874,368 policy/attack/world decisions**, and **2,304 authentic integer-witness comparisons**. Permanent read-only CI pins both original SHA identities and repeats the independent re-evaluation.
