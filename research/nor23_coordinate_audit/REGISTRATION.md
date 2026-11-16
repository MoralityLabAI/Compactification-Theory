# NOR-23 registered protocol: randomized exact coordinate audits

**Registered 2026-11-16 before generation of fresh NOR-23 public sources, hidden target alterations, randomized coordinate challenges or outcome data.** Continuation of merged [NOR-21 #185](https://github.com/MoralityLabAI/Compactification-Theory/pull/185) and [NOR-22 #187](https://github.com/MoralityLabAI/Compactification-Theory/pull/187). Companion formal work will be in MathCan.

## Claim and scope

NOR-21 demonstrates public-only indistinguishability of two target futures, and NOR-22 tests target-dependent modular fingerprints from a trusted publisher (including explicit p and p*q collisions). NOR-23 gives a trusted external witness permission to reveal **exactly k target OUTPUT coefficients** after a public-only candidate future has been frozen, so as to test the coordinate-information budget. These values are an **additional trusted information channel**. An audit pass at k<m means compatibility on sampled positions, **not** complete future correctness, cryptographic authentication, or unrestricted theorem-prover soundness.

This is a controlled synthetic CPU study with a fixed 16-program RRC25 grammar and independently generated public input/output examples. No neural training. Preserve NOR-21's streamed exact public checker and original interpreter semantics without modification.

## New public sources and sealed worlds

* **12 fresh distinct complete-public sources**: two replicate cohorts 0,1 × observation degrees D∈{512,1024} × original NOR-21 families {wide_dense_small,wide_sparse_small,wide_rational_short}. Query input/output degree Q=3D/2, with m=Q+1 coefficients. Two full D-degree demonstrations per source.
* New generator seed: original RRC25 SEED + **94323000** + 1000003*rep + 100009*degree_index + 10007*family_index. Sample the first case (up to 100 attempts) with a uniquely inferred public header, a nonempty exact 16-bit program mask, and a **single shared hypothetical full query output** among accepted programs. Require 12 different full public digests. Source selection may use public compatibility and hypothetical candidate predictions only.
* Public fixture records demo coefficients, query input, header, source attempt, exact public mask and public-input digest. **No private planted program or true target output**, no altered target coordinates, and no witness answers appear in the public source fixture.
* Freeze a full public-only candidate prediction per source using NOR-21 stream16. The sealed native generator reconstructs the planted program and exact true target **only after the public proposal has been fixed**. The valid world and perturbed worlds share the same public observation and same candidate prediction.
* Precommitted, **blind** attack coordinates for each source and r use a distinct PRNG stream seeded by SHA-256 of registered ATTACK_SEED=202611230071, public digest, r. They are drawn independently of the later audit challenge. Attack radii r∈{1,4,16} and shift magnitudes δ∈{1, 4294049777} (NOR-22 p*q), with shifts applied to r distinct true-future output positions.
* Challenge budgets **k∈{0,1,4,16,64}**. Generate **4096** uniformly sampled k-subsets (without replacement) independently per distinct source and k, using SHA-256-derived deterministic CHALLENGE_SEED=202611230072 and trial ordinal. Reuse identical challenges across r and δ for paired comparisons. These seeds make CI replay deterministic; the *blind* threat model requires attack positions fixed before challenge sampling. Known code or seeds do not constitute cryptographic secrecy.
* The **adaptive informed adversary** instead sees each challenge set and then selects the first r unchecked coordinates, which exists because k+r≤80<m. It alters the actual hidden target only after seeing the audit set. This is a separate control with a stronger adversary, not a simulation of an unknowable future.
* Trusted witness: on the selected k indices, return the **actual hidden target's exact integer coefficients** to the verifier. Compare them to the frozen candidate at each sampled position. Reject/abstain on any mismatch. k=0 returns the unchanged public-only answer. All targets outside NOR-21's planted program family are labeled accordingly.
* Authentic witness availability, correctness and timing are **explicit assumptions**. If the witness is forged, arbitrary answers can make any audit pass; we do not claim to establish witness trust.

There are 12 independent source identities, 3 blind radii, 2 shift magnitudes, 5 budgets and 4096 deterministic challenge draws per source/budget, with both threat models. Do not report the many paired challenge/world decisions as independent source samples.

## Frozen mathematical expectation and checks

Let m=Q+1, r changed output coordinates, and k sampled distinct indices.

* Under independent uniform k-subset sampling **after** a fixed r-subset alteration, exact miss probability is C(m-r,k)/C(m,k) (zero if k>m-r); detect probability is one minus this ratio. The formula is independent of the nonzero integer shift magnitude.
* With known challenge positions and r≤m-k, an adaptive attacker can choose only unchecked positions: zero detections and 100% false acceptances on altered worlds.
* Valid futures must pass 100% of audits. Public-only candidate and mask must stay identical across target worlds and audit budgets.
* k=m would detect every nonzero alteration and establish equality of the entire target sequence under an honest witness. It is a mathematical boundary, not measured by the registered k≤64 experiment.
* For each (degree,k,r,δ), calculate per-source successes, degree-pooled detection rate over 6×4096 challenges, exact hypergeometric prediction, and raw counts. A 5-standard-error tolerance plus 0.015 absolute allowance is used as a deterministic sanity check for the blind Monte Carlo; record all misses and any failure. Adaptive false acceptance must be exact. No significance tests treat challenge draws as independent source examples.
* Measure per-source cumulative challenge+comparison CPU nanoseconds using CPython 3.12 in GitHub Actions, report paired k-to-k differences and amortized cost per audit. Python time is an observed engineering metric without a preregistered speed threshold.
* All decisions are source-locked with SHA-256 of the public fixture and result; a separately implemented replay auditor must rederive candidate choices, attacks, sampling and truth checks from the original RRC25 interpreter (and public source generator).

## CI and outcome discipline

Unit tests must cover all k-subsets exhaustively on small finite m, adversary timing, unseen-coordinate evasion, valid-witness pass, δ-independence, public/hidden isolation and full-source candidate parity. CI must run the full 12-source experiment under a **40-minute** budget and an independent replay audit; archive the original outputs before writing interpretations. If a run exceeds the budget or a source fails to generate, record the failure instead of lowering sample counts or changing seeds. Any change to this frozen protocol requires a dated amendment committed before new held-out data.
