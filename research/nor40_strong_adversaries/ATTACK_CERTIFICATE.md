# NOR-40 attack optimization certificate and threat-model interpretation

This is a mathematical and methodological note written after the original prospective [REGISTRATION.md](REGISTRATION.md) commit `dbc1adbe0e30b9c0e36421cf07cebcf376d6a1ec`. It does not amend any experiment budget, source, attacker algorithm, control or science gate.

## Objective and witnesses

For a fixed source and a fixed target auditor distribution, let `M` be the future output-coordinate count (769 or 1537), `T_{b,i}` be the set of **CAL challenge IDs** where component branch `b` inspects coordinate `i`, and `w_b ≥ 0` the exact integer policy weight, with `Σ_b w_b=256`. For any distinct four-coordinate attack support `A`, define

```
J(A) = Σ_b w_b * | union_{i∈A} T_{b,i} |
```

`J(A)` is an integer count of component-weighted CAL schedules that detect a precommitted attack. A lower `J` is a more evasive CAL-trained attack. EVAL schedules are generated separately and are not used anywhere in this function.

**Singleton values.** Define `s(i)=Σ_b w_b |T_{b,i}|` and let `s_(4)` be the fourth-smallest value among the full `M` singleton values, counting distinct coordinate indices and breaking ties by index.

For any four-distinct `A`, each component's union contains `T_{b,i}` for every `i∈A`. Every weight is nonnegative. Consequently

```
J(A) ≥ s(i) for every i∈A
J(A) ≥ max_{i∈A} s(i) ≥ s_(4).
```

Hence `L=s_(4)` is a **sound global lower bound** on the optimum over all `binom(M,4)` attack supports. If `U=J(A_best)` is the best scored feasible support, then

```
L ≤ min_{A:|A|=4} J(A) ≤ U.
```

The gap `U-L` is an upper bound on how far `A_best` could lie above the true global *CAL* minimum. It may be loose. Reported `U` is not called a proof of global optimality. If an improved global lower bound becomes available from a genuine full-domain MILP/LP solver, it requires a separate prospective protocol or explicitly post-hoc qualification.

## Restricted exact 16-coordinate enumeration

The public CAL-only candidate pool consists of all four coordinates in the best of 32 full-domain greedy restarts, plus the lowest singleton-exposure other coordinates in ascending `(s(i),i)` order until 16 distinct indices have been selected. The evaluator enumerates all `binom(16,4)=1820` subsets and selects the exact least `(J(A), lexicographically sorted A)`. This is a **complete certificate for the 16-coordinate pool**. Since each pool candidate is a feasible full-domain attack, the unrestricted optimum is no worse than the restricted optimum, which itself is no better than the best attack found after CAL-only strong candidate selection.

The eight final strong attacks are the eight best distinct candidates from eight CAL256/8 baseline greedy attacks, 32 CAL2048 greedy restarts, eight locally refined starts and the exact restricted optimum, repeating the best candidate when fewer than eight unique supports exist. Therefore the mean **full CAL objective** of the final strong eight can only decrease relative to the eight baseline attacks: they are in the candidate pool, whose best eight are selected by the same full CAL score. This guarantee is **not a theorem that EVAL detection improves**, since the CAL scores are only an empirical estimate of the defender's true random policy.

## Why the independent EVAL gate matters

Overfitting can reduce `J(A)` on 2048 CAL schedules while losing detection evasion on a fresh 512-schedule EVAL draw. Gate 1 uses the EVAL score itself, requiring the strong global7 attacker to lower detection by at least **0.1 percentage points on average** with at least 24 of 48 distinct original source identities improving. A failure is retained even when CAL dominance and all bracket checks pass.

Relative to a pure uniform16 full-output k16 distribution, NOR40's seven defenders retain the exact full-output unrestricted r4-support minimax benchmark:

```
u(M,16,4)=1-C(M-16,4)/C(M,4).
```

The conditional guaranteed uniform-exploration contribution of `global7_mix` is `(7/16)u` for every attack fixed before an independently hidden audit draw, provided the externally authenticated true-target witness is available. In contrast, a schedule-informed attacker can choose four of the unchecked output coordinates and make the authentic queried transcript identical to the original.

## Statistical unit

Each public source identity has one learned attack game. The 8 CAL baseline starts, 32 full-CAL starts, local swap trials, candidate-pool subsets and 512 EVAL schedules are algorithmic observations within the same source. NOR40 has 48 source-level registered comparisons, nested in eight replicate blocks each containing six related degree/family cells. The replicate-block intervals and attack uniqueness are descriptive. Any optimally selected or retrospectively retuned attack family using EVAL violates the prospective separation and requires an independent fresh cohort.

## Lean companion

The [merged MathCan PR #141](https://github.com/patrickdugan/MathCan/pull/141), commit [`61d653c`](https://github.com/patrickdugan/MathCan/commit/61d653c5bf9d1a4e1c9483a9c0c21b0677513869), kernel-verifies the four-distinct-rank pigeonhole statement and the fourth-smallest singleton lower bound under sorted-rank and nonnegative union-exposure hypotheses. It also verifies nonnegative integer policy-weight monotonicity. The full-domain CAL optimum bracket continues to have a potentially nonzero gap.
