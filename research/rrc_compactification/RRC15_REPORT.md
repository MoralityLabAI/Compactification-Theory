# RRC-15: frozen learned search transfer to new graph wiring

**Result, November 10, 2026.** The 7,353-weight RRC-14 recurrent edit controller was applied with **zero weight updates** to new cascaded-branch and additive-join formal-series graphs. The public interpreter changed, while the three two-bit program slots, 12 full candidate-query budget and model observation sizes stayed fixed. All reported program acceptances passed independent exact integer coefficient checks.

## Main held-out transfer

Each case supplies two public input/output demonstrations through degree 12 and a query input. Later query-output coefficients through degree 32 are held back. The following 144 examples are selected to be *inexpressible by all 64 original fork-join graphs at the observed prefix*, so they measure genuine finite-class semantic transfer.

| Method | Cascade novel signed /48 | Cascade product /24 | Additive novel signed /48 | Additive product /24 | Total /144 |
|---|---:|---:|---:|---:|---:|
| Frozen recurrent 12-query edits | **27** | 7 | 25 | 12 | **71** |
| Same GRU, hidden state reset | 8 | 2 | 26 | 10 | 46 |
| Frozen matched feedforward edits | 15 | 1 | 21 | 12 | 49 |
| Untrained residual frontier | **27** | **13** | 23 | 11 | **74** |
| Visible-seeded random frontier | 16 | 8 | 19 | 10 | 53 |
| Fixed first-twelve graph catalogue | 0 | 0 | 21 | **14** | 35 |
| Exhaustive exact 64-candidate solver | 48 | 24 | 48 | 24 | 144 |

Every accepted program in the four positive transfer groups also predicts the held-out degree-32 continuation exactly. The exact-64 solver's 144 successes exceed the registered 12-call budget; this row is an upper control. All methods use the same new interpreter semantics when searching and verifying.

**Recurrent-state dependence is topology-specific.** On 72 cascade-novel examples, the frozen GRU reaches 34/72 with history versus 10/72 when state is reset, and the matched feedforward reaches 16/72. On 72 additive examples, history makes little difference (37/72 versus 36/72). The untrained residual frontier leads the combined table, 74/144, at lower counted compute. Consequently the proposed recurrent-superiority gate **fails**, despite substantial transfer and a causal history ablation on cascade wiring.

The paired case audit, aggregated across the 144 novel cases, records 34 recurrent-only and 9 reset-only acceptances (37 accepted by both), and 31 recurrent-only versus 9 feedforward-only acceptances. Against the untrained residual frontier the overlap is 41 successes; recurrent-only and residual-only cases number 30 and 33 respectively. These are exact within-roster comparisons, without inferential significance claims beyond this selected single seed.

## Controls and scope

| Split | Cases | Recurrent verified | Feedforward verified | Untrained residual verified | Full exact verified |
|---|---:|---:|---:|---:|---:|
| Fresh original fork | 32 | 26 | 20 | 16 | 32 |
| Cascade graphs also expressible by old fork | 24 | 13 | 16 | 14 | 24 |
| Negative observed-prefix mutations, two topologies | 20 | 0 | 0 | 0 | 0 |
| Paired future-collision samples, two topologies | 24 | 8 | 12 | 10 | 24 |

The 24 collision items comprise 12 base/mutant pairs. Full exact search accepts all 24 observed prefixes, predicting 12 correctly and making 12 unavoidable false future acceptances. The frozen recurrent policy accepts eight collision prefixes and predicts four correctly; matched visible inputs yield identical search traces in every arm. Exact rejection of the 20 negative samples proves unrepresentability only within this declared finite 64-graph language and observation horizon.

## Resource ledger on the 144 novel examples

| Policy | Mean graph queries | Mean nominal integer work /case | Mean dense model MACs /case |
|---|---:|---:|---:|
| Frozen recurrent | 9.38 | 2,184 | 77,363 |
| Frozen GRU, state reset | 10.01 | 2,469 | 83,201 |
| Matched feedforward | 9.77 | 2,375 | 82,081 |
| Untrained residual | **8.40** | **1,890** | **12,936** |
| Visible-seeded random | 9.70 | 2,534 | 12,936 |
| Fixed first 12 | 10.50 | 2,359 | 0 |
| Complete 64-graph search | 28.34 | 7,387 | 0 |

Each edit candidate query executes both degree-12 demonstrations, charging eight primitive applications. The integer work proxy counts the particular unary nodes and binary join, so cascade and additive queries have different costs. Model MACs include the common frozen RRC-13 initializer (12,936 dense MACs per case), except for the uninitialized catalogue/exhaustive controls. These arithmetic proxies omit actual big-integer bit cost, nonlinearities, cache/heap operations and elapsed time; the integer and floating-point units are separately reported. The frozen GRU also carries its prior 7,353 trained parameters, the 7,393-parameter feedforward control and 6,148-parameter common initializer. This evaluation trained no new network; RRC-14's 35,631 teacher neighbor queries remain a prior training resource.

The test strata are selected by observed-prefix old-fork equivalence, which gives a strong within-task semantic shift while conditioning the case distribution. The original RRC-14 test seed and grammar were known, and this result is a single exploratory instance seed. No model architecture search or held-out tuning was performed after observing these scores.

## Audit

- The mathematical note proves the unit-series closure, describes the multiplicative endomorphism aliases, and gives the deterministic feedback and collision limits.
- The 16-test new suite checks independent old-fork parity, new operator semantics, source separation, actual interpreter-call count, policy blindness to hidden/future data, exact candidate budgets, reversible edits, negatives, collisions, source hashes and frozen checkpoint locks.
- A full standard-library replay regenerates the source-bound receipt and all trajectory hashes. RRC-13/RRC-14 code and checkpoints are unchanged, so their existing regression checks remain available.

**Next direction:** Train a topology-conditioned controller using a strictly separate set of wiring families, then assess transfer on genuinely unseen joins with equal source rosters and a blind split. Exact dynamic-programming and public-interpreter control arms should remain the resource baseline; a stronger learned result must beat residual search under the same graph queries and total work accounting.
