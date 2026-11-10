# RRC-17: topology-conditioned candidate quotient and exact budget audit

**CPU study executed November 10, 2026.** One source-locked semantic-probe design and 336 fresh positive cases across fork, cascade, additive and Hadamard graph wiring, plus 16 observed-invalid negatives and eight late future-collision pairs. Three further 128-case Hadamard seed checks were executed with frozen rules and frozen neural checkpoints. No training occurred in RRC-17.

## Twelve-query comparison on 336 positive cases

| Method | Accepted /336 | Mean two-demo graph queries | Mean nominal integer work /case | Dense neural MAC proxy |
|---|---:|---:|---:|---:|
| **Twelve largest probe classes** | **247** | 7.04 | 1,303.6 | 0 |
| Class mass divided by graph work | 246 | 7.15 | 1,268.0 | 0 |
| First 12 original program spellings | 160 | 9.35 | 1,697.4 | 0 |
| Frozen RRC-16 recurrent | 207 | 7.35 | 1,506.1 | additional neural work |
| Frozen RRC-16 feedforward | 209 | 7.55 | 1,527.5 | additional neural work |
| Untrained residual frontier | 174 | 9.07 | 1,803.3 | frozen initializer |
| Complete probe quotient, 24–33 calls maximum | **336** | 8.77 | 1,820.7 | 0 |
| Complete 64-program catalogue | **336** | 17.63 | 3,635.0 | 0 |

The 12-class prior heuristic outperforms the frozen neural policies on the combined test without any new training, but consumes a **shared offline interpreter-derived quotient**. This experiment strengthens the exact search baseline; it does not demonstrate a recurrent-model advantage. The full quotient comparator matches the 64-program solver throughout the positive test at approximately half the online graph queries and nominal integer operations. The theoretical classes remain probe-inferred, with final candidate acceptance independently checked by exact integers.

## Previously withheld Hadamard join, new instances

| Method | Uniform signed (96) | New-join, old-grammar-inexpressible (48) | Product sources (48) | Total /192 |
|---|---:|---:|---:|---:|
| Twelve largest classes | 75 | 37 | 36 | **148** |
| Class mass per work | 75 | 37 | 37 | **149** |
| Fixed first 12 | 56 | 22 | 25 | 103 |
| Frozen GRU | 56 | 29 | 21 | 106 |
| Frozen feedforward | 56 | 28 | 23 | 107 |
| Untrained residual frontier | 39 | 26 | 25 | 90 |
| Complete quotient / exact 64 | 96 | 48 | 48 | 192 |

The Hadamard 12-class order uses the Hadamard *public interpreter*, supplied at evaluation time. Its classes are computed from four random synthetic sources rather than from these held-out targets. The 48 Hadamard novel cases were explicitly selected for absent old-topology aliases; they therefore have a different prior distribution from uniform syntax.

## Three additional independent Hadamard seeds

| Seed | Largest 12 classes | First 12 spellings | Frozen GRU | Frozen FF | Complete exact |
|---|---:|---:|---:|---:|---:|
| `2026115017`, 128 cases | 85 | 67 | 61 | 67 | 128 |
| `2026116017`, 128 cases | 95 | 71 | 60 | 67 | 128 |
| `2026117017`, 128 cases | 89 | 65 | 59 | 53 | 128 |
| **Combined 384** | **269** | **203** | 180 | 187 | 384 |

These were added after the initial design and first positive result. The class partition and order remained unchanged, and the additional instances never influenced training or method selection. They support a limited robustness claim over fresh random sources and seeds.

## Controls and resource ledger

- All 16 deliberately inconsistent Hadamard observed inputs are rejected by both complete solvers, and every checked method abstains after failed exact checks.
- Each of eight paired base/future-mutant controls has identical public inputs, query and candidate trace. The complete 64-graph and complete quotient methods accept all 16 observed prefixes, make eight correct future predictions, and make **eight false future acceptances**.
- Probe partition counts are 25 (fork), 24 (cascade), 33 (additive), and 24 (Hadamard). Four independent field probes per program, plus two independently seeded stress probes and integer-source comparisons, yielded no observed false class merges. These samples are *not* a universal symbolic proof of equivalence.
- Constructing the shared quotient costs **1,024 graph executions on public synthetic inputs**, or **143,104 nominal integer-loop operations**, before the first case is evaluated. That is about **389 extra integer-loop units per case** if amortized over the 368-case primary roster, or 190 over all 752 evaluated cases including the 384 independently seeded checks. Storage of the interpreter, classes, probes, input coefficients and code is separate. A change to the interpreter requires rebuilding the quotient.
- Exact inference queries execute each candidate on two observed demonstrations. Each query costs eight primitive applications. Frozen neural controls also perform dense MAC operations and retain previously trained weights and teacher-supervised training privileges. Integer arithmetic and FP MACs are not converted into one invented performance metric.

The method has a sharply restricted optimality statement: with a uniform prior on the 64 syntax programs and *Boolean class membership feedback only*, testing the twelve classes with greatest syntax mass maximizes the prior success probability. The real residual interface is richer than Boolean feedback, and the test source distribution includes class aliases, so this statement does not bound general interactive search policies. Random-probe aliasing is governed by a conditional Schwartz–Zippel bound; see [RRC17_THEORY.md](RRC17_THEORY.md).

## Audit and decision

The 17-test dedicated suite checks class coverage, independent semantic probes, future blindness, exact paid-call counts, evidence boundaries, negative cases, collisions and receipt integrity. The frozen previous checkpoint digests remain unchanged. `bench17.py --check` recomputes all 368 primary outcomes, the three 128-case replications, the quotient partitions and complete method score tables, then compares its output byte-for-byte with the checked JSON receipt. Regressions for RRC-15/16 are part of the lightweight workflow.

**Next scientific gate:** use symbolic normal forms or a proof assistant to certify equivalence classes rather than relying on random probes; only then train a model to predict the equivalence class *without invoking the public interpreter on 1,024 synthetic examples*. Compare total training, compilation, lookup, runtime and exact-verifier work against the quotient method established here.
