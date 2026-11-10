# RRC-24: Late-divergence symbolic search and learned base-order ranking

**Executed 2026-11-10, CPU-only, exploratory single training seed.** An exact
formal-series grammar of 132 spellings creates 32 indistinguishable delayed
factor variants below degree 64 for each of four base operations. Three small
CPU models learn to order *the four base operations*, with a shared exact
solver recovering the delayed factor at test time. Every candidate is checked
against two public degree-192 demonstrations; independent query outputs through
degree 256 provide future grading.

## Primary 240-case test, two full base checks

| Source family (40 each) | GRU, 609 weights | FF, 588 weights | Shuffled GRU | Static base order | Exact short-prefix order |
|---|---:|---:|---:|---:|---:|
| Signed, training family | 35 | 22 | 22 | 26 | 40 |
| Partition products, training family | 40 | 40 | 19 | 21 | 40 |
| Sparse signed, training family | 33 | 32 | 15 | 18 | 40 |
| Rational reciprocal, unseen | 27 | 31 | 14 | 22 | 40 |
| Convolution-squared, unseen | 35 | 27 | 16 | 19 | 40 |
| Late-onset input, unseen | 25 | 25 | 15 | 17 | 17 |
| **All 240** | **195** | 177 | 101 | 123 | **217** |

These are **future-correct, independently verified executable programs**;
all accepted normal positive cases predict the withheld degree-256 query
correctly. The GRU's held-out two-query score exceeds static ordering by
72 programs and the matched feedforward control by 18. A deterministic
short-degree exact ordering wins by 22 programs and also uses less counted
work. Allowing four complete factorized checks yields **240/240** for the GRU
and for exhaustive symbolic search. The GRU top-one ordering gets **137/240**.

The exact short-prefix sorter executes four public unary operators on **both**
demonstration prefixes through degree 16, prior to its two full macro checks.
These eight short computations are additional to its full candidate budget
and included in its work ledger. The trained neural model does not invoke an
interpreter to rank candidates. All ranking policies subsequently rely on
**the same privileged factorized exact suffix solver**, which finds the
first residual coefficient and reconstructs the delay and amplitude.

## Algebra and why the short-prefix tie matters

The DSL is \(G(F)=O(F)(1+cq^K)\), with
\(O\in\{I,N,R,S\}\), eight allowed delays from 64 to 176 in steps of
16, four signed nonzero amplitudes, and a null suffix. For fixed \(O,F\),
all 32 delayed choices agree through degree 63. Exactly at their delay
\(K\), the first nonzero coefficient in \(G-O(F)\) is \(c\). The teacher
therefore supplies an exact algebraic decomposition instead of having the
model search over all 132 literal spellings. The solver validates that the
inferred \(K,c\) belong to the declared grammar.

The `late_source` family contains only zero coefficients through degree 16
beyond the constant unit, so every base and every delayed factor gives the
**same neural input**. For a uniform hidden base prior, any fixed top-two
policy has theoretical expected success \(2/4\). The sampled 40-case
values range between 15 and 25 successes across neural/static controls;
they do not establish learned semantics from absent evidence. This family
explains the residual error in the otherwise perfect exact-prefix sorter,
which gets 200/200 on the five other positive families.

### Complete literal-program search comparison

A separately selected twelve-case diagnostic runs the **literal 132-program
syntax catalogue** with paid repeated verification at degrees 16, 64, 128
and 192. Its first twelve syntax candidates recover **1/12** cases.
The complete 132-candidate scan recovers **12/12**, using **956 literal
candidate attempts** and 3,490 two-demonstration stage executions.
The factorized four-base solver recovers all twelve by inferring \(K,c\) after
each tested base. These are *different kinds of computational access*: one
macro check invokes a supplied algebraic suffix inversion. The benchmark
records candidate counts, full base executions, work proxies and the
literal search's repeated staged executions separately.

## Resource ledger

| Method, 240 positive cases | Future correct | Mean paid full base checks | Mean exact-work proxy /case | Dense model MACs /case |
|---|---:|---:|---:|---:|
| Learned GRU, two checks | 195 | 1.429 | 10,059 | 8,459 |
| Matched feedforward | 177 | 1.442 | 10,379 | 576 |
| Shuffled-label GRU | 101 | 1.717 | 24,296 | 8,459 |
| Static two-base order | 123 | 1.754 | 2,031 | 0 |
| Exact degree-16 prefix rank, two full checks | **217** | **1.117** | **7,898** | 0 |
| Exact degree-16 prefix rank, up to four checks | **240** | 1.267 | 11,586 | 0 |
| Exhaustive four-base factorization | **240** | 2.525 | 20,801 | 0 |

The work proxy includes reciprocal triangular sums and linear shift/residual
work; its units differ from neural floating-point MACs. The very cheap
static order often tries only `I` and `N`, sacrificing accuracy. This
is one reason total work cannot be compared without including recovery.

### Actual CPU timing on a separate 36-case sample

Two warm, shuffled passes per arm, 72 measurements, Python 3.13.5 on AMD EPYC
9V74. Every method uses its production portable-Python inference and the
same exact public demonstration verifier. Cold training, case generation,
future grading and setup are excluded.

| Method | Warm median milliseconds/case | Correct across 72 executions |
|---|---:|---:|
| GRU, two checks | 0.860 | 64 |
| Feedforward, two checks | 0.103 | 54 |
| Static two-base order | 0.039 | 36 |
| Exact prefix rank, two checks | **0.096** | 60 |
| Exact prefix rank, four checks | **0.113** | **72** |
| Exhaustive four-base factorization | 0.431 | **72** |

The exact four-check prefix rank reaches every sampled future correctly
with a median **7.6× lower** than the two-check learned GRU. Median times
alone do not represent full latency distributions; sums, p90 and separate
route/verification stages remain in the machine-specific timing JSON.

## Robustness and controls

Two *post-primary*, frozen-weight 144-case seed checks give:

| New batch | GRU, 2 checks | FF, 2 checks | Static, 2 checks | Exact prefix, 2 checks | Factorized four-base |
|---|---:|---:|---:|---:|---:|
| Seed 2026127024 (144) | 123 | 119 | 72 | 135 | 144 |
| Seed 2026138024 (144) | 113 | 99 | 67 | 132 | 144 |
| **Both (288)** | **236** | 218 | 139 | **267** | **288** |

These seeds were added after the primary score was observed and are
therefore robustness observations, without independent preregistration.

All **12 intentionally inconsistent** public demonstration pairs are
rejected by complete four-base search. Eight identical observed-prefix
base/future-mutant pairs are constructed with a hidden \(q^{224}\) factor.
Both members of each pair induce identical predictions and interpreter traces.
Complete search accepts 16 visible prefixes, predicts exactly eight futures
correctly and incurs eight unavoidable future errors. Finite degree-192
verification establishes equality on the observed inputs only.

Training uses 960 examples across three families, 192 development examples,
and 340 Adam updates of batch size 96 for each of three models. The GRU has
609 trainable FP32 weights and 11 live recurrent floats; the feedforward
control has 588 weights, a 3.6% smaller checkpoint. The shuffled GRU has
609. Training supervises only the **base token**. The fixed graph interpreter,
coefficient data, training labels, symbolic suffix solver, and all inference
runtime memory are separate costs. The first training run and an independent
second run export byte-identical checkpoint JSON files.

## Reproduce and decision

The 20 RRC-24 unit tests pass. Standard-library frozen inference matches
PyTorch's GRU and feedforward outputs within 2e-5 on the tested cases.
RRC-22/RRC-23 suites contribute 30 passing regression tests. The complete
source-bound deterministic receipt includes all primary cases, literal
search traces, two post-primary seed checks, test-source locks, checkpoint
hashes and collision invariants. The measured timing JSON is separate and
is never required to match physical times in CI.

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_delay24.py' -v
python research/rrc_compactification/bench24.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_scale23.py' -v
python -m unittest discover -s research/rrc_compactification -p 'test_meter22.py' -v
```

**Decision:** Learned candidate ordering improves over the fixed syntactic
prior on this chosen long-prefix benchmark, including on two shifted input
families. A mathematically informed exact short-prefix search retains better
accuracy and lower observed CPU latency, while an exact q-adic factorization
largely eliminates the large delayed-suffix syntax search. This specific
benchmark supports further work on harder, nonfactorable aliases rather than
another scaling run of the same hand-supplied suffix solver.
