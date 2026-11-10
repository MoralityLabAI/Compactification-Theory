# RRC-18: exact finite algebraic quotient and supervised class prediction

**Executed November 10, 2026.** A deterministic polynomial normalizer
certifies the previously probed classes through degree 12, and three
CPU-trained models propose exact quotient representatives from paired raw
coefficient demonstrations. All candidate outputs are verified with a
separate integer interpreter. The result shows strong *within-source*
class ranking by recurrence and weak source-family transfer.

## Mathematical certificate

| Topology | Exact degree-12 classes | RRC-17 randomized probe agreement | Graph spellings |
|---|---:|---|---:|
| Fork | 25 | exact partition match | 64 |
| Cascade | 24 | exact partition match | 64 |
| Additive | 33 | exact partition match | 64 |
| Hadamard | 24 | exact partition match | 64 |

The exact normalizer operates on `Z[x_1,...,x_12]` coefficient
polynomials. It certifies class equality for **all integer input series
through degree 12**, independent of finite-field random probes.
There are 256 public graph/topology symbolic normalizations per certificate
construction. The generated polynomial vectors together contain 48,556
monomial entries (representation-size proxy). The RRC-17 comparison
independently evaluates 1,024 public finite-field probe graph executions.
Tests confirm concrete equality on 512 additional random integer series
and reject a known different graph. The proof scope is finite; no global
identity over every coefficient or Lean kernel proof is claimed.

## Measured held-out results, 12 exact checks per case

| Direct ordering | Signed sources /240 | Product-shift /160 | Total positive /400 |
|---|---:|---:|---:|
| Recurrent 6,240-param GRU | **233** | 92 | **325** |
| Feedforward 6,235-param MLP | 205 | 94 | 299 |
| Shuffled-label GRU | 168 | 90 | 258 |
| Certified quotient, descending class size | 155 | **110** | 265 |
| First-12 syntax catalogue | 122 | 85 | 207 |
| Exhaustive certified quotient (more calls) | 240 | 160 | 400 |

The recurrent model's unverified top-choice output alone passes exact
checks on **86/240 signed** and **18/160 product** cases. Ranking up to
12 representatives raises those rates to 233/240 and 92/160 respectively.
All trained methods see the **same public** paired input/output series and
topology label. Neural models use no candidate interpreter previews to
rank programs; exact execution starts after the list has been fixed.

### Per-topology recovery at budget 12

| Topology | Signed GRU /60 | Signed FF /60 | Signed size prior /60 | Product GRU /40 | Product size prior /40 |
|---|---:|---:|---:|---:|---:|
| Fork | 60 | 54 | 40 | 18 | 29 |
| Cascade | 59 | 53 | 41 | 32 | 35 |
| Additive | 55 | 41 | 31 | 17 | 20 |
| Hadamard | 59 | 57 | 43 | 25 | 26 |

Paired case outcomes sharpen the same comparison. On signed-series cases,
the recurrent ranking succeeds where feedforward fails **31** times, with
**3** feedforward-only successes; against class-mass ordering the counts
are **82** recurrent-only and **4** mass-only. On product sources the
recurrent/feedforward exclusive outcomes are **21 versus 23**, and the
recurrent/class-mass exclusive outcomes are **21 versus 39**. Both methods
are graded on exactly the same instances and exact interpreter.

The product-source shift is the critical negative control: the GRU's
92/160 approaches the shuffled-target model's 90/160. The size-ordered
certified quotient performs better (110/160) with zero neural inference.
This single seeded pilot therefore supports **in-distribution learned
ranking** and fails the stronger distribution-shift recurrence criterion.
The model did not invent operators, class semantics or an interpreter.

## Exact verification and uncertainty

All 20 constructed inconsistent demonstration cases are rejected by every
12-query and exhaustive arm; each reported abstention follows an exact
checker rejection. Eight base/mutant pairs have equal observed series,
topology and query input but differ at a hidden future coefficient. The
complete certified quotient and the complete 64-syntax search accept
all 16 visible cases and predict exactly eight futures correctly, with
eight false future acceptances. Every deterministic candidate trace is
identical within the corresponding pair. Finite coefficient verification
cannot certify an infinite formal-series identity or a withheld future
continuation.

## Measured inference cost, same exact query budget

| Arm, signed cases | Mean exact graph queries | Mean nominal integer work | Dense forward MACs / case |
|---|---:|---:|---:|
| Recurrent top-12 | 3.5125 | 917.58 | **53,216** |
| Feedforward top-12 | 4.5958 | 1,167.29 | **6,120** |
| Certified class-mass prior | 7.4833 | 1,599.22 | 0 |
| Complete certified quotient, larger budget | 10.0125 | 2,362.53 | 0 |

| Arm, product-shift cases | Mean exact graph queries | Mean nominal integer work | Dense forward MACs / case |
|---|---:|---:|---:|
| Recurrent top-12 | 8.4188 | 2,148.74 | **53,216** |
| Feedforward top-12 | 8.1688 | 1,977.79 | **6,120** |
| Certified class-mass prior | 7.3813 | 1,563.09 | 0 |

Each graph query evaluates **both** observed degree-12 demonstrations and
charges the appropriate topology's unary and join coefficient-work proxy.
A GRU's 900 Adam updates, minibatch 128, process at least 115,200
forward training-example visits, with **6.13 billion estimated forward
MACs** excluding backpropagation; the FF uses approximately 0.705
billion. The shuffled control adds another 6.13 billion forward MACs.
These are counted neural operations and exclude PyTorch's backward pass,
tokenization/feature transformation, memory operations, sample generation
and actual elapsed time. The exact class compiler, 256 graph normalizations,
class tables, independent integer checker and all model weights are
additional resources. None of the listed operation units can be summed
as a single hardware-independent cost.

The bare FP32 checkpoint sizes are 199,680 bits (GRU) and 199,520 bits
(MLP), excluding JSON metadata. The GRU retains 32 FP32 state values.
At a fixed syntax-only two-bit-per-token code, one graph costs six bits,
plus a public topology ID; this does not account for the interpreter,
compiler, training examples, class tables or retained coefficients.

## Training and reproducibility

Training used 3,600 examples and a disjoint 400-case development roster,
with all four topologies and signed integer input series. Recurrent dev
canonical-class top-12 accuracy was **95.75%**, compared with **85.25%**
for feedforward and **66.25%** for the shuffled-label recurrent.
Development labels are training-grammar class representatives, so these
figures are diagnostics and do not measure certified held-out execution.
An independent retraining run reproduced all three checkpoint JSON files
byte-for-byte. Nineteen RRC-18 tests and 32 existing RRC-16/RRC-17
regressions passed locally; the source-locked JSON receipt reproduced
exactly. Tests check no inference-time graph executions, two executions
per charged query, no input leakage, concrete polynomial parity and
future-pair invariance.

## Decision and next test

The finite semantic quotient can now be **certified rather than inferred
probabilistically**, and a small neural class ranker improves success
within the signed-series family. The strongest test failed: that advantage
did not survive the new source generator against an exact symbolic class
prior. RRC-19 should train on multiple algebraic source families with
family-balanced objectives and test an unseen operator or topology. It
should also compare a mixture of learned and fixed class priors against
fully measured compile/training/verification budgets, using fresh seeds
and avoiding the public RRC-18 roster for model selection.
