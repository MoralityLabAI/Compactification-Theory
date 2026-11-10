# RRC-17: exact quotient-search exploratory protocol

**Executed:** 2026-11-10, standard-library CPU. The design follows publicly known RRC-11–RRC-16 findings, and the protocol was written during execution rather than externally preregistered. RRC-16's Hadamard interpreter and frozen recurrent/feedforward checkpoints were already available. All results are finite coefficients and this one experiment is not an independent, blind proof of graph equivalence.

## Hypothesis and holdout

A topology-conditioned semantic quotient built using public random formal-series probes should reduce the number of *exact* full graph queries needed to identify a correct candidate in the four-topology 64-program grammar. A static 12-query top-class-prior method will be compared with RRC-16's frozen GRU, matched feedforward model, and untrained frontier. The software retains all RRC-16 source and checkpoint digests; **RRC-17 performs zero new model optimization**.

- Offline public probes: four independent vectors uniform in `F_(2^31-1)^12`, seed `2026111017 + 19011`. No test hidden labels, observed targets or future outputs enter the quotient builder. Cache all four topology partitions and representatives before any test search.
- Primary evaluation: 96 uniform Hadamard signed-source cases, 48 Hadamard cases selected for absence of old-topology observed-prefix aliases, 48 Hadamard arithmetic-product sources, plus 48 each of fork, cascade and additive from uniform syntax. Total 336 positives, all from a fresh deterministic seed independent of RRC-16.
- Controls: 16 selected inconsistent Hadamard demonstration pairs, chosen after exact 64-class rejection; eight Hadamard base/future-mutant pairs (16 records), identical on all public inputs and diverging at coefficient degree 15.
- Three additional 128-case Hadamard sets with seeds `2026115017`, `2026116017`, `2026117017`, each 25% arithmetic-product sources, are **post-design robustness checks**, never used in class ranking or model selection. Their outcomes are identified separately from the main suite.
- All normal program labels are drawn uniformly from the syntax language; a correct alternative program is accepted if its public observed coefficients match. The `hadamard_novel` stratum is conditionally selected using the *public degree-12 observations* and can change syntax-class frequencies. The main result reports that selection separately.

## Equal-information methods

**Frozen policy arms:** RRC-16 GRU with state retained, identical checkpoint with state erased after failed candidates, matched feedforward and untrained residual frontier, with the exact RRC-13 initializer they already include. Each executes up to 12 unique full-degree candidate graphs and sees only previously executed feedback.

**Untrained arms:** first 12 canonical programs; 12 uniformly shuffled class representatives from a deterministic visible-data seed; first 12 canonical distinct classes; top 12 classes by number of syntax spellings; top 12 by syntax mass divided by exact per-query work; complete quotient representative list (24–33, higher budget); full 64-program exact search (higher budget). No candidate-output preview during test-time ranking. The quotient's offline construction cost is reported separately and never assigned zero.

At test time every candidate uses the **same** degree-12 exact integer two-demonstration verifier. A new graph is charged once, with two graph executions, eight primitive calls and topology-specific coefficient-loop work. A winning representative must be among executed candidates. Scoring future outputs through degree 32 is independent and inaccessible during inference. Count neural MAC proxies and integer-loop proxies separately.

## Required gates

1. Exhaustive partition: all 64 programs in exactly one quotient class per topology; the generated class counts must match frozen values. Other field-probe seeds and unrelated signed integer sources must show no observed within-class difference.
2. Every accepted program passes actual public integer coefficient verification. An adversarial future or hidden-label replacement must preserve the entire candidate trace. Patch the interpreter in tests and check exactly two executions per charged graph query.
3. Complete full-DSL search accepts every positive case and rejects all selected observed-inconsistent cases. Complete quotient search must match its result on the declared roster, or the experiment records the failure of representative sufficiency.
4. Every partial-budget arm uses at most 12 distinct full graph queries. The offline 1,024 probe executions and 143,104 nominal units are reported separately, with amortization assumptions stated.
5. Exact collision pairs must produce identical traces. The complete solver makes exactly eight wrong future predictions across eight base/mutant pairs. A valid degree-12 certificate cannot be called an infinite-series identity or a future guarantee.
6. The source-hashed saved receipt must regenerate byte-for-byte with no network, downloads or retraining. The earlier RRC-14/15/16 tests remain unmodified and are run in light CI.

## Reproduce

From Compactification-Theory root, Python 3.10+ standard library:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_quotient17.py' -v
python research/rrc_compactification/bench17.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_topology16.py' -v
python -m unittest discover -s research/rrc_compactification -p 'test_topology15.py' -v
```

The three frozen RRC-16 checkpoints remain unmodified, with explicit SHA-256 identity checks. The quotient method is symbolic/interpreter-assisted and its superiority, if observed, does not establish a neural generalization improvement.
