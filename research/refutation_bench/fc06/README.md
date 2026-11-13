# FC-06: strict-budget proof-rule choice with reachable distractors

FC-06 addresses [FC-05's ten-action saturation](../fc05/POSTHOC_SATURATION.md).
Its [preregistration](PREREGISTRATION.md) was committed as
`81f53bcf606e45d8b3fea9acba7907641c339cab`, before constructing
any FC-06 source graph or evaluating models.

Every graph has exactly 12 proposition atoms: an initial fact, a goal
backbone of d implications, 10-d reachable distractors and one
unreachable guard. A valid goal has an **exact shortest proof of d<=5**
novel rule applications, while ten novel consequences are reachable
altogether. Therefore a five-action cap is *binding*. A legal but
distracting rule consumes scarce proof budget.

A neural actor must choose an ordered list of no more than five
original rule IDs. The independent FC-05 trace checker verifies
each rule and its known antecedents, enforces the shorter budget,
and accepts only complete goal derivations. Accepted model traces
become Lean 4.19.0 theorem proofs using their **exact rule order**.
A failed proof proposal abstains. Negative refutations require an
independently checked complete Boolean countermodel.

## Frozen source family split

| Group | Cases | Goal-chain d | Family provenance |
| --- | ---: | --- | --- |
| Train | 768 | 2-4 | star_decoys, ladder_decoys, fork_decoys |
| Development | 192 | 2-4 | cross_decoys |
| Held-out same depth | 192 | 2-4 | zigzag_lures, nested_fork, late_spoke, alternating_gate |
| Held-out deeper | 192 | 4-5 | Same four unseen families, distinct instances |

The test splits each have 96 valid and 96 invalid examples, with
24 valid and 24 invalid in every test family. Every sample is
checked by both operational Horn closure and exhaustive truth
tables; positives have minimal BFS proof traces and negatives have
complete countermodels. Conservative WL structural fingerprints
are disjoint from all earlier FC-02 through FC-05 cases and across
FC-06 splits.

To protect the held-out experiment, train data were committed
before adding the test generator. The training GitHub Actions job
checks out only the source at the old commit and asserts that
`test_data.py` does not exist.

| Artifact | Frozen commit | Canonical JSONL SHA-256 |
| --- | --- | --- |
| Train-only source | `1bf514325788b633432eb9f3e49627bd923a216e` | `c841727d6a3e82c5ce5769b2b324e9bde078d15930d03fad5462292657d73ccf` |
| Held-out source | `63c7c411e32a25b447b1ca92ae0db9e33b135d27` | `6d570d3c6b21466c8df0032cac0cab115ebf3a586e82160f18b0a96f9efb9d9e` |

[Train source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414157412)
passed seven tests and compiled **480 positive Lean reference theorems**.
[Held-out source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414735943)
passed seven tests and compiled **192 positive Lean reference theorems**.
The other 480 training/development and 192 test cases have verified
countermodels. Both source CI jobs inspected endpoint theorems and
found no extra Lean axioms.

## First non-neural held-out control

The following was generated **before neural evaluation** from the
fixed source and preserved without filtering or resampling cases:

| Five-action policy | Same-depth valid proofs /96 | Deep valid proofs /96 |
| --- | ---: | ---: |
| Uniform random legal | 4 | 2 |
| Immediate goal if legal, else random | 26 | 4 |
| Exact minimal BFS (oracle upper bound) | 96 | 96 |

The large gap between heuristic success and exact shortest proofs
establishes a meaningful rule-selection challenge. It does not
predict whether the trained neural model will solve it.

## Reproducibility

From the repository root:

~~~sh
python -m unittest -v research.refutation_bench.tests.test_fc06_train
python -m unittest -v research.refutation_bench.tests.test_fc06_test

python -m research.refutation_bench.fc06.train_data \
  --out /tmp/FC06_train.jsonl \
  --lean /tmp/FC06_train.lean

python -m research.refutation_bench.fc06.test_data \
  --out /tmp/FC06_test.jsonl \
  --lean /tmp/FC06_test.lean

lean /tmp/FC06_train.lean
lean /tmp/FC06_test.lean
~~~

A separate private LoopedTransformers branch trains the FC-04 negative
gate on FC-02 and jointly fine-tunes a copied LR-01 graph cell and
4,705-parameter rule head on **FC-06 training-positive prefixes**.
The test evaluation compares the learned head against random legal
and goal-first ordering under the same five-action budget, with
separate verifier, neural inference, and Lean compilation receipts.

This is an exact formal-logic surrogate. Real-world mathematical
manuscript assessment still requires a verified semantics-preserving
source-to-formal transformation.
