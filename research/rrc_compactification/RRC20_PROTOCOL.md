# RRC-20: exploratory calibration and source-composition transfer protocol

**Execution date: 2026-11-10.** The training-source calibration idea was
chosen after RRC-19's public results. This protocol and the recorded
metrics were finalized in the same exploratory development session. It
is not an externally registered or independently blinded comparison.

## Frozen components and additional supervision

The three RRC-19 checkpoints, RRC-18's certified class partition and the
four algebraic graph interpreters are reused without modification.
RRC-20 may update only a 60-parameter adaptive gate, a 4-parameter
constant gate, and an equally sized shuffled-target calibration control.
All calibrators receive the **420 existing RRC-19 development examples**,
including teacher-supplied exact canonical labels. These data were used
for prior RRC-19 development diagnostics, so they are exposed calibration
examples with a history of model selection.

Training uses 500 FP64 Adam steps at learning rate 0.035. The adaptive
and shuffled models penalize mean squared matrix weights by 0.002.
A source-only feature extractor reads exactly two input prefixes and the
public topology. A convex mixture of frozen expert class distributions
ranks candidates. The constant gate learns only four biases; uniform
mixture is an untrained default and the class-size prior is a separate
symbolic control. Shuffled-label supervision changes the gate's labels
and leaves the three trained experts intact.

## New held-out evaluation

RRC-20 builds **fresh cases** with seed 2026111020, excluding every public
RRC-19 training, development and test visible-prefix signature. Each of
four graph topologies appears within every source family. Cases are
sampled uniformly over the 64 source graph syntaxes, while source
selection is independent of the model or withheld target results.

| Group | Cases | Source generator |
|---|---:|---|
| Signed, product, sparse | 48 each | Familiar generators, fresh instances |
| Twisted product | 96 | `P(-q)` from an arithmetic-part product |
| Rational product | 96 | `P(q) / S(q)` with sparse finite `S` and unit constant |
| Mixed signed/product | 96 | Product of signed coefficients and an arithmetic-part product |
| Cubic sparse | 96 | Third convolution power of a sparse unit series |
| Selected inconsistent demonstrations | 12 | Degree-seven observed mutation, rejected by all 64 graphs |
| Late future collisions | 12 | Six matched visible base/mutant pairs, different future degree 15 |

The four compositional source families are unseen as *generators* during
all RRC-19 and RRC-20 optimization. Their building blocks are known
public algebraic primitives. All graph topologies, the candidate grammar
and its verifier remain familiar. Test-source family labels are used for
reporting and excluded from the gate's feature vector.

## Exact evaluation and comparison

- At most 12 distinct certified graph-class representatives are proposed
  by every budgeted arm. Each one invokes the exact interpreter on both
  observed degree-12 demonstration pairs. Candidate execution occurs
  only **after** ranking and costs the full two-demo graph evaluation.
- Controls: frozen balanced GRU, frozen skewed GRU, frozen balanced FF,
  class-size prior, fixed twelve syntax graphs, uniform four-expert
  mixture, constant-calibrated mixture, shuffled-target gate. The
  complete class solver can spend up to 33 verification calls.
- Report accepted finite programs, future-correct predictions through
  degree 32, false future acceptances, abstentions, candidate calls,
  integer-loop work, dense MACs, model and gate parameters, teacher
  supervision, and source/program compiler costs.
- Retain the per-case paired differences and the full hashed trajectory
  checks. Every selected candidate must have a valid exact prefix
  certificate. The full-class baseline must accept all 528 positive
  cases and reject all 12 selected negatives.
- Every future collision pair must have identical model proposals and
  search traces, even when the hidden grading answer changes.

A robust calibrated gate would need to exceed the untrained uniform
mixture and strong frozen single experts on the combined 384 *novel*
source-construction cases while paying for its extra inference. The
negative result is recorded without architecture or threshold retuning.

## Reproduction

From the Compactification-Theory repository root, Python 3.10+ standard
library is sufficient for all frozen-model evaluations:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_family20.py' -v
python research/rrc_compactification/bench20.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_family19.py' -v
python -m unittest discover -s research/rrc_compactification -p 'test_certify18.py' -v
```

Optional independent gate retraining uses CPU PyTorch and must produce
byte-identical files:

```bash
python research/rrc_compactification/train20.py --out-dir /tmp/rrc20_retrain
for role in constant adaptive shuffled; do
  cmp /tmp/rrc20_retrain/rrc20_${role}.json research/rrc_compactification/results/rrc20_${role}.json
done
```

The receipt contains SHA-256 hashes of all dependencies, three gate
checkpoints, all previously frozen expert checkpoints, calibration
roster, all new split rosters, individual method traces, and exact score
tables. A source or weight change invalidates the recorded receipt until
independently reviewed and regenerated.
