# RRC-03: frozen interactive extension to RRC-02

**Registration (2026-11-09):** supervised CPU policy pilot with exact oracle,
source-hashed independent replay and a mathematical capacity bound.

## Hypothesis class and information boundaries

- Baseline is the first Rogers–Ramanujan product, whose coefficient `f_n` is
  positive at every nonnegative degree because parts of size one are allowed.
- `H` candidates consist of one unchanged baseline and `H-1` single-defect
  products multiplied by `1/(1-q^K)`, `K=65..63+H`.
- All candidate coefficients `f_0..f_64` are identical. Hidden candidate
  identity never enters student input, policy history or final MAP decoder.
- One query chooses integer `n=65..63+H` and returns the **single bit**
  `1{ candidate_coefficient(n) > baseline_coefficient(n) }`.
- One bit costs one query regardless of degree, with the substantial exact
  integer-computation cost accounted for separately. Full numeric coefficients,
  exact Euler exponents and candidate identity are never delivered to a model.
- Uniform prior; classification output is the smallest index in the exact
  posterior interval. Budget `B=0..6`. Only legal degrees are accepted.

## Registered baselines and exact gates

1. Independent local-gap and product counting, plus independent convolution
   and Euler-transform defect construction, must agree for each hypothesis.
2. Every candidate must share degrees `0..64`. The binary comparison must
   equal the declared threshold predicate at every legal query.
3. Analytical capacity: adaptive `min(2^B,H)/H`, fixed `min(B+1,H)/H`.
4. An independent Bellman DP must agree with the adaptive certificate for a
   uniform prior; small non-uniform prior fixtures exercise generality.
5. Corrupting a query bit must alter the posterior or yield contradiction.
6. A policy cannot access true hidden `K`; the simulator supplies only the
   query response. Source-bound trace hashes capture every requested query
   and decoded answer, so scoring drift is detected.

## Training and held-out split

Two models contain **exactly 353 trainable parameters each**. The recurrent
model is a 16-dimensional shared tanh update over `(normalized-query,
response,step,hypothesis-count)` and a scalar action head. Its 353-parameter
feedforward control reads the full six-slot feedback history, with a fixed
mask and public count; it receives at least as much observable information
as the recurrent model. Both use the same loss, teacher examples, optimizer
steps and batch size.

- Training: candidate counts `H=9,17,33,65`, 1,000 seeded trajectories for
  each count, random hidden candidate, `55%` midpoint vs `45%` random legal
  informative query selection. Every resulting nonterminal history produces
  an exact midpoint imitation target; 19,604 training examples in the
  recorded seeded run.
- Adam, 1,500 updates, batch 256, learning rate `0.01`, per-model seed
  `20261109+77` and `+78`. No GPU. Pure Python portable weight replay
  is numerically checked against trained PyTorch weights.
- Held-out counts `H=13,25,49`. Their exact *candidate-set sizes* and
  evaluation query transcripts are inaccessible to training and tuning.
  The underlying single-defect family and some individual defect degrees
  overlap the training `H=65` pool. This measures size/history transfer,
  not a new generating-function family. `H=33` is also audited as an
  in-domain, exhaustive candidate roster.
- Controls: analytic adaptive midpoint, analytic best fixed-query policy,
  full-history feedforward with matched parameter budget, and feedback-censored
  inference for both students. Censored policies continue to receive the
  public query degrees and the exact scoring decoder receives real feedback,
  isolating the causal contribution of feedback to action selection.

## Scoring, audit and interpretation

For every `H` and budget, enumerate **all** hypotheses. Store exact
classification numerator and denominator, distinct response transcripts,
requested query counts, wasted query counts and SHA-256 of every full
candidate transcript. Report the 4-query frontier and the 5/6-query
failure cases. The exact posterior decoder is separately charged rather
than credited to learned neural inference.

Replay source and checkpoint SHA-256, roster hash and floating-point weights
are recorded. Student replay and theorem tests run under Python's standard
library in lightweight GitHub Actions; retraining is optional PyTorch and
is intentionally absent from CI. RRC-03 is a policy-learning result on a
*known binary threshold family*. Neural recovery of a periodic grammar
from raw sequence coefficients remains a different, unrun experiment.
