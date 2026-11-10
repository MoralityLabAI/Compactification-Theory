# RRC-20: calibrated class mixtures and exact finite verification

## Mathematical setting

The evaluator uses RRC-18's certified equivalence relation on 64
three-operation graphs `post(left(F) JOIN right(F))`, with four public
wirings (fork, cascade, normalized addition and coefficientwise Hadamard).
The coefficientwise polynomial normalizer works in
`Z[x_1,...,x_12]`; it proves equality of each grouped graph's first
thirteen coefficients for **all** integral unit input series. It does not
certify equality at unbounded degree.

For an observed pair of two input/output demonstrations `X`, a public
wiring `tau`, and one canonical representative `p` in the certified class
set `C_tau`, the earlier frozen models produce logits `z_e(p|X)`.
This phase uses three previously trained models:

* balanced RRC-19 recurrent (6,240 weights);
* skewed RRC-19 recurrent (6,240 weights);
* balanced RRC-19 feedforward (6,235 weights).

They are converted into normalized class distributions by

`P_e(p|X,tau) = exp(z_e(p))/sum_{r in C_tau}exp(z_e(r))`.

The fourth expert is the certified class-mass prior
`P_mass(p|tau) = |[p]_tau|/64`. It uses the complete public equivalence
class table and has offline compiler/interpreter costs, even though it
contains no trainable neural weights.

## Learning a small gate

For fourteen bounded, public source-series statistics `phi(X,tau)`, the
adaptive gate has 60 real coefficients (4 times fourteen plus four biases):

`w_e = exp(b_e + W_e . phi) / sum_j exp(b_j + W_j . phi)`.

The resulting ranked distribution is an exact convex mixture

`P_theta(p|X,tau) = sum_e w_e P_e(p|X,tau)`.

The constant gate fixes `W=0` and learns four biases. The uniform mixture
uses four weights of `1/4` without fitting. A 60-parameter negative control
scrambles *only the gate's* canonical-class targets within each topology;
all three frozen neural experts still retain their legitimate supervised
training. Its performance cannot be interpreted as an untrained network's
accuracy.

The gate minimizes observed calibration log-loss on 420 formerly exposed
RRC-19 development records, using 500 full-batch FP64 Adam updates and a
small L2 penalty on the adaptive matrix. Each record's canonical label is
provided by the certified algebraic teacher. Consequently the gate's
improved calibration accuracy measures fitting to a familiar supervised
distribution. It supplies no theorem about out-of-family expected risk.

The source-family name, hidden graph and withheld degree-13..32 query
output do not enter any inference features. The source-feature vector
contains nonzero density, signedness, coefficient-size statistics, parity
sparsity, early-versus-late growth, first nonzero position and four public
wiring bits. Frozen neural predictors also receive the public output
coefficients of both demonstrations. No graph candidate is executed to
score or gate any unverified proposal.

## Executable finite certificates

The gate emits an ordering of distinct canonical class representatives.
The verifier executes at most twelve selected programs *on both observed
input series* through degree twelve. Accepting a candidate certifies

`G_p(F_i)_n = Y_{i,n}` for `i in {1,2}` and every `0<=n<=12`.

This is an exact statement about the observed coefficients. Query outputs
through degree 32 are graded independently. Complete certified search
can test up to 33 classes and is an upper control for this finite grammar,
with a larger interpreter budget.

For `K>12`, changing the future answer at degree `K` while keeping the
two demonstrations and query input fixed produces indistinguishable
observations. Deterministic candidate orderings are identical, and one
candidate continuation can match at most one of a paired base/future
mutant. RRC-20 explicitly contains six such pairs. The cubic-sparse
source family also creates an ordinary instance where an accepted
12-degree candidate extrapolates incorrectly, demonstrating the
finite-horizon qualification on an unmutated source.

## Resource comparison

An adaptive mixture evaluates all three pretrained networks:
`2 * 53,216 + 6,120 + 56 = 112,608` dense forward MACs per case
(plus nonlinearities and feature extraction). The constant and uniform
mixtures use 112,552; a single recurrent costs 53,216, and feedforward
alone 6,120. The adaptive gate's storage is 60 FP64 coefficients (3,840
bare bits) plus metadata; previously trained expert weights total 18,715
FP32 coefficients (598,880 bare bits). The class table, verifier and
interpreter are additional resources.

Every exact candidate check executes both degree-12 demonstrations and
is charged twice the appropriate topology-specific integer coefficient
loop cost. Integer arithmetic and neural MAC counts are separate units;
big-integer bit complexity, Python overhead and GPU time are not inferred
from either proxy. The prior RRC-19 neural training and the 256 exact
polynomial graph normalizations remain prerequisites, as do 420
calibration teacher labels and all model evaluations needed to prepare
the gate dataset.
