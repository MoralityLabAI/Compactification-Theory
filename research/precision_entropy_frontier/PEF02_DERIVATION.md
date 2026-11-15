# PEF-02 | Robust history geometry, recurrent quantization and prior leakage

A continuation of [PEF-01](README.md). The source motivations remain OpenAI
mathematics #148 (entropy-rate dimension) and #140 (memory/precision).
PEF-02 derives independent finite statements about an explicit recurrence.

## 1. Exact geometric margin for recent-history suffixes

Let M=2^k, k>=1, L>=t>=1, and define the exact length-L orbit

    x_L(b) = (M-1) sum_{i=0}^{L-1} b_i M^i / M^L.

Fix the t newest input bits. The set of exact states from all older histories
lies in an interval of length M^-t - M^-L, from the all-zeros older history to
the all-ones older history. The lower endpoints for distinct suffixes are

    S(c) = (M-1) sum_{j=1}^t c_j M^-j.

The minimum difference between distinct sorted S(c) is (M-1) M^-t.
Proof: if their first difference occurs at digit r, the remaining terms
can offset at most M^-r - M^-t, leaving at least
(M-2)M^-r + M^-t >= (M-1) M^-t. The bound is achieved by two suffixes
differing at their last retained digit with earlier retained bits equal.

Subtracting the maximum old-history tail gives the **exact** positive
cross-suffix gap

    g(k,L,t) = (M-2) M^-t + M^-L.                 [PEF-02 A]

The formula was independently compared with every pair of adjacent finite
class hulls for k=1..4, L=1..7, t=1..L, using exact integers.

**Sharp analog adversarial threshold.** All exact trajectory suffixes
remain perfectly decodable from an adversarially perturbed real readout
y=x+e, |e|<=epsilon, precisely when 2 epsilon < g for worst-case uniform
guarantees. At equality the two closest class intervals touch. The
adversary may choose different errors on different trajectories.

If the recurrence has an additive state error e_j after each step with
|e_j|<=eta, its end-state deviation is bounded by

    E_L = eta sum_{r=0}^{L-1} M^-r
        = eta (1-M^-L)/(1-M^-1).

Thus the sharp analog sufficient/necessary limit for the error model is
2E_L < g, with every sign of the disturbances available to the adversary.

For a final floor fixed-point quantizer Q_B(x)=floor(2^B x), the
**sufficient** condition for disjoint suffix code sets is

    2 E_L + 2^-B <= g.                           [PEF-02 B]

A shared code would force two perturbed real values within distance
strictly less than one bin width, contradicting [B]. This bound depends
on bin width and need not be optimal for a particular quantizer phase.

For k=2,L=8,B=8, t=4, g=513/65536 and the fixed-point bin is 1/256,
so analog endpoint tolerance is epsilon < 513/131072, while an endpoint
fixed-point sufficient guard is epsilon <= 257/131072. For bounded
per-step errors divide these epsilon bounds by the geometric gain.

A key Pareto phenomenon: lower dimension D=1/k makes the retained-suffix
count smaller at fixed B, yet can widen the real geometric gaps between
the remaining suffix clusters. Compressed representations may preserve
fewer historical distinctions more robustly.

## 2. Exact commutation for floor-quantized recurrent states

Let u_j=floor(2^P x_j), and quantize **after every recurrence**:

    u_(j+1) = floor((u_j+(M-1)2^P b_j)/M).

For all k,P,L and binary histories, the final u_L equals the endpoint
quantizer floor(2^P x_L). This follows from the elementary identity
for integral C and integer M>0:

    floor((floor(z)+C)/M) = floor((z+C)/M).

The result is special to this integer-base digit contraction and floor
quantizer. In an explicit nearest-integer half-up rounding control,
k=2, P=3, chronological history [1,0,0] gives stagewise code **1**
and endpoint-only code **0**. These differences are witnessed by the
Python oracle.

This distinction matters for looped layers: repeated rounding can
preserve an exact digit shift or cause history-sensitive errors,
depending on the operation, rounding mode, and geometry. FP16/BF16
are relative/exponent formats outside this ideal fixed-point theorem.

## 3. Gauge and source-prior caveats

An absolute B-fractional-bit readout after a gain z=2^r x has the same
code as Q_(B+r)(x); amplification can seem to increase the amount
of recoverable history with unchanged fractal dimension. It also expands
the necessary integer range, so it is not a free fixed-word-size gain.

For an affine coordinate change z=s x+a, s>0, use a calibrated readout

    C_B^[a,a+s](z)=floor(2^B (z-a)/s).

It is exactly equal to the original Q_B(x). Benchmarks on recurrent
activations should report the calibration range and distinguish affine
coordinate gauge changes from actual lost dynamical information.

**Source-prior leakage.** Let the chronological binary input form a
stationary symmetric Markov chain with P(b_j=b_(j-1))=s and b_0 fair.
For t>=1, the same code C_B deterministically exposes exactly the t
newest bits. All older individual interventions with newer inputs
fixed still have **zero** causal readout effect. Yet a Bayes-optimal
decoder of the entire L-bit history has exact average success

    MAP(s) = max(s,1-s)^(L-t),                   [PEF-02 C]

and conditional entropy

    H(history|C_B)=(L-t) h2(s) bits.

For t=0, the corresponding MAP is (1/2)max(s,1-s)^(L-1).

At k=2,L=8,B=8,t=4 and s=19/20 the full-history Bayes MAP rate is
(19/20)^4 = **130321/160000**, while the uniform-source rate was 1/16.
The history channel is unchanged. Prediction from a source prior can
appear like increased latent memory.

For an older bit i < L-t, its observational agreement with the first
retained bit b_(L-t) is

    P(b_i=b_(L-t)) = [1+(2s-1)^(L-t-i)]/2,

which can be high despite zero direct causal dependence in the readout.
Observational linear probes must therefore be paired with causal
interventions and conditional controls to measure actual latent retention.

## 4. Evidence and next mechanistic-interpretability experiment

The PEF-02 tests exhaust every suffix hull up to L=7, compare all
recurrent floor and endpoint-only codes across a parameter grid,
verify the nearest-rounding counterexample, test source-conditioned
Bayes MAP exactly using Fraction, and test affine calibrated readouts.
The CI stores an exact machine-readable receipt without floating point.

To apply to an actual trained looped transformer, preregister:
- one fixed base/instruct checkpoint and recurrent layer with a fixed
  architecture, input family and extraction hook;
- train-only calibration range and truly measured numeric format;
- a heldout causal bit-flip intervention sweep over historical positions;
- a probe trained only on observational source data, with shuffled and
  balanced controls, plus exact native-state lower baseline;
- a separate well-defined readout-noise budget and depth schedule.

Report bitwise direct influence, sufficiency/necessity of the hidden
state for history reconstruction, source-prior-only MAP performance,
and activation Jacobian singular values. PEF-02 makes no claim of
tested learned-model capability or floating-point robustness.

Run:

    python -m unittest discover -s research/precision_entropy_frontier -p 'test_pef02.py' -v
    python research/precision_entropy_frontier/pef02.py --suite --out /tmp/pef02.json

The pure mathematical finite certificates are in
patrickdugan/MathCan/MathCan/PrecisionEntropyRobust.lean.
