# PEF-01 | Entropy-rate dimension meets memory-limited inference

**November 15, 2026.** A new, independently proved finite applied-mathematics
campaign for the compact-model depth–precision frontier and reversible
activation-history tests. Prior TFH-01 mathematics has been merged.

## OpenAI math source pair

- **#148:** *The entropy-rate dimension formula for self-similar measures
  on the line* (2026-09-24).
  https://github.com/openai/math/blob/main/preprints/The-entropy-rate-dimension-formula-for-self-similar-measures-on-the-line-September-24-2026/main.pdf
  The source concerns h_RW/chi and self-similar measures with exact overlaps.
- **#140:** *Memory and precision in noiseless Gaussian regression*
  (2026-09-27).
  https://github.com/openai/math/blob/main/preprints/Memory-and-precision-in-noiseless-Gaussian-regression-September-27-2026/paper.pdf
  The source concerns the sample burden of a one-pass Gaussian estimator
  with O(d^2) persistent bits.

PEF-01 borrows the structural questions of contraction-dimension and
finite memory. The finite theorem below is independently derived and
tested; it neither imports the Gaussian sample lower bound nor claims
the OpenAI manuscripts certify anything about a trained transformer.
OpenAI's research repository labels parts of its collection as
research data at varying stages of external mathematical verification.

## New finite applied theorem

Consider a model of depth-L input-conditioned recurrence, initially x0=0:

    x_(i+1) = (x_i + (2^k-1)b_i) / 2^k,
    b_i uniform in {0,1}, k >= 1.

The exact end-state is

    x_L = (2^k-1) sum_{i=0}^{L-1} b_i (2^k)^i / (2^k)^L.

A fixed-point observer stores only C_B = floor(2^B * x_L),
with a precision budget of B >= 0 bits after the binary point.

**PEF-1 (exact suffix fibre theorem):** Put

    t(k,L,B) = min{L, ceil(B/k)}.

Every output of C_B identifies exactly the last t chronological input
bits. All other older bits are hidden. There are 2^t distinct
readout codes, and each code has exactly 2^(L-t) histories.

Proof: each base-(2^k) expansion digit is either all binary zeros
or all ones. B high-order binary fractional digits reveal floor(B/k)
whole digits and, when there are B mod k further digits, the leading
part of the next digit also identifies its input bit. Every earlier
digit lies entirely below the quantization cut. Thus each
observable suffix has 2^(L-t) uniform completions.

**PEF-2 (finite Bayes recovery):** An arbitrary deterministic or
randomized history decoder using only C_B has optimal average
whole-history success probability

    P(success) = 2^(t-L).

The conditional Shannon entropy in the history is exactly L-t bits.
Every equal-size code fibre allows one correct representative; this
upper bound on success holds even for an unbounded-compute decoder.

**PEF-3 (sharp precision budget):** At positive depth L the smallest
number of fixed-point bits allowing exact history recovery is

    B_min = k(L-1)+1.

This is a finite precision-depth transition, sharper than the
kL digits needed to reproduce the full exact scalar numerator.

## Link to dimensions and new measurable predictions

For this separated Bernoulli IFS the geometric dimension is
D = h_RW/chi = log(2)/(k log(2)) = 1/k, consistent with #148.
Hence the depth slope of memory needed to recover *all* events is
k=1/D fixed-point bits per loop. In this example, contraction
and low-dimensionality increase the loss of distant history at fixed
absolute readout precision, while recent bits survive.

At B=8, depth L=8:
- k=1 (dimension 1) recovers 8 bits, exact-history success 1;
- k=2 (dimension 1/2) recovers 4 bits, success 1/16;
- k=3 (dimension 1/3) recovers 3 bits, success 1/32.
For k=2, L=12, B=8 the exact-history success is 1/256.
At k=2,L=8 the exact-recovery threshold is 15 precision bits.

These are **finite mathematical predictions for this affine toy**. A
real looped transformer has learned nonlinear dynamics, noise, hidden
channels, representation-dependent and often relative-precision
quantization; an empirical experiment would need measured local
Jacobian spectra, intervention-retrodiction probes, and documented
precision calibration before generalizing. A folded SAE need not
behave like the scalar IFS.

## Applied experiment and controls

The standard-library Python oracle:
- enumerates all length-L binary histories for L<=12;
- separately computes the exact recurrence using Fraction and its
  integer numerator using base powers;
- classifies every code fibre and checks equal sizes and suffix
  bijectivity;
- flips **every input bit** of every trajectory to certify exactly
  which past events remain causal for the stored code;
- calculates optimal Bayes decoding success and conditional entropy
  without floating-point entropy estimates;
- records an immutable compact suite of cases for k=1,2,3 and L=8,12.

All computations run on CPU and use only exact integers or rationals.
No training, GPU use, model-output evaluation or held-out contamination
takes place. This is a calibration benchmark ready to act as a null
for future recurrent-activation retention experiments.

## Independent reproduction

From the repository root:

    python -m unittest discover -s research/precision_entropy_frontier -p 'test_pef01.py' -v
    python research/precision_entropy_frontier/pef01.py --suite --output /tmp/pef01-suite.json
    python research/precision_entropy_frontier/pef01.py --k 2 --depth 12 --bits 8

The pure MathCan companion adds kernel checked depth-four suffix and
fibre results plus all-depth quaternary threshold arithmetic:
patrickdugan/MathCan/MathCan/PrecisionEntropy.lean.

Any actual trained-model capability or mechanistic-retrodiction result
will require a new frozen protocol and independent native baselines.


## PEF-02: robust, gauge-calibrated and schedule-sensitive history

Follow-on mathematics and exact CPU checks are recorded in
[PEF02_DERIVATION.md](PEF02_DERIVATION.md) and
[PEF02_SCHEDULE.md](PEF02_SCHEDULE.md).

For constant k and L>=t>=1, the sharp analog cross-suffix gap is
g=(2^k-2)*2^(-kt)+2^(-kL). Integer floor quantization at every step
exactly equals endpoint-only floor quantization for this toy, while
nearest rounding has an explicit discrepancy.

For a stationary binary Markov source with persistence s, Bayes
whole-history success max(s,1-s)^(L-t) can be much higher than the
fair-source baseline even though old-bit interventions with the recent
suffix fixed have no effect on the readout.

With varying chronological contraction widths k_j, the ordered schedule
determines the finite readout margins. At L=3 and B=6, schedules
(2,2,4) and (2,4,2) both retain two bits while their exact gaps
are 9/256 and 57/256. Schedule (4,2,2) retains all three bits.

Reproduce:
    python -m unittest discover -s research/precision_entropy_frontier -p 'test_pef02.py' -v
    python -m unittest discover -s research/precision_entropy_frontier -p 'test_pef02_schedule.py' -v
    python research/precision_entropy_frontier/pef02.py --suite
    python research/precision_entropy_frontier/pef02_schedule.py --suite

These are finite affine recurrences, with no measured trained-model
capability gain. The prospective experiment should compare causal
history interventions and observational prior probes on frozen looped
transformer activations with calibrated numerical readouts.
