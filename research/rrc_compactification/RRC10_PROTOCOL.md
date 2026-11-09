# RRC-10 experiment contract

**Status:** exploratory CPU pilot, prepared and executed on 2026-11-09, following the RRC-09 negative transfer result. This protocol accompanies the measured implementation. It was not preregistered in an external immutable registry.

## Frozen interface

- An episode contains two independent raw coefficient demonstrations, with 25 observed coefficients (degrees 0–24) each. The hidden rule is multiplication by an unknown polynomial \(H\); the model receives neither its coefficients nor its generating template during inference.
- Query input is an independent 49-coefficient series, and its target output is withheld through degree 48. The wrapper is public and drawn from identity, sign substitution, reciprocal and degree doubling. Wrapper semantics are supplied to every method.
- Training has 128 episodes from 8 single-monomial insertion families \(H=1+q^k\), \(k=1..8\), and the development group has 24 more. A local exact teacher derives \(h_n\) from the **training demonstrations only**. Both trainable arms fit two rational coefficients from the same 6,144 supervised rows. The recurrent arm has a public triangular convolution state; the local feedforward arm has only \(f_n,g_n\).
- Test has 48 polynomial multipliers with unseen \(k\in\{9,10,11,12\}\) and new double, signed, or triple coefficients at degree \(2k\); 48 further episodes compose those held-out multipliers with the public N/R/S wrappers. The new test source inputs, multiplier families, and k values are absent from training. Test labels and future coefficients never influence weight fitting.
- Eight paired late-defect episodes (16 items) differ only at hidden degree 29. Sixteen deliberately inconsistent second-demonstration cases must be rejected.

## Baselines and gates

1. Fit and compile the two-coefficient causal model with fixed nearest-integer snapping tolerance 1/8, requiring coefficients in [-2,2]. Verify exact training solution and source digest.
2. Compare with parameter-matched raw-local regression, the direct exact triangular division algorithm, and a no-demo zero-weight arm. The exact algorithm is supplied with the same causal operator structure, so any learned success relative to the raw-local baseline is an **architecture/basis advantage**, not novel primitive discovery.
3. Each output must verify all 25 observed coefficients in **both** demonstrations before being eligible for future correctness. A mismatch or inconsistent demonstration is an abstention, with the explicit first fault recorded when available.
4. Evaluate full query output through degree 48 with independent exact product and wrapper implementations; do not expose hidden coefficients to the model for refinement. Every collision pair must have identical proposals and exactly one future-correct member.
5. Source SHA-256 hashes, roster hashes, and trained checkpoint hashes are recorded in a byte-reproducible JSON receipt. Standard-library-only CI reruns the tests and compares full canonical receipt bytes. Training replay must reproduce rational weights exactly.

## Interpretation boundary

The hidden operator lies in a **known multiplier family**. Its coefficients are new to the student at inference, and are recovered from demonstrations using a prewritten convolution basis. Thus the experiment tests few-shot *instance identification and composition* of a supplied algebraic operator type. Novel nonlinear operator-family inference and automatic invention of convolution remain open; this is a focused diagnostic after RRC-09.
