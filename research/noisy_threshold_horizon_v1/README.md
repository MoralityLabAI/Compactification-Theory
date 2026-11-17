# NHB-01: noisy thresholds, horizons and compact decision interfaces

Started November 14, 2026 after merging KSR into Compactification Theory main.

Source papers:
- OpenAI Mathematics [#119](https://github.com/openai/math/blob/main/lean/docs/119.md), *Sharp binary-information contraction on the discrete cube* (September 24, 2026).
- OpenAI Mathematics [#127](https://github.com/openai/math/blob/main/lean/docs/127.md), *Average sensitivity of polynomial threshold functions* (September 25, 2026).

[THEORY.md](THEORY.md) combines their reported information-contraction and threshold-sensitivity results with an exact noisy recurrence. It gives a conditional two-sided information/error guarantee for a compact low-degree monitor with declared uniform approximation error. The elementary derivations also give a sharp terminal-only horizon cutoff, a minimum memory certificate for arbitrary posterior-threshold queries, and a biased-event false-stability control.

## Exact executable models

Python 3.10+ and the standard library are sufficient. The finite benchmarks cover a coordinate, majority, parity, a quadratic product of two majorities and a rare-alert conjunction. Every state is enumerated. We compare factorized noise with a direct transition oracle, exact Fourier spectra with flip influences, optimal Bayes errors with explicit readouts, and the output-state count needed for different decision interfaces.

    python -m unittest discover -s research/noisy_threshold_horizon_v1 -p 'test_*.py' -v
    python research/noisy_threshold_horizon_v1/reference.py --out /tmp/nhb01_exact.json
    python research/noisy_threshold_horizon_v1/decision_memory.py --out /tmp/nhb01_decisions.json

The registered step noise is 1/20; the independent initial target flip rate is 1/10. Horizons are 0,1,2,4,5,8,16. A separate completely noisy rare-event control uses crossover 1/2 and no initial target flips. All probabilities, spectra and influences use exact Fractions. Entropies and inverse-entropy source bounds are numerical and tested with tolerance 2e-11.

The first-order affine surrogate uses the target's complete finite-population Fourier projection. It is an oracle construction, not a fitted checkpoint or a claim of generalization. Bayes and posterior-value codes likewise have static computation/table costs separate from their output-state bits.

## Main application

For looped-transformer or SAE monitoring, a proposed low-degree probe's error, the noise law, and persistent history access must be measured separately. The terminal-only theorem permits no cached target or trajectory. A one-bit memory that already stores the target changes the interface completely. The exact finite benchmark is a control for those future experiments; no native neural checkpoint is evaluated in this round.

The original OpenAI manuscript results are used conditionally. Finite tests do not establish their general proofs, and the supplied comparator declarations alone do not establish kernel verification. The accompanying MathCan bridge covers selected finite recurrence and threshold-separation arithmetic only.
