# COE-03: precision/time observer capacity and controlled quantization pilot

**Date:** 2026-11-15. Follows COE-01 and COE-02 on the existing draft PR.

A finite equilateral `K_n` evaluator with ℓ1 output margin 1 and total output Lipschitz budget 2 requires `n-1` one-bit features. Its continuous scalar observer needs `Theta(log n)` feature dimensions, with an explicit small-amplitude construction. If those outputs must occupy a uniform `b`-bit grid while retaining `O(log n)` coordinates, the necessary and sufficient order of precision is `Theta(log log n)`. Reusing `w` binary output channels over T observable steps requires `w*T >= n-1` when the **whole transcript's total ℓ1 budget is 2**. Each theorem's hypotheses and proofs appear in [MathCan COE-03](https://github.com/patrickdugan/MathCan/blob/feat/coe01-chromatic-observer-20261115/research/compactification/SOFT_PRECISION_DEPTH.md).

The construction has no dependence on the unverified deep proofs in its source-motivating OpenAI Math families 098 and 332. The finite metric is synthetic. The 10-bit outputs are numeric values, not 10 separate unit-amplitude Boolean probes.

## Repeat all checks

Python 3.10+ standard library suffices, from repository root:

```sh
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_*.py' -v
python research/chromatic_observer_gym_v1/soft_precision_codes.py
python research/chromatic_observer_gym_v1/train_soft_probe.py
```

The cumulative suite has 30 tests, including 9 new COE-03 cases. A `Random(2305)` 1,024-state, 256-head, ten-bit-grid code yields Hamming extrema `91,168` over all 523,776 pairs. The rational amplitude is `12/1023`, so minimum and maximum ℓ1 distances are exactly `364/341` and `672/341`. The deterministic code hash is pinned in the test. The binary equivalent requires 1,023 output channels under the **same** ℓ1 budget/margin conditions. These counts describe observer output dimension, not stored parameter or data bits.

The K5 exact continuous three-feature certificate and all 64 coordinatewise threshold patterns are independently checked, along with a binary four-feature witness. The temporal reference packs an n-state one-hot transcript into `ceil((n-1)/w)` steps of width w, and checks the full transcript budget.

An exploratory standard-library training pilot fits a three-output linear probe on five frozen one-hot states. It uses projected pairwise squared-hinge subgradients, seeds 0 to 9, a fixed learning rate and 4,000-step limit. Across ten runs with all ten edges used in training, 10/10 reach tolerance `1e-3`. Threshold at `0.5` yields zero valid binary K5 codes. An exploratory seven-edge train/three-edge withheld evaluation gives 10/10 train and 2/10 withheld passes. The held-out split was explored before commitment; these proportions are diagnostic, not model generalization statistics. Test replay checks them under Python 3.12 CI.

### Work that remains

- Compare a named frozen looped-transformer or SAE checkpoint after extracting latent activation vectors and independently measured evaluator edges.
- Establish a blinded train/test task split before modeling, with quantized and unquantized head budgets fixed.
- Separate observer width, bit precision, wall time and full-transcript memory during recurrent inference.

Only synthetic finite codes and the specified CPU training pilot have been executed.
