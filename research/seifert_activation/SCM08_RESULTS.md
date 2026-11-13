# SCM-08: exact-real source clean tubes, CPU receipt

**2026-11-13.** Python 3.13, PyTorch CPU, SymPy 1.14.0. No RMP trained checkpoint or D-shared dictionary was available. The experiment tested a four-dimensional RMP-like synthetic decoder with LayerNorm, explicit causal softmax attention, unbiased exact-GELU MLP, tied or untied blocks, optional fixed embedding reinjection and real sine/cosine positional encoding over a frozen float32 frequency table interpreted as exact dyadic input.

## Executed checks

`python -m unittest discover -s tests -q` passed **160/160 tests**: 130 historical SCM-01 through SCM-07 cases plus 30 new SCM-08 cases. Dedicated input tests verify rational pi/Machin bounds, sin/cos Taylor intervals, LayerNorm point enclosures, rounding propagation and input-token source hashing. Clean tube tests cover beta=0, tied and untied recurrence, FF-U reinjection, exact attention image bounds, source-derived RMS and mathematically proven null-braid cases. Auditor tests verify refusal on mismatched task hashes and output overwrite.

A 128-dimensional, 64-token input bootstrap completed in 7.41 seconds on this CPU host and produced a maximum rational center uncertainty of 7.87e-16 in the ideal source graph. This performance probe does not execute an eight-visit trained model.

The deterministic reference is in `results/scm08_cpu_reference.json` and records the source SHA-256 of the three new Python modules. It is produced by `scm08_reference.py` with seed 617.

| Untrained synthetic arm | Max clean tube radius at visit 3 | Source-global pair upper, visit 3 | Source-derived local pair upper, visit 3 | Source-only ideal null-braid visits |
| --- | ---: | ---: | ---: | --- |
| LOOP-T, beta 0 | approximately 0 | 0 | 0 | 3 / 3 |
| LOOP-T, beta 0.01 | 0.001433 | approximately 1.4e-5 | approximately 1e-8 | 3 / 3 |
| LOOP-T, beta 0.1 | 0.014327 | approximately 3.5e-4 | approximately 3e-7 | 3 / 3 |
| FF-U with reinjection, beta 0.01 | 0.001445 | approximately 3e-5 | approximately 1e-8 | 3 / 3 |

The table gives order-of-magnitude descriptions; the source JSON records exact rational upper bounds. Each null-braid result concerns the untrained ideal real graph and its specified fixed readout chart.

## Evidence boundary

The clean enclosure proof is independent of captured float activations for its defined exact-real graph. The source frequency-table hash is a declared model parameter; producing identical tables on a different CPU/GPU must be verified separately. The source-only scheme uses coarse attention and MLP image bounds, so larger trained models and late visits may yield wide tubes. The numerical PyTorch activations, actual sin/cos and GELU implementation errors and real-model plotted coordinates remain outside this certificate. No neural topology or compression gain was measured.

## Larger-shape source-only negative control

`python scm08_large_reference.py --out /tmp/scm08_large_ref.json` reproduces an untrained model with 128 hidden dimensions, four heads, 64 tokens, eight tied visits and beta 0.25. The source-only clean-state tube reaches a maximum radius of **17.498** at visit eight. At nominal relative injection radius `10^-6`, the source-global projected pair derivative bound is **6.8731e26** and the SCM-08 derived local bound is **2.6833e8**. All eight independent null-braid tests refuse certification. The source-only computation is a sound mathematical enclosure for the specified ideal real graph but remains much too conservative to prove the synthetic late-visit topology. The executable reference and JSON live in `scm08_large_reference.py` and `results/scm08_large_cpu_reference.json`; a dedicated CI test checks the refusal behavior.

This negative result motivates a source-verified approximate trajectory with independent local residual/error accounting, rather than repeated global attention and MLP image bounds.
