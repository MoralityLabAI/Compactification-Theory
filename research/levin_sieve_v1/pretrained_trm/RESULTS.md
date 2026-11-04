# LSV-05/06 first pretrained checkpoint evaluation, 2026-11-04

**Executed:** [GitHub Actions run 41788086893](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41788086893), 10 local tests plus the independent pretrained checkpoint benchmark, successful on CPU with PyTorch `2.7.1+cpu`. **Status:** evaluation of a third-party TRM-inspired 44,001-parameter network, not the canonical 7M TRM and not a Qwen checkpoint. Independent training occurred upstream. Our benchmark did not retrain the model.

- Upstream revision: `f0406195fece4d0db29ced28f3a69d1993d95b1e`.
- Checkpoint Git blob: `dfbe8ffbeb52f2c32f0c44fd326cd26c96272c54` (181147 bytes).
- Checkpoint SHA-256 from successful CI, now pinned: `3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1`.
- Frozen 64-puzzle roster digest: `eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611`.
- [Original raw CI artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41788086893/artifacts/13302461345).

| Clean recurrence depth | Exact solved / 64 | Accuracy on 30 missing cells |
|---:|---:|---:|
| 1 | 0 | 81.9792% |
| 2 | 2 | 91.6146% |
| 3 | 13 | 94.2708% |
| 4 | 20 | 95.3125% |

**Damage after the first pass.** Inject uniform hidden noise with max absolute amplitude bounded by the registered value, then run the three remaining passes. First, for seed 34, amplitudes `0.02 / 0.10 / 0.50` yielded `20 / 19 / 15` exact solved out of 64. For seed 89, the respective counts were `20 / 21 / 17`. Each run holds puzzle roster and model weights fixed. The maximum absolute latent difference after the last pass was `0.126 / 0.555 / 1.992` in seed 34 and `0.087 / 0.443 / 1.955` in seed 89. These final differences exceed the injected magnitudes, so the hidden dynamics are **not observed to contract** on these traces. Relative task robustness is an empirical observation, not a proven recurrent repair guarantee.

**Stale whole-state substitution.** Replacing one intermediate update with an older stage-one state produced exactly 13/64 solved, equal to the clean 3-step result. This controller loses one effective update, so it tests reduced refinement rather than the coordinate-wise delayed contract from LSV-02.

**Storage rounding.** Rounding all frozen weights to BF16 and converting back to FP32 for matrix operations changed 2 of 5184 cell predictions. It retained 20/64 complete solutions, and missing-cell accuracy was 95.4167%. This isolates weight-storage quantization from BF16 arithmetic kernels. The exact-rational global upper bound on readout-logit differences is about `1.08e43`, whereas the largest observed FP32 logit difference was `0.16163`; this massive slack prevents a useful *global* numerical accuracy guarantee. The mathematical bound excludes runtime roundoff.

**Provable unconditional state bound.** Every exact-real post-LN2 coordinate is bounded in magnitude by `8.828797` under original stored FP32 weights or `8.820313` under BF16-stored weights. This follows from the LayerNorm architecture for arbitrary inputs. The independent proof certifies bounded state magnitude, with a separate output-goal readout stability test in [LSV-06](MARGIN_CERTIFICATE.md).

The follow-up CI run using the fully pinned SHA-256 and margin checker is tracked separately. This first record remains unchanged so the analysis is traceable.

## LSV-06 exact rational post-hoc readout certificates

**Second successfully completed [CI run 41788844627](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41788844627)** rechecked the locked upstream checkpoint SHA-256 and passed **13/13 CPU tests**. The [immutable run artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41788844627/artifacts/13302457818) and [committed receipt](results/pretrained_trm_lsv06_20261104.json) contain every evaluated noise stream, including the raw certificate counts.

| Hidden damage (after first recurrence) | Noise seed | Exact Sudoku solved / 64 | Readout-stable cells certified / 5,184 | Observed same exact affine label / 5,184 |
|---:|---:|---:|---:|---:|
| 0.02 | 34 | 20 | 5,176 | 5,181 |
| 0.02 | 89 | 20 | 5,174 | 5,180 |
| 0.10 | 34 | 19 | **5,124** | 5,174 |
| 0.10 | 89 | 21 | **5,124** | 5,173 |
| 0.50 | 34 | 15 | 4,860 | 5,129 |
| 0.50 | 89 | 17 | 4,874 | 5,127 |

All numbers are on the *same* frozen 64-puzzle roster and hash-verified model. The post-hoc bound uses the final hidden error measured per cell. It proves stability of that fixed affine decoder inside the measured error ball; it gives **no ex-ante bound on what hidden error the recurrent model will produce**. The global recurrence-Lipschitz estimate is far too conservative for that. Hidden-state error after 0.5 damage reached a maximum close to 2 after the remaining passes, while the overwhelming majority of cell predictions retained their clean labels. This motivates quotient-/goal-aware certification of the network instead of requiring strict hidden-space contraction.
