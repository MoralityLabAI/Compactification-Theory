# FHC-01 experimental protocol

Status: prospective; no trained model experiment is claimed here.

## Target

Measure minimum empirical encoder dimension at distortion D as the
future observation horizon T and semantic state-cover resolution change.
Define Phi_T(h)=(w_1 G(F(h)),...,w_T G(F^T(h))) with explicit bounded
observable G, recurrent map F, nonzero weights and declared task input.

The metric d_T(h,k)=||Phi_T(h)-Phi_T(k)||_2 may be a pseudometric.
When the infinite future limit is used, require square-summable weights.
Raw hidden-state Euclidean geometry already lives in a fixed dimension;
the #098 compact doubling obstruction is relevant only after the
future-observation geometry is independently shown to have those properties.

## Frozen variables

Store checkpoint/tokenizer SHA-256 and full-state accounting, including
external clock, prompts, caches and recurrence. Freeze random seeds,
independent prompt families, intervention grammar and future metric.
Hold out complete prompt/trajectory families before fitting embeddings.

Sweep T=1,2,4,8,16 and supported widths d=2,4,8,16,32,64.
Prespecify D=1.5 and D=2.0. Compare dimension-matched PCA, Gaussian
projection, SAE and nonlinear encoders at the same parameter budget,
precision, fitted sample count and compute. Record any lookup table
and stateful encoder side memory in the RC-01 resource ledger.

## Interpretation

On each finite held-out set, report pairwise semantic distances,
encoded distances, a single calibration-only global scale,
worst-case multiplicative distortion and every violating pair.
A random finite sample supplies an empirical statistic. Uniform
guarantees require a proved epsilon-net or explicit covering certificate.
Any one violation certifies failure of that particular candidate.

The reversible shift-register control gives exact linear rank
min(T,m). Confirm it before neural trials. Use a non-doubling equilateral
finite packing control to check the volume obstruction. No experiment
can claim recreation of #098 without reproducing its compact witness.

A future prospective model-level finding requires new frozen training
seeds and a decision rule published before inspecting outcomes.
