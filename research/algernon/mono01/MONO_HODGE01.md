# MONO-HODGE-01: Bayesian evidence order and causal one-shot surgery

**Protocol date:** 2026-11-04. **Execution status:** CPU mathematical prototype tested on explicitly synthetic affine operators. The 27B Hugging Face exports were not readable from the current runtime (the public pages and their raw files returned access errors); **no real-archive analysis or new Qwen edit has been executed**. No H100-hours were spent.

## Public, fixed-source provenance

These two archives come from separate checkpoints and experimental questions. Their existing evaluation sets are now public and cannot be called pristine holdouts.

| Archive | HF dataset | Commit | Prior supplied finding |
| --- | --- | --- | --- |
| ALG-HODGE-27B-01 | [AlephFunk/alg-hodge-27b-01](https://huggingface.co/datasets/AlephFunk/alg-hodge-27b-01) | `09b4e483d9a7bb0f4f36871a2c9523a45623d05a` | Qwen3.5-27B posterior probe MAE 0.9510 nats; shuffled 1.504; constant 1.434; order dependence; tested edit gave no utility benefit |
| M-VOI-01 | [AlephFunk/m-voi-01-qwen3-8-27b](https://huggingface.co/datasets/AlephFunk/m-voi-01-qwen3-8-27b) | `906399c2b7934ca8316498d86ba0a8f4058cd204` | Qwen3.8-27B continuation advantage decodable but weaker than public-state signal; selected edit did not reliably improve stopping |

The numerical findings above are from the operator-supplied experiment handoff, not measurements independently reproduced in this module. `repaired_v2/` is described as authoritative for HODGE; the initial capture is a failed historical run. **Before processing:** examine archive manifests, original schemas, retention notes and licenses; keep failed/superseded evidence separate and never merge rows from distinct versions. Commit fingerprints and actual fetched file hashes in the new receipts.

## Distinguish three different mathematical quantities

**1. Readout order sensitivity.** For a matched evidence multiset `{a,b}` with task-defined exchangeability, compare the two final prompted posterior readouts:

\[
D_{ab}=\widehat L(a,b)-\widehat L(b,a),\qquad
E_{ab}=\tfrac12\big(|\widehat L(a,b)-L^*|+|\widehat L(b,a)-L^*|\big).
\]

The readout gap is observable from paired records and does not identify an internal transition operator. Even `D=0` can accompany a consistently wrong posterior. Equal or balanced token lengths, fixed prompt templates, same evidence multiset, a shared answer parser, and genuinely order-invariant task truth must be checked. Ordering can carry legitimate time or reliability information in other task families.

**2. Order-antisymmetric activation geometry.** When both completed-prompt final hidden states share a valid activation basis at the same hook point, define rows of `D = h_ab - h_ba`, with a nominal reference matrix `X`. The most order-sensitive directions relative to reference variability solve

\[
D^\top D\,v/n=\lambda C v,\qquad C=X^\top X/N+\epsilon I.
\]

Calculate the small Gram kernel `D C^{-1} D^T/n`, using Woodbury to invert `C` without forming a large square matrix. These eigenmodes are **diagnostics**. Removing them can erase useful evidence information, and they are not automatically causal edit directions.

**3. Operator commutator.** With an executable state-transition mechanism `T_e^W` and a fixed, comparable state gauge, define

\[
\Omega^W_{a,b}(h)=T_b^W(T_a^W(h))-T_a^W(T_b^W(h)).
\]

This requires replay or an independently verified transition abstraction. Two final-prompt activation vectors cannot supply the two counterfactual operator compositions, particularly in transformer architectures with position-sensitive caches and Gated DeltaNet state. Measure downstream corrected decisions and preservation even when an operator commutator decreases.

A naïve Hodge "curl" constructed by subtracting a scalar potential along the edges of a graph of *complete evidence permutations* telescopes and is identically closed; it is not evidence of a learned defect. A meaningful discrete-curvature test must instead operate on identified **local evidence-update operators** or on a separately specified, potentially non-integrable predicted increment field.

## One-shot editor and theorem-sized toy

Use a fixed left direction `u` obtained solely from **calibration** causal tests; restrict the weight edit to `ΔW = u z^T`. For an actual replayable transition, approximate the Jacobian of vectorized commutator residual `r(W)` under perturbations along `u e_j^T` by central differences. Compute one Gauss-Newton step:

\[
\boxed{z^*=-(J^TJ+\lambda C_S)^{-1}J^Tr(W),\qquad \Delta W=u(z^*)^T.}
\]

`C_S` is the regularized covariance of nominal activations on allowed input columns. The result is rank one at most. For a nonlinear model, this is a local linearization requiring independent rollout validation, step-size checks, frozen support and a reference-retention test.

The CPU demonstration uses the **synthetic** affine evidence operator

\[
T_e^W(h)=(I+W)h+e.
\]

Its commutator is exactly `W(a-b)`, and the test places a rank-one defect in one evidence-sensitive column while keeping an independent nonzero transform in other coordinates. The ground-truth target and causal left direction are known by construction. Therefore the demonstration checks the solver and cannot be interpreted as discovery of a real model circuit.

**Non-collapse condition:** Always measure posterior accuracy and downstream utility alongside order invariance. A constant posterior readout can have zero order gap and poor Bayesian accuracy. Require independent functionality and base-capability retention tests before claiming a useful edit.

## CPU commands

From repository root with NumPy installed:

```bash
python -m unittest discover -s research/algernon/mono01/tests -p 'test_hodge01.py' -v
python research/algernon/mono01/hodge01.py
python research/algernon/mono01/hodge01_analyze.py --toy
```

To map the HODGE archive, **inspect and explicitly normalize its authenticated `repaired_v2/` paired readouts** into JSONL records with fields:

`family_id`, `log_odds_ab`, `log_odds_ba`, `true_log_odds`, and optional numerical `state_ab`, `state_ba` vectors measured at the same layer and hook location.

Use a disjoint nominal-state `.npy` with shape `[N,d]` for the optional spectral diagnostic:

```bash
python research/algernon/mono01/hodge01_analyze.py \
  --pairs normalized_pairs.jsonl --reference nominal_states.npy --output hodge_results.json
```

For M-VOI, first construct an `.npz` **with `allow_pickle=False`-compatible arrays** named `public_state` `[N,s]`, `latent` `[N,z]`, `continuation_advantage` `[N]`, and `family_id` `[N]` (Unicode/string array). Run:

```bash
python research/algernon/mono01/hodge01_analyze.py \
  --voi voi_normalized.npz --output voi_results.json
```

The M-VOI score is the paired improvement in **group-held-out mean squared prediction error** from adding the latent variables to a public-state ridge predictor. This is a predictive incremental-value test, not a conditional mutual-information estimator or a causal attribution.

## Frozen next experimental gates

1. **Archive inspection, CPU:** inspect both Hub commits and choose canonical capture files, identifying field keys, row counts, family splits, activation dimensions, available probe checkpoints, and the task generator. Preserve a machine-readable mapping from source bytes to each normalized row. Review HODGE's `repaired_v2` and its matched-batch limitation. Set aside the failed initial capture as historical.
2. **Observational calculations, CPU:** compute order gap and posterior error on matched, task-exchangeable pairs. Bootstrap by independent generator family. Estimate whitened order-sensitive modes with nominal activations from the same checkpoint. For M-VOI, cross-fit state-only and state-plus-latent predictors with family-disjoint splits and report out-of-fold errors. This is retrospective descriptive work; publication used the original evaluation roster.
3. **Causal intervention, 27B GPU only after precommit:** obtain an executable, cache-correct Qwen3.5-27B checkpoint and hook; replay transitions at fixed state/position conventions. Establish whether relevant order defects can be altered and whether the readout and utility move beneficially. Use a fixed support/direction and a single finite-difference update, with matched random/norm controls, restored base, and untouched **newly generated** final families.
4. **Cost gate:** declare a strict total H100-hour budget, local preprocessing bill, token and rollout count before dispatch. This note authorizes **zero GPU expenditure**.

**Falsification:** ordered readouts prove irreducibly prompt-semantic; detected modes fail intervention; commutator reduction destroys posterior accuracy; group-held-out predictive gain disappears; or edited decisions remain unchanged. Preserve negative results.

## Prior experiment caveats

The old HODGE report describes an initial isolated-prompt control failure, followed by a repaired **matched-batch** comparison. That experiment cannot be silently re-described as an isolated-prompt intervention. The old reported 31.67 H100-hour debit is a conservative ledger value and is not a measured bill. Neither that budget nor the two already-public test rosters constitutes new evidence from MONO-HODGE-01.