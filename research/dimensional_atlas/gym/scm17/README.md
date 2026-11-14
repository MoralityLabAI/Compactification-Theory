# SCM-17: trained-RMP SAE + local margin-fiber transfer

**Status (November 14, 2026):** executable interface and CPU mock test, trained RMP evaluation gated by external weights. This research track lives on top of SCM-16 PR #137. It changes neither the existing RMP gates nor the checkpoint files. SCM-17 creates a distinct trainable adapter with frozen model and dictionary parameters.

## Why a new kind of fiber

SCM-16 used a deliberately low-rank *linear* head to obtain a known exact final-readout kernel. RMP's final head instead is `F.linear(final_norm(h), embedding.weight)`. The direct final decision margin is nonlinear. SCM-17 therefore fits a **label-blind local logit-margin Jacobian frame** on the selection half of existing `monitor_calibration`. For each clean task, take the top-1 predicted logit versus its next two logits, differentiate those margins with respect to the *final pre-readout answer-position hidden state*, concatenate the gradients, then take the top eight right singular vectors (or fewer). Their orthogonal complement is the **approximate local decision-silent fiber**.

The frame is frozen before SAE-conditioned VPD training. A rank cap can omit gradients: the captured gradient-energy fraction is recorded; the launcher refuses below the minimum 0.75. A terminal fiber patch need not leave **all** logits or margins exactly invariant. In particular, finite patches can change a margin at second order because of final LayerNorm. The campaign reports actual causal effects rather than assuming them.

For a retained frame `Q`, the shared VPD correction is

    raw_v = (((h_v @ V) * tanh(R a_v) * tanh(g_v)) @ U) * scale
    visible_v = (raw_v @ Q) @ Q.T
    fiber_v = raw_v - visible_v

with fixed `D-shared` SAE `a_v` and visit-conditioned trainable gates. The joint, visible-only, fiber-only, visit-reversed and zero-SAE-code arms use the *same trained adapter*. The recurrent source's parameters and dictionary remain frozen. The trained adapter changes the effective operation at successive visits and is a VPD-inspired low-rank correction, rather than a recovered Goodfire weight decomposition.

## RMP integration

SCM-17 calls the authentic `research_gym.interp.phase2.load_model` and `core_examples`, then `AnalysisModel.run(tokens, hook)`. The hook sees each `context.after` **after RMP's FF-U input reinjection**, leaving the source implementation untouched. The `LOOP-T s1` and `FF-U s3` configurations are the original C3 cells, with explicit source hashes and integrity sidecars.

The campaign uses `monitor_train` to update the adapter. It fits the Jacobian frame on eight group-distinct examples from the selection half of `monitor_calibration`. The reporting half of `monitor_calibration` is scored after freezing. These are previously examined RMP data, so success there is exploratory; this run never opens the sealed test. Sample selection is balanced across pointer chase, gated pointer chase, reachability and Sudoku.

### Preflight on the RMP host

From a `Compactification-Theory` checkout containing SCM-17, with a sibling `LoopedTransformers` checkout and its existing local artifacts:

```bash
python -m research.dimensional_atlas.gym.scm17.campaign --mode preflight \
  --model 'LOOP-T s1' \
  --rmp-root ../LoopedTransformers \
  --runs ../LoopedTransformers/experiments/recurrence_monitorability_v0/runs \
  --artifacts /path/to/rmp_v0/artifacts \
  --monitor-bundle /path/to/rmp_v0/bundle \
  --dictionary /path/to/selected/shared_dictionary.pt \
  --metrics ../LoopedTransformers/experiments/recurrence_monitorability_v0/phase2/dictionaries.json \
  --output /path/to/new/s17-loop.json
```

Repeat with `--model 'FF-U s3'` and its own selected dictionary and output. The manifest also pins the local SCM-17 adapter and campaign source, so later code changes invalidate an existing lock. The preflight returns `blocked_external_artifacts` and precise missing labels when resources are absent. When preflight is ready, run `--mode freeze --lock-file /path/to/new/scm17.lock.json` to freeze source hashes. For training, use `--mode train --lock-file /path/to/new/scm17.lock.json --device cpu --steps 60`. It refuses output overwrite, checks the RMP source and checkpoint receipt, loads the D-shared dictionary with `weights_only=True` and validates the registered selection metadata. Heavy outputs and adapter weights remain outside GitHub.

## Local and hosted tests

```bash
python -m unittest discover -s research/dimensional_atlas/gym/scm17/tests -v
python -m research.dimensional_atlas.gym.scm17.reference --check \
  --out research/dimensional_atlas/gym/scm17/results/scm17_mock_reference.json
```

The reference is a randomly initialized, simplified recurrent decoder implementing the same visit-hook contract, with a nonlinear LayerNorm head and a frozen TopK encoder. It is **not an RMP trained checkpoint**. The experiment is a bridge and a causal-null control, rather than a measured transfer result. See [PROTOCOL](PROTOCOL.md), [THEORY](THEORY.md), [RESULTS](RESULTS.md).
