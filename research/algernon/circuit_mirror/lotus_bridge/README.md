# Algernon-02B: LOTUS post-loop circuit mirror

**Execution status (2026-11-04):** the A2b GPU run on the published 3B checkpoint is done: [results/algernon-02b/report.md](results/algernon-02b/report.md). It reproduced GSM8K (0.650 on 200 items) and passed bitwise hook parity, and the registered Phase 4 gate failed (`no_iterative_state_signal`). Held-out decodability at depths 9-16 is 0.185 for the primary recurrent view against 0.341 for the prompt-only start token, and LOTUS's own pointer answers are at chance. The A2c follow-up (2026-11-05, [results/algernon-02c/report.md](results/algernon-02c/report.md)) trained upstream's GPT-2 LOTUS recipe on the pointer task. Both arms learn states decodable within the trained depth, but no transition carries past it, and both fail the same gate. All reports are indexed in [results/README.md](results/README.md). Constraint-guided correction and any Circom/RV32 execution **have not run**. The separate 222-parameter toy pilot used privileged multiplicative features, not LOTUS representations.

## What this instrument actually captures

Upstream [LOTUS](https://github.com/yingfan-bot/lotus/tree/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e), `scripts/lotus.py`, implements an initial prefix pass followed by `n_looped_iters` refinements of the latent-token region. Each refinement passes the previous loop-region hidden states through the original embeddings and calls `efficient_forward`. The `intercept_loop_states` context records **both** the last latent-position vector and the mean of the final `c_thought` latent-position vectors after each call, including the initial pass. It restores the original function even after an exception.

This is instrumentation of the published network's internal activations. No new loop is implemented in Algernon, no LOTUS weights are altered, and the exact pointer-machine constraints are used only for *post hoc evaluation*. The capture path discards full-vocabulary intermediate logits, which the LOTUS forward uses for optional visualization and losses. A short two-loop parity check compares hidden vectors with and without this optimization before collecting the roster.

### A2b: block-aligned capture (registered in [EXPERIMENT.md](../EXPERIMENT.md#h-a2b-block-aligned-capture-causal-controls-and-the-phase-4-gate-registered-2026-11-04-before-any-lotus-gpu-run))

Every loop refines *all* latent blocks, and the published configuration supervises block *b* with CoT step *b* after the final loop, so a reasoning step lives in a block rather than in a loop iteration. `capture.capture_case` therefore records, for every pass, every 25-token block (last latent and mean), the start-latent token (prompt-only, never re-injected) and upstream's own intermediate-logit argmax over the loop region. With generation on, these are the states the model's own answer is decoded from.

| Module | Role |
| --- | --- |
| `counterfactual.py` | Frozen relabelled copies of the 48 held-out cases (`frozen_counterfactual_6x16.json`): map conjugated by a fixed derangement, trace relabelled. Never equal to a train or calibration instance; sink duplicates of other held-out cases are listed. |
| `gpu_run.py` | One gated GPU job: unmodified upstream `eval.py` on 200 GSM8K test items, weight fingerprint, full-depth hook parity (bitwise), two-case smoke, 16-case calibration subset, full roster, relabelled roster, answers at 6 loops and with no latents, drift recapture. |
| `analysis.py` | CPU decoders for the `diagonal`, `final`, `lastblock`, `pass0` and `static` views, the registered controls and symbolic baselines, the Phase 4 gate, `metrics.json` and `records.jsonl`. |

Upstream pins `transformers==4.46.2`; its `_clone_kv_cache` uses `DynamicCache.key_cache`, which later transformers releases removed, so run LOTUS in an environment with that version (torch 2.7.0+cu128 here).

```bash
python -m research.algernon.circuit_mirror.lotus_bridge.gpu_run \
  --upstream ../lotus \
  --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json \
  --counterfactual research/algernon/circuit_mirror/lotus_bridge/frozen_counterfactual_6x16.json \
  --hf-revision b392d2cb7aaa73475b93028221523c47f49f66a2 --run-dir runs/algernon/lotus-02b
python -m research.algernon.circuit_mirror.lotus_bridge.analysis \
  --run-dir runs/algernon/lotus-02b \
  --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json \
  --counterfactual research/algernon/circuit_mirror/lotus_bridge/frozen_counterfactual_6x16.json \
  --out runs/algernon/lotus-02b/analysis
```

`gpu_run.py` never downloads: the pinned snapshot must already be in the Hugging Face cache, and `HF_HUB_OFFLINE=1` is set for the upstream evaluation subprocess. The upstream GSM8K test file comes from `preprocessing/gsm_icot.bash` (convert with UTF-8 I/O on Windows).

### Scientific question

Can a **single linear decoder** trained on actual LOTUS post-loop activations for transition depths 1–8 recover machine states at unseen depths 9–16, on held-out graph-generator families? We measure:

- state accuracy at each depth;
- commuting-square defect `decode(h[t+1]) != successors[decode(h[t])]`;
- whole-trace acceptance by the independent algebraic constraint checker;
- changes under repeated stage-zero and shuffled-stage controls.

These diagnostics establish whether pretrained LOTUS activations carry useful structure for this out-of-domain pointer task. A positive result would motivate training a latent-to-Circom feedback bridge; this stage itself does not establish a constraint-guided reasoning circuit. LOTUS's published checkpoint was trained on arithmetic word problems, so this transfer is exploratory.

## Frozen data and code

- `frozen_roster_6x16.json` is generated by `roster.build_roster(seed=20261104,n=6,steps=16,train=80,calibration=24,test_per_family=24)`: 152 unique `(successor_map,start)` instances.
- Training: `random_function`, calibration: `permutation`, held-out test: `long_cycle` and `sink`, with no reused mapping/start pair across splits.
- Same input prompt and same 400 latent slots on every row. It includes successor map and starting state, never the answer trace.
- Probe is trained only on stages 1–8 on training-family cases, while pooling choice uses calibration-family stages 1–8. The held-out families and depths 9–16 remain outside both optimization and model selection.
- Prior circuit-mirror exact verifier and Circom-source emitter are reused unchanged. The compiled Circom witness / SNARK gate remains open.
- LOTUS source commit is `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`, under the upstream MIT license. Checkpoint: `yingfanbot/gsm-lotus-llama3b`, its **resolved Hugging Face revision is frozen at execution time** and saved in the manifest. This folder does not bundle model weights.

## CPU-only commands

Run from `Compactification-Theory/`:

```bash
python -m unittest discover -s research/algernon/circuit_mirror/tests -v
python -m research.algernon.circuit_mirror.lotus_bridge.roster \
    --out /tmp/roster.json
```

The generated roster should match `frozen_roster_6x16.json` byte for byte under the same JSON serializer and command settings. The strict roster hash is also checked on every training/evaluation run. No GPU or external model needed for these commands.

## Actual LOTUS checkpoint extraction (GPU host)

Create a separate upstream checkout, pin the commit, install its documented environment, and run on CUDA when ready. Work within a **fixed memory and compute budget**; a 3B BF16 model on an RTX 5080 has not been measured by this project yet. The script never automatically rents cloud GPUs.

```bash
git clone https://github.com/yingfan-bot/lotus.git ../lotus
git -C ../lotus checkout --detach eb77e2f7909c5006f58ff0ad7cd6629b942caa9e
# Install the pinned upstream environment (see ../lotus/README.md).

# Start with two cases. Generates an explicitly incomplete smoke manifest.
python -m research.algernon.circuit_mirror.lotus_bridge.capture \
  --upstream ../lotus --allow-download \
  --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json \
  --loops 16 --c-thought 25 --limit 2 \
  --out runs/algernon/lotus/smoke.npz

# After the smoke fits the available VRAM, do the complete fixed roster.
python -m research.algernon.circuit_mirror.lotus_bridge.capture \
  --upstream ../lotus --allow-download \
  --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json \
  --loops 16 --c-thought 25 \
  --out runs/algernon/lotus/full.npz

python -m research.algernon.circuit_mirror.lotus_bridge.probe \
  --roster research/algernon/circuit_mirror/lotus_bridge/frozen_roster_6x16.json \
  --features runs/algernon/lotus/full.npz \
  --out runs/algernon/lotus/probe_report.json
```

The capture tool resolves and freezes the exact Hugging Face commit **before** loading it. For offline use, supply `--hf-revision <40-hex-sha>` and ensure the checkpoint is already cached. An explicit `--allow-download` is required before any model download. Logs include input token hashes, per-case times, feature byte hash, source revision, model precision, and peak allocated GPU memory. The readout script refuses incomplete smoke artifacts and any roster, feature-byte, or ordering mismatch.

## Evaluation interpretation

The linear probe sees features that come from a model with full access to the problem's successor map. Successful classification can therefore arise from decoding an ordinary input feature rather than latent algorithmic iteration. Constant-stage and shuffled-stage controls test for iteration dependence, with an exact-solver ceiling. Further work should also compare to an input-only feature probe and a similarly budgeted explicit-CoT baseline before claiming general algorithmic scaling.

Predictions are **never corrected** using the exact verifier. Every candidate trace is checked against its public successor map and start value, and the verifier reports invalid traces, even when the final node happens to coincide with the true answer. Stage-16 accuracy and defect rate are emphasized; every intermediate depth is retained.

### Risk boundaries

- This uses an external checkpoint trained for GSM8K arithmetic, so no guarantee of useful pointer representations is implied.
- The memory-saving hook changes *only the returned intermediate logits*, restoring the unmodified upstream function after each call. The first-case parity test must succeed on-device.
- Deep recurrence (`n=16`) exceeds the checkpoint's documented default six-loop regime; failure or activation blow-up is a legitimate negative result.
- The 400 latent tokens and 16 recurrent passes may be costly; execute one smoke case and inspect VRAM, latency, model loading and CUDA compatibility before the full roster.
- Publicly readable frozen labels are for evaluation. Any future weight-space optimizer must use fresh, inaccessible final test instances and a newly frozen protocol.
