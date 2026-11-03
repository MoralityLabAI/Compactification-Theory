# Algernon-01 protocol: evidence gates and falsification

**Draft registry date:** 2026-11-03. The hypotheses below are registered as directions for subsequent experiments. A specific numerical threshold, model candidate, dataset roster and execution hash must be frozen *before* the corresponding experiment is run. The scaffold is not itself a published preregistration of an executed trial.

## Separation of evidence

- **External evidence:** Qwenputer's published RV32 programs and trace checks; Takehi et al.'s published p50c edit and benchmark results. Quote these as externally reported.
- **This repository's evidence:** only commands that we execute, paired with command logs, versions and retained raw outputs.
- **Hypotheses:** trainable grafts, latent direct execution, structural reuse of compiled DeltaNet RAM, 27B transfer and precision-layer repairs. Keep labelled as untested until tested.

## Staged experiment

### Gate 0: CPU-only source and contract
- Verify source and checkpoint revisions and their licenses.
- Inventory actual old CoT-looped/TRM/LDT weights, separated from architecture sketches.
- Freeze one group-disjoint task set with training, calibration and test splits; record SHA256 for each.
- Test register schema and record manifest. Refuse missing/changed revisions.
- Record reference-model CPU exact outputs for small state-transition and arithmetic tests.

### Gate 1: compiled correctness
**H-C:** A compiled exact kernel retains the independent interpreter's input-to-state-to-output transition.

Precommit cases: arithmetic signs, overflow, boundary indexing, reset, branch, invalid input, full trace length, memory read/write. Compare state bytes and output bytes. Match both isolated and batched execution where available. Include a deliberately corrupted instruction as a negative control. Missing or skipped cases are reported separately.

Reference parity of the circuit demonstrates correct execution; it does not demonstrate reasoning or learned adaptation.

### Gate 2: recurrent graft causality
**H-R:** A learned recurrent graft improves group-disjoint reasoning task accuracy at some K > 1, beyond matched-parameter feedforward and extra-token baselines, while preserving the Qwen base skills within a pre-set tolerance.

Candidate sites: Qwen3.5-0.8B layers 8,12,15,19,21. Candidate depths K=0,1,2,4,8. Freeze the site/depth selection using calibration data; evaluate the chosen configuration once on test.

Controls: base model, zero-gate identity, constant-state recurrence, shuffled latent trajectory, random recurrent weights, equal-parameter feedforward bridge and a token-budget matched text-CoT baseline.

Primary report: group-level paired outcome differences with uncertainty, absolute accuracy and total inference-time cost. Report p50/p95 latency, peak GPU memory, per-sample generated tokens, internal refinement steps, model forward operations, KL to base, retention set accuracy and invalid-state rate. Correct for multiple comparisons or quarantine site-search results as exploratory.

**Falsifies the strong graft claim:** equal-compute feedforward performs as well; added steps give no held-out lift; ablations leave accuracy intact; or retention/compute penalties exceed frozen limits.

### Gate 3: sparse editing
**H-W:** A supported receiver repair improves a working graft, relative to support- and norm-matched random edits, with a fixed edit budget and out-of-sample gain.

First independently reproduce the *published fixed* 0.8B patch with the [HybridTRMLDT replication package](https://github.com/MoralityLabAI/HybridTRMLDT/pull/5). Freeze model hashes, roster, prompt and decoding settings. For graft-specific edits, disallow editor access to test questions and evaluate all candidates on an independent untouched test split. Compare random edits at matched BF16 effective support; record rank/norm differences separately.

Primary report: accuracy, token-cut rate, looping, rescue/regression pairs, patch sparsity, per-layer relative Frobenius norm and base-skill drift. Hold out a final set for claims.

### Gate 4: joint processor
**H-J:** Exact in-weight circuit plus learned recurrent graft improves problems requiring both deterministic state manipulation and heuristic search, beyond either component and a host-tool oracle subject to clearly stated interface constraints.

Controls: compiled-only, graft-only, joint, disabled bridge, falsified exact output, shuffled latent and ordinary Qwen plus externally computed answers. Preserve tool accessibility parity and count CPU/host work in each arm.

Primary report: exact execution fidelity, joint task performance, added wall-clock time, GPU-seconds, trace tokens and any extra trainable or nonzero compiled weights.

**Falsifies the strong synergy claim:** the joint arm has no reliable improvement over the better single component at matched compute and information access.

## Evaluation hygiene

1. Train, calibration and test are separated by underlying problem generator / state seed, not just text.
2. Every subject/task group carries a stable identifier; paired bootstrap clusters by group.
3. The evaluator checks format-independent correctness where tasks permit it, and separately tracks format failures and timeouts.
4. Classify each run as simulated, reference-executed, on-device, or published external. Keep metrics for each category distinct.
5. Record interventions as reversible weight/state deltas with SHA256, model and revision; never overwrite original checkpoints.
6. Do not call future inference-time recurrent state updates weight training. If local gradient training is used, register trainable parameter count and training tokens.
7. Keep budget limits, stop criteria and maximum trials in the manifest before launching. No automatic rented GPU sessions.

## Stop conditions

- Nonfinite residual, invalid projection or incorrect exact-machine state: disable graft and preserve a failure receipt.
- Off-by-one trace failure, rounding-induced mismatch or missing checkpoint hash: invalidate that benchmark arm.
- Unresolved rights to training data or unknown retained checkpoint source: exclude the model from the candidate list.
- Evidence limited to calibration: publish as exploratory only.

## Output layout

Each real experiment writes to an ignored run directory, not to this registration:

~~~
runs/algernon/<run_id>/
    manifest.json
    records.jsonl
    metrics.json
    report.md
    source_hashes.json
~~~

The report must explicitly state **what ran**, where it ran, and which claims were learned from the measured results.
