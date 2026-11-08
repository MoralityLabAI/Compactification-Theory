# Silico work order: TRSI-BONSAI-SFT-01

Work in MoralityLabAI/Compactification-Theory, under `research/algernon/terminal_rsi_bonsai_gym/`. Record the actual checkout commit before editing. Preserve earlier Algernon results and its external-baseline pivot. The research seed is https://arxiv.org/abs/2610.22192; verify the paper/version and distinguish its published method from the extensions below.

## Objective and authority

Prepare verified terminal-agent trajectories and test whether an SFT adapter transfers useful teacher behavior into a low-bit 27B student for the user's 16 GB RTX 5080. The target is a candidate low-bit runtime; architecture, tokenizer and train/serve compatibility must be measured. Teacher hints are Terminal-RSI-27B; student hints are Bonsai/Bonsai 2 27B; original-base hint is Qwen3.8-27B. None of these hints is an immutable model pin or an availability claim.

This commit authorizes CPU preparation only. `model_manifest.json` defaults to zero GPU seconds, zero paid calls and zero optimizer steps. Stop before downloads, paid teacher inference or training until a separate authorization record supplies finite limits. Do not launch from CI. Never use production credentials or funded transactions. Record blockers and completed CPU artifacts when an authorization or required checkpoint is absent.

## Phase 0: reproduce the instrument

Run the 30 gym tests and smoke command in README. Require 120 fixture episodes, 60 scripted successes, zero model-absent successes and an exported training-only fixture set. Preserve rejected traces. Test a forged-success record, recomputed-hash tampering, a holdout relabel and the public-pass/private-fail control. Fixtures are marked ineligible for learned training.

## Phase 1: inventory exact model and data provenance

Resolve teacher, original base and student from publisher-owned model cards and accessible weights. Pin repository revisions and weight/config/tokenizer SHA-256 values; record license terms and reviewer. Record tensor shapes, vocabulary and chat-template conventions. Treat binary, ternary and unpacked/quantized training representations as separate candidates. Published GGUF inference support does not establish packed backward support.

Recover the actual existing Bonsai adapter method from BitAgent artifacts. Claims of decoded tensor parity establish only the measured decode operation. Require a frozen-base forward pass, adapter gradients with a nonzero finite norm, an optimizer update that changes only approved adapter tensors, save/reload parity, and parity against the actual exported deployment representation. Save hardware, compiler/kernel, runtime and conversion commits. Add numerical tolerances before seeing parity results; keep zero-adapter and round-trip controls. A representation fallback gets its own arm and cannot inherit a native-packed result label.

Populate model_manifest.json and store six reviewed, hashed evidence receipts for forward, adapter backward, save/reload, export parity, tokenizer mapping and tensor mapping. Run preflight. Hash checks establish provenance consistency; a reviewer must assess the evidence contents.

## Phase 2: connect a native gym behind a host process

Keep this fixture API reproducible. Add a separate adapter around a pinned terminal benchmark or existing licensed repository-maintenance tasks. Use isolated ephemeral containers with read-only task/verifier mounts, no host socket, no production secrets, no network by default, cgroup/process/time/output ceilings, and a clean workspace per root. Test escape surfaces before permitting untrusted generated shell commands. Model execution and tool execution must be separate processes with restricted channels.

Adapt existing Control-Harness INSPECT/ACT/VERIFY contracts after reading their current schemas. The host owns permissions, approval nonces and event boundaries. An independent final verifier sees protected state. Actor messages cannot supply a success label, mutate verifier code, choose a new split or mint approvals. Public tests may guide recovery; strict success remains protected. Preserve a model-absent run and the strongest available deterministic baseline.

Freeze root/template lineage and family-disjoint evaluation manifests before teacher sampling. Preserve variants and rewrites in their original root split. Keep all original failures, timeouts, malformed calls and denials with reason codes. Record expected versus observed coverage by family. Treat imported public traces as their own source arm with license and contamination checks.

## Phase 3: bounded data canary after authorization

Proposed ceiling, subject to a smaller signed authorization: one GPU, 20 wall minutes including model load, 32 teacher episodes, 250,000 generated tokens across every retry, and 32 optimizer steps. Charge teacher generation, rewrite attempts, rejected candidates, evaluation and loading to the same ledger. Stop on the first exhausted ceiling and retain partial output. No automatic extension or second job.

Use family quotas fixed before sampling. Generate raw teacher traces with public context only. Rewriting may use diagnostic guidance in a separate source arm; all rewritten procedures must then succeed in a fresh public-only workspace with new synthetic approvals. Private hints, protected tests and verifier results never enter SFT targets. Keep an acceptance-stage record for each attempted trajectory rather than retaining only successes. If a family has zero accepted samples, retain a support-gap record.

## Phase 4: SFT and exports

Implement the smallest adapter update supported by the parity gate. Candidate starting configuration: rank 8, alpha 16, dropout 0, microbatch 1, gradient accumulation 8, sequence length 512, learning rate 2e-5, five paired seeds [11,23,37,53,71] for the later comparison. The bounded canary uses seed 11 only. Confirm target module names from the pinned architecture. Parameters are proposals to freeze after CPU inspection and before training.

Optimize assistant-action cross entropy over verified public transcripts with padding and system/user/tool tokens masked out. Unit-test actual token-level masks against the pinned chat template. Keep a small causal-prefix check to ensure no future tool observation enters an earlier action's input. Tool responses and rejected attempts remain diagnostic data. They are not positive assistant targets.

Baseline distillation is supervised behavior cloning. Optional teacher KL requires compatible tokenization and measured teacher distributions. Log top-k truncation mass and alignment conventions; never substitute a scalar agreement score for KL. Terminal reward optimization and critic updates belong in later separately registered experiments.

Evaluate the frozen untrained low-bit student before updates, then save adapters and test exact reload plus deployment export. Measure actual peak VRAM with context/cache/buffers on the target runtime. The user's 16 GB target is a measured acceptance constraint. Report model load and adapter memory separately. Keep adapter composition disabled until independently checked.

## Metadata contract

Store run/config/model/dataset/host/verifier hashes; root and family lineage; split; training stage; checkpoint selection rationale; train and serve quantization; action/rejection/effect traces; boundary counters; public-test and protected-final outcomes; partial-check counts; stop reasons; token usage; latency; measured GPU seconds and peak memory; retry totals; and source/provenance labels. Use null plus an unavailable reason for unmeasured fields.

At explicitly sampled steps record selected-token log probabilities, action entropy where supported, adapter gradient norms, loss components, masked token counts, learning rate and optimizer step. Optional layer probes store module names, sampling schedule and aggregate norms or low-rank sketches with capped bytes. Preserve RNG states and probe overhead. Full hidden-state dumps need separate storage authorization.

Critic diagnostics remain offline/read-only in this SFT canary. Record strict-success predictions and family-conditioned pass-count distributions only when a critic exists. Join outcome-reweighting analysis to `../critic_handoff_01/`; compare token/event clocks offline using HybridTRMLDT's reference. Keep monitor choice fixed across actor snapshots; any refreshed monitor needs a separate calibration arm at a declared FPR target.

## Experiment and promotion plan

After the canary, propose a paired 2x2 experiment: raw versus verified rewritten demonstrations, crossed with the selected training representation versus actual low-bit serving representation. Compare frozen base, trained adapter and model-absent/strong script controls on identical roots and token/turn budgets. The native teacher provides a reference with its full acquisition cost recorded.

Primary outcome is protected strict completion on sealed roots. Report family breakdowns, length/horizon slices, unauthorized attempted actions, unauthorized effects, parser failures, recovery and coverage. Use paired root-clustered intervals and include every planned seed. Select checkpoints on validation only and spend the sealed test once. Reused attacks under different seeds remain clustered. A small clean canary cannot certify a rare-risk target.

Stop promotion for any unauthorized effect, provenance/split leak, adapter export mismatch, budget violation or missing receipt. Also stop when the trained model fails to beat the strongest simple baseline within the preregistered effect margin. Preserve negative results and write a next-action decision bounded by evidence. Successful local preparation does not authorize scaling.

## Deliverables

Return the exact source commit, reviewed model manifest, compatibility receipts, frozen split hashes, raw JSONL trajectory/acceptance ledgers, public-only SFT dataset with token-mask tests, adapter/export artifacts when authorized, metrics by source/family/seed, compute ledger and a factual results report. Label CPU mechanics, native inference, training and deployment checks independently. Leave unresolved gates explicit.
