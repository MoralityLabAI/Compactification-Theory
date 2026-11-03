# Agent workorder: Algernon-01

> **2026-11-03 supersession:** [PIVOT_EXTERNAL_BASELINES.md](PIVOT_EXTERNAL_BASELINES.md) is now the active execution order. Checkpoint inventory and legacy training-log recovery below are optional retrospective work, not exit conditions for Phase A/B. Begin with pinned Training-Free Looped Transformers on supported Qwen3, independently replicate a baseline, then investigate ReLIT/retrofitted recurrence. Preserve original tasks as an archival workorder.

**Mission:** turn the research scaffold into a minimal, independently tested graft experiment. Work in this folder, preserve upstream code ownership, and leave machine-readable receipts for every measurement.

## Phase A: inventory and freeze (CPU; first PR)

1. Resolve the current full commit SHA of divagr18/qwenputer and the published experiment log revision. Lock model revision, architecture config, tokenizer and licenses; preserve their source notices.
2. Search the accessible HybridTRMLDT, CoT-embedding and looped-transformer repos for real trained checkpoints. Record path, optimizer recipe, dataset, recurrence schedule, tensor shapes, measured performance and hash. An unverified anecdotal run is not a candidate.
3. Produce an interface matrix: Qwen3.5-0.8B, Qwen3.5-4B, former recurrent modules, delta-head topology, projection widths and memory footprints. Specifically check Qwenputer's two-value-head dependence.
4. Freeze train, calibration, and evaluation rosters. Assign problem-group IDs and ensure train/test group disjointness.
5. Add a lockfile, a provenance script, and executable unit tests. Keep all model weights out of Git.

**Exit A:** a reviewer can enumerate exact model revisions, validate no evaluation leakage, and decide which checkpoints are usable.

## Phase B: reference compiled processor (CPU first)

6. Run Qwenputer's upstream RV32 reference tests under its pinned environment. Record the exact command, pass/fail, instruction count and full revision. Do not attribute upstream's published results to our own run.
7. Implement a small fixed arithmetic/constraint kernel suitable for an LDT task and prove parity against an independently written Python evaluator on seeded test cases. Vary signs, boundaries, invalid inputs and long traces.
8. Measure code bytes, reference execution time and trace-token estimate as separate quantities.

**Exit B:** exact correctness on held-out cases in reference execution, with raw receipts.

## Phase C: 0.8B recurrent receiver (GPU optional, controlled)

9. Use a frozen Qwen3.5-0.8B checkpoint. Implement a reversible hook at candidate receiver layers and log the pre/post residual. Confirm K=0, zero gate exactly recovers the base.
10. Train only P/Q and a tiny recurrent F at first, using group-disjoint calibration and a fixed budget. Sweep K=0,1,2,4,8 and matched-parameter feedforward control. Record all seeds, tokens and GPU-seconds.
11. Compare site, effective rank, random direction and shuffled recurrence; calculate clustered intervals over problem groups. Report KL, semantic regressions, invalid projected states and accuracy.
12. Use the released weightspace p50c fixed edit only as a separate replicated arm. Apply sparse receiver edits to the graft only after it proves helpful; every edit needs a hash, support mask, matched random control and independent evaluation.

**Exit C:** a graft-specific causal gain at a justified resource cost; otherwise publish a negative result.

## Phase D: joint compiled + learned kernel

13. Design an explicit circuit-to-latent interface. Implement deterministic output receipts and ensure the learned component never rewrites the exact reference assertion.
14. Compare compiled only, recurrent only, combined, disabled component, shuffled state and host CPU-tool oracle on identical tasks.
15. Measure effective bits per modified weight, activation/gradient rank, long-horizon memory fidelity, quantization sensitivity and latency.
16. Make a separate GPU feasibility plan for Qwenputer-4B: reported text-only BF16 checkpoint is 18.7 GB, above 16 GB VRAM. Preserve exactness through any offload/precision change or report loss of bit-exactness.

**Exit D:** independent gains that require both components, with transparent compute cost and unchanged frozen-base skills.

## Reporting format

For each executed experiment add one immutable timestamped results folder containing:
- manifest.json: source SHAs, model revisions, task split hashes, evaluator version, execution device, precision, seed, limits;
- records.jsonl: per-item answer, reference, stop reason, token count, exact-state receipt, intervention arm;
- metrics.json: accuracy, paired delta and CI, number of invalid states, base-skill drift, VRAM, wall time, GPU-seconds;
- report.md: preregistered hypothesis, deviations and interpretation, with explicit separation of observations and ideas.

**Rules:** no automatic large GPU spend; respect existing experiment owners, preserve test/eval boundaries, mark untested paths, reject hash mismatches and do not merge on the strength of a synthetic proxy score.

## Requested handoff

Start by completing Phase A and Phase B. Open a PR with an audited checkpoint inventory, a bit-exact CPU reference test and a proposed choice of first LDT microtask. Schedule 0.8B training only once the retained model and test harness have passed those gates.
