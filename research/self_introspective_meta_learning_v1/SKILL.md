---
name: self-introspective-task-graph-learning
description: Draft procedure for measured self-diagnosis, bounded candidate training, and fresh-evidence promotion across a versioned task graph. Use with an explicitly configured learner and experiment budget.
---

# Self-introspective task-graph learning

Read [README.md](README.md) before the first run. This is a proposed operator skill. The task runner and model adapters remain implementation work. The accompanying Python module covers arithmetic only.

## Goal

Choose small, testable training interventions from measurements of the learner. Improve a frozen task-family objective while preserving established competences and accounting for all experiment costs. Record failed ideas so the experiment policy can learn from them.

## Required inputs

Use a run manifest with an incumbent checkpoint hash, a versioned task graph, generator/scorer hashes, complete execution settings, the resource ceiling, a global promotion-attempt counter, total statistical error allowance and retained reference checkpoints. Bind the declared training and evaluation units. Access to model activations must be confirmed by a working collector.

A configured proposal endpoint may use the local 27B model. A task environment with exact state transitions can supply successor labels. Each such privileged source must be named in the manifest.

## Procedure

1. Freeze the epoch's task-family weights, endpoints, resource score and gain margins. Carry forward the retention registry and spent error allowance.
2. Collect development diagnostics. Keep every observation tied to its checkpoint, task contract and collector version.
3. Ask the proposer for one target node, a supported diagnosis, a competing explanation, one update operator and numerical effect/cost predictions.
4. Validate the proposal against the available operator catalog and budget. Save rejected proposals with reasons.
5. Clone the incumbent and train the candidate within the configured limits. Save the optimizer/random-state semantics needed for replay.
6. Use development data to screen candidates. Count all search work. Select one candidate before asking for promotion evidence.
7. Lock the candidate and comparator hashes, declared endpoints, sample counts and attempt number. Allocate alpha_t=delta/[t(t+1)] across every endpoint. All earlier failed promotions have already consumed their indices.
8. Obtain fresh conditionally independent paired blocks through the separate evaluation broker. Keep the candidate frozen throughout the batch.
9. Check the fixed-potential gain, requested task gain and every protected-family floor against its immutable reference. Use the registered uncertainty method. Missing evidence produces `insufficient_evidence`.
10. Promote only when every condition passes. Preserve the parent and full receipt. Retire the evaluated batch and mark every disclosed statistic as feedback.
11. Update the experiment archive. A later controller-policy update receives a new version and a fresh meta-evaluation contract.

## Diagnosis rules

A collision between encoded states with different legal futures can motivate a larger or better code. Successor errors with recoverable initial state motivate a dynamics experiment. Long-horizon failures invite numerical stability checks. Adapter placement and effective rank require matched-compute controls. Any claimed transfer edge needs held-out target results under a fixed interface.

Closure residuals, probe decoding and model-written explanations are diagnostic inputs. The scored capability endpoints remain fixed during the epoch. Test a diagnostic on random initialization before using it to infer a learned mechanism.

## Outputs

Write `proposal.json`, `candidate_manifest.json`, `development_receipt.json`, `promotion_request.json`, `promotion_receipt.json`, `exposure_events.jsonl` and an append-only incumbent history. The request includes the sampled population and independent block definition. Record rejected and inconclusive attempts as well as promotions.

## Failure recovery

A scorer change, pre-freeze look at promotion outcomes, overlapping batch or missing reset invalidates that promotion request. Preserve the affected artifacts and consume the attempt index. Acquire a fresh batch after repair. A hardware interruption retains a checkpoint and partial-cost record; resume only with matching execution semantics. Keep an established task's reference through graph-version changes so tolerable losses cannot accumulate silently.

## Scope

This skill authorizes no job on publication. Actual execution requires a configured runtime and an explicit budget. FAW/FRC oversight belongs to a later version. Current statistical guarantees assume the acquisition conditions stated in the protocol. A fixed finite fixture result supports its specified domain; broader capability claims require the corresponding fresh task families.
