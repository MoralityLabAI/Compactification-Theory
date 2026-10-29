# Plan R bounded-power bench: prospective integration contract

**Interface status:** Draft only. This package neither imports nor patches BitAgent, Control-Harness, CryptoSwarm Arena, or any wallet. The canonical CryptoSwarm Arena GitHub destination is unresolved. Resolve exact owner, branch, and review ownership before adding an adapter.

## Minimal adapter envelope (`plan_r_event_v0`)

Each event in a proposed synthetic test should contain:

```json
{
  "schema": "plan_r_event_v0",
  "run_id": "synthetic-run-id",
  "step": 1,
  "actor_id": "isolated-actor",
  "model_artifact_sha256": "64-lowercase-hex-digest",
  "policy_sha256": "64-lowercase-hex-digest",
  "state_before_sha256": "64-lowercase-hex-digest",
  "action": "read_a",
  "authority_scope": "synthetic-read-only",
  "verdict": "allowed",
  "effect_committed": false,
  "state_after_sha256": "64-lowercase-hex-digest"
}
```

Before any external action, the adapter must evaluate the authorization verdict and emit an event with distinct **request**, **decision**, and **effect** stages. `denied`, `allowed`, and `unknown` must be distinguishable, with `unknown` failing closed. The hash fields bind exact serialized artifacts and declared state schemas; a digest or self-reported `allowed` by itself provides no authority or verified attestation. A separate privileged oracle retains undisclosed synthetic ground truth. Use deterministic seeds and local fake endpoints only.

## Project boundaries

- **CryptoSwarm Arena:** owner of scenario builders, adversarial actors, oracle-only state and synthetic swarm trajectories. Import reference cases or produce equivalent traces once its repository/owner is verified. Reuse existing evidence and control gates where feasible.
- **BitAgent:** gateway decision authority for permitted model-generated proposals. Any adapter consumes a pinned, read-only decision-log export. Wallet/signature release stays with the existing wallet/approval boundary; the benchmark never requests real signing, withdrawals, orders, network publication, or payment.
- **Control-Harness:** control policy, evaluation orchestration, and monitor quality/false-positive measurements. Reuse existing `INSPECT → ACT → VERIFY` phase semantics if present at the pinned target revision, while preserving decision-only gate ownership.
- **Compactification Theory:** exact finite reachability, history-equivalence oracle, model/system capability separation, and falsifying witnesses. This directory is its only initial code owner.

## Preflight before a learned-model experiment

Pin the target repositories and model artifact hashes. Verify branch ownership and active PR overlap. Declare each tool, authority scope, memory lifetime, peer communication channel, external resource cap, release delay, and simulated unsafe condition. Freeze matched inference budgets and held-out tasks, independent seeds, and pass/fail criteria. Separate changes in solver success from changes in permissions. Record task completion, unauthorized-effect rate, number of denied requests, latency, and evidence quality under blinded evaluator-controlled labels.

Run a mock-only adapter and an independently replayed fixture before considering a local small-model pilot. Keep network access disabled, use synthetic identities and assets, and require a specific approval for any device or service deployment. Real-model trials and cryptographic attestation are future work.
