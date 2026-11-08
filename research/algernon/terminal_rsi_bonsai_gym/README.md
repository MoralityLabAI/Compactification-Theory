# TRSI-BONSAI-SFT-01 transfer gym

CPU fixture runner and Silico work order, prepared 2026-11-09. Native model inference, packed adapter training, Terminal-Bench integration and exported-adapter parity are **unrun**. Exact teacher/student identities remain unresolved in `model_manifest.json`; names from the discussion are search hints.

## Run from the repository root

```bash
python -m unittest discover -s research/algernon/terminal_rsi_bonsai_gym -p 'test_*.py' -v
python research/algernon/terminal_rsi_bonsai_gym/gym.py smoke --out /tmp/trsi-smoke-01
python research/algernon/terminal_rsi_bonsai_gym/gym.py export --traces /tmp/trsi-smoke-01/trajectories.jsonl --out /tmp/trsi-smoke-01/reexport.jsonl
python research/algernon/terminal_rsi_bonsai_gym/gym.py preflight --manifest research/algernon/terminal_rsi_bonsai_gym/model_manifest.json --artifacts-root /tmp/trsi-evidence
```

The final command intentionally exits 2 while model pins and evidence are missing. Preflight checks file hashes and declared outcomes supplied by a trusted reviewer. It cannot certify the truth of an arbitrary evidence document. It has no launch function. All output files use create-only writes.

## Implemented surface

`gym.py` implements six deterministic in-memory task families: arithmetic-config repair, structured-data transformation, transient-error recovery, delayed multi-step completion, text-file transformation and injection pressure. Allowed operations are typed `read`, `write`, `advance`, `test`, and `finish`. The host limits paths and steps. An independent final judge checks actual state after a finish request; visible tests are deliberately weaker. This is a CPU mechanics fixture, with no command execution or native terminal isolation claim.

Replay externally supplied JSONL actions with:

```bash
python research/algernon/terminal_rsi_bonsai_gym/gym.py replay --family transform --root 0 --seed 11 --actions actions.jsonl --out episode.jsonl
```

An action line has exactly the keys allowed for its operation, for example `{"op":"read","path":"input.json"}`. Writes additionally require `value`; test/finish/advance have only `op`. Duplicate keys, shell requests, protected paths and oversized requests are rejected and recorded. This CLI replays a pre-recorded stream. A feedback-driven model process and native container runner remain integration work in `SILICO_PROMPT.md`.

## Records and train/export boundary

The trajectory envelope pins the runner/verifier bytes, host policy, task identity, group split and actor label. Every event records the raw action hash, parsed action, public observation, state fingerprints, host-owned decision boundary, timing and hash-chain predecessor. Full raw text is retained within the byte envelope. Oversized text is represented by its hash. Model IDs and inference/critic/quantization fields remain null until measured by a native runner.

Failure trajectories remain in `trajectories.jsonl`. Successful training-split traces with zero rejected requests are independently replayed before export. Export reconstructs an allowlisted public transcript, supplies assistant-message loss boundaries, deduplicates identical root/seed/action trajectories, and excludes validation/test roots. Private final-judge outcomes and state hashes stay outside model messages. A hash chain detects changes relative to a trusted root; it supplies no signature or external authenticity guarantee.

Every exported smoke record is labeled `synthetic_fixture=true` and `eligible_for_model_training=false`. These fixtures test plumbing and should not be mixed into claimed teacher distillation data. Root-group splits cover retries and variants; native datasets additionally require frozen template/family-disjoint holdouts and duplicate detection before any sampling.

## Research joins

Use sibling `../critic_handoff_01/` for critic transport and distributional reward readouts. Event-clock references live in `MoralityLabAI/HybridTRMLDT/research/terminal_rsi_01/`; stagewise control is owned by `MoralityLabAI/Control-Harness/research/terminal_rsi_control_01/`. Replay selection and low-bit trajectory bounds live in `patrickdugan/BitAgent/research/terminal_rsi_rewrite_01/`. These packages supply mathematical references; live schema compatibility has not been established.

`CPU_RESULTS.json` records the local smoke result. A scripted baseline solving its fixtures measures mechanics. Learned capability claims require the native matched-budget comparison in the work order. Earlier Algernon results and its external-baseline pivot remain unchanged.
