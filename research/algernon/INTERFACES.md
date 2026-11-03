# Phase A interface matrix: LoopedTransformers → Algernon

**Status: metadata-derived design constraints.** The checkpoint inventory contains tensor shapes and byte hashes. No local training logs, model outputs or Qwen residual activations were available to certify a graft.

| Family found in inventory | Total files | Tensor layout | Parameters per checkpoint | Possible interface |
| --- | ---: | --- | ---: | --- |
| Weight-tied | 45 | one `blocks.0` pair of 128×128 linear maps plus two 128-biases | 33,024 | 128-dim state bridge, recurrence schedule to recover |
| Untied, 16 blocks | 12 | 16 sets of in/out projections | 528,384 | step-specific 128-dim stack; do not treat as tied recurrence |
| Untied, 32 blocks | 12 | 32 sets | 1,056,768 | step-specific 128-dim stack |
| Untied, 64 blocks | 8 | 64 sets | 2,113,536 | step-specific 128-dim stack |

All 77 inventory entries collapse to **59 distinct SHA-256 hashes**. Recovery folders contain 18 redundant aliases. These counts are based on the committed inventory JSON, and do not establish accuracy or compatibility.

The checkpoint schema lists `cell_id`, `exposure`, `optimizer`, `state_dict`, and `step` as top-level keys. Current metadata extraction only retained the names and tensor shapes. The actual optimizer *configuration*, training dataset, recurrence timing, target/task identity, latent semantics, and verified held-out performance must come from source receipts. Names such as `R64` and `E4096` are filename clues rather than certified schedules or training steps.

## Proposed bridges and constraints

| Receiving substrate | Known feature width / role | Proposed bridge | Required gate |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 1,024-dimensional residual in 15/19 MLP down-projections | `P:1024→128`, `Q:128→1024` | trained reversible receiver, zero-gate parity, ablation at equal FLOPs |
| LoopedTransformers tied checkpoint | 128-dim in/out projections, 33,024 saved model parameters | repeat one 128-dim block according to *logged* schedule | reproduce its standalone held-out result first |
| LoopedTransformers untied checkpoint | 16/32/64 sequential 128-dim blocks | choose explicit pass schedule; count all weights and compute | step semantics must be recorded before calling it recurrent |
| Qwenputer RV32 on Qwen3.5-4B | 11 compiled layers in front of pretrained model; DeltaNet CPU memory | compiled exact circuit / token bridge | upstream RISC-V reference suite and trace hash parity |
| Joint Algernon | separate token-clock RAM and algorithm-clock recurrence | typed exact-output record to latent state; gated residual write | independent causal ablations and retained skills |

**Architecture boundary:** Qwenputer's published RV32 core assumes two value heads per key head, while Qwen3.5-0.8B's layout gives one. Compiled 4B and sparse-edit 0.8B experiments remain separate until a new head map is constructed and tested.

Qwenputer reports an approximately **18.7 GB** merged BF16 text checkpoint and roughly 31 GB host RAM for its merge; a 16 GB RTX 5080 must use a specifically tested offload/quantization route.

## Candidate selection rule

A checkpoint becomes a *source-verified candidate* only when a receipt references its exact 64-digit checkpoint SHA-256 and a local experiment log whose SHA-256 is independently recalculated, with:

- optimizer recipe and code commit;
- dataset ID and SHA-256;
- recurrence schedule;
- a held-out evaluation, sample count and split hash.

Run `python -m research.algernon.inventory.audit_inventory --inventory research/algernon/inventory/checkpoint_inventory.json --out runs/algernon/checkpoint_audit.json` from the repository root. Add `--receipts /path/to/receipts.json --logs-root /path/to/LoopedTransformers/experiments` after the old logs have been inspected. Qualifying receipt status measures provenance completeness, **not model performance replicated here**.

## First microtask

`research/algernon/exact_kernel.py` solves tiny directed-graph reachability through repeated bitset closure and verifies it against an independently implemented BFS. It supplies step-indexed states and group-disjoint graph families for learning/recurrent ablations. It is **ordinary CPU code**, serving as a future reference for a compiled circuit, not a Qwenputer layer.
