# Algernon Phase A/B status, 2026-11-03

## Completed and independently inspected

- Main commit `319ce426ddb30b16a07ff4e5dae31590b95e2206` introduced the user's checkpoint inventory and script. It was merged into the draft PR branch by merge commit `dfd22bc38d37407d42d9cc9606a6bcd3216d6a87`.
- The JSON contains 77 checkpoint entries grouped into 9 experiment/recovery directories and 59 unique byte-level SHA-256 hashes; 18 entries are duplicates. 45 describe a tied 33,024-parameter module. The remaining 32 describe untied 16/32/64-block stacks.
- All original inventory entries have `performance_recorded=false`. Log-backed qualified graft candidates: **zero**.
- `inventory/audit_inventory.py` groups aliases by SHA, validates shape metadata, and requires a locally hash-verified experiment log plus optimizer/dataset/schedule/held-out evaluation evidence before a candidate becomes source-verified.
- The Qwenputer upstream source ref and the Weight-Space Programming logs are pinned in `source_lock.json`. These are source-revision observations, not evidence of a new run.

## Executed CPU microkernel (independent preliminary Phase B)

On Python 3.13.5 CPU, the local script `exact_kernel.py` was tested against an independent BFS oracle. Five unit tests tested exact-parity logic and another five tested inventory/provenance rules: **10/10 passing**. The graph fuzz test includes 1,000 generated examples. A further fixed-roster suite generated 250 cases, 50 per graph family, with zero bitset/BFS mismatches. The immutable receipt containing exact hashes is `cpu_reference_receipt.json`.

The result tests deterministic lattice refinement. **Qwenputer RV32 in-weight execution was not run.**

## Remaining gates

1. Run the committed inventory audit against the original local experiment directory, and provide/verify source-log-backed performance receipts. Never guess schedule from `Rxx` filename fragments.
2. Recover at least one prior independently evaluated recurrent checkpoint and its original train/test splits before selecting a graft.
3. Execute upstream RV32 reference tests at pinned commit `bec1a0c9654ac8151b6f36ed0154a19c53141384`:
   ```bash
   git clone https://github.com/divagr18/qwenputer.git
   cd qwenputer
   git checkout bec1a0c9654ac8151b6f36ed0154a19c53141384
   cd experimental/riscv
   uv sync --extra hf
   uv run pytest tests/test_rv_ref.py -q
   ```
   For the full circuit tests use `uv run pytest tests -q`. Record skip counts, wall time, installed lockfile and raw stdout. GPU execution is a separate step.
4. Reproduce its reference outputs on the same graph microtask. A Python reference pass cannot substitute for a compiled neural circuit trace.
5. Freeze final train/cal/test splits and compute budgets for the 0.8B residual-receiver experiment.

## Commands on a clone of Compactification-Theory

```bash
python -m unittest discover -s research/algernon/tests -v
python -m research.algernon.validate_register
python -m research.algernon.inventory.audit_inventory --inventory research/algernon/inventory/checkpoint_inventory.json --out runs/algernon/checkpoint_audit.json
python -m research.algernon.exact_kernel --output runs/algernon/cpu_reference.json
```

The current environment could inspect GitHub source through a connected account but could not resolve github.com for cloning and had CPU-only PyTorch. The upstream Qwenputer suite and the local LoopedTransformers checkpoint files thus remain unavailable for direct execution here.
