# Gate-Compiled Protocol Transformer: ARC-AGI-2 sandbox pilot

Experiment date: 2026-11-04. This is the GCPT subfolder for the CPU/RAM ARC-AGI-2 pilot, migrated from the `loop-capacity` staging experiment.

## Observations

| Condition | Exact test grids (public selected subset) | Tasks solved | Status |
| --- | ---: | ---: | --- |
| Frozen generic program search | 0 / 8 | 0 / 6 | Code frozen before importing task contents |
| Exploratory glyph grammar | 1 / 8 | 1 / 6 | Grammar written after public-task exposure |

The correct task was `136b0064`, correct on attempt 1. Six tasks were selected by file size and lexical order, **not as a representative or blinded ARC-AGI-2 evaluation**. The pilot provides **no measured full-set or private leaderboard score**.

## Architecture

The GCPT intermediate level separates finite neural semantic cells from exact loop control. The core local gate cell implements all 16 binary Boolean functions with visible role keys and a physically tied two-pass Transformer. Exact MSB-first register scheduling composes learned equality and less-than relations into an unsigned comparator.

For the exploratory ARC task, a task-specific panel/glyph grammar is fitted from provided demonstration pairs, then a 9,784-parameter two-visit Transformer learns the glyph/action rows. Exact code owns program counter, cursor, bounds and raster commit; the 9,378-parameter gate model supplies finite predicates. This extension was written after seeing public task material.

The pilot validated 64/64 Boolean rows, 4,096/4,096 six-bit comparisons, and one public ARC output. A separate 256-sequence synthetic audit tested the glyph-executor semantics within the discovered four-symbol grammar. One-visit inference was already accurate on the glyph rows. No relative architectural advantage from loop depth is established.

## Source and audit status

The complete executed experiment is preserved in the accompanying `gcpt_arc2_sandbox.zip` download from the 2026-11-04 ChatGPT run, including checkpoints, six pinned public task files, raw predictions, execution traces, tests and scoring receipts. **The source archive has not been uploaded to this GitHub branch.** This folder presently contains the independently runnable finite-gate module and experiment documentation; the remaining Python modules and checkpoints still need to be imported from that archive. Do not treat this branch as the complete reproduction.

The full original bundle contained `gates.py`, `glyph_vm.py`, `solver.py`, `run.py`, `run_glyph.py`, `score.py`, `import_subset.py`, and `test_gcpt_arc2.py`. The original sandbox tests passed 16/16 on the packaged bundle.

## Next decisive experiment

Freeze a task-agnostic compiler and grammar before opening another public ARC task; hold out ARC-AGI-2 evaluation tasks, score one or two attempts per test grid, and compare equally sized one-visit, two-visit and low/high-quotient cells with exact traces, program-specific semantic rows and counterfactual wire interventions.

Data provenance: https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49
