# RRC-14: reversible edits with strict interpreter-query accounting

This CPU-only experiment extends [RRC-13](RRC13_REPORT.md) with a learned
recurrent graph-edit policy that can revise previous operations and restore
cached search states. The graph language is the supplied 64-program
fork/join algebra over `I,N,S,R`. It receives two public demonstrations
and tests at most 12 complete graphs, with every interpreter call charged.

**Result:** the trained GRU recovers 53/160 previously unseen `S/R`
compositions and 16/64 shifted-source cases. A matched feedforward policy
recovers 63/160 and 14/64, while an untrained residual-frontier policy
gets 78/160 and 16/64. The fully exact solver recovers all 224. Thus
reversible search helps over the frozen RRC-13 one-shot initializer,
while the main recurrent-superiority comparison fails.

[Report](RRC14_REPORT.md) · [Protocol](RRC14_PROTOCOL.md) ·
[Mathematics](RRC14_THEORY.md) · [Frozen result](results/rrc14_pilot.json) ·
[Exact tests](test_graph14.py) · [CPU workflow](../../.github/workflows/rrc-14.yml).

From the repository root (Python 3.10+):

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_graph14.py' -v
python research/rrc_compactification/bench14.py --check
```

The checked replay uses the Python standard library. Training and
independent checkpoint regeneration require PyTorch CPU and are optional.
The previous RRC-13 checkpoint is deliberately shared and pinned by hash.
