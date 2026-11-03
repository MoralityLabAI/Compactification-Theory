# Algernon-02 receipts (CPU-only)

- `cpu_reference_4x8.json`: 192 independently checked task traces, 48 each from random-function, permutation, long-cycle, and sink families. Python R1CS checker with BN254 Fr, full trace and public inputs bound, no circuit/compiler execution. A deliberately corrupted intermediate state failed verification.
- `pointer_4x8.circom`: generated source that describes the same transition relation. This was **not compiled or witness-tested by Circom** in this environment; its syntactic/semantic compatibility is an open follow-up.
- `pilot_6_300.json`: one-seed CPU PyTorch toy (6 nodes, 222 parameters, 300*128 one-step training examples) with 18 evaluation cells, 40 tasks per cell. The network receives explicit input products and thus has strong architectural prior; held-out families test recurrence and readout, not discovering arbitrary programs.

All receipts have explicit execution class labels and fixed seeds. Scripts write subsequent runs outside Git to `runs/algernon/circuit_mirror/` by default. Do not train on or adjust thresholds using these held-out samples and then report an independent holdout score.
