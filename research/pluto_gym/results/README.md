# Executed smoke evidence, v1

Run `0732e92cd3aa8bc6e44e` used the default configuration and optimization seed 0.
`smoke_manifest_v1.json` freezes its source hashes, seed partitions and rule catalog.
The full receipt retains every per-seed evaluation result and adaptation step.
Base and final checkpoints accompany the sanitized designer traces.

The learning run performed 200 base updates and 480 adaptation updates per
adaptive arm. Every update used a batch of 32. The host replay held up to 512 rows
and allocated 8,704 payload bytes; the final buffer contained 80 examples.
The eight-unit target MLP had 33 float64 parameters and 264 weight bytes.

| Arm | Held-out original-priority reward | Held-out reversal-priority reward | Original-rule retention |
|---|---:|---:|---:|
| Frozen | 0.8 | 0.0 | 0.5 |
| Random curriculum | 0.8 | 0.509375 | 0.5 |
| Offline regret curriculum | 0.8 | 0.509375 | 0.5 |
| Exact interface ceiling | 0.8 | 0.9 | 0.5 |

The adaptive arms tied. This smoke result demonstrates executable learning and
measurement while leaving a substantial target error on the reversed rule.
The trained component sees four possible cue/rule pairs. Public scheduling solves
the changed deadline and reward priorities. Broader strategy generalization is
outside this result.

`tensorcraft_surrogate_v1.json` records eight seeded trials using the actual
pinned TensorCraft combat simulator. It includes the request seed, variant,
scenario, adapter hash and dependency-lock hash. Its surrogate evidence label is
preserved. Native StarCraft never launched during these checks.

`verification_v1.json` records both test runs: 59 discovered, 55 passed and four
skipped in each of ordinary and optimized Python. Five offline integration tests
used the actual 70-file pinned upstream snapshot. Three HTTP loopback cases
skipped because the sandbox rejected sockets; the Gymnasium checker skipped
because its optional dependency was absent. The real subprocess timeout test
executed. Provider calls and GPU training both remained at zero.

Reproduce the learning experiment from the repository root:

```bash
python -m research.pluto_gym run --output runs/pluto-gym/reproduce --seed 0
```

The byte hashes bind the recorded source and artifacts. Wall times and process
RSS vary across executions. Numerical portability across NumPy/BLAS versions
requires comparison under a declared tolerance; the tested runtime records
NumPy 2.3.5.

