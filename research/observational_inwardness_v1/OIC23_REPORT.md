# OIC-23: Executed finite certificates for fault-robust sensor queries

**Status (11 November 2026): exact deterministic experiment completed for n=2,3,4. No trained model is part of this stage.**

OIC-22's 145-parameter learned scorer missed its registered ten-point robustness-improvement gate at n=6 and n=8. The next exact experiment tests whether the general lower bound \(Q\ge n(e+1)+r\) is attainable, with one labeled one-bit sensor selected per clock step and unknown hidden shift and starting phase.

## Exact exhaustive results

| n | Necessary OIC-T37 flip bound | True minimum with one substitution | One erasure | One substitution plus one erasure |
|---|---:|---:|---:|---:|
| 2 | 4 | 6 | 4 | 8 |
| 3 | 6 | 8 | 5 | 9 |
| 4 | 8 | 10 | 7 | 12 |

The computed one-flip minima follow \(2n+2\) on the three tested instances. A general proof remains open. For n=5, the first searches reached a cap of 300,000 explored states without a completed answer; no feasibility status is inferred.

## Adaptive-versus-fixed separation

At n=4 and Q=10 the adaptive strategy succeeds under every possible single substitution. The complete fixed-schedule enumeration (262,144 words up to translation symmetry) gives a maximum inter-shift Hamming distance of 2, below the distance-three threshold for correcting one substituted report. Fixed policies succeed at the corresponding one-flip thresholds for n=2 and n=3.

These are *worst-case correct identification* results over every shift, starting phase, and fault path. Earlier OIC-21 and OIC-22 numbers are Bayes accuracies at Q=n+2; the objectives and budgets differ.

## Evidence and commands

- `oic23_exact.py`: finite Boolean AND/OR policy solver for simultaneous bounded substitutions and marked erasures, with exact candidate-mask states and a full adaptive query tree.
- `oic23_reference.py`: deterministic certificate generator and **independent** scenario-level replay, exhaustive fixed-schedule search and SHA-256 archive checks.
- `results/oic23_reference.json`: nine exact thresholds, unsuccessful and successful explored-state counts, policy-tree sizes, and fixed-schedule enumerations.
- `results/oic23_certificates.json.gz.b64`: compressed deterministic archive containing all nine successful policy trees.
- `test_oic23_exact.py`: source and receipt tests, tampering tests, and exact adaptive/fixed separation checks.

From the repository root:

```sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic23_exact.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic23_exact.py' -v
python research/observational_inwardness_v1/oic23_reference.py --check
```

The policy trees are generated from exact integer bit operations. The reference replay checks every legal true-world fault path independently of the candidate-mask recurrence. Certified finite coverage does not provide a Lean proof of a universal formula or a guarantee for any untested n.
