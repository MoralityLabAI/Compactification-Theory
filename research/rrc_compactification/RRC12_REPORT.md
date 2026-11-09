# RRC-12: direct recurrent graph proposer, one-seed CPU pilot

**Executed November 10, 2026.** New small CPU benchmark extending RRC-11. Frozen one-scan GRU and matched feedforward checkpoints, a shuffled-label negative control, optional frozen-weight extra scans, exact bounded search and an untrained interpreter-assisted preview sorter. Two raw input/output demonstrations per case, observed through degree 12. A single direct proposal is checked *after* the model has finished inference. Exact future grading extends to degree 32.

## Results on new graph combinations

| Predictor | Familiar 68 | Held-out S+R graphs 160 | Shifted product-source 64 | Held-out program recovery |
|---|---:|---:|---:|---|
| GRU, one trained scan, 4,044 parameters | **17** | **0** | **0** | Failed |
| GRU, two frozen scans | 17 | 0 | 0 | Failed |
| GRU, four frozen scans | 17 | 0 | 0 | Failed |
| Feedforward, 3,990 parameters | 12 | 0 | 0 | Failed |
| Shuffled-label GRU | 1 | 0 | 0 | Failed |
| Untrained 64-preview + four exact checks | 67 | 159 | 62 | Tool-assisted |
| Exhaustive 64-graph exact synthesis | 68 | **160** | **64** | Class-complete |

The recurrent model transfers a few familiar classes better than the feedforward control, but **recurrence does not produce a verified novel composition**. Additional loops make no difference in accepted-graph totals. The direct models reject all 20 constructed out-of-grammar examples *through the external verifier*. There are eight identical-prefix collision pairs (16 entries). Exact and preview methods accept all 16 visible prefixes and have exactly eight incorrect future predictions. Every direct neural proposal in the collision group fails prefix verification, so the neural models receive zero future-success credits there.

The finite 64-graph exact search finds every program in its declared class. The preview sorter, which evaluates every candidate through degree four, misses one of 160 familiar-source unseen combinations and two of 64 shifted-source unseen combinations within the four-check budget. Its near-perfect rate is due to explicit public interpreter calls, not a trained model.

## Search and compute ledger

| Arm | Mean exact integer coefficient-work / unseen-composition case | Dense FP MACs / case | Full candidate checks / case |
|---|---:|---:|---:|
| Direct GRU, 1 scan | 1,014 | 90,240 | 1.00 |
| Direct GRU, 4 scans | 1,014 | 359,808 | 1.00 |
| Direct feedforward | 1,014 | 3,944 | 1.00 |
| Untrained 64-candidate preview | 10,620.34 | 0 | 1.02 |
| Exhaustive exact synthesis | 18,717.81 | 0 | 35.92 |

The direct model saves integer verifier calls because most proposals are rejected. Its extra floating-point inference work and low success rate prevent interpreting the smaller integer-work count as a better solver. The exact-search work is a coefficient-loop proxy, and the neural values are estimated dense multiply-accumulates; a total cross-hardware price has **not** been measured. CPU time is further affected by the portable pure-Python implementation.

## Training and transfer context

Training uses 2,040 random-series demonstrations from 17 finite-probe graph classes. Development has 320 independent examples from those classes. Eight new classes with combined degree substitution `S` and reciprocal `R` are excluded; all four operators appear during training. The GRU has 4,044 FP32 parameters versus 3,990 for the feedforward model (1.35% fewer). Both train with the same deterministic 420 minibatches at batch 128 and Adam learning rate 0.004. The model checkpoints have identical source/roster hashes and contain complete portable inference weights. Development graph-token accuracy is 61.15% for the GRU and 56.15% for the feedforward control; these are training-family diagnostics, not claims of execution correctness.

The out-of-grammar cases are **selected after exact search rejection** and provide a protocol negative check, not unbiased anomaly detection. The source-family shift uses periodic-part generating functions rather than the signed small coefficients used for model training. The 25-graph quotient is empirical over fixed probes and has not been formally proven as a complete semantic equivalence quotient. The eight new test classes were excluded from this training roster, but RRC-11 publicly exposed related graph structures and the operator library. Nothing in this pilot measures scaling to a pretrained general-purpose model.

## Verification

The 13-test suite covers exact graph algebra, semantic split audits, data leakage, future collision bounds, source/checkpoint hashes, train-only generation, no interpreter access during neural inference, PyTorch-to-portable parity, and regression totals. The saved full receipt regenerates byte-for-byte. Independent CPU retraining produced byte-identical JSON checkpoints for all three models.

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_graph12.py' -v
python research/rrc_compactification/bench12.py --check
# Optional:
python research/rrc_compactification/train12.py --out-dir /tmp/rrc12_retrain
```

### Next falsifiable question

RRC-13 should replace the flat three-token classifier with a **program-construction controller** that can issue one partial graph node and query its consequences, while charging those interactions. Include an exact dynamic-programming or prefix-pruned synthesis comparator and an untrained active-query controller with the same observation and interpreter budgets. Also require fresh source families and graph classes to prevent repeated evaluation on exposed holdouts.
