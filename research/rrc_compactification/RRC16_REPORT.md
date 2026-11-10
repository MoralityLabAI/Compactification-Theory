# RRC-16: topology-conditioned graph edits on a withheld Hadamard join

**Executed 2026-11-10, CPU-only exploratory pilot.** The primary held-out join was excluded from all 360 training and 90 development cases; three previous graph wirings supplied teacher-supervised edit trajectories. A small recurrent head was fine-tuned from the frozen RRC-14 checkpoint, with a matched feedforward head and history, descriptor, and shuffled-target controls. Every accepted candidate was executed exactly against two public degree-12 demonstrations and forecast on a separately withheld degree-32 query output.

## Main outcome: new join transfer

| Method | Signed sources | Product sources | Total held-out |
|---|---:|---:|---:|
| Fine-tuned GRU, stateful | 38/80 | 14/40 | **52/120** |
| Same GRU, state reset | 42/80 | 18/40 | **60/120** |
| Same GRU, topology features zeroed | 37/80 | 14/40 | **51/120** |
| Fine-tuned feedforward | 43/80 | 18/40 | **61/120** |
| Shuffled-target fine-tuned GRU | 40/80 | 21/40 | **61/120** |
| Frozen RRC-14 recurrent | 34/80 | 12/40 | **46/120** |
| Untrained residual frontier | 31/80 | 14/40 | **45/120** |
| Fixed first-12 candidate order | 41/80 | 21/40 | **62/120** |
| Complete 64-candidate exact solver | 80/80 | 40/40 | **120/120** |

All 120 held-out Hadamard cases are **inexpressible by each old topology** on their observed degree-12 prefixes. The complete 64-program search finds a valid Hadamard graph for every case; the trained GRU finds 52 under the 12-query budget. The matched feedforward and a simple fixed 12-candidate search perform better. **The registered recurrent-superiority test fails.**

The recurrent model has a modest positive difference against its *frozen RRC-14* predecessor, 52 versus 46 successful cases. Holding the same new recurrent checkpoint fixed while resetting its history improves recovery from 52 to 60; the feature-censored ablation finds 51. This is evidence against recurrent-state benefit for this shift. Familiar-wiring test accuracy is 41/60 for the newly trained recurrent model, and 41/60 for its frozen predecessor. The shuffled-target fine-tune reaches 61/120, but its starting checkpoint already has supervised RRC-14 weights; this control is a **corrupted fine-tuning treatment**, not a randomly initialized learner.

## Cost ledger, combined 120 new-join cases

| Method | Mean exact graph queries | Mean integer-loop proxy | Mean FP dense MACs |
|---|---:|---:|---:|
| Fine-tuned GRU, stateful | 8.97 | 1,546.57 | 76,076 |
| Same GRU, state reset | 8.84 | 1,683.28 | 74,598 |
| Same GRU, topology features zeroed | 9.08 | 1,570.83 | 76,938 |
| Fine-tuned feedforward | 8.77 | 1,509.73 | 74,774 |
| Shuffled-target fine-tuned GRU | 8.79 | 1,502.58 | 74,166 |
| Frozen RRC-14 recurrent | 9.32 | 1,455.35 | 77,685 |
| Untrained residual frontier | 9.95 | 1,589.47 | 12,936 |
| Fixed first-12 candidate order | 9.37 | 1,256.88 | 0 |
| Complete 64-candidate exact solver | 20.12 | 3,320.42 | 0 |

All edit policies pay for their selected full-degree two-demo graph queries, with 12 queries as a hard ceiling. The complete solver's larger search uses approximately 20.12 graph queries per positive test case on average. Each graph query requires eight public primitive applications. Arithmetic-loop units, model MACs and Python execution times are not commensurable. No performance claim divides one by the other.

The recurrent head stores **7,497 FP32 parameters** (239,904 bare weight bits), while the feedforward stores **7,549** (241,568 bare weight bits). Both are deployed with the same **6,148-parameter frozen RRC-13 initializer**. Prior RRC-14 fine-tuning supplies an additional privileged training history. The new teacher used **16,578 exact candidate-feedback calls** on the 360-case training roster, each executing two demonstrations; it also generated development and shuffled-label traces. This supervision cost is excluded from inference but cannot be ignored in model-efficiency conclusions.

## Negative controls and finite uncertainty

All 12 deliberately inconsistent observed-prefix inputs are rejected by the exact 64-candidate solver. Every bounded learned controller also abstains because the exact verifier rejects its proposed graphs. The six paired future mutants share all visible inputs and queries with their bases. The exact solver accepts all 12 observed prefixes, predicts six futures correctly and makes six false future acceptances. Every method follows identical candidate traces within each collision pair.

These are certificates for a chosen 64-graph language through degree 12, and separate finite predictions through degree 32. There is no inference that a candidate program is an infinite-series identity.

## Training and audit

Training used 260 CPU Adam updates, batch 64 and learning rate 0.002. The fine-tuning roster includes 120 examples per old topology. Teacher-forced development edit accuracy is 33.42% for recurrent, 29.37% for feedforward and 21.77% for shuffled fine-tuning. These development metrics assess next-edit labels and should not be interpreted as complete-program success. The trainable models are initialized from frozen RRC-14 weights in the old 132-feature columns; newly exposed three topology columns are zero-initialized before training. A public Hadamard interpreter is made available to *all* test-time search methods, including the baselines.

The 15 new tests cover exact formal-series arithmetic, old-topology parity, source-family isolation, real interpreter-call metering, independent held-out certification, paired collision invariance and Torch-to-portable-inference parity. An independent CPU training run reproduced all three frozen JSON checkpoints **byte-for-byte**; the complete source-bound benchmark receipt also regenerated exactly. No GPU resources or model downloads were used.

## Reproduce

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_topology16.py' -v
python research/rrc_compactification/bench16.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_topology15.py' -v
# Optional PyTorch CPU retraining:
python research/rrc_compactification/train16.py --out-dir /tmp/rrc16-retrained
```

The next useful gate is a **new grammar with withheld primitive semantics**, since the present transfer inherits the entire test-time interpreter and public structural descriptor. The fixed-catalogue result suggests that a future learned method should be compared against optimized static candidate orderings across several independently chosen topology shifts, with pretraining and training-time interpreter calls accounted for.
