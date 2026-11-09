# RRC-13: active partial-graph construction, measured CPU pilot

**Completed 2026-11-10.** An exploratory 6,148-parameter recurrent
controller emits graph nodes one at a time and uses exact residuals from
its own partial executions. The matched passive controller has the same
parameter count, training roster, minibatches and optimizer budget. A
frozen active-network ablation disables feedback at inference. The
fully independent verifier checks completed programs. Graph topology,
operator implementations and residual representation are supplied.

## Held-out executable-program results

| Evaluation | Active 2 queries | Active, feedback off | Matched passive | Stagewise greedy | 64-preview / 4 checks | Exact search |
|---|---:|---:|---:|---:|---:|---:|
| Familiar graph classes (64) | **25** | 8 | 12 | 22 | 64 | 64 |
| Unseen S/R compositions (120) | **0** | 0 | 0 | 25 | 120 | 120 |
| Unseen S/R + product inputs (48) | **0** | 0 | 0 | 8 | 45 | 48 |
| Constructed out-of-grammar (16) | 0 | 0 | 0 | 0 | 0 | 0 |
| Paired future-collision prefixes (16) | 0 | 0 | 0 | 2 | 16 | 16 |

Every accepted program on the familiar, unseen and shifted sets
predicts the future correctly through degree 32. All 16 out-of-grammar
samples are rejected by the full-prefix verifier, by construction of the
negative roster. The full search and preview comparator accept the
16 collision prefixes but make eight false future predictions because
the pairs share all observable inputs. Stagewise greedy accepts only two
of those prefixes and predicts one correctly. Neural proposals in the
collision group fail observed-prefix checks and abstain.

The previous frozen 4,044-parameter RRC-12 direct GRU, evaluated on
this fresh roster without retraining, recovers 14/64 familiar programs
and 0/168 held-out composition programs. Its distinct optimization
history rules out treating it as the matched passive model.

## Where feedback helps and where it fails

Familiar-case exact program recovery is **25/64** with active residual
feedback versus **12/64** for the equal-sized passive model. Replaying
the identical active checkpoint with feedback disabled falls to **8/64**,
which is a stronger within-model causal ablation for the provided
feedback interface. Withheld compositions remain **0/168** for both
trained controllers, including the 48 new product-source cases. The
registered out-of-family transfer gate therefore **fails**.

The untrained greedy program constructor reaches **33/168** on the
same unseen compositions after evaluating four candidate operations
per graph slot. An untrained 64-candidate preview procedure reaches
**165/168**, while complete exact search reaches **168/168**.
These tool-assisted methods execute additional candidate graphs using
public algebra; their success is evidence of a useful interpreter, not
of neural zero-shot compositional discovery.

Training uses 1,200 examples from 17 classes and a 160-case
same-class development set. Teacher-forced development token
accuracy is 74.17% for the active model and 63.54% for the passive;
those metrics are optimistic relative to free-running program recovery.
The two trained models have 6,148 FP32 parameters and 24 FP32
recurrent-state coordinates each. Independent retraining reproduced
both JSON checkpoints byte-for-byte.

## Cost accounting on the 120 unseen-composition cases

| Method | Mean public partial queries | Mean full candidate checks | Integer-loop proxy / case | FP dense MACs / case |
|---|---:|---:|---:|---:|
| Active learned | 2 | 1 | 3,042 | 12,936 |
| Matched passive | 0 | 1 | 1,014 | 12,936 |
| Untrained greedy construction | 12 | 1 | 13,182 | 0 |
| Untrained 64-preview / 4 checks | 64 | 1.025 | 10,630.9 | 0 |
| Complete exact search | 0 | 35.358 | 18,437.9 | 0 |

The counting convention charges a full-degree (12) graph execution
on both demonstrations for each partial query. A full certificate costs
two graph executions. Preview search charges every degree-4 candidate,
including those never selected for full verification. Exact search
short-circuits failed candidates after their first demonstration.
These are nominal arithmetic-operation proxies and dense MAC counts,
not wall-clock measurements or interchangeable cost units. Interpreter
code storage, big-integer arithmetic, source generation, neural
training and independent future grading have separate costs.

**Training-time privilege:** the active model precomputed 20
counterfactual partial-query feedback vectors per training case and
per development case, each with two demonstration executions. The
passive model consumed zero feedback features. The active student
uses only two selected partial queries per case at inference. Training
benefits from explicit canonical graph labels, a known graph skeleton,
and a supplied algebraic simulator.

## Audit and interpretation

The 16-test suite checks independent exact formal-series identities,
semantic class boundaries, held-out generator isolation, the literal
partial-interpreter call budget, absence of hidden/future access,
PyTorch-versus-portable inference parity, exact negatives, collision
invariance, and the frozen receipt. The replay recomputes source and
checkpoint SHA-256 hashes and every stored score. Local independent
retraining produced identical active/passive JSON checkpoints.

**Decision:** intermediate algebraic feedback helps this small
recurrent controller reconstruct known graph families. It has not
recovered any graph from the excluded compositional classes. Future
work should train on a richer distribution of paths, preserve
class-disjoint evaluation and compare a learned search controller
against a candidate-pruned exact dynamic-programming oracle at
matched interpreter budgets. RRC-11/12/13 heldouts are public and
must be treated as development context in subsequent iterations.
