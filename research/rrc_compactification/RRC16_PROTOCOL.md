# RRC-16: multi-topology fitting and a withheld Hadamard join

**Executed:** 2026-11-10. One exploratory CPU pilot with frozen deterministic seeds and source-bound results. This document records the implemented protocol; it is not an externally preregistered or blind multi-seed study. RRC-11 through RRC-15 and their previously exposed results were available while designing RRC-16.

## Training/holdout boundary

Three wiring families appear in training: fork, cascade, and additive. Hadamard coefficientwise multiplication appears only in the held-out generator and evaluation interpreter. The model's supervised targets are the index of a best **single-token neighbor edit**, selected using nine exact training-only counterfactual candidate executions per nonterminal state. It does not see the held-out graphs during training.

- **Training:** 120 unique examples per wiring, 360 cases total, all with signed integer source series, and two four-step trajectories per case.
- **Development:** 30 cases per wiring, 90 total, with the same rules and independent samples. These results are diagnostic.
- **Held-out seen wirings:** 60 cases, 20 per trained topology, with fresh signed series.
- **Held-out Hadamard join:** 80 signed-series cases and 40 product-partition-source cases, both selected to have no matching graph in *any of the three old topologies* on the observed coefficients. Selection uses only two public demonstrations, independent of policy predictions or withheld future outputs.
- **Hadamard negative controls:** 12 selected degree-seven output mutations, rejected by all 64 Hadamard graphs at degree 12.
- **Future-collision controls:** six base/future-mutant pairs (12 records). Both members share the same visible demos and query, with a degree-15 future difference.

Every result is based on two degree-12 demonstration input/output series. The held-out degree-13–32 output of an independent query is used **only** for final grading. Exact visible signatures are deduplicated between train/dev/test except for the six deliberately paired collisions.

## Models, constraints and matched search

The RRC-16 recurrent head has 7,497 parameters, the matched feedforward head has 7,549, and a shuffled edit-label recurrent head has 7,497. All models initialize from their corresponding frozen RRC-14 edit heads, copying the 132 prior feature columns and setting the three newly added public topology-feature columns to zero. The same frozen RRC-13 initializer supplies each policy's starting graph. Shared training: 260 Adam updates, minibatch 64, learning rate 0.002, deterministic single-thread CPU PyTorch, and identical minibatch indices. No model selection or checkpoint changes followed the first full held-out evaluation.

Policy state receives the raw demo coefficients, a candidate graph one-hot code, exact residual feedback **only after the candidate is executed**, and the public topology descriptor. Test-time inference contains no graph label, query future, unseen candidate prediction, full catalogue score, or training optimizer. The same recurrent checkpoint is evaluated with its state reset at every failed candidate, and separately with its three topology descriptor values censored. These are single-checkpoint ablations.

All adaptive edit arms are limited to **12 unique complete candidate queries**, with two demonstration executions and eight primitive applications each; a hash-set cache prevents duplicate executions. The untrained residual-frontier comparator receives the same initial graph, feedback and call cap. The fixed first-12 canonical syntax control has no trained initializer. Exhaustive 64-program search is a higher-budget, exact upper control. All arms use the **public Hadamard implementation** when testing the unseen wiring; its implementation and the interpreter are costed separately from learned weight storage.

Report acceptance and degree-32 future prediction, false future acceptance, abstention, graph queries, demo executions, primitive applications, nominal integer loop work and floating-point dense MACs. Never mix the latter two in one cost number. Include the training teacher's counterfactual execution cost and prior-model history in the accounting.

## Acceptance and falsification

- Required exactness: independently check every candidate output against both observed demonstrations; a proposed program only receives credit if both matches are exact. Complete 64-program search must accept all 120 normal Hadamard cases and reject all 12 chosen out-of-grammar Hadamard cases.
- Required held-out semantics: no training/development call to the Hadamard implementation; all 120 positive new-join cases must lack a matching old-wiring program on the degree-12 observations.
- Causal controls: within-checkpoint recurrent-state and topology-feature ablations, same-budget feedforward, shuffled fine-tuning, old frozen GRU, untrained frontier and fixed catalogue. Pair the exact same test cases; report zero/negative effects.
- Budget controls: the number of actual `execute` calls equals twice the number of graph queries, every evaluated graph is unique, and no unchecked program may be emitted as an accepted result.
- Future uncertainty: all six paired future collisions must yield identical candidate-search traces for each arm and six future errors for complete search.
- Reproducibility: the benchmark receipt hashes all defining source, prior and current checkpoint bytes, datasets and all trajectory digests. Replaying must compare the complete JSON result byte-for-byte. Independent retraining is optional in CI and must be separately compared to the checkpoint files.

**Decision gate:** Learned recurrent search superiority on the new join requires exceeding the matched feedforward, the same checkpoint with state reset, and the strongest untrained 12-query method in accepted and future-correct graphs while disclosing all compute. Failure is an expected valid result.

## Reproduce

From the Compactification-Theory repository root (Python 3.10+ standard library):

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_topology16.py' -v
python research/rrc_compactification/bench16.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_topology15.py' -v
```

Optional CPU PyTorch checkpoint retraining:

```sh
python research/rrc_compactification/train16.py --out-dir /tmp/rrc16-retrained
cmp /tmp/rrc16-retrained/rrc16_recurrent.json research/rrc_compactification/results/rrc16_recurrent.json
cmp /tmp/rrc16-retrained/rrc16_feedforward.json research/rrc_compactification/results/rrc16_feedforward.json
cmp /tmp/rrc16-retrained/rrc16_shuffled.json research/rrc_compactification/results/rrc16_shuffled.json
```

No model weights are downloaded, no GPU training is required, and CI never retrains or accesses held-out labels through the model.
