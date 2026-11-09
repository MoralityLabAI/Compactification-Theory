# RRC-14: fixed-budget, reversible graph-search CPU protocol

**Executed November 10, 2026.** This single-seed exploratory study was
planned and implemented within the conversation, with its test roster
exposed after the first trained evaluation. It is neither an externally
preregistered benchmark nor a multi-seed replication. The RRC-11/12/13
graph language and earlier results were available during experiment design.

## Frozen inputs and visibility

RRC-14 uses two independent public formal-series input/output demonstrations
through degree 12, with a target graph from the RRC-11/12 three-slot DSL.
The student reads signed-log coefficient features from those demonstrations,
its currently tested graph, and exact residual feedback **only after**
choosing and executing that graph. Its returned program is executed on a new
query through degree 32 solely by an independent scorer after inference.

- Seed `2026111014` generates 960 training and 128 development examples
  from the 17 previously known graph classes. All four operators appear
  individually and in familiar combinations.
- Held-out semantic classes combine `S` and `R` in eight excluded graph
  classes: 160 signed-source examples and 64 arithmetically generated
  product-series examples. The independent same-class test contains 80
  familiar-graph examples.
- Twenty mutated observed examples were accepted only if exhaustive
  64-graph search failed through the observed degree. Eight base/mutant
  pairs duplicate all visible data and differ at a future coefficient.
- Train, development and held-out visible demonstration prefixes are
  deduplicated except for the eight deliberate paired future collisions.
  `dataset(include_test=False)` never constructs any test source.

Training and evaluation are family-disjoint by operator-composition class,
with an additional source-generator shift. Previously exposed RRC classes
and operators remain known context. One seed provides exploratory evidence.

## Learning interfaces and test-time budget

- An RRC-13 active checkpoint is used as the **common frozen initializer**,
  with feedback disabled. Its zero-query inference and 12,936 MACs are
  charged to every edit policy.
- Every edit policy can execute at most **12 unique complete graphs** on
  both public demo prefixes. Each query returns the exact degree-12
  residual and is itself the finite verifier. No final free graph
  execution is permitted. A vertex cache avoids duplicate interpreter
  calls when the priority frontier backtracks.
- One trained GRUCell edit policy has 7,353 FP32 parameters, and a matched
  two-layer feedforward edit policy has 7,393 (0.54% more). They receive
  exactly the same 132 current-state features. An equal-size shuffled
  edit-label GRU is the negative control.
- The two additional ablations reuse the *same* GRU checkpoint: reset
  its hidden state every edit, or censor the residual-feature bus. The
  exact verifier remains active so that rejected candidate scores are
  always checked independently.
- Training uses two trajectories per training case, one from the frozen
  initializer and one from a two-edit-corrupted familiar graph. At each
  nonterminal step the teacher selects the best of nine neighbors using
  **training-only** exact executions. Each trajectory has four steps.
  Training uses 420 Adam updates, batch 96, learning rate .004,
  deterministic single-thread CPU PyTorch, and matching minibatch draws.
  The shuffled control permutes edit targets on the same trajectories.
- Inference and the checked replay use standard-library Python; PyTorch
  is optional for independent checkpoint retraining and parity tests.

## Comparators and accounting

Budgeted controls from the identical frozen initializer and the same
12 unique graph-query limit: lexical frontier, deterministic random
frontier, and a feedback-residual frontier that prioritizes outgoing
edits from better-scoring **already evaluated** parent states. A fixed
12-canonical-program catalogue runs without the initializer, and the
frozen initializer alone receives one exact check. The historical
64-candidate degree-four preview search and complete 64-graph exact
search are reported as higher-resource upper controls, with their
actual preview costs, short-circuited demo executions, and full checks
reported separately.

Store accepted and correctly forecast graphs, forced abstentions, false
future acceptances, actual unique graph queries, total demo executions,
primitive calls, nominal integer-loop work, FP MACs and nonlocal frontier
restorations for every arm and split. The search must use no additional
candidate executions after reaching its budget. The independent grader
reexecutes a proposed program only after selection; this audit call is
excluded from the model's resource meter and cannot influence its choice.

## Decisive tests

1. The reversible Hamming graph has nine legal single-token edits per
   vertex, with a valid reverse edge. Independent formal-series arithmetic
   and expected graph noncommutativity remain correct.
2. A patched interpreter-call counter must equal each model's reported
   query count; every tested graph is unique, within budget. No hidden
   graph or query-future attributes are required by policy inference.
3. The training generator cannot access test graph lists. The checkpoints
   and saved receipt must lock all defining source files and the prior
   frozen initializer. Portable logits must match PyTorch at multiple
   recurrent steps.
4. The repeated visible collision pairs produce identical adaptive
   proposals, including the random control. The exact complete baseline
   must reject all 20 chosen inconsistent cases and make exactly eight
   false future predictions for the paired late mutants.
5. A strong recurrent out-of-class claim requires beating the matched
   feedforward and the strongest untrained 12-call frontier on executable
   held-out programs, with dense MACs and interpreter calls both counted.
   Failure of that comparison remains part of the report.

## Reproduction

Run from Compactification-Theory root with Python 3.10+:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_graph14.py' -v
python research/rrc_compactification/bench14.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_graph13.py' -v
```

Optional CPU PyTorch training reproduction:

```sh
python research/rrc_compactification/train14.py --out-dir /tmp/rrc14-retrain
cmp /tmp/rrc14-retrain/rrc14_recurrent.json research/rrc_compactification/results/rrc14_recurrent.json
cmp /tmp/rrc14-retrain/rrc14_feedforward.json research/rrc_compactification/results/rrc14_feedforward.json
cmp /tmp/rrc14-retrain/rrc14_shuffled.json research/rrc_compactification/results/rrc14_shuffled.json
```

A source/weight mismatch fails the receipt check. The independent CPU
retraining above was completed and matched every checkpoint byte.
