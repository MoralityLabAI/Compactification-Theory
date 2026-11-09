# RRC-13: executed exploratory protocol and audit scope

**Date:** 2026-11-10. CPU-only, one fixed seed and two matched student
checkpoints. The protocol was developed during implementation. Earlier
RRC-11/RRC-12 holdouts and the DSL were already exposed; the new seed
supplies fresh instances and a held-out graph-class split without claiming
an externally blind or multiply replicated study.

## Task and information access

Every sample contains two formal-series input/output demonstrations,
observed through degree 12, from an unknown graph
`post(left(F)*right(F))` over `I,N,S,R`. A model chooses `left`, obtains
one exact residual vector for that partial graph against the observed
outputs, chooses `right`, obtains a second residual, then chooses `post`.
The interpreter only executes the **selected** partial graph at each step.
Its implementation and the fixed feedback vocabulary are supplied.
All accepted program predictions are independently verified on both
full degree-12 demonstrations, with degree 13–32 held back for grading.

The active controller never sees hidden source graph labels, output
coefficients beyond the observation horizon, a catalogue of all candidate
programs, or the exact search result during inference. Training uses
teacher-provided graph tokens and fixed scheduled sampling of off-path
partial feedback. The passive network has identical parameters and
optimizer steps and receives zero partial-feedback vectors. A third
ablation disables feedback at evaluation in the **same trained active
checkpoint**, retaining its weights unchanged.

## New deterministic roster

Seed `2026111013` produces 1,200 training cases and 160 development
cases with 17 empirical semantic graph classes. All four operators appear
in training. Eight different graph classes combine `S` and `R` in
compositions excluded from training; the test contains 120 new instances
of those classes with signed raw series and 48 new instances with
product-partition sources. Additional held-out cases comprise 64 new
training-class graphs, 16 mutated observed outputs selected only after
64-candidate exact-search rejection, and eight late future-collision
pairs (16 records). Training/demo-prefix signatures are deduplicated
across groups except for the explicit paired collisions.

Training invokes `dataset(include_test=False)`. Held-out graph programs,
source shifts, out-of-grammar mutations and future targets never enter
parameter updates or model selection. The fixed evaluation roster is
now public, so any RRC-14 optimization needs new seeds and additional
family shifts.

## Learning and resource controls

- **Active GRUCell:** 104 observed raw features go through a 24-unit
  tanh encoder, followed by three shared-weight GRUCell steps, 4-way
  output head, and a 16-number feedback bus. Exactly **6,148 FP32
  parameters**, 24 FP32 decoder-state values.
- **Passive GRUCell:** same complete architecture, exactly **6,148**
  parameters, with its feedback bus fixed to zero.
- Both models use 480 Adam updates, minibatches of 128, learning rate
  0.003, one CPU thread, identical minibatch and scheduled-sampling masks,
  and separately fixed initialization seeds. The first third of training
  is teacher forced; subsequent steps replace 35% of previous decisions
  with the model's own argmax to expose off-path feedback. Training labels
  are teacher canonical graph tokens, including exact syntax equivalents.
- **Frozen RRC-12 comparator:** 4,044-parameter previous direct GRU on
  the new roster, without retraining. Its distinct training roster makes
  it a historical comparator rather than a matched control.
- **Untrained stagewise greedy:** tries four operators at each of three
  decision positions, using public interpreter residuals for every
  candidate, with 12 partial queries per sample and one final certificate.
- **Untrained full-catalogue preview:** executes 64 degree-4 graph
  previews, then tests up to four candidates exactly through degree 12.
- **Complete bounded exact synthesis:** checks the 64 legal syntax
  programs against the same demonstrations, with exact early termination.

Every method records accepted observed programs, correct degree-32 future
predictions, false acceptances and abstentions. Record counts of partial
queries, full-verification attempts, primitive calls, nominal integer-loop
work and dense FP MACs separately. Teacher-forced development token
accuracy is a diagnostic rather than an execution certificate.

## Success criteria and claims gate

The primary out-of-class target is a **valid executable program on unseen
`S/R` compositions** after a single completed active construction.
A positive active-learning claim requires a verified result exceeding the
matched passive controller, the feedback-disabled active ablation, and
reported tool-assisted baselines. A familiar-family gain alone qualifies
as a narrower within-family outcome. Each negative and collision example
must preserve its failure or exact information limit.

No finite test proves a universal formal-series identity or model-based
invention of graph topology. The grammar, interpreter, feedback algebra,
supervised targets and partial-program oracle are supplied explicitly.

## Reproduction

From the repository root, standard-library Python 3.10+ is sufficient for
CI checks and inference:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_graph13.py' -v
python research/rrc_compactification/bench13.py --check
```

Optional deterministic CPU training uses PyTorch; the two checkpoint
JSON files must match the frozen files byte-for-byte:

```bash
python research/rrc_compactification/train13.py --out-dir /tmp/rrc13-retrained
cmp /tmp/rrc13-retrained/rrc13_active.json research/rrc_compactification/results/rrc13_active.json
cmp /tmp/rrc13-retrained/rrc13_passive.json research/rrc_compactification/results/rrc13_passive.json
```

The complete receipt records source, model, and roster SHA-256 digests.
Changing an executable module or a checkpoint requires an explicit new
receipt and another review. On CI without PyTorch, two training/parity
unit tests skip cleanly while exact arithmetic, source locks, inference,
and certificate checks still run.
