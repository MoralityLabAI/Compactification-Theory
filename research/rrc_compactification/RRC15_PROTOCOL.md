# RRC-15: frozen cross-topology transfer protocol

**Executed:** November 10, 2026, one fixed exploratory CPU seed. This protocol and its source-bound receipt were prepared during the experiment, after the RRC-14 models and prior held-outs were available. It is an independent *instance* roster with newly specified graph-wiring classes, not an externally preregistered or truly blind result.

## Hypothesis and eligibility

The primary hypothesis is that the frozen RRC-14 recurrent edit policy can reuse its learned search dynamics after the public interpreter switches from `post(left(F)*right(F))` to (a) `post(left(F)*right(left(F)))` and (b) `post(left(F)+right(F)-1)`. The three-token syntax, 64 candidate labels, nine reversible neighbors, 104 raw observations, 16 residual features and 12-query search frontier remain fixed. No model retraining, checkpoint selection or new layers are permitted.

New topology alone can leave equivalent old expressions, especially for cascade wiring. The roster therefore distinguishes old-fork-unexpressible cases, cascade aliases, and the unchanged fork anchor. An old-fork alias is identified using **only the two public observed degree-12 demonstrations**, exhaustively checking 64 old programs. This condition is a declared source selection rule; it is not feedback supplied to any policy during inference.

## Dataset and split

Seed `2026111015` generates 244 independent records, including twelve deliberately duplicated *visible* future-collision pairs. Inputs are signed integer unit series (coefficients from -2 through 2) or a held-out arithmetic-part product-series source. Product sources must expose a nonzero first or second coefficient. The counts are:

| Split | Cases | Graph semantics | Source and eligibility |
|---|---:|---|---|
| Fork anchor | 32 | Original | Fresh signed series |
| Cascade novel | 48 | Cascaded right branch | Signed; **no** old-fork observed-prefix equivalent |
| Cascade alias | 24 | Cascaded right branch | Signed; an old-fork observed-prefix equivalent exists |
| Cascade product | 24 | Cascaded right branch | Product source; old-fork-inexpressible |
| Additive novel | 48 | Additive normalized join | Signed; old-fork-inexpressible |
| Additive product | 24 | Additive normalized join | Product source; old-fork-inexpressible |
| Negative cases | 20 | 10 each new topology | Degree-seven observed-output mutation, rejected by all 64 graphs |
| Future collisions | 24 | Six base/mutant pairs per new topology | Identical visible data and query, degree-15 future mutation |

All normal generated examples have a valid executable graph in their declared topology. The two negative splits are selected only after exhaustive finite-class rejection. Source selection does not consult frozen policy outputs, scores, or hidden future predictions. This one seed, its hand-designed shift classes, and all model scores are now public.

## Frozen comparison conditions

- Frozen 7,353-parameter RRC-14 GRU edit controller and an identical checkpoint with recurrent history erased at every failed candidate.
- Frozen 7,393-parameter RRC-14 full-feedback feedforward edit control.
- Two untrained 12-query frontier policies using the same RRC-13 frozen initializer: observed-parent residual ranking and deterministic visible-seeded random priority.
- A fixed first-twelve-graphs canonical catalogue without a learned initializer, using exactly the same full-degree candidate execution contract.
- Complete 64-candidate exact search for a **larger-budget upper reference**.

The frozen recurrent, reset, feedforward, residual and random arms all start from the same feedback-disabled RRC-13 initializer. Each uniquely tested candidate consumes one graph query, executes both degree-12 demonstrations and returns the same exact residual statistics. Inference has no test-time access to graph label, source generator identity, hidden graph, future target or any unexecuted candidate response. The topology itself is public to the environment's interpreter; the model receives no explicit topology-ID input channel. Search over the reversible graph can restore parent states without charging an already evaluated graph again. Each model's final selection must be exactly verified on observed coefficients before future grading.

Report accepted executable graphs, correct predictions through degree 32, false future acceptance, visible-prefix aliases, exact candidate queries, primitive applications, nominal coefficient-work proxy, FP32 dense MAC proxy and paired per-case outcomes. An improvement limited to a topology or source family is reported narrowly. A strong learned-search superiority claim requires the frozen GRU to exceed the matched FF and the strongest untrained 12-call policy on the combined novel topology shifts, with computation and earlier training privilege disclosed.

## Reproduce

All evaluation and CI replay use Python 3.10+ standard library with frozen JSON checkpoints:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_topology15.py' -v
python research/rrc_compactification/bench15.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_graph14.py' -v
```

The exact receipt saves executable source and frozen checkpoint SHA-256 hashes, roster digests, 10 split-level score tables and candidate-trajectory hashes. Old RRC-13/RRC-14 modules and checkpoints remain unchanged. A mismatched source or receipt fails CI. No new neural optimization or downloads are invoked. Every independently graded future execution is excluded from the policy's interpreter-call budget.
