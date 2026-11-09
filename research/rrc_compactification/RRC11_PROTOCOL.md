# RRC-11 exploratory CPU protocol

Executed November 10, 2026. This file documents one exploratory fixed-seed experiment written during its implementation. The test roster is exposed, so later model selection must use new held-out seeds.

## Evidence and split

A case supplies **two independent input/output demonstration series** through coefficient degree 12 and a fresh input query; the requested output continues through degree 32. Hidden graph labels and future coefficients never enter inference. We train a four-weight linear preview ranker on 160 examples from nine branch graphs: left/right in `{I,N,R}`, post always `I`. Development uses 32 new cases from the same nine graphs. Evaluation uses 80 examples whose three-node graphs contain previously unseen `S` or post-`R` structure, 20 deliberately inconsistent observed demonstration pairs, and eight pairs duplicated at visible degree 12 but mutated at unseen degree 15.

The recurrent/fork-join computation belongs to the **supplied formal-series interpreter**. The four-weight learned ranker is a lightweight nonrecurrent surrogate. This experiment therefore tests transfer of a ranking policy across novel graph syntax, with a public program interpreter, rather than making a claim about a neural model inventing graph structure unaided.

## Controls and acceptance

- Exact enumerative program synthesis tests all 64 legal graphs against the same two degree-12 demonstrations. It can stop after finding a matching candidate. Acceptances must be exactly verified on both demonstrations.
- The learned ranker and an untrained deterministic residual heuristic each preview all 64 graphs through degree four on both demonstrations. Only their first four ranked programs receive full degree-12 checks. Report total previews and work, including failed candidates.
- A blind one-candidate control uses the first training-grammar candidate with no new demonstrations for graph choice. Its candidate still requires finite verification.
- Report accepted finite programs, degree-13–32 extrapolation, wrong future acceptances, mean candidate checks, primitive calls and nominal coefficient arithmetic. Checkpoint SHA-256, training roster digest and source hashes are pinned in the replay.
- Late identical-prefix pairs receive identical proposals. A future mismatch does not revoke a valid finite observed-prefix certificate. Out-of-grammar cases must not be mislabeled as universal unsatisfiability.
- A recurrent-search superiority claim would require a recurrent learner and lower total search work than the untrained residual sorter. This experiment has neither result; its findings are an interpretable baseline for a later model design.

## Reproduce

From repo root, Python 3.10+ standard library:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_graph11.py' -v
python research/rrc_compactification/train11.py --out /tmp/rrc11_ranker.json
python research/rrc_compactification/bench11.py --check
```

Compare `/tmp/rrc11_ranker.json` byte-for-byte against the frozen checkpoint; no GPU, downloads or PyTorch needed. Tests cover independent polynomial arithmetic, grammar boundaries, late collisions, candidate-call accounting and source-hashed receipt replay.
