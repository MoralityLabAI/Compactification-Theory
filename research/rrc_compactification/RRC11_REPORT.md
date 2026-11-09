# RRC-11: unknown fork/join graph identification with few-shot demonstrations

**Executed November 10, 2026.** One deterministic standard-library CPU pilot, with 160 training examples, 32 development examples and 116 held-out evaluations. The graph *syntax* is hidden, while its operator interpreter is public to the search methods. A tiny learned four-weight candidate ranker uses two input/output demonstrations to sort proposed graphs. Its training only includes nine simple branch graphs. Evaluation includes unseen graph compositions with `S` and post-`R` nodes.

## Frozen results

| Method | New graphs, accepted and future-correct /80 | Out-of-grammar rejection /20 | Correct collision futures /16 | Nominal work / unseen graph case |
|---|---:|---:|---:|---:|
| Complete finite exact search, up to 64 graphs | **80/80** | **20/20** | 8/16 | **10,418.85** |
| Learned four-weight preview ranker, four full checks | 79/80 | 20/20 | 8/16 | 10,633.01 |
| Untrained deterministic preview ranking, four full checks | 79/80 | 20/20 | 8/16 | 10,633.01 |
| Blind first training-graph candidate | 5/80 | 20/20 by rejection | 0/16 | 538.69 |

The learned ranker and the untrained comparator have **identical recorded aggregate scores and compute cost** on all four evaluation splits. A restricted four-check budget misses one of the 80 new-graph cases. Full exact search finds every registered in-grammar case, at a slightly lower nominal coefficient-work cost than the preview methods on the unseen-graph split.

All eight future-collision pairs share identical demonstrations and query input; the target future series differs at degree 15. The exact solver produces the same graph for each pair and is correct on precisely one member. Full search's rejection of the 20 corrupted demonstration cases is complete only with respect to the specified 64-graph language through degree 12.

## Training and information

The learned candidate score has four floating-point weights `(0.10000000000000002, 0.1688888888888889, 0.09680000000000002, 0.0608)`. A deterministic pairwise-margin update made six training updates across 12 passes over 160 cases. Supervision uses graph executability and exact agreement on training demonstrations. A fresh checkpoint replay reproduces the saved JSON byte-for-byte. The ranker receives the count of exact preview coordinates, first mismatch position and signed residual statistics from **64 candidate executions** on both input/output demonstrations. Therefore its successful generalization depends on public interpreter access to unseen graph structures.

The fixed graph topology is `post(left(F)*right(F))` with the four known unary operators identity, sign twist, degree doubling and inversion. This is 64 total candidate graphs, with equivalent syntaxes permitted. It is a more expressive *fork/join* graph than a two-token sequential operator composition, but graph topology and each primitive's implementation are specified by hand. **No recurrent neural model was trained in RRC-11.** The ranker is a measured baseline for later direct recurrent graph induction.

## Resource ledger and scope

The four candidate-ranking weights are the only learned parameters (four Python doubles in the frozen checkpoint). Public graph syntax alone costs six bits, excluding the interpreter and output coefficients. Preview ranking spends 512 primitive applications and 9,600 nominal coefficient-loop operations per case before full verification. Full search short-circuits after the first demonstration mismatch. The operation proxy charges 3*(degree+1)^2 per graph/demo and excludes big-integer bit costs, interpreter code storage, Python runtime, data generation, and training work. These accounting limitations must accompany any efficiency claim.

The new 13-test suite checks formal series identities, hidden graph family isolation, finite negative examples, matched collision proposals, preview computation cost and source/checkpoint hashes. Standard-library replay compares complete receipt bytes. This run is exploratory and the test seed is now exposed. Its outcome gives **no learned advantage over a deterministic preview heuristic**.

## Reproduce

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_graph11.py' -v
python research/rrc_compactification/bench11.py --check
python research/rrc_compactification/train11.py --out /tmp/rrc11_ranker.json
cmp /tmp/rrc11_ranker.json research/rrc_compactification/results/rrc11_ranker.json
```

The next experiment should encode demonstrated series into a recurrent latent state **without candidate previews**, train a graph predictor on disjoint graph shapes, and measure verified executable-graph recovery at a matched inference budget against exact search and the above preview heuristic. The trained recurrent model must infer graph syntax from examples, with exact candidate execution reserved strictly for post-proposal verification.
