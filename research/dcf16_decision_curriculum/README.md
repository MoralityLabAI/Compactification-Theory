# DCF-16: verifier-grounded curriculum of computational decisions

A dataset of 920 inference decisions, each grounded in an exact verifier. It is built from the DCF-14 problem generator, operators and verification paths. Every record contains:
- the problem: variables, prior, factor scopes, sensor kinds, observations, noise, parity query;
- the observation history;
- candidate dependency representations, each marked sufficient or not: unary evidence only, one-hop neighbourhood, query component, min-fill elimination order with induced width, loop cutset;
- the operators available under the record's budget, with exact counted costs, posteriors, actions and regrets;
- the selected strategy and its recurrent depth, plus intermediate receipts (elimination trace, BP iterations);
- the final decision or abstention;
- the exact verification outcome: the reference posterior, which independent exact paths agreed, and the truth bit;
- the regret relative to the best available decision;
- supervision targets for a compact recurrent model (relevant variables, dependency edges, log2 exact cost, algorithm, recurrent depth, reference probability, abstention);
- hard-negative flags, set where a superficially plausible representation (unary evidence, one-hop neighbourhood) or BP reaches a different decision than the exact posterior.

| Split | Families | Sizes | Examples | Hard negatives | Selected: exact / mbe / bp / particle / abstain |
|---|---|---|---:|---:|---|
| train | chain, tree, grid, sparse | 8–32 | 600 | 324 | 283 / 18 / 3 / 0 / 296 |
| val | chain, tree, grid, sparse (fresh seeds) | 8–32 | 120 | 62 | 56 / 5 / 0 / 0 / 59 |
| heldout | star, dense, hyper, ktree, xor_noiseless, weak_dense (12–64); chain, grid, sparse (48, 64) | 12–64 | 200 | 117 | 78 / 16 / 3 / 1 / 102 |

Held-out families never occur in train or val, and held-out sizes of the shared families exceed every training size. Canonical problem hashes do not repeat across splits. `curriculum16.py verify` re-derives every reference posterior and regret from the serialised record (0 mismatches over 920 records; `verify16.json`). Forty generated held-out problems had no independently verifiable reference (dense and weak_dense at n ≥ 48) and are excluded rather than labelled.

Where it is used:
- **Data Goblin**: a schema, a loader that enforces split hygiene, and per-target training views.
- **EvalMath**: a compute-aware decision and algorithm-selection evaluation on the held-out split.

The results files record commits and scores for both. No model was trained on this curriculum in this session.

What the data is not: natural-language reasoning traces, or problems outside binary factor graphs with parity queries. The selection label uses a fixed cost weight (λ = 0.05) and is one defensible target among several. Half the labels are "abstain", because exact utility is zero whenever the posterior is uncertain; DCF-14 shows that this imbalance pushes learned selectors toward abstention.

## Reproduce

    python curriculum16.py generate     # deterministic; MANIFEST.json records uncompressed SHA-256 per file
    python curriculum16.py verify
    python -m pytest tests

Requires `../dcf14_width_routing` (fg.py, gen.py, width14.py), NumPy 2.3.5, scikit-learn 1.7.2 (imported by width14).
