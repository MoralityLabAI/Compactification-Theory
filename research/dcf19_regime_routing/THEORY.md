# DCF-19 theory

**Why coarse routing restores batching.** With B episodes per step and one operator per step, kernel time per step is `a_op + b_op·Σ work`. The fixed overhead `a_op` is shared by all B queries. Per-query routing with k distinct operators in a step pays up to `Σ_k a_k`. It also pays feature extraction per query, which in DCF-15 was about 0.5 ms of Python, larger than any modelled kernel except exact at B = 1. A regime decision costs `O(B·F)` once per epoch, amortised over `B·M` queries.

**What regime routing can win.** Its ceiling is the epoch oracle, which picks per batch-epoch the operator with the higher realised `Σ EU − λ·kernel`. Regime routing captures part of the gap between the better fixed operator and that oracle only if its statistic ranks batch-epochs by `ΔU = Σ(EU_exact − EU_bp) − λ·Δkernel`.

**Why S is a blunt statistic.** S sums the maximum information pending sensors could carry about their scopes. Three things separate it from the gain of exact over BP:
1. Most queries are answered correctly or abstained on by both operators, whatever S is.
2. BP is already exact on block graphs whose pending dependencies form a tree.
3. The utility gain concentrates in queries whose BP posterior sits near the 0.8/0.2 thresholds.

A sharper statistic would estimate the expected decision change, for example the batch fraction of queries with a BP margin below δ. It costs one BP pass, which the BP operator computes anyway.

**Equivalence of trace and online execution.** Mesh state depends only on absorbed evidence and slow merges; neither depends on which operator answered earlier queries. Every policy's outcome is therefore a deterministic function of a trace that records both operators' answers and work. `tests/test_regime19.py` checks this query by query against direct online runs.
