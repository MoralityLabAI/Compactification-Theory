# RH-COMP-01 / RH-DAG-01 / RH-RAM-01: verifier-sensitive proof compactification

2026-11-27. Status: **source measurements and synthetic analyzer checks; preregistration draft for measured verification**.

## Evidence ledger

The source is [Sernau et al., grh-lean](https://github.com/lsernau/grh-lean/tree/72fead102f663d5c7a407964a13dc0cf9d977207), pinned at `72fead102f663d5c7a407964a13dc0cf9d977207`. Proof credited to Levent Alpöge and Claude; rewrites credited to the upstream authors. Lean v4.35.0, Mathlib `56a005855ea17e0b0958f73ec801ae140b0a1679`. The [one-line edition](https://github.com/1337isnot1337/riemann-hypothesis-minimal) is a string-literal archive/extractor and is excluded from mathematical size comparisons. The [MIT verification report](https://math.mit.edu/~drew/shortgrh/) is external evidence, not our replay. The [Lean FRO/Mathlib statement](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/) concerns the original proof.

| Metric | Readable | Compact |
|---|---:|---:|
| Lean source modules (tree count) | 803 | 794 |
| Lean source bytes (tree count) | 30050271 | 7420187 |
| Lib source modules | 495 | 573 |
| New source modules | 306 | 219 |
| New source bytes | 12416147 | 2393231 |
| Export bytes (external) | 4485867644 | 5172274528 |
| Export declaration records (external) | 216571 | 204583 |
| lean4lean peak GB (external, reported units) | 332 | 7 |
| lean4lean time (external) | 9305 s | 1719 s |

Full source bytes shrink 4.05x; New bytes shrink 5.19x. Export grows 15.3%. The RAM ratio 47.4x is checker-specific: Lean's own default/paranoid checker checked the readable export in under 8 GB. The reported dominant time cost is New.Variety.tiltTheory (7917635 ms) in readable src/Lib/PadicComparison/Primitive/PeriodCoefficients.lean. Compact counterpart: src/Lib/PadicComparison/Primitive/TiltedTheory.lean. The readable definition names several support lemmas; the compact definition assembles related subarguments inline. This comparison motivates reduction/conversion profiling, but does not isolate causality; both editions change many other definitions.

## Executed source graph pilot

`source_receipt.json` contains the pinned tree counts and a partial induced import graph for 4 readable and 11 compact modules, extracted from the first 40 lines. Unresolved imports are explicitly listed. These unequal samples cannot compare total graph topology. On compact sampled nodes lexical and greedy frontier peaks are 39679 and 40462 source bytes respectively (greedy loses by 783). Both policies are source-level scheduling proxies only. The analyzer's ten fixture assertions passed in the JS runtime. Full upstream checkout scan and Node CLI execution remain unrun.

Run on an operator machine:

    git clone https://github.com/lsernau/grh-lean.git
    git -C grh-lean checkout 72fead102f663d5c7a407964a13dc0cf9d977207
    node analyze.mjs --self-test
    node analyze.mjs grh-lean > full_source_graph.json

The CLI requires a pinned checkout with unchanged tracked files. Review header grammar against Lean's parser, validate resolved internal imports, and use compiled declaration dependencies for the real experiment. Source-byte weight is intentionally not an RSS estimator. Do not infer module unloading is supported by Lean.

## Mathematical model

For a DAG G with dependencies d(v), evaluated set S and weights w(v), define L(S) = {u in S: there exists v outside S with u in d(v)}. In a no-recompute abstract machine, allocating v next costs c(S,v) = sum_{u in L(S)} w(u) + w(v). The exact minimal peak on a small DAG obeys

    F(empty)=0
    F(S union {v}) = min over admissible predecessors S of
        max(F(S), c(S,v)).

The state is sufficient only under this toy lifetime model. A reduction machine requires additional state: shared expression DAG, conversion cache, transparency policy, local contexts, live objects and recomputation schedule. In particular, equal theorem statements and equal export bytes do not imply equal runtime or peak memory.

For a proof representation P and checker K define C(P,K)=(source bytes, export bytes, time, peak RSS). Hard acceptance first: identical challenge statements, exactly allowed transitive axioms, every configured independent checker accepts, fixed toolchain. Optimize Pareto cost among accepted candidates. Timeouts/OOM are censored costs, never kernel rejection labels.

## Next campaign (draft; freeze before evaluation)

1. Extract compiled declaration edges for each edition from the independently verified export; hash exports and challenge types; record version/flags and host topology. Report declaration-level ancestry of tiltTheory and top theorem roots separately from source import graph.
2. Produce tiny closed-root conversion-stress fixtures with controlled variants: auxiliary lemmas vs inline terms; repeated shared subexpressions; reducible/opaque constants where type checking remains valid. Rebuild from fresh state, verify exact target types and axiom closure; hold out entire construction families. Preserve trust boundary and never weaken an existing theorem or turn its body into an axiom.
3. Baselines: original source ordering, lexical topology, greedy frontier, exact subset DP on <=18 synthetic nodes, matched random ready order, and independent checker default. Model policies must use identical authorized rewrite menu and actual verification budget. Record build time, checker-only time, peak RSS, export bytes, term-node counts, declaration counts and conversion events separately; report uncertainty with repeated cold runs and pair by root.
4. Resource gates: start with synthetic and isolated small-root slices at 512 MiB, 2 GiB, 6 GiB and 12 GiB; operator-provided hard cgroup/job caps including descendants. Never attempt the readable full lean4lean replay on a local workstation. Full compact replay is a separate operator task; upstream recommends 24 GB for build and 64 GB for Comparator.
5. Primary prospective endpoint: fraction of fresh equivalent tasks accepted under the same hard memory/time cap; secondary: paired log peak RSS/time among jointly accepted roots, total cost including failed candidates and verifier calls. Seed-group by dependency neighborhood, not random proof tokens. No claim of universal savings or new mathematics. A positive result must transfer across held-out families and report a second checker separately.

Related work: MathCan RH proof distillation and DataGoblin RLT external Lean/MCP verifier interfaces. Keep the already public RH editions in reference/evaluation-only custody; use independently created and frozen training roots for generalization.
