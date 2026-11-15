# NOR-15D: full public acceptance-mask capacity bound

**Retrospective secondary analysis.** NOR-15C established that no four-check fixed candidate ordering, shared by the forty aliased late_source cases, can reach more than 15 future-correct outcomes on that frozen test sample. The exact NOR-15C JSON and its original SHA-256 are already source-controlled in [results/nor15c_full.json](../nor15c_verifier_alias/results/nor15c_full.json).

NOR-15D asks a different question: suppose a stronger verifier spends all sixteen public demonstration checks, learns which candidate templates were accepted, and chooses one candidate based *only on the complete 16-bit acceptance mask*. How much can this newly exposed public verification signature recover?

For each full public accepted-set signature \(A\), the cases sharing that signature form a group \(G_A\). Let \(F_i\subseteq A\) denote the future-correct templates for case \(i\). If a signature-only policy selects the same candidate \(c_A\in A\) for all \(i\in G_A\), its future-correct total is bounded by

\[
\sum_A \max_{c\in A}\sum_{i\in G_A}\mathbf 1\{c\in F_i\}.
\]

This upper bound is attained by an independent optimal choice within every group. Randomizing the candidate given the same signature cannot improve *expected* correctness, since a convex average of per-template success counts is bounded by their maximum.

## Exact result from the frozen forty cases

| Information available to the hindsight selector | Maximum future-correct /40 |
| --- | ---: |
| Original GRU four-check ranking | 8 |
| Any common fixed four-check ordering, stop at first public accept | 15 |
| Complete 16-bit accepted-template signature, choose one accepted candidate per signature | **25** |
| Entire source-specific future truth | 40 |

There are fourteen distinct complete public acceptance masks. The most populous class has **thirteen distinct cases**, all sixteen candidate templates pass both public demonstrations for every one of those thirteen, and even the best single template is future-correct in only **three**. Ten future-correct outcomes are therefore lost in that equivalence class by every policy restricted to its full acceptance mask.

Across all fourteen classes, the signature-specific hindsight maxima sum to **25**, leaving **15 cases** that even the optimally chosen signature-only candidate fails. The original NOR-15C source has a future-correct candidate for every one of its forty cases. The failure arises from aliasing under the chosen observable signature, rather than a failure of the complete program family to contain a correct candidate.

## Experimental and causal scope

The bound reuses the *same* NOR-15B held-out examples, along with the source-locked exhaustive NOR-15C full public and future acceptance masks. The oracle's selections use hidden future correctness and are computed **after** evaluation. They are non-deployable hindsight ceilings, never newly validated learning results.

A policy that examines the complete degree-192 public demonstration coefficients, issues a richer algebraic query, or continues evaluating candidates after its first public acceptance may separate cases that have identical binary template-acceptance signatures. The 25/40 bound does not apply to that richer policy family and does not establish any distribution-level conditional mutual information statement. An independent fresh-source test would be necessary for new generalization claims.

The computation reads the immutable NOR-15C JSON, validates its SHA-256, partitions by exact public accepted-set bitmask, and enumerates every accepted candidate within each partition. The full per-signature counts and chosen hindsight template are returned as an auditable numeric certificate.

## Reproduce

    python -m unittest research.nor15d_mask_capacity.test_mask_ceiling -v
    python research/nor15d_mask_capacity/mask_ceiling.py --check \
      --out /tmp/nor15d_mask_capacity.json
