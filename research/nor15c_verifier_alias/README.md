# NOR-15C: exact four-check order ceiling for one aliased recurrent input

**Retrospective protocol, 15 November 2026.** This analysis follows the frozen NOR-15B holdout from merged [NOR-15 PR #164](https://github.com/MoralityLabAI/Compactification-Theory/pull/164). All 40 late_source examples used in that study have distinct observed source cases but the exact same original 9×8 model-visible input hash. The trained hidden state, terminal nonlinear features, and initial candidate logits therefore agree on these 40 examples.

The 40 examples produced all 20 original-GRU false accepts in the 1,040-case NOR-15B evaluation. An identical hidden state is an information bottleneck for a representation-only ranking policy, although subsequent exact public verification uses individual source observations and can return different outcomes.

## Formal finite decision-list model

Let \(\mathcal C=\{0,\ldots,15\}\) be the candidate templates. For case \(i\), let \(A_i\subseteq\mathcal C\) be the candidates accepted by the exact degree-192 public verification. Let \(F_i\subseteq A_i\) be those that also match the withheld degree-256 future. The native policy chooses one ordered length-four tuple \(q=(q_1,q_2,q_3,q_4)\) of distinct templates, then stops at the first \(q_t\in A_i\), or fails if all four reject. Its correctness and false-accept outcomes on case \(i\) are then determined by membership of that first accepted template in \(F_i\).

For all 40 inputs with a common model-visible tensor, a deterministic representation-only candidate ranker fixes one tuple \(q\). Every deterministic policy that sees only failed public checks before the first accepted template likewise follows one fixed ordered rejection path.

There are exactly

\[
(16)_4=16\cdot15\cdot14\cdot13=43{,}680
\]

distinct top-four orders. [ceiling.py](ceiling.py) evaluates every one using public acceptance sets, then determines the **hindsight** maximizer of future-correct cases, breaking ties by fewer false accepts and fewer verification checks. The same finite enumeration yields the lowest false-accept count achievable for each attainable future-correct total.

## Source, charges and leakage boundary

The RRC-25 checkpoint remains SHA-256 \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`. The original NOR-11/12 train and development roster hashes and the four NOR-15B source manifests are pinned. The 40 exact source cases are deterministically reconstructed and source identity is checked. The original frozen GRU's top-four policy must reproduce its earlier 8/40 future-correct and 20/40 false accepts on these same cases.

A public acceptance mask is determined with 16 exact checks of both observed demonstrations, for **640 additional retrospective public checks**. For every accepted candidate, the hidden future grader is then called, with each call separately counted. These retrospective oracle labels and the test-future-dependent optimum never train or select any implementable predictor. Its performance is a **test-label hindsight ceiling** on the fixed-four ranking family and is not a valid held-out model-performance estimate.

The original four paid checks per case are preserved in the separate benchmark; the present mask enumeration is an auxiliary research audit. A stronger inference process allowed to inspect the full public source, run longer proofs, or continue after finding a public acceptance can exceed the ranking family's ceiling, so the result is not a general observability impossibility theorem.

## Claims and output

The output records all 40 public acceptance bitmasks and after-the-fact future-correct masks, unique public-acceptance-signature counts, every candidate budget, the original four-template ranking, the exact best and Pareto-style minimum-false-accept frontier, and all source and checkpoint hashes.

The preregistered interpretive question is whether the performance ceiling of this **restricted deterministic four-check ranking family** remains low despite the availability of multiple exact public-accepted candidates, or whether the frozen original ranker's ordering has substantial headroom. The result is descriptive and cannot be optimized into a deployable policy without new training data and a separately locked test.

## Run

    python -m unittest research.nor15c_verifier_alias.test_ceiling -v
    python research/nor15c_verifier_alias/ceiling.py --check \
      --out /tmp/nor15c_full.json

The companion CI independently checks all source hashes, reconstructs the exact enumeration from stored masks without calling the original enumeration function, and verifies native original outcomes. The full JSON and SHA-256 will be committed as a separate immutable receipt after the first hosted execution.
