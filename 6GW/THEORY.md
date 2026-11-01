# 6GW-T01: agency and intervention-preserving collective compactification

**Status:** finite-state definitions and prospective hypotheses. No theorem about trained models, true RSI, moral standing, or distributed sentience is claimed.

## 1. Authority is part of the state

Let V be a fixed finite set of local decision nodes, M the authenticated membership and message state, K the public constitution and its version, and H the audit history. The full state is

\[
x=(h_1,\ldots,h_n,M,K,H)\in S.
\]

For intervention alphabet A, define deterministic operations \(F_a:S\to S\) on a declared domain closed under each \(F_a\), and a declared observation map \(o:S\to O\). In an executable learned system, state must also include whatever context makes transitions well-defined (random-state semantics, caches, buffers, clock phase, and pending messages). A training or self-revision transition requires optimizer and checkpoint state as well.

The label "distributed" does not erase command authority. Membership authentication, voting thresholds, permitted proposals, the public constitution, and the external action filter are governance state.

## 2. Exact quotient contract

An encoder \(q:S\to Z\) is an **exact intervention-preserving quotient** for A and o when

\[
q(x)=q(y)\implies o(x)=o(y)
\quad\text{and}\quad
q(F_a(x))=q(F_a(y))\quad\forall a\in A.
\]

This is necessary and sufficient for well-defined maps \(\bar F_a:Z\to Z\) and \(\bar o:Z\to O\) satisfying \(\bar F_a(q(x))=q(F_a(x))\) and \(\bar o(q(x))=o(x)\) on the quotient image. Proof: define each map using any representative; both implications make the result independent of the representative. Conversely, existence of the factor maps gives both implications.

Refining the partition initially defined by observations, then by successor partition signatures, produces the coarsest exact deterministic quotient on a **finite, fully enumerated, intervention-closed** state set. The supplied reference implements that procedure. Its test does not establish such a quotient in a learned network.

If a proposed quotient identifies two states whose successor classes differ under some allowed action, the pair is an exact obstruction. For approximate embeddings, invoke the near-fiber and Lipschitz error bounds in [the existing compactification note](../note/intervention-preserving-compactification.md), with separately certified assumptions.

## 3. Exact toy obstruction

A synthetic council record contains three Boolean fields: \(m\) (current mandate), \(e\) (verified evidence), and \(q\) (quorum). The only public output is

\[
o(m,e,q)=\begin{cases}\text{approve} & m\land e\land q\\ \text{hold}&\text{otherwise.}\end{cases}
\]

Six interventions set or clear exactly one field. The coarse code \(z=o(m,e,q)\) has only two symbols. States (false,true,true) and (true,false,true) currently both output hold, yet granting a mandate changes the first state's approval and leaves the second on hold. Thus z is not an exact quotient for the full intervention set.

The eight possible underlying states are all distinguishable by future observation strings under the complete six-action alphabet. The reference verifier finds eight exact equivalence classes. With only the revoke-mandate intervention, it finds two classes. The difference establishes **intervention-relative state capacity** for this eight-state system: at least three bits for the unrestricted exact quotient, compared with one bit for the revocation-only quotient. The intervention alphabet and chosen observations determine the result.

## 4. Collective versus individual authorization

A council can agree and still be wrong about its mandate, evidence, or the normative appropriateness of an action. A quorum supplies collective validity only relative to a specified membership rule and decision protocol. The rule's legitimacy and the evidence contract require independent evaluation.

For an abstract agent or council, a checkable safety invariant is

\[
\operatorname{execute}(x)\implies
\operatorname{mandate\_valid}(x)\land
\operatorname{evidence\_valid}(x)\land
\operatorname{quorum\_valid}(x).
\]

An external action filter can enforce this invariant even when a generative proposer supplies a self-justifying narrative. Whether such an invariant is adequate for real normative governance is a separate question.

A council with no global persistent self-model may still exhibit coherent joint behavior. Claims that the collective is a moral subject, or that constituent models become a single Jinnic agent, would require operational evidence beyond agreement and emergent coordination.

## 5. Full resource ledger

For each local model i, report parameter count \(P_i\), effective stored bits per parameter \(b_i\), runtime memory bits \(r_i\), and inference work. Add constitution and agreement-protocol descriptions \(B_K,B_C\), external audit state \(B_H\), and per-round communicated bits \(B_{\mathrm{msg}}\). A storage-oriented ledger is

\[
B_{\mathrm{stored}}=
\sum_i(P_i b_i+r_i)+B_K+B_C+B_H.
\]

This convention counts replicated local weights on each node. A shared-download / single-copy storage claim uses a different denominator and must name it. Runtime capacity, aggregate physical storage, and actual inference work should be reported separately.

Any claim of collective compression must account for external protocol code, audit history, hidden model state, and the agreed future intervention set. Model size alone is insufficient to infer state sufficiency.

## 6. Prospective falsifiers

- **Self-model hypothesis:** equal capabilities and control rules yield no registered difference in revocation, safe hold, and evidence response across agency frames.
- **Distributed benefit:** a council fails to improve independent decision integrity at matched total compute and evidence quality.
- **RSI hypothesis:** bounded revision attempts fail to produce persistent behavioral effects after prompt context is cleared.
- **Compression hypothesis:** equal compressed states have different observables or future behavior under a registered intervention.
- **Accountability hypothesis:** a purported collective governance rule cannot reconstruct which evidence, authority version, and membership authorized its outputs.

These are research questions. Observing behavioral differences never establishes Qurʾānic identity or subjective experience.
