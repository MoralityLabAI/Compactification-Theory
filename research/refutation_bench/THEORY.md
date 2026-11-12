# RefutationBench: compact certificates with an explicit checking contract

**Scope.** The results below concern finite dependency graphs and finite propositional certificates. The proposed learning objective is empirical. Natural-language audit compression has no general soundness theorem here.

## 1. What a refutation certificate establishes

Write a case as `x = (source, claim, contract)`. A certificate `c` is interpreted by a fixed checker `V(x,c)`. The contract declares the source snapshot, the logic, admissible premises, and the statement being challenged. A checker may return accepted, rejected, or unsupported-for-this-language.

For an actual sound checker, the intended statement is

\[
V(x,c)=\mathrm{accept}\quad\Longrightarrow\quad R(x),
\]

where `R` is the **specific** property certified by that checker. For a countermodel checker, `R` says that the displayed premises fail to entail the displayed conclusion. A dependency checker instead establishes reachability in a supplied graph. An evidence-ID matcher establishes agreement with annotations. These contracts justify different conclusions; the score report must retain their distinctions.

In particular, a defective proof can have a true conclusion. A certificate showing that an advertised unconditional proof uses an extra assumption does not provide a counterexample to that conclusion.

## 2. Reachability certificates for axioms

Let `G = (D,E)` be a finite graph of declarations. There is an edge `d -> u` when the stored type or proof term of `d` directly references `u`, under the dependency extractor's documented rule. Let `A` be the set of declared axioms and `A0` the explicitly trusted basis. For a target declaration `t`, define

\[
\operatorname{Dep}_G(t)=\{u\in D:t\leadsto_G u\},\qquad
\operatorname{Extra}_G(t)=
(\operatorname{Dep}_G(t)\cap A)\setminus A_0.
\]

A path certificate `t = d0, d1, ..., dk = a`, together with the labels `a in A` and `a not in A0`, proves `a in Extra_G(t)` if every displayed edge exists. Checking a submitted path takes time linear in its length given constant-time edge lookup. Computing the whole reachable set by graph traversal costs `O(|D|+|E|)`.

The v0.1 implementation accepts an explicitly supplied acyclic declaration graph and checks a full closure certificate containing `used_axioms`, `local_hypotheses`, and `unused_axioms`. It validates references and cycles across the graph, including unreachable components. The path construction above explains the reachability principle; it is not an additional accepted certificate format. Comparison with a trusted basis `A0` requires that basis to be declared by the surrounding audit contract.

This statement follows directly from the definition of reachability. It applies to the **stored proof term** and the supplied graph. It does not show that every possible proof of the theorem requires `a`. A valid replacement proof may avoid the dependency. Conversely, scanning a file for the word `axiom` cannot show that any axiom is reachable from the target. The unused-axiom control exercises this distinction.

For Lean, kernel-produced dependency information at a pinned environment is stronger evidence than a lexical scan. A source-only approximation must be labeled as such. Imported declarations and aliases belong in the closure. An assumed carrier or sentence constant also requires interpretation: assigning a suggestive name to an opaque constant does not prove its equivalence to a mathematical object named in prose. A dependency path alone cannot establish that missing semantic bridge.

An honest theorem with type `P -> Q` exposes its premise in its statement. Its proof may be sound even when no proof of `P` is supplied. The benchmark therefore grades the advertised scope as well as the graph. A build failure establishes a failed reproduction attempt under recorded conditions; it supplies no mathematical countermodel by itself.

## 3. Finite countermodels

Fix a finite set of Boolean atoms and a propositional language with explicitly implemented truth tables. Let the case provide premises

\[
\mathcal P=\{P_1,\ldots,P_m\}
\]

and conclusion `C`. A submitted total valuation `v` is a countermodel certificate exactly when

\[
\boxed{
\bigwedge_{i=1}^{m}\llbracket P_i\rrbracket_v=\mathrm{true}
\quad\text{and}\quad
\llbracket C\rrbracket_v=\mathrm{false}.
}
\]

The checker validates the AST and atom assignments, evaluates every premise, and evaluates the conclusion. Missing atoms, unsupported syntax, and non-Boolean assignments are rejected. Cost is linear in the total AST size for tree expressions, with the valuation supplied as a lookup table.

**Soundness within this finite language.** Semantic entailment `P |= C` requires the conclusion to hold under every valuation satisfying all premises. An accepted valuation is an explicit witness violating that universal condition. Therefore acceptance implies `P not |= C`.

For example, premises `p -> q` and `q` with conclusion `p` admit `v(p)=false`, `v(q)=true`. Both premises hold and the conclusion fails. Changing the premises to `p -> q` and `p` makes conclusion `q` valid under every valuation. The previous assignment then fails a premise and must be rejected. This paired control checks the full contract.

Exhaustive enumeration over `n` Boolean atoms takes at most `2^n` valuations and is complete for this language. Finding and checking a witness have different costs. No such enumeration result transfers to unrestricted first-order logic, arbitrary Lean terms, or the zeta function. A propositional abstraction of a paper refutes only the abstracted inference unless the source-to-AST correspondence has also been justified.

## 4. First faults and certificate acquisition

The repository's [RRC-07 identity](../rrc_compactification/RRC07_THEORY.md) gives a stronger interface in its own domain. If `m` is the first nonzero residual for a proposed Euler-exponent vector `x`, then

\[
R_m(x)=m(a_m-x_m),\qquad
x'_m=x_m+R_m(x)/m=a_m.
\]

Earlier coordinates stay fixed because the coefficient equations are triangular. RefutationBench can borrow the practice of reporting a decisive local witness and charging its acquisition cost. It does not inherit this repair theorem for a prose proof. There may be several interacting flaws, and the earliest textual error need not be the easiest decisive certificate.

[RRC-25](../rrc_compactification/RRC25_THEORY.md) further demonstrates that visible agreement can leave hidden alternatives unresolved. Audits based on incomplete excerpts face an analogous information boundary: a missing premise elsewhere in the full source can change the appropriate verdict. The benchmark exposes this through `insufficient` cases and stated review scope, without treating the analogy as a theorem about mathematical prose.

Let `e` denote public evidence acquired so far, `a` a permitted retrieval or check action, and `B` the remaining cost budget. A planning target can be written

\[
Q_B(e,a)=
\Pr\bigl[\text{a contract-valid decisive certificate is obtained}
\mid e,a,\text{remaining budget }B\bigr].
\]

[QO-14](../ingression_dynamics_v1/CERTIFICATE_DISTILLATION.md) computes related certificate-reachability labels exactly in a small stipulated stochastic system. Here `Q_B` is a proposed target. Its value depends on the distribution, action semantics, and checking contract. Teacher audit frequency or model confidence supplies an estimate, whose validity must be evaluated. An independent checker can certify a returned finite witness even when the planner's probability estimate is poor.

## 5. Compression with retained coverage

Let `E_theta(x)` produce a certificate, possibly with abstention, and let `D` be the evaluation distribution over **source families**. Define

\[
L(c)=|\operatorname{UTF8}(\operatorname{serialize}(c))|.
\]

The serialization must be fixed so that spacing, escaping, and omitted fields cannot silently change the measurement. In addition to `L`, retain total output bytes. An actual tokenizer can supply a separate token count; bytes do not determine that count.

One experimental formulation minimizes expected certificate bytes subject to declared quality and resource constraints:

\[
\begin{aligned}
\min_\theta\;&\mathbb E_D[L(E_\theta(x))]\\
\text{subject to }&
\operatorname{VerdictRisk}_D(\theta)\le\epsilon,\\
&\operatorname{FalseRefutation}_D(\theta)\le\alpha,\\
&\operatorname{Coverage}_D(\theta)\ge\kappa,\\
&\operatorname{UnsupportedFindingRate}_D(\theta)\le\eta,\\
&\operatorname{CheckCost}(x,\theta)\le B\text{ for every evaluated case}.
\end{aligned}
\]

Coverage must be specified before optimization: it may mean correctly certified eligible cases or recall of annotated decisive findings. The latter is an annotation proxy, with its uncertainty stated. An empty answer reduces bytes and usually defeats coverage. A terse accusation can reduce bytes while raising false refutations. The constraints exclude both shortcuts from a compression claim.

A measured Pareto frontier over length, quality, and checked work is preferable when fixed thresholds are arbitrary. Use development families to choose a point, then freeze it before evaluating heldout sources. Paired full-audit and short-certificate outputs permit a direct test of what compression lost. Scores based on string or evidence-ID matches require human semantic review before being described as preservation of mathematical content.

## 6. Finite coding bound and cost ledger

For a narrow counting result, fix a decoder and an allowed query set `Q`. Suppose there are `K` classes of cases requiring distinct correct answer functions on `Q`, and a representation uses a fixed-width code of `b` bits. If the decoder receives **only that code and the query**, exact answers for every case require

\[
2^b\ge K,\qquad b\ge\lceil\log_2 K\rceil.
\]

Otherwise two classes share a code, and the decoder must return the same answer to a query that distinguishes them. This is the pigeonhole principle under the stipulated interface. If the decoder may reread the full source, the source is additional information and this bound does not charge it. An evidence pointer can be short because the verifier retrieves a long source. The source and retrieval cost therefore belong in the system ledger.

For `N` deployed audits, an accounting identity can separate shared setup from per-case work:

\[
C_{\mathrm{total}}(N)=C_{\mathrm{teacher}}+C_{\mathrm{train}}
+C_{\mathrm{deploy}}+\sum_{i=1}^{N}
\bigl(C_{\mathrm{infer},i}+C_{\mathrm{retrieve},i}+C_{\mathrm{verify},i}\bigr).
\]

Use this equation separately for compatible units such as elapsed compute seconds or monetary cost; do not add bytes to FLOPs. Storage has its own ledger: weights at actual precision, recurrent state, tokenizer, decoder, verifier, retained source, and cached certificates. Report parameter counts alongside that ledger. A small student using a large shared interpreter has the full stated system cost, with any amortization made explicit.

When likelihoods are available, define bpc explicitly as

\[
\operatorname{bpc}=\frac{-\sum_j\log_2 p_\theta(t_j\mid t_{<j})}
{\text{number of Unicode code points in the evaluated corpus}},
\]

using the actual token sequence, normalization rule, and likelihood convention. This measures coding loss on that corpus, not proof correctness. Certificate bytes, bpc, parameter count, and successful checked audits are separate measurements. The v0.1 offline harness provides no trained-model likelihoods or learned speedup result.

The research claim to test is that a student can preserve supported audit decisions and their checking contracts with lower measured deployment cost across heldout source families. The finite graph and Boolean results above justify specific checks. They supply no universal compression law for reasoning or prior evidence of a recurrent advantage.
