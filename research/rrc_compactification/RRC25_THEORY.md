# RRC-25: Delayed branch identification beyond the first residual

## Formal-series grammar

Let \(F\in1+q\mathbb Z[[q]]\), and let \(O,U,V\) be chosen from identity \(I\), sign substitution \(N:F(q)\mapsto F(-q)\), formal reciprocal \(R:F\mapsto1/F\), and degree substitution \(S:F(q)\mapsto F(q^2)\). The public interpreter computes

\[
G_{O,K,\Delta,c,U,V}(F)
=O(F)+cq^K+cq^{K+\Delta}U(F)+cq^{K+2\Delta}V(F).
\]

The finite syntax permits \(K\in\{64,80,96,112\}\), \(\Delta\in\{24,32\}\), \(c\in\{-2,-1,1,2\}\), and three four-way operator choices. There are exactly \(4\cdot4\cdot2\cdot4\cdot4\cdot4=2048\) literal programs. All operations are exact on the truncation through degree 192, since inversion is triangular and the other operations are causal. Two input/output demonstrations supply the observed coefficients. A separate query input with output through degree 256 is reserved for grading.

**Why the first residual is insufficient.** Fix \(O,K,\Delta,c\). For any two choices \((U,V)\) and \((U',V')\), their outputs agree at degrees \(0\) through \(K+\Delta\), because the two unknown transforms have unit constant term. Subtracting \(O(F)\) produces its first nonzero term \(cq^K\); after removing that term, the next nonzero term is \(cq^{K+\Delta}\). These observations identify the header \((O,K,\Delta,c)\) when \(O\) is correctly chosen, but are identical for all sixteen branch pairs. A concrete witness \(F(q)=1+2q+q^2+\cdots\) with \(U=I\) versus \(U'=N\) differs at coefficient \(K+\Delta+1\). Holding \(U\) fixed, \(V=R\) versus \(V'=N\) differs at coefficient \(K+2\Delta+2\). Thus one first-fault coefficient, even supplemented by the next delayed onset, cannot determine the complete suffix.

## Header decoding and fixed-budget verification

For each of four public candidates \(O\), independently calculate \(G(F_i)-O(F_i)\) for both demos. The earliest nonzero term must have a declared delay and amplitude; after removing it, the next nonzero term must occur at a declared spacing with the same amplitude. Both demos must identify the same header. **This is a public exact algebraic preprocessing step**, shared and charged to every ranking method. The teacher uses the true header during supervised training, while inference reconstructs it from the observed data. This is a substantial structured prior that the network does not learn.

The network receives two short windows of normalized public residual coefficients, each aligned using the decoded header. It ranks sixteen candidate pairs \((U,V)\) without executing a candidate transform. Each subsequently proposed pair is independently executed on both full degree-192 demonstration prefixes. The verifier never reads the hidden graph or query future. Acceptance establishes the two degree-192 equalities, and the 12-query method checks at most four pairs. The higher-budget complete factorized solver can inspect up to sixteen for each header, at most 64 full checks. The exact short-window comparator explicitly evaluates all sixteen templates against both demos to degree \(K+2\Delta+8\) via shortened operator traces before submitting a full candidate. Its previews are an additional measured resource.

## Observational equivalence and future uncertainty

A finite demonstration may leave two different suffix operations observationally equivalent. For example, when \(F_i=1+q^{L_i}P_i(q)\) with large \(L_i\), the value of \(U(F_i)\) below degree \(L_i\) is 1 and similarly for \(V(F_i)\). If \(K+2\Delta+L_i>192\), the second suffix operator is **completely invisible** through the observed truncation. An exact accepted class is then an equivalence class of candidate programs on these specific examples, not an identification of the source's hidden program. On the registered late-onset roster 28 of 32 cases admit more than one verified template, with up to all sixteen pairs matching the observed data. Hence even a class-complete finite solver can select a valid program whose future continuation differs from the generating program.

For a second, deliberately adversarial control, take a base target \(Y\) and replace its *future* coefficient at degree 224 by \(Y_{224}\pm1\), holding both demonstrations and query input fixed. Every deterministic prefix-based policy, including one with additional exact candidate calls on the same demonstrations, returns an identical ordered trace for both records. Exactly one member can agree with any single proposed future output. The six registered pairs therefore produce six false future acceptances for a complete visible-prefix solver. This does not turn a finite certificate into an infinite identity proof.

## Cost accounting and comparison scope

The finite DSL uses two bits per operator label, a fixed bounded delay/spacing/amplitude code, plus a shared interpreter. These syntax bits omit the model weights, decoding and inversion implementations, training demonstrations, learned header features, public coefficient arrays and computational state. The recurrent classifier has 1,000 FP32 weights and 12 live FP32 state values; the MLP has 995 FP32 weights. The portable model computes nine GRU steps, approximately 6,672 dense MACs, versus 968 MACs for the MLP. Both are scored under the same four *full* candidate-verification budget. All methods also pay for the public four-base header scan. The preview method additionally executes sixteen short pairs per compatible header; the full factorized solver may execute up to 64 long pairs. The nominal work proxy uses RRC-24's triangular reciprocal and linear other-operation estimates and explicitly includes calls to compute preprocessing, input features, previews and every submitted template. It is not elapsed time or exact big-integer bit complexity. Actual timed Python CPU results are stored in a separate machine-dependent receipt.

The mathematical identities are classical formal power-series algebra. The positive or negative empirical result concerns learnable template ordering under a supplied decomposition, **not** discovery of the operator vocabulary, new infinite Rogers–Ramanujan identities, or an intrinsic advantage of recurrence.
