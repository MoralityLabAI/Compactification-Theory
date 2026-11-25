# DCF-14 theory

**Exact cost of elimination.** For an elimination order σ on the query component, bucket elimination performs `Σ_v 2^{|U_v|}·|B_v|` table products, where `U_v` is the scope union of bucket `B_v`. Its largest table has `2^{w_σ+1}` entries, where `w_σ` is the induced width of σ. Both quantities depend only on factor scopes, so they are known before execution (`fg.symbolic_ve`, tested equal to executed counts). Treewidth is `min_σ w_σ`; min-fill and min-degree give upper bounds, and the degeneracy bound gives a lower bound.

**Why width rather than state dimension.** The joint belief state has `2^n` entries, but exact inference over a parity or marginal query needs only the query's component (DCF-13 T2), and within it only tables of size `2^{w+1}`. Hence `log cost ≈ w + log(#factors)`, independent of n for bounded-width families (chains and trees cost O(n) at width 1).

**Cutset conditioning** with cutset C enumerates `2^{|C|}` assignments and peels the remaining factor forest. A forest always has a variable in at most one multi-variable factor, so peeling never enlarges scopes. Cost: `2^{|C|}·O(Σ|f|·2^{|f|})`. Because `w ≤ |C| + 1` for loop cutsets, cutset cost upper-bounds the elimination exponent.

**Where width is the wrong currency.**
1. *Algebraic structure:* noise-free XOR systems are linear over GF(2). The posterior is uniform on an affine subspace, and the query parity is determined iff the query vector lies in the row space: O(F·n) bit operations at any width.
2. *Approximation accuracy* depends on coupling strength and loop structure, not width; weakly coupled dense graphs are well approximated by BP.
3. *Decision value:* when every action's expected utility is near zero, abstention is optimal and no inference is worth its cost.

**Approximate operators under a budget.** Mini-bucket elimination with i-bound i caps tables at `2^{i+1}` entries. Summing within each mini-bucket gives an upper bound on each query-configuration mass, but no bound on their ratio. Loopy BP is exact on trees. Likelihood weighting gives an ESS-based interval; it is used for abstention but is not a guarantee.

**Verification.** A reference posterior counts as verified only when two independently coded exact paths agree within 1e-8: einsum VE, plus at least one of log-space enumeration, batched cutset conditioning, or broadcast VE with a different ordering.
