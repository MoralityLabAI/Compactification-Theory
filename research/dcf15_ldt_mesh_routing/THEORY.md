# DCF-15 theory: routing objective for an inference mesh

**Units.** The latent state splits into units, and a partition groups units into blocks. Each block holds an exact table over its own states, conditioned on the evidence whose scope lies inside the block. Evidence that spans blocks is a pending dependency. In LDT terms the explicit lattice state `a_t` is, per block: the table, the pending dependencies, the entropy, the available operators, the symbolic cost of each, the verification stamp (the evidence count at the last exact pass) and `KL_hat`.

**Objective.** For a query and candidate operator a ∈ {local, cluster BP, exact on relevant component, abstain}:

`J(a) = E[R_hat(a)] + λ C(a) + μ KL_hat(a)·[a = local] + γ F(a)`

- `R_hat`: predicted decision regret. Zero for exact by construction; for abstain, the utility the oracle would have earned.
- `C`: modelled kernel seconds, `a_op + b_op · work`, fitted to timed batched NumPy kernels.
- `KL_hat`: estimated discarded correlation, `Σ (ln 2 − H(e_j))` over pending sensors touching the query blocks. `ln 2 − H(e)` upper-bounds the mutual information a binary sensor with flip rate e can carry about its scope.
- `F`: fragmentation. γ is charged when the operator's kernel has not yet been launched in the current batch step.

**Exactness.** Pending dependencies define a block graph. The query's relevant component is the set of blocks reachable from the query blocks through pending dependencies. By DCF-13 T2 applied at block level, exact inference over that component equals the global posterior; `tests/test_mesh15.py` checks this against independent elimination. Local answers are exact when the query blocks have no pending dependencies.

**Information lost by a partition.** For the true posterior p and a block partition Π, `KL(p ‖ ∏_B p_B)` equals the multi-information across blocks. It is zero iff the blocks are independent a posteriori. Merging two blocks removes their mutual term, at the price of tables that grow from `16^|B|` to `16^{|B|+|B'|}`. The slow timescale greedily merges the pair with the largest `KL_hat`, if it exceeds τ.

**Batching economics.** With B episodes per step, the kernel cost is `Σ_{ops used} (a_op + b_op Σ work)`. Heterogeneous routing launches more distinct kernels per step, so the fixed cost `a_op` is shared across fewer episodes. Homogeneous fixed routing has the most sharing.

**Why a learned router must clear a high bar.** Every router that reads unit features pays the feature cost, here in Python. The learned router also pays inference and amortised training. Its regret predictions must beat the symbolic thresholds by more than these costs.
