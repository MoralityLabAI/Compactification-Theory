# RRC-22: arithmetic work and selective-search break-even mathematics

Let the fixed bounded graph grammar have certified class representatives \(P_\tau\) for topology \(\tau\), and let \(x\) be a pair of observed degree-12 demonstrations. A verification of candidate \(p\) executes the exact graph program on **both** demonstrations and compares its coefficient vectors with the observed outputs. Each policy produces an ordered list of distinct representatives, stopping when one verifies or its query allowance expires.

Write \(C_{\rm init}\) for class-table compilation and other shared setup, \(D(x)\) for source-shift detection, \(N(x)\) for neural inference where required, \(R(x)\) for class-order computation, and \(V(p,x)\) for exact execution of the two demonstrations. If \(A(x)\) is the set of graph candidates actually executed by a selective policy, its total deployable path time is

\[
T_{\mathrm{select}}(x)=D(x)+\mathbf{1}_{\mathrm{neural}}(x)N(x)+R(x)
          +\sum_{p\in A(x)}V(p,x).
\]

For complete certified search with ordered candidates \(B(x)\),

\[
T_{\mathrm{complete}}(x)=R_{\mathrm{mass}}(x)+\sum_{p\in B(x)}V(p,x).
\]

The shared symbolic compiler contributes to cold requests separately. Neural training and calibration creation are sunk acquisition costs, and should be charged or amortized over the intended workload. The break-even condition over a roster is

\[
\sum_x[D(x)+\mathbf{1}_{\mathrm{neural}}N(x)+R(x)-R_{\mathrm{mass}}(x)]
\le \sum_x\left[\sum_{p\in B(x)}V(p,x)-\sum_{p\in A(x)}V(p,x)\right].
\]

The right-hand side is an **elapsed time saving**, which depends on the actual graph programs executed and their source coefficients. A raw candidate-count improvement is insufficient, since convolution/inversion and additive/Hadamard joins have different costs.

## Metered integer arithmetic

The independent exact interpreter tracks every finite multiplication \(a\cdot b\) and addition \(a+b\), including intermediate sums inside truncated convolution and reciprocal-series inversion. With \(\beta(n)=\mathrm{bit\_length}(n)\), report

\[
W_\times=\sum_{\text{multiply calls}}\beta(a)\beta(b),\qquad
W_+=\sum_{\text{add calls}}\max(\beta(a),\beta(b)).
\]

These are transparent integer-operand size proxies. They do not equal measured CPU instructions. They ignore Python allocation/branch costs, large-integer algorithm switching, and the fact that multiplication by zero is much cheaper than multiplying arbitrary operands of the same width. Record the maximum operand and intermediate bit width, integer comparison counts and the actual arithmetic-call count. The metered interpreter must produce the same exact integer series as the production interpreter on all tested graph/topology combinations. Measure wall time on the *uninstrumented* production interpreter to avoid attributing instrumentation overhead to policy execution.

For degree \(T\), ordinary convolution needs \(\sum_{n=0}^T(n+1)= (T+1)(T+2)/2\) coefficient products, plus additions. Unit-series reciprocal inversion uses \(\sum_{n=1}^T n=T(T+1)/2\) products. Normalized additive or Hadamard joins require \(O(T)\) coefficient operations. Candidate mix, input bit widths and verification order therefore govern cost independently of the number of attempted programs.

## Full-accuracy finite baseline

If all source programs on the roster belong to the declared finite certified graph language, exhaustive class search can accept every uncorrupted demonstrated prefix. The selective policy with bounded or conditionally expanded candidate budget can still abstain on valid cases, since it may explore a different order and stop before the matching class. The RRC-21 positive roster contains 480 such in-grammar cases. The six future-collision pairs retain an unidentifiable degree-15 distinction, so every prefix-only method must produce the same prediction for both members of each pair. This is separate from the computational cost comparison.

The degree-12 class equivalence certificate is exact for integer inputs to that truncation. It does not extend by computation alone to all infinitely many coefficients. Timed CPU results are also not a general complexity theorem: these measurements use one Python implementation and a specific CPU with fixed short inputs.
