# QO-11: Monotone certification gaps and expected one-step acquisition

**2026-11-04.** Finite state/action hit/hold dynamics and synthetic observations. The statistical guarantee remains the unchanged QO-07 anytime e-process.

Let C_t be a nonempty rectangular uncertainty region with C_(t+1) contained in C_t. For each allowed partition pi, QO-07 defines robust one-step error R_t(pi) and optimistic error lower bound L_t(pi). They are respectively monotone nonincreasing and nondecreasing under interval refinement.

Set R*_t = min_pi R_t(pi), L*_t = min_pi L_t(pi), G_t = R*_t - L*_t. Finite minimum preserves each direction. For a robust-minimizing partition pi_R, L*_t <= L_t(pi_R) <= R_t(pi_R) = R*_t. Therefore

    0 <= G_(t+1) <= G_t.

A feasible certificate R*_t<=tau or infeasible certificate L*_t>tau persists after further nonempty interval refinements. These are deterministic finite mathematical conclusions. The QO-07 e-process supplies the separate time-uniform probability that the true rates lie in C_t.

For cell c with s hits in n draws, the **planning-only** Beta(1,1) predictive rate is p_hat=(s+1)/(n+2). The one-step acquisition score is

    A_t(c) = G_t - [p_hat*G_next_hit + (1-p_hat)*G_next_hold].

The selector maximizes its nonnegative rational score after computing both hypothetical QO-07 interval updates. This Beta predictive has no role in confidence coverage. Rare hypothetical empty intervals trigger an explicit fallback.

**Plateau obstruction:** With two wholly unobserved cells in one macro class, the other cell remains [0,1] after one probe. Both possible next gaps equal 1/2, so A_t(c)=0 for either cell. Use unchanged QO-07 bottleneck or width as recorded fallbacks. Such a myopic acquisition is not globally optimal.

The source-fixed pilot has 5 synthetic cases, seeds 401/409/419/421, query checkpoints 32/96/192, and policies round_robin, width, bottleneck, expected_gap_bottleneck, expected_gap_width. Three cases were published in QO-10; two are new source-frozen digital cases. All paired outcomes, negative results, statistical failures and exact lookahead computation counts remain in the source-hashed receipt. No neural training or transfer is claimed.

Sources: [QO-07](ANYTIME_ACTIVE_QUOTIENT.md); [QO-10](DECISION_BOUNDARY_RESULTS.md); Howard et al. (2021) https://doi.org/10.1214/20-AOS1991; Zhao et al. (2021) https://proceedings.mlr.press/v130/zhao21c.html.
