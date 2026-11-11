# OIC-33: Proof-grade adaptive seven-phase minimization

**Prospective, 11 November 2026.** OIC-32 resolves the fixed global optimum `Q_fixed^*(7, one flip)=21`. The exact adaptive optimum remains in the computer-certified interval `15..20` after exhaustive Q14 UNSAT and an independently verified Q20 policy. OIC-29 Q15–19 search caps are **inconclusive**.

## Primary target

Decide Q19 with a complete finite proof or a verified winning policy. Extend to Q18/Q17/Q16/Q15 only after resolving Q19. Retain the 49-world two-error-layer belief state, all 42 affine phase gauges, legal adversarial online report branches, and the sound pairwise future-disagreement cut (OIC-T44). Each cap/timeout is logged as inconclusive.

Seek a **multihypothesis** impossibility cut stronger than the pairwise condition: a set of at least three shifts may remain impossible to separate under any common future action schedule even when no single pair saturates OIC-T44. Prove the cut on the explicit finite game, then compare it against unpruned exact searches on n<=4 before applying it at n=7. Preserve an independently checkable complete UNSAT trace or a winning policy tree with SHA256 and every real-world fault-path replay.

## Secondary target

Provide a structural bound on adaptive-vs-fixed resource savings at all prime phase counts. OIC-32 establishes the fixed one-flip optimum `3p`; its three-address block code is a control for every prime `p>=5`. Report each strategy's raw sensor bits, address bits, policy description, exact belief-state bits, host memory and actual search work. A preconfigured fixed schedule pays its instruction length separately from live online link bits.

A synthetic cyclic monitor is a mathematical test bed. Any transfer of these guarantees to trained recurrent networks requires a validated observation/transition homomorphism and independent held-out evaluation.
