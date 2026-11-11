# OIC-23: Certified robust query schedules above the information lower bound

**Prospective (11 November 2026).** OIC-22's learned policy failed the preregistered ten-point single-substitution gain at Q=n+2, and OIC-T36 shows a universal full-shift guarantee requires Q≥n(e+1)+r. The next exact study starts at this necessary horizon rather than assuming a subcritical monitor can become perfect through training alone.

## Registered objective

For small n=2,3,4,5, determine whether the lower bound Q=2n is attainable for one substitution, and Q=2n+1 for one substitution plus one marked erasure. Use a Boolean adversarial game over both shift and initial phase. Return either a complete policy tree with disjoint shift receipt sets (a *verified* certificate) or a failure witness giving two different shifts, phases and permitted fault patterns yielding the same received receipt.

## Methods and resource controls

Use exact zero/one mismatch sets or an explicitly budgeted product automaton for a simultaneous flip and erasure. Cache canonical candidate states and retain every distinct public receipt branch. Prove soundness of any memoized quotient used to prune the tree. First enumerate n=2,3 exactly. Then n=4,5 use certified branch-and-bound with resource caps stated in the receipt; timeouts are inconclusive. Keep online response bits, address bits, stored/preprogrammed schedules and adaptive policy description bits separate. Analyze the single erasure alone as its own channel.

## Learned-policy comparisons

Only after the exact oracle is available, train five independent 145-parameter scorers on a fault-correcting teacher using n=3,4,5, and evaluate n=6,8,12 on a frozen policy at Q≥2n. Include the original OIC-21 policy and OIC-22 negative student. Compare all-position fault recovery and clean degradation at **identical Q and addressing costs**. Publish worst-case phase and shift errors, positive/negative branch counts, failure witnesses and CPU/host-memory costs.

The theorem applies to a synthetic cyclic finite-state sensor interface. No inference about physical inwardness or large-model private reasoning is warranted.
