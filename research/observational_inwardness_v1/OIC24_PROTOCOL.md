# OIC-24: Does robust query complexity have a closed form?

**Partially executed, November 11, 2026.** The n=5 one-erasure threshold and the one-flip eleven-query impossibility are certified in [OIC24_NOTE.md](OIC24_NOTE.md). The twelve-query one-flip feasibility question and larger cycles remain prospective. OIC-23 establishes exact finite robust identification budgets through n=4 and a strict adaptive/fixed separation at n=4, Q=10.

## Registered questions

1. Test the conjecture \(Q^*_{\mathrm{flip}}(n)=2n+2\) for n≥5, beginning with n=5, while imposing exact node-expansion budgets and reporting timeouts as inconclusive. Seek an analytic construction and a lower bound stronger than OIC-T37.
2. Determine the minimum error-and-erasure horizon at n=5 and n=6, examining whether a parity or gcd-class obstruction changes the finite pattern.
3. Quantify description length of the adaptive policy tree and per-round address bits. Compare fixed and adaptive schedules at **identical** communication budgets, including an offline-programmed fixed schedule.
4. If computational pruning is introduced, prove that quotienting under uniform phase translations or shift-label automorphisms preserves the worst-case target and all future sensor outcomes, including erasure reports.

## Verification contract

Use exact integer state masks and save constructive policy trees. For each claimed feasible case, independently enumerate all true shifts, initial phases, and legal online faults; for each claimed impossible case, retain the full Bellman specification, explored-state counts and a reproducible, termination-complete search. Register node-cap and memory limits before moving to larger n.

Optional learned policy comparison occurs only after a certified exact reference is available at the same observation and address budget. Do not conflate Bayes success with a zero-error adaptive guarantee. The synthetic observation model supplies no claim about sentience, cosmology or pretrained neural systems.
