# DCF-16 theory: what a label certifies

**Sufficiency of a representation.** A candidate representation R (a set of variables together with the factors inside it) is *decision-sufficient* for a record when the Bayes action computed from the posterior restricted to R equals the Bayes action under the full posterior. The query component is always sufficient (DCF-13 T2), as are the elimination order and the cutset (each is an exact algorithm over the component). The unary-evidence and one-hop representations are sufficient only by luck of the instance. Records where they fail are the hard negatives. A model that relies on local evidence alone scores well on easy records and fails exactly these.

**Selection label.** Among the operators that fit the budget, the label maximises `EU(a | p*) − λ·ops/2^B`, where p\* is the verified reference posterior. Ties go to abstention. "Exact" resolves to the cheapest exact operator. Because the label is computed from p\*, it is a verified optimum for the stated objective, not a demonstration of what a model can infer from the inputs alone.

**Recurrent depth target.** For elimination, depth is the number of non-empty buckets; for BP, the iterations used; for abstention, 0. Depth is therefore a property of the chosen algorithm, which makes it a supervised target for adaptive computation time. It is not a lower bound on what any recurrent model needs.

**Verification.** A record exists only if its reference posterior was confirmed by an independent exact path (DCF-14 verification). Regret and the targets are deterministic functions of the stored record and are re-derived by `verify` and the tests.

**Contamination control.** Splits are disjoint by family or size, and by canonical-problem SHA-256. Training views (Data Goblin) must never read held-out records; the loader enforces this.
