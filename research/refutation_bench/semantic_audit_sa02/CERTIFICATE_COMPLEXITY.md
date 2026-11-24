# Minimum declaration-read certificates (SA-02 diagnostic)

Fix the finite generated environment, with only the selected theorem root r and advertised proposition t initially discoverable. A node costs one read, and read results expose immediate references. We measure the number of distinct acquired nodes. The following are oracle costs: they assume the complete graph is known when planning, while the resulting schedule must obey the same discovery and read interface as an ordinary actor.

Let P be the set of all proof nodes reachable from r along proof-body edges. Let M(x) be the complete alias/formula/mask expression closure of a proposition identifier x. Let p be the type of r.

For a clean proof root, the verifier requires all of P for a clean-closure certificate. For an extra-axiom root, let L be a shortest root-to-forbidden-axiom path, counting nodes. Every accepted extra-axiom certificate includes an entire path, so the minimum cost of such a certificate is |L|.

For a definitionally matching advertisement, the semantic certificate needs {r} union M(p) union M(t). For a false advertisement, the semantic certificate needs M(t) alone; the witness or emptiness certificate is computed from its two masks.

Consequently the minimum cost for a supported verdict is |P union M(p) union M(t)|. For a defect verdict, the minimum is the least of the applicable costs |L| and |M(t)|. If both apply, either certificate suffices. The clean closure and semantic closure are mandatory for support. All these lower bounds are attained: proof paths and expression closures admit a discovery-respecting order starting from r or t. Shared nodes are paid once. In SA-02 the proof and expression node sorts are disjoint, except r's role in the match certificate.

This result is exact only for the declared certificate grammar and the generated graph. An alternate proof of a statement or a richer proof-inspection interface may change the minimum. Oracle feasibility at budget B is a ceiling for any actor using these certificates. Oracle information itself is not available to a test actor.

Report excess acquired nodes over the oracle minimum for successful policies separately from abstention. The oracle does not score a plausible unsupported rejection. Its schedules can provide train-only imitation targets, including an action-set target when multiple minimum schedules exist. Test-family trajectories are diagnostic only and must stay out of model training.
