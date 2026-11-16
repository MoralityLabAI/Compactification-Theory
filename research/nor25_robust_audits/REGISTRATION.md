# NOR-25 prospective registration: uniform-reserve robust coefficient auditing

**Registered 2026-11-16, before generating any NOR-25 source fixture, challenge schedules, attack positions, or outcomes.** Base: merged NOR-24 PR #192 at `f28edf3b65fb4b834b661ead28d23b290e6f88e3`. This is a new source experiment and preserves the NOR-24 archive and registered result. If a correction is needed, append a dated amendment; preserve the first registration commit.

## Fixed hypothesis and threat model

NOR-24's sequential posterior achieved 34.9699% primary weighted detection against 34.8674% for static prior and 6.1458% for uniform, but its off-template 16-query detection was zero. NOR-25 tests whether **eight learned-risk inspections followed by eight uniform inspections over the remaining coordinates** retain a positive source-level template advantage while recovering an explicit worst-case blind-attack detection floor.

The adversary fixes a set of exactly r altered coefficients BEFORE the audit samples its secret random schedule. Alterations use an honest exogenous exact target witness, integer deltas 1 and 4294049777, and are judged by intersection between the inspected positions and changed set. For an informed adversary that sees the realized schedule and can choose alterations afterward, the fixed-blind lower bound DOES NOT APPLY: all partial audits can be evaded when r unchecked coordinates exist. This negative control is mandatory.

## Pinned priors, new sources, and isolation

- Reuse the original RRC25 exact sixteen-template family, NOR-21 streamed public verifier, and NOR-24 public-only proposal interface without modifying their source code.
- Source generation: independently seeded **12 new held-out source identities**, replicates **4 and 5**, degrees **512 and 1024**, all three original NOR-21 families at each degree. Local source seed `corpus.SEED + 99325000 + 1000003*replicate + 100009*degree_index + 10007*family_index`; accept first attempt among 100 with unique public digest, unique public header, nonzero native mask and unanimous full-output prediction. Output lengths m=769 or 1537. These sources were absent from NOR-24 training and evaluation. No model training occurs in NOR-25.
- Source-only fixture holds two observed degree-D examples, the public query input, original native public acceptance mask, inferred header, seed, replicate, and public SHA; it excludes the planted program, true future, witness values, and attack positions. Save its bytes and SHA-256 BEFORE evaluating attack labels.
- Freeze the NOR-24 TRAIN-only fitted prior as exact fractions: tail 295/772, magnitude 246/772, lag 153/772, full-uniform background 78/772. Do not refit on NOR-25 test identities or outcomes. The 96-element tail, highest-absolute-coefficient, and delay+spacing supports remain public and predeclared. The off-template stress support is the complement of the union of those three structural regions.

## Policies, randomization, and primary decision

- Audit budgets k=(4,16,64), changed-coordinate radii r=(1,4,16), two exact nonzero deltas, 1024 independently salted trial ordinals per held-out public source and policy. Test kinds: tail, magnitude, lag, uniform background, and separate off_template. Each source/kind/r attack set is sampled ONCE, from public-only support with SHA-256 seed stream **202611250072**, before generating audit schedules; it is held fixed across all challenge ordinals, policies and deltas.
- Seven nonprivileged policies: uniform random without replacement; frozen static trained-prior rank; original clean-path sequential posterior; shuffled-prior ranking; hybrid_25 (25% exploration), **hybrid_50 (primary, 50% exploration)**, hybrid_75 (75% exploration). Include a privileged known-support oracle separately, never in the primary comparison.
- For each hybrid policy and budget k, first choose the top `t=k-u` ranked static-prior coordinates, then draw exactly `u=floor(f*k)` unseen positions uniformly without replacement from their complement using a distinct SHA-256 audit stream **202611250073**, source digest, policy, k, and trial. At k=4/16/64 the reserve sizes are (1,4,16), (2,8,32), (3,12,48). Schedules for different k are separately salted, and need NOT be nested. Baseline static and sequential policies retain their original top-k prefix. No audit selector reads private truth, blind altered positions, attack kind, or test results.
- Primary grid k=16, r=4, delta=1. Source-weighted in-template mixture fixed at (tail .4, magnitude .3, lag .2, uniform background .1), as in NOR-24. **PASS** iff hybrid_50 beats uniform on at least 9/12 independent sources, yields mean in-template detection gain >= +0.10 absolute vs uniform, exceeds static ranking on the off-template fixed attack on at least 10/12 independent sources, AND attains >= 0.02 source-mean off-template detection. Record failure without adapting the protocol. Additional static-retention ratio, source-specific changes against posterior, and alternative reserve allocations are descriptive only; no second primary gate is picked after data.
- Verify exact magnitude invariance, legitimate-target full acceptance, and informed-adversary evasion in every source/attack/policy/budget cell. Report individual source differences and pooled numbers; challenge replicas share 12 source units.

## Mathematical guarantee, independent of source distribution

Let U be the m possible output coordinates, let T be any fixed size-t set of risk-ranked coordinates, let A be any **fixed** size-r attack subset selected before the exploration randomness, and let R be a uniform size-u subset of U\\T, independent of A. The exact blind detection probability is 1 if A meets T; otherwise it is

    1 - C(m-t-r,u)/C(m-t,u).

Thus the minimum over all fixed size-r attacked sets (where m-t>=r and m-t>=u) equals this same hypergeometric value. For u=0 and a feasible disjoint attack the worst-case detection is zero. The finite combinatorial formula gives no computational randomness-security guarantee: a pseudorandom SHA256 seed in public CI is a reproducibility device rather than hidden production entropy.

## Receipt and review gates

- Commit this registration before producing any fixture or experimental result.
- Independent fixture generator parity checks versus the original RRC25 native 16-template public mask; reconstruct sealed true targets only AFTER freezing public proposals and policy schedules.
- Exhaustive small-m combinatorial oracle for hybrid blind detection probability and adversary-first/eavesdropping counterexample. Per-source SHA256 summaries of attack positions, schedule sampling and observation rows; independent second-process replay on the committed source-only fixture.
- GitHub Actions Python 3.12 CPU; tests and heldout run are separate jobs. Commit source only in first pass; attach unaltered fixture and complete outcome as CI artifact. If and only if the first run succeeds, preserve exact original bytes in a committed immutable archive and audit their SHA256 hashes.
- Interpretation is conditional on a synthetic finite attack model, authentic external target values, fixed blind attacks, disclosed support templates and the independent audit-randomness assumption. No complete-target soundness or live-verifier security claim is implied.
