# NOR-24 prospective registration: trained sequential witness auditing

**Frozen on 2026-11-16 before generating any NOR-24 training or held-out sources, attack labels, audit challenge schedules or results.** Continues merged NOR-23 (Compactification-Theory PR #191; MathCan #112).

## Sources and information

- Preserve original exact sixteen-program RRC25 and NOR-21 stream16 full-degree public verifier. 24 distinct source inputs: replicates 0,1 TRAIN and 2,3 TEST; each degree D=512 or 1024 and each of three NOR-21 families. Query OUTPUT length m=3D/2+1 (769 or 1537). Seed: original corpus.SEED + **94324000** + 1000003*rep + 100009*degree_index + 10007*family_index. First source within 100 attempts with unique public header, nonempty accepted mask and unanimous full query prediction. Distinct public hashes across train and test.
- Public fixture contains only two observed D-degree demos, query INPUT, inferred header, original exact acceptance mask, attempt/seed and public digest. There are NO private planted programs, true future outputs, altered coordinates, witness values or test attack labels. Freeze one exact public predicted query OUTPUT before regenerating the true program and native target in a postchoice grader.
- Four predeclared attack supports visible from the frozen candidate output and public header: tail = last 96 coordinates, magnitude = 96 largest abs(candidate coefficient) coordinates with ties at smaller indices, lag = 96 consecutive coordinates starting at header.delay + header.spacing, uniform = all m indices. An off_template stress attacker selects outside the union of the first three supports; exclude it from the primary mixture and require at least 16 eligible positions. The model is explicitly supplied these four templates. It learns mixture weights only, so results are conditional on correct support-template assumptions.

## Registered learning, policies, attacks

- Fit only on 64 labeled synthetic attack episodes per TRAIN source (768 episodes). Each training attack family drawn at weights tail .40, magnitude .30, lag .20, uniform .10, and four distinct attack positions sampled uniformly from that support. Draw using SHA256-derived seeds **202611240071**, public digest and episode number. Add one pseudocount to each of four family counts; fitted priors are frozen before TEST auditing. Family labels are supervised data, never inferred from test targets.
- TEST attacks: kinds tail, magnitude, lag, uniform and off_template; changed coordinate counts r={1,4,16}; integer deltas δ={1,4294049777}. For each TEST source/kind/r, fix the blind set with SHA256 seed **202611240072**, public digest, kind and r, INDEPENDENT of later audit draws. Both magnitudes reuse that set.
- Audit budgets k={4,16,64}, **1024** trial ordinals per TEST source/arm. Each policy draws an ordered 64-coordinate clean-witness-path schedule using SHA256 seed **202611240073**, digest, arm, and ordinal. Budget checks use nested prefixes so source/world/trial comparisons are paired.
- Arms:
  1. uniform: independently shuffled range(m).
  2. static_prior: rank coordinates by sum_f pi_f * 1{index in support_f}/|support_f|; seeded randomized ties.
  3. posterior: sequentially select highest conditional probability of an attack at next index, given all previous authentic audit checks matched. Calibration radius rho=4. For each family f, after t_f matched checks in S_f: likelihood L_f=choose(|S_f|-t_f,rho)/choose(|S_f|,rho); posterior family weight pi_f*L_f normalized; next-index score sum_f posterior_f * rho*1{i in S_f}/(|S_f|-t_f). Select maximum among unchecked, break equal scores using independent seeded tie randomness. The clean-path schedule is extensionally the actual sequential policy until first mismatch; stop at mismatch.
  4. shuffled_prior: uniform random permutation of the full static_prior coordinate order, retaining the rank distribution but destroying its localization.
  5. oracle_support: privileged upper bound given the true attack family, selecting a seeded random order inside its support. Never give the other arms this family label.
- A separately labeled informed adversary sees each policy's realized k-coordinate schedule and chooses all r changes in unchecked positions of its declared support, which is always feasible since k+r<=80 and |S|>=96. Expected exact detection zero.

## Frozen scoring and control

- On each test source, each kind×radius×arm×k calculate blind detection counts /1024, informed-evasion counts, exact original public mask parity and valid-target acceptance. Reuse both δ values to verify exact magnitude invariance. Report source-level deltas versus uniform, shuffled and static.
- Primary evaluated at k=16,r=4,δ=1: weighted detection over four in-template kinds using weights tail .4, magnitude .3, lag .2, uniform .1. **PASS only if posterior exceeds uniform on >=9/12 held-out sources, exceeds shuffled on >=9/12, and gains >=0.10 absolute in pooled source-weighted mean over uniform.** Failure retained as FAIL with all observations. No predefined advantage requirement for posterior versus static.
- Off-template attacks are a separate stress readout and excluded from the primary weighted mean; do not upgrade passes on k<m sampled coordinates into complete target correctness. Valid target must always pass, and informed attacker must always evade. Differences under δ=1 and δ=4294049777 must be exactly zero.
- Independent raw 24-source identity checks, source-locked candidate freezing, training counts, fitted weights, per-source world records, SHA256 receipt and an auditor importing original RRC25 but NOT the NOR-24 experiment implementation. Run original Python 3.12 Ubuntu CPU CI under 40-minute hard cap with independent replay and preserve first raw bytes. No retroactive change of grid or success threshold.

## Formal companion

MathCan NOR-24: same authentic checked-value transcript implies the same deterministic next query; exact combinatorial number of r-coordinate attacks remaining consistent with a clean audit transcript in a declared support is choose(s-t,r). The abstract proofs do not establish cryptographic trust or external random-seed secrecy.
