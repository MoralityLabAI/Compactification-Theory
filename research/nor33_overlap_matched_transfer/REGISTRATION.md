# NOR-33 prospective registration: exact overlap-matched OOD audit transfer

**Frozen November 17, 2026, before generation of any NOR-33 source, attack, schedule, or grade data.** This is the first NOR-33 commit. Companions: merged NOR-32 [Compactification-Theory #201](https://github.com/MoralityLabAI/Compactification-Theory/pull/201) and [MathCan #123](https://github.com/patrickdugan/MathCan/pull/123). Retain original outcomes and the registered FAIL if any criterion fails. All later protocol changes require a dated amendment before new data.

## Identification problem and frozen auditors

NOR-32's six newly defined attack families produced a preregistered OOD failure. Curvature96 was algorithmically new yet overlapped the old *magnitude* support by 74.75/96 indices and the old *tail* support by 54.96/96, and hybrid_50 detected more of these attacks than the NOR-31 mixture. This study **intervenes on actual support overlap**, rather than grouping source attacks by their nominal generation mechanism. The base six procedures remain exactly those of NOR-32, with a seventh **stratified_random** matched comparator.

Define historical public-risk union U(source) = tail96 ∪ magnitude96 ∪ lag96 from the original NOR-24 public-prediction templates. The complement is V = {0,…,m−1} \ U. Both have ≥96 indices at m=769 or 1537, or the source is a preregistered abort. Only the public proposed output and header define U, so sealed target values cannot enter the conditioning.

Freeze NOR-31 six distinct auditor classes, in order (uniform, static_prior, posterior, hybrid_25, hybrid_50, hybrid_75), and integer weights **(2,0,1,1,4,8)/16**. Keep unchanged NOR-25 full k=16 audit schedules, NOR-24 fixed training prior (295,246,153,78)/772, exact RRC25 16-program public verifier, and external exact integer target witness. No refit or budget change.

## Public-source roster and isolation

Generate **24 genuinely new public-only sources** with RRC25 replicate indices **40,41,42,43** × observation degree **512/1024** × three original NOR-21 wide families. Source seed = corpus.SEED + **130033000** + 1000003*replicate + 100009*degree_index + 10007*family_index; query input/output degree 3D/2 and output length m=769/1537. Accept the first case in ≤100 attempts with exactly one public header, nonzero exact native public 16-mask and unanimous full hypothetical query output across accepted templates; otherwise stop, retaining the failed registration. Require all 24 source hashes distinct and disjoint from 60 earlier source identities: original NOR-26/30/31 36 plus original NOR-32 24, checked at immutable NOR-31 and NOR-32 first-run archived fixture SHA-256s. The public fixture includes two observed degree-D demos, query INPUT, inferred header, seed, accepted attempt, full-public hash, native 16-mask. It contains **no planted program, true future output, altered coordinates, or target witness values**. Freeze one public full query-output proposal per source with NOR-21 stream16 before reconstructing hidden truth.

## Exact support and attack-overlap matching

Seven **source-visible complete rankings** over all m output positions:
- central: ascending |2i−(m−1)|, ties by index;
- stride17: all m positions in cyclic order (int(public_sha[:16],16) + 17*j) mod m for j=0,…,m−1; gcd(17,m)=1;
- low_abs: ascending |predicted_output_i|, ties index;
- curvature: descending |pred_(i−1 mod m)−2 pred_i+pred_(i+1 mod m)|, ties index;
- anti_prior: ascending frozen NOR-24 marginal risk score, ties index;
- hash: ascending SHA256(UTF8("NOR33|"+public_sha+"|"+decimal(i))), ties index;
- stratified_random: ascending SHA256(UTF8("NOR33-MATCHED|"+public_sha+"|"+decimal(i))), ties index.

For each family and *stratum* z∈{0,1,2,3,4}, construct a size-96 support by taking the first **24z** ranked positions in U and the first **96−24z** ranked positions in V. Each support therefore has exactly 24z positions inside the old risk union and 96−24z outside. Both support parts must have enough coordinates. Every source has exactly **7×5=35** support cells; the family mechanism still orders positions *within* each overlap block.

From each support cell, draw **8 fixed blind radius-four changed-coordinate sets** from a **conditional exact-overlap distribution**: select precisely z of the 24z support-U positions and 4−z from its 96−24z support-V positions, uniformly without replacement in each block. For z=0 and z=4 the empty part contributes zero positions. All such attacks therefore have **exactly z/4 changed positions inside U**, giving true actual altered-coordinate matching across *all seven families* at each stratum. The intervention deliberately changes the marginal attack-placement distribution from NOR-32; it is a prospective causal overlap-control experiment, not a replication of NOR-32's original random-within-family rates. Attack sets are fixed with SHA-derived stream **202611330071** before evaluation schedules are sampled.

## Equal-budget audit scoring and registered controls

Use **512** distinct challenge ordinals per source and auditor, with policy schedules from unchanged NOR-25 and newly domain-separated SHA stream **202611330072**. Each full plan inspects exactly k=16 distinct target-output indices. Evaluate all 35×8 fixed attacks per source against each of the six pure auditors, plus exact NOR-31 weighted-expected mixed detection and one independently seeded realized mixed-policy rollout (seed **202611330073**). Under the authentic external witness model, detect iff any sampled coefficient differs. Both nonzero integer shifts **1** and **4294049777** produce identical decisions; independently check actual target-coefficient comparisons for at least four trials per arm/support-cell/attack set while full evaluator uses equivalent exact set/bitset intersection. A separate fully informed attacker choosing four uninspected indices after an actual schedule must evade.

For every source, family, stratum, arm, compute exact detected and total 8×512=4096 counts and a theoretical conditional detection rate given each realized plan: with a=24z support-U positions, b=96−a support-V, inspected within each support t_U and t_V, the fraction of uniform *conditional* attack sets detected is

    1 − [C(a−t_U,z)/C(a,z)] · [C(b−t_V,4−z)/C(b,4−z)].

Average that exact expectation over the 512 source/arm schedules; publish empirical 8-attack rates beside it. The resulting discrepancy is a finite Monte Carlo residual, not an independent-source significance test. Keep the 24 public source identities as independent experimental units. Report each of seven families separately, including curvature's residual against the matched stratified_random family after controlling *both support overlap and exact actual attack overlap*.

## Frozen scientific tests

Define, for each source and z, the seven-family equal-weight mean d_z = detection_rate(NOR31_mix) − detection_rate(hybrid_50) over 7×8 fixed attacks ×512 plans. Define residual at z=2 as e = [hybrid_50−mix on curvature] − [hybrid_50−mix on stratified_random]. Test five independent gates:

1. **Overlap-gradient:** d_0 > d_4 on ≥18/24 source units, with mean d_0−d_4 ≥ **+0.015 absolute**.
2. **Low-overlap mixture advantage:** d_0 > 0 on ≥18/24 source units, with equal-source mean d_0 ≥ **+0.002**.
3. **High-overlap targeted advantage:** d_4 < 0 on ≥18/24 source units, with equal-source mean d_4 ≤ **−0.005**.
4. **Family residual after exact matching:** curvature has a stronger hybrid_50 advantage than stratified_random at z=2 on ≥16/24 source units and mean residual e ≥ **+0.005**. This is explicitly intended to distinguish old-union overlap from within-union risk ranking; it may fail.
5. **Exact integrity:** all new sources and 35 support cells verified, exact source-native public mask/full future parity, zero honest witness false rejection, positive ideal mixture uniform-reserve blind detection floor, zero fully schedule-informed detections, exact nonzero-shift equivalence, all 7×5×6×24 observed cells independently replayed, and original source/attack/schedule SHA-256 receipt preserved.

**Scientific PASS iff all five gates pass.** A failed test retains its outcome as FAIL with raw bytes, complete paired source/family/stratum counts and conditional exact combinatorial predictions. Equal-budget pure uniform is a diagnostic policy, with theoretical k/m and the NOR-32 uniform-attack duality as check; report it alongside mix and hybrid_50. This study makes no universal generalization claim outside the finite synthetic RRC25 model.

## Formal companion

MathCan NOR-33 will prove an exact **two-block hypergeometric numerator/denominator product** for altered sets constrained to z inside and 4−z outside a declared source-support partition, and prove conditional expectation invariance for any two audits with identical within-part inspection counts. It will retain the NOR-32 uniform-attack no-free-lunch theorem and a constructive schedule-informed alternate-target evasion boundary.

Witness authenticity, deployment entropy secrecy, unknown-source distributions, and general prover soundness remain external assumptions. SHA-derived audit streams are deterministic replay instruments, not cryptographic secret randomness.
