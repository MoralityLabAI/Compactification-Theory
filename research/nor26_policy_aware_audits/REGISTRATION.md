# NOR-26 prospective registration: policy-aware schedule-blind attack game

2026-11-16. **This registration is the first NOR-26 commit, before NOR-26 fixture generation and challenge data.** Parent study NOR-25 is merged Compactification-Theory #193 at b5bbc34b8692f33f771676df4fc258fd75b72ee0; MathCan #114. Preserve this first commit. Amendments must be append-only and dated.

## Threat model and hypothesis

NOR-25's hybrid_50 detected 3.17% of fixed off-template four-coordinate attacks, with a combinatorial guarantee for fixed blind attacks. Here a **policy-aware** attacker knows the public source, audit algorithm, NOR-24 prior, and a set of simulated **calibration** schedules. It must choose its four altered coordinates before the actual **evaluation** audit randomness is sampled; it cannot see realized held-out audit schedules. The question is whether the eight-uniform reserve retains meaningful detection against attacks optimized for the policy's distribution. An informed attacker that knows the realized evaluation schedule is a separately registered zero-detection control.

The fully uniform source-independent attack is included on eight fresh sets per source, to test attack-set idiosyncrasy. We distinguish mathematical exact probability floors from observed finite-sample detections, and reproducible public seeds from secret randomness.

## New source identity and isolation

- Twelve fresh public-only RRC25 inputs, replicates (6,7), observation degrees (512,1024), all three original NOR-21 source families, output lengths 769 or 1537. Source seed formula: corpus.SEED + 109326000 + 1000003*replicate + 100009*degree_index + 10007*family_index. Take first of at most 100 generating attempts with a uniquely inferred public header, nonzero native accepted 16-template mask and unanimous future prediction. The seed space differs from NOR-24 and NOR-25. Freeze source-only fixture bytes and SHA256 *before* attack or challenge generation.
- Fixture includes two D-degree observed demos, query input, public header, native public acceptance mask, seed, attempt and public digest; no generating program, target future, witness, or attacked coordinates. Generate source-only proposals before private reconstruction, with parity against the native RRC25 verifier. Hold original pretrained program family and NOR-21 stream verifier fixed.
- Reuse exact NOR-24 TRAIN-only prior (295,246,153,78)/772, 96-position tail/magnitude/lag windows and all-output uniform background. No refitting.

## Registered policies and attack computation

- Single primary k=16, r=4, integer deltas 1 and 4294049777. Seven unprivileged audit arms: uniform, static_prior, posterior, shuffled_prior, hybrid_25, hybrid_50, hybrid_75. Every selector is reused unchanged from NOR-25. Privileged true-support oracle is excluded from policy-aware game.
- Per public source **256 simulated calibration schedules**, drawn using SHA256 calibration phase stream **202611260074**. For each auditor policy calculate an integer bitset for every output coordinate marking the calibration schedules that inspect it. Attack optimization greedily selects four *distinct* coordinates minimizing the union of covered calibration schedule bits; deterministic ties are reordered by SHA256 stream **202611260075**, source digest, policy name, attack replica and coordinate. This is an explicitly approximate four-position distribution-aware best response; no global-optimum claim. Exactly **8** attacker tie seeds, replicas 0..7, per source/policy. Retain duplicate solutions and report distinct count.
- Fresh-blind control: eight radius-four uniformly random attack sets per source from the whole coordinate universe, independent stream **202611260076**, shared across policies.
- Per public source, generate **512 evaluation audit schedules** with distinct phase stream **202611260077**. Phase key is the SHA256 digest of stream, phase, and source digest. The evaluation schedules are independent of calibration and the attack tie seeds. Attacker choices are sealed before the evaluation schedule loop. Each of the seven attacker policy strategies is cross-tested against all seven auditors, producing a 7x7 detection matrix. Each uniform-blind attack is evaluated against every auditor on those same challenge schedules. The 8 attack replicates and 512 draws share one source unit.
- Fully informed attacker chooses four unchecked coordinates after each realized schedule and must evade; honest exact target always passes; both nonzero delta magnitudes must agree. Full source and per-attack positions plus all per-source counts retained.

## Primary prospective PASS/FAIL

For each source and policy, diagonal self-play detection is the average over 8 policy-aware attack replicas x 512 held-out challenge ordinals, where the attacker optimized on calibration schedules against that same auditor.

Primary PASS if and only if, at k=16 and r=4: **(1)** hybrid_50 diagonal beats static_prior diagonal on at least 10 of 12 sources, **(2)** source-mean hybrid_50 diagonal detection >= 0.020, **(3)** exact Python finite minmax formula agrees with 1-C(m-12,8)/C(m-8,8) and yields a strictly positive bound for every source, **(4)** the fully-informed attack evades all tests. All gates must pass; preserve an observed FAIL. Mean detection under uniform-fresh, source-level paired deltas, policy cross-play matrix, and attack calibration/evaluation generalization are descriptive, not alternative selection gates.

## Mathematical target and scope

If an attack A of fixed radius r is chosen before audit coins, any targeted prefix T of size t (including arbitrary independent randomness in T) followed by u uniform samples from its complement detects with certainty when T meets A, and otherwise detects with probability exactly 1-C(m-t-r,u)/C(m-t,u). Conditioning on T and averaging over arbitrary public-risk prefix distributions preserves this lower bound. An attacker with access to the realized full schedule may instead select r unchecked coordinates and evade. Formalize conditional counts and the finite mixture bound in MathCan without assuming cryptographic entropy or witness authenticity.

First commit only freezes registration. Follow with source generation code, exact controls, separate-process verifier replay, and path-scoped Python 3.12 CPU CI with 45-minute job cap. Preserve initial original raw fixture and outcome bytes and SHA256. Reproduce all outcomes with separate processes that share no NOR-26 implementation imports. No complete-target correctness or live-verifier security is claimed.
