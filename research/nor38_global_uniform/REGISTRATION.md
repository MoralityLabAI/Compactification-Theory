# NOR-38 prospectively frozen global-uniform-reserve curvature auditing

**Original NOR-38 registration: 2026-11-17, before generating any NOR-38 source, attack, CAL, EVAL, or result data.** This file is the first scientific-protocol commit on this dedicated branch; the commit SHA is recorded as `FIRST_REGISTRATION_COMMIT` in the implementation after committing this file. The complete NOR-37 first source/grade and NOR-36 first receipts remain immutable. Any changed threshold, treatment, cohort, or attacker protocol requires a dated amendment and a NEW independent source cohort.

## Scientific question, prediction, and threat model

NOR-37 passed its five frozen gates: stochastic50_h50 improved deterministic curvature50 h50 self-play by +1.0071 percentage points and preserved 32.65% of the top-curvature advantage on 24 sources. Its own attacker self-play detection was 2.2725%, compared with 4.0278% for frozen NOR31 original_mix and 5.9631% for fully uniform16. The sourcewise worst of ten greedy attacker classes was 1.7883% stochastic50, 3.5119% original_mix, and 5.2694% uniform16. All are synthetic source-conditioned calibrated-attack results.

NOR-38 tests whether **fresh hidden full-output uniform exploration** can restore adversarial coverage while preserving high-curvature attack sensitivity. The attack fixes any four altered positions before each 16-coordinate audit. An attacker may learn a policy distribution and see 256 public CAL challenge schedules; independently generated 512 EVAL schedules are hidden until after committing the attack. Source-generating private truth and an authentic external coefficient witness are restored/assumed only after public source-only proposals, CAL attacks, and EVAL schedules are frozen.

The defense samples 16 DISTINCT output coordinates in every realized challenge, for future length m=769 or 1537. Existing NOR25 original six component mixture weights `[2,0,1,1,4,8]/16` remain unchanged within the nonuniform mixture branch. `atom_curvature50` is the original NOR35 deterministic per-atom curvature transform applied to the original NOR25 arm, preserving exactly the eight original atom inspection quotas **on that curvature branch**. A genuinely uniform branch samples from *all m output coordinates* and therefore is intentionally not subject to the old eight-atom quota restrictions.

Seven fixed k16 auditor distributions:

- `uniform16`: old fully uniform all-coordinate16.
- `original_mix`: old untransformed NOR31 six-arm mixture.
- `deterministic_mix`: original NOR35 curvature50 transformed six-arm mixture.
- `stochastic50_mix`: original NOR37 random within-atom curvature50 six-arm mixture.
- `atom_random_h50`: NOR35 atom-quota-preserving random hybrid50.
- `global7_mix`: **primary**, independent whole-output uniform16 branch with probability 7/16; deterministic_mix curvature branch with probability 9/16.
- `global8_mix`: **sensitivity**, uniform16 branch with probability 8/16; deterministic_mix otherwise.

The new whole-output branch is independent of all source-public curvature scores, CAL plans, target-output witness values, and attacker choices. The two global mixtures are distributions, not reweighted per-coordinate inspection counts. Attack calibration and cross-play integrate policy branch weights analytically using integer weights with common denominator 256: 16q for the uniform arm and `(16-q)*[2,0,1,1,4,8]` for curvature component arms, where q=7 or 8. Frozen NR31 mixture weights are scaled by 16 for equivalent policy-neutral comparison. Diagnostic realized branch draws must be independently salted and never used for calibration objective.

## Exact global guarantee and its limits

For a precommitted fixed r=4 changed-coordinate set A and an independently sampled uniform k=16 audit set Q over m outputs, the exact detection probability is
`u(m)=1-binomial(m-16,4)/binomial(m,4)`.
No k16 audit distribution can have worst-case detection greater than u(m) against unrestricted precommitted four-coordinate support choices, since averaging over every equally likely four-subset makes detection exactly u(m). Full-output uniform16 attains the upper bound for each fixed four-set. For each new mixture with whole-output probability `alpha=q/16`, `Pr[detect|A] >= alpha*u(m)` regardless of curvature branch. This bound requires a genuine independent hidden draw; publicly replayable SHA-derived random streams are scientific fixtures, not deployed cryptographic secrets. A schedule-informed adversary choosing unchecked positions after seeing the realized 16 coordinates still evades.

Exact source-conditional TOP-curvature detection is averaged over *all* 2+2 altered sets from original public top-48 support in/outside the frozen old risk union. Its estimate is a rational analytic average across 512 EVAL schedules and mixture weights, separate from eight sampled blind attacks.

## 48 fresh TEST sources, public freeze before private truth

Replicates 60..67 inclusive × observed degree 512 and 1024 × the exact original three NOR21 wide families, producing 48 independent full public source identities. Source seed uses original RRC25 `corpus.SEED + 130038000 + 1000003*replicate + 100009*degree_index + 10007*family_index`. The first admitted attempt out of ≤100 must pass the original 16-template native acceptance mask, yield a single consistent header, and produce unanimous full query continuation across all accepted native classes. Input query degree is 3D/2. Store only the two public demonstration pairs, full public query input, inferred header, original native acceptance mask, public SHA, replicate, family, seed and selected attempt. The first source-only fixture must exclude and pin all 180 original historical public SHA identities (the 156 archived through NOR-36 plus NOR-37's independently source-frozen 24), verified from immutable first archives. It must omit the planted program, private true future, witness values, attack locations and actual audit rolls. Generate the complete public-only fixture after registration, then freeze its bytes and SHA before restoring any private source program.

## New symmetric greedy attacker and controls

Each of the seven defender distributions receives eight independent source-seeded greedy restarts and 256 CAL schedules. For each of four greedy steps, search every one of the m output coordinates; greedily minimize the union of CAL trial IDs exposed by chosen four-coordinate attack positions, weighted by exact component mixture weights. Tie order is source/restart seeded. Attack classes are optimized separately against the global7 and global8 mixture distributions, including their public uniform probabilities; the global policies have their own greedy adversaries. Fix attacks before drawing the 512 independent EVAL plans. Cal/Eval use distinct NOR38 phase SHA streams and the 512 EVAL decisions are never used to pick attacks.

Five source-public blind control attack families with eight radius-four attacks each: top-curvature, bottom-curvature, original eight-atom-quota-matched, full-output uniform, and lower-calibration-exposure support chosen from the least-inspected public-coefficient positions of `deterministic_mix`. Freeze blind attacks before EVAL plans; the low-exposure support must be derived from CAL only. Record the complete **7x7 + 5 control-class** cross-play, source-level paired scores, restarts unique-set counts, exact TOP/BOTTOM and uniform controls, 512 realized diagnostic branch counts, and m-stratified results. All decisions check authentic native integer target coefficient witness for nonzero shifts +1 and +4294049777. A realized-schedule-informed four-coordinate attack is an explicit zero-detection control.

Keep per-source rates as independent primary observational units, with replicate-group summary as a sensitivity analysis, rather than treating the eight attack restarts or 512 schedules as independent sources. Report original full receipts and exact SHA. Learned greedy attacks are approximate, never global minimax certificates.

## Five frozen scientific gates (FAIL preserved)

**Gate 1: global-reserve robustness relative to deterministic curvature mixture.** `global7_mix` self-play against ITS OWN greedy attacker exceeds `deterministic_mix` self-play against ITS OWN greedy attacker on >=36 of 48 independent sources, with equal-source mean difference >=+0.010 (1 percentage point).

**Gate 2: competitive full-calibrated-class robustness.** On each source, compute the MINIMUM detection rate across all SEVEN independently calibrated greedy attack classes for each defender. Require `global7_mix` sourcewise minimum >= `original_mix` sourcewise minimum minus 0.010 on >=36/48 sources; mean global7-vs-original minimum difference >=-0.010; and mean global7 sourcewise minimum >=0.025. All are absolute detection-probability units.

**Gate 3: retain feature alignment.** Exact TOP-curvature conditional detection of `global7_mix` exceeds `atom_random_h50` on >=36/48 and by equal-source mean >=+0.020; additionally global7 mean gain must retain >=20% of the positive `deterministic_mix` minus `atom_random_h50` TOP gain. Record global8 sensitivity without determining PRIMARY from it.

**Gate 4: analytic global support and witness integrity.** Exact rational whole-output uniform16 detection equals `u(m)` for every source; original components and new mixtures have correct exact integer weights (denominator 256), separate 256 CAL and 512 EVAL seeds, k=16 distinct authentic witnesses, per-atom quota invariance on curvature branch, full-output uniform branch that can sample coordinates outside the old inspected-risk union, native original program acceptance, zero honest false rejection, identical decisions under both nonzero witness shifts, fully schedule-informed evasion, and exact theoretical mixture floor `alpha*u(m)` for every precommitted support (enforced by component nonnegativity). Finite EVAL-sample hit frequencies are NOT required to exceed the exact probability floor individually.

**Gate 5: replay and artifact integrity.** All 48 public source digests unique and excluded from 180 archived historical identifiers; first fixture generated after this registration; attacks fixed before EVAL; mandatory source-only fixture SHA and graded result SHA; NOR37 original 4,857,464-byte source SHA `f969925a159aa3e36d7061e2e65e0a6b9e6e0f82634e2a7e52b4dac874df6710` and 559,101-byte grade SHA `4852bd7de8012484db34404519553ae4d1e9f388203f2fdbf8b7516a534b684c` remain untouched; separately implemented full source/greedy/auditor/heldout cross-play replay matches the frozen first results exactly. Preserve original FAIL if any scientific gate misses.

## MathCan companion

Prove the finite uniform k-subset vs arbitrary r-subset maximin identity and global-uniform-mixture detection floor in Lean 4.35, in addition to the previously merged MathCan NOR37 per-atom quota theorem. Integrate this with the source-locked NOR38 result only after first-run evidence is frozen; math theorem and empirical policy-aware result have distinct premises.
