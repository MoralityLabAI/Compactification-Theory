# NOR-37 prospective registration: stochastic within-atom curvature with guaranteed exploration

**First NOR-37 commit, 2026-11-17, BEFORE generating any NOR-37 source, attack, CAL schedule, EVAL schedule or outcome.** Extends merged NOR-36 applied #210 and MathCan #130. The original NOR-36 source/grade SHA pins and scientific results remain immutable. Keep a failed NOR-37 result as FAIL. Any changed threshold, policy rule, source roster or attacker protocol requires a dated amendment and a fresh prospectively generated source cohort.

## Scientific question and frozen policies

NOR-35's PUBLIC curvature50 deterministic prefix increased source-aligned TOP-curvature detection by +16.73 percentage points, but NOR-36's matched-budget policy-aware greedy attacker reduced its self-play detection to 1.26%, below original hybrid_50's 3.05%. NOR-37 replaces each predictable targeted prefix with RANDOM within-top-window selection and introduces an independently randomized all-atom exploration branch. Attackers know the distribution and all public scores but do not know the realized EVAL audit coins before committing altered coordinates.

All policies inspect EXACTLY k=16 distinct coefficients of the true target continuation through an assumed authentic external target-dependent witness. Base schedules, historical source-risk prior, six component NOR25 policies and frozen NOR31 integer mixture weights [2,0,1,1,4,8]/16 remain unchanged. Each source/trial/component audit preserves the same eight original tail/top-magnitude/lag atom quotas t_j under every within-atom transformation.

Auditors evaluated in identical-budget 10-column calibrated cross-play:
1. original_h50, original NOR25 hybrid50;
2. atom_random_h50, original NOR35 full uniform redraw within each old atom;
3. deterministic_h50, original NOR35 curvature50 deterministic per-atom prefix;
4. stochastic25_h50, new within-atom randomized curvature with 1/4 uniform exploration;
5. stochastic50_h50, new with 1/2 uniform exploration, PRIMARY;
6. stochastic75_h50, new with 3/4 uniform exploration;
7. uniform16, unchanged original NOR25 k16 uniform;
8. original_mix, exact original six-component NOR31 mixture;
9. deterministic_mix, six-component original NOR35 curvature50 transformed mixture;
10. stochastic50_mix, the new 1/2-exploration transform of all six components using original NOR31 weights.

### Exact stochastic algorithm, all public-only

For each source, original NOR25 component arm, source/trial, and old atom j (j=0..7), let n_j be that atom's public size and t_j the original number of inspected positions, sum_j t_j =16. If t_j=0, inspect none. Compute public cyclic curvature score |y_(i-1)-2y_i+y_(i+1)| from the full proposed continuation y, ties by ascending index. For stochastic25/50/75 generate a fresh independent per-atom coin c∈{0,1,2,3} using SOURCE/TRIAL/ARM/ATOM/VARIANT and PHASE-separated SHA-derived random stream 202611370076. If c < q, where q=1,2,3 respectively, draw ALL t_j positions uniformly without replacement from atom j ("explore"). Otherwise:
- let deterministic targeted count h_j=ceil(t_j/2),
- window size L_j=min(n_j,max(2h_j,ceil(n_j/8))),
- select h_j positions UNIFORMLY WITHOUT REPLACEMENT from the first L_j highest-curvature ranked positions in atom j, and
- select t_j−h_j remaining positions UNIFORMLY WITHOUT REPLACEMENT from that atom excluding the selected h_j positions ("random reserve").
Every branch inspects exactly t_j distinct positions; the targeted choice is randomized over a public top window rather than a deterministic prefix. The independent per-atom exploration coin ensures a positive uniform branch probability for any atom carrying t_j>0, conditional on an attacker fixed before actual audit coins. Do not assert a positive floor for attacks restricted exclusively to atoms whose audit quota is zero in every policy trial.

Original, atom_random, deterministic_curvature50 variants are imported UNCHANGED from NOR35; phase domain separation for CAL vs EVAL also applies to the old random redraws.

## New 24-source independent TEST cohort

New original RRC25 PUBLIC-ONLY source identities: replicates 56,57,58,59 × observed demonstration degree 512/1024 × the three NOR21 wide input families. Source seed = original corpus.SEED + 130037000 + 1000003*replicate + 100009*degree_index + 10007*family_index. Query future length m=769 or 1537. Accept first source in ≤100 attempts with exactly one inferred original public header, nonempty exact native sixteen-template accepted-program mask and unanimous full hypothetical query continuation. Public fixture contains two observed D-degree input/output demos, public query INPUT, header, source seed, attempt, mask and full public SHA digest. **No planted program, private true future, attacked positions, target-witness answers or actual audit schedules.** Freeze all full public proposals before restoring the original native generating programs.

All 24 public source SHA identities distinct and disjoint from **156 archived previous original source identities** NOR26,30,31,32,33,34,35,36. NOR36 source fixture 4,224,502 bytes SHA256 881bf776e8e77eff6b8a5406789d07a4ced2eed9394a62b27d02d835c99355a4 and complete first grader 341,232 bytes SHA256 ffc9ebaf5b8e16c93e01c2c71cc387e4c8a455505f5af8cb3c9cad42ea9a44b5 are mandatory original SHA-locked inputs. Source exhaustion or disjointness failure aborts, retaining evidence.

## Registered attack calibration and held-out evaluation

For EACH of the ten frozen defender distributions, eight source-seeded four-position greedy attacks are learned using exactly 256 CAL schedules (stream 202611370071), with tie/restart seed 202611370072. Greedy objective at each step minimizes union of inspected CAL challenge trial IDs covering any selected attack coordinate. For mixed defenders, the objective uses the original frozen 16-quantum weighted SUM over component-policy detected CAL trial IDs. All attackers search the ENTIRE m-coordinate output domain with four sequential steps and identical eight-restart budgets. Attacker knows all policy distributions and source public curvature, including stochastic exploration fractions. Attacks are fixed BEFORE 512 distinct EVAL audit schedules from phase-separated stream 202611370073. Mixture policy realization draw stream 202611370075 is a diagnostic only.

Additional independent BLIND source-public control attacks (8 source-seeded fixed sets/family, stream 202611370074): TOP_CURVATURE 2 positions from top-48 within old union U plus 2 from top-48 outside; BOTTOM_CURVATURE 2+2 from low-48 inside/outside; EIGHT_ATOM_MATCHED redraw preserving the TOP attack's exact joint eight atom quota vector, and FULL_UNIFORM four-position attacks from all m. All control attacks freeze before EVAL plans. Score full 10×10 calibrated attack/auditor cross-play and four blind attack classes against every defender. Every plan uses exactly 16 true target-output coefficient queries and matches original NOR25 component's eight atom quotas per source/trial. Compare changed outputs with two nonzero shifts +1 and +4294049777; exact detection iff any checked coordinate differs. A fully realized-schedule-informed adversary may choose four unchecked coordinates and must evade as a stronger threat-model control.

## Exact analytic controls, no sample reuse

For each fresh source compute original NOR35 TOP_CURVATURE exact conditional detection under uniformly sampled r=2+2 positions in predeclared public top 48+48 support, using exact rational 1−C(48−t_U,2)C(48−t_V,2)/C(48,2)^2, averaged over all 512 EVAL schedules per policy. The full-uniform four-position attack exact expectation 1−C(m−16,4)/C(m,4) is policy invariant, as is the EIGHT_ATOM_MATCHED quota-conditioned expectation across any transformations sharing the original per-trial t_j vector. Record these exact rationals separately from eight finite blind-control attacks and their observed hits.

All primary self-play and worst-class cross-play comparisons average over 24 INDEPENDENT source identities (one source unit each), not over restarts/challenges as independent observations. Calibrated greedy attacks are APPROXIMATE learned best responses, not exact unrestricted minimax.

## Five frozen scientific PASS gates

Scientific PASS iff ALL five:
1. **Self-play robustness improvement:** stochastic50_h50 detection on attacks calibrated to stochastic50_h50 exceeds deterministic_h50 detection on attacks calibrated to deterministic_h50 on ≥18/24 new source identities, with equal-source mean improvement ≥ **+0.005 absolute** (0.5 pp).
2. **Worst-class robustness improvement:** for each source compute MINIMUM detection rate over all TEN calibrated attacker classes for stochastic50_h50 and for deterministic_h50 separately. The stochastic50_h50 minimum exceeds the deterministic_h50 minimum on ≥18/24 sources, and source-mean minimum improvement ≥ **+0.005 absolute**.
3. **Feature-aligned gain retained:** stochastic50_h50 TOP-curvature exact conditional detection minus atom_random_h50 is positive on ≥18/24 sources and source-mean ≥ **+0.020 absolute** (2.0 pp); further require mean stochastic gain ≥ **25%** of mean deterministic_h50 minus atom_random_h50 TOP gain, where the deterministic gain must be positive.
4. **Mixture robustness and transfer:** stochastic50_mix self-play against its OWN eight calibrated attacks beats deterministic_mix self-play on ≥18/24 source identities with source-mean improvement ≥ **+0.005**; and stochastic50_mix TOP-curvature exact detection exceeds original_mix TOP-curvature exact detection on ≥18/24 with source-mean improvement ≥ **+0.020**.
5. **Exact scientific integrity:** 24 original new public sources SHA-disjoint from 156 historical; original 16-template mask and native target parity; original per-atom t_j preserved by EVERY transformed plan with k=16; full 256 CAL versus 512 distinct held-out EVAL schedules; all attacker sets frozen before independent EVAL; real source target-witness checks identical under both nonzero shifts; honest original never rejected; complete schedule-informed evasion; uniform-attack exact invariance and eight-atom matched attack conditional equality; exact positive probability of an independent all-uniform exploration branch in any sampled atom with t_j>0; and a separate independently implemented replay of source programs, greedy attacks, all transformed plans, conditional exact controls and EVERY original world decision. Preserve original source/grade SHA bytes and registered PASS/FAIL.

Report all 10×10 cross-play, sensitivity at stochastic25/50/75, source-level paired rates, attacker uniqueness, TOP/BOTTOM control losses, pure uniform performance and lowest-risk-policy cases. Any failed gate remains scientific FAIL in source control, with original full raw receipts and reproducible independent analysis.

## Lean companion and security scope

For a fixed atom carrying n positions with t>0 audited, and a fixed blind attack changing r>0 positions before fresh audit coins, an explore-with-probability p>0 branch has conditional exact uniform attack detection 1−C(n−r,t)/C(n,t). Mixing this branch with ANY source-public curvature-targeting distribution gives overall detection ≥p·(1−C(n−r,t)/C(n,t)), even if the adversary knows the policy. Under independent per-atom choices, products of no-hit probabilities give an exact conditional finite distribution; no unconditional floor applies when all attacked atoms have zero inspection quota. For an equal-budget audit that preserves all old atom quotas, the expected detection under a fixed quota-conditioned UNIFORM-WITHIN-ATOM attack distribution is the same as prior NOR34, irrespective of curvature weighting. A realized-schedule-informed attacker can always select unchecked positions and obtain the same exact target-witness transcript.

Theorem hypotheses include an authenticated external true-target-output coefficient witness, an attacker fixed before independent hidden genuine audit randomization, and valid source-public family membership. SHA-based trial seeds are PUBLIC reproducibility fixtures, not deployed secret coins; greedy attack optimization is empirical, not a global minimax proof.
