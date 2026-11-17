# NOR-31 prospective registration: alias-collapsed, template-constrained minimax auditing

**Frozen 2026-11-17, before NOR-31 source generation, attack calibration or test scoring.** Extends merged NOR-30 #198 and MathCan #120. Amendments to the protocol must be dated and committed before new held-out data.

## Identification and fixed auditor family

NOR-30 trained 13/16 shuffled_prior and 3/16 uniform. The shuffled_prior policy uniformly permutes all output coordinates before reading a precomputed rank order; for any static ranking independent of the permutation, the first k selected indices are **uniform k-subsets**. That policy duplicates uniform sampling rather than adding genuine auditor diversity. NOR-31 therefore collapses this alias.

Six eligible, distinct k=16 auditor arms, in order: **uniform, static_prior, posterior, hybrid_25, hybrid_50, hybrid_75**. The historical shuffled_prior remains an attacker class and negative equivalence control but may receive no defender weight. The original exact RRC25 sixteen-template public generator/checker is unchanged. An authentic external exact coefficient witness and ideal independent audit sampling are explicit assumptions.

## Two pinned historical TRAIN cohorts; no new TEST information

Historical adversarial matrix: NOR-26 original 12-source 7×7 cross-play, source SHA256 **e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f**, 217954-byte original result SHA256 **a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a**. Preserve all seven attacker rows and project columns to the six eligible defenders. Each historical cell has 49152 decisions.

Historical structured-coverage cohort: NOR-30 original **12-source** TEST result, explicitly promoted to NOR-31 prior TRAIN evidence, public fixture SHA256 **9961afd81cac78b32fc5280fe54291f7db4dca0cf85cb6624560ddac420e1878**, 337563-byte outcome SHA256 **4189235f7cb5c14f3c7d1028abda264d893c7b5e5d2eda0c4c1de091f052b9c8**. Read original integer detections for eight independent radius-four attacks per source in each of tail, magnitude, lag and uniform supports. Compute exact weighted numerator per auditor using family weights **4,3,2,1**, denominator 12×4096×10.

Exhaustively search every six-class 16-integer-quantum mixture (sum weights=16) constrained by:
- uniform weight 2–12 quanta;
- historical weighted in-template detection ≥**0.18 absolute**, using exact integer comparisons;
- maximize minimum over seven NOR-26 attacker-row weighted integer detection counts, then maximize sum of seven row scores, then lexicographically smallest weight tuple.

Freeze training matrix, template integer numerators, unique historical source hashes, exact feasible candidate count, trained six-class weight tuple and original SHA-256 receipt in a **dedicated dependent TRAIN GitHub Actions job before producing any fresh TEST public fixture**. Report unconstrained and no-template-constrained diagnostics. Zero substitute outcomes if optimization or source selection fails.

## Prospective new TEST sources and policy-aware attacks

Generate **12 genuinely new public-only RRC25 inputs** from replicate indices (24,25) × D∈{512,1024} × the three original NOR-21 families. Source seed formula is original RRC25 SEED + **130031000** + 1000003×rep + 100009×degree_index + 10007×family_index. Query OUTPUT lengths m=769/1537. Select the first of at most 100 source attempts with a uniquely inferred public header, nonempty exact 16-template acceptance mask, and unanimous query prediction. Require all twelve complete-public hashes distinct and disjoint from BOTH historical TRAIN cohorts. The public fixture includes two observed degree-D input/output demos, query INPUT, header, seed, attempt, exact public mask and digest. It excludes planted program, future target, changes and witness answers. Freeze all public candidate outputs before reconstructing any sealed true future.

The eight policy-aware attacker classes are the original NOR-26 **seven** (including historical shuffled_prior) plus **mixed_best_response** directly optimized against the new frozen mixture. Each class has **eight** distinct-coordinate radius-four approximate greedy restarts per source, chosen against **256** calibration audit schedules before any of the **512** separately salted evaluation schedules. An attack restart may repeat its chosen set; report distinct count. The mixed best response minimizes a weighted sum of union-of-calendar exposure counts across the six eligible auditors. All attacks commit four positions before held-out audit coins. Also evaluate eight independent uniformly sampled blind attacks, and eight attacks on each predeclared support: tail, magnitude, lag, uniform, and off_template. Template controls use only existing NOR-24 public-defined supports and are not attacker-optimized.

For each source draw independent policy schedules using the unmodified NOR-25 k=16 definitions, with new phase streams: calibration **202611310091**; attacker tie/fresh seeds **202611310092**; evaluation **202611310093**; mixed-policy rollout **202611310094**; template attacks **202611310095**. Mix expected detection equals the integer-weighted average of the six arm outcomes divided by 16; separately report a realized independently drawn mixed-policy choice per trial. Same altered coordinates evaluated under shifts δ=+1 and +4294049777 must produce identical exact witness decisions. A fully schedule-informed attacker that picks four unchecked positions after revelation must evade every partial audit.

## Frozen five success gates

At k=16,r=4, the primary registered PASS requires ALL five gates:
1. Trained mixture against its own eight mix-aware calibrated attacks beats fixed hybrid_50 against **its own eight** calibrated attacks on **≥9/12** fresh TEST sources.
2. Equal-source mean of that paired self-response improvement ≥**+0.005 absolute** (0.5 percentage point).
3. Mixture's minimum expected detection across the eight policy-aware attacker classes exceeds hybrid_50's minimum across the same eight on **≥9/12** fresh sources.
4. TEST weighted in-template detection over the 4:3:2:1 support-family mixture ≥**0.17 absolute** across twelve sources, and per-source trained mix template coverage ≥**75%** of fixed hybrid_50 template coverage on **≥9/12** sources. This gate prevents the NOR-30 uniform-degeneracy outcome.
5. All exact original public-mask/native-query checks, true in-family future parity, source distinctness, honest-target acceptance, δ invariance, positive ideal source-conditional mixture blind floor, fully informed evasion and independent complete replay pass.

Retain a scientific FAIL if any gate misses. Report original uniform, hybrid_50 and all six auditor controls, source-level paired differences, 8×6 cross-play, uniform/off-template attacks, historical template-feasibility and expected-versus-realized mixture scores. Treat 12 public sources as the independent unit; eight attacker restarts and 512 challenges remain within-source correlated.

## Formal and security boundary

MathCan NOR-31 will kernel-prove that a uniformly permuted fixed ranking has exact uniform k-subset coverage, that collapsing duplicate distributions preserves finite-game expected payoffs, that a nonnegative mixture inherits a linear template-coverage lower bound, and that positive uniform/hybrid reserve mass gives an exact conditional blind floor when attacks are fixed BEFORE independent hidden audit coins. An attacker seeing the actual complete schedule may still evade.

External truth-witness authenticity, OS randomness, PRF assumptions, public pseudorandom-seed secrecy and unknown attack-distribution transfer are not proved by this finite synthetic experiment.
