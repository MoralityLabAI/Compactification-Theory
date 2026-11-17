# NOR-34 preregistration: explain curvature residual with eight historical-risk atoms

**Frozen 2026-11-17 before any NOR-34 source fixture, attack, audit schedule, result, or training. This is the first NOR-34 experiment commit.** Extends merged NOR-33 #203 and MathCan #125. No post-data changes to source roster, success thresholds, attacks, or family definitions. Later changes require dated prospective amendments, and the original PASS or FAIL is preserved.

## Hypothesis and frozen model

NOR-33 reported +3.116 percentage points of hybrid_50's curvature-versus-matched-random advantage over the NOR31 mix at two altered output coefficients within the public old-risk union U=tail96 ∪ magnitude96 ∪ lag96. NOR-34 asks whether matching the *full joint three-bit membership* of each altered coordinate (bit1 tail, bit2 magnitude, bit4 lag, atoms 0–7) explains ≥70% of that remaining source-level effect.

Freeze the original NOR25 k=16 exact audit schedules for six policy arms (uniform, static_prior, posterior, hybrid_25, hybrid_50, hybrid_75), and original NOR31 mixture quanta [2,0,1,1,4,8]/16. Keep the original NOR24 prior [295,246,153,78]/772 and exact 16-template RRC25 public verifier unchanged. An honest external exact true-target coefficient witness is explicitly assumed.

## 24 new sealed sources

Generate source-distinct RRC25 public examples: replicates 44,45,46,47 × degrees D=512/1024 × three original NOR21 wide input families. Source seed = original corpus SEED + 130034000 + 1000003*replicate + 100009*degree_index + 10007*family_index. Query output length m=769/1537. First case among ≤100 attempts with exactly one inferred public header, nonempty exact native public 16-template acceptance mask, and unanimous full candidate query continuation. Source exhaustion aborts. Demand all 24 complete-public SHA identities distinct and disjoint from **84 previous original** NOR26/30/31/32/33 inputs. The NOR33 source-only fixture and complete grade are pinned at original SHA256 16f3b397aea17adbd1462666142a13e933c8da67a1286e1e6418d32b22090db8 and 88d37a957f7d3be42b76213b351df921a08556654a90a2ed21df1b66788d6a62. The new public fixture includes two observed degree-D demonstrations, query INPUT, header, seed, selected attempt, native mask and public digest. It includes **no planted private program, future target, altered positions, or target witness values**. Freeze all full public predictions before private original program reconstruction.

## Matched attack distributions, fixed before independent audit challenges

- From public candidate prediction construct historical tail96, top-magnitude96 and lag96, U and its complement V. Each coordinate belongs to exactly one 3-bit atom j∈{0..7}. Empty atom categories permitted.
- Curvature rank: descending absolute cyclic second difference of PUBLIC predicted coefficients, ties by smaller index. Take top **48** rank positions in U and **48** in V as a fixed 96-position curvature support C. Abort if each block lacks 48.
- For each new source generate **16 paired attacks** before audit schedules using deterministic public SHA streams: CURVATURE: choose 2 positions uniformly within C∩U and 2 within C∩V (stream 202611340071). FINE MATCH: for this attack's exact 8-atom quota vector q_j, redraw q_j from the ENTIRE source atom j, independently without replacement (stream 202611340072). COARSE MATCH: redraw 2 from full U and 2 from full V (stream 202611340073). Each attack changes exactly four coefficients, each pair shares 2+2 union membership and FINE preserves all eight q_j exactly.
- Evaluate each of the 3×16 attacks on 512 NEW original NOR25 k=16 distinct-coordinate audit schedules per source/arm, stream 202611340074. Independently sample a realized mixed policy from source-bound stream 202611340075 as a control. The mixture PRIMARY is the exact conditional weighted expected detection of six arms. Both nonzero target shifts +1 and +4294049777 must agree; directly check at least four chosen plans per source/arm/attack arm with authentic true-target equality. Separately construct an informed four-coordinate change after exposing the realized schedule and confirm zero detection. All attacks and plans freeze before private target restoration.

## Exact analytical estimands, on the same frozen 512 schedules

Use exact integer binomials and Fraction; no finite attack draw affects the analytic primary:
- CURVATURE law: uniformly choose 2 positions from the fixed 48 curvature-support U block and 2 from the 48 outside. Per-plan miss = C(48−t_CU,2)/C(48,2) * C(48−t_CV,2)/C(48,2).
- COARSE law: choose 2 from entire U and 2 from entire V. Miss = C(|U|−t_U,2)/C(|U|,2) * C(|V|−t_V,2)/C(|V|,2).
- FINE law: first choose 2+2 curvature positions as CURVATURE, retain their eight atom quotas q_j, then independently redraw q_j uniformly from the entire same atom j. Per-plan miss is the expectation over q of PRODUCT_j C(n_j−t_j,q_j)/C(n_j,q_j), where n_j is category population and t_j inspected count. Compute the exact expectation by enumerating unordered pairs of category labels independently in U and V. For equal labels j, probability weight C(s_j,2)/C(48,2) and no-hit C(n_j−t_j,2)/C(n_j,2). For two distinct labels j,k use s_j*s_k/C(48,2) and no-hit (n_j−t_j)/n_j*(n_k−t_k)/n_k. Multiply U and V block expectations. Empty atoms skip; invalid positive quotas abort.
- For source and conditional law X define H_X = analytic detect(hybrid_50) − analytic detect(frozen mix). Coarse residual R_C=H_CURVATURE−H_COARSE; fine residual R_F=H_CURVATURE−H_FINE. Report all per-source exact means and the sampled paired empirical equivalents separately. The curvature-minus-coarse and curvature-minus-fine residuals are *contrasts of attack distributions conditioned on identical public sources and audit schedules*, not evidence of causal effects in unknown real adversaries.

## Precommitted scientific gates, source unit n=24

PASS iff **ALL FIVE**:
1. R_C>0 on ≥18/24 source pairs, source-mean R_C ≥ **+0.015 absolute** (1.5 pp).
2. |R_F|<|R_C| on ≥18/24 and **|mean R_F| ≤0.30*|mean R_C|**, with nonzero mean R_C. This is ≥70% mean residual attenuation after eight-category matching.
3. **|mean R_F| ≤0.005 absolute** (0.5 pp).
4. Empirical 16-attack source-mean curvature-minus-fine and curvature-minus-coarse hybrid-advantage residuals agree with their analytic counterparts to within **0.010 absolute** each; zero legitimate-witness false rejection and exact both-shift invariance.
5. 24 new SHA-distinct sources disjoint from 84 historical; public native 16-mask and full target parity; exact 48+48 curvature supports, 2+2 altered coordinates per attack, exact 8-pattern quota match, correct 2+2 coarse match; positive independent uniform-reserve mixture floor, schedule-informed 0% detection, and a **separate-process full independent replay** of all 24×3×16×6×512 decisions, analytic counts and originally frozen source/grade SHA receipts.

Preserve the first original scientific FAIL with all raw evidence if any gate fails. Report the historical union, all eight category population/support/inspection counts, 16 attack quota vectors, pure uniform and hybrid50 rates, every per-source residual and all theoretical-vs-empirical deviations. All repeated attacks/schedules belong to their 24 source units; never count them as independent sources.

## Formal companion and assumptions

MathCan proves an eight-block extension of NOR33: for a fixed attack category quota vector q, the number of missed distinct radius-four attacks is PRODUCT_j C(n_j−t_j,q_j), out of PRODUCT_j C(n_j,q_j). Equal atom populations and inspected-count vectors imply identical conditional counts for any quota vector; the equality survives any finite mixture over quota vectors. The source-independent uniform-attack and fully schedule-informed evasion theorems remain applicable.

External witness authenticity, hidden private audit coin independence, public seeded PRNG security, and arbitrary prover soundness are NOT established. Deterministic SHA-derived Python streams are reproducibility tools rather than deployment secrets.
