# DISTILL Campaign Report 01: Neon Research Teacher Qualification, Live Free Inkling-2-Mini & Claude Fable 5.2 Distillation

**Date:** 2026-11-28  
**Repository:** `MoralityLabAI/Compactification-Theory` (PR #314, branch `research/distill-gym-v01-20261127`)  
**Custody & Admission Policy:** All provider outputs (Tinker free tier and Anthropic API) are archived locally outside Git under **`custody: reference_only`** and **`training_admission: false`**. Only scrubbed aggregate metrics, provenance manifests, and separately generated **`source_kind: synthetic_oracle_only`** curricula are committed.

---

## Executive Summary

Across all nine phases of the DISTILL research-teacher campaign, we audited the prior 215-slot Gemini 4 Neon pilot, ran matched 62-packet (124-slot) DISTILL-05 research examinations and 16 paired longitudinal research trajectories across **Gemini 4 Neon** and live zero-cost **Inkling-2-Mini** (`thinkingmachines/Inkling-2-Mini:peft:65536:free`, verified `$0.00`), triaged the 12 hardest cases against a frozen 30-case held-out split, executed a calibrated **Claude Fable 5.2** (`claude-fable-5-2`) paid research campaign under a single canonical `$200.00` transactional spend ledger (`$22.83` actual spend across 53 settled calls; `$167.17` active cap unspent; `$10.00` hard reserve untouched), subjected all outputs to adversarial multi-teacher criticism and executable Python verification, executed 6 surgical Lane D repairs (`6/6` defect+repair verification scripts passing), and built a new synthetic oracle-verified DataGoblin mechanism curriculum (`Distill/mechanism_curriculum.py`, `69` disjoint exact-rational rows).

### Headline Findings

1. **Exact answers mask derivation defects (Phase 1):** Independent audit of all `215/215` prior Neon pilot slots scored `1471/1720` (`85.5%`), catching **3 hard `mathematical_validity = 0` errors** (`QSM-03` wrong `tanh((log 9)/4) = 0.268` vs `0.500`; `SCM-23` degenerate `d=2` LayerNorm witness with `sigma=0`; `HG-07` false `[16,0]` row-span claim) and 10+ complexity/witness flaws beneath `30/30` exact procedural pass. Separate retrospective self-review (`1558/1720`, `90.6%`) caught the same 3 hard errors while exhibiting a `+0.298` optimism gap on `explanatory_value` (`1.977` vs `1.679`).
2. **Matched DISTILL-05 research exam (Phase 2):**
   - **Neon (`runs/neon-research09`, frozen `1591c6aa...`):** `62/62` format valid, `31/31` exact research decisions passed (using public prompts + standalone arithmetic verification, zero oracle access). Adversarial audit of all `124/124` slots exposed 3 `math=0` Section-2 counterexample/design errors and systemic gaps in `resource_accounting` (`1.032/2.00`) and `falsifiability` (`1.355/2.00`).
   - **Live Free Inky (`runs/research-05`, `$0.00` verified):** `62/62` finished (`125,576` input / `409,059` output tokens, `4,085.6s`), `29/62` (`46.8%`) format valid (`33/62` hit the `8,192` token ceiling at `effort=0.6`), `18/31` (`58.1%`) exact pass — and **`18/18` (`100%`) exact pass conditional on format-valid Section 1 responses**.
3. **Longitudinal 8-pair frontier benchmark (Phase 3):**
   - **Neon (`8/8` complete, `62` steps):** Total realized net utility **`208`** (mean `26.00`), **`0` conditional one-step optimal-continuation gap** across all 62 visited states, **`0` false recommendations** — matching the exact Bayes-optimal oracle (`208`) and beating `reuse_first` (`143`, mean `17.875`) by **`+65`** net utility (`7/8` worlds).
   - **Live Free Inky (`0/8` complete, `$0.00` verified):** `5` accepted steps across 8 arms before `8/8` terminal `ValueError` caused by `clean=False` at `program_runner.MAX_GEN = 4096` (`effort=0.6`).
4. **Claude Fable 5.2 operational & scientific calibration (Phases 4–7):**
   - **Always-on adaptive thinking discovery:** At default `effort="high"` and `max_tokens=8192`, all 3 preflight calls (`$1.025`) spent `8,192/8,192` output tokens on `thinking_tokens`, emitting `0` visible characters. Calibrating to `output_config.effort = "medium"` (or `"low"` for dense multi-defect repairs) with `output_cap = 24576` eliminated thinking starvation across all 13 expanded cross-repo research forms (`13/13 end_turn`) and all 24 official triage shortlist packets (`24/24 end_turn`, `$5.586` total).
   - **Official `Distill.research_paid` shortlist (`freeze_id: ad215ada...`):** Immutable verbatim score = **`7/24` strict format valid, `1/12` strict exact pass** (`b31f83a944`). Read-only post-hoc diagnostic revealed `0/17` markdown fences; `11/17` dropped one closing brace `}` between answer 1 and answer 2 (`},{"question_id":...` vs `}},{"question_id":...`) and `6/17` added 1 unrequested extra key (`check_note`, `assumptions`, `__drop__`). After read-only brace/key cleanup, **Fable solved `12/12` (`100%`) of the hardest frontier decisions purely in-head without tools**.
   - **Matched 7-packet semantic audit (14 strict-valid slots):** Fable scored **`185/204` (`90.7%`)** vs Neon **`175/204` (`85.8%`)** (`+4.9 pp`), leading on `falsifiability` (`24/24` vs `20/24`), `resource_accounting` (`11/14` vs `8/14`), and `assumptions/evidence_scope` (`100%`), while Neon led on `mathematical_validity` (`24/28` vs `22/28`) and `experimental_identification` (`10/12` vs `8/12`).
5. **Adversarial multi-teacher criticism & executable repair (Phase 8):**
   - All `12/12` directly parsed expanded Fable Section-18 Python scripts passed their own assertions on first run (13th passed after fixing 1 missing `}`), **yet adversarial out-of-box Python stress testing exposed 6 severe mathematical/experimental defects** (e.g., LayerNorm invariance rendering Fable's `SCM-23` pre-LN repair a `6.66e-16` no-op; `MoLT` GRU Lipschitz bound `>= 1` for every GRU; `DISTILL-10` 5.0-gap receipt-flag soundness counterexample; `LaCo` quadratic-vs-exponential drift misfit; `DataGoblin` Phase-9 exact-oracle promotion deadlock).
   - Live zero-cost **free Inkling-2-Mini** (`$0.00`, `effort=0.25, max_gen=8192`) produced `3/6` clean independent critiques and caught an additional arithmetic error in Fable's `laneC-datagoblin-transfer-03` Section 9 (`W=3, n=4` sign-test `p = 5/16`, not `1/16`).
   - Targeted **Lane D (`$20` cap)** calibrated v2 repairs (`$2.967` for 6 calls) fixed all 6 defects with **`6/6` runnable pure-Python defect+repair verification scripts passing**.

---

## Phase 1 — Audit of Prior Neon Pilot (`runs/neon-pilot`, 215 Slots)

Replayed `runs/neon-pilot` (`plan_id: 7196340a619f6cd1dbe11e08b548f5ce40463937263b682d2e1ed30dc258f627`, 55 packets, 215 answered slots, 30/30 exact procedural pass). Independent audit (`runs/neon-audit/`) and separate retrospective self-review (`runs/neon-self-review/`) both verified `all_planned_slots_reviewed: true` (`215/215`).

| Dimension (Max 430) | Independent Audit | Mean (/2.00) | Retrospective Self-Review | Mean (/2.00) | Self - Indep |
|---|---:|---:|---:|---:|---:|
| `mathematical_validity` | 394 / 430 | 1.833 | 405 / 430 | 1.884 | +0.051 |
| `assumptions_and_scope` | 367 / 430 | 1.707 | 373 / 430 | 1.735 | +0.028 |
| `explanatory_value` | 361 / 430 | 1.679 | 425 / 430 | 1.977 | **+0.298** |
| `checkability` | 349 / 430 | 1.623 | 355 / 430 | 1.651 | +0.028 |
| **Total (Max 1,720)** | **1,471 / 1,720** | **1.710 (85.5%)** | **1,558 / 1,720** | **1.812 (90.6%)** | **+0.101** |

- **Three verified `mv=0` defects (independently caught by both reviews):** (1) `QSM-03` (`21a08477`): `tanh((log 9)/4)` miscomputed as `2-sqrt(3)=0.268` instead of `0.500`, making the claimed Birkhoff bound unsound; (2) `SCM-23` (`78c23531`): `d=2` witness `h=(1,-1), delta=(0,2)` yields `h+delta=(1,1)` with `sigma=0` (singular; LayerNorm rotation requires `d>=3`); (3) `HG-07` foundation (`53e7dfcc`): false claim that `[16,0]` brings `q=13` into `row_span(A)` while derivation silently switches to `[4,0]`.
- **Highest-insight slots beyond exact answers:** `RH-PROOF` (`ce7478c8`, 40/40, minimal kernel constant cone `Dep(c)` vs `Imports(M)`), `BROKEN-VERIFIER` (`aa1304c1`, 39/40, NP/co-NP triple certificate), `MOLT-HYBRID-01` frontier (`cc2caa0e`, nilpotent `A^D=0` + 7-parallel-path mod-7 structural-vs-semantic invalidation separation), `COE-01` frontier (`13a00fd9`, `K_4` + complement matching exposing `ceil(log2 chi)` plateau), `BSAR-02` standard (`4f601759`, 2-point failure of pointwise-to-integral Lagrange lift), and `ASM-01` (`072e95ef`, necessary-and-sufficient condition for `VoM > 0`).

---

## Phase 2 — DISTILL-05 Research Examination (62 Packets / 124 Slots)

| Metric | Gemini 4 Neon (`runs/neon-research09`) | Live Free Inky (`runs/research-05`) |
|---|---|---|
| Exam / Plan ID | `534fe109...` / freeze `1591c6aa...` | `92b9f62a...` (`effort=0.6`, `max_gen=8192`) |
| Packets finished / format valid | **62 / 62** (100%) | 62 / **29** (46.8%; 33 truncated at 8,192) |
| Research decisions exact pass | **31 / 31** (100%)\* | **18 / 31** (58.1%; **18/18 = 100%** on valid S1) |
| Feasible & certificate-consistent | 31 / 31 | 18 / 31 (13 `exact_not_checked` due to truncation) |
| Provider tokens / wall time / cost | External operator agent (standalone Python check) | 125,576 in / 409,059 out / 4,085.6s / **`$0.00`** |

\*Neon answered strictly from `runs/neon-research09/public/*.txt` with zero access to `Distill/` oracle/solver files and strict Section 1 / Section 2 context isolation; standalone Python arithmetic verification was used on public statements before freezing (`1591c6aa...`).

**Neon Independent Adversarial Audit (`runs/neon-research-audit/`, `124/124` verified):**
- Section 1 (`case.solve` mean `1.936`, `research.interpret` mean `1.774`): caught 2 math slips in `research.interpret` (`489f9935`, `cffcaec0`) and weak `alternative_hypotheses` (`1.419/2.00`).
- Section 2 (`research.design` mean `1.661`, `research.obstruction` mean `1.885`): caught 3 hard `math=0` defects (`8a018ae7` broken maximin obstruction, `e160de6d` dominated non-interference witness, `acec3391` wrong Hamming distances & cost `21` vs `23`), plus systemic gaps in `resource_accounting` (`1.032/2.00`, 30/31 missing multi-arm/sham rollout totals) and `falsifiability` (`1.355/2.00`, 20/31 advance-only thresholds).

---

## Phase 3 — Longitudinal Research Pairs (`runs/teacher-pairs09`, 16 Arms)

Eight paired frontier worlds (`campaign_id: 2c467808...`: 4 `shared_control`, 4 `shifted_control`), evaluated in isolated per-arm contexts. Logical work, model tokens, and CPU wall time are kept strictly separate.

| Pair ID | Neon Net Util | Neon 1-Step Gap | Neon Credits / False Recs | `reuse_first` (Same World) | Exact Oracle (Same World) | Live Free Inky (`$0.00`) |
|---|---:|---:|---:|---:|---:|---|
| `shared_control-0` | **27** | 0 | 9 / 0 | 23 | 27 | 0 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shared_control-1` | **27** | 0 | 9 / 0 | 11 | 27 | 1 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shared_control-2` | **27** | 0 | 9 / 0 | 11 | 27 | 2 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shared_control-3` | **24** | 0 | 8 / 0 | 11 | 24 | 1 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shifted_control-0` | **24** | 0 | 8 / 0 | 11 | 24 | 0 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shifted_control-1` | **28** | 0 | 8 / 0 | 24 | 28 | 0 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shifted_control-2` | **27** | 0 | 9 / 0 | 23 | 27 | 1 acc, 1 invalid (`clean=False`, 4096 tok) |
| `shifted_control-3` | **24** | 0 | 8 / 0 | 29 | 24 | 0 acc, 1 invalid (`clean=False`, 4096 tok) |
| **Total / Mean** | **208 (26.00)** | **0** | **68 / 0** | **143 (17.875)** | **208 (26.00)** | **0/8 complete** (5 acc, 15,189 in / 48,336 out, 522.3s) |

- **Contextualization against DISTILL-08 `budget_rule`:** Per `Distill/TEACHERS.md`, these 8 paired worlds are descriptive realized trajectories and do not share the DISTILL-08 priced-work meter (`budget_rule` beats `reuse_first` by `+0.0663/+0.0657` priced utility via ~270 saved logical work units). Notably, on `shifted_control-1`, Neon discovered the exact nuisance-calibration shortcut (`anchor` `screen` [cost 1] + `direct_audit` [cost 3] deduces `nuisance = screen XOR anchor`, saving 1 credit vs `control` [cost 2] to fund both `transport` certifications and shifted-control `out_of_scope` decoding for `+28` net utility).

---

## Phases 4–7 — Anthropic Preflight, Triage, Canonical `$200` Ledger & Fable 5.2 Campaign

### Provider Verification & Terms
- **Verified model ID:** `claude-fable-5-2` (authenticated `GET /v1/models` and zero-cost `POST /v1/messages/count_tokens`).
- **Verified standard rates (Fast mode disabled):** `$8.00 / MTok` input, `$40.00 / MTok` output (billed output includes always-on adaptive `thinking_tokens`).
- **Output-use terms:** Anthropic Commercial Terms prohibit using Claude API outputs to train or distill competing AI models without explicit written permission. All Fable artifacts remain strictly `custody: reference_only`, `training_admission: false`.

### Canonical `$200.00` Global Spend Ledger (All 53 Settled Calls; 0 Pending; 0 Overruns)

| Lane | Authorized Cap | Calls | Input Tokens | Output Tokens (incl. Thinking) | Actual Billed (USD) | Remaining Cap (USD) |
|---|---:|---:|---:|---:|---:|---:|
| `original_research` (Preflight + 6 Expanded + 24 Triage Shortlist) | `$80.00` | 32 | 63,560 | 256,472 | **`$10.767352`** | `$69.232648` |
| `negative_results` (Preflight + 4 Expanded) | `$50.00` | 5 | 8,973 | 50,420 | **`$2.088584`** | `$47.911416` |
| `adversarial_design` (Preflight + 3 Expanded) | `$40.00` | 4 | 7,306 | 40,011 | **`$1.658888`** | `$38.341112` |
| `criticism_repair` (6 v1 + 6 Calibrated v2 Repairs) | `$20.00` | 12 | 23,180 | 203,214 | **`$8.313992`** | `$11.686008` |
| **Hard Reserve (Untouched)** | **`$10.00`** | 0 | 0 | 0 | **`$0.000000`** | **`$10.000000`** |
| **TOTAL GLOBAL LEDGER** | **`$200.00`** | **53** | **103,019** | **550,117** | **`$22.828816`** | **`$177.171184`** (`$167.17` active + `$10` reserve) |

### Phase 5 Teacher Triage & Official Paid Shortlist (`runs/fable-paid`)
- Frozen independent held-out roster (`runs/research-heldout`, seed `2026112991`, `exam_id: 2a69faac...`, 30 fresh cases / 60 eligible packets) created strictly prior to selection.
- `Distill.teacher_triage` (`triage_id: 206263b9...`) selected the 12 hardest development cases (24 packets, both sections included, zero overlap with holdout).
- `Distill.research_paid plan` (`plan_id: 78352e24...`, `input_cap=8192, output_cap=24576`) executed all 24 packets (`$5.586168`, 24/24 `end_turn`), froze verbatim (`freeze_id: ad215ada...`), graded (`7/24` strict valid, `1/12` strict exact pass; `12/12` diagnostic exact pass after single-brace/extra-key strip), and underwent full adversarial review on all 14 strict-valid slots (`185/204 = 90.7%` vs Neon `175/204 = 85.8%`).
- **Runner improvement recommendation:** Future Anthropic calls for `distill.response.v2` should pair `output_config.effort = "medium"` and `max_tokens >= 16384` with Anthropic Structured Outputs (`output_config.format` JSON schema) to eliminate the single-brace closure and extra-key failures without affecting mathematical reasoning.

---

## Phase 8 — Cross-Repo Expanded Campaign, Independent Criticism & Lane D Repairs

Thirteen source-pinned 18-section research forms were generated across `Compactification-Theory`, `MathCan`, `EvalMath`, `SelfModels`, `HybridTRMLDT`, `DataGoblin`, `LoopedTransformers`, `BPD`, `GPTStoryworld`, `jinn-or-beast-paper`, and `RSITopology` (13/13 `end_turn`; 12/12 directly parsed Section-18 Python scripts passed on first run; 13th passed after fixing 1 missing `}`).

Independent adversarial Python stress-testing (Neon) + live free Inky criticism (`$0.00`, 3/6 clean JSON critiques at `effort=0.25`) produced the following audit outcomes and 6 verified Lane D v2 repairs:

| Case ID | Domain | Audit Score | Verified Material Defect Caught | Lane D v2 Repair & Executable Check |
|---|---|---:|---|---|
| `laneA-mathcan-bsar16i-02` | MathCan BSAR-16I | **8/10** | Global DE + ray sweeps (`<=10^4`) + 2D RK4 confirmed all 3 PDE inequalities and tightened constants (`6->4, 15->7.86, 4->16/9`). | Verified sound (no repair needed). |
| `laneA-evalmath-rht07-04` | EvalMath RHT-07 | **7/10** | Prop 1 + 18-element Graver enumeration verified; Hitting-Set `z_u` cap typo (`|U|` vs `m`) noted. | Minor scope note recorded. |
| `laneA-distill-quotient-01` | DISTILL-10/11 | **6/10** | Real `Distill` 5.0 `V*` soundness counterexample when dropping receipt flags; XOR screens fix 0 bits; non-uniform `gcd`-marginals required. | **Repaired & PASS** (`$0.515`: 414 states -> 162 Q-classes, cancellation mass `0.6087`, zero soundness gap). |
| `laneA-molt-prefix-06` | MoLT / CCA-12 | **6/10** | GRU Lipschitz bound `>= 1` for every GRU; scalar product `2.5^t` explodes on `J=[[0.5,2],[0,0.5]]`. | **Repaired & PASS** (`$0.496`: old GRU bound `1.05` -> tube-local `0.3354`, sampled sup `0.2305`; weighted Lyapunov norm certifies non-normal decay). |
| `laneA-selfmodels-qsm03-05` | SelfModels QSM-03 | **6/10** | Missing `}`, 2x Brier mismatch, provably non-Markovian `n=3` `log(P_a/N_a)` quantization, Lloyd-Max beating uniform at `K=5`. | **Repaired & PASS** (`$0.634`: boundary-constrained Lloyd-Max hits `0.0` regret with lower Brier than uniform; `n=3` counterexample verified). |
| `laneB-bpd-scm23-03` | BPD / SCM-23 | **4/10** | Pre-LN affine rescaling is mathematically a `6.66e-16` no-op while breaking `ker(W_head)` (`4.6e-15 -> 2.28`); 4x Popoviciu scaling error; Inky also caught `chi^2` stat (`10.12` vs `11.67`). | **Repaired & PASS** (`$0.422`: LN-tangent / kernel-preserving intervention verified with `||W_head delta|| < 1e-12` and nonzero LN shift). |
| `laneB-looped-laco-crr-02` | LaCo / CRR | **4/10** | Static-deficit omission misidentified quadratic `T^2` linear drift as exponential `1.1^T`; halting `||Delta h_t||/||h_t||` decays as `O(1/t)` under divergence. | **Repaired & PASS** (`$0.394`: quadratic vs exponential model separated; step-ratio contraction certificate + leaky contractive update verified). |
| `laneC-datagoblin-transfer-03` | DataGoblin Phase 9 | **4/10** | Exact-oracle `PROMOTE` deadlock, conjunction power collapse (`5.7%` / `0.02%`), impossible `2x` microsecond CPU cap; Inky caught `W=3, n=4` sign-test `p=5/16` vs `1/16`. | **Repaired & PASS** (`$0.505`: headroom-normalized promotion + Holm-Bonferroni power `0.993` + separated `L/T/C` meters verified). |

---

## Phase 9 — DataGoblin Integration Preparation

1. **Versioned reference-only artifact inventory:** Committed at `Distill/reports/DISTILL_NEON_FABLE_INVENTORY.json` (`53` Anthropic provider receipts, all campaign/plan/freeze/triage IDs, local archive SHA-256 digests, review totals, and `custody: reference_only`, `training_admission: false`). Raw provider outputs remain strictly outside Git.
2. **Separately generated synthetic, oracle-verified curricula (zero provider text admitted):**
   - `runs/research-curriculum-v1` (`manifest_id: 4b4de3a7...`, `source_kind: synthetic_oracle_only`, `training_admission: false`).
   - `runs/program-curriculum-v1` (`manifest_id: 3ec52d88...`, 4 splits with exact rational baselines, `training_admission: false`).
   - **New module `Distill/mechanism_curriculum.py`** (`manifest_id: 37e24214...`, `69` disjoint exact-rational rows across `train [24]`, `calibration [9]`, `heldout_roots [18]`, `heldout_structural [18]` covering `receipt_quotient`, `belief_quantization`, and `overhead_regime_switch`; unit-tested in `Distill/tests/test_mechanism_curriculum.py`). On `heldout_structural`, `naive_brier_minimizer` incurs `131/3300` regret, `naive_posterior_merge` incurs `281/72` soundness regret, and on `heldout_roots` `always_dense_learned` incurs `1529/1200` regret vs `cheap_analytic_budget_switch` `57/80`.
3. **Proposed DataGoblin Transfer Experiment (incorporating Lane D v2 repairs):**
   - **Splits:** Strictly disjoint `train`, nested `val_tune` (for retuning analytic baselines), `val_headroom` (unbiased oracle-headroom estimation `H_s`), `heldout_roots` (same families, fresh seeds), and `heldout_structural` (unseen graph topologies, shifted nuisance couplings, non-binary budgets, and 3-state/asymmetric losses).
   - **Cheap strong baselines:** `stop`, `one_step`, `information_then_stop`, `reuse_first`, `budget_rule`, and budget-conditioned `cheap_analytic_switch`.
   - **Separated resource meters:** Logical search work `L`, model tokens `T`, and amortized wall CPU time `C` (never conflating neural ms with `if`-statement us; compare neural students on matched `L` and total priced utility `U - lambda*L`, with `C` reported as a separate engineering tier).
   - **Promotion / Kill gate (deadlock-free):** Stage 0 audits oracle headroom `H_s` on `val_headroom`; splits with `H_s < eps_s` enter a **non-inferiority tier** (avoiding exact-oracle deadlock), while headroom-positive splits evaluate dimensionless headroom capture fraction `Delta_s / H_s` with paired Wilcoxon / exact sign tests under Holm-Bonferroni correction calibrated to `>= 0.80` joint power (`0.993` verified at `p_win = 0.75`). Training admission remains `false` until the owning DataGoblin pipeline approves admission and clears these matched-resource held-out gates.
