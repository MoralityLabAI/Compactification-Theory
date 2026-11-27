# DISTILL-04: Live Free Inkling-2-Mini Examination Campaign (`INKY_FREE_01`)

> **Custody & Scope:** Reference-only public development evaluation (`training_admission: false`). Raw live response archives, SQLite journals, and credentials remain strictly outside Git. This report contains only aggregate metrics, cryptographic receipt digests, and concise analytical summaries of representative successes and failures, pending verification of publication and training-use rights.

## 1. Environment, Model Identifier, and Confirmed Expenditure

| Field | Value |
| :--- | :--- |
| **Requested Sampling Model** | `thinkingmachines/Inkling-2-Mini:peft:65536:free` |
| **Provider-Confirmed Sampler Base Model** | `thinkingmachines/Inkling-2-Mini:peft:65536:free` (`model_path: null`, verified via `SamplingClient.get_base_model()` and `RestClient.get_sampler()`) |
| **Provider Model Weight Revision** | `null` (not exposed by Tinker API) |
| **Tokenizer / Renderer Metadata Model** | `thinkingmachines/Inkling-2-Mini` (metadata only; never sampled) |
| **Renderer** | `tml_v0` (`TmlV0Renderer`, `H200kBaseChatTokenizer`, stop token `[200006]`) |
| **SDK / Cookbook / Renderer Versions** | `tinker 0.40.1`, `tinker-cookbook 0.7.1`, `tml-renderers 0.5.0`, `torch 2.15.0+cpu` |
| **Orchestration Host** | CPU-only, Python 3.12.3 (Linux x86_64 via WSL2; required because `tml-renderers>=0.4.2` publishes `manylinux`/`macosx` binary wheels only) |
| **Context / Decoding Caps** | Default context `12,288`, hard cap `16,384`, temperature `0.3`, `num_samples=1`; pilot `max_generation=4096, effort=0.9`; scaled batches `max_generation=8192, effort=0.6` |
| **Paid Model Fallback / Training** | Disabled (zero paid sampling calls, zero Fable calls, zero training or checkpoint calls) |
| **Confirmed Free-Tier Expenditure** | **`$0.00`** (`AdminClient.get_billing_usage()` confirmed `estimated_cost_usd=0.0` and `effective_rate_usd_per_million_tokens=0.0` on all `thinkingmachines/Inkling-2-Mini:peft:65536:free` prefill and sample billing events; an unrelated prior paid SFT session at `09:00 UTC` preceded this campaign) |

### Preflight SDK Compatibility Findings

1. **Linux/macOS Wheel Constraint (`tml-renderers`):** On Windows, PyPI resolves only the empty `tml-renderers 0.0.1` stub and downgrades `tinker-cookbook` to `0.5.9`, causing `ModuleNotFoundError` when loading `thinkingmachines/Inkling-2-Mini`. Running CPU-only orchestration in WSL2 (`x86_64`) cleanly installed `tinker 0.40.1`, `tinker-cookbook 0.7.1`, and `tml-renderers 0.5.0`.
2. **Thinking-Block Leakage on Truncation (`visible_text` fix):** When generation hits `max_tokens`, `TmlV0Renderer.parse_response` returns `ParseTermination.MALFORMED` with a raw decoded string containing `<|message_model|><|content_thinking|>...` rather than structured content blocks. `Distill/inky_free.py:visible_text()` was patched after Request #1 to strip `<|content_thinking|>` fallback strings (retaining only visible text after `<|content_text|>`, or `""` if thinking never terminated) and `TinkerFree.__init__()` now explicitly asserts `client.get_base_model() == FREE_MODEL`.

---

## 2. Campaign Footprint, Token Throughput, and Latency

Three live campaigns were executed sequentially via the persistent SQLite journal and independently replayed (`replayed: true`, zero pending requests, zero cap overruns, zero checker errors):

| Campaign | Plan ID (prefix) | Config (`max_gen` / `effort`) | Finished / Planned | Question Slots (Finished / Valid) | Format Valid | Truncated | Invalid JSON/Schema | Input Tokens | Output Tokens | Sampling Time (s) | Mean Latency (s) | Effective Output tok/s | Effective Total tok/s |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pilot (`inky-pilot`)** | `fcfa44e7afb5` | `4096` / `0.9` | 55 / 55 | 215 / 42 | 12 (21.8%) | 42 (76.4%) | 1 (1.8%) | 58,632 | 216,160 | 2,134.62 | 38.81 | 101.26 | 128.73 |
| **Scaled Procedural (`inky-full`)** | `009602940584` | `8192` / `0.6` | 100 / 960 | 300 / 288 | 96 (96.0%) | 3 (3.0%) | 1 (1.0%) | 108,578 | 305,247 | 3,346.13 | 33.46 | 91.22 | 123.67 |
| **Full Design (`inky-design-full`)** | `47feb4965156` | `8192` / `0.6` | 75 / 547 | 375 / 360 | 72 (96.0%) | 0 (0.0%) | 3 (4.0%) | 77,790 | 230,840 | 3,074.40 | 40.99 | 75.08 | 100.39 |
| **Total Live Footprint** | — | — | **230** | **890 / 690** | **180 (78.3%)** | **45 (19.6%)** | **5 (2.2%)** | **245,000** | **752,247** | **8,555.15** | **37.20** | **87.93** | **116.57** |

- **Tokens per question slot:**
  - Pilot (`4096 / 0.9`): `272.7` input / `1,005.4` output tokens per question (`3,930.2` mean output/request; median `4,096`).
  - Scaled procedural (`8192 / 0.6`): `361.9` input / `1,017.5` output tokens per question (`3,052.5` mean output/request; median `2,752`).
  - Full design (`8192 / 0.6`): `207.4` input / `615.6` output tokens per question (`3,077.9` mean output/request; median `2,921`).
- **Response-level abstention:** `0 / 690` valid question slots used `"status": "abstain"` (one genuine abstention attempt occurred in `DG-AGN-TINKER-01` standard design, which failed schema validation by emitting explanation fields instead of `{"reason": ...}`).
- **Truncation diagnosis in the `4096 / 0.9` pilot:** All 55 rendered pilot prompts were only `1,011–1,185` tokens (`1,066` mean), leaving `~11,200` tokens unused inside the `12,288` context budget. Under `effort=0.9`, `25 / 42` truncated pilot responses (`16` procedural, `9` design) exhausted all `4,096` tokens inside `<|content_thinking|>` without emitting visible text, and `17 / 42` (`4` procedural, `13` design) truncated mid-JSON. Moving to `max_generation=8192, effort=0.6` (strictly within the `12,288` default context) increased format validity from **`21.8%` to `96.0%`** while lowering mean output tokens per request.

---

## 3. Exact Correctness by Gym and Difficulty

### 3.1 Phase 2 Pilot (`runs/inky-pilot`, `4096 / 0.9`, 55 requests)

| Subset | Finished | Format Valid | Truncated | Invalid | Exact Pass | Exact Fail | Exact Pass (of Valid) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Procedural — `foundation` | 10 | 5 | 4 | 1 | 5 | 0 | 5 / 5 (100%) |
| Procedural — `standard` | 10 | 3 | 7 | 0 | 3 | 0 | 3 / 3 (100%) |
| Procedural — `frontier` | 10 | 1 | 9 | 0 | 1 | 0 | 1 / 1 (100%) |
| **Procedural Total** | **30** | **9 (30.0%)** | **20 (66.7%)** | **1 (3.3%)** | **9 (30.0%)** | **0** | **9 / 9 (100%)** |
| **Design Sections (`standard`, 25 gyms)** | **25** | **3 (12.0%)** | **22 (88.0%)** | **0** | — | — | — |

The 3 valid pilot design sections were `IRC-PG-01`, `ARG-01`, and `AUX-GYM`. The sole pilot invalid JSON (`ASM-01` foundation) computed the exact right mathematical answer (`{"action":0,"probability":"9/13"}`) but flattened the `fields` keys directly into each answer object.

### 3.2 Phase 4 Scaled Procedural Batch (`runs/inky-full`, `8192 / 0.6`, 100/960 cases)

| Procedural Family (Gym) | Finished | Format Valid | Truncated | Invalid | Exact Pass (Strict) | Exact Fail (Strict) | Quoting-Only `exact_fail` (`"abstain"` vs `"\"abstain\""`) | Genuine Math Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ARG-01-ORACLE` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `ASM-01` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `AUX-GYM` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `BSAR-02` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `MOLT-HYBRID-01` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `RLT-ROLLOUT` | 10 | 10 | 0 | 0 | 10 (100%) | 0 | 0 | 0 |
| `IRC-PG-01` | 10 | 10 | 0 | 0 | 9 (90%) | 1 | 0 | **1** |
| `COE-01` | 10 | 8 | 1 | 1 | 8 (80%) | 0 | 0 | 0 |
| `BROKEN-VERIFIER` | 10 | 10 | 0 | 0 | 7 (70%) | 3 | **3** | 0 |
| `HG-07` | 10 | 8 | 2 | 0 | 6 (60%) | 2 | **2** | 0 |
| **Total (10 families)** | **100** | **96 (96%)** | **3 (3%)** | **1 (1%)** | **90 (90.0%)** | **6 (6.0%)** | **5** | **1** |

| Difficulty (`inky-full`) | Finished | Format Valid | Truncated | Invalid | Exact Pass (Strict) | Exact Fail (Strict) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `foundation` | 40 | 39 (97.5%) | 0 | 1 | 35 (87.5%) | 4 (all 4 `"abstain"` quoting) |
| `standard` | 30 | 30 (100.0%) | 0 | 0 | 30 (100.0%) | 0 |
| `frontier` | 30 | 27 (90.0%) | 3 | 0 | 25 (83.3%) | 2 (1 `"abstain"` quoting, 1 `IRC-PG-01` math error) |

### 3.3 Phase 4 Full Design Batch (`runs/inky-design-full`, `8192 / 0.6`, 75/547 packets)

All **25 registered gyms** were examined across all **3 difficulties** (`25 foundation`, `25 standard`, `25 frontier`, 5 questions/packet = `375` slots):
- `foundation`: `24 / 25` valid (`1` invalid: `MOLT-HYBRID-01` flattened `fields`).
- `standard`: `23 / 25` valid (`2` invalid: `DG-AGN-TINKER-01` invalid abstain payload; `AUX-GYM` unescaped LaTeX `\escape` in JSON).
- `frontier`: **`25 / 25` (`100%`) valid**, `0` truncations.
- **22 of 25 gyms achieved `3 / 3` valid design packets**; `AUX-GYM`, `DG-AGN-TINKER-01`, and `MOLT-HYBRID-01` achieved `2 / 3`.

---

## 4. Independent Semantic Audit (Separate from Exact Scores)

All 12 valid pilot packets (`42` question slots) plus a 6-gym frontier design sample from `runs/inky-design-full` (`30` question slots) — **18 packets, 72 question slots total** — were audited independently using the 4-dimension `Distill.review` rubric (`0–2` per dimension) and cryptographically bound via `Distill.review.validate_review()` (`training_admission: false`).

### 4.1 Aggregate Semantic-Review Means (Out of `2.000`)

| Cohort | Audited Packets | Audited Slots | Mathematical Validity | Checkability | Assumptions & Scope | Explanatory Value |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pilot Procedural (`4096/0.9`)** | 9 | 27 | **1.963** (`53/54`) | **2.000** (`54/54`) | **2.000** (`54/54`) | **1.852** (`50/54`) |
| **Pilot Standard Design (`4096/0.9`)** | 3 | 15 | **1.467** (`22/30`) | **1.600** (`24/30`) | **1.733** (`26/30`) | **1.533** (`23/30`) |
| **Scaled Frontier Design (`8192/0.6`)** | 6 | 30 | **1.633** (`49/60`) | **1.833** (`55/60`) | **1.867** (`56/60`) | **1.767** (`53/60`) |
| **All Audited Slots Combined** | **18** | **72** | **1.722** (`124/144`) | **1.847** (`133/144`) | **1.889** (`136/144`) | **1.750** (`126/144`) |

### 4.2 Per-Packet Cryptographically Validated Review Receipts

| Campaign | Gym | Kind | Difficulty | Slots | Math | Check | Assump | Expl | `response_sha256` (prefix) | `review_sha256` (prefix) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| Pilot | `COE-01` | procedural | foundation | 3 | 6/6 | 6/6 | 6/6 | 6/6 | `f5c956a7e95e` | `49154b6f34fb` |
| Pilot | `RLT-ROLLOUT` | procedural | foundation | 3 | 6/6 | 6/6 | 6/6 | 5/6 | `b8a5da7d6991` | `b90f1f960fd7` |
| Pilot | `AUX-GYM` | procedural | foundation | 3 | 6/6 | 6/6 | 6/6 | 6/6 | `8ec09416f668` | `53dbabf1f6ab` |
| Pilot | `ARG-01-ORACLE` | procedural | foundation | 3 | 6/6 | 6/6 | 6/6 | 6/6 | `9c5a007710b8` | `3b0c3e1ec9e8` |
| Pilot | `BROKEN-VERIFIER` | procedural | foundation | 3 | 6/6 | 6/6 | 6/6 | 6/6 | `bf9de3df22f4` | `4e1507f27316` |
| Pilot | `RLT-ROLLOUT` | procedural | standard | 3 | 5/6 | 6/6 | 6/6 | 5/6 | `df3fa3ee00d3` | `8e28cde93a15` |
| Pilot | `AUX-GYM` | procedural | standard | 3 | 6/6 | 6/6 | 6/6 | 5/6 | `27adfe1a6cd3` | `ee0c8c7aede9` |
| Pilot | `BROKEN-VERIFIER` | procedural | standard | 3 | 6/6 | 6/6 | 6/6 | 6/6 | `79ce6ff2c2f7` | `d155a4475ce7` |
| Pilot | `RLT-ROLLOUT` | procedural | frontier | 3 | 6/6 | 6/6 | 6/6 | 5/6 | `2619af0b5541` | `0a8fdd189439` |
| Pilot | `IRC-PG-01` | design | standard | 5 | 3/10 | 5/10 | 6/10 | 6/10 | `67573847e4ab` | `71d6b965dced` |
| Pilot | `ARG-01` | design | standard | 5 | 9/10 | 9/10 | 10/10 | 7/10 | `d55bfff0029a` | `7c26e76179bc` |
| Pilot | `AUX-GYM` | design | standard | 5 | 10/10 | 10/10 | 10/10 | 10/10 | `06b7b9bbc190` | `aabf31aa1d2b` |
| Design-Full | `IRC-PG-01` | design | frontier | 5 | 3/10 | 7/10 | 8/10 | 6/10 | `021df2abf5b0` | `ffcefb782238` |
| Design-Full | `SCM-23` | design | frontier | 5 | 10/10 | 10/10 | 10/10 | 10/10 | `15b7c822b590` | `7e14472a68a2` |
| Design-Full | `BPD-DT01` | design | frontier | 5 | 8/10 | 8/10 | 9/10 | 9/10 | `cbba3fa7c022` | `f248746c4514` |
| Design-Full | `QSM-03` | design | frontier | 5 | 8/10 | 10/10 | 9/10 | 9/10 | `3124633b0dc6` | `f4c1c62ce184` |
| Design-Full | `CSC-01` | design | frontier | 5 | 10/10 | 10/10 | 10/10 | 10/10 | `a8fb4d42e107` | `4f3ed4b7980d` |
| Design-Full | `MORAL-HYSTERESIS` | design | frontier | 5 | 10/10 | 10/10 | 10/10 | 9/10 | `bae95b1dee8e` | `4db263f77083` |

---

## 5. Qualitative Synthesis: Representative Successes, Failures, and Capability Boundaries

*(Analytical summaries only; verbatim model archives remain private outside Git.)*

### Representative Strong Solutions
1. **First-Principles Nonlinear Geometry (`SCM-23`, frontier design, `15b7c822b590`, `40/40`):** Correctly derived the first-order LayerNorm transport Taylor expansion around $h$, exhibited the exact constant-shift null-space obstruction $\delta = c\mathbf{1}$ where $\mathrm{LN}(h+\delta)=\mathrm{LN}(h)$ identically, counted degrees of freedom ($\le 3$ inside rank-4 frame $F$ intersected with the norm-matching sphere $2 h_r\cdot\delta + \|\delta\|_2^2=0$), and formulated the explicit 4-variable frame-coordinate quadratic for norm-matched transport.
2. **Spectral Control Sensitivity (`CSC-01`, frontier design, `a8fb4d42e107`, `40/40`):** Derived $F(p)=U\operatorname{diag}((1+\alpha\lambda_i)^{-p/2})U^\top$, constructed a checkable 2-node Laplacian witness with exact trace $1+(1+2\alpha a)^{-p/2}$ and exact relative weight-perturbation sensitivity $(1 + 2\alpha\delta/(1+2\alpha a))^{-p/2} - 1$, and properly distinguished finite-precision Lanczos/LOBPCG residuals from interval certificates.
3. **Algebraic & Combinatorial Obstructions (`ARG-01-ORACLE`, `COE-01`, `BROKEN-VERIFIER` procedural):** Proved both directions of $w=\lceil\log_2\chi(G)\rceil$ (`COE-01`), exhibited ring-localization failure $F(q)=2+q$ lacking an inverse in $\mathbb{Z}[[q]]$ while invertible in $\mathbb{Q}[[q]]$ (`ARG-01-ORACLE`), and produced paired consistent GF(2) witness assignments proving ambiguity when removing an anchor (`BROKEN-VERIFIER`).
4. **Experimental Design & Epistemic Calibration (`MORAL-HYSTERESIS` frontier `bae95b1dee8e`; `AUX-GYM` standard `06b7b9bbc190`):** Accurately captured why `AUX-GYM` tests trace-consistency against majority/final-answer shortcuts, and why `MORAL-HYSTERESIS` requires multiset-hashing checks, paired endpoints, family-level splits, blinded bilingual human rating reliability gates, and separation of representation divergence from causal proof. Consistently avoided claiming unexecuted code runs.

### Representative Failures and Diagnosed Capability Gaps
1. **Systematic Semantic Misreading of `IRC-PG-01` Parameters (`67573847e4ab`, `021df2abf5b0`, and procedural frontier `509130c23e`):** Across standard design, frontier design, AND the single genuine mathematical error in the 100-case procedural batch (`509130c23e`: `{fronts:6, command_ticks:4, switch_ticks:1, deadline:5}` -> answered `2` instead of `1`), Inky Mini interpreted `command_ticks` and `switch_ticks` as *action-count budgets* (`k <= command_ticks`, `k-1 <= switch_ticks`, total `2k-1 <= deadline`) rather than *per-action tick durations* (`k*c + (k-1)*s <= deadline`). This shows vulnerability to ambiguous parameter naming whenever units are not explicitly stated in the packet.
2. **Metric-Name Conflation (`QSM-03`, frontier design, `3124633b0dc6`):** Saw `d_H` / Hilbert-distance in the context, proved a valid Le Cam inequality $\mathrm{TV}(p,q)\le\sqrt{1-\mathrm{BC}^2}\le\sqrt{2(1-\mathrm{BC})}$, but substituted the **Hellinger distance** formula while labeling it "Hilbert-distance", missing QSM-03's actual Hilbert projective metric and Birkhoff contraction theorem.
3. **Vacuous Algebra & Unstated Isometry (`BPD-DT01`, frontier design, `cbba3fa7c022`):** Wrote $\|P_{l+1}x_{l+1} - P_{l+1}F_l(x_l)\|\le\varepsilon_l$ even though $x_{l+1}=F_l(x_l)$ makes the left-hand side identically zero, and wrote a lower bound on $\|P_{l+1}x_{l+1}\|$ implicitly assuming $F_l$ is norm-preserving.
4. **Hard Frontier Combinatorial/GF(2) Search Exhaustion (`inky-full` 3 truncations at `8,192` tokens):** The only 3 requests that still exhausted `8,192` tokens at `effort=0.6` were `COE-01` frontier (`n=7`, 20 edges, $\chi=5$, $w=3$) and two `HG-07` frontier (`n=9`, 5-equation GF(2) span tests). Without tool use, text-only backtracking over dense chromatic coloring and 9-bit GF(2) elimination remains a genuine reasoning-depth ceiling.
5. **Weak `case.boundary` Witnesses on Simple Procedural Families:** On `RLT-ROLLOUT` (3/3 pilot packets) and `AUX-GYM` standard, Inky Mini satisfied `case.boundary` via trivial input-value perturbations (changing an operand or `x_0`) rather than structural failure modes, plus a minor prose counting slip ("both `if_even_add` instructions" when three existed in `df3fa3ee00d3`).
6. **JSON-Schema & Quoting Failure Modes (Accounts for 5/6 `exact_fail` and 5/5 `format_invalid`):**
   - Double-JSON-string trap: `5/5` `"abstain"` cases in `inky-full` (`2 HG-07`, `3 BROKEN-VERIFIER`) had 100% valid derivations but emitted `"answer": "abstain"` instead of `"answer": "\"abstain\""`.
   - Inverse typing trap: `COE-01` foundation (`ac9ade5156`) emitted `"answer": 2` (JSON int) instead of `"answer": "2"`.
   - Flattened `fields` object (`ASM-01` pilot, `MOLT-HYBRID-01` design), wrong abstain payload keys (`DG-AGN-TINKER-01` design), and unescaped LaTeX backslashes (`AUX-GYM` standard design).

---

## 6. Recommended Improvements to Examination Forms

1. **Adopt `max_generation=8192, effort=0.6` as the default reasoning-model evaluation profile** (within `CONTEXT=12288`): cut truncation from `76.4%` to `1.7%` across `175` scaled requests while preserving high derivation quality. Keep `4096 / 0.9` only as a stress-test ablation.
2. **Eliminate the double-JSON-string trap on `case.solve.answer`:** In `render_prompt()` (or a future `distill.response.v2`), either permit native JSON values for `answer` or include an explicit two-line formatting example showing `"answer": "2"` and `"answer": "\"abstain\""`, alongside explicit reminders for `"status": "abstain"` (`{"reason": "..."}`) and doubled LaTeX backslashes (`\\sum`, `\\equiv`).
3. **Disambiguate parameter units in adapter contexts:** Explicitly define parameter semantics in adapter `context` (e.g., `IRC-PG-01`: `command_ticks` = ticks consumed per command, `switch_ticks` = ticks consumed per front switch; `QSM-03`: Hilbert projective metric vs Hellinger distance; `ARG-01`: concrete 3-symbol grammar) so design packets test mathematical reasoning rather than lexical guessing.
4. **Tighten `case.boundary` prompt requirements:** Explicitly instruct that input-value perturbations that leave the algorithm valid do not satisfy `case.boundary`; require a structural obstruction, ring/domain breakdown, or identifiability collapse.
5. **Surface `section` and `difficulty` telemetry in `Journal.summary()`** and keep `visible_text()` defensive against renderer fallback control tokens (implemented in this PR).

---

## 7. Prioritized Examination Syllabus for Future Fable Reference Campaign

Because Inkling-2-Mini already saturates `6/10` procedural families (`100%` exact on `ARG-01-ORACLE`, `ASM-01`, `AUX-GYM`, `BSAR-02`, `MOLT-HYBRID-01`, `RLT-ROLLOUT`) and achieves `96%` valid JSON at `8192 / 0.6`, a paid **Fable** reference campaign should concentrate budget where Inkling-2-Mini showed genuine mathematical, algorithmic, or experimental-design ceilings rather than re-running saturated foundation items:

| Priority | Target Scope | Rationale from Inky Mini Baseline | Recommended Budget Allocation |
| :--- | :--- | :--- | :--- |
| **P0 — Hard Frontier Procedural** | `HG-07` (`frontier`, $n=9$ GF(2) span), `COE-01` (`frontier`, dense $\chi(G)$ coloring), `BROKEN-VERIFIER` (`frontier` multi-component parity), `IRC-PG-01` (`frontier` deadline bounds) | Sole families with `8,192`-token exhaustion (`3` frontier cases) or conceptual/math errors (`IRC-PG-01`). Highest signal per token for comparing reasoning depth. | ~35% of Fable budget (e.g., 24–32 frontier cases/family across these 4 families) |
| **P0 — Frontier Design Sections (Math-Heavy & Adversarial Gyms)** | `SCM-23`, `CSC-01`, `QSM-03`, `BPD-DT01`, `RH-PROOF`, `PPT-A-B`, `LDT-RED01`, `SCCG-04`, `AEG-01` (`frontier`, full sections including `advanced.obstruction` and `advanced.redesign`) | Exposes subtle proof defects (vacuous `BPD-DT01` algebra, `QSM-03` Hilbert-vs-Hellinger conflation) and tests whether Fable can construct non-trivial obstructions and decisive follow-up experiments rather than templates. | ~40% of Fable budget (complete frontier sections for these 9 gyms) |
| **P1 — Research-Taste & Experimental-Control Sections** | `core.controls`, `core.causal`, `core.compactification`, `core.research_taste`, `core.calibration` across `MORAL-HYSTERESIS`, `LRM-REV-03`, `RLT-POSTFAILURE`, `DG-AGN-TINKER-01`, `MOLT-GOBLIN-01` (`standard` & `frontier`) | Tests falsifiable experimental design, sham/matched-compute controls, leakage boundaries, and honest calibration under incomplete specifications. | ~20% of Fable budget (sections 2–5 of full design exams, which were beyond section 1 in this run) |
| **P2 — Calibration Anchor Slice** | Small stratified anchor (`2` cases/family x `6` saturated families + `3` foundation design sections) | Confirms cross-model calibration and reproduces the `100%` baseline without wasting paid budget on easy arithmetic/modular traces. | ~5% of Fable budget |
