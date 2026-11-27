# DISTILL-04: Direct Gemini 4 Neon Pilot Evaluation (`NEON_PILOT_01`)

> **Custody & Scope:** Reference-only public development evaluation (`training_admission: false`). Raw response archives and SQLite journals (`runs/neon-pilot/`) remain outside Git. All 55 packets were answered directly from the public packets without calling local solver tools or `exact_value()`, then validated and graded through the Distill `Journal` runner (`grade_instance` / `evaluate` / `replay`). Semantic review of Neon's prose derivations remains `pending` independent human review.

## 1. Execution Protocol & Side-by-Side Baseline Comparison

- **Examinee:** Gemini 4 Neon (direct public-packet pass, zero solver/oracle tool calls during solving)
- **Roster:** Exact 55-packet pilot roster (`seed=20261127`, `per_family=3`, `include_design=True`, `215` question slots: `30` procedural across `10` families x `3` difficulties + `25` standard design sections across all `25` registered gyms)
- **Plan ID:** `7196340a619f6cd1dbe11e08b548f5ce40463937263b682d2e1ed30dc258f627` (replayed `replayed: true`, zero cap/checker errors)
- **Expenditure:** `$0.00` Tinker API calls

| Metric (Same 55-Packet Pilot Roster) | Inkling-2-Mini Pilot (`4096 / 0.9`) | Inkling-2-Mini Scaled (`8192 / 0.6`)* | Gemini 4 Neon Direct Pilot |
| :--- | ---: | ---: | ---: |
| **Total Packets Finished** | 55 / 55 | 175 (`100` proc + `75` design) | **55 / 55** |
| **Format Valid (All)** | 12 / 55 (21.8%) | 168 / 175 (96.0%) | **55 / 55 (100.0%)** |
| **Format Valid (Procedural)** | 9 / 30 (30.0%) | 96 / 100 (96.0%) | **30 / 30 (100.0%)** |
| **Format Valid (Design)** | 3 / 25 (12.0%) | 72 / 75 (96.0%) | **25 / 25 (100.0%)** |
| **Truncated / Invalid JSON** | 42 / 1 | 3 / 4 | **0 / 0** |
| **Exact Procedural Pass (Strict)** | 9 / 30 (30.0%) | 90 / 100 (90.0%) | **30 / 30 (100.0%)** |
| **Exact Pass — `foundation`** | 5 / 10 | 35 / 40 | **10 / 10 (100.0%)** |
| **Exact Pass — `standard`** | 3 / 10 | 30 / 30 | **10 / 10 (100.0%)** |
| **Exact Pass — `frontier`** | 1 / 10 | 25 / 30 | **10 / 10 (100.0%)** |

*\*Scaled Inkling-2-Mini numbers come from `runs/inky-full` (`100/960` procedural) and `runs/inky-design-full` (`75/547` design) in [`INKY_FREE_01.md`](INKY_FREE_01.md).*

---

## 2. Key Technical Highlights (Compared to Inkling-2-Mini Findings)

1. **Exact Procedural Frontier Resolution (`30 / 30`):**
   - Solved `IRC-PG-01` frontier (`509130c23eb4`, `{fronts:6, command_ticks:4, switch_ticks:1, deadline:5}`) cleanly as `1` using per-action durations $T(k)=4k+(k-1)\le 5$ (correcting the action-budget misreading that caused Inkling-2-Mini's sole procedural math error).
   - Handled the double-JSON-string contract on `HG-07` foundation (`53e7dfcca414`, `"\"abstain\""`), with an explicit GF(2) kernel witness pair (`x_A=12`, `x_B=24`).
   - Computed degree-8 formal reciprocal `[1,-3,13,-50,199,-786,3106,-12280,48532]` (`ARG-01-ORACLE` frontier), 6-vertex 11-edge complement clique cover (`COE-01` frontier, $\chi=3, w^*=2$), 7-step mod-17 affine composition (`AUX-GYM` frontier), and 9-node strict-prefix DAG fixed point and depth-6 nilpotency bound (`MOLT-HYBRID-01` frontier) directly without solver tools.
2. **Structural `case.boundary` Obstructions:**
   - Replaced trivial input perturbations with structural obstructions across all 10 families: signed 4-bit overflow and piecewise parity breaking affine collapse (`RLT-ROLLOUT`), non-invertible multiplier / composite modulus collapsing bijectivity (`AUX-GYM`), non-unit constant term in $\mathbb{Z}[[q]]$ and zero-divisor ring $(\mathbb{Z}/6\mathbb{Z})[[q]]$ (`ARG-01-ORACLE`), integral-lifting failure of pointwise Lagrange identity and Minkowski indefinite form (`BSAR-02`), $d_H \ge 2$ Hamming packing and $\lceil\log_2 \chi\rceil$ quantization gap (`COE-01`), and $m$-parallel-path modular cancellation separating structural vs semantic invalidation (`MOLT-HYBRID-01`).
3. **First-Principles Coverage Across All 25 Design Sections (`125` Slots):**
   - `IRC-PG-01`: derived exact deadline-binding bound $k^* = 2$ on `pluto-new-toy-01` ($T(2)=8 \le 12 < 13 = T(3)$).
   - `QSM-03`: used the true **Hilbert projective metric** $d_H(u,v) = \log(\max_i (u_i/v_i) / \min_j (u_j/v_j))$ with Birkhoff contraction factor $\tanh(\Delta(T)/4)$ and emission-scaling isometry (avoiding the Hellinger-distance conflation).
   - `BPD-DT01`: derived non-vacuous orthogonal Pythagorean retention/leakage decomposition $\|P_{l+1}\delta_{l+1}\|^2 + \|(I-P_{l+1})\delta_{l+1}\|^2 = \|\delta_{l+1}\|^2$ with local Lipschitz bound.
   - `SCCG-04`: derived exact affine-sphere closed form $g_S^\top \alpha_0 + \|\Pi_{\ker P_S} g_S\|_2 \sqrt{\rho^2 - \|\alpha_0\|_2^2}$ and squared rational threshold verification in $\mathbb{Q}$.
   - `LRM-REV-03`: exhibited colliding $K_{\text{init}}=6$ histories proving the 6-bit separation between posterior sufficiency and online-update sufficiency.
   - `COE-01`: proved $w^*(C_7) = \lceil\log_2 3\rceil = 2$ with explicit 2-bit assignment `[00,01,00,01,00,01,10]`.

---

## 3. Complete 55-Packet Graded Manifest (`replayed: true`)

| Gym ID | Kind | Difficulty | `packet_id` (prefix) | `response_sha256` (prefix) | Exact Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `IRC-PG-01` | procedural | foundation | `1e241b65edb3` | `c796ffae73b2` | pass |
| `COE-01` | procedural | foundation | `43b4b8ce46ae` | `680ae99c3cfb` | pass |
| `RLT-ROLLOUT` | procedural | foundation | `95a63ea1b1ab` | `fcdbd4a8d9d9` | pass |
| `AUX-GYM` | procedural | foundation | `84e2b908bd8d` | `9023d243c4b6` | pass |
| `ARG-01-ORACLE` | procedural | foundation | `f7ca7a463b3d` | `8fdc603493f9` | pass |
| `BSAR-02` | procedural | foundation | `d53c6c1772fe` | `bde1c98a27fe` | pass |
| `MOLT-HYBRID-01` | procedural | foundation | `e079d298b29b` | `a352302a2b2d` | pass |
| `ASM-01` | procedural | foundation | `b30dfbe3ebdf` | `6f8c2dd55419` | pass |
| `BROKEN-VERIFIER` | procedural | foundation | `5d0f90dd25c2` | `6f1cc945b543` | pass |
| `HG-07` | procedural | foundation | `53e7dfcca414` | `0d7f0fdcc43e` | pass |
| `IRC-PG-01` | procedural | standard | `f4edad9f813a` | `c98f214dd8b5` | pass |
| `COE-01` | procedural | standard | `b439e49ac68a` | `06fb1d7cdaf8` | pass |
| `RLT-ROLLOUT` | procedural | standard | `38b369a4ee6b` | `38108f1e40c0` | pass |
| `AUX-GYM` | procedural | standard | `2d04b0f1615c` | `5b4ceeadb6f7` | pass |
| `ARG-01-ORACLE` | procedural | standard | `4bcc3cd85fbf` | `2cc9932e7ef6` | pass |
| `BSAR-02` | procedural | standard | `4f601759ac7b` | `0dada0a56453` | pass |
| `MOLT-HYBRID-01` | procedural | standard | `91d69f959630` | `ec94beff1614` | pass |
| `ASM-01` | procedural | standard | `8a2a9832b9d3` | `e7ca69e1d564` | pass |
| `BROKEN-VERIFIER` | procedural | standard | `47b98ec8c21e` | `55a0d9dcc3b9` | pass |
| `HG-07` | procedural | standard | `b12fe04f796b` | `3072a2a978f0` | pass |
| `IRC-PG-01` | procedural | frontier | `509130c23eb4` | `251c629fef0d` | pass |
| `COE-01` | procedural | frontier | `13a00fd9c2de` | `e29edf5972f1` | pass |
| `RLT-ROLLOUT` | procedural | frontier | `ffb100ad5525` | `1189a66b1937` | pass |
| `AUX-GYM` | procedural | frontier | `4a7f481c6c87` | `6cedab6ef757` | pass |
| `ARG-01-ORACLE` | procedural | frontier | `cecb0f7b4447` | `2ae6ac41c387` | pass |
| `BSAR-02` | procedural | frontier | `c2f55ddc03af` | `7b69aa4047b3` | pass |
| `MOLT-HYBRID-01` | procedural | frontier | `cc2caa0e7dcc` | `715c785daf51` | pass |
| `ASM-01` | procedural | frontier | `0163b8ba46e8` | `99cd75539218` | pass |
| `BROKEN-VERIFIER` | procedural | frontier | `160decb424bb` | `e142bd127f7c` | pass |
| `HG-07` | procedural | frontier | `b990ecbaec56` | `6ab5b54b2de3` | pass |
| `IRC-PG-01` | design | standard | `c1d71893359e` | `d87540ea2bbb` | not_checked |
| `COE-01` | design | standard | `f5bc247125a4` | `77c413e5b03e` | not_checked |
| `ARG-01` | design | standard | `8d2e2148001c` | `a1cd53ee3d81` | not_checked |
| `AEG-01` | design | standard | `013e64b278f4` | `306aeda78583` | not_checked |
| `SCM-23` | design | standard | `78c23531d984` | `2b850a4a605a` | not_checked |
| `RLT-ROLLOUT` | design | standard | `57a3dc7b13c6` | `7ccc2be61105` | not_checked |
| `DG-AGN-TINKER-01` | design | standard | `c4f55b3dac5b` | `85cbc77998df` | not_checked |
| `MOLT-GOBLIN-01` | design | standard | `8c18bebad330` | `54b7354f394b` | not_checked |
| `RLT-POSTFAILURE` | design | standard | `d963e0396f8d` | `db5beece2d12` | not_checked |
| `PPT-A-B` | design | standard | `5abc1b99d3e8` | `8f0679b15867` | not_checked |
| `LRM-REV-03` | design | standard | `1d2455302024` | `11538ddf5691` | not_checked |
| `AUX-GYM` | design | standard | `4aa206ce7d5b` | `8223049df043` | not_checked |
| `BPD-DT01` | design | standard | `42ae52f9d29f` | `ef9e1d33d9c8` | not_checked |
| `SCCG-04` | design | standard | `dfbed1560927` | `538c7c7bcb20` | not_checked |
| `ARG-01-ORACLE` | design | standard | `3721b94cce6a` | `5961ba139f31` | not_checked |
| `RH-PROOF` | design | standard | `ce7478c853c5` | `ea28ea875cdc` | not_checked |
| `BSAR-02` | design | standard | `2ccd75dc13e1` | `f55b85f45bcf` | not_checked |
| `LDT-RED01` | design | standard | `9ba3ed5ce960` | `e91c39ae2557` | not_checked |
| `MOLT-HYBRID-01` | design | standard | `118388d7f32b` | `8c3f81e205e7` | not_checked |
| `ASM-01` | design | standard | `072e95ef3ffc` | `dbf461ec5414` | not_checked |
| `QSM-03` | design | standard | `21a08477a9f5` | `9d2581516e24` | not_checked |
| `BROKEN-VERIFIER` | design | standard | `aa1304c1df57` | `fd86496347ae` | not_checked |
| `CSC-01` | design | standard | `3444b2f10e6f` | `e953bfa19289` | not_checked |
| `MORAL-HYSTERESIS` | design | standard | `aafdd92e1673` | `cbfb5d3c1871` | not_checked |
| `HG-07` | design | standard | `0bb3168f1e29` | `c89f27be896b` | not_checked |
