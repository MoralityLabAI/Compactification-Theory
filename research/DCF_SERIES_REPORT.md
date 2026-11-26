# The DCF series (DCF-01 to DCF-16): what it establishes

2026-11-25. Synthetic CPU studies, no paid compute. The motivating premise was a claimed matrix-multiplication exponent of 9/4. It is treated conditionally throughout, and nothing here depends on or tests it. A cheaper dense primitive lowers the cost of every dense step; the question here is how much general dense computation structured inference makes unnecessary in the first place.

Receipt status: DCF-01–12 and DCF-14A–C report results whose raw receipts live in conversation archives, not on this branch. DCF-13, DCF-13D, DCF-14 (`dcf14_width_routing`), DCF-15 and DCF-16 commit source, protocols, hash locks, raw receipts and tests.

## 1. Dense computation

- **Exact structure removes dense work outright when it exists.** Query-directed component inference ignores disconnected evidence exactly (DCF-13 T2, kernel-checked in MathCan); its median speedup over full enumeration is 13.7× in DCF-13A (K = 8, 12). The DCF-13D cost hybrid's median is 153× with K = 16 included. Elimination replaces `2^n` by `2^{w+1}`. A 256-variable chain costs a few thousand operations.
- **Its cost is known before you pay it.** For a fixed elimination order the work count is a function of the scopes alone (4,608/4,608 exact matches in DCF-13D). Budgeted abstention therefore never wastes partial work.
- **Dense factorised recurrence (DCF-01/02) is cheaper but not exact.** Low-rank recurrence matched dense numerics in DCF-01 and improved short horizons in DCF-02, but continuous latent drift erased the gain at L16. The exact gains in this series come from discrete structure (quotients, components, widths), not from cheaper dense algebra.

## 2. Structured reasoning

- **Width is the right currency for elimination, not for reasoning in general (DCF-14).**
  - Min-fill width predicts held-out elimination cost with R² .86, against negative R² for state dimension and component size.
  - It explains only 36% of the variance in the best exact cost.
  - It misprices algebraic structure: GF(2) decides noise-free parity systems in 90 operations, where elimination at width 7 needs 4,523.
  - It does not predict approximation error or decision value.
- **Model uncertainty reconnects what structure disconnected (DCF-13D).** A sensor law shared across components couples them through its parameter, doubling to tripling the relevant variable set. Treating that parameter exactly (Bernstein-form BMA) beats plug-in estimation. Certifying decisions over the shared parameter, rather than per factor, raises safe coverage from 1.65% to 39.8% with zero negative-utility actions.
- **Higher-order dependencies need higher-order detectors (DCF-07), and repartitioning loses information (DCF-08).** Merging and splitting units has a measurable KL price, and the router must pay it knowingly.

## 3. Recurrence

- **Exact finite-state quotients give unbounded exact execution (DCF-03/04)**, when they can be found by symbolic merging and active queries. Passive neural discovery found them far less often.
- **Structured beliefs compress state without loss where independence holds (DCF-05/06, DCF-09).** Context-conditioned beliefs help under uncertain context.
- **Recurrent depth as a supervised target (DCF-16)** is defined by the verified algorithm: elimination steps, or BP iterations. It is available as a training signal; no model was trained on it here.

## 4. Adaptive specialization

Across DCF-10/11/12, 13C, 13D-A, 14 and 15, adaptive routing selects the right operator but repeatedly fails to beat a strong fixed or symbolic baseline on cost:
- DCF-10: joint gate 2/24.
- DCF-11: 0/24.
- DCF-12: 7/120 even fully compiled.
- DCF-13C: 6/24 preliminary, 7/24 reproduced.
- DCF-13D: the cost-estimated hybrid is exact but passes the latency gate in only 4/36; routing bookkeeping exceeds the saving.
- DCF-14: the learned selectors lose to exact-else-mini-bucket in held-out expected utility (MLP 17 wins vs 86 losses). Structural feature extraction costs 9.9 ms per decision in Python.

What does survive is symbolic adaptivity with exactness: exact when the symbolic cost fits the budget, a principled approximation otherwise, and certified abstention. Learned routing has not earned promotion anywhere in the series, and not only because of its cost: the DCF-14 and DCF-15 learned routers lose on expected utility before any routing cost is charged.

## 5. Batching economics

- **Batching rewards homogeneity.** In DCF-15 the exact kernel's per-query cost falls 3.7× from batch 1 to 256 under a fixed policy (1.80 to 0.48 ms). The learned router launches three distinct kernels per step at B ≥ 32, so its cost falls only 2.3×. DCF-11/12 found the same thing for collective scheduling. Adaptive specialisation and accelerator batching conflict unless routing decisions are coarse (per batch or per slow epoch) rather than per query.
- **Slow, structural adaptivity coexists with batching.** DCF-15's slow-timescale unit merges are made by a fixed structural rule every 8 steps and keep a single execution path per step. They cut discarded correlation 36% and raise local-answer utility 41% in every held-out condition. Per-query learned routing lost 0/6. The privileged regret oracle also lost on total cost, because knowing regret costs 7.3 ms per query.
- **Coarse routing removes the cost problem but not the signal problem (DCF-19).** Choosing one operator per batch per slow epoch costs 0.005 ms per query, 100–140× less than DCF-15's per-query routers, and keeps one kernel launch per step. On the mean it is the best non-oracle policy on the DCF-15 mesh (.1080 vs fixed BP .1073; the epoch oracle reaches .1089). It still fails its per-condition gate (1 win, 5 losses), because its structural statistic does not separate the epochs where exact inference pays. DCF-19 also found that DCF-15's offline tuning charged batched launch overhead per query, which biased the symbolic router against exact. DCF-19B replicated the result on fresh seeds (.1130 vs .1123 mean; R1 1 win, 5 losses) and found why routing cannot win much on this mesh: exact changes BP's decision on only 1.6% of queries, and decision-margin signals are mostly false alarms (6% precision).
- **Router cost is binding for symbolic and oracle routers; learned routers also lose on accuracy.** The router's own execution decides the outcome for the DCF-13D hybrid (bookkeeping), for DCF-15's symbolic router against fixed BP (equal utility, 0.5 ms routing), and for the privileged DCF-15 oracle (.1190 before routing, .0460 after). The learned routers in DCF-14 and DCF-15 lose on expected utility with no cost charged, so compiling them alone would not pass their gates.

## 6. Data and evaluation

- DCF-16 turns the verified machinery into 920 training and evaluation records, with family- and size-disjoint held-out splits enforced by hash and by the Data Goblin loader. Each record carries candidate representations marked sufficient or not, operators with exact costs, receipts, verified posteriors and regret. Hard negatives (117/200 held-out) isolate cases where local evidence looks adequate but is not: a unary-evidence shortcut scores −.053 on them, the symbolic router +.54.
- The EvalMath held-out baseline that any compact recurrent model must beat is symbolic, with verified EU .352 at selective risk .047 (oracle .446). The label space shifts on held-out (GF(2) and particle labels are unseen in training). This is a deliberate test of whether a model learns *when* a representation suffices rather than which label is common.

## Promotion criteria, series-wide

| Criterion | Met? | Where |
|---|---|---|
| Better verified utility at matched cost | Yes, for exact BMA over plug-in at the same polynomial representation; yes, for exact-else-MBE over always-BP at lower cost | DCF-13D C; DCF-14 |
| Lower cost at matched verified utility | Yes for exact component/elimination vs enumeration; no for adaptive vs strong fixed routing at the condition level | DCF-13A–D; DCF-10–14 |
| Constructive exactness guarantee with materially reduced cost | Yes: component restriction, symbolic elimination cost, credal certificate (zero negative-utility actions with 24× the coverage of pointwise bounds) | DCF-13, MathCan `DCFDecision.lean` |
| Learned router beats strongest symbolic after charging costs | No | DCF-14, DCF-15 |
| Coarse (regime) router beats strongest fixed policy per condition | No (best on the mean only) | DCF-19 |

## What is not established

- No general matrix-exponent conclusion.
- No accelerator, TPU or energy measurement.
- No real-language-model result.
- No claim that these synthetic factor-graph families represent intelligence-relevant workloads.
- Python timings are unequal-optimisation comparisons wherever an implementation differs (Numba vs NumPy).
- Result JSON files written on Windows contain CRLF line endings. Re-running a `report` stage on a non-Windows host rewrites them with LF, so `SHA256SUMS` fails although the values are identical; check hashes before running report stages.

## Where the work lives

| Piece | Location |
|---|---|
| DCF-13/13D experiments, receipts, theory | `research/dcf13_decision_compactification` |
| DCF-14 width routing and shared inference library (`fg.py`) | `research/dcf14_width_routing` (pilots 14A–C: `research/dcf14_structural_width`) |
| DCF-15 mesh routing | `research/dcf15_ldt_mesh_routing` |
| DCF-16 curriculum data | `research/dcf16_decision_curriculum` |
| DCF-19 regime routing | `research/dcf19_regime_routing` |
| Kernel-checked lemmas | [MathCan `MathCan/DCFDecision.lean`](https://github.com/patrickdugan/MathCan/blob/main/MathCan/DCFDecision.lean) |
| LDT mesh units and J router | [HybridTRMLDT `research_gym/ldt_mesh`](https://github.com/MoralityLabAI/HybridTRMLDT/tree/master/research_gym/ldt_mesh) |
| Dataset schema, loader, supervision views | [DataGoblin `schemas/dcf16_decision_trace.schema.json`, `src/datagoblin/datasets/dcf16.py`](https://github.com/MoralityLabAI/DataGoblin/blob/master/src/datagoblin/datasets/dcf16.py) |
| Compute-aware decision evaluation | [EvalMath `src/spiral_ln/evaluation_dcf16_decision.py`](https://github.com/MoralityLabAI/EvalMath/blob/main/src/spiral_ln/evaluation_dcf16_decision.py) |
