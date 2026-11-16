# NOR-18: certified proof coverage on a common arithmetic-work budget

Protocol registered **16 November 2026 before fresh NOR-18 source generation or evaluation**. Extends [NOR-17 #173](https://github.com/MoralityLabAI/Compactification-Theory/pull/173), [NOR-16 #170](https://github.com/MoralityLabAI/Compactification-Theory/pull/170) and kernel-certified sound prefix elimination [MathCan NOR-17 #92](https://github.com/patrickdugan/MathCan/pull/92).

## Research question

In NOR-17, a depth-160 source-prefix screen recovered 28/192 risk-bounded certificates with four full public checks and zero false accepts. The prefix filter added **3,072 short candidate comparisons**, plus substantial exact arithmetic work. Because each reciprocal-series unary operator involves quadratic coefficient work while identity/negation/substitution are linear, counting four-check calls and sixteen cached-screen calls as interchangeable is misleading.

NOR-18 compares candidate verification under one auditable **coefficient-operation proxy** per case and per policy. It measures whether choosing a shallower or deeper public prefix increases certified future correctness after the screen itself and simulated public query continuations consume the same budget. It also includes a batched degree-192 full-public comparison that reuses the four unary transforms across the entire candidate family. That arm is explicitly billed for **16 full-degree public template comparisons**.

## Locked observation and source populations

- Source model: the original 1,000-parameter RRC25 GRU, unchanged, checkpoint SHA-256 \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`.
- Public data: two input/output demonstrations through degree 192 and query INPUT through degree 256. Withheld query OUTPUT and the planted program identity are seen only by the sealed native grader after every arm has selected or abstained.
- Frozen prefix degrees: **112, 128, 144, 160, 176**. The 192-degree *batch* arm computes all sixteen public acceptance decisions at full degree using shared public unary transforms. "None" is an unscreened certificate control.
- Cost multipliers: **4, 8, 16**. The per-case public full-check reference unit is \(\lceil (\sum_{j=0}^{15} C_{\mathrm{full},j})/16 \rceil\), computed from the public header alone. The total cap is \(\mathrm{multiplier}\times\mathrm{unit}\). This is an **arithmetic proxy** rather than a wall-clock or bit-complexity bound. All policies pay for their publicly computed query-output predictions; a plain original-GRU greedy reference does not need query simulations.
- Fixed policy family: original greedy ranking, no-prefix predictive-consensus certifier, predictive-consensus certifiers for each frozen prefix degree, and the cached full-depth batch mask. Additional degree-160 controls use the original GRU order and a seeded public-query-hash order, with the same screening and query simulation work.
- Fresh source seeds: \`corpus.SEED+93518000+1000003*rep+10007*family_index\` across **4 source-disjoint cohorts** of 48 late_source, 12 signed and 12 rational cases each. Primary: **192 complete-public-distinct late_source cases** sharing one original GRU-visible tensor. Secondary: 96 nonlate unique-GRU-input controls. The generator excludes NOR11–17 public source cases and previously evaluated nonlate model feature hashes.
- The cohort sizes, cost formula, budget multipliers, degree sweep and primary comparison are fixed here before the first held-out execution.

## Exact arithmetic-work proxy and accounting

For degree \(n\), charge the public unary operator \(\mathsf R\) (reciprocal) \(n(n+1)/2\) multiply-accumulate coefficient operations, and each of \(\mathsf I,\mathsf N,\mathsf S\) exactly \(n+1\) primitive coefficient operations. The latter are conservative O(n) proxies. Candidate combination work is the public output copy and delayed branch product-additions; a full verification additionally compares each output coefficient with the observed demonstration.

Write \(u_o(n)\) for the above per-operator cost and \(a_j(n)\) for the candidate's output construction plus equality comparisons. With two public demonstrations, one independently computed full candidate check has cost

\[
C_{\mathrm{full},j}(n)=2[u_{\mathrm{base}}(n)+u_{u_j}(n)+u_{v_j}(n)+a_j(n)].
\]

The **cached sixteen-candidate public screen**, which actually shares each of four unary evaluations per demonstration, has work

\[
C_{\mathrm{screen}}(n)=2[\sum_{o\in\{\mathrm I,\mathrm N,\mathrm R,\mathrm S\}}u_o(n)
+\sum_{j=0}^{15}a_j(n)].
\]

At full degree 192 this makes a *batched full mask*, with sixteen full candidate comparisons included in the screen's cost. Other prefix levels use the same cache algebra, but a surviving candidate still needs its separately charged full degree-192 check before the verifier returns a result. Constructing sixteen hypothetical candidate query outputs from the **public** degree-256 query input incurs a separate once-per-policy charge

\[
C_{\mathrm{query}}=\sum_o u_o(256)+\sum_j a^{\mathrm{build}}_j(256).
\]

No shared cross-arm caches are given free budget credit. A policy may run only if its entire query and/or screening setup fits the budget, and can perform each later full candidate check only when its charged work also fits. If it cannot afford a necessary step, it abstains. Original greedy controls are separately metered and may return a public-accepted result whose future is false; risk-controlled arms demand **one accepted full-public witness and predicted-output unanimity across every still-possible program**.

The native RRC25 public header discovery and one frozen-GRU ranking pass are common fixed costs and omitted from *all* arms' incremental budgets. The logged proxy excludes Python allocation, integer bit-length, memory and actual measured runtime; no hardware speedup is inferred from a smaller number. On these exact scalar-operation conventions, the cached unary part costs strictly less than sixteen independent candidate evaluations, since the shared four maps are evaluated once per demonstration. A MathCan companion proves the corresponding finite arithmetic bound.

## Predeclared outcomes and controls

The principal outcome is **certified future-correct cases under exactly the same total arithmetic-work cap**. All certificate outputs must be future-correct on the sealed grader under synthetic generator membership. Any false accept invalidates the experiment or the model-family premise. The exact remaining-program mask must include every full public accepted candidate, independently checked **after** every policy has frozen its choice.

The **preregistered primary hypothesis** is that depth 176 at the four-unit cap produces a positive paired net certified-coverage gain against **both** depth 160 and the unscreened certificate at that cap. Each of these two-sided exact paired sign p-values must be at most **0.025**, with no increase in false accepts. The score-ranked versus original-order and hash controls, deeper budget tiers, the full depth-192 cached batch, and other prefix depths are secondary. Held-out best-degree/frontier selection is descriptive and cannot count as an independent, preregistered win.

Report by family and cohort: certification counts, false accepts, abstentions, total coefficient-operation proxy work, setup/query/screen/full-charge breakdown, actual full candidate checks, batch full-public comparisons, and fraction of the nominal work cap consumed. Record a source-locked full JSON of all per-case input hashes, output prediction hashes and verification traces; charge an additional **sixteen full checks per case** for a post-selection integrity audit outside the policies' inference budgets.

## Interpretation and limitations

A shorter screen could be more work-efficient even when it rejects fewer impossible candidates. The cached full-depth batch is a real public-algorithm variant with a different cost structure, not an ordinary four-check policy. Equal-budget claims are restricted to the frozen arithmetic proxy and its explicit source-operation assumptions. A change to the underlying source generator, candidate family, verifier implementation, or bigint cost model requires a new registration. No positive conditional mutual information or general Lean-proof soundness result follows from this synthetic benchmark.

## Reproduction

\`\`\`sh
python -m unittest research.nor18_costed_prefix.test_experiment -v
python research/nor18_costed_prefix/experiment.py --check \
  --repetitions 4 --late-per-repetition 48 \
  --signed-per-repetition 12 --rational-per-repetition 12 \
  --out /tmp/nor18_full.json
\`\`\`
