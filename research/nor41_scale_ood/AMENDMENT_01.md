# NOR-41 pre-data amendment 01: greedy duplicate-support tie handling

**Status: protocol revision BEFORE first NOR-41 source fixture or any NOR-41 CAL/EVAL data.** The original prospective scientific registration remains immutable at `87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f`. This amendment was motivated by the independent replayer parity test on a **historical SHA-archived NOR-40 source**, which is excluded from the 64 NOR-41 source identities. The NOR-41 program generator and native target have not been executed on the registered reps 84–99.

## Exposed deterministic edge case

The registered attacker recipe selected the first eight distinct supports by CAL score from sixteen greedy restarts, two CAL-only swap refinements and one exact pool16 optimum, then repeated its best candidate if fewer than eight distinct supports were available. The same registration required the selected eight average CAL detection cost to be **no greater than the eight strongest original greedy starts**.

When greedy starts return duplicates of an exceptionally evasive support, these two constraints can conflict. Historical-source tests found the independent implementation's CAL dominance assertion failing on a genuine archived NOR-40 public source. This is a deterministic algorithm-definition error discoverable before any NOR-41 TEST source is created. It does not change the risk budget or top-curvature scientific question.

## Frozen prospective resolution

Use the full CAL-only candidate set: 16 original greedy radius-four supports, two swap-refined supports from the lowest-cost **two greedy restarts** (duplicates allowed), and the exact size-four optimum from the predeclared 16-coordinate CAL-only pool. Evaluate each support's **same-budget** integer detection objective. The original greedy reference is the **eight lowest-cost greedy restarts including duplicate supports**, with total CAL objective `G`.

Sort the **unique** candidate attacks by `(CAL objective, sorted support indices)`. For `q` from `min(8, number of unique candidates)` descending to 1, form the eight-attack multiset:
`first q unique best scored candidates + (8-q) additional copies of the single best candidate`.
Choose the **largest q** whose summed CAL objective is ≤ `G`. Such a q always exists, because q=1 repeats the candidate best support and its objective is ≤ the best original greedy support. This rule maximizes attack diversity subject to the exact registered CAL-dominance gate, uses only CAL, and creates exactly eight r4 supports. Archive `q`, the original greedy eight sum and the selected eight sum. Ties resolve lexicographically.

The intended scientific gate remains unchanged: `selected_best8_cal_sum ≤ greedy_best8_cal_sum`. All 64 source identities, registered fresh source seeds and original independent source-only freeze, four CAL budgets, 16 greedy restarts/budget, best-two swap cap, exact 16-pool enumeration, 1024 EVAL schedules, five scientific thresholds and original blinded EVAL endpoints are fixed. The observed number of distinct selected supports is now an explicit optimization-diagnostic output.

**Integrity rule:** this amendment and corrected implementation must pass the independent archived-history parity test BEFORE the first full NOR-41 public-only source fixture job. A post-data adjustment to this rule would require a new cohort and registration. Original first NOR-41 results, whether PASS or FAIL, must retain their exact SHA bytes.

The source-cohort rule, fixture schema and theory premises are unchanged. Reproducible public SHA streams remain scientific fixtures rather than security-grade independent entropy.
