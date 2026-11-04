# QO-09: A fractional-information probe policy under QO-08's paired harness

**Protocol fixed November 4, 2026, before execution of the complete QO-09 report.** This extends [QO-08](PAIRED_PROBE_PILOT.md) with a fourth probe allocation strategy and a permutation control, while retaining its original three methods unchanged. The original QO-08 environments and their previous results are **already public** and form a retrospective regression panel. Two additional small synthetic cases are fixed in source before reading their first completed QO-09 report, forming an exploratory extension rather than an independently sealed evaluation.

The central question is whether a **static allocation derived from a valid QI-03 fractional-cover certificate**, under the same 128 Bernoulli query budget, helps a QO-07 quotient-certification task whose true hypothesis family is different. This is an experiment about the *transfer of an experiment-design heuristic*. An optimal lower-bound allocation for one declared decision problem is not automatically an optimal policy for another.

## 1. Prior mathematical source and the certified source design

The [EvalMath QI-03 theorem](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/active_quotient_fractional_information.md) derives a necessary evidence budget for **baseline-versus-alternatives** Bernoulli hypothesis identification. With baseline rate p=1/2, changed rate q=5/8, and baseline error tolerance delta=1/20, its fractional LP covers all alternative supports by optimal nonnegative weights x for named probe cells.

We construct a **public alternative family determined solely by the dimensions of the state/action interface**, before any evaluated true rates or Bernoulli observations enter the policy. Enumerate cells in state-major order, and partition them into successive blocks of at most six cells. In each block with at least three cells, include the three two-cell alternative supports {0,1}, {1,2}, {0,2}, each local to the block. Assign primal weight 1/2 to each of these three cells and dual weight 1/2 to each of the corresponding alternatives. Remaining cells have singleton alternatives with primal and dual weight 1. A short block has only singleton alternatives.

For six cells, the exact fractional primal allocation is

\[
x=(1/2,1/2,1/2,1,1,1),
\quad \sum_c x_c=9/2.
\]

For eight cells, two more independent singleton constraints give an exact objective of 13/2. All alternatives meet \(\sum_{c\in C_j}x_c\ge1\), and all probe cells satisfy the dual bound \(\sum_{j:c\in C_j}y_j\le1\). The primal and dual objectives are equal. By **finite LP weak duality**, formalized in [MOR-08](https://github.com/patrickdugan/OntologicalProofs/blob/main/Neoplatonic/LTR/MorphospaceInformationCover.lean), these are globally optimal fractional solutions for their constructed alternative families.

The QI-03 reference LP enumerator has a six-cell scale cap. Our eight-cell solution uses disjoint block certificates and directly verifies the *global* primal/dual inequalities. The tests also independently enumerate all half-integer candidate allocations on up to eight cells, confirming the optimum. This is an exact transfer of the QI-03 **optimization mathematics**, with no runtime dependency on the EvalMath repository.

## 2. From LP solution to a query policy

Normalize the positive primal vector to shares \(\rho_c=x_c/\sum_i x_i\). At each query, the \`fractional_cover\` policy chooses

\[
 c_t \in \arg\min_c \frac{N_c(t)}{x_c},
\]

where \(N_c(t)\) is the previously observed number of queries at cell c. Cell IDs break ties deterministically. The policy sees the count ledger and the public dimensions. Its selection is fixed before drawing the next outcome. Each weight is strictly positive, so every state/action cell is sampled; there is no hidden oracle access to the evaluated rate table.

The **rotation control** \`fractional_cover_rotated\` cyclically moves the alternative-support masks and their optimal weights by half the number of cells. Its primal/dual objectives and total query budget remain equal to the original design, while privileged cell indices change. A benefit shared by rotations would support an explanation in terms of the *weight histogram* and broad coverage. A benefit that depends strongly on rotation would suggest incidental alignment with particular public state/action indices.

The fractionally weighted schedule is an **allocation heuristic**. QI-03's necessary information lower bound concerns its declared baseline and alternatives; the QO-08 feasibility task has distinct rates, thresholds, and goal classes. The experiment therefore makes **no transfer of a QI-03 error guarantee** to the QO-08 environment. Its actual confidence certificates are supplied by the independently valid QO-07 anytime beta-binomial e-process, conditional on its addressed-IID sampling assumptions.

## 3. Frozen casebook and matched budget

The original QO-08 casebook includes three familiar synthetic tasks: \`two_cluster_feasible\`, \`separated_infeasible\`, and \`weak_margin_feasible\`. Their results are already known and are labelled retrospective in the QO-09 output.

Two additional synthetic cases are declared before the new pilot:
- \`crossed_clusters_feasible\`: four transient states, two actions, a two-class limit, and tolerance 7/20, with high/low hit probabilities crossing between actions
- \`crossed_infeasible\`: three transient states, two actions, a one-class limit, and tolerance 1/10

The source [fractional_cover_probe.py](fractional_cover_probe.py) contains every true Fraction rate and case label. The true table is supplied to the **simulator and post-run auditor** and is never accessed by the fractional selector. Any proposed neural or stronger generalization experiment would need independently sealed holdouts.

The five policies evaluated are round-robin, confidence-interval width, robust bottleneck, QI-03 fractional cover, and the rotated fractional control. The **same six QO-08 seeds** (11, 19, 23, 37, 41, 53) and **same cumulative checkpoints** (24, 64, 128 queries) apply. Alpha=1/20, ten exact-rational confidence-bound bisection steps, and the class budgets are shared. Every policy receives the same deterministic per-cell Bernoulli draw stream prefix for the same scenario/seed and consumes exactly 128 queries by its final checkpoint. The existing [QO-08 stream scheduler](paired_probe_pilot.py) is used unchanged.

This defines 5 scenarios × 6 seeds × 5 policies = **150 complete trajectories**, 450 checkpoint records, and **19,200 individual Bernoulli queries**. The casebook source is the authoritative preregistered specification. All paired differences, including negative results, are preserved.

## 4. Evaluation and source-bound audit

Every checkpoint retains the complete three-way result: \`certified-feasible\`, \`certified-infeasible\`, or \`undetermined\`. Additional fields include the robust upper prediction-error bound, true optimum (audit-only), actual frozen macro-model error (audit-only), post-selection partition regret, all per-cell query counts, coverage failures, and incorrect certificates. An incorrect certificate *on the valid confidence event* raises an error.

For every matched seed and case, report the exact Fraction of

\[
  E_{\text{baseline frozen}}-E_{\text{fractional frozen}},
\]

alongside the difference in robust upper bound and a count of seed-wise wins/ties. Positive means lower observed true error for the fractional policy; negative means the baseline did better. The rotated policy is a named negative control.

The required source-hashed machine receipt is saved at [results/fractional_probe_pilot/receipt.json](../../results/fractional_probe_pilot/receipt.json). It records the complete 450 checkpoint rows, casebook, both certified primal/dual allocations for every case, grouped outcomes, paired differences, and the SHA-256 of every complete trajectory's deterministically reproducible draw sequence. Source SHA-256 hashes additionally bind the QO-09 code and tests to the unchanged QO-07/QO-08 generators and the receipt script. The compare-or-fail verifier reruns **every trajectory** before acceptance.

Reproduce from the repository root:

    python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_fractional_cover_probe.py' -v
    PYTHONPATH=. python scripts/verify_fractional_probe_pilot.py --check

Both commands run in the read-only GitHub workflow. No model inference, paid API tokens, cloud GPU time, or physical experiments are used.

**Reporting rules:** Preserve all unresolved trials and confidence-sequence failures. Call the QO-08 cases retrospective and the two new cases exploratory, avoid selection on observed results, and treat paired differences as descriptive across six seeds. Record algorithm CPU/planning overhead separately before any future claim of superior economic efficiency. An LP optimum certifies only its stated training-family fractional covering problem.
