# OTTD-CAP-01 scientific protocol v0.1, 2026-10-30

Base repository: Compactification-Theory main c438bf55ecebbd84392157f52141ca3153864626. Real benchmark reference: thesophiaxu/openttd-bench fee25ab19837959fd9c9749808b117144b9d1b3e. All executed evidence in this revision is synthetic.

## Defined finite process

Let route i have publicly known build cost c_i and income pair g_i > b_i. Its private quality z_i is an independent fair bit. At simulated month t, the world contains spendable cash C_t, cumulative operating profit P_t, paid capex K_t, active route mask A_t and a transient last-survey reply. The public observation comprises these fields and the public route catalog. A survey reveals (route, bit) at the next decision and expires after the subsequent action. Each survey, build or wait consumes exactly one month. Already-active routes produce income during an action. Building deducts c_i immediately, with income from the new route beginning next month.

The policy maximizes E[C_T-C_0] for the declared synthetic prior. The full-information oracle knows all quality bits and is an upper control. The upstream OpenTTD score uses cumulative operating profit and relative rank; there is no claim of equivalence between net cash and that grade. Fees, salvage, inflation, debt, spatial capacity, demand shocks, traffic, opponent actions and nonlinear route congestion are absent from this initial micro-world.

For a posterior ensemble B with shared public observation and positive-probability hypotheses, the exact history-aware value is

    V(B,t) = max_{a legal} sum_o P(o | B,a) V(B_next(o,a), t+1)

with terminal V the posterior mean of net cash. The solver partitions every reachable successor by complete public observation and uses exact Fractions. It enumerates the entire finite action/feedback tree for n <= 3 and T <= 7. It therefore solves an *expected value* problem for the declared synthetic kernel. Predictive equivalence for every possible adaptive continuation, minimum automaton size and general POMDP information bounds remain with the separately pinned EvalMath IRC-02 reference.

## Memory budget

Scout retains at most floor(B / w) route-quality records, w = ceil(log2(n)) + 1. The deterministic survey order comes from public month t; no hidden phase index is supplied. Route catalog, fixed code, private mutable state, source checkpoint, encoder, token budget, CPU time, action time, server latency and replay space belong to different ledger categories. The synthetic run records actual peak retained bits and decision count. No training or model inference cost is claimed.

A student policy with a smaller cache can lose past survey replies. Compare paired seeds under frozen route distributions and public catalog. The exact oracle is the ceiling for *all history-aware* finite policies in this synthetic process, while the clairvoyant oracle grants private quality from time zero. For a real OpenTTD experiment, opponent behavior and the real-time clock must be fixed or paired, and all action/observation/billing records must be retained.

## Checks and failure conditions

Run standard and optimized Python unittests. Check that building reduces cash but not operating profit, income begins next month, quality leaks only through permitted feedback, terminal actions cannot extend the horizon, and illegal builds raise. Test JSON-roundtripped hash-chain receipts and detect tampering. Compare policy expectation against the exact posterior-history oracle and the clairvoyant control. Include the single-route hand-worked case: c=3, good=8, bad=-10, T=3. Survey and then build only if good yields E[net cash] = 5/2; the full-information upper control is 13/2.

The bridge must reject unknown or disallowed commands, require explicit write enablement, preserve validation flags, and detect both outer request errors and nested game execution failures. A read-only state probe must issue zero batch actions. A real map inspection can start the benchmark's live clock; this risk remains even with write access disabled.

A failed test, source mismatch, forged or inconsistent hash receipt, unexpected private observation, or a measured policy value beyond the exact expectation ceiling rejects the corresponding scientific claim. ATIF usage metadata is historical evidence; operating-profit measurements require the authoritative upstream grading snapshot.

## Native milestone

After the micro-world CI passes, provision a real upstream seat through its own runner. Validate a deterministic transport planner against the upstream schema and authenticated envelope. Add a policy-action compiler for construction and orders with request receipts and measured clock cost. Run paired real games with the same seeds and opponents before comparing a learned TRM/HRM controller to scripted baselines. Keep native scores in a distinct result namespace with provenance and uncertainty.
