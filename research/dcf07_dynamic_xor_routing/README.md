# DCF-07: Dynamic higher-order dependency routing

2026-11-25. CPU-only exploratory study following DCF-06: six systems, 24 context graphs, 468 primary test cells and 144 repaired follow-up cells. Primary protocol and gym hashed locally before execution; follow-up and numerical repair separate. No external preregistration, real LDT model, GPU, energy or inference timing measurement.

## Environment and information boundary

K=6,8; seeds 431,433,439. Four context graphs per system: independent flips, random reciprocal pairs, random disjoint XOR triples (leftovers independent), novel connected XOR cycle (each bit controlled by next two in a random permutation). Graph and error rates hidden. Observable context ID, bit identities, command semantics, independent fair reset prior, parity hypothesis class and maximum two parents supplied. A command flips a target unconditionally if no parents, otherwise by XOR of parent bits; noise and single-bit read follow. Rates shared across contexts: emission U(.05,.18), idle U(.015,.07), command error U(.03,.12). Context switches preserve hidden state. Novel cycle has its own reset calibration and validation; this is adaptation, not zero-shot generalization. Zero-shot comparator uses pooled seen graph for novel context with no novel probes. Evaluation schedules deterministic and context visible each step.

## Higher-order reset identification

Reset, idle/read target j, idle/read every other bit once, command/read target. Independent fair resets and independent noises imply pre-command bits remain independent fair. Let E be (-1) to the parity of target pre/post observations, and R_p be (-1) to the read of candidate parent p. For nonempty true parent set P, E is parent parity sign times independent target error parity. Therefore E[E product(R_p:p in S)] is zero for S != P and equals (1-2e_j)^2(1-2f_j) product(1-2e_p:p in P) for S=P. For unconditional flip, mean(-E) equals (1-2e_j)^2(1-2f_j), and nonempty moments vanish. This is identification under the supplied reset/probe assumptions, not a general passive causal discovery theorem.

Screen sets of size 0,1,2 on 2048 reset words per target/context; select largest moment and confirm on 4096 fresh words. First-order restricts size <=1; pooled averages seen-context moments. Require lower bound score - sqrt(2 log(12K/.05)/4096) > .02. One-sided Hoeffding and union bound over at most 12K confirmations controls incorrect positive bounds per system under independence, not power or population generalization. Unconfirmed hypotheses fall back to empty. Estimate emission via two reads without target action, idle via two reads with intervening idle, and command error by corrected parity moment median across seen contexts. Rate probes 8192 trials per design per target. Mean absolute errors: emission .002193, idle .007715, command .011667.

## Graph results

|Estimator|Exact context graphs|Correct parent sets /168|Correct nonempty /120|Correct XOR /78|False edges|
|---|---:|---:|---:|---:|---:|
|Second-order|24/24|168|120|78|0|
|First-order|12/24|90|42|0|0|
|Pooled|6/24|48|0|0|0|

Second-order confirms all 168 assignments; first-order confirms 90, 78 XOR cases default to empty. Pooled always predicts unconditional flip. Graph calibration 1,032,192 reset words / 8,404,992 observations, including novel probes (one quarter); separate rate calibration 688,128 words / 1,376,256 observations. Expensive calibration; no sample-efficiency claim. Contexts within each system share rates and are not independent seeds.

## Dynamic cluster inference

Greedy graph-hyperedge partition with cap 1,2,3,4. Degrees of freedom = sum(2^size - 1); stored arrays contain sum(2^size) FP entries, excluding temporaries/parameters. Given correct graph/rates and independent reset, a partition without cross-edges is exact. For cross-cluster XOR, multiply signed parity expectations across factors, retain joint parity within each factor; update only observed factor. On changing partitions, approximate old product by product of new-cluster marginals. This preserves those marginals, is exact if new partition coarsens old, and forward KL projection cost equals sum new cluster entropies minus old joint entropy. Splitting can permanently discard correlations. No uniform error guarantee.

Static validation 192 words x64/context, minimize NLL+.0004*(coords-K). All six select cap1 independent, cap2 pairs, cap3 triples; novel cycle cap3 twice, cap4 four times. Full learned graph agrees with known-graph estimated-rate oracle. Shuffled graph preserves parent-set size, may retain some correct edges. GRU64: 600 AdamW steps lr .003, batch64 alternating passive seen corpus (6144 length32 words), seen graph probes and rate probes, with replacement. 16385/17153 params K6/8, aggregate training 47.67 CPU seconds with concurrency. GRU receives no novel probes/validation and receives extra passive seen data. Unequal structure and training exposure, especially novel context; preliminary comparison, not a tuned neural upper bound.

Mean conditional oracle KL, 256 words x128 per scenario/system, three seeds per K:

|K|Scenario|Learned full|First-order full|Pooled full|Shuffled full|Marginal|Adaptive|Zero shot|GRU64|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|6|independent|.000792|.000792|.000792|.000792|.000792|.000792|.000792|.184973|
|6|pairs|.000886|.000886|.122073|.073307|.017810|.000886|.000886|.164466|
|6|triples|.000902|.123468|.123468|.089915|.031529|.000902|.000902|.159042|
|6|novel cycle|.000907|.121225|.121225|.087538|.031719|.018031|.121225|.157874|
|6|switch seen|.000824|.054381|.096024|.064166|.015753|.002716|.002716|.173242|
|6|switch novel|.000804|.075590|.105810|.073184|.019316|.005487|.040854|.168577|
|8|independent|.000474|.000474|.000474|.000474|.000474|.000474|.000474|.211743|
|8|pairs|.000490|.000490|.134887|.098087|.017933|.000490|.000490|.194477|
|8|triples|.000508|.103776|.103776|.078870|.023365|.000508|.000508|.192296|
|8|novel cycle|.000502|.132833|.132833|.096839|.032461|.013822|.132833|.186919|
|8|switch seen|.000494|.045269|.095632|.068555|.013723|.002799|.002799|.202103|
|8|switch novel|.000482|.073982|.109413|.079086|.018086|.005391|.045411|.198026|

Across six switch-novel systems adaptive .005439 vs marginal .018701 (70.9% reduction); full .000643, first-order .074786, pooled .107611, shuffled .076135, zero shot .043132, GRU .183301. K8 adaptive mean 15.4167 degrees vs 255 full (16.54x); maximum 17 or 30 in novel context. Switch seen K8 mean 11.75. Static selection and context-conditioned partition switching, no autonomous context detection. All 468 primary cells finite. Six independent full-joint enumeration checks max 9.44e-16, 18 static component checks max 2.22e-15, 18 causal prefix checks max 0.

## Follow-up: switching frequency and numerical failure

Post-primary, same graph/rates, fresh RNG. Compare primary reset-selected dynamic partition, uniform cap selected on switched validation, fixed union partition (cap1..4), all dynamic caps and full. Switched validation 128 words x96, period8, same penalty. Test 192 words x128 at periods1,8,32. Fixed uses correct context-conditioned transitions; only partition fixed. Switch validation chooses cap3 four times, cap4 twice; fixed cap3 all six.

Original rapid-switch test produced nonfinite cells at K8 seed439 period1: reset-selected, switch-selected, dynamic3. Repeated repartition amplified factor mass roundoff. Original failed log/receipt retained (Python JSON NaN). Repair explicitly normalizes old and new factors at repartition, asserts finite/nonnegative/normalized. Replay identical RNG: all 144 follow-up cells finite; no retuning graph or objectives. Six projection tests preserve marginals and exact coarsening, max discrepancy 3.33e-16.

|K|Period|Reset-selected KL|Switch-selected KL|Fixed-selected KL|Full learned KL|
|---:|---:|---:|---:|---:|---:|
|6|1|.011895|.012061|.009599|.000851|
|6|8|.005484|.005657|.008653|.000814|
|6|32|.004269|.004372|.008027|.000827|
|8|1|.010504|.010769|.009660|.000483|
|8|8|.005220|.005601|.009218|.000492|
|8|32|.003114|.003506|.008289|.000507|

Mean coordinates K6: 11.42 reset-selected, 11.08 switch-selected, 14 fixed, 63 full. K8: 15.42,14.33,16.67,255. Dynamic lower mean KL at periods8/32 with fewer coordinates; fixed lower at period1 but uses more coordinates. Switch-aware validation reduces coordinates at cost of increased raw mean KL vs reset-selected. No universal switch threshold or hardware advantage established.

## Reproduction

Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, one CPU thread per process. From this directory:

    sha256sum -c PRE_RUN_SHA256
    bash run_all.sh
    sha256sum -c FOLLOWUP_PRE_RUN_SHA256
    python followup.py > followup_failed.log 2>&1
    mv followup_receipt.json followup_failed_receipt.json
    sha256sum -c REPAIR_PRE_RUN_SHA256
    python stable_followup.py > followup.log 2>&1

Conversation artifact dcf07_reproducibility.zip contains detailed report, exact sources, six GRU checkpoints, raw primary and both follow-up receipts/logs and SHA256SUMS; SHA256 f0f5f0b4ec4054e3035ff859a544c88b234c09764a320eff4bb890668d40133f. No paid compute. Original gym frozen to reproduce failure; repaired runner is not a universal numerical stability guarantee.

## DCF-08 candidate

Budgeted routing controller estimates graph dwell time and loss of discarded correlations, chooses retain/merge/split, charges reconstruction costs. Test unseen schedules, drift without context labels, noisy context labels, nonparity interactions, matched calibration budgets, stronger recurrent and particle-filter baselines. Register fresh query and quality gates; measure actual compute and energy separately. No claim of actual LDT deployment.
