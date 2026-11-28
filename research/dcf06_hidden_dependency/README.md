# DCF-06: Hidden dependency discovery and joint predictive-state compactification

2026-11-25. Exploratory CPU study. Primary protocol and executable gym hashed locally before execution (PRE_RUN_SHA256); no external preregistration or blinded adjudication. Separate post-hoc follow-up with fresh validation/test streams. No GPU, energy, LDT router or language-model measurement.

## Environment and information boundary

K=4,8; families independent, random disjoint reciprocal pairs, random directed cycles; seeds 311,313,317 (18 systems). At matched K/seed, families share noise rates and permutation order, so do not count them as independent population replicates. Each reset draws independent fair hidden bits. Step chooses action index j, command c, observation index r. Flip j by c XOR noise if independent, or by c*s[parent(j)] XOR noise if coupled. Noise rates are idle U(.015,.07), commanded error U(.025,.15), observation error U(.07,.20). Read bit r with independent emission error. Learners receive action/read indices, commands and noisy observations, never latent bits or true graph/rates. The bit count, reset prior, single-parent hypothesis class, command structure and observation semantics are supplied. Full joint oracle has true graph and rates. Passive and active discovery use different observation protocols.

Active probe: reset; idle/read target j, idle/read candidate p != j, command/read j. Score D=P(pre/post read disagreement | observed p=1) minus P(disagreement | observed p=0). With independent fair resets and independent errors its expectation is (1-2e_j)^2*(1-2f_j)*(1-2e_p) for the true parent; zero for unrelated candidates or an unconditional flip. This follows from the parity of the two read errors and flip error, and from the posterior difference of the fair controlling bit conditional on its noisy read. Only one candidate per target is selected using 512 words/pair; confirm on 1024 fresh words/target. Lower bound subtracts sum of sqrt(log(2K/alpha)/(2n_y)) for y=0,1; alpha=.05 per system. Accept when lower>.05. Conditional on nonempty groups and independent confirmation, one-sided Hoeffding and union bound control incorrect positive lower bounds across the K selected candidates; not an across-system guarantee or guarantee of power. Missing an edge means the rule did not confirm it, not proof of independence.

Passive heuristic: use commanded target-read events with a previous target read and candidate parent read, with no intervening actions on either corresponding variable since its read. Compare disagreement conditional on the last parent read; at least 16 events per group; select largest score if >.15, otherwise independent. This is a heuristic observational score, not a generic causal identification theorem. No extra queries beyond the common training corpus. Shuffled control replaces each confirmed parent by a different nonself candidate; preserves number of confirmed edges. No graph tuning on held-out outcomes.

## DCF-06A: graph discovery and prediction

| graph estimator | exact graphs | correct node assignments /108 | true edges found /72 | false edges | missed true edges |
|---|---:|---:|---:|---:|---:|
|passive|18/18|108|72|0|0|
|active|15/18|105|69|0|3|
|shuffled active|6/18|36|0|69|72|

Active costs 423,936 extra 3-step probe words (1,271,808 noisy observation labels) across 18 systems. Passive training corpus 4,096 length-32 words per system (131,072 labels, 2,359,296 total). Three active misses all selected the correct candidate at screening but confirmation lower bound fell below .05; two misses share K8/seed311 noise and parent. Exact graph recovery is small-family empirical performance, not a graph-discovery guarantee.

Fit all graph hypotheses using the same passive data: one per-bit idle/commanded noise and emission parameter (3K), Adam lr .035, 150 updates, batch64, length32. Equal graphs share fitted checkpoint; 45 unique filter fits. GRU64 uses 600 updates, lr .003, same corpus, 18 fits. Different priors/optimization prevent identifying architecture alone as causal; no tuned GRU upper-bound claim. 63 unique training runs, 108 named checkpoint files including aliases. Sum training-loop wall times 229.775s with four concurrent jobs, not a single-device performance benchmark.

Product-of-clusters posterior: exact transitions within a cluster, external parent approximated by its marginal, exact read update of the observed cluster only. For a partition into sizes m_i, degrees of freedom = sum_i(2^m_i-1); stored implementation arrays have sum_i 2^m_i entries. For true graph with no intercluster edges, independent reset and single-bit observations, factorization is exact by induction. Otherwise dropping correlations and cross-cluster observation updates generally introduces error. Fit rates once with mean field; reuse them for cluster variants to isolate the inference change (could bias learned rates). Greedily merge highest-confidence graph edges under size cap 1,2,4. Validation on fresh 256 length-64 trajectories chooses NLL+.0005*(coordinates-K) among independent, passive singleton and active singleton/pair/4-cluster options, excluding known and shuffled controls. Test 256 fresh trajectories per length 32/128 and per random or shifted policy (repeated target/parent/target triples, without resets inside trajectories). Per-system metrics retain paired word losses and oracle conditional KL. Main 864 test rows.

Primary length128 random-policy conditional KL, mean across 3 seeds (nats per prediction):

| family | independent | shuffled | passive singleton | active singleton | active cap2 | active cap4 | known cap4 | selected | GRU64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4 independent|.000137|.000137|.000137|.000137|.000137|.000137|.000137|.000137|.120242|
|4 pairs|.048167|.046476|.012722|.012722|.001233|.001233|.001233|.001233|.093053|
|4 cycle|.046148|.044714|.013541|.015512|.010429|.003435|.000535|.004198|.093372|
|8 independent|.000469|.000469|.000469|.000469|.000469|.000469|.000469|.000469|.176063|
|8 pairs|.063499|.062507|.015885|.017326|.003517|.003517|.002028|.003517|.153093|
|8 cycle|.061421|.060389|.017785|.019222|.012739|.006671|.006324|.012866|.152623|

Primary selected choices: active singleton 6 (all independent), active cap2 8 (all pairs, two 8-cycles), active cap4 2 (4-cycles), passive singleton 2 (one 4-cycle, one 8-cycle). Under shifted policy, selected KL averaged over both widths is .000335 independent, .002105 pairs, .007360 cycles vs .000335/.058403/.057865 for independent model. These are finite-horizon expected conditional KL estimates on realized histories, not a uniform guarantee for unseen policies.

## DCF-06B: passive clustering, fresh streams, exploratory

After primary results, add passive graph clusters with same learned rates, new validation and test RNG, same penalty. Keep previously selected primary arm as frozen comparator. 180 extra test rows. No training on these fresh streams. Mean conditional KL on 128-step fresh random-policy tests:

| family | passive singleton | passive cap2 | passive cap4 | prior selected | passive selected | selected coords |
|---|---:|---:|---:|---:|---:|---|
|4 independent|.000138|.000138|.000138|.000138|.000138|4,4,4|
|4 pairs|.012101|.001235|.001235|.001235|.001235|6,6,6|
|4 cycle|.013449|.006613|.000524|.004159|.002440|15,15,6|
|8 independent|.000472|.000472|.000472|.000472|.000472|8,8,8|
|8 pairs|.015814|.001997|.001997|.003607|.001997|12,12,12|
|8 cycle|.017900|.009904|.006298|.012683|.008611|12,17,11|

Shifted-policy passive selected KL: .000133,.000921,.002019,.000540,.001500,.007572 in table order. At K8, pairs use 12 vs 255 joint degrees of freedom and recover 87.4% of singleton KL; cycle uses 11–17 coordinates and recovers 51.9%. All independent cases select singleton, all pairs select cap2; cycles select cap2 (3) or cap4 (3). This is static per-system validation selection, not dynamic per-instance routing or actual memory packing. No measured accelerator savings.

Post-hoc nested passive corpus ablation, same primary systems: exact graphs 8/18 at 128 words, 18/18 at 512/1024/2048/4096. Extra graph-only mixed directed forest cases (K4/8; seeds401,409,419, roots and nonroot children): passive 6/6, 21/21 edges; active 5/6, 19/21 edges; neither produces false edge. No predictive filter fit for mixed extension. These cases broaden the topology family modestly; no general hidden graph identification claim.

## Checks and reproducibility

90 singleton NumPy vs PyTorch comparisons max discrepancy 4.068e-7. Exact independent/pair component vs full joint checks (12 systems) max 2.554e-15. Independent explicit source-to-destination enumeration of full joint transitions on 18 fixtures max 1.110e-15. All 108 named checkpoint causal-prefix tests pass (flip observations at index8 onward, predictions through index8 unchanged, max delta0). These checks do not certify approximation accuracy. Python3.13.5, torch2.10.0+cpu, numpy2.3.5. From this directory:

    sha256sum -c PRE_RUN_SHA256
    bash run_all.sh
    python followup.py

dcf06_reproducibility.zip in research conversation includes exact source, protocol, raw receipts, all named checkpoints, logs and SHA256SUMS. SHA256 1f206126aaccde3abc243d7b874eb1bda82f181bb747c644d126f9a974da22e6. No paid compute. All results and failures recorded. No inference timing, energy, GPU/TPU or general dense matrix exponent conclusion.

## DCF-07 candidate: dependency-guided routing under changing structure

Move beyond static single-parent systems and per-system calibration. Randomize graph by episode with observable context, allow multiple parents and higher-order interactions with indistinguishable pairwise marginals (e.g. XOR), and test a held-out graph family. Compare fixed graph, context-conditioned graph predictor, adaptive cluster router, known structure oracle and strong recurrent controls under common observation and state/compute budgets. Predeclare query and false-edge costs, calibration/abstention and conditional-KL gates; record out-of-distribution failures. For a real LDT model, log selected experts/edges, actual recurrent passes, end-to-end latency and joules per correct task on matched hardware. No extrapolation from DCF-06 to general AGI implied.
