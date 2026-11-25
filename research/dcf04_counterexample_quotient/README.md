# DCF-04: Counterexample-guided compactification without known cardinality

Date 2026-11-25. Executed CPU-only exploratory study. Follow-up to DCF-03; no language-model, accelerator or energy measurement.

## Systems and budgets
24 freshly generated minimal reachable binary-output deterministic finite automata, alphabet size four. Initial 16 systems: n in {4,8,12,16}, two seeds per n and each of random permutation/random arbitrary map families. Additional eight off-grid systems: n in {5,9,13,20}, one seed per family. Oracle draws random binary acceptance maps and rejects nonminimal/unreachable systems. Learners receive alphabet, known initial-state convention, and binary observations only. No cardinality, primitive decomposition, transition law or accepting state supplied. Neural model selection grid {4,8,12,16} is shared across systems, including off-grid. Every operation is exposed; no unseen-operation claim. The 24 systems are independently generated, but are not a sealed prospective benchmark.

Neural soft automaton: learned unconstrained stochastic transitions and readout; 200 AdamW updates, lr .025, 48 length-four words/update; supervise binary prefix outputs plus initial observation. Each candidate uses same passive draws and validation traces, selects by length-eight BCE, never audit outcome. Four candidates/system. Per system 38,401 training label draws including initial bit once, 2,048 validation draws, 154,400 training loss-label uses over four candidates; repeats are common. All 96 checkpoints retained. Harden by row argmax and output threshold. For guidance, greedily choose up to six length<=4 distinguishing suffixes of the candidate automaton, then suffix-close. Random sham matches number/length of proposed probes before suffix closure (closure can differ). Repair tests proposal first, then initializes an observation table with counterexample prefixes and proposed suffixes. It does not optimize neural weights on the counterexample.

Active symbolic baseline is standard closed/consistent observation-table learning (L* style). Exact equivalence teacher accesses full target/candidate product graph; count teacher calls, visited pairs and edges separately. Sampled teacher only queries output membership on up to 64 words each at lengths 1,2,4,8,16,32 per attempt; stop at first disagreement. All distinct membership words including sampled tests charged to 2,000 budget, EQ attempts capped at 20. The final audit is exact in both modes and provides no repair feedback. A sampled teacher's acceptance is not an equivalence proof.

## Initial 16 systems
Passive validation-selected neural exact: 4/16, all n=4. 12/64 candidate checkpoints exact (not a model-selection result). Complete trace accuracy at lengths 4/16/64: hardened 90.84/36.55/25.56%; soft 91.31/36.10/24.55%, 512 traces per system/length. The long-horizon exact rate is dominated by easy systems. No neural exact success at n>=8. Selected neural models with exact behavior are overparameterized and admit smaller reachable minimal quotients.

| teacher/arm | exact | membership | EQ calls |
|---|---:|---:|---:|
|exact/cold|16/16|3977|68|
|exact/random|16/16|4421|38|
|exact/neural suffix|16/16|4136|32|
|exact/neural repair|16/16|4059|42|
|sampled/cold|16/16|8699|67|
|sampled/random|15/16|9301|36|
|sampled/neural suffix|16/16|8804|31|
|sampled/neural repair|16/16|8836|41|

Sampled/random accepted an incorrect 12-state map hypothesis (seed 2026113710; counterexample [3,2,1,1]); independent exact audit rejected it. Do not infer correctness from the random equivalence test. 

## Off-grid extension and combined results
Off-grid neural selected exact 1/8 (5-state map, selected k=16), 3/32 candidate fits exact. All eight active arms solve every off-grid system exactly within budgets, including n=20 > largest neural candidate. No causal claim on training budget or architecture.

Combined 24 systems:
| teacher/arm | exact | membership | EQ calls | exact teacher visited pairs | membership tokens |
|---|---:|---:|---:|---:|---:|
|exact/cold|24|6360|105|1258|32555|
|exact/random|24|7355|59|777|37015|
|exact/neural suffix|24|6524|46|647|32012|
|exact/neural repair|24|6500|63|942|33214|
|sampled/cold|24|14330|106|0|138252|
|sampled/random|23|15202|57|0|145899|
|sampled/neural suffix|24|13366|45|0|130745|
|sampled/neural repair|24|13436|60|0|132813|

Neural guidance decreases exact-teacher calls and visited pairs; membership count rises 2.58% relative to cold. Sampled neural suffix reduces distinct memberships 6.73%, but ignores substantial passive training cost. Aggregate neural training loop time 18.45 seconds, 921624 training label draws +49152 validation label draws across systems before deduplication, 3,705,600 training loss-label uses. All cold exact learners require only 6360 distinct chosen membership words and .032s aggregate active learner time in this synthetic oracle setting. No evidence for end-to-end neural efficiency.

## Passive state-merging control
Red-blue greedy deterministic state merging on a labeled prefix tree, using only the original passive traces. All observed prefix labels preserved. Missing transitions totalized by rejecting sink. Two arms: training-only length4 draws, and training plus original length8 validation observations (these become training data for the symbolic arm, whereas neural uses them only for selection; same observation access, different use). Deduplication leaves 8184 and 31839 distinct labeled words respectively across 24 systems. 
| method | exact | merge time |
|---|---:|---:|
|neural selected|5/24|18.45s training|
|passive merging, length4|6/24|0.079s|
|passive merging, train+validation|21/24|0.532s|
|passive merging + exact repair|24/24|0.532s +0.011s|
|passive merging + sampled repair|24/24|0.532s +0.046s|

Merge times exclude data preparation; neural times exclude data preparation/selection/test. Repair query counts are additional to passive observations, with no cross-cache reuse: exact 1642 membership, 27 EQ, 10214 membership tokens; sampled 8435 membership, 27 EQ, 108483 membership tokens. Cold active symbolic uses fewer total labels than passive+repair when passive traces must be collected specifically for the task. When passive traces already exist, symbolic proposal/repair is a useful reuse route. No global cost optimum claimed.

## Independent audit
Independent partition refinement of disjoint union of candidate and target confirms equivalence iff their start states share a stable partition. This is algorithmically distinct from product BFS used in training/primary audit. Replayed all 96 neural checkpoints, 192 active tables, 48 passive tables, 48 passive repair tables; all primary equivalence verdicts agree. Every successful active table minimizes to true cardinality. BFS counterexamples checked against integer execution. Accepted exact tables certify every finite word, conditional on the specified finite oracle; this says nothing about arbitrary partially observable stochastic environments.

## Reproduction
From this directory:
    python gym.py run
    python offgrid.py offgrid
    python audit.py .
    python passive.py .
    python passive_repair.py .
Local Python 3.13, PyTorch 2.10.0+cpu, NumPy 2.3.5, one CPU thread. Raw receipts, checkpoints, scripts and SHA256SUMS in dcf04_reproducibility.zip supplied with the research conversation. No paid compute. Exact model cardinalities and transition matrices appear only in the environment and post-hoc audit; learner interfaces use observations and teacher responses.

## DCF-05 candidate
A useful frontier is approximate or noisy history equivalence where cheap exact membership/complete equivalence are unavailable. Introduce hidden stochastic transducers and noisy observations, continuous recurrent state, expensive query budget and a held-out statistical witness suite. Compare passive state merging with uncertainty-aware refinement, neural predictive states and hybrid proposals under equal label/compute cost. Report finite-horizon error and calibrated abstention; do not claim exact certification from sampled checks. Also test algorithmic systems with symbolic state explosion where minimal table is too large, using compositional programs rather than a flat DFA.
