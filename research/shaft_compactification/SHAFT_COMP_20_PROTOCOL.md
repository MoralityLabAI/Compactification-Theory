# SHAFT-COMP-20: Goal-code versus persistent temporal-state counterexample

**Registration date:** November 11, 2026, before running COMP-20 neural training and inspecting held-out outcomes. All tests are local, in-memory synthetic models with intact weights. This is a scientific test of behavioral-state observability, not a model-persistence or evasion experiment.

## Motivation and exact target

COMP-19 found a small positive AUC gain from a 36-output, two-query behavioral probe over an equal-capacity repeated-immediate-output control. Its tested task's final output after two new inputs depends on the goal table and the new inputs; the historical input before those two queries is forgotten. Therefore COMP-19 alone cannot establish that its probe recovers **persistent temporal state**.

COMP-20 constructs a finite recurrent task in which a previous-input register changes a persistent latch *after* the current output, and the latch affects all later outputs. With alphabet V=6, goal g∈{0,...,63}, state (g,l,p), latch l∈{0,1}, previous input p∈{0,...,5}, and fresh input q:

1. Output **before** changing the latch: y = bit(g,q) XOR l.
2. Update previous-input register: p' = q.
3. Update latch by XOR with a trigger T(p,q).

Two preregistered triggers:
- **repeat-toggle**: T(p,q)=1[q=p].
- **successor-toggle**: T(p,q)=1[q=(p+1) mod 6].

Initial state is (g,0,0). The full state space has 64×2×6=768 elements per family. Both rules preserve a bit of accumulated parity arbitrarily far into the future under common inputs once the two states' previous-input registers have synchronized.

The present four/six-query counterfactual response signature B1(g,l,p) depends only on (g,l), so same-goal, same-latch states with different previous symbols are behaviorally identical under the immediate observer. The two-query final signature B2(g,l,p) depends on p because its first query updates the latch. A goal-only signature G(g), formed by six table bits repeated six times, is identical for such states.

**Exact finite preregistered tests:** exhaustively enumerate all 768 states, form B1, B2, and G signatures for both triggers; compute the partition sizes and identify same-(g,l) pairs with p≠p' distinguished by B2. Verify that some such pairs remain distinguished under a common 32-token continuation, since the trigger difference from the first input survives as persistent latch difference. Prove the alias/split witness in Lean 4 using a small fixed goal and two previous inputs. The precise finite counts are outcomes, not assumed here.

## Optional learned-GRU experiment

For both families, train a 24-hidden-unit TaskMemoryGRU with 6-symbol inputs and 64 task embeddings on the same 12-step binary output supervision as COMP-19, modified only by the new latch update label rule. 2,400 Adam steps, batch 128, LR=.01, length 12, clip norm 2.0. Report independently sampled native task accuracy at lengths 12 and 32. No early stopping or dropping poorly trained models.

Train three *equal 36-output* linear probes, each on the **same** frozen recurrent states from depths 4/8/12, 12 histories per goal per depth, Adam 400 updates at LR .04 with BCE:
- Wgoal: six canonical **ground-truth goal table bits** repeated six times (privileged goal supervision, clearly reported).
- Wdup: six native immediate-query bits repeated six times.
- W2: 36 native final responses to ordered two-query continuations.

Each probe has equal output dimension, optimizer and number of examples. Wgoal's privileged goal labels mean the comparison is an upper-bound reference for goal-only encoding, not a matched behavioral-supervision-only contrast.

Model seed assignments (calibration then held-out):
- repeat-toggle: calibration **2001,2002**; held-out **2003,2004**.
- successor-toggle: calibration **2011,2012**; held-out **2013,2014**.

For each seed evaluate state depths 12 and 40, 64 goals × 8 independent same-goal state pairs/depth. Both states in a pair receive independently sampled input histories. Eligibility: the two frozen models' immediate native six-query signatures match **and** their Wdup-predicted six-bit signatures (one out of each repeat block) match. Wgoal and W2 are not used to decide eligibility.

Diagnostic continuation: five common tokens per pair. Directional score S_P = sqrt((||O_P,A Δ||²+||O_P,B Δ||²)/(2×36)), where O stacks probe weight P transported by the same GRU recurrent Jacobians through diagnostic steps 0..5, Δ is the initial state displacement. Higher scores predict delayed native disagreement.

Outcome: two **independent, shared within pair** future sequences of length 32, with native six-query signature comparisons at steps 8,16,32. Event Y=1 if any inspected future signature differs. Diagnostic inputs and outcome inputs are independent. Report signed and absolute AUC(S_W2)-AUC(S_Wgoal), as primary, and AUC(S_W2)-AUC(S_Wdup) as secondary; no feature/threshold tuning on held-out data. Use 1,000 bootstrap replicates and 1,000 within-family/model/depth label permutations for a centered two-sided test. Report per-model and per-family results, event prevalence, and probe-teacher bit agreement.

Calibrate source FPR thresholds from pooled calibration **negatives** at the 90th percentile (NumPy higher method, strict >) separately for Wgoal, Wdup and W2. Commit numeric thresholds before fitting any held-out models. Apply unchanged to held-out families and report false-positive and true-positive fractions. No claim about operational transfer unless both family held-out FPRs for W2 are <=0.15.

Primary transfer support requires positive AUC improvement in both family-pooled comparisons and a positive paired-bootstrap 95% lower bound on pooled AUC difference. The study reports failure to learn the task or zero eligible positives as limitations, without replacing seeds or modifying the objective.

## Scientific limitations

The exact finite-state theorem is independent of any trained model. The trained-GRU portion is a test of whether learned recurrent dynamics and simple linear probes recover the finite distinction. Any positive result remains specific to known synthetic goals and intact model weights, and does not establish existence or recovery of unknown objectives in frontier models or autonomous persistence.
