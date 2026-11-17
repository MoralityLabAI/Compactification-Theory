# SHAFT-COMP-20: Persistent temporal state beyond goal signatures

**Registered:** November 11, 2026. The protocol was committed as 9201423bc6b5895ed1773a15b683d385b9481b08 before running the experiments. The calibration thresholds were frozen in commit c9b352beb574a0a4c6ffec3dedd824c458dd4be0 before any held-out fitting.

**Verdict:** Exact finite-state temporal-state separation **established**. Registered learned-GRU transfer **failed**, with all eight neural models below the base task-capability gate.

This research uses isolated, in-memory synthetic task models with full weights retained. It contains no mechanism for real-device persistence, covert communication, unknown-goal retrieval or recovery after weight deletion.

## Scientific motivation

SHAFT-COMP-19 found a small predictive gain from a two-query probe over an equal-output-capacity immediate-response probe. In its task family, the response after two fresh queries can depend on a goal table whose representation is already in the state while forgetting the previous token before the query pair. COMP-20 asks whether the probe can extract **persistent causal temporal information beyond a goal code**.

## Exact finite state and behavioral quotient

The new task's state is (g,l,p), where g is one of 64 six-bit goal tables, l is a binary latch, and p is one of six previous tokens. For query q, the output uses the **current** latch:

\[
y((g,l,p),q)=\mathrm{bit}(g,q)\mathbin{\mathrm{XOR}}l.
\]

Only after output does the state update:

\[
(g,l,p)\mapsto(g,l\mathbin{\mathrm{XOR}}T(p,q),q).
\]

Two preregistered transition triggers were tested. Repeat-toggle uses T(p,q)=1 if q=p. Successor-toggle uses T(p,q)=1 if q=(p+1) mod 6.

The full state space has 64×2×6=768 states. The executable finite enumerator computes the goal-only code, every immediate binary query response, and the final binary response after all 36 ordered two-query sequences.

| Result | Repeat-toggle | Successor-toggle |
|---|---:|---:|
| Distinct goal-only signatures | 64 | 64 |
| Distinct immediate behavioral signatures | 64 | 64 |
| Distinct two-query final signatures | 384 | 384 |
| Distinct joint goal + immediate + two-query signatures | 768 | 768 |
| Same-goal, same-latch ordered pairs with different previous token | 3,840 | 3,840 |
| Such pairs distinguished after two queries | 3,840 | 3,840 |
| Such pairs separable after a chosen common 32-input continuation | 3,840 | 3,840 |

For any equal-g, equal-l states with distinct previous tokens p and p', a first query q=p in repeat-toggle, or q=(p+1) mod 6 in successor-toggle, flips exactly one latch. Afterwards both states have the same previous-symbol register, so further common queries cause identical latch updates and preserve the latch discrepancy. It remains observable after 32 steps.

A two-query signature alone retains a twofold ambiguity from simultaneously complementing the goal table and flipping the latch. Combining it with the goal code distinguishes all 768 states. The precise quotient structure is therefore:

\[
|S/{\sim_{\mathrm{goal}}}|=64,\quad
|S/{\sim_{\mathrm{current}}}|=64,\quad
|S/{\sim_{2}}|=384,\quad
|S/{\sim_{\mathrm{goal,2}}}|=768.
\]

The finite Python reference exhaustively checks each state and every one of the 3,840 ordered same-goal/latch previous-input pairs per family. Lean 4 checks concrete witness lemmas for immediate equivalence, two-query separation, and survival after a shared 32-token continuation for both families. The Lean witnesses are finite examples; all-pair enumeration remains a separate executable reference.

## Neural training protocol and failure

Eight newly seeded six-token GRUs, 24 hidden units each, were trained for **2,400 Adam steps** with batch 128 and length-12 target sequences. Each learned model sees a goal-table ID in its initial embedding, then must learn the latch and previous-input dynamics from binary output supervision. There was no early stopping or post-hoc seed replacement.

Two calibration seeds and two held-out seeds per family were preregistered:

| Seed | Family | Split | Fresh 12-step task accuracy | Fresh 32-step task accuracy |
|---|---|---|---:|---:|
| 2001 | Repeat | Calibration | 78.7% | 63.9% |
| 2002 | Repeat | Calibration | 69.5% | 58.3% |
| 2003 | Repeat | Held-out | 78.3% | 61.7% |
| 2004 | Repeat | Held-out | 68.3% | 57.6% |
| 2011 | Successor | Calibration | 66.1% | 56.1% |
| 2012 | Successor | Calibration | 68.5% | 57.5% |
| 2013 | Successor | Held-out | 74.0% | 60.5% |
| 2014 | Successor | Held-out | 82.3% | 62.9% |

**All eight models failed the preregistered 90% base-capability requirement.** Their empirical behavior differs substantially from the finite automaton, limiting what the probe evaluation can establish.

Three **36-output equal-capacity** linear probes were fitted to the same 2,304 frozen hidden-state training examples per model:

- Goal probe: the six true table bits repeated six times. Goal-ID supervision is privileged, clearly distinct from behavioral probe supervision.
- Immediate-duplicate probe: each of the model's six native one-query behavioral responses repeated six times.
- History probe: all 36 native final behavioral responses to ordered two-query continuations.

All probes used BCE and 400 Adam updates with separately fixed initializations. A pair was eligible when its immediate native six-query outputs and duplicate-probe representative query bits agreed. The probes' directional scores used independent five-token diagnostic trajectories and a common initial state difference. Delayed divergence was detected on either of two separate shared input sequences at steps 8, 16, or 32.

### Registered held-out prediction

| Metric | Goal probe | Immediate-duplicate probe | History probe |
|---|---:|---:|---:|
| ROC-AUC | **0.492745** | 0.449681 | **0.476740** |
| Frozen source threshold | 1.193325 | 2.333848 | 2.246707 |
| False positives / 816 negatives | 426 | 210 | 201 |
| True positives / 200 native divergences | 99 | 62 | 64 |
| False-positive rate | 52.2% | 25.7% | 24.6% |
| True-positive rate | 49.5% | 31.0% | 32.0% |

Across 4,096 generated held-out state pairs, 1,016 satisfied the eligibility constraint and 200 exhibited native delayed divergence. The **primary history-minus-goal AUC was −0.016005**. The registered 1,000-resample within-model bootstrap 95% interval is **[−0.071810,+0.041796]**; centered within-family/model/depth 1,000-permutation **p=0.559441**. The registered neural-transfer support criterion failed.

At the original frozen threshold, history FPR reached **24.63%**, exceeding the preregistered maximum 15% in the pooled held-out test. Repeat-toggle alone reached 71.43% false positives while successor-toggle reached 7.68%, showing substantial cross-family score-scale instability.

### Within-family results

| Family | Eligible | Native divergences | Goal AUC | History AUC | Difference |
|---|---:|---:|---:|---:|---:|
| Repeat-toggle | 325 | 108 | 0.616103 | 0.414661 | −0.201442 |
| Successor-toggle | 691 | 92 | 0.510017 | 0.431770 | −0.078247 |

The pooled AUC difference is smaller in magnitude because cross-family positive-negative ranking comparisons depend on the score scales and prevalence distributions of both families.

### Intended versus learned dynamics

For the held-out sampled histories, the exact finite automaton predicts delayed divergence on **699 eligible pairs**. The trained models show **200 native divergences**, with **37.3%** binary agreement on which pairs diverge. The trained system therefore provides poor fidelity to the causal transition under study. This is a learning-capability failure, and the neural AUC result should be read in that context.

## Reproducibility

The SHAFT-COMP-20 branch includes the finite enumerator, formal Lean witnesses, the original training and probe script, calibration threshold commitment, frozen numerical summary, and a standard-library independent Mann–Whitney AUC audit.

Reproduction from the repository root:

1. Run the finite reference with: python -m research.shaft_compactification.persistent_latch_finite
2. Run the frozen AUC audit with: python -m research.shaft_compactification.comp20_audit
3. Run standard-library tests with: python -m unittest discover -s research/shaft_compactification/tests -v
4. Regenerate held-out model seed 2003 using the persistent_goal_control.py script with family repeat_toggle, phase heldout, seed 2003, and the SHA-256-locked calibration threshold JSON. Use the same procedure for the other seven registered seeds.
5. Merge the eight per-model output JSON files with merge_comp20.py. The full raw receipts, model and probe digests, and reproduced seed-specific outputs are packaged in the conversation archive.

The calibration receipt was frozen before held-out evaluation. A separate independent replay of held-out seed 2003, and another of seed 2013, reproduced byte-identical JSON results. The complete eight-model replay was interrupted by execution timeouts, and the report makes no full-replay assertion.

## Consequence for future tests

The finite proof shows exactly what a genuine temporal-state code must distinguish. Recovering that code in a trainable neural agent requires a model that first learns its intended latch and previous-input transitions reliably. A separate registered follow-up could introduce supervised auxiliary latch and previous-symbol heads, with untouched original training controls and new holdout seeds, then test whether history-aware probes retain an advantage once the base model meets the capability gate.

This result is a precise finite-state causal distinction accompanied by a failed neural instantiation under the preregistered recipe.
