# SHAFT-COMP-23: Behavioral-only factorization of persistent temporal state

**Preregistered November 11, 2026.** [Protocol](SHAFT_COMP_23_PROTOCOL.md), commit \`fae6c5c6843ee8f2a8f85686012ee559dc37d281\`. **Calibrated before heldout:** frozen commit \`5bb2e00777aeea7a88b5b2e374abfa21d3892647\`, with independent full local calibration SHA-256 \`730efbe3a5d5cb9cd772171988e32a8bac5a2e2b6a0cd79ed55331c44e36fbc4\`. Both calibration base models passed the prespecified 90% native task-accuracy gate.

**Registered primary result:** across four *fresh* heldout GRUs, the behavioral-only 900-parameter structured decoder reaches **89.3643% native two-query bit accuracy** at depth 40, while the 900-parameter direct linear decoder reaches **51.7225%**. The pooled difference is **+37.6418 percentage points**, model-stratified 1,000-bootstrap 95% interval **[+37.1127,+38.1702] percentage points**. Every heldout seed favors structured decoding. The registered support criterion is met.

A third, nonlinear direct-decoder control with 890 trainable parameters achieves **75.2157%** bit accuracy. The structured decoder's improvement over this control is also positive (95% bootstrap interval **[+13.7586,+14.5264] percentage points**). The result therefore extends beyond a pure linear-versus-nonlinear capacity contrast, although its known transition-law architecture remains a decisive inductive bias.

This experiment uses synthetic six-query tasks, intact 24-unit GRU weights and a known finite transition law. It does not measure hidden objectives in frontier models, autonomous device behavior, or weight restoration after deletion.

## 1. Mathematical question

SHAFT-COMP-22 found an exact compact quotient of the persistent-latch model but its stronger supervised decoder used *privileged goal, latch, and previous-token labels*. COMP-23 asks whether those same quantities, up to observational gauge, can be **inferred from native two-query behavioral targets alone**.

For goal table g∈{0,...,63}, binary latch l, and previous query p∈{0,...,5}, define the effective behavioral table

\[
b=g\oplus(63\cdot l).
\]

For ordered future queries (a,c), the exact final response is

\[
\boxed{B_2(a,c)=b[c]\oplus T(p,a).}
\]

The registered triggers are \`repeat_toggle\` with T(p,a)=[a=p] and \`successor_toggle\` with T(p,a)=[a=(p+1) mod 6]. For a fixed second query c, precisely **one** of the six first-query rows complements b[c]; five retain it. A majority vote across the rows recovers b[c], and the row of complements identifies p (via the inverse successor mapping where relevant).

This yields a **constructive inversion from 36 behavioral bits to the effective six-bit table and previous token**. In each family, the Python reference enumerates all 768 canonical (g,l,p) states, recovering all 384 distinct behavioral codes and reconstituting the complete 36-bit signature. Canonical goal/latch complement symmetry remains unidentifiable, as it should.

[BehaviorOnlyIdentification.lean](lean/Shaft/BehaviorOnlyIdentification.lean) proves that the majority of each column recovers its effective goal bit, that the uniquely complemented row recovers the previous query, and that the resulting decoder is a left inverse of the two-query observation map. It also establishes injectivity on behavioral codes. The Lean theorems concern exact finite behavior. Neural decoding is a separate empirical hypothesis.

## 2. Information-matched learned decoders

All three trainable heads receive **the same 2,304 frozen hidden-state examples and the same 36 native behavioral bits as their only supervision**, with 600 Adam updates at learning rate 0.04. Training states consist of 12 independent histories per each of 64 goals at depths 4,8,12. No canonical goal, latch, previous-token, or effective-table label enters any decoder's gradient computation.

| Learned decoder | Architecture | Parameters | Training loss |
|---|---|---:|---|
| Direct36 linear | 24→36 affine logits | 900 | BCE on native 36 binary responses |
| Factorized900 | 24→24 tanh→12 latent logits | 900 | BCE on native 36 binary responses, via differentiable trigger-law mixture |
| Direct36 nonlinear | 24→14 tanh→36 logits | 890 | BCE on native 36 binary responses |

Factorized900's first six logits parameterize Bernoulli probabilities s_c for each effective goal bit. Its second six logits parameterize a softmax distribution over previous query p. For any first query a, the trigger probability t_a is the appropriate marginal of the predicted previous-token distribution. Its differentiable two-query output probability is

\[
P(\widehat B_2(a,c)=1)=s_c(1-t_a)+(1-s_c)t_a.
\]

This is the Bernoulli XOR mixture. At evaluation, the 36 probabilities are thresholded at 1/2. The latent goal and previous-token estimates are audited against oracle labels *only after evaluation*. The known trigger law and its hard-coded two-row factorization are inductive biases; information matching here refers to labels and training examples, not to a completely unconstrained model class.

**Important base-model supervision boundary:** the six recurrent models were themselves trained under COMP-21's **aux_state** recipe, using privileged latch and previous-token labels in training-only heads. These heads are absent from decoder fitting. The experiment isolates privileged **decoder** supervision, while conditioning on recurrent representations shaped by privileged **base-model** training.

## 3. New base models and reproducibility gate

Both calibration and all four heldout seeds are new. Base model training follows the frozen COMP-21 recipe for 3,200 Adam steps, with goal embeddings and 24 recurrent hidden units.

| Model | Family | Fresh native accuracy at 12 steps | Native accuracy at 32 steps | Gate |
|---|---|---:|---:|---|
| 2301 | Repeat, calibration | 96.94% | 96.15% | Passed |
| 2311 | Successor, calibration | 99.70% | 99.51% | Passed |
| 2302 | Repeat, heldout | 99.76% | 99.62% | Passed |
| 2303 | Repeat, heldout | 99.92% | 99.88% | Passed |
| 2312 | Successor, heldout | 91.90% | 90.57% | Passed |
| 2313 | Successor, heldout | 99.87% | 99.85% | Passed |

Both calibration models were trained and their full receipts SHA-256 pinned **before any heldout training began**. The heldout experiment script requires the byte-exact frozen calibration receipt and its SHA-256 on command line before fitting a heldout model. No model or seed was replaced based on performance. In each heldout model, test inputs use fresh independent natural histories at depths 12 and 40, with 16 histories per goal per depth (1,024 fresh states at each depth).

The trained experiment script on GitHub is byte-identical to the executable used locally: Git blob SHA \`d521b96255d3bc06e7771048a70325b094260aaf\`, source SHA-256 \`17bf025a2e24f01c78d0bf11590e04a24b20d268f55ed4469af7813b8f39736f\`.

## 4. Neural heldout results

### Prespecified primary, recurrent depth 40

| Decoder | Correct native bits / 147,456 | Native micro bit accuracy | Complete native 36-bit signature correct / 4,096 |
|---|---:|---:|---:|
| Direct linear (900 parameters) | 76,268 | 51.7225% | 0 |
| Direct nonlinear (890 parameters) | 110,910 | 75.2157% | 0 |
| **Behavioral-only factorized (900 parameters)** | **131,773** | **89.3643%** | **1,698 (41.4551%)** |
| Exact automaton oracle (zero fitted parameters) | 143,165 | 97.0900% | 3,059 (74.6826%) |

The primary bitwise gain is +0.376418. Model-stratified bootstrap 95% CI: [0.371127,0.381702]. The nonlinear direct control achieves considerably better bit accuracy than the direct linear control, yet it also fails to construct any full 36-bit signature correctly in the sample. The factorized decoder improves on both without additional supervision labels.

### Each heldout model, depth 40

| Seed | Family | Direct linear bits | Factorized bits | Direct nonlinear bits | Factorized full signatures |
|---|---|---:|---:|---:|---:|
| 2302 | Repeat | 52.4604% | 90.8122% | 76.5842% | 50.4883% |
| 2303 | Repeat | 50.4313% | 89.5237% | 74.8454% | 49.7070% |
| 2312 | Successor | 53.1548% | 88.7722% | 73.7657% | 19.7266% |
| 2313 | Successor | 50.8436% | 88.3491% | 75.6673% | 45.8984% |

The factorized-minus-linear bit-accuracy gains are +0.383518, +0.390923, +0.356174 and +0.375054 respectively. All four heldout models meet the sign condition in the preregistered support rule. The comparatively weak successor seed 2312 barely passes the base capability gate and has the lowest full-signature accuracy. These seed differences limit generalization.

### Additional depth and latent diagnostics

At **depth 12**, factorized bit accuracy is 89.5013%, direct linear 51.6093%, direct nonlinear 75.5052%, and oracle 97.3151%. Factorized full-signature accuracy is 44.1650%, compared with one full signature each for the two direct controls.

At **depth 40**, the factorized model's inferred previous-token classifier agrees with the oracle previous-token register in about **99.27%** of cases; its inferred *entire* effective six-bit goal table agrees in about **46.39%** of states. Latent agreement is an audit against privileged test labels that were unseen during decoder fitting.

The exact automaton yields 97.09% native bit accuracy and 74.68% full-signature agreement, showing that the frozen GRUs still differ from their ideal tasks in some states, especially after 40 steps. This finite-oracle ceiling is distinct from decoder error.

## 5. Receipts and audits

- [Preregistered protocol](SHAFT_COMP_23_PROTOCOL.md).
- [Frozen calibration receipt](results/comp23_calibration_freeze.json) and full calibration SHA-256 noted above.
- [Executed training/evaluation code](experiments/behavior_only_factorization.py).
- [Exact constructive behavioral-only inverse](behavior_only_finite.py), [tests](tests/test_behavior_only_finite.py), and [Lean proof](lean/Shaft/BehaviorOnlyIdentification.lean).
- [Frozen numerical results](results/comp23_summary.json) and [standard-library count audit](comp23_audit.py).

All six local model experiments were independently replayed from the same new seeds, **with byte-identical original and replay JSON for every one**. The full raw per-state result rows, both full executions, the original model checkpoints and hashes are preserved in a conversation ZIP archive. The compact GitHub receipt records the complete per-model SHA-256 hashes, aggregate counts, and the within-model bootstrap.

The main test has 4,096 fresh heldout states at depth 40, equivalent to 147,456 binary response decisions. The confidence interval resamples states **within each of four fixed model seeds**. It is not a confidence interval across potential neural architectures or task families.

This does not establish an unknown latent goal in a frontier model or any cross-device persistence. It provides an interpretable example of how a known transition law helps train a compact decoder under exactly matched behavioral supervision.

## Reproduction

From the repository root with CPU PyTorch 2.10.0 and NumPy 2.3.5:

\`\`\`bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.behavior_only_finite
python -m research.shaft_compactification.comp23_audit
python -m research.shaft_compactification.experiments.behavior_only_factorization \
  --family repeat_toggle --seed 2301 --phase calibration \
  --out /tmp/comp23_cal_repeat.json
\`\`\`

Run the successor calibration seed 2311, freeze the full calibration SHA-256 receipt, and only then run heldout seeds 2302,2303,2312,2313 with the \`--calibration-freeze\` and \`--freeze-sha256\` arguments. The accompanying ZIP archive provides the original calibration receipt and executable replay commands. Light CI checks the standard-library reference, frozen aggregate counts, experiment Python syntax and Lean 4.28 kernel. Full CPU retraining is intentionally opt-in.

## Next falsifier

A new task family with **multiple trigger rows**, a larger alphabet, or an unknown trigger function would test the dependence on the supplied transition-law inductive bias. Further, a new LSTM task model could test whether this behavioral-only factorization transfers across recurrent architectures. These would need new preregistered seeds, and any adaptive method for discovering a trigger law would have to earn its own heldout test.
