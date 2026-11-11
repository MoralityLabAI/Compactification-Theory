# SHAFT-COMP-23: Information-matched behavioral factorization on fresh models

**Registration:** 2026-11-11, before training new models or evaluating new decoder outcomes. This protocol follows the exact 384-class behavioral gauge theorem in COMP-22 and its privileged-label decoding result. The experiment uses only local synthetic six-symbol GRUs with intact weights.

## Scientific question

Can a **structured factorized decoder**, trained exclusively on the **same native 36 two-query behavioral targets** as a direct decoder, predict two-query responses more accurately on fresh recurrent models?

The ideal finite behavior is B2(g,l,p)[a,b]=bit(g XOR 63*l,b) XOR T(p,a). The factorized decoder learns the effective six-bit table and a *latent distribution* over six possible previous tokens using gradients from B2 labels alone. It never sees goal, latch or previous-token labels during decoder fitting. Hidden previous-token logits need only be behaviorally identifiable, and the exact finite reference tests this claim.

## Model cohort

Two families: repeat_toggle T(p,a)=1[p=a] and successor_toggle T(p,a)=1[a=(p+1) mod 6]. The base model is the frozen COMP-21 architecture: RecurrentMemory('gru',6) with 24 hidden units, 64 goal embeddings, and the original two-layer nonlinear task readout. All **new** model seeds:

- repeat_toggle: calibration 2301, heldout 2302 and 2303.
- successor_toggle: calibration 2311, heldout 2312 and 2313.

Train all six base models using COMP-21 **aux_state** training: 3,200 Adam steps at LR 0.01, batch 128, length 12, task BCE + latch BCE + 0.25 previous-token CE, gradient clip 2, training-only auxiliary heads. Task IDs and register labels are available to the *base-model trainer*; the decoder experiment removes all privileged labels from **decoder training**. The base models, architecture, weight updates, data and task streams are unchanged within the prescribed recipe. Verify on 1,024 fresh length-12 sequences (gate >=90% bit accuracy) and separately on length-32 sequences. Keep every seed, including gate failures. Fit/evaluate a decoder only for gate-passing models; if fewer than three heldout gate passes, classify the probe experiment inconclusive.

## Three decoders and parameter controls

For each gated frozen base model, sample the **same 2,304 training states** from 64 goals ×12 independent contexts × depths 4/8/12. Labels are **only** the frozen native model's 36 binary outputs for ordered two-query continuations at each state. No oracle goal, latch, previous symbol, or canonical effective table is used for any train-time decoder update.

1. **Direct36-linear:** a 24→36 learned affine readout (900 trainable parameters), optimized by BCE.
2. **Factorized-900:** 24→24 tanh →12 logits (900 trainable parameters: 600+300). Its first six logits represent an effective-table Bernoulli distribution. Its remaining six logits represent a softmax distribution over previous tokens. Under the known fixed finite trigger, compute for every pair (a,b) the soft XOR probability s_b(1−t_a)+(1−s_b)t_a, where s_b=sigmoid(goal-logit_b) and t_a is the probability of T(p,a)=1 under the predicted previous-token distribution. Apply BCE to these 36 probabilities and the **same native B2 target**. The fixed trigger law, which is a symbolic inductive bias, is disclosed. No latent labels.
3. **Direct36-MLP control:** 24→14 tanh→36 logits (890 parameters) optimized by BCE. It is a closely parameter-matched nonlinear baseline to separate the effect of nonlinear decoder capacity from the transition-law factorization.

Training: Adam LR=0.04, **600 steps** for all three decoder types, independent fixed initialization derived from model seed, identical training-state and label tensors, no validation-based early stopping. Record all train losses and hashes. Report the factorized model's effective-table exact/bit correctness and previous-token classification accuracy as **posthoc diagnostics only**, using finite oracle state labels excluded from fitting.

## Fresh heldout evaluation

For each gated model evaluate natural states from **64 goals ×16 fresh contexts** at both recurrent depths 12 and 40 (1,024 states per depth, per model), with input histories seeded independently of probe training. Native frozen-model two-query signatures are the **primary target**; exact automaton signatures are the secondary target and an oracle ceiling.

Metrics: micro-bit and full 36-bit exact accuracy per model, depth, family and decoder, plus baseline task accuracy, finite oracle-vs-native mismatch, training-context count and hashes.

**Registered primary contrast:** pooled heldout **depth-40 micro-bit accuracy** of Factorized-900 minus Direct36-linear, using the same fresh evaluation states. This primary metric is chosen because COMPLETE 36-bit signatures were almost always incorrect for a direct linear probe in COMP-22. The comparisons against Direct36-MLP and against exact signature accuracy are prespecified secondary metrics.

Compute 1,000 stratified bootstrap samples of **state indices within model seed** at depth 40, reporting the percentile 95% CI for the primary difference and model-specific improvements. **Success criterion:** positive pooled bootstrap lower bound AND positive depth-40 differences for >=3 of the 4 heldout model seeds, among gate-passing models. Otherwise report negative/inconclusive without modifying the decoder architecture, training loss, orientation or seeds.

**Calibration gate:** train only calibration seeds first, record base accuracies and decoder results, and commit a calibration receipt before any heldout base model training. No threshold tuning is allowed from calibration outcomes; all hyperparameters, decoder families, the primary metric and eligibility are frozen here.

## Verification

Repeat each complete neural output deterministically using CPU PyTorch 2.10.0+cpu and NumPy 2.3.5, with 3 CPU threads and deterministic algorithms. Archive base checkpoint hashes, decoder hashes, numeric outcomes and full state-level counts, SHA-256 receipts and a separate exact finite Python+Lean result confirming the factorization and behavioral gauge. Light CI checks stdlib reference/tests, frozen numeric summaries, source syntax and the Lean kernel.

The study uses explicit synthetic tasks and intact trained weights. It provides no evidence of unknown objectives, covert agent persistence, off-device reconstruction or survival after model deletion.
