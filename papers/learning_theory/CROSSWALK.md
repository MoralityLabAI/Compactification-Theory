# Classical learning theory × Compactification-Theory × EvalMath

This is an application map, with established mathematical ancestry distinguished from proposed neural experiments.

## PAC/VC, finite description and conditional generalization

Valiant (1984) and Vapnik–Chervonenkis (1971) define learning relative to a hypothesis class and a sample distribution. Rissanen (1978) and Blumer et al. (1987) motivate coding complexity. If the entire task-dependent predictor family has at most 2^D descriptions, for n independent bounded-loss examples and W predeclared intervention words, a standard finite union bound gives

\[
\sup_{f,w}|R(f,w)-\hat R(f,w)|
\leq \sqrt{\frac{D\ln 2+\ln(2|W|/\delta)}{2n}}.
\]

The encoder, adapter, routing and decoder must be counted. This may be numerically vacuous for large networks. A candidate fixed before independent evaluation permits a simpler pointwise bound. See [EvalMath IC-01](https://github.com/MoralityLabAI/EvalMath/tree/main/research/interventional_compactification).

## The minimum state needed for future work

Angluin (1987), Rivest and Schapire (1989), and Littman, Sutton and Singh (2001) connect active tests with hidden-state distinguishability. With operation maps F_a and task readout g, define

\[
x\sim x'\iff g(F_w(x))=g(F_w(x'))\ \text{for every admitted }w.
\]

The finite exact quotient has |Q| distinct classes; a fixed-length exact code requires at least ceil(log2 |Q|) bits. For finite horizons the relevant state is graded by remaining time. The intervention alphabet determines which futures count. The theorem is a classical automata/predictive-state construction with our explicit model-resource and intervention contracts.

## Compression and causal representations

Shannon, Solomonoff, Rissanen and Tishby give distinct ways to interpret a short code. The proposed intervention-preserving code must retain features relevant to future task readouts. In the Z31 shear fixture, five bits can encode the present sum (31 possibilities), while ten fixed bits are needed for joint present and post-shear outputs (961 possibilities). A ten-bit nominal budget is a necessary capacity statement, and a specific quantizer may still collide. Compare [Compactification-Theory PR #28](https://github.com/MoralityLabAI/Compactification-Theory/pull/28) and Geiger et al. (2025) for causal abstraction.

## Recurrence, optimization and physical cost

Robbins–Monro, Sutton, Watkins–Dayan, Schmidhuber and Hochreiter–Schmidhuber provide foundations for iterative fitting, temporal prediction and persistent state. RC-01 treats mutable bits, recurrent steps, physical memory and host/device traffic as separate resources. Its uniform symbolic restoration condition is conditional on actual one-step error and separation bounds, which a learned neural implementation must still satisfy.

## Learning a curriculum or optimizer

Baxter (2000) formalizes inductive-bias learning across tasks; Andrychowicz et al. (2016), Finn et al. (2017) and Graves et al. (2017) provide learned-optimizer, meta-initialization and learning-progress curricula. SIML-01's proposed experiment-policy learner needs new graph instances and matched resources to demonstrate the effect of measured self-diagnosis. Random and behavioral-only selectors provide controls.

## Continual learning and adaptive evaluation

Kirkpatrick et al. (2017) and Schwarz et al. (2018) motivate concrete retention baselines. SIML fixes a task-family potential within an epoch and keeps immutable anchors for earlier skills. Hoeffding (1963) gives fixed-n confidence calculations; Dwork et al. (2015) warns that repeated disclosure of evaluation scores makes selection adaptive. EvalMath extends this with exact feedback-risk toy calculations and filtration analyses. SIML's initial promotions use fresh paired blocks and spend a summable error allowance across all attempts. Those guarantees require a scorer that measures the declared skill and an independently enforced acquisition boundary. FAW/FRC are deferred.

## Immediate research obligations

1. Replace the finite tabular recurrent reference with a neural student and test held-out action compositions across independent training seeds.
2. Compare diagnostic-guided updates to random task choice and a learning-progress selector, with the same full compute.
3. Check transfer to new intervention families and changed task generators.
4. Plot capability versus task-dependent description bits and physical resource cost.
5. Keep a distinct untouched semantic-family exam after the curriculum stabilizes.
