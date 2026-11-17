# Learning theory: primary papers and mathematical bridges

32 annotated sources, from Shannon (1948) through modern deep-learning theory. Bibliographic links were checked against author, publisher or archival metadata on 2026-11-02. The notes are original commentary, and source publications remain with their rights holders.

Read the [cross-repository mathematical crosswalk](CROSSWALK.md), [paper-by-paper notes](READING_NOTES.md), [machine-readable catalog](catalog.json) or [BibTeX](REFERENCES.bib).

## Information theory

- [**A Mathematical Theory of Communication**](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) (1948; Claude E. Shannon). Code length and capacity count transmitted information; this does not itself prove learnability.

## Optimization

- [**A Stochastic Approximation Method**](https://doi.org/10.1214/aoms/1177729586) (1951; Herbert Robbins and Sutton Monro). Source for iterative noisy optimization under explicit assumptions.

## PAC, VC and learnability

- [**On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities**](https://doi.org/10.1137/1116025) (1971; Vladimir N. Vapnik and Alexey Ya. Chervonenkis). Connect finite hypothesis classes and function covers to generalization.
- [**A Theory of the Learnable**](https://doi.org/10.1145/1968.1972) (1984; Leslie G. Valiant). Specify the learner's sampling and computational protocol for every task.
- [**The Strength of Weak Learnability**](https://doi.org/10.1007/BF00116037) (1990; Robert E. Schapire). Evaluate proposed weak-skill composition on untouched task families.
- [**Mathematical Theory of Deep Learning**](https://arxiv.org/abs/2407.18384) (2024; Philipp Petersen and Jakob Zech; modern bridge). Approximation theorems require proven regularity for the neural quotient.
- [**A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models**](https://doi.org/10.1146/annurev-statistics-040522-013920) (2025; Namjoon Suh and Guang Cheng; modern bridge). Distinguish approximant existence from what optimization actually learns.

## Compression and inductive inference

- [**A Formal Theory of Inductive Inference, Part I**](https://doi.org/10.1016/S0019-9958(64)90223-2) (1964; Ray J. Solomonoff). Algorithmic descriptions and priors need separate physical compute accounting.
- [**Modeling by Shortest Data Description**](https://doi.org/10.1016/0005-1098(78)90005-5) (1978; Jorma Rissanen). Description cost includes encoder, adapter, routing, decoder and advice.
- [**Occam's Razor**](https://doi.org/10.1016/0020-0190(87)90114-1) (1987; Anselm Blumer and Andrzej Ehrenfeucht and David Haussler and Manfred K. Warmuth). Small consistent hypotheses must count the entire task-dependent description.
- [**The Information Bottleneck Method**](https://arxiv.org/abs/physics/0004057) (1999; Naftali Tishby and Fernando C. Pereira and William Bialek). Future intervention targets change which information the bottleneck must preserve.

## Automata, predictive states and recurrence

- [**Learning Regular Sets from Queries and Counterexamples**](https://doi.org/10.1016/0890-5401(87)90052-6) (1987; Dana Angluin). Use shortest separating intervention words as active counterexamples.
- [**Inference of Finite Automata Using Homing Sequences**](https://doi.org/10.1145/73007.73047) (1989; Ronald L. Rivest and Robert E. Schapire). Identify distinct latent states through controlled action sequences.
- [**Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks**](https://doi.org/10.1162/neco.1992.4.1.131) (1992; Jürgen Schmidhuber). Fast internal memory may avoid constant large-backbone updates.
- [**Long Short-Term Memory**](https://doi.org/10.1162/neco.1997.9.8.1735) (1997; Sepp Hochreiter and Jürgen Schmidhuber). Compare learned recurrent retention with explicit symbolic memory.
- [**Predictive Representations of State**](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) (2001; Michael L. Littman and Richard S. Sutton and Satinder Singh). Direct antecedent to action-conditioned predictive state in IRC-02.

## Online and reinforcement learning

- [**Learning Quickly When Irrelevant Attributes Abound: A New Linear-threshold Algorithm**](https://ai.stanford.edu/~pabbeel/depth_qual/littlestone1988.pdf) (1988; Nick Littlestone). Mistake bounds inspire sequential evaluation of a curriculum policy.
- [**Learning to Predict by the Methods of Temporal Differences**](https://doi.org/10.1007/BF00115009) (1988; Richard S. Sutton). Train recurrent predictions of future outcomes in addition to immediate outputs.
- [**Q-Learning**](https://doi.org/10.1007/BF00992698) (1992; Christopher J. C. H. Watkins and Peter Dayan). A tabular learning benchmark for adaptive control with a fixed environment.
- [**A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting**](https://doi.org/10.1006/jcss.1997.1504) (1997; Yoav Freund and Robert E. Schapire). Multiplicative-weight allocation supplies a curriculum scheduler baseline.

## Representation and distillation

- [**Representation Learning: A Review and New Perspectives**](https://doi.org/10.1109/TPAMI.2013.50) (2013; Yoshua Bengio and Aaron Courville and Pascal Vincent). Feature quality and disentanglement require a transfer experiment.
- [**Distilling the Knowledge in a Neural Network**](https://arxiv.org/abs/1503.02531) (2015; Geoffrey Hinton and Oriol Vinyals and Jeff Dean). A student's distillation fidelity needs fresh compositional evaluation.
- [**Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**](https://www.jmlr.org/papers/v26/23-0058.html) (2025; Atticus Geiger and others; modern bridge). Require commutation under a declared intervention grammar and controlled alignment maps.

## Meta-learning and curricula

- [**A Model of Inductive Bias Learning**](https://doi.org/10.1613/JAIR.731) (2000; Jonathan Baxter). Judge learned experiment policy on new task-graph instances.
- [**Learning to Learn by Gradient Descent by Gradient Descent**](https://papers.nips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf) (2016; Marcin Andrychowicz and Misha Denil and Sergio Gómez Colmenarejo and Matthew W. Hoffman and David Pfau and Tom Schaul and Brendan Shillingford and Nando de Freitas). Compare learned optimizer and experiment chooser at matched total compute.
- [**Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**](https://proceedings.mlr.press/v70/finn17a.html) (2017; Chelsea Finn and Pieter Abbeel and Sergey Levine). Meta-initialization and adaptation speed are alternative experimental controls.
- [**Automated Curriculum Learning for Neural Networks**](https://proceedings.mlr.press/v70/graves17a.html) (2017; Alex Graves and Marc G. Bellemare and Jacob Menick and Rémi Munos and Koray Kavukcuoglu). Behavioral learning-progress scheduling is the principal SIML baseline.

## Continual learning

- [**Overcoming Catastrophic Forgetting in Neural Networks**](https://doi.org/10.1073/pnas.1611835114) (2017; James Kirkpatrick and others). Parameter-importance retention compared with immutable skill anchors.
- [**Progress & Compress: A Scalable Framework for Continual Learning**](https://proceedings.mlr.press/v80/schwarz18a.html) (2018; Jonathan Schwarz and others). Continual consolidation is an alternative to unlimited adapter growth.

## Evaluation and adaptive validity

- [**Probability Inequalities for Sums of Bounded Random Variables**](https://doi.org/10.1080/01621459.1963.10500830) (1963; Wassily Hoeffding). Fixed-sample concentration for paired fresh promotion blocks.
- [**Generalization in Adaptive Data Analysis and Holdout Reuse**](https://arxiv.org/abs/1506.02629) (2015; Cynthia Dwork and Vitaly Feldman and Moritz Hardt and Toniann Pitassi and Omer Reingold and Aaron Roth). Repeated score disclosures need an explicit information budget.
- [**Fantastic Generalization Measures and Where to Find Them**](https://research.google/pubs/fantastic-generalization-measures-and-where-to-find-them/) (2020; Yiding Jiang and Behnam Neyshabur and Hossein Mobahi and Dilip Krishnan and Samy Bengio). Validate feature rank and other diagnostics before treating them as predictive.
