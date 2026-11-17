# SIML-01: self-introspective meta-learning across a task graph

**Draft 0.1 | 2026-11-02 | Compactification-Theory**

This draft turns our compactification mathematics into a learning procedure. A controller inspects measurements of its learner, predicts which small change will help, trains a bounded candidate, and tests that prediction on fresh task instances. Accepted candidates become the next incumbent. The controller also learns from its unsuccessful experiments.

The intended first proposer is a configured local 27B model. Smaller recurrent learners, routed adapters and task-state encoders provide the initial editable components. The same backbone can eventually serve as proposer and learner through separate checkpoint snapshots. Checkpoint hashes and runtime compatibility remain run-manifest fields.

**FAW and FRC are deferred.** This version covers capability learning and the validity of the measurements used to steer it. The evaluator assumes honest acquisition, a fixed task specification and access separation. Scalable oversight and adversarial evaluator compromise need a subsequent protocol.

Status: research proposal with an executable finite-state CPU reference under [RUNNER.md](RUNNER.md). [SKILL.md](SKILL.md) is the operator entrypoint. [microlearner.py](microlearner.py) fits a tiny recurrent transition-table student to supervised one-step data; [promotion.py](promotion.py) runs the bounded promotion/retention ledger. [gate_math.py](gate_math.py) provides arithmetic checks. Neural training, a 27B proposer and a separately credentialed evaluator remain future integrations. The reference cannot establish whether evaluation data were acquired independently.

## 1. The research question

Can measured self-diagnosis help a learner choose updates that produce more held-out transfer per unit of total compute?

The proposed mechanism has an explicit causal chain:

`measured learner state -> diagnosis -> predicted update effect -> bounded training -> independently measured effect -> updated experiment policy`

For example, a collision witness may show that a code has discarded a variable needed after an intervention. The controller can allocate an additional state coordinate. When the information is already recoverable but its learned transition fails to compose, successor consistency becomes a different candidate intervention. A long-horizon numerical failure suggests a precision or restoration experiment.

The experimental claim concerns improved choices by this controller. A compelling explanation, a better probe score, and a higher development score each receive their own records. Promotion depends on the frozen task contract and fresh measurements.

## 2. Mathematical dependencies and ownership

Use the following inspected versions. Existing experiments keep their own protocols; this folder changes none of their sources.

| Source | Imported content | Role here |
|---|---|---|
| [EvalMath IC-01](https://github.com/MoralityLabAI/EvalMath/blob/957ab08d87c28792f8bea1770156ea14511b3182/research/interventional_compactification/RESEARCH.md) | Behavioral quotient, fiber obstruction, simulation bound and covering-number certificate | Specify state sufficiency and design diagnostic tests |
| [EvalMath evaluation calculus](https://github.com/MoralityLabAI/EvalMath/blob/957ab08d87c28792f8bea1770156ea14511b3182/paper/evaluation_design_calculus.md) | Typed game, controller, sampling, acquisition, judgment, aggregation and claim | Define each task node and each graph edit |
| [EvalMath selection risk](https://github.com/MoralityLabAI/EvalMath/blob/957ab08d87c28792f8bea1770156ea14511b3182/paper/selection_risk_continuations.md) | Post-selection risk and continuation-sufficient audit state | Account for score disclosures and retained provenance |
| [EvalMath IRC-02](https://github.com/MoralityLabAI/EvalMath/blob/957ab08d87c28792f8bea1770156ea14511b3182/paper/interface_relative_compactification.md) | Timed predictive equivalence and finite history-code bounds | Separate interface limits from learner errors |
| [CT RC-01](https://github.com/MoralityLabAI/Compactification-Theory/blob/bd1aabd2d58d074f1ea4c2f35ecee151b093abf7/research/recurrent_circuits_v1/THEORY.md) | Description/state/resource ledger and conditional robust recurrence | Charge the complete system and investigate long-horizon failures |
| [CT task closure](https://github.com/MoralityLabAI/Compactification-Theory/blob/6594586c74b38ce6cd410c56e4e087a8742d801c/research/dynamical_closure/TASK_CLOSURE.md) | Exact task quotient and monotone fixed-target Bayes risk along nested codes | Measure when a specified compression path loses relevant information |
| [CT transfer](https://github.com/MoralityLabAI/Compactification-Theory/blob/6594586c74b38ce6cd410c56e4e087a8742d801c/research/dynamical_closure/TRANSFER_AND_GENERALIZATION.md) | Density-ratio and changed-target bounds | Limit claims across graph edges |
| [CT neural recoverability](https://github.com/MoralityLabAI/Compactification-Theory/blob/6594586c74b38ce6cd410c56e4e087a8742d801c/research/dynamical_closure/NEURAL_RECOVERABILITY.md) | Exploratory task-only and successor-consistency comparison | Seed one update family and its controls |
| [CT shear](https://github.com/MoralityLabAI/Compactification-Theory/blob/6594586c74b38ce6cd410c56e4e087a8742d801c/research/dynamical_closure/SHEAR_CLOSURE.md) | Present/future distinction with an exact bit requirement | Test whether diagnostics notice a newly relevant hidden coordinate |
| [CT Pluto-Gym](https://github.com/MoralityLabAI/Compactification-Theory/blob/bd1aabd2d58d074f1ea4c2f35ecee151b093abf7/research/pluto_gym/PROTOCOL.md) | Existing curriculum interface, checkpoints and resource records | Supply an eventual execution adapter |

The PR #28 sources are pinned to an open branch. This draft imports their statements by reference. General mathematical implementations remain owned by EvalMath. The new contribution proposed here is a task-graph training protocol with a measurable self-model and an adaptive promotion contract. The concentration tools and quotient constructions have established precedents.

## 3. What the graph contains

A node is an evaluation contract

\[
v=(G_v,\Pi_v,P_v,Z_v,J_v,\mathcal A_v,C_v;\mathcal U_v,H_v,\mathcal R_v).
\]

The first seven entries follow EvalMath's evaluation calculus. They fix the world, permitted controller interface, sampling law, acquired evidence, scorer, uncertainty procedure and intended claim. The remaining entries specify interventions, horizon and resource limits. Store generator and scorer hashes, semantic family identity, reset rules, loss scale and sampling unit.

A graph has several edge types. A prerequisite edge is a testable claim about useful preparation. Transfer edges store signed estimates from contemporaneous comparisons. Composition uses a hyperedge when several parent skills are required. Interference edges record where an update damages another competence. Interface-expansion edges change which observations or operations are available. Restrict the prerequisite subgraph to a DAG; the transfer graph can contain cycles.

Difficulty has several coordinates. Increasing horizon and adding a new intervention are separate edits. An edge must say which coordinate changed and which semantics remain fixed. A known quotient can certify an information requirement in an exact fixture. An unmeasured transfer edge remains a hypothesis.

Task families receive weights at the family level. Ten paraphrases of the same rule share that family's budget. The controller can propose new nodes, whose status starts as `exploratory`. An epoch's primary target distribution stays frozen. Admitting a new family creates a new graph version and a new baseline report while carrying forward the evaluation-attempt counter and retention registry.

### Initial graph

| Node | Task contract | Held-out axis |
|---|---|---|
| T0 | Eight-state hidden-bit world with present and reveal targets | Fresh instances and designated reveal words |
| T1 | Compositions of T0's known primitives | Whole action compositions |
| T2 | Longer recurrent execution | Lengths beyond training, with reset and precision controls |
| T3 | Affine binary circuits with parity, shifts and writes | Operation sets and circuit instances |
| T4 | Modular addition with shear | Joint present/future prediction under fixed code budgets |
| T5 | Symbolic Storyworld rules with executable consequences | Rule combinations, relabelings and delayed effects |
| T6 | Routed modules solving conjunctions of learned skills | New module compositions and routing contexts |
| T7 | Partially observed, timed planning via an IRC/Pluto adapter | Interface modes and deadline combinations |

T0 -> T1 -> T2 is the first local path. T3 and T4 provide independent capacity diagnostics. T5 then tests a richer symbolic domain. The T5/T2 -> T6 composition edge is a hypothesis. T7 waits for an adapter that preserves its declared observation and timing contract. These nodes describe proposed contracts; this folder supplies no generators or model outcomes for them.

## 4. The learner and its measured self-model

Let the deployed learner be

\[
\Theta=(W,\phi,\psi,\rho,\pi,q,\mathcal M),
\]

where W is the backbone, phi the adapters, psi recurrent modules, rho routing, pi the state encoder, q readout, and M persistent memory. Any component that changes predictions belongs to the candidate identity. Include tokenizer, decoding, tool versions, context, precision, optimizer state and random-state semantics when relevant. A hidden-state boundary must carry the KV cache, position and phase needed to define its transition.

A self-model record is a measured feature vector plus forecasts. Record source hashes and confidence with each entry. Suggested diagnostics include held-out development loss by family, one-step closure residual, multi-step composition error, near-fiber witnesses, quantization sensitivity, effective gradient rank at declared tolerances, intervention responses, forgetting, and measured cost. Full descriptions of encoders and probes are charged to the resource ledger.

Before an update, the proposer writes a structured record:

```text
incumbent_hash, graph_version, target_node
hypothesis_id, diagnostic_receipts, competing_explanation
update_operator, affected_components, training_budget
predicted_primary_gain, predicted_transfer, predicted_retention
predicted_cost, uncertainty, discriminating_test
```

Measure forecast quality on subsequent development outcomes with a proper score, such as squared error for a bounded predicted gain. Log every attempted update, including rejected candidates. Predictions about untried updates require randomized shadow experiments or another identified design; an archive of chosen actions alone does not establish their counterfactual value.

Self-diagnosis can read instrumented activations only when the runtime exposes them. A black-box proposer receives the diagnostics supplied by the external collector. Its prose cannot manufacture a missing measurement.

### Diagnosis-to-experiment map

| Observed evidence | Candidate experiment | Required control |
|---|---|---|
| Same code, different relevant futures | Add a state coordinate or change the encoder | Equal total code-bit budget alternative |
| Initial state decodes; successor composition fails | Add state-consistency loss or repair the transition | Task-only training with matched updates |
| Low adapter feature diversity at the chosen site | Move or spread a fixed parameter budget | Same data and training work |
| Error grows with recurrent horizon | Change precision, restoration or memory refresh | Matched logical work and complete traffic accounting |
| A dormant variable appears after an intervention | Extend the task's represented state | Present-only and intervention-aware versions |
| An update damages an earlier family | Add replay, isolate a module or revise routing | Same total storage and optimizer budget |
| Performance sits at the interface oracle ceiling | Propose a separately versioned interface task | Same-interface comparison retained |
| A diagnostic fires before any learning | Run random-init and shuffled-feature falsifiers | Identical diagnostic fitting budget |

## 5. How the compactification mathematics guides an update

### 5.1 State sufficiency and capacity

For fixed observable g and operation grammar U, define

\[
h\sim h'\iff g(F_w(h))=g(F_w(h'))\quad\forall w\in\mathcal U^*.
\]

On a finite declared domain, exact state coding requires

\[
B_{\rm live}\ge\lceil\log_2|Q|\rceil.
\]

Finite-horizon versions are horizon-graded: a transition consumes remaining horizon. A fixed H quotient becomes an autonomous transition state only with the necessary closure or clock structure. Adding allowed operations can refine the quotient even when the present output stays unchanged.

In the shear fixture, h(a,b)=a+b and S(a,b)=(a,a+b) over Z_31. The pair (h,h after S)=(s,s+a) identifies all 961 states. Current output needs five bits; joint current/future coding needs ten. The unrestricted Bayes decoder is an information oracle with its own complexity. A learned decoder must earn its own held-out result.

For nested codes C_k and fixed target Y, squared Bayes risk

\[
R(C_k;Y)=\mathbb E\operatorname{Var}(Y\mid C_k)
\]

can only increase under deterministic coarsening. Fix the measure, target scale and future-word family before assigning a threshold k*. A newly relevant latent variable can change the threshold. The pilot modular-addition warning remained active at initialization and the recorded runs failed to grok; this motivates the random-init control, with source [CYCLIC_GROKKING_RESULTS](https://github.com/MoralityLabAI/Compactification-Theory/blob/6594586c74b38ce6cd410c56e4e087a8742d801c/research/dynamical_closure/CYCLIC_GROKKING_RESULTS.md).

### 5.2 Dynamic closure

Measure

\[
r_a(h)=\|\pi(F_a(h))-G_a(\pi(h))\|.
\]

Under a declared L-Lipschitz abstract transition, a near-collision gives the obstruction

\[
\epsilon\ge\tfrac12[\|\pi F_a(h)-\pi F_a(h')\|-L\|\pi(h)-\pi(h')\|]_+.
\]

The norm and scale are fixed. A numerical estimate of L is a diagnostic until certified. Uniform one-step error epsilon and a D_q-Lipschitz readout give

\[
\text{output error at H}\le\eta+D_q\epsilon\sum_{j=0}^{H-1}L^j
\]

for the same supplied action sequence on both systems. Closed-loop action agreement also requires a decision-margin argument. A small sampled average residual gives a distributional observation; coverage assumptions are needed for a uniform bound.

In exact toy worlds, successor-state supervision can implement

\[
\mathcal L=\mathcal L_{\rm task}+\lambda_{\rm dyn}\|z_{t+1}-\operatorname{sg}(e(x_{t+1}))\|^2.
\]

Ground-truth successor states are privileged supervision and must be charged. Real-model variants need a justified target representation and collapse controls. The initial experiment freezes the encoder or constrains it with independently measured readout performance; a jointly shrinking encoder otherwise makes the residual easy to optimize.

### 5.3 Transfer and statistical learning

When deployment reweights a fixed target with dnu/dmu <= M,

\[
R_\nu(C;Y)\le M R_\mu(C;Y).
\]

For a changed target with L2(nu) difference sigma,

\[
\sqrt{R_\nu(C;Y')}\le\sqrt{M R_\mu(C;Y)}+\sigma.
\]

These constrain optimal recoverability. Fitting a predictor adds estimation and optimization error. Direct samples from the intended deployment law are the preferred promotion test. A guessed density-ratio constant supplies no transfer certificate. Signed paired improvements cannot in general be multiplied by M using the nonnegative-loss inequality.

For a fixed full predictor class of at most 2^D descriptions, bounded loss, W predeclared words and n IID initial cases, a Hoeffding union bound yields

\[
R_\mu(f,w)\le\widehat R_\mu(f,w)+
\sqrt{[D\ln2+\ln(2|W|/\delta)]/(2n)}
\]

simultaneously over the class and words. Include encoder, routing, selected adapters, auxiliary tables and task-dependent advice in D. A frozen common backbone can be conditioned on when it is independent of the sampled labels. For large models this bound can be vacuous. Promotion below uses a candidate frozen before fresh evidence, allowing a pointwise conditional test. None of these inequalities guarantees that an optimizer will find a useful representation.

### 5.4 Recurrent stability and total cost

RC-01 gives a conditional restoration test

\[
e_{t+1}\le\rho e_t+\eta,\qquad \rho<1,\quad\eta\le(1-\rho)r.
\]

With adequate code separation, errors remain in the intended decoding tube. Test the assumptions on the actual learned module, including idle memory and addressing. Report description bits, persistent mutable bits, training work, recurrent steps, VRAM, host RAM, optimizer state and host/device bytes separately. Saved parameters can carry costs elsewhere in this ledger.

## 6. One bounded learning attempt

The epoch fixes the task-family distribution, scorer and resource caps. The incumbent stays frozen while candidate branches are trained.

1. Collect development diagnostics from the incumbent and update the experiment archive.
2. Select a reachable frontier task or a retention repair. Predict the result of one named update operator.
3. Clone the incumbent and apply a bounded training change. Initial operators are adapter placement/rank, a state-consistency loss, recurrent-depth schedule, state precision, replay allocation and module routing. Each attempt chooses one primary intervention; coupled bundles receive their own operator identity.
4. Use development evidence to discard weak candidates. An exploration branch may execute a bounded sequence of speculative steps to cross a plateau. Its budget and selection work are recorded in full.
5. Choose one final candidate and freeze its full execution identity. Lock comparator hashes, endpoint list, margins, sample counts and the attempt index.
6. Request fresh paired evaluation blocks from the promotion broker. The learner has no further updates during that request.
7. Apply every declared acceptance condition. A successful candidate replaces the incumbent. Rejections and insufficient-evidence results remain in the archive with their costs.
8. Retire that batch from future holdout use. Any information returned from it enters the exposure ledger. Update the experiment policy using information permitted by the declared channel.

A checkpoint can be developed aggressively through many internal trials before requesting promotion. The finite confidence budget counts every promotion attempt. Diagnostic failures and rejected proposals also count toward compute.

## 7. A fixed potential with explicit retention floors

Fix task-family weights w_v for an epoch. Let s_v(Theta,x) lie in [0,1]. An optional resource score c(Theta,x) lies in [0,1] after normalization by a fixed cap; exceeding a hard cap fails the run separately. Choose lambda >= 0 before the epoch and define

\[
u(\Theta,x)=\frac{s(\Theta,x)+\lambda[1-c(\Theta,x)]}{1+\lambda},\quad
\Phi(\Theta)=\mathbb E_{v\sim w,x\sim P_v}u(\Theta,x).
\]

The initial capability pilot uses lambda=0. An efficiency study freezes its own lambda and reports raw task performance and physical costs alongside Phi. This keeps the meaning of the scalar fixed for its claimed hill climb.

A capability promotion requires a lower confidence bound on Phi(candidate)-Phi(parent) of at least a predeclared positive eta. It also requires the requested target gain, plus every protected-family floor. A transfer claim requires its own unseen composition or neighbor endpoint.

For each protected family v, keep an immutable reference checkpoint a_v and a total tolerance epsilon_v:

\[
\mathbb E[s_v(\Theta_{\rm candidate})-s_v(a_v)]\ge-\epsilon_v.
\]

This is a cumulative allowance relative to that reference. Comparing only against the previous checkpoint would permit a small loss at every step. Newly mastered nodes obtain a reference; later graph versions preserve it. Replacing a reference requires separately certified non-degradation against the earlier one, or retaining both checks.

For v0, test all protected families. Larger graphs may omit an endpoint only after establishing exact preservation for the complete routed computation on that family. Unchanged adapter bytes alone cannot establish this when routing, shared weights or context has changed.

Task weights and scorer changes create a new epoch report. Append historical scores under their original definitions. Split nodes inherit their family's total weight, so duplicating easy tasks cannot increase their contribution.

## 8. EvalMath promotion contract

### 8.1 Information channels

Development data and diagnostics may be reused for optimization. The promotion channel receives one frozen candidate and fresh task instances. It returns the registered outcome and uncertainty summary; every returned field counts as feedback. A separate end-of-epoch exam uses held-out semantic families or graph compositions with a pinned scorer. Once its result informs a change, that exam is marked exposed and the next confirmatory claim requires fresh evidence.

The broker's generator, scorer and random state are outside the proposer's writable files. This is a v0 engineering assumption. Hashes establish recorded identity, while independent acquisition needs an actual implementation and audit. Secret seeds from a familiar generator protect instance reuse; broader transfer needs held-out semantics and independently authored falsifiers.

### 8.2 A conditional guarantee across adaptive proposals

Let F_(t-1) include everything available before promotion attempt t. Candidate, comparators, endpoint definitions, sample counts and graph version are fixed conditional on this history. Endpoint j supplies n_(tj) conditionally IID fresh paired block differences

\[
D_{tji}=S_j(\Theta'_t,X_i)-S_j(\Theta^{\rm ref}_j,X_i)\in[-1,1].
\]

Pairs share the task instance. Episodes or independently generated worlds are the sampling units; correlated turns within one episode remain one block. Persistent cross-episode state requires a different sampling contract or independent whole-run blocks.

Choose a total failure allowance delta in (0,1). For every attempt, including rejections, allocate

\[
\alpha_t=\frac{\delta}{t(t+1)},\qquad
\alpha_{tj}=\frac{\alpha_t}{m_t},
\]

where the finite endpoint count m_t is frozen before sampling. A two-sided interval has radius

\[
r_{tj}=\sqrt{\frac{2\ln(2m_t/\alpha_t)}{n_{tj}}}.
\]

The factor 2 accounts for the range length of paired differences. Use [mean-r,mean+r], clipped to [-1,1]. Apply the lower endpoint to each declared gain or retention floor.

**Proposition: adaptive promotion validity.** Under the acquisition and bounded-loss assumptions, with probability at least 1-delta every interval at every attempted promotion covers its conditional population difference. Consequently every accepted inequality holds for its declared target law.

**Proof.** Conditional on F_(t-1), Hoeffding gives failure probability at most alpha_(tj) for endpoint j. Sum over endpoints to obtain at most alpha_t. Take expectations and sum over attempts. The series sum_t 1/[t(t+1)] equals one. Adaptivity in candidate selection and finite endpoint count is allowed because each is fixed before its fresh batch. Dependence across endpoints does not affect the union bound.

This is a standard conditional-concentration construction applied to the graph protocol. It places no restriction on how many development candidates were tried before the frozen promotion candidate. Selecting a winner after seeing a shared promotion batch changes the test and requires simultaneous candidate accounting.

Fixed sample size is part of this reference. A second look consumes a fresh attempt and fresh data. A future sequential implementation can use valid confidence sequences or a predefined look-wise budget. Reusing cumulative samples with the fixed-n formula after discretionary stopping is outside this guarantee.

A small batch can produce `insufficient_evidence`. The reference deliberately preserves that outcome. Exact finite enumeration can establish exact fixture results at lower cost; its claim stays scoped to that finite task. Variance-aware bounds can replace Hoeffding after an independently tested implementation is frozen. A small effect cannot receive a stronger certificate by relabeling correlated tokens as samples.

### 8.3 What hill climbing earns

On the simultaneous-validity event, a fixed-potential epoch with every accepted gain at least eta satisfies

\[
\Phi(\Theta_K)\ge\Phi(\Theta_0)+K\eta.
\]

Since Phi lies in [0,1], there are at most floor((1-Phi(Theta_0))/eta) such promotions in that epoch. This follows by telescoping the accepted differences. It certifies progress on the stated potential. Discovery of new task families, convergence to a global optimum and transfer to unrepresented capabilities remain empirical questions.

The retention reference gives a second useful property: the allowed loss on each protected family stays epsilon_v through arbitrarily many accepted steps, under the simultaneous-validity event and an unchanged family contract. The allowance never becomes K*epsilon_v.

The attempt counter is global across graph versions. Starting another epoch cannot restore previously spent statistical error allowance. A new separately reported study can declare a new budget; combined claims must account for all studies used in selection.

## 9. Learning the experiment policy

The initial proposer uses a fixed prompt and measured diagnostics. Build an archive containing both failed and successful updates, with costs and comparator identity. Fit a predictor

\[
\hat f_\omega(z_t,v,a)\longrightarrow(\widehat{\Delta\Phi},\widehat{\Delta s_v},\widehat{\text{retention}},\widehat{\text{cost}}).
\]

Use development outcomes for frequent updates. The scheduler can rank reachable actions by predicted learning progress per total cost, with an explicit exploration allocation. This ranking is a heuristic under a changing learner; stationary-bandit regret guarantees require assumptions absent here.

An example budget split is 70% promising frontier updates, 20% retention/replay and 10% randomized diagnostic experiments. These are tunable draft defaults. Freeze them for each comparison. The reference executes no such allocation.

At the next epoch, a controller adapter or a small learned selector may replace the fixed policy. That replacement receives its own version. A meta-learning claim requires improved downstream learner performance on fresh graph instances at matched total compute, compared with the same proposer whose experiment policy stayed fixed. Reporting a better fit to its old experiment archive cannot establish that claim.

A learned graph edge also needs a contemporaneous control. Clone the same incumbent and compare training on the proposed prerequisite with matched alternative training, then score the target. An evolving model's before/after scores alone confound transfer with its other updates.

## 10. Experiments that decide whether introspection helps

The main question needs matched curriculum controls. Use the same candidate operators, task access, data exposure, trained learner and budget for every arm. Charge proposer tokens and diagnostic extraction even when they come from a subscription.

| Arm | Selection method | Purpose |
|---|---|---|
| A | Fixed human curriculum | Reference sequence |
| B | Random eligible task and update | Selection baseline |
| C | Development learning-progress scheduler | Curriculum benefit without internal diagnostics |
| D | Same proposer as E with behavioral traces and shuffled/masked internal diagnostics | Information ablation with matched calls |
| E | Measured self-introspective proposer | Proposed method |
| F | E plus a trained experiment-policy adapter on prior graph instances | Meta-learning extension |

Start by comparing C and E after an engineering smoke run. Add D before attributing any gain to internal diagnostics. F needs fresh graph instances and a frozen archive split. Match complete inference and training work, or report an explicit cost frontier where budgets differ. Retain paired independent training seeds for method comparisons; the promotion theorem concerns a particular frozen candidate and does not turn episodes from one training run into independent method replications.

Primary endpoint: epoch-end performance under the fixed graph distribution at the full resource budget. Secondary records include unseen composition transfer, worst protected-family regression, forecast calibration, accepted/rejected/inconclusive counts, and cost to a preregistered capability threshold. Log the raw score vector to expose scalar tradeoffs.

Required falsifiers include random-initialization diagnostics, label-preserving surface changes, dormant-variable reveals, wrong-order compositions, random features matched for precision and model complexity, and tasks that expose evaluator shortcuts. Any change to a falsifier after inspecting its result is recorded as a development revision.

## 11. Implementation sequence and run boundaries

**First build:** implement T0/T1/T4 with exact environment oracles, a tiny recurrent learner and a fixed task selector. Reproduce full-model composition before compressing its state. Implement the promotion batch lifecycle and demonstrate rejection of a regression that a scalar score would hide. Replay the exposure ledger from saved records.

**Next:** connect a configured 27B proposer to the typed proposal endpoint. It chooses among already implemented updates and receives the measured self-model. Use a hard job budget and checkpointed execution. Training dtype, quantized runtime, adapter merge/export, activation hooks and optimizer memory require an on-device compatibility smoke test.

**Then:** compare C/E/D on fresh graph instances and inspect total cost. Add T5 only with an independent symbolic scorer. Routed modules and T6 become the next composition study. Controller-adapter training follows a sufficient archive of varied outcomes and a fresh meta-evaluation partition.

FAW/FRC integration is a later workstream. The protocol currently relies on correctly implemented measurements and access separation. Any use in open-ended deployment needs a separate review of those assumptions.

## 12. Acceptance of this draft's implementation

A future runner is ready for its first model study when it can replay a task graph and all candidate identities; produce a failure receipt for a changed scorer, reused promotion batch, missing endpoint or exceeded budget; and distinguish rejection from insufficient evidence. It must preserve the anchor registry across graph edits and show that held-out information stays outside diagnostic collection until the batch is retired.

The local arithmetic tests check the spending identity, paired range factor, regression floors, an exact shear fixture and a restoration inequality. They exercise neither a learner nor the broker. No numerical test here certifies an acquisition assumption or formalizes the proofs in Lean.

```bash
python -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
```

## Primary literature

[Graves et al., Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003) treats task selection through learning progress. [Dwork et al., Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629) supplies the wider context for adaptive evaluation. The present v0 uses fresh batches and a summable error budget; a reusable-holdout algorithm would need its own implementation and assumptions. The repository sources above identify the precedents behind causal abstraction, observability and predictive-state representations.

## 13. Executable v0.2 finite reference

[RUNNER.md](RUNNER.md) documents the trained-table recurrent student and paired promotion broker, its exact finite task scores, the execution commands and its trust boundary. The runnable example completes two promotions using a fixed diagnostic heuristic. The scientific endpoint for a neural meta-learning claim remains prospectively specified in Sections 9–11.


## SIML-01N: neural curriculum pilot

[NEURAL_PROTOCOL.md](NEURAL_PROTOCOL.md) defines the tied-GRU student, task splits and matched-update comparison. [NEURAL_RESULTS.md](NEURAL_RESULTS.md) records the five-seed exploratory outcomes and the shuffled-diagnostic falsifier. The source files include an optional local 27B proposal-only adapter; it was not used to generate model output. The complete source-hashed five-seed receipt is in the downloadable neural artifact, with GitHub source verification pending its import. A separate optional CPU PyTorch workflow runs newly generated smoke checks. FAW/FRC remain deferred.
