# CGC-01: protected causal compactification of shared-weight recurrent computation

**Mathematical research addendum, 18 November 2026.** Bridges [BPD SF01–SF05B](https://github.com/MoralityLabAI/BPD), the [RC-01 recurrent circuit quotient](../recurrent_circuits_v1/THEORY.md), [LSV-15 exact LayerNorm gauges](../levin_sieve_v1/pretrained_trm/GAUGE_QUOTIENT.md), [LSV-18 semiconjugacy](../levin_sieve_v1/pretrained_trm/LSV18_SEMICONJUGACY.md), [SCM-15 causal dimension atlas](../dimensional_atlas/SCM15_THEORY.md), [SMC-15 native near-null SAE pairs](../real_recurrence_sae_v1/SMC15_RESULTS.md), the [NOR local contraction certificates](../nor01_numerical_orbit_sketch/NOR04.md), and the [precision/description frontier](../precision_entropy_frontier/README.md).

**Evidence:** Propositions below are mathematical deductions, some resting on classical linear-algebra and state-space results, with deterministic finite-dimensional Python and exact-rational witnesses in [`finite_audit.py`](finite_audit.py). They have **not** been Lean-kernel formalized and have **not** been measured on a newly trained BPD/SAE/TRM model. In BPD itself, the registered SF03 strong incremental test **failed 0/32 roots**, SF04 public-safe partial-coalition gate **failed 3/8**, and the newer SF05B source-closure ablations were development-only and failed their combined public/authorized objectives. These failures remain evidence, not overridden results.

## 1. Define the observer and the computation being compacted

Consider a complete recurrent state `h in R^d` and a time-homogeneous physical transition `F_x`, with any time-varying forcing `x` explicitly supplied or included in the state. Fix the state `h` and a declared physical Euclidean inner product. Collect protected outputs into a differentiable map `P: R^d -> R^p`: e.g. actual current normalized logits and future **unauthorized** readouts under a specified query history. Collect desired downstream task outputs into `T: R^d -> R^m`: e.g. authorized source retrieval, task success margins, or future reasoning-program readouts. Multiple horizons and contexts can be stacked in `T` with *predeclared* output weights/whitening. Crucially, all future maps must execute the **true** recurrent continuation, not a probe or teacher-forced proxy.

At the same complete recurrent state define

\[
B=D P(h),\quad K=\ker B,\quad Q\in\mathbb R^{d\times k},\quad Q^TQ=I_k,\quad\operatorname{im}Q=K,\quad A=D T(h)Q.\tag{1}
\]

The `k=d-rank(B)` directions form the **local protected-readout tangent kernel**. They leave `P` unchanged only at first order. If an exact architectural gauge is known, such as the channel-shift symmetry of a LayerNorm, its finite/infinite-radius invariance is a separate stronger statement. `P` must represent the real monitor information set; protecting a single scalar when full vocabulary logits or an entire observable trace are available is insufficient for a transcript privacy claim.

The *safe task rank* is

\[
r_{\rm task}(h;P,T)=\operatorname{rank}(A)=\operatorname{rank}(DT(h)|_K).\tag{2}
\]

It is relative to the chosen physical metric, output metric, horizon and task/observer pair. It counts independent **first-order task effects accessible while holding the protected output fixed to first order**. It does *not* count the number of neural mechanisms or establish a global invariant quotient. In particular, selecting a BPD component by large post-rollout gradient can recover an ordinary task-sensitive direction without discovering a reusable parameter circuit.

## 2. Protected task spectrum and the optimal local block

**Theorem CGC-01 (protected Eckart–Young compactification).** Let the singular values of `A=DT(h)Q` be `sigma_1>=...>=sigma_k>=0`, padding zeros when needed. Among all `r`-dimensional orthogonal subspaces `E` of `K`, with physical projector `Pi_E`,

\[
\min_{\dim E=r}\|DT(h)(\Pi_K-\Pi_E)\|_{\rm op}=\sigma_{r+1},\quad
\min_{\dim E=r}\|DT(h)(\Pi_K-\Pi_E)\|_F^2
=\sum_{j>r}\sigma_j^2.\tag{3}
\]

A minimizer is the physical span of the first `r` **right singular directions of `A` mapped back by `Q`**. The protected task Gramian in kernel coordinates is

\[
G_K=A^TA=Q^T DT(h)^TDT(h)Q.\tag{4}
\]

For an isotropically chosen intervention `delta` on the radius-`rho` sphere **inside K**, and the top-`r` projector, the exact first-order mean-square omitted task response is

\[
\mathbb E\|DT(h)(I-\Pi_E)\delta\|^2
=\frac{\rho^2}{k}\sum_{j>r}\sigma_j^2,\quad k>0.\tag{5}
\]

**Proof.** Every `E` inside `K` corresponds to an orthogonal projector `V V^T` in the `Q` coordinates. The linear omitted map is `A(I-VV^T)`. Its rank-`r` complement is `AVV^T`, a rank-at-most-`r` approximation to `A`; truncated SVD attains the classical Eckart–Young–Mirsky minima in both norms. For the spherical statement, `E[delta delta^T]=(rho^2/k) Pi_K`; trace the omitted squared response. QED.

**Rank lower bound.** If a differentiable `r`-coordinate encoder `phi` and decoder `D` exactly factor `T` on an open local neighborhood of `h` **within the affine slice `h+K`**, the chain rule implies `rank(DT(h)Q)<=r`. This is a necessary local dimension for exact task behavior on the slice. For an `epsilon`-approximation, the tail in (3) is a *first-order error frontier*, not a claim of exact factorization. A local task map being representable says nothing about whether an encoder can evolve autonomously with the recurrent transition.

**Important identifiability qualification.** Singular values and spectral error are well defined in the **fixed physical metric**. If `sigma_r=sigma_{r+1}`, the selected rank-`r` subspace need not be unique. A nonorthogonal physical change of hidden-state coordinates changes Euclidean singular values unless the metric is transformed with it. By contrast, the *component-coordinate* gauge `V -> V R, U -> R^{-1} U` in BPD leaves the physical hidden state and physical operator fixed, so it leaves this task Gramian unchanged.

The result provides a falsifiable definition of **local protected causal dimension**. It does not imply that the weight matrix's own top singular vectors optimize task transport. For example, `W=diag(10,1)` and task readout `T(h)=h_2`: the top physical-weight SVD selects the first coordinate yet the task reads only the second. This exact counterexample is in the tests.

## 3. A nonlinear local compactification guarantee

**Theorem CGC-02 (finite-radius task-error certificate).** Suppose `T` is `C^2` on the closed Euclidean ball `B_rho(h)`, with uniform Hessian bilinear operator norm `||D^2 T|| <= M` on that ball. Let `E` be the top-`r` protected task block from CGC-01. For every `delta in K` with `||delta||<=rho`,

\[
\|T(h+\delta)-T(h+\Pi_E\delta)\|
\le\rho\sigma_{r+1}+M\rho^2.\tag{6}
\]

**Proof.** Expand each task value around `h`: `T(h+v)=T(h)+DT(h)v+R(v)` with `||R(v)|| <= M||v||^2/2`. The linear difference is bounded by `sigma_(r+1)||delta||`, since `delta-Pi_E delta` lies in `K`. Bound the two Taylor remainders by `M(||delta||^2+||Pi_E delta||^2)/2 <= M rho^2`. QED.

Thus a **sufficient, local** dimension certificate under error budget `epsilon` is the smallest `r` for which `rho*sigma_(r+1)+M*rho^2 <= epsilon`. This bound can be conservative even at full rank because it compares two independent Taylor remainders. It does not prove that a proposed projection preserves the protected outputs at finite radius; `P` needs its own nonlinear bound or an exact level-set correction. This distinction is central to SF05.

For trained transformer features, validated Hessian bounds from the SCM-14 program can in principle supply `M`. A floating Jacobian/SVD alone gives a diagnostic curve rather than an exact certificate.

## 4. Multichannel finite public-level-set correction

**Theorem CGC-03 (minimum-norm correction and second-order residual).** Assume the *current protected readout* has an **exact affine/architectural** kernel `K_0` with orthonormal column basis `Q_0`; along this kernel it is truly unchanged for finite edits. Let `P_U: R^d -> R^p` be the future protected public output under a fixed query program. Given a current-dark candidate displacement `d in K_0`, define

\[
 e=P_U(h+d)-P_U(h),\quad J=DP_U(h+d)Q_0.\tag{7}
\]

If `J` has full row rank and smallest singular value `sigma_min(J)>=gamma>0`, the minimum-Euclidean-norm **linearized** correction inside `K_0` is

\[
 s=-Q_0 J^\dagger e,\quad \|s\|\le\|e\|/\gamma,\quad
 D P_U(h+d)s=-e.\tag{8}
\]

If the Hessian bilinear operator norm of `P_U` is at most `M_U` along the segment from `h+d` to `h+d+s`, the **actual finite public residual after one exact step** obeys

\[
 \|P_U(h+d+s)-P_U(h)\|
 \le\frac{M_U}{2}\|s\|^2
 \le\frac{M_U}{2\gamma^2}\|e\|^2.\tag{9}
\]

**Proof.** In kernel coordinates `s=Q_0 z`, the equation `J z=-e` has the least-norm solution `z=-J^dagger e` by the Moore–Penrose characterization; full row rank makes it exact and bounds the pseudoinverse norm by `1/gamma`. Taylor expansion about `h+d` cancels the zeroth and first-order residuals, leaving the Hessian remainder. QED.

This extends the *scalar* source-locked SF05-A2 Newton construction to a collection of **joint public outputs**. In a language model that may mean all monitored normalized logits/token sites, where full row rank can be impossible if `p>dim K_0`. If `J` lacks full row rank, an arbitrary public residual may be unreachable by such a repair; the actual least-squares leftover is `(I-JJ^dagger)e`. Any apparent success after silently collapsing a full output vector to one scalar is observer-dependent. A public-only output correction can also move the authorized target, so source-specific task retention must be checked after the finite edit.

A simple analytic negative control is `P_U(h)=h_0+h_1^2` at `h=0`, with current readout `q(h)=h_0`. The finite edit `d=(0,0.2,0)` is tangent-safe because the initial public gradient in the `h_1` direction vanishes, yet changes the actual future public output by `0.04`. A Newton step at `h+d` is `s=(0,-0.1,0)`, which leaves residual `0.01`, exactly the quadratic bound for `M_U=2` and `||s||=0.1`. Stage A found a more complicated trained-model failure: projecting the *original* physical donor displacement onto the tangent plane could worsen finite public behavior.

## 5. From local rank to an exact dynamic quotient

The protected task spectrum above is **not** an exact recurrent compactification. To make the distinction constructive, specialize to a real-affine controlled recurrence on the *full declared state domain*,

\[
 h_{t+1}=A_a h_t+b_a,\quad y_t=C h_t,\quad a\in\mathcal U.\tag{10}
\]

Let `V_0=rowspan(C)` and iteratively close the output row space under every permitted action,

\[
V_{n+1}=V_n+\sum_{a\in\mathcal U}V_n A_a,\qquad V_*=\lim_nV_n.\tag{11}
\]

**Theorem CGC-04 (minimal real-linear affine recurrent quotient).** If `B` is a row basis of `V_*`, of rank `r_*`, there exist matrices `D_a` and a readout `R` such that

\[
 B A_a=D_a B,\qquad C=RB,\qquad
 z'=D_a z+B b_a,\quad y=Rz,\quad z=Bh.\tag{12}
\]

Every **linear** encoder preserving `C` and admitting such closed affine reduced transitions for all actions has rank at least `r_*`. Moreover, for any states `h,h'`, all future output words agree under the same action program if and only if `B(h-h')=0`. Over the full continuous state space, the quotient has *dimension* `r_*`, while a finite bit count requires a declared precision and bounded domain.

**Proof.** The row-space iteration spans precisely the rows `C A_{a_k}...A_{a_1}` for all finite operation words. Their common kernel is the set of invisible state differences under every future word. Closure guarantees `BA_a` lies in the rowspace of `B`, giving `D_a`, while `C` lies there as well and factors as `RB`; offsets descend as `Bb_a`. Conversely, any closed linear encoder must contain the rows of `C` and be invariant under all `A_a`, hence contain `V_*`. QED. This is the standard linear-systems observability construction over the reals, parallel to [RC-01's exact F2 circuit theorem](../recurrent_circuits_v1/THEORY.md), rather than a claimed novel result in control theory.

For a **single** fixed real-affine map, the finite observability matrix is

\[
\mathcal O_H=\begin{bmatrix}C\\CA\\\vdots\\CA^{H-1}\end{bmatrix},\quad
G_H=\mathcal O_H^T\mathcal O_H.\tag{13}
\]

Its rank is nondecreasing in horizon `H`, and stabilizes at the exact quotient rank by `H=d` (Cayley–Hamilton). The Gramian satisfies `G_(H+1)-G_H=(CA^H)^T(CA^H)` positive semidefinite. The singular tail of a finite **protected** observability stack supplies the task-relative local analogue of this exact invariant row-space closure, but says nothing about whether the proposed protected kernel remains invariant during rollout.

**Contraction connection.** If `||A||op = lambda < 1`, the unobserved squared Frobenius energy beyond a finite horizon is bounded by

\[
\sum_{j=H}^\infty \|C A^j\|_F^2
\le\frac{\|C\|_F^2\lambda^{2H}}{1-\lambda^2}.\tag{14}
\]

For a nonnormal recurrence, `spectral radius(A)<1` alone gives **no** `lambda<1` bound in this physical norm and does not prevent large early amplification. The toy `A=[[0.5,3],[0,0.5]]` has spectral radius 0.5 yet `||A||op>3` at the first step. The NOR block-derivative/invariant-basin certificates are therefore the more relevant route to trained looped networks than assuming a spectral-radius contraction.

## 6. Independent information/precision obstruction

**Proposition CGC-05 (finite distinguishable-future code lower bound).** Fix an input/decoder/side-information contract. Suppose `N` reachable states have pairwise task-future outputs at least `Delta` apart, under one declared future protocol. An encoder-decoder with worst-case task-output error strictly less than `Delta/2` must assign **distinct codes** to those `N` states. Any fixed-length quantized code needs at least `ceil(log2 N)` bits, and if it uses `r` mutable coordinates with at most `b` bits each then

\[
 r b\ge\lceil\log_2 N\rceil.\tag{15}
\]

**Proof.** If two separated targets shared a code, their common decoded output would be within less than `Delta/2` of each, violating the triangle inequality. Thus there are `N` distinct codes. Count them. QED.

This is a **global finite-code requirement**, distinct from the differential task rank. For example, NOR's two separated recurrent attracting basins require one terminal basin bit for a joint compact code even when each individual basin contracts to a locally zero-dimensional limit. Conversely, a small observed local tangent dimension says nothing about the number of discrete basins, the precision required to distinguish nearly coincident futures, or the cost of computing the code. Treat the encoder's parameters, query-dependent selection, external state and decimal precision as part of RC-01's resource ledger.

## 7. Gauge, causal witnesses, and a new compactness frontier

For a physically frozen BPD matrix `W=VU`, let any invertible `R in GL(c)` act as `V'=VR, U'=R^{-1}U`. The full write `W`, its physical output-SVD projector and its functional continuation are unchanged. A singleton/diagonal mask is generally **not** invariant: on an invertible factorization, it must commute with each aligned relative gauge transformation to define the same physical intervention (see [BPD optimal basis/invariant-mask theorem](https://github.com/MoralityLabAI/BPD/blob/main/mechinterp/bpd/theory/OPTIMAL_BASIS_AND_MASKS.md)). The task-conditioned spectral block in Theorem CGC-01 is likewise invariant to *component-coordinate reparameterization* because it is constructed from the unchanged physical dynamics and actual task/observer derivatives.

These are different mathematical quotients:

| Quotient / dimension | What it preserves | What it does not yet establish |
|---|---|---|
| Factor-gauge physical block `VU` | Same weight operator | Identifiable singleton neuron/SAE feature |
| Protected tangent `K=ker DP` | First-order protected output | Finite public output preservation |
| Task spectrum on `K` | Best linearized causal task response for a given block rank | Causal component specificity or autonomous recurrence |
| Exact affine row-space quotient `V_*` | Every declared future readout under every action | Nonlinear-model compression or finite-bit storage |
| Finite code lower bound | Distinguishable task-future classes at a specified margin | Trained decoder, inference cost, or reachable-basin classification |

A well-specified **compact causal computation frontier** should therefore report

\[
\mathcal F_{\mathrm{CGC}}=(r,\ b,\ T,\ C_{\rm compute},\ \epsilon_{\rm task},\ \epsilon_{\rm public},\ \epsilon_{\rm closure},\ \epsilon_{\rm gauge};\ \mathcal O,\ \mathcal U,\ \mathsf{family}).\tag{16}
\]

Here `r` is selected **physical causal subspace dimension**, `b` is retained precision per coordinate, `T` is execution depth and `C_compute` includes any learned encoder/policy and oracle calls. The observer information `O`, legal interventions `U`, source family and task/public metrics must be declared. Finite-radius and rollout-closure failures remain separate from weight reconstruction or probe decodability.

**Central extrapolated hypothesis, unproved:** for some trained compact recurrent models, a low-dimensional gauge-stable physical subspace approximates several task-conditioned future readouts at multiple passes while preserving designated public outputs, and the dimension remains stable under deeper shared-weight execution. This is falsifiable: the dimension may increase with horizon, factor fits may be degenerate, gradient-matched shams may reproduce the effect, or exact current readout matching may yield too few naturally reachable interventions.

## 8. Prospective BPD / Compactification / EvalMath experimental contract

The accompanying [CGC-01 protocol](PROTOCOL.md) defines a source-only reference suite and a separate unrun trained-model study. The scientific target is a **causally sufficient, gauge-stable block across recurrent depth**. Candidate models: compact TRMs and looped transformers with true per-pass readouts, accompanied by synthetic source-binding and exact EvalMath program-state tasks with new independently generated source families.

Predefine a full normalized-logit or task-protocol readout, compute the protected kernel and task Gramian on **development** states, select physical subspaces by the task singular spectrum and compare them with physical-weight SVD, independent BPD factorizations, naturally active SAE subspaces, hidden-state random-dark controls, norm-and-future-Jacobian matched shams, and no-edit/donor full-state positive controls. Then **freeze** the basis, rank and radius before evaluating different model roots, heldout programs and additional loop counts.

The strong hypothesis is that a *small physical block*, with the same selection rule, transfers actual heldout causal task effects beyond the source/calibrated shams while remaining numerically public-safe and maintaining task utility. Require the complete teacher and BPD native execution path, finite-radius perturbation sweeps, all attempted/eligible donor denominators, gauge rotations, output whitening, model-root-level uncertainty and a phase-disjoint dictionary-training control. Source-only spectral identities cannot satisfy those gates. The sealed ARG-03-N split remains unused, and existing BPD and Compactification-Theory receipts are unchanged.

## Sources and scope

- Classical best rank-`r` truncation: [Eckart–Young theorem, EPFL notes, Thm. 9.1](https://www.epfl.ch/schools/ic/ipg/wp-content/uploads/2019/09/ITSP2019_Sept20.pdf).
- Classical matrix model reduction and the Moore–Penrose pseudoinverse: [Max Planck Institute, Matrix Equations and Model Reduction](https://www.mpi-magdeburg.mpg.de/2517709/MathBasics.pdf).
- Internal contextual sources: [BPD SF03](https://github.com/MoralityLabAI/BPD/blob/main/mechinterp/bpd/theory/sf03_tangent_null/RESULTS.md), [BPD SF04](https://github.com/MoralityLabAI/BPD/blob/main/mechinterp/bpd/theory/sf04_source_binding/RESULTS.md), and [BPD SF05B Stage 2-B PR #64](https://github.com/MoralityLabAI/BPD/pull/64), with the last remaining a development-only result at the time of this note.
- Existing work in this repository: RC-01, LSV-15/18, SCM-14/15, NOR-04/10, SMC-15/27 and the precision-entropy frontier; see links at the top. New novelty is in the **combined observer-protected causal compactness contract**, with explicit negative controls and numerical certificates. The individual SVD, observability, pseudoinverse and counting statements are classical or straightforward deductions.
