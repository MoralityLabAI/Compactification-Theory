# RC-01: resource-bounded compactification of recurrent circuits

Date: 2026-10-29. Evidence: handwritten derivations, exact finite reference checks,
and cited primary sources. The statements below have not been checked by a Lean
kernel. This is an additive specialization of QPC-T01. Existing EvalLang, E10,
IC-01 and loop-capacity protocols retain their owners and definitions.

## 1. Computational object and resource ledger

Write a recurrent computation as

\[
 h_{t+1}=F_\theta(h_t,p,x,u_t),\qquad y_t=G_\theta(h_t).
\]

Here p is a program, x is ordinary task input, and u is the declared operation or
intervention. The complete state includes any loop counter, instruction register,
KV cache and persistent recurrent memory used by the transition. A counter supplied
by a host is still a resource. An uncharged input-dependent scheduler can encode
the answer. A terminal decoder with an uncharged lookup table has the same issue.

Use a resource vector rather than parameter count alone:

\[
 \mathcal R=(D_\theta,D_p,D_{\rm auxiliary},B_{\rm live},T,C,
             M_{\rm GPU},M_{\rm host},Q_{\rm H2D},Q_{\rm D2H},\varepsilon).
\]

D records description bits, B records persistent mutable bits, T counts recurrent
steps, C counts actual arithmetic work, M records physical memory residency, and Q
records transfers in bytes. Quantization precision belongs to both the description
and state encodings. GPU VRAM is itself DRAM; here host DRAM means CPU/system RAM.

For a fixed hardware/kernel family H, define an achievable region

\[
 \mathfrak C_{H}(f,\mathcal U,\varepsilon)
 =\{\mathcal R:\text{an implementation with resources }\mathcal R
 \text{ preserves the specified outputs/interventions to error }\varepsilon\}.
\]

Its nondominated points form the proposed resource frontier. This is a definition
and research target, not a solved optimization problem. Uniform worst-case and
held-out-distribution versions are different regions and must be labelled.

Qwenputer supplies a practical compiled hybrid-memory example [S1]. Its autoregressive
microinstructions and Gated DeltaNet state have to be included when relating it to a
latent-depth loop. The stroboscopic transition can be one complete instruction;
its cost includes every forward pass and trace token. Giannou et al. [S2] already
construct looped programmable transformers. The contribution sought here is an
explicit task-relative state/precision/I/O account with falsifiable experiments.

## 2. Reuse theorem and the description bound

**Proposition 1 (conditional instruction simulation).** Suppose a fixed recurrent
block realizes each instruction of a specified finite ISA on an encoding of a
p-word machine state, with at most c recurrent microsteps per instruction. A valid
program executing t instructions has an execution using at most ct microsteps,
the same block weights, and at most

\[
 B_{\rm live}\le p b_{\rm word}+B_{\rm registers}+B_{\rm counter}+B_{\rm buffers}
\]

mutable bits. Immutable program and input storage are recorded separately.

**Proof.** Initialize corresponding machine and neural states. Each completed
instruction preserves the encoding relation by hypothesis. Induction over the t
instructions gives corresponding outputs and the stated step count. The fixed
machine storage layout gives the live-state bound. This proposition is conditional
on an actual instruction implementation and numerical assumptions; it does not
compile an arbitrary small pretrained model by assertion.

For a Boolean DAG, a legal register/pebbling schedule gives such a program using a
universal Boolean gate, reads, writes and control. Gate recomputation can lower
live storage while raising executed instructions. Store the schedule and graph in
D_p, or charge their generator and execution. An arbitrary untied network can be
interpreted by a small controller while its original weights live in program data.
Its total description then still includes those weights. Repeated structure is
what can provide a description saving.

**Proposition 2 (finite-description counting).** Freeze the architecture, decoder,
input domain and side information. Let all task-dependent descriptions occupy at
most D bits, padded to a fixed length, and allow K externally selected global
runtime settings. The resulting family has at most K*2^D functions. For N independent
fair target labels, the probability that some member makes at most e errors is

\[
 \Pr(\text{fit})\le
 \min\left(1,\;K2^{D-N}\sum_{j=0}^{e}{N\choose j}\right).
\]

**Proof.** Each description/runtime pair selects one prediction vector. A Hamming
ball of radius e contains the displayed binomial sum. Count their union and divide
by 2^N. Overlap only makes the upper bound looser. For 0<=e/N<=1/2 the familiar
entropy bound gives K*2^{D-N(1-H_2(e/N))}.

A fixed adaptive halting policy is part of the description. Choosing a separate
runtime per input adds a scheduler description. Targets stored in initial persistent
memory, prompts, adapters or a decoder must be charged when they depend on the
training labels. Ordinary query input is not task-specific advice. This extends the
existing loop-capacity counting result without changing it.

## 3. Exact minimal state for affine binary circuits

Let the concrete domain be all of X=F_2^m. A finite operation alphabet acts by

\[
 F_a(x)=A_a x+b_a,\qquad E(x)=Cx.
\]

Every allowed intervention is included as an operation. Compare two states under
identical future operation words, allowing observation at the empty word too.
All immutable side information is fixed in this comparison. Define

\[
 V_0=\operatorname{rowspan}(C),\qquad
 V_{k+1}=V_k+\sum_{a\in\mathcal U} V_k A_a,\qquad V_*=\bigcup_k V_k.
\]

**Theorem 3 (minimal intervention-preserving affine quotient).** Set r=dim(V_*).
Then:

1. Future-observation equivalence is x~x' iff v(x-x')=0 for every v in V_*.
2. There are exactly 2^r equivalence classes. Any exact finite-state representation
   sufficient for all these future observations needs at least r bits.
3. If B has a basis of V_* as its rows, z=Bx attains that bound. For every a there
   is a unique matrix D_a satisfying BA_a=D_aB; the quotient transition is
   z'=D_a z+B b_a. The readout satisfies C=RB for some R.
4. Gaussian row-space closure computes V_* after at most m-rank(C) strict dimension
   increases, followed by a stability check. Runtime is polynomial in m and the
   number of operations under ordinary exact linear algebra.

**Proof.** Differences evolve as A_a(x-x'), since the translations cancel.
Outputs after any word are therefore differences in the rows C A_word. The closure
spans precisely all such rows, proving (1). Its annihilator has dimension m-r and
2^{m-r} points; its cosets partition 2^m states into 2^r equal classes. Distinct
classes cannot share a sufficient encoding, establishing the lower bound even for
nonlinear encoders. The rank-r linear map B is onto F_2^r. Invariance V_* A_a subset
V_* expresses each row of BA_a in the independent basis B and gives D_a. The same
argument gives R. Encoding after transition equals transition after encoding:
B(A_ax+b_a)=D_aBx+Bb_a. Finally, each unfinished closure increases dimension,
which is bounded by m. This proves the algorithmic claim.

This is the observability/minimal-realization argument specialized to controlled
finite-field circuits. It does not claim novelty for observability theory. The
reference independently compares it with exhaustive finite-state partition
refinement, including the translations and all quotient transitions.

**Constructive examples.** Reading only the first of m bits under identity dynamics
requires one bit. Add a cyclic shift and every bit becomes observable, so r=m.
A parity readout under permutations alone has r=1. With independently permitted
writes x_j:=0, its immediate post-write parity has row 1^T+e_j^T. Subtracting the
original parity row isolates e_j^T. Thus even a parity-only evaluator needs r=m
when arbitrary overwrites remain possible. The same conclusion holds when writes
to one are also supplied; their affine offsets are retained by the quotient.

This is a concrete reason to choose the intervention alphabet before measuring
compression. A passive low-rank probe can measure one bit while intervention-
preserving state requires every memory bit.

**Scope.** A restricted reachable state set can reduce the number of classes.
Finite-horizon equivalence uses rows through that horizon; unless closure has
stabilized, the transition goes from the H-horizon representation to H-1.
Nonlinear Boolean circuits admit finite partition refinement, with exponential
worst-case enumeration in m. The polynomial row-space algorithm applies to the
affine class. General transformer extraction remains a separate obligation.

## 4. Exact RAM, numerical precision and indefinite symbolic execution

For row-address GDN memory S, write

\[
 T(S)=\alpha(I-\beta kk^T)S+\beta k v^T.
\]

At alpha=beta=1 and k=e_i, this overwrites row i and preserves every other row.
Its state-linear part has eigenvalue 0 on the addressed direction and 1 on all
other address directions. With beta=2 it has eigenvalue -1 on that direction.
The local spectral behavior depends on which information the instruction keeps.

**Proposition 4 (global contraction excludes nontrivial permutations).** A map
which permutes a finite set of at least two distinct codewords cannot have a global
Lipschitz constant strictly less than one on that set.

**Proof.** A permutation has finite order q. For two different codewords c,d,
F^q(c)=c and F^q(d)=d. Global contraction would give
||c-d||<=rho^q||c-d|| with rho<1, a contradiction.

Multiple stable codewords can still have locally contracting neighborhoods.
That is the useful condition for digital recurrent computation.

**Theorem 5 (uniform robust execution).** Let c(z) encode a finite symbolic state
with code separation Delta. For every legal instruction a and every perturbation
||e||<=r suppose the implemented step obeys

\[
 \|\widehat F_a(c(z)+e)-c(\delta_a z)\|\le\rho\|e\|+\eta.
\]

Assume 0<=rho<1, 2r<Delta, eta<=(1-rho)r, and initial error at most r.
Nearest-codeword decoding then gives the correct symbolic execution for every
finite number of steps, as long as all these uniform hypotheses continue to hold.

**Proof.** The inequality maps each radius-r code tube into the next one. Induction
preserves the tube and 2r<Delta makes decoding unique at every step. More explicitly,
e_t<=rho^t e_0+eta(1-rho^t)/(1-rho). This is a conditional mathematical guarantee
against bounded error, with no stochastic independence assumption.

**An explicit finite-precision NAND construction.** Define

\[
 R(s)=2\operatorname{ReLU}(s-1/4)-2\operatorname{ReLU}(s-3/4),
 \qquad N(x,y)=1-R(\operatorname{ReLU}(x+y-1)).
\]

If x and y lie within 1/8 of their intended bits, N returns the exact NAND bit:
false AND inputs give an inner value <=1/4, while true AND inputs give >=3/4.
The dyadic thresholds and integer slopes have finite descriptions. The theorem
allows an additional error of at most 1/8 after each restored update. The reference
checks all gate corners and a recurrent parallel NAND ring. These finite checks
supplement the explicit inequality proof.

All live coordinates need the stated per-step bound. Restoring only the current
output leaves idle cells exposed to accumulated error. Options include exact
inert bit storage, a bound on idle-cell corruption, or a refresh schedule whose
traffic and computation are charged. The construction uses hard exact addressing;
soft attention contributes another error term.

For M softmax choices with correct-score gap at least g, the total wrong mass is

\[
 \epsilon_{\rm addr}\le
 \frac{(M-1)e^{-g}}{1+(M-1)e^{-g}}.
\]

Hence g>=log((M-1)(1-epsilon)/epsilon) suffices to keep wrong mass below epsilon.
With binary data stored within radius r_mem, a convex softmax read has error at
most r_mem+epsilon_addr. The NAND margin requires that total to be <=1/8 before
other kernel errors. Finite logits, normalized keys, position codes and arithmetic
rounding must be checked for the actual model. A constant-size arithmetic core
does not automatically provide cost-free addressing at arbitrary memory size.

**Weight-to-state bridge.** Suppose within every correct tube a weight perturbation
adds at most K_theta*||delta_theta||_infty to one-step error. Uniform rounding over
[-R_w,R_w] with b bits can use error bound R_w/2^b. With numerical error eta_num,
a sufficient precision condition is

\[
 K_\theta R_w2^{-b}+\eta_{\rm num}\le(1-\rho)r.
\]

When the right-hand residual is positive, it is enough to choose

\[
 b\ge\max\{1,\lceil\log_2(K_\theta R_w/((1-\rho)r-\eta_{\rm num}))\rceil\}.
\]

The bound is independent of the requested horizon. K_theta is a weight-sensitivity
constant and rho a within-tube state-sensitivity constant; they have distinct domains.
The zero K_theta case is immediate and avoids the displayed logarithm. Without a
restoration bound, a neutral transition only gives e_T<=e_0+T*eta. This supplies a
sufficient safe horizon, not an assertion that a given learned model fails there.
A deliberately drifting identity map makes the linear envelope sharp in that class.

## 5. Hardware compactification and a locality budget

Compiling a controller into GPU-resident weights can save repeated weight loads.
Whether its weights remain in L2 cache is a measurable hardware question. VRAM
residency alone supplies no on-chip-cache guarantee. A dense attention read still
scans its declared keys; logical RAM and constant-latency physical RAM differ.

Use an instruction DAG and legal schedules to charge live intermediates and I/O.
Hong and Kung's red-blue pebble model [S6] is the classical two-level framework.
Its lower bounds concern specified computation graphs and operation rules;
an algebraically different algorithm requires its own graph and argument.

For work C and device-memory traffic Q_v, peak admissible rates give the basic
necessary wall-time bound

\[
 \tau\ge\max(C/F_{\max},Q_v/B_{v,\max},Q_\uparrow/B_{\uparrow,\max},
                         Q_\downarrow/B_{\downarrow,\max}).
\]

Measured sustained rates give a practical cost model, not an unconditional theorem.
Serial data dependencies add latency, and overlap determines how near the lower
bound an implementation runs. Profile each direction rather than adding rates.

For a state of S bytes spilled and reloaded after every k steps during T steps,
the explicit schedule in the supplied benchmark moves

\[
 Q_\uparrow=Q_\downarrow=S\lceil T/k\rceil.
\]

This schedule includes a final roundtrip for consistent accounting. It does not
reduce single-session peak state memory: it measures transfer cost and bit-preserving
transport. A multitenant capacity test must actually evict inactive states.

In an idealized serial model with loop compute tau_loop and roundtrip startup 2l,
steady-state transfer overhead fraction is

\[
 f_{IO}\approx\frac{S/B_\uparrow+S/B_\downarrow+2l}{k\tau_{\rm loop}}.
\]

Keeping it below eta requires k at least the numerator divided by eta*tau_loop.
This is the residence-time experiment: larger DRAM capacity helps when enough work
can be completed between transfers or independent work hides their latency.

For autoregressive full-attention blocks, R loop-indexed caches cost

\[
 M_{KV}=2B_{batch}L_{block}RNH_{KV}d_{head}s_{elem}.
\]

A first-loop shared cache with later-loop windows of w tokens changes the cache
factor RN to N+(R-1)*min(w,N), before adding any separate recurrent states. This
is an architecture/representation choice whose behavioral effects must be tested.
It is not an automatic consequence of sharing weights. Current shared-memory and
cross-loop-codec studies [S4,S5] support testing both trained sharing and preservation
of loop-specific information.

The October 2026 CoT-internalization construction [S3] illustrates why physical
accounting matters: it reduces visible positions by using separate per-loop caches;
its own total cache and serial-work ledgers remain unchanged under that folding.
The memory-budget separation [S7] studies a genuinely compressed mutable-state
regime under explicit uniformity and finite-precision assumptions.

## 6. Parameter sensitivity under tied recurrence

To connect with the existing gradient-rank experiments, suppose F is differentiable
along a fixed trajectory and let

\[
 J_t=\partial_hF_\theta(h_t),\quad B_t=\partial_\theta F_\theta(h_t),
 \quad S_t=\partial_\theta h_t.
\]

The chain rule gives

\[
 S_{t+1}=J_t S_t+B_t,\qquad
 S_T=\Phi(T,0)S_0+\sum_{t=0}^{T-1}\Phi(T,t+1)B_t,
\]

where Phi(T,t+1)=J_{T-1}...J_{t+1}, with the empty product equal to identity.
The same parameter perturbation enters every term. This is distinct from a control
model in which an independent perturbation can be selected at each time. The
Krylov span [B,AB,...] of that independently controlled model can overestimate the
degrees of freedom of a tied network. With one scalar parameter, A swapping two
coordinates and B=(1,0)^T, the two-step sensitivity is (1,1)^T of rank one; the
independent-control matrix [B,AB] has rank two.

For N evaluated examples and W trainable parameters, the N-by-W output-gradient
matrix always has rank at most W. Repeated steps can change which parameter
directions matter and improve conditioning. The finite-description bound remains
in force. Proposed measurements should include the task-conditioned singular
spectrum of this actual tied-parameter Jacobian, alongside local state restoration
margins. A state-transition norm cannot be substituted for weight sensitivity.

## 7. Proof and experiment status

The affine quotient is exact on its declared finite domain. The restoration proof
is conditional on uniform margins and bounded perturbations. The hardware schedules
are explicit accounting identities with a cost model for timing. No conclusion
about training reliably discovering these circuits follows from their construction.

`reference.py` and `test_reference.py` give new finite checks for this note.
`gpu_dram_bench.py` is a systems/transport pilot, including an untrained full-sequence
looped transformer. GPU execution, learned circuit recovery and Qwenputer checkpoint
replication remain unperformed in this work. The experimental protocol supplies
separate decision criteria for each of those measurements.

## Primary sources

[S1] Divyansh Agrawal, Qwenputer (2026-10-22).
https://divagr.com/blog/qwenputer
Author-reported compiled hybrid experiment; the write-up was rechecked for this note.

[S2] Giannou et al., Looped Transformers as Programmable Computers, ICML 2023.
https://proceedings.mlr.press/v202/giannou23a.html
https://proceedings.mlr.press/v202/giannou23a/giannou23a.pdf
PDF pages 2 and 4 inspected, including instruction/memory interface.

[S3] Zhou and Lu, Internalizing Chain-of-thought into Looped Transformers,
arXiv:2610.25998, 2026-10-19. Sections 3 and 4.4.
https://arxiv.org/abs/2610.25998
The stated construction uses per-iteration KV caches and a loaded-cache interface.

[S4] Monea et al., The Surprising Effectiveness of Shared Memory in Looped
Transformers, arXiv:2610.02383, 2026-10-01.
https://arxiv.org/abs/2610.02383

[S5] O'Neill and Reid, Looped Latent Attention: Cross-Loop KV Compression for
Looped Transformers, arXiv:2607.15456v2, 2026-07-24.
https://arxiv.org/abs/2607.15456v2

[S6] Hong and Kung, I/O Complexity: The Red-Blue Pebble Game, STOC 1981.
https://www.eecs.harvard.edu/~htk/publication/1981-stoc-hong-kung.pdf
PDF page 2 inspected for the graph and fast/slow memory rules.

[S7] Haozhou Zhang, Chain-of-Thought and Compressed Looped Transformers:
A Memory-Budget Separation, arXiv:2605.30757v1, 2026-05-29, Sections 3-4.
https://arxiv.org/html/2605.30757v1

Related repository snapshots read for this extension:
Compactification-Theory fc27a7d33aa483411219b8da3991254d1fd174d3;
loop-capacity docs/note.md at e1042ad66a48df10b25be70c4e72efe588107018.
