# HC-01: intervention-sufficient learned chunking

Date: 2026-10-29. Status: conditional mathematical statements, exact finite-state reference, prospective neural tests. No trained H-Net checkpoint has been inspected in this package. The proofs below are handwritten and have not been kernel-checked.

## Motivation and scope

The 2025 H-Net paper introduced end-to-end, context-dependent dynamic chunking of byte streams. On 2026-10-29 Albert Gu described a newer H-Net scaling study reporting 3–5x controlled efficiency improvements and an approximately 10x large-model training-data/compute comparison. Those are author-reported results, not independent measurements in this repository. They motivate asking which *future distinctions* a learned chunk representation preserves.

Existing [IC-01](../../experiments/interventional-compactification/README.md) asks when an encoding commutes with declared interventions. [RC-01](../recurrent_circuits_v1/THEORY.md) computes exact minimum state for affine binary machines under a declared operation alphabet. This note specializes those questions to hierarchical, prefix-causal chunking, whose boundaries change with context. It makes no claim that predictive compression entails semantic or intervention-preserving compression.

## 1. Complete causal state

Let `A` be a finite alphabet of raw input bytes and permitted interventions. The complete state immediately before the next input is

\[
s_t=(h_t,r_t,b_t,\phi_t,c_t,\tau_t,\xi_t),
\]

where the components represent, as required by the actual implementation, backbone state, routing state, incomplete-chunk buffer, hierarchy phase, decoder/cache state, byte and chunk positions, and any stochastic state. Omitting a component requires a proof that the next update and observable readout remain functions of the retained state. Treating a latent chunk token as the entire state can hide answer-bearing information in the encoder, buffer or decoder.

For each admissible action `a`, including byte append and each separately declared counterfactual action, require a causal one-step transition `F_a:S→S` and an observable `o:S→Y`. Actual chunk emission can take zero or more internal microsteps per byte. Charge those steps and retain alignment with the raw byte clock. The intervention domain must be closed under all claimed `F_a`.

A learned encoding `E:S→Z` is evaluated relative to this full computational state. If the architecture is stochastic, include RNG state in `S` for pathwise claims or state a probabilistic transition kernel and a distributional comparison separately. The exact Python oracle in this package handles only deterministic finite systems with discrete observations.

## 2. Right-congruence and a minimum-state bound

For a finite deterministic machine `M=(S,A,F,o)`, with the empty action word allowed, define

\[
s \sim s'
\quad\Longleftrightarrow\quad
o(F_w(s))=o(F_w(s'))\;\;\text{for every }w\in A^*.
\]

Here `F_w` applies the actions in the fixed order of `w`.

**Proposition HC-1 (future-observation quotient).** The relation `~` is a right congruence: if `s~s'` then `F_a(s)~F_a(s')` for every `a`. The quotient `S/{\sim}` admits well-defined transitions and observable outputs. If it has `N` classes, any finite code from which *all* the declared future observations can be recovered requires at least `ceil(log2 N)` bits. Encoding the equivalence class attains that state-count bound, with the transition/readout implementations charged separately.

**Proof.** The empty word gives equality of the present outputs. For each `a`, prepend `a` to every future word, which proves that the successors are equivalent. Thus transitions and observations descend to the equivalence classes. Two inequivalent states have a distinguishing suffix, so an observation-sufficient encoding must separate them. At least `N` codewords are therefore necessary, requiring `ceil(log2 N)` bits under fixed-length binary coding. The canonical quotient uses `N` codes. QED.

This is the classical behavioral-equivalence/Myhill–Nerode construction for deterministic transducers, applied here as a proposed test of learned chunks. The theorem itself is established automata theory. The prospective contribution concerns identifying the concrete neural state, intervention grammar, resource accounting and transfer behavior.

**Stronger factorization requirement.** For a *specified* encoding `E`, a deterministic abstract transition `G_a` satisfying `E(F_a(s))=G_a(E(s))` exists exactly when

\[
E(s)=E(s')\quad\Rightarrow\quad E(F_a(s))=E(F_a(s'))
\]

for every `a`. A readout `q` satisfying `o=q\circ E` additionally requires equal current outputs inside each fiber. Preserving all future observations is weaker than factorization for arbitrary over-refined encodings. The reference tests include a representation that is observationally sufficient while lacking a well-defined encoded successor.

**Finite horizon.** Define `s~_H s'` by matching current observations and all suffixes of length at most `H`. This partition can grow finer as `H` increases. Its transition maps naturally from horizon `H` to horizon `H-1`; the countdown must join the abstract state if used as a time-homogeneous transition system. A fixed-horizon quotient is not automatically a full congruence.

## 3. The buffered-byte counterexample

Take four two-bit states `S={00,01,10,11}`. Only the first bit is observed. With the action `hold` alone, two equivalence classes suffice, so the first-bit representation is exact and one state bit is enough.

Now admit `shift`, which swaps the two positions. The pair `00` and `01` has the same initial readout and the same proposed one-bit code. After `shift` the readouts are `0` and `1`. Every state becomes distinguishable by an admissible future suffix: the exact quotient needs four classes and two bits.

The code in `reference.py` computes the minimum quotient and outputs the witness `("shift",)`. This is a deterministic finite analogy for a dynamic chunker discarding a buffered byte before an allowed continuation can make it matter. It is **not** a demonstration about the behavior of Cartesia's learned H-Net.

For an actual byte stream, test suffixes that shift boundaries, rename symbols, reorder admissible game actions or introduce previously unseen syntax. A good chunk boundary is a candidate abstraction; behavioral preservation depends on the future interface being evaluated.

## 4. Approximate and stochastic representations

On a fixed reachable domain `K`, with a declared norm, abstract transition `G_a`, decoder `q` and embedding `E`, define

\[
\rho_a(s)=\|E(F_a(s))-G_a(E(s))\|,
\qquad
\rho_o(s)=d_Y(o(s),q(E(s))).
\]

For a uniformly `L`-Lipschitz abstract update and a uniform one-step bound `rho_a≤epsilon`, the existing IC-01 induction gives, for matching external action words of length `T`,

\[
e_T\le\epsilon\sum_{j=0}^{T-1} L^j.
\]

A further Lipschitz readout bound converts this to output error, under its own approximation assumption. For a stochastic next-byte predictor, choose a declared distributional metric, such as total variation or a bounded proper score, and establish the corresponding transition/readout inequalities. Mean held-out errors give distributional evidence, not uniform bounds.

The near-fiber obstruction from IC-01 applies unchanged. For any `s,s'`,

\[
2\epsilon\ge \|E(F_a(s))-E(F_a(s'))\|-L\|E(s)-E(s')\|.
\]

Every term depends on the chosen action, representation, norm and verified `L`. A changed chunk boundary can cause discontinuity; a neural routing score's smoothness does not by itself make discrete chunk decisions Lipschitz.

**Conditional boundary-margin lemma.** Suppose a boundary is emitted exactly when a real routing score `g(s)` is positive and `g` is `L_g`-Lipschitz in a fixed norm. If `|g(s)|>L_g\delta`, every perturbation of size at most `delta` preserves that boundary decision. Proof: `|g(s')-g(s)|≤L_g\delta<|g(s)|`, so the sign cannot flip. This local statement assumes a valid bound on the *complete* router state and says nothing about subsequent steps unless margins remain valid.

## 5. Resource frontier, relative to original bytes

For a trained chunking system, report a resource vector

\[
\mathcal{R}_{HC}=
(D_{\mathrm{router}},D_{\mathrm{enc}},D_{\mathrm{backbone}},
 D_{\mathrm{dec}},B_{\mathrm{buffer}},B_{\mathrm{live}},C_{\mathrm{train}},
 C_{\mathrm{infer}},M_{\mathrm{VRAM}},M_{\mathrm{host}},Q_{\mathrm{IO}},\varepsilon).
\]

Descriptions include quantized parameters, learned schedules and external dictionaries. Live state includes partial chunks, state-space memories and caches. Externally chosen routing or a pretrained teacher boundary generator has a resource cost. Track bytes consumed, emitted chunks, training FLOPs, wall time and byte-aligned evaluation loss independently. Comparing 'tokens' across different learned boundaries without normalizing to original bytes can distort data efficiency.

Define an experimental tradeoff objective, **not an H-Net training objective**,

\[
J(E,G,q)=L_{\mathrm{byte}}+\lambda C_{\mathrm{train}}
 +\mu D_{\mathcal U}(E,G,q)+\nu B_{\mathrm{live}},
\]

where `D_{\mathcal U}` measures error on a predeclared family of intervention-conditioned futures. The multipliers and normalization must be frozen before optimization. This objective seeks a Pareto frontier among prediction, recoverable future distinctions and cost; it does not assert that a single scalar objective identifies every meaningful abstraction.

## 6. Falsifiable predictions

- **HC-P1 (conjecture):** At matched *source bytes*, parameter budget and measured compute, a jointly learned chunker improves held-out bits-per-byte or rule-transfer success over a static/frozen boundary policy on at least one out-of-domain symbolic family. Failure across preregistered families and seeds rejects that bounded claim.
- **HC-P2 (conjecture):** Learned boundaries preserve more intervention-conditioned future distinctions at equal code and predictor budget than a fixed-stride or boundary-shuffled control. The finite-state collision rate and neural near-fiber errors are the probes.
- **HC-P3 (conjecture):** Joint boundary learning especially helps unseen formal syntax and Storyworld rule mutations, after accounting for tokenizer/pretokenizer priors and adaptation compute. A null result under matched controls counts against the interpretation.
- **HC-P4 (conditional):** Given a certified uniform router margin and Lipschitz bound, boundary decisions within the certified perturbation radius remain unchanged. Failure requires a violated hypothesis, numerical mistake or implementation defect; measured margins alone are not a certificate.

A reported 3–5x compute advantage is evidence about a specific benchmark setup; it does not prove HC-P1 through HC-P3, a minimum-state theorem for neural models, or consumer-GPU inference speed.

## Sources and ownership

- Hwang, Wang and Gu, [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955), 2025; [reference implementation](https://github.com/goombalab/hnet).
- Cartesia, [Hierarchical modeling](https://www.cartesia.ai/blog/hierarchical-modeling), 2025.
- Albert Gu, [2026-10-29 announcement](https://x.com/_albertgu/status/2115502110578290900). Results are claims of the authors pending independent reproduction.
- Local theory: [IC-01](../../note/intervention-preserving-compactification.md), [RC-01](../recurrent_circuits_v1/THEORY.md), [QPC-T01](../qwenputer_v1/README.md).

This package is additive. It does not alter the frozen E10 selector comparison, the IC-01 protocol, existing Qwen/LoRA data, or RC-01 resource tests.
