# PGC: assumptions, operational influence, and identifiability

Research audit, 26 November 2026. This note assesses the proposed pregeometry program and its first finite experiments. A proof about an observation map establishes a statement about that map. Claims about physical spacetime require a further bridge from the primitive model to measured physics.

## 1. The five obligations

| Target | What a successful derivation must supply | Present status |
| --- | --- | --- |
| A Lorentzian manifold | A specified primitive model, a coarse-graining map, a continuum regime, and a proof that its effective local geometry has one temporal direction | Open research obligation |
| A universal null boundary | A characteristic cone shared by the relevant physical excitations, with justified Lorentz symmetry and a controlled regime of validity | Open research obligation |
| Four spacetime dimensions | A selection mechanism whose admissible inputs permit other dimensions, together with dimension diagnostics and finite-size or continuum control | Open research obligation |
| Einstein variables and dynamics | A reason the effective observables admit a metric description and satisfy the stated locality, covariance, field-content, and derivative assumptions | Open research obligation |
| The gravitational coupling | A derived coefficient, its sign and scale dependence, and a relation to independently defined matter and length scales | Open research obligation |

In Lorentzian geometry the causal cone includes timelike directions; its boundary contains null directions. The target therefore concerns the origin of the cone field and its characteristic boundary. An observed upper speed bound supplies less structure than a Lorentzian metric.

## 2. What the reconstruction theorems establish

**Established theorem under stated assumptions.** Malament's original 1977 paper defines a spacetime as a connected, smooth, four-dimensional manifold without boundary, equipped with a smooth Lorentz-signature metric and time orientation. For past- and future-distinguishing spacetimes, a bijection preserving the chronological relation in both directions is a smooth conformal isometry. His curve-preservation theorem has a different hypothesis: preservation of continuous timelike curves in both directions. [M1]

The mathematical conclusion fixes geometry within the stipulated class up to a conformal factor. Applying it to primitive operational data requires an existence argument placing those data in that class. In particular, this use of the original theorem already supplies dimension four and Lorentzian signature. Reconstructing a conformal metric also leaves its volume or scale factor to further information.

Sorkin's causal-set notes make the approximation problem explicit. Their dimension estimators are designed for intervals approximating Minkowski Alexandrov intervals, with sufficient manifold-like structure. The notes distinguish these estimators from the conjectured uniqueness of a continuum approximation. [S1] The status described here is the status of those cited results; the note makes no claim about the complete 2026 literature on continuum reconstruction.

## 3. Correcting the operational influence definition

**Elementary proposition, proved here.** Assume a tensor factorization

\[
\mathcal H=\mathcal H_A\otimes\mathcal H_B\otimes\mathcal H_R
\]

and an unconditional completely positive trace-preserving map \(\Phi_A\) acting on \(A\). Its dual is unital. Thus every observable supported on the disjoint factor \(B\) obeys

\[
(\Phi_A^*\otimes\mathrm{id}_{BR})(I_A\otimes O_B\otimes I_R)
=I_A\otimes O_B\otimes I_R.
\]

Consequently an instantaneous expression of the form

\[
\sup_{\Phi_A,O_B}\left|\omega(\Phi_A^*O_B)-\omega(O_B)\right|
\]

vanishes for disjoint factors, including entangled input states. The tensor extensions are implicit in that shorthand. Conditioning on a selected measurement outcome changes the mathematical operation and requires its outcome record to be specified.

A useful delayed definition declares a later channel \(\mathcal T_t\), an allowed state set \(\mathcal S\), and allowed interventions \(\mathcal I_A\):

\[
I_{A\to B}(t)=\sup_{\rho\in\mathcal S,\,\Phi_A,\Psi_A\in\mathcal I_A}
\frac12\left\|
\operatorname{Tr}_{\bar B}\mathcal T_t[(\Phi_A\otimes\mathrm{id})(\rho)]
-\operatorname{Tr}_{\bar B}\mathcal T_t[(\Psi_A\otimes\mathrm{id})(\rho)]
\right\|_1.
\]

This normalization lies in \([0,1]\). It measures optimal distinguishability of the two output states on \(B\). The equivalent supremum over Hermitian observables of operator norm at most one has an additional factor \(1/2\). An expectation-value supremum needs such a norm bound or a bounded measurement range; scaling an unrestricted observable otherwise makes every nonzero response arbitrarily large.

The definition supplies register identities, subsystem composition, allowed preparations, a channel, and a temporal index. Each belongs in the assumption ledger. Inferring structure from such experiments concerns the declared interface and its evolution. Extending the result to an origin of subsystem factorization or time requires additional work.

The preparation class can change the measured influence. For the binary rule \(B=A\oplus R\), an independent uniform hidden bit \(R\) makes the marginal distribution of \(B\) uniform for every setting of \(A\). Fixing \(R\) gives perfect influence from \(A\) to \(B\). The two experiments use different background contexts. A report must specify which contexts the supremum ranges over.

### Influence edges and partial orders

**Exact counterexamples.** A threshold on influence need not produce a transitive relation. Consider binary channels \(A_0\to B_1\to C_2\), each independently flipping its input with probability \(1/10\). Their single-link influence is \(4/5\); the composed influence is \(16/25\). At threshold \(7/10\), both adjacent edges survive while the endpoint edge fails. An assumed temporal ordering remains available throughout this example.

Cancellation can occur even at zero threshold. Apply \(\mathrm{CNOT}_{A\to B}\) twice. The first gate permits influence from \(A_0\) to \(B_1\), and an intervention at \(B_1\) affects \(B_2\). The complete pair of gates is the identity, so changing \(A_0\) leaves \(B_2\) unchanged. Declaring interventions at intermediate events changes which channel is tested.

Register-level influence can also have cycles. A SWAP exchanges the states of two registers, creating both \(A\to B\) and \(B\to A\) when event times are omitted. Keeping input and output times separate removes this particular cycle through the supplied update order. Taking a transitive closure constructs a reachability relation whose edges can exceed the directly measured influence relation.

## 4. Finite propagation and Lorentz structure

**Exact counterexample.** On an integer lattice, let \(x_i(t+1)=x_{i-1}(t)\). Changes propagate exactly one site per update. The rule singles out lattice sites and an update clock, and its causal graph has discrete translation symmetries. A finite propagation radius therefore permits models with such a preferred microscopic structure.

The original Lieb–Robinson theorem assumes a quantum spin system with a finite-range interaction and space and time translation structure. It bounds distant commutators outside a sufficiently fast propagation envelope. This result supplies a rigorous locality estimate for that model class; exact vanishing outside a cone is stronger than the stated asymptotic bound. [L1]

A Lorentzian limit calls for separate evidence about the low-energy dispersion relations and boost symmetry. Cone universality adds another condition: multiple propagating sectors must share the same effective characteristic boundary in the claimed regime. A common upper bound on their speeds permits slower sectors and distinct characteristic surfaces.

## 5. Keep the dimension diagnostics explicit

| Quantity | Definition or measurement | Information needed |
| --- | --- | --- |
| Manifold dimension | Local charts with open subsets of \(\mathbb R^d\) | A topology and a manifold claim |
| Hausdorff or volume-growth dimension | Scaling of volume with radius | A metric, a measure, and a scale range |
| Spectral dimension | \(d_s(\sigma)=-2\,d\log P_{\rm return}(\sigma)/d\log\sigma\) | A diffusion generator and diffusion-time window |
| Order dimension | Minimum number of linear orders whose intersection is the given partial order | A partial order; its interpretation as spacetime needs another argument |
| Minkowski embedding dimension | Smallest dimension supporting a specified kind of causal embedding | The admissible embedding class, density or faithfulness conditions, and tolerance |
| Representation dimension | Size of a chosen latent space or code | A representation class, numerical precision, and complexity accounting |

These definitions supply distinct questions. Their values can agree in a controlled geometric regime; that agreement needs justification.

Ambjørn, Jurkiewicz, and Loll measured a scale-dependent spectral dimension in a CDT ensemble built from causal four-dimensional piecewise-flat manifolds. The paper reports values near four at large scales and near two at small scales. Its topological building blocks, diffusion observable, and dynamical effective dimension play different roles. [D1] Recovering a dimension used by a synthetic generator validates a reconstruction procedure within that test. A mechanism that selects four dimensions must define a broader admissible ensemble or dynamics and establish why the four-dimensional regime dominates.

## 6. Einstein dynamics and the coupling

**Conditional classification.** Lovelock classifies symmetric, divergence-free rank-two tensors constructed locally from the metric and its derivatives through second order. In four dimensions, the available tensors are constant linear combinations of the Einstein tensor and the metric. [G1] Writing

\[
a\,G_{ab}+b\,g_{ab}=T_{ab}
\]

exposes the remaining coefficients. The classification requires a metric description and a spacetime dimension; its assumptions restrict the local field equations. Establishing those assumptions from microscopic observables remains a separate task. Additional fields, higher derivatives, and nonlocal dependence expand the available theories.

**Conditional thermodynamic derivation.** Jacobson assumes local Rindler horizons, an entropy density proportional to horizon area, the Unruh temperature, local equilibrium, and stress-energy conservation. In units \(c=k_B=1\),

\[
S=\eta A,\qquad T=\frac{\hbar\kappa}{2\pi},\qquad
\delta Q=T\,dS
\]

lead to

\[
G_{ab}+\Lambda g_{ab}=\frac{2\pi}{\hbar\eta}T_{ab},
\qquad G_N=\frac{1}{4\hbar\eta}.
\]

The original paper explicitly leaves \(\eta\) undetermined and \(\Lambda\) free. [J1] With SI units restored and entropy written \(S/k_B=\eta A\), dimensional consistency gives \(G_N=c^3/(4\hbar\eta)\). A microscopic calculation of \(\eta\), including its physical area unit, would supply the missing scale input. The local Lorentzian horizon structure already enters the derivation.

**Induced-gravity mechanism with inputs.** Sakharov expands the effective vacuum action in curvature and identifies its term proportional to \(R\) with the Einstein action. His original discussion explicitly ties the coupling to particle masses, their equations of motion, and a momentum cutoff, and also includes higher-curvature terms. [K1] Curved geometry and quantum matter enter the setup. This mechanism provides a way to calculate a gravitational coefficient after the microscopic content and regulation have been supplied. A PGC proposal would need to derive or justify those inputs and state the regime in which the Einstein term controls the dynamics.

## 7. Identifiability from operational records

**Elementary theorem, proved here and formalized separately.** Let \(W\) be a class of worlds, \(T\) a space of accessible records, and \(o:W\to T\) the observation map. Let \(p:W\to P\) be a target property. If

\[
o(w_0)=o(w_1),\qquad p(w_0)\ne p(w_1),
\]

then a decoder \(d:T\to P\) cannot satisfy \(d(o(w))=p(w)\) for every \(w\in W\). Indeed, applying \(d\) to the equal records forces equal decoded values, contradicting the displayed difference in \(p\). Every deterministic postprocessing of the records preserves their equality.

This theorem applies to a physical target only after one exhibits the required pair of models within the specified admissible class. A finite observation window can, for example, miss a controller that agrees through the last tested step and changes its rule afterward, if that controller belongs to the class. Extra inaccessible degrees of freedom offer another conditional example. Restricting the class can remove such ambiguities and then requires a physical reason for that restriction.

Compression selects among descriptions under declared coding choices. An objective such as \(L(M)+L(D\mid M)\) requires a model class, a code, and a data-fidelity convention. The selected description inherits those choices. A short representation of finite records supplies evidence about that code and those records; physical dimension and metric signature need an identifiability argument tying them to accessible interventions.

## 8. Scope of the first experiment

The finite-circuit experiment can establish exact register influence, demonstrate cancellation, check equivariance under register relabeling, and expose indistinguishable record classes. Its report should state the declared register factorization and update schedule. A successful continuum result would additionally need a defined large-system family and a convergence argument, followed by the five physical obligations in section 1.

The companion Lean module formalizes four statements about equality under an observation map. Its worlds and properties are arbitrary types. It contains no Lorentzian manifold, continuum limit, quantum channel, dimension-selection rule, or gravitational equation. “Observation-preserving” in that module means preserving an accessible record; quantum trace preservation concerns a different operation.

**LEAN-CHECKED:** `OperationalIdentifiability.lean` compiled with Lean 4.35.0. The compiler returned exit code zero and printed four transitive-axiom reports, each empty. `lean_validation.json` pins the source hash and compiler version; `compiler_release_receipt.json` records verification of the downloaded release against the digest reported by the official Lean GitHub release API.

## Primary sources

- **[M1]** D. B. Malament, *The class of continuous timelike curves determines the topology of spacetime*, Journal of Mathematical Physics **18**, 1399–1404 (1977), [DOI](https://doi.org/10.1063/1.523436). Definitions in section 2 and Theorems 1–2. [Readable scan text](https://www.scribd.com/document/490611872/Malament-D-The-CLass-of-Continuous-Timelike-Curves-etc-Spacetime-1977-pdf). The publisher endpoint was unavailable during this audit; the author's [publication list](https://sites.socsci.uci.edu/~dmalamen/bio/vita.html) confirms the paper and links its publisher version.
- **[S1]** R. D. Sorkin, *Causal Sets: Discrete Gravity*, [arXiv:gr-qc/0309009](https://arxiv.org/abs/gr-qc/0309009), especially “Dimension and length” and the preceding discussion of continuum approximation. [Full text](https://arxiv.org/html/gr-qc/0309009v1).
- **[L1]** E. H. Lieb and D. W. Robinson, *The finite group velocity of quantum spin systems*, Communications in Mathematical Physics **28**, 251–257 (1972), [DOI and original abstract](https://link.springer.com/article/10.1007/BF01645779).
- **[D1]** J. Ambjørn, J. Jurkiewicz, and R. Loll, *Spectral Dimension of the Universe*, Physical Review Letters **95**, 171301 (2005), [arXiv:hep-th/0505113](https://arxiv.org/abs/hep-th/0505113). [Full text](https://arxiv.org/html/hep-th/0505113v2), especially “Measuring the spectral dimension.”
- **[G1]** D. Lovelock, *The Einstein Tensor and Its Generalizations*, Journal of Mathematical Physics **12**, 498–501 (1971), [DOI](https://doi.org/10.1063/1.1665613), [publisher abstract](https://pubs.aip.org/aip/jmp/article-abstract/12/3/498/223441/The-Einstein-Tensor-and-Its-Generalizations).
- **[J1]** T. Jacobson, *Thermodynamics of Spacetime: The Einstein Equation of State*, Physical Review Letters **75**, 1260–1263 (1995), [arXiv:gr-qc/9504004](https://arxiv.org/abs/gr-qc/9504004). [Full text](https://arxiv.org/html/gr-qc/9504004v2), equations (1)–(6) and the paragraphs specifying the entropy density and local-equilibrium assumptions.
- **[K1]** A. D. Sakharov, *Vacuum quantum fluctuations in curved space and the theory of gravitation* (1967/1968), reprinted in Soviet Physics Uspekhi **34**, 394 (1991), [official PDF](https://www.ufn.ru/ufn91/ufn91_5/ufn915i.pdf), [DOI](https://doi.org/10.1070/PU1991v034n05ABEH002498).
