# Algernon algebraic machine: executable mathematics (2026-11-04)

**Status:** deterministic CPU tests were executed on constructed finite-state examples. These tests do not establish that LOTUS or Qwen implements these maps. Lean source is provided for a kernel audit and is only considered proved in Lean after its CI job passes. No Circom compiler or Qwenputer runtime is invoked by this package.

## Research question

Can a trained recurrent representation preserve the algebra of an instruction set, so that valid primitive transitions compose into correct deeper programs? This module isolates the formal interface before any GPU experiment.

Mathematical roles:

| Object | Algebraic interpretation | Executable artifact |
| --- | --- | --- |
| Instructions | Free monoid over instruction alphabet | `machine.py`, `lean/Algernon/Semiconjugacy.lean` |
| Decoder | Quotient/factor map π from hidden state H to machine state S | `machine.py` |
| Computation graph | Cellular sheaf with edge maps Aₑ and coboundary δ | `sheaf.py` |
| Inconsistent trace | Affine 1-cochain outside im(δ), yielding class in H¹ | `sheaf.py` |
| Sparse repair | Low-rank intervention supported on U in linearized dynamics | `sparse.py` |
| Depth robustness | One-step L¹ transition error accumulates at most linearly | `sparse.py` |

## 1. Instruction algebra and semiconjugacy

For each instruction `a`, let `Fₐ : H → H` be the hidden machine and `Tₐ : S → S` its exact semantics. A decoder π should obey `π ∘ Fₐ = Tₐ ∘ π` on every *reachable* hidden state.

By induction on word length, the property extends to every program composed from those instructions. This is a **semiconjugacy** (a factor map when π is surjective); injectivity and complete state reconstruction are extra conditions. The kernel proof in `Semiconjugacy.lean` includes word compositionality, quotient congruence, invariant-subset iteration, semantic periodic returns, and a three-edge cycle's equivalence with a fixed point of its monodromy. It assumes the generator property; finite training examples alone do not prove it.

`machine.py` exhaustively checks 27 deterministic three-state transitions lifted to six explicitly constructed hidden states and all words over two instructions of length 0–5. Corrupting one hidden transition creates a measurable defect. The finite tests are controls for a future learned neural decoder.

**Conjugacy and period:** if `Fᵐ(h)=h` and π semiconjugates F and T then `Tᵐ(πh)=πh`. The minimum semantic period therefore divides m when a semantic cycle exists. A hidden cycle may be a multiple of a semantic cycle, since π can identify different hidden states.

## 2. A finite-field cellular sheaf on an oriented cycle

Let the cycle have m vertices with fiber `Fₚᵈ` at each vertex. Each edge carries a d×d linear transition `Aᵢ` and

`(δs)ᵢ = sᵢ₊₁ − Aᵢ sᵢ`, with indices modulo m.

`H⁰ = ker δ`, `H¹ = coker δ`. For a single directed cycle with `M=Aₘ₋₁...A₀`, eliminating the intermediate vertex variables gives:

`H⁰ ≅ ker(I−M)` and `H¹ ≅ coker(I−M)`.

For a prescribed affine 1-cochain b, a global state assignment exists **iff** b∈im δ. Equivalently `(I−M)s₀ = c`, where `c=bₘ₋₁ + Aₘ₋₁bₘ₋₂ + ... + Aₘ₋₁...A₁b₀`. This gives a computable obstruction over finite fields.

A useful test is the three-edge identity cycle over F₅. Increments (1,1,1) sum to 3, so global consistency is impossible; increments (1,1,3) sum to 0 and admit a solution. A *twisted* cycle with monodromy 2 has H⁰=H¹=0. The sheaf invariants depend on transition maps as well as the graph topology.

**Boundary:** This is a linear cellular sheaf, not the nonlinear Circom constraint variety. R1CS Boolean/one-hot/range constraints must be checked separately. This module uses a small-prime modular reference solver. A Circom witness is only considered accepted when its complete polynomial constraints and public-input binding pass the exact checker in `../spec.py`.

## 3. Sparse weight repair as a linear intertwiner

For local real-valued linear maps, decoder `P:[h]→[s]`, neural step `M:[h]→[h]`, semantic transition `A:[s]→[s]`, define `D=PM−AP`. A support subspace `U:[h]→[r]` permits edit `ΔM=UB`. The optimal residual is

`min_B ||D+PUB||_F = ||(I−Π_{PU})D||_F`.

An exact repair exists precisely when `im(D) ⊆ im(PU)`. `sparse.py` computes the least-squares edit and independently checks its residual. In the built example one edit direction removes a norm-3 defect; an orthogonal direction leaves norm 3. These are **local Jacobian algebra** statements, not weight-edit results for Qwen, and do not account for nonlinear state shifts or discrete quantization.

## 4. Depth-error theorem

For a column-stochastic transition A and decoded distributions qₖ, nonexpansiveness in L¹ yields

`||qₙ−Aⁿq₀||₁ ≤ Σₖ ||qₖ₊₁−Aqₖ||₁`.

If the true initial state is eₛ and `||q₀−eₛ||₁ ≤ ε₀`, then the target error is ≤ `ε₀+Σ εₖ`. When this total is <1 the correct target has majority mass, so argmax is correct. The guarantee needs accurate initialization and decoder semantics. A constant uniform state may have zero local defect and still represent no meaningful computation.

## Commands

From repository root, with NumPy installed:

```bash
python -m pip install -r research/algernon/algebraic_machine/requirements.txt
python -m unittest discover -s research/algernon/algebraic_machine/tests -v
python -m research.algernon.algebraic_machine.experiments --out runs/algernon/algebraic_machine/reference.json
```

For the independent Lean project, from `research/algernon/algebraic_machine/lean/`:

```bash
lake build
lake env lean Algernon/Semiconjugacy.lean
```

The GitHub Actions workflow runs CPU tests, compares the deterministic JSON receipt, and runs the Lean build. The Lean toolchain is pinned locally to `v4.28.0` with no Mathlib dependency.

## Next use with LOTUS

1. Capture genuine post-loop hidden states using `../lotus_bridge/` and train π on a frozen, group-disjoint roster. Measure one-step defect on held-out states, including counterfactual prompt changes.
2. Estimate local Jacobian `M` of the recurrent step and `P` of the decoder on calibrated states. Compare low-rank edit supports by `im(PU)` coverage of D. Include random and norm-matched controls.
3. Measure whether latent cycles are compatible with target periods and whether decoded transitions form globally consistent traces. A low linear defect is diagnostic and does not replace exact polynomial witness validation.
4. Audit GPU cost, quantization sensitivity, model skill retention, and depth generalization before attempting a real Qwen weight splice.

## Evidence status

The frozen CPU reference JSON in `results/reference.json` contains `execution_class: cpu_reference_only`. The finite models are constructed examples, and the least-squares repair uses NumPy. Lean CI is independently reported. This package contains no learned model parameters, trained decoder, zk proof, or measured neural gain.
