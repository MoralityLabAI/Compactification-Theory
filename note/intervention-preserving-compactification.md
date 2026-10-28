# Intervention-preserving compactification

Status: research bridge and proposed experiment, 2026-10-28. This addition contains no new model measurements or Lean proof compilation.

The canonical mathematical note, Python reference, tests, and source-hashed replay receipt live in [EvalMath](https://github.com/MoralityLabAI/EvalMath/tree/adeb45fbb6b611e0e1fc38989e28f0d4d621698b/research/interventional_compactification). The local [experiment protocol](../experiments/interventional-compactification/README.md) applies that work to this repository's capacity and precision questions. The implementation is maintained in EvalMath; this repository does not vendor a second copy.

## What the reduction must preserve

Fix the computational state h, an intervention-closed domain K, operation maps F_a, and observables g. A representation pi:K -> Z proposes abstract operations G_a and a readout q. Its errors are

    rho_a(h) = ||pi(F_a(h)) - G_a(pi(h))||
    rho_g(h) = ||g(h) - q(pi(h))||.

All metrics and encoding scales must be fixed before comparison. A layer activation must carry sufficient surrounding context for F_a to be a function. For token-level dynamics this can require KV caches, position, and the computation phase. Training dynamics additionally require the optimizer and random-state semantics.

The near-fiber obstruction gives a falsification target. If G_a is L-Lipschitz and the uniform commutation error is at most epsilon, then every pair satisfies

    2 epsilon >= ||pi(F_a(h)) - pi(F_a(h'))|| - L ||pi(h) - pi(h')||.

A pair with a positive right-hand side forces an error floor for the entire declared L-Lipschitz model class. Numerical implementations require rounding/error control before treating the computed number as a rigorous lower bound. The imported EvalMath reference is exploratory floating-point code.

With readout approximation error eta, readout Lipschitz constant D, and matching externally supplied action words of length T, the handwritten simulation argument gives

    output_error(T) <= eta + D epsilon sum_{j=0}^{T-1} L^j.

The hypotheses must hold on the trajectories being compared. Closed-loop action agreement additionally needs a decision-margin argument at every step.

## Capacity and precision accounting

The existing [bits/depth/Lipschitz note](https://github.com/MoralityLabAI/Compactification-Theory/blob/f33f6db451c081a8a4ec8c92f81ebafc84b26bb4/note/bits-depth-lipschitz.html) studies representation resources and approximation complexity. This extension asks how those resources change when future interventions must remain predictable.

Record compressed dimension k alongside the number of bits used to encode each coordinate. A fixed k-coordinate representation using b bits per coordinate has at most 2^(k b) codewords. Learned encoder parameters, the abstract transition/readout parameters, and any external lookup table must also be charged to the system's description budget. A real-valued coordinate with unrestricted precision or a complicated encoder can hide substantial information.

The weight-to-function sensitivity discussed in the existing note and the state-transition Lipschitz constant L above have different domains. Their values cannot be substituted for one another without a connecting argument. Normalization also changes distances and constants, so report it explicitly.

The Petersen–Zech source map in EvalMath identifies approximation results for Barron, compositional, and manifold-regular functions. Applying those rates to a quotient requires separately establishing its existence and regularity. A small empirical activation rank alone supplies neither an intervention-preserving quotient nor its required coverage.

For evaluation, the complexity budget covers the learned encoder together with G_a and q. Statistical covers concern this full loss family when it was selected using the same data. Independent held-out evaluation of a fixed fitted hypothesis supports a simpler distributional claim. A uniform certificate additionally requires a valid net of the declared intervention domain and residual regularity bounds.

## Meaning of compactification

This work uses the term for representation reduction. Topological compactness, finite-dimensionality, and decidability each require additional hypotheses. Behavioral equivalence over all admissible future words defines an exact quotient. With a finite remaining horizon, equivalence is horizon-indexed and transitions consume that horizon; a fixed-horizon equivalence cannot automatically be treated as a congruence.

The reachable state domain should be declared from the start. A cached finite dataset is not automatically closed under every specified intervention. An experiment that measures held-out mean error should report that distributional scope.

## Ownership and evidence

EvalMath owns the derivations and executable reference. Compactification-Theory owns this resource-accounting interpretation and the prospective compression protocol. Existing EvalLang work keeps its current owners. The protocol is separate from the frozen E10 selector comparison and does not change its hypotheses, selectors, seeds, or acceptance rules.

At import, this repository's inspected baseline is `f33f6db451c081a8a4ec8c92f81ebafc84b26bb4`. Existing adapter experiments, results, and follow-up registrations remain unchanged. The linked EvalMath receipt reports five synthetic reference tests; it provides no validation of a transformer reduction or a percolation advantage.
