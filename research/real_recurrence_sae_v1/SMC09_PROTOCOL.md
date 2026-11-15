# SMC-09: Verified finite endpoint ordering of learned SAE causal directions

**November 15, 2026.** Continuation of merged [SMC-08 source-locked curvature results](SMC08_RESULTS.md) and [MathCan OLM-08](https://github.com/patrickdugan/MathCan/pull/70). Companion pure mathematics: [OLM-09 exact endpoint dominance](https://github.com/patrickdugan/MathCan/blob/research/olm09-endpoint-dominance-20261115/research/orbit_lock/OLM09_ENDPOINT_CAUSAL_DOMINANCE.md).

## The gap addressed

The SMC-08 pretrained Sudoku recurrent model showed a 15/16-puzzle mean advantage of active SAE decoder directions over seeded isotropic controls at fixed perturbation radii. A global exact-real Hessian upper bound of about \(10^{53.606}\) is mathematically sound but too large to certify any of the experimentally useful radii. The LSV-17/SMC-08B locally propagated Arb Taylor bounds were much tighter yet could certify a nonzero signed winner-versus-runner margin displacement only up to radii roughly \(10^{-8}\).

This stage changes the **mathematical certification goal**. For each frozen source hidden state and selected intervention direction, the exact recurrent model is evaluated at both the clean and perturbed endpoints using outward-rounded Arb interval arithmetic. The resulting ten-logit output displacement is enclosed directly, and the two squared-L2 effect intervals are compared.

This certifies **finite endpoint effect ordering** at the evaluated radius for the *exact-real mathematical network with frozen binary weights and inputs*. It is not a uniform curvature statement over all intermediate points or nearby directions. The direction itself is selected by numerical SAE training/Jacobian estimation and then frozen for validated endpoint evaluation.

## Source lock and experimental contract

- Frozen external 44,001-parameter TRM-inspired Sudoku recurrence, exact source commit f0406195fece4d0db29ced28f3a69d1993d95b1e.
- Model checkpoint SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1, verified before weights-only loading.
- Roster SHA-256 eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.
- Training 32 Sudoku puzzle IDs, validation 16, heldout 16, with all tokens from each puzzle in one cohort. Repeated test inspection from SMC-07/08 makes this explanatory follow-up rather than independently confirmatory evaluation.
- SAE: frozen recipe TopK=4 ReLU 48→96→48, 300 optimizer steps, validation checkpoint selection; training seeds 71,89,113.
- For each heldout puzzle, the first blank goal token receives an intervention at the post-stage-one hidden state. Freeze the other 80 tokens. Evaluate the subsequent three recurrent blocks and all ten final goal logits.
- Directions: highest-active learned SAE decoder atom, uniformly sampled inactive atom, seeded isotropic vector, and *numerically estimated* leading right singular direction of the 10×48 goal-logit Jacobian.
- Radii: **0.02, 0.2, 0.8**. Each direction is interpreted as its 48 exact FP64 dyadic components and renormalized by its *exact-real* Euclidean norm. Radius is an exact input rational. This ensures equal intervention norm at the mathematical definition level.
- For each feature and radius, record the Arb interval for the **squared** Euclidean norm of the ten-logit displacement from the clean baseline. Declare a strict active-over-control (or control-over-active) comparison only when the two interval enclosures are disjoint. Overlapping enclosures are **inconclusive**.
- Run 160-bit Arb arithmetic via the existing LSV-17 validated erf-GELU/LayerNorm forward implementation, using precisely the stored checkpoint binary weights and original FP32 stage-one anchor interpreted as an exact binary state.

## Certificate theorem

Let \(I_u\) and \(I_v\) be valid interval enclosures of the squared output displacements of the exact-real model. [OLM-09](https://github.com/patrickdugan/MathCan/blob/research/olm09-endpoint-dominance-20261115/research/orbit_lock/OLM09_ENDPOINT_CAUSAL_DOMINANCE.md) proves

\[
\inf(I_u)>\sup(I_v)
\quad\Longrightarrow\quad
\|F(h+ru)-F(h)\|_2>\|F(h+rv)-F(h)\|_2.
\]

It avoids the OLM-08 curvature premise by evaluating the two specific endpoints. The resulting certificate is valid **at the declared amplitude for the declared exact input**, with no claim of the same ranking at every smaller radius. For an uncertainty neighborhood around the intervention direction, a separate Lipschitz/error budget would still be needed.

## Execution and provenance

Source:
- [Validated endpoint solver](smc09_endpoint_arb.py)
- [Exact-unit and interval-ranking controls](test_smc09_endpoint_arb.py)
- [Pilot and full hosted workflow](../../.github/workflows/smc09-endpoint-interval.yml)

Reproduce from the repository root:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc09_endpoint_arb.py' -v
    python research/real_recurrence_sae_v1/smc09_endpoint_arb.py --fetch --max-test-puzzles 2 --sae-seeds 71 --radii 0.02,0.2 --out runs/smc09/pilot.json
    python research/real_recurrence_sae_v1/smc09_endpoint_arb.py --fetch --max-test-puzzles 16 --sae-seeds 71,89,113 --radii 0.02,0.2,0.8 --out runs/smc09/full.json

The full receipt preserves each puzzle/seed/direction/radius interval and its comparison decision. Once the workflow succeeds, a compact result summary can be committed without copying the full CI artifact. Any failure to separate intervals must be recorded as an inconclusive case; it is inappropriate to substitute FP64 numerical measurements for a missing interval certificate.

**Interpretation boundary.** Even a rigorously certified finite difference does not establish that an SAE feature corresponds to a semantic unit of Sudoku reasoning or improves performance. The experiment tests a narrower mathematical claim: a selected active decoder direction changes the exact pretrained network's output more or less than an explicit control under a fixed equal-norm intervention.
