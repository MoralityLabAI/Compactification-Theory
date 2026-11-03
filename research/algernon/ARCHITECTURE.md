# Algernon architecture: compiled exact kernel + recurrent learned kernel

## Four distinct computational objects

1. **Token-time state:** for a DeltaNet head, a matrix S_t changes as token t is processed. One-hot keys and controlled gates support addressed reads and writes. Qwenputer uses this to store CPU memory.
2. **Algorithm-time recurrence:** a reused operator F updates a latent reasoning state z_k across k internal refinement iterations. This is the TRM/LDT target.
3. **Layer-depth propagation:** the frozen Qwen transform maps h_l to h_(l+1) with distinct pretrained layer weights. A graft at layer l changes downstream computation.
4. **Weight edit:** a permanent sparse delta to selected matrices, which may help Qwen consume a verified graft output but does not by itself establish algorithm insertion.

These operate on different clocks and state spaces. The experiment must explicitly define the conversion between them.

## Proposed joint model

Given prompt x, a frozen Qwen residual h_l(x), optional exact-machine state m_t, and a learned recurrent kernel:

~~~text
m_(t+1), c_t = C(m_t, instruction_t)          exact compiled transition
z_0 = P(h_l, c_t, mask)                       state encoder
z_(k+1) = F_phi(z_k, h_l, c_t)                shared recurrent refinement
r = Verify(Project(z_K), c_t)                typed/semantic check
h'_l = h_l + gate(h_l,r) * Q(z_K, r)         bounded, reversible residual write
y = Qwen_frozen_tail(h'_l)                    continue language processing
~~~

The first R-track prototype omits C and passes a zero-valued exact-state input. The C-track can be tested with its reference machine independently. Integrate only after each component passes its own gate.

The gate starts at zero. Record per-token and per-layer gated-write norms and clamp them relative to the original residual norm. Fail closed on nonfinite values or invalid typed projections. Weight-space edits later have a separate edit budget and baseline.

### Proposed objective (empirical)

L = L_task + lambda_KL * KL(p_graft || p_base) + lambda_cost * measured_compute
    + lambda_regress * existing_skill_regression + lambda_state * invalid_state_rate

Measure the terms separately. Gradient training is permitted **only** for the bridge/recurrent track with declared data. The released Qwen weight-edit reproduction is a distinct no-gradient intervention. Do not mix gradient and no-gradient evidence under the same label.

## Starting placements

Use Qwen3.5-0.8B to sweep layers 8, 12, 15, 19, 21 (zero indexed), as candidates rather than causal truths. The published sparse repair acted in 15 and 19. Compare a single receiver with distributed low-rank receivers at equal total trainable parameters.

For each candidate site:
- estimate effective residual-feature rank on calibration-only examples;
- record normalized Jacobian/sensitivity of the output to the injected state;
- check that P and Q are trained without exposure to test labels;
- compare K=0,1,2,4,8 (optional 16) recurrence;
- add a no-recurrence feedforward control with matched trainable parameters and wall-clock FLOP/time receipts;
- add shuffled-trajectory, zero-state, random-direction and frozen-random-kernel controls;
- report both learning gains and base-model retained skills.

**Failure mode to watch:** an added recurrence can merely increase the opportunity to loop or inject an answer label. Held-out, disjoint tasks and no-answer-leakage checks are mandatory.

## Exact kernel interface (proposed)

The compiled kernel should export a strict record:

~~~json
{"program_sha256":"<sha>","input_sha256":"<sha>","state_before_sha256":"<sha>","state_after_sha256":"<sha>","output_bytes_hex":"...","instruction_count":0,"trace_tokens":0,"exact_reference_match":true}
~~~

The joint processor consumes the *meaning* of this record through an explicitly learned interface; it never treats the reference comparison as user-visible reasoning. The reference machine remains outside the model for evaluation only.

Qwenputer's existing call bridge is token-mediated. Direct latent-state execution, without Qwenputer trace tokens, is a **new design to test**; the current implementation cannot be treated as demonstrating it.

## Position relative to the prior projects

- Compactification Theory supplies adapter capacity and precision tests, including multi-site receiver comparisons.
- HybridTRMLDT supplies explicit recurrence, typed membrane/projections and VPD diagnostics.
- Qwenputer supplies a compiled circuit construction and exact execution control.
- Takehi et al. supply the frozen sparse-repair example and bit-exact checkpoint reconstruction.

A valid Algernon result requires that the new capability survives counterfactual intervention on the added component. Promotion to a 27B Bonsai/1-bit target requires a separate quantization and throughput study.

## Learning criterion

A learning computer passes when an adaptive neural component demonstrably improves on *new held-out problems*, under an experiment that distinguishes learning in its parameters from computation with a fixed program. Report whether adaptation occurred by gradient update, residual memory write, inference-time recurrence, or sparse model edit, because these mechanisms represent different kinds of learning.
