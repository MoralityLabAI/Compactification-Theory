# LSV-13: Validated local rank of a frozen pretrained recurrent model

**Run date:** 2026-11-04. **Model:** 44,001-parameter externally pretrained TRM-inspired Sudoku model (MIT-licensed third party), source revision `f0406195fece4d0db29ced28f3a69d1993d95b1e`, checkpoint SHA256 `3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1`, 64-case roster SHA256 `eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611`.

## Main results

Two CPU GitHub Actions runs with `python-flint==0.9.0` used the LSV-12 frozen selection of four input coordinates per case, along with the four goal-history observations at depths 0 through 3 after stage one:

- [First complete 64-case determinant audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41817808622): 64/64 selected 4x4 Jacobian minor determinant balls excluded zero, all 64 final winning and runner-up classes were strictly separated in the exact-real model. The verifier completed in 54.818 seconds on a CPU runner.
- [Second complete 64-case derivative-error audit](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41818387044): 64/64 rank-four witnesses; all 1,024 binary64 autograd derivatives were individually compared to rigorous Arb-enclosed exact-real derivatives; all 64 cases met the previously frozen strict maximum-entry error threshold from LSV-12. The per-case strict budget checks passed in 72.165 seconds.

The lower bound on the absolute determinant computed independently from the first run's **serialized 16-entry interval tables** has a minimum of about `1.09853e-5` across the 64 cases. This is a conservative lower bound obtained from the exact rational Leibniz expansion on decimal endpoints; the final repository receipt should be used for the complete raw evidence.

## Claim strength

For each selected frozen stage-one state, the exact-real shared block has goal-history Jacobian rank four. By the chain rule, a smooth local state encoder with fewer than four real coordinates cannot exactly reproduce the complete four-component goal history through a smooth decoder on a neighborhood of that point.

This result concerns the **exact-real lift of checkpoint weights and observed binary32 stage-one states**. It does not certify the floating-point hardware's forward passes, a finite perturbation radius, general task-level reasoning, convergence, or a global finite-dimensional invariant quotient. The 4-row goal-history rank is at its maximal possible value and is not, by itself, evidence of surprising cognitive competence. The reusable applied technique is the inexpensive discovery of small numerical witnesses followed by exact-ball differentiation and independent rational proof replay.

See [theorem and provenance](VALIDATED_MINOR.md), [Arb verifier](validated_minor.py), [source-locked runner](run_validated_minor.py), and [standalone rational verifier](verify_serialized_minors.py).
