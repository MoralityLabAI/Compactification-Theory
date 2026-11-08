# Transporting a critic through self-rewrites

## 1. Exact finite-state transport

Fix a common Markov state space, discount 0 <= gamma < 1, and policy-induced kernel P. Let r include the policy-dependent expected immediate reward. Old and new value functions satisfy their Bellman equations. Subtracting them gives

    V_new - V_old = (I - gamma P_new)^(-1)
        [r_new - r_old + gamma (P_new - P_old) V_old].

The two terms isolate reward drift and policy/dynamics drift. The rational reference verifies this identity on 96 seeded finite examples, with substochastic kernels representing omitted absorbing states.

For any inherited candidate v, define b = r_new + gamma P_new v - v. The contraction inequality gives

    ||v - V_new||_infinity <= ||b||_infinity / (1-gamma).

The bound needs the true full-state residual. A minibatch residual under an old visitation distribution cannot stand in for this quantity. Generalization, coverage and model-error bounds would be needed for an empirical certificate.

## 2. Transfer can be harmful

In a one-state discounted chain with gamma=.9, old per-step reward 1 gives V_old=10. Changing the reward to zero makes V_new=0. A reset critic is exact and the inherited critic is maximally stale. The matched reference table compares reset and inherited initializations after four Bellman sweeps in each policy-shift/reward-shift cell. Its positive cells and adverse control are synthetic arithmetic evidence.

For a stochastic continuing chain and affine reward change r'=a r+b, V'=aV+b/(1-gamma). That shortcut generally fails for task-dependent nonlinear reward calibration or a different absorption-time distribution.

## 3. Preserve outcome components

A reward-separable critic can carry successor features:

    psi_pi(s) = E_pi[sum_t gamma^t phi_t | s_0=s],
    V_(pi,w)(s) = psi_pi(s)^T w.

For fixed policy/dynamics and reward features, changing w requires a new linear readout. Policy changes still change psi. This follows the successor-features transfer framework; the proposed application is to recursive agent training.

For bounded terminal tasks with gamma=1, store predicted strict success, a family-conditioned distribution over passed checks, and separately predicted cost/unauthorized attempts. Then for divisor D,

    E[R_SPR | s] = Pr(S=1 | s) + sum_p Pr(P=p | s) p/(p+D).

The expectation of P alone is insufficient. A distribution putting half its mass on 0 checks and half on 100 has E[P]=50. With D=100, true expected progress reward is .25; substituting the mean yields 1/3. The reference tests this discrepancy. All value predictions remain advisory; the deterministic host owns permissions.

## 4. What compression must preserve

Let z=f(h) compress history. For all histories sharing z and all permitted actions, require equal immediate reward laws, equal distributions over successor z, and equal admissible-action sets. Finite-horizon Bellman induction then gives a common value for each z and preserves optimal action values on this quotient.

Authority is part of the state: two histories saying 'approved' may differ in nonce, amount or expiry. Merging them violates the action-set premise. A summary can preserve value under one fixed policy and still be unsuitable for policy improvement.

An approximate finite-horizon variant, for rewards in [0,Rmax], common action sets, reward error at most epsilon_r and transition total-variation error at most epsilon_p, yields

    error_H <= H epsilon_r + Rmax epsilon_p H(H-1)/2.

Proof: the one-step expectation difference is at most (H-1)Rmax epsilon_p, followed by the previous horizon's value error. Summing the recurrence gives the expression. Uniform actionwise assumptions are essential. Reuse the repository's existing quotient/oracle ownership when testing this condition.
