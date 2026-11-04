# LSV-01: Certified asynchronous goal repair

Date: 2026-11-04. Evidence: analytic proofs, exact rational CPU reference, finite tests and explicit counterexamples. These are applications of classical small-gain theory, nonnegative matrix analysis and asynchronous iterations. There is no claim of a new general mathematical theorem, a Lean-checked proof, a measured neural mechanism or biological validation.

## 1. System and assumptions

Take a finite network whose error relative to a declared target is e(t) in R^n. At time t exactly one coordinate i updates; all other coordinates are unchanged. Assume the uniform, global componentwise contract

    |e_i(t+1)| <= sum_j A[i,j] |e_j(t)| + eta[i],                 (1)

where A is nonnegative and eta[i]>=0. A proposed positive vector w and 0<=rho<1 certify small gain if

    sum_j A[i,j] w[j] <= rho*w[i] for all i.                    (2)

A schedule is B-fair in aligned blocks when every aligned consecutive block of B updates visits every coordinate. B>=n, and repetitions within the block are permitted. Define the weighted error, noise and repair radius by

    E(t)  = max_i |e_i(t)| / w[i]
    delta = max_i eta[i] / w[i]
    R     = delta/(1-rho).

## Theorem T1: robust local-to-global repair

For every trajectory satisfying (1), (2) and B-fairness,

    E(t) <= R + rho^floor(t/B)*max(E(0)-R,0),                   (3)

for all integer t>=0. For every tolerance tau>R, a sufficient number of complete blocks to enter the goal tube E<=tau is the smallest k with

    rho^k*max(E(0)-R,0) <= tau-R.                               (4)

Once entered, the goal tube stays invariant under all admitted updates.

**Proof.** Suppose the weighted error at the beginning of a block is at most R+d, with d>=0. Every update obeys

    |e_i(new)|/w[i] <= rho*(R+d)+delta = R+rho*d.              (5)

The other coordinates are unchanged and bounded by R+d, so the total weighted error remains at most R+d throughout that block. Every coordinate has been updated at least once by the end of the block; on its last update it was at most R+rho*d. Thus the error at the end of the block is at most R+rho*d. Set d_0=max(E(0)-R,0) and iterate by induction to prove (3) at block boundaries; use the within-block bound for other times. For tau>=R, rho*tau+delta<=tau, proving invariance. QED.

**Sharpness.** For the one-component system e(t+1)=rho*e(t)+delta with e(0)>=R and B=1, equality in (3) holds after every update. The code checks an exact six-step witness.

## Theorem T2: constructive rational certificates

For any nonnegative rational matrix A with spectral radius less than 1, the exact rational vector

    w=(I-A)^(-1)*1 = sum_(k>=0) A^k*1                         (6)

exists, satisfies w>=1 componentwise, and gives Aw=w-1<w. Therefore

    rho=max_i (Aw)[i]/w[i] < 1

is an exact certificate. Conversely, if a positive w satisfies Aw<=rho*w with rho<1, the spectral radius of A is less than 1.

**Proof.** The Neumann series converges under the stated spectral-radius assumption. Its terms are nonnegative, including the identity term, hence w>=1. Multiplying (I-A)w=1 gives Aw=w-1<w. Conversely, the matrix norm induced by the weighted infinity norm equals max_i (Aw)[i]/w[i] for a nonnegative A. Spectral radius never exceeds an induced norm. QED.

The Python reference constructs w by exact rational Gaussian elimination and checks Aw<=rho*w. Resolvent weights are sufficient and may be poorly conditioned or yield a rho much larger than the optimal achievable rate.

## Theorem T3: exact two-node cycle criterion

For A=[[0,a],[b,0]] with a,b>=0, a positive-weight contraction exists exactly when ab<1. If a,b>0, the least possible rho equals sqrt(ab), attained when w_1/w_2=sqrt(a/b).

**Proof.** Set r=w_1/w_2>0. The two inequalities require rho>=max(a/r,b*r). Their product is ab; therefore the maximum is at least sqrt(ab). When a,b>0, choose r=sqrt(a/b) to attain equality. If one factor is zero, the infimum rate is zero (not necessarily attained with finite positive weights), and ab<1 still admits a rho<1. When ab>=1, no such rho can exist. QED.

**Concrete witness.** Choose a=3, b=1/10 and w=(11/2,1). The maximum ordinary row sum is 3, so the ordinary sup-norm contraction test fails. Weighted ratios are 6/11 and 11/20, giving rho=11/20. Under eta=(1/100,1/100), delta=1/100 and R=1/45. Starting from e=(10,-3), E(0)=3. Formula (4) certifies E<=1/10 after seven complete B=2 blocks. The demo verifies six blocks of exact arithmetic and reports both actual and guaranteed error.

## Theorem T4: switching modes with a shared envelope

Let every update choose one of finitely many nonnegative gain matrices A_m. Assume each mode obeys (1), replacing A by A_m, and every mode is entrywise bounded by a common Abar. If Abar*w<=rho*w with rho<1, equation (3) holds for arbitrary switching among the modes, provided the coordinate schedule remains B-fair.

**Proof.** Entrywise dominance and nonnegative errors give a local inequality with Abar at every step. Apply T1. QED.

**Counterexample.** Each of A_0=[[0,3],[0,0]] and A_1=[[0,0],[3,0]] is nilpotent, with spectral radius zero. Update coordinate 0 under A_0 and coordinate 1 under A_1, repeating in B=2 blocks. The second error coordinate grows by a factor of 9 per block. Thus separate mode certificates do not establish arbitrary-switching stability. The test checks growth by 9^4 after four blocks.

The code computes a common entrywise-maximum envelope. This method is conservative, because different modes may maximize different coefficients within the same row.

## Lemma T5: deriving the contract from Jacobians

Let U be a convex domain containing the target x*. Suppose every local update F_i is continuously differentiable on U, F_i(x*)=x*_i, and the derivative bounds

    |partial F_i(x)/partial x_j| <= A[i,j]

hold for every x in U. Then

    |F_i(x)-x*_i| <= sum_j A[i,j]*|x_j-x*_j|, for x in U.     (7)

Adding a disturbance of magnitude at most eta_i yields contract (1).

**Proof.** Integrate the directional derivative of F_i along the segment from x* to x, which stays in U by convexity:

    F_i(x)-F_i(x*) = integral_(s=0)^1 sum_j
       partial_j F_i(x*+s(x-x*))*(x_j-x*_j) ds.

The triangle inequality and uniform derivative bounds give (7). QED.

A neural-model application must prove that U contains all reachable states and remains invariant under admitted perturbations. Empirical sampling of Jacobians does not establish a global derivative bound.

## Relationship to other repository research

[IDQ-01](../ingression_dynamics_v1/THEORY.md) establishes controlled dynamical quotients. [IDQ-03](../ingression_dynamics_v1/GOAL_RECOVERY.md) proves pathwise goal-hit transfer for a goal-visible exact quotient. LSV-01 gives a distinct robust bound for asynchronous local updates. When an application satisfies both sets of hypotheses, its goal-preserving quotient transfers the event certified by LSV-01 to the corresponding microscopic goal event.

## Scope and validity

The goal norm and interventions must be declared before measuring performance. The local contract must hold on the intended state domain, beyond a finite sampled trace. Failure of the weighted-gain test may simply mean the chosen componentwise envelope is too loose. Update starvation, stale neighbor reads, concurrent writes, and disturbances above eta fall outside T1. The finite rational tests audit the executable examples; the proofs above are mathematical arguments, not Lean kernel results. No claims about experience, souls, or biological consciousness follow from them.
