# KSR-01C: dependencies introduced by a compressed state

2026-11-14. These deductions are independent of the external OpenAI #229/#236 claims. Their probability proofs are handwritten; exact finite controls and a frozen RRC-25 checkpoint audit accompany them. The original KSR-1 through KSR-6 statements retain their independent symmetric-broadcast assumptions.

## KSR-7: the sibling dependence hidden by coarse labels

Let H be a full parent state. Given H, draw children H1 and H2 independently from the same transition kernel. Let C=c(H) be a compressed parent label, and let Zi=c(Hi) be the child labels. Write

    p_j(h) = Pr(Z_i=j | H=h).

For any parent label a of positive probability, define

    T_a(j) = E[p_j(H) | C=a],
    J_a(j,k) = Pr(Z1=j, Z2=k | C=a).

Conditional independence at the full state gives J_a(j,k)=E[p_j(H)p_k(H)|C=a]. Therefore

    J_a - T_a T_a^T = Cov(p(H) | C=a).

The matrix on the right is positive semidefinite, and its trace is

    Delta_a = sum_j Var(p_j(H)|C=a)
            = Pr(Z1=Z2|C=a) - sum_j T_a(j)^2 >= 0.

In particular, two same-label parents with different child-label distributions certify Delta_a>0, provided each has positive probability. Conversely, Delta_a=0 exactly when all p_j(H) are constant almost surely within the parent-label fibre. In that case arbitrary numbers of independently generated children have independent coarse labels conditional on C=a.

Proof: expand the conditional second moments; a sum of nonnegative conditional variances is zero precisely when each coordinate is almost surely constant. Conditioning on the constant vector then factors the full sibling-label law.

This is a one-step statement on the declared parent-state distribution. A stationary coarse Markov process requires consistency on every relevant reachable-state distribution and at later depths as well. A fitted single-step transition matrix does not supply that consistency.

## KSR-8: a posterior accuracy floor with arbitrary prior and dependence

Let R be a finite root label with prior pi_j=Pr(R=j), and Y any observation. Choose real, finite-variance scores S_j(Y), one for each label j. Define

    a_j = Cov(1{R=j}, S_j),   v_j = Var(S_j).

Set a_j^2/v_j to zero when v_j=0 (then a_j=0). The optimal Bayes decoder of Y satisfies

    P_MAP >= max(max_j pi_j,
                 sum_j pi_j^2 + sum_j a_j^2/v_j).

Proof: let eta_j=Pr(R=j|Y). The tower property gives a_j=Cov(eta_j,S_j). Cauchy-Schwarz implies Var(eta_j)>=a_j^2/v_j. For each posterior vector, max_j eta_j>=sum_j eta_j^2. Taking expectations proves the second lower bound. Ignoring Y gives the first bound.

No uniform prior, tree structure, symmetry or conditional sibling independence is required. With a uniform root, symmetric broadcast scores, and the KSR-1 eigenfunction covariance, this reduces to the earlier 1/q+(q-1)/(q V_L) expression.

The implementation uses S_j equal to the number of observed leaves with label j. Its exact Fraction result applies to a declared finite joint law of root and leaf-count vector. A law assembled from a finite root cohort is a valid finite population for this theorem. Estimating its moments from samples does not automatically yield a population confidence bound. The native audit enumerates the intervention law on a frozen finite root cohort; its count MAP decoder uses that same finite prior.

## KSR-9: shared innovations change the second-moment recurrence

Consider the uniform q-state symmetric channel P_lambda on a b-ary tree. At each vertex, independently choose one of two ways to generate its children: with probability 1-rho draw child labels independently from P_lambda; with probability rho draw a single channel label and copy it to all b children. Subsequent subtree innovations are independent conditional on these labels. Every individual edge has the same original marginal channel in both cases.

For the usual contrast g_j(s)=q*1{s=j}-1, the normalized depth-L sum still satisfies

    E[g_j(root) M_{j,L}] = q-1.

The unconditional cross moment of two distinct child contrasts is

    E[g_j(child_1) g_j(child_2)]
      = (q-1)[lambda^2 + rho(1-lambda^2)].

Expanding the square of the sum of b descendant-subtree estimators gives

    V_0=1,
    V_{L+1} = V_L/(b lambda^2)
             + (b-1)/b * [1 + rho(1-lambda^2)/lambda^2].

The earlier posterior argument remains applicable using this corrected V_L. The common-innovation term increases variance without changing the single-edge channel. For t=b lambda^2>1 its finite limit is

    V_infinity(rho) = [(b-1)/b * (1+rho(1-lambda^2)/lambda^2)] * t/(t-1).

This supplies a sufficient reconstruction floor in the stated correlated model. It does not assert that t=1 is the exact threshold for all correlated branching processes.

### An exact falsification example

Take q=3, b=4, lambda=1/2, rho=1, and depth one. Every leaf is the same single channel draw. The true optimal root accuracy is 2/3. Incorrectly applying the independent-child formula gives V_1=7/4 and a claimed lower bound of 5/7, which exceeds the true optimum. The corrected variance is V_1=4, producing the valid lower bound 1/2.

Both channel marginals are identical; a one-edge transition audit cannot distinguish this example from the independent broadcast. The finite code independently enumerates complete histogram laws to verify the variance correction and Bayes inequalities.

## KSR-10: minimum refinement needed to recover one-step independence

Fix a finite parent-state support with positive mass at every supported state. Let p_h be its distribution of original child labels. A deterministic refinement D of C must retain the original parent label: C is a function of D.

For each original label a let k_a be the number of distinct vectors p_h among supported h with c(h)=a. Any refinement for which the original child labels become conditionally independent must have at least

    S_min = sum_a k_a

distinct supported values. To prove this, apply KSR-7 within each D-fibre: two different p_h values cannot share a D-label. Since D retains C, values from different original C-fibres also cannot merge.

The bound is attained by D=(c(H),p_H). It gives a fixed-length representation of at least ceil(log2 S_min) bits, and such a finite code exists. The exact finite certificate groups equal rational child-law vectors and records their parent indices.

The new label restores independence of the ORIGINAL child labels for the ONE declared transition kernel and support. It does not yet define a recursively closed Markov representation of D at subsequent generations. Computing p_H may consume all input-bank transitions; storing its codebook and computing the refinement have separate costs. The finite eight-row intervention bank quantizes this example's probability vectors; a different bank may change the partition.

## Frozen recurrent-model interpretation

The RRC-25 GRU has twelve real hidden coordinates and a trained sixteen-way output head. We keep its weights frozen, replay the nine original public feature rows, and observe the first-template U index of the highest-ranked candidate. Independent edges select one of eight calibration feature rows and execute the native recurrent step. Such extra rows form an artificial continuation outside the originally specified nine-row ranking task.

The finite audit checks the conditional covariance matrix, label-prior imbalance, symmetry residual and two-step transition composition. The root-cohort count likelihoods are constructed by bottom-up convolution given full hidden states, which keeps the independence that does hold at that level. It then evaluates KSR-8 on the resulting exact finite discrete law. The original uniform symmetric broadcast floor is never substituted for this model's measured law.

This work measures how a particular trained ranker's coarse label discards state relevant to subsequent labels. It supplies no claim about native LoopedTransformers checkpoints, hidden chain of thought, or an infinite-tree phase transition in a neural network.
