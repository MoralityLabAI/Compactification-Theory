# UG-LAT-02: exact q=2 recursive gadget and a 36/35 separation ceiling

Date: 2026-11-04. Evidence: written mathematical proof, exhaustive exact finite kernels, and tested recurrence evaluation. The Lean companion certifies the rational recurrences separately. This concerns the manuscript's fixed-B recursive construction at F=F2, B=F2^3, before its final quotient and alphabet enlargement.

## The fixed recursive object

Set P0=R0=B and C0=lambda0=id. Index every parent by all pairs (A,J), where A is one of seven lines of F2^3 and J:B->U_A is one of 168 binary isomorphisms. Thus there are 1176 children, fixed in the map, at each node. For t+1, aggregate all J C_t(child) to (x,y), then set C_(t+1)=y+Q(x). The shift space R_(t+1) consists of child shifts whose translated aggregate has x=0; lambda_(t+1) returns its y component. The U_A span X plus B, so lambda stays onto. The input dimension is 3*1176^t bits.

Noise selects a uniform leaf and adds an independent uniform B-symbol there. A full linear lift is a linear family z->psi_z on P_t, indexed by B*, with psi_z restricted to R_t equal to z composed with lambda_t. Its detection probability is Pr[there exists z: psi_z(noise) != 0]. This definition fixes the relation between linear observations and the nonlinear output symmetry.

The executable height-one map contains all 1176 children. Larger depths use exact finite transfer kernels for this same fixed tree. The input tree is not materialized at larger depths. No stochastic fitting is used.

## Exact nonlinear transfer

Equivariance makes the child's initial output uniform even conditional on its difference and perturbation. The parent aggregate is consequently uniform and independent of the aggregate difference. Conditional on a nonzero child difference, uniform J sends it uniformly onto U_A minus zero. Enumerating 7 lines, 7 nonzero updates and 8 initial x values gives 392 outcomes:

    Pr[parent difference = 0] = 1/7
    Pr[parent difference = b] = 6/49 for each nonzero b.

A zero child difference remains zero. Therefore, for every depth t,

    p_t = Pr[C_t(input+noise) != C_t(input)] = (7/8)(6/7)^t.

The implementation constructs this kernel by evaluating the quadratic map, then iterates it. The closed form is checked independently.

## Minimization over every full linear lift

At a node with character subspace S, every lift induces one linear extension map gamma:S->X*, shared by all outgoing children. It must be fixed before that child's A,J are sampled. On line A=<v>, the restricted row family is kappa(z)=(gamma(z),z)|U_A. Its kernel is nonzero precisely when v belongs to S and

    v dot gamma(v) = v1*v2*v3.

Since a binary line has one nonzero vector, rank falls by at most one. Uniform J makes the surviving character space uniform among subspaces of its surviving rank. The code enumerates all 16 subspaces, all 1017 linear extension maps, and 7119 map/line restrictions. It verifies the kernel criterion against direct restriction matrices and checks uniform dual orientation against all 168 isomorphisms.

The greatest number of rank-losing lines is 1, 3, and 6 at ranks 1, 2, and 3. For ranks at most two, the cubic restricted to S is a quadratic function with zero constant term and can be realized as z dot gamma(z). At rank three, every quadratic has even total parity on F2^3, while z1*z2*z3 has odd parity, so agreement on all seven nonzero vectors is impossible. Gamma=0 attains agreement on six.

For clarity, the Bellman step first fixes gamma, then averages over A,J. A negative control interchanges these operations: at height one it incorrectly returns 3/4 instead of the valid minimum 43/56.

### Why the recursion gives the exact optimum

For any parent lift, select a linear section of kappa at each child. This gives a lift on its image; detecting with that subfamily implies detection with the original family. The mean optimal child detection therefore lower-bounds any parent lift.

For the reverse inequality, choose an optimal child lift on each image, and define the parent functional by summing these child functionals composed with kappa. On parent shifts, the shared gamma annihilates the horizontal aggregate and the vertical term is the required z(lambda). A noise perturbation visits only one child, so detection is exactly its chosen image-lift detection. Thus each Bellman minimum is attained by an actual linear lift of the fixed tree. Child choices are preassigned throughout the tree and depend only on their path; future noise choices are unavailable.

Writing d_r(t) for the minimum at rank r gives

    d_0(t)=0
    d_1(t+1)=(6/7)d_1(t)
    d_2(t+1)=(4/7)d_2(t)+(3/7)d_1(t)
    d_3(t+1)=(1/7)d_3(t)+(6/7)d_2(t)
    d_r(0)=1-2^(-r).

The closed forms are increasing with rank, which validates choosing the maximal rank loss at each stage. They yield

    d_1(t)=(1/2)(6/7)^t
    d_2(t)=(3/4)(6/7)^t
    d_3(t)=(9/10)(6/7)^t-(1/40)(1/7)^t.

Consequently

    d_3(t)/p_t = 36/35 - 1/(35*6^t).

This is an all-depth statement for the specified q=2 construction. Both probabilities tend to zero; the relative advantage stays below 36/35. The original arbitrary-small-noise versus constant-linear-detection theorem selects a growing character rank and a sufficiently large field before setting depth. Its parameter regime remains separate from this exact small-field study.

## Evidence and reproduction

Seven local unittest methods passed normally and under python -O. They exhaust all local maps, check the closed forms against exact Bellman values through depth 64, and reject altered receipts. The general finite-to-recursive equivalence above is a written proof; the entire gadget has not been formalized in Lean. The Lean companion is limited to recurrence identities and conditional harmonic-potential arithmetic.

    python -m unittest discover -s research/levin_sieve_v1 -p 'test_unique_games_recursive_v02.py' -v
    python research/levin_sieve_v1/unique_games_recursive_v02.py --out /tmp/ug-lat02.json
    python research/levin_sieve_v1/unique_games_recursive_v02.py --check /tmp/ug-lat02.json

The existing LSV-01 workflow discovers these tests. All runs use CPU exact arithmetic. No neural checkpoint, GPU allocation, or Goodfire service is involved.

Source: OpenAI, The Unique Games Theorem, section 02-latent.tex, blob 8e1c7ad0b873bd91f77e0611dae3d689b7c180a9, inspected 2026-11-04:
https://github.com/openai/math/blob/main/preprints/The-Unique-Games-Theorem-September-23-2026/build/sections/02-latent.tex

The local q=2 envelope is derived here; no priority or novelty claim is made.
