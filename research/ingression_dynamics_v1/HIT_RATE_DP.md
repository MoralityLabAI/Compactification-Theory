# QO-02: A closed-form goal-aware rate–distortion law

**2026-11-04.** A tractable special case of [QO-01](GOAL_AWARE_RATE_DISTORTION.md). All propositions below are elementary finite optimal-clustering results with complete proofs. The result concerns a declared **synthetic absorbing-goal Markov family**, not biological morphogenesis or learned neural abstractions. Its connection to one-dimensional minimax clustering is prior art; the closed-form family supplies an exact baseline for future quotient-learning experiments.

## Model and exact one-step distortion

Let the microstates be transient states 1,...,n and a single absorbing goal \(\dagger\). For each transient state \(i\), the `hit` action goes to \(\dagger\) with probability \(p_i\) and stays at \(i\) otherwise. The `idle` action fixes every microstate. The declared goal is the indicator of \(\dagger\). We require **zero goal error** at every microstate, so its macrostate is separate from the transient ones.

For any partition of transients into blocks \(B_1,\ldots,B_k\), the projected hit law from state \(i\in B_j\) is \((1-p_i)\delta_{B_j}+p_i\delta_{\dagger}\). The optimal TV center within a block has goal probability \((\min_{i\in B_j}p_i+\max_{i\in B_j}p_i)/2\). Thus the exact worst-case one-step autonomy defect is

\[
\boxed{\epsilon(B_1,\ldots,B_k)=\tfrac12\max_j\bigl(\max_{i\in B_j}p_i-\min_{i\in B_j}p_i\bigr).}\tag{1}
\]

This is a special case of IDQ-02's Chebyshev-radius theorem. The state-bit budget \(b\) permits at most \(k=\min(n,2^b-1)\) transient classes, since the goal occupies one additional distinct class.

## QO-02/T1: an optimal partition can be contiguous

**Theorem.** Sorting the \(p_i\), some optimal partition into at most \(k\) blocks consists of consecutive intervals of sorted rates.

**Proof.** Let an arbitrary partition have maximal within-block range \(D\). Every block is covered by a real interval of width \(D\), so the sorted rate points can be covered by at most \(k\) such intervals. Starting at the smallest uncovered point, greedily cover every point within \(D\) to its right. Any interval that includes this smallest point extends no farther right than the greedy interval. Repeating this exchange argument proves the greedy cover uses at most as many intervals as any feasible cover. Each greedy interval selects consecutive sorted points; hence an interval partition of at most \(k\) groups also achieves maximal range at most \(D\). An optimum exists among finite partitions, so one contiguous optimum exists. QED.

## QO-02/T2: exact polynomial-time dynamic program

For sorted \(p_1\le\dots\le p_n\), define \(D(j,i)\) as the minimum possible maximum **radius** over partitions of the first \(i\) points into exactly \(j\) nonempty contiguous groups. Set \(D(0,0)=0\) and infeasible cases to infinity. Then

\[
D(j,i)=\min_{j-1\le h<i}\max\left(D(j-1,h),\frac{p_i-p_{h+1}}2\right).\tag{2}
\]

**Proof.** In an optimal contiguous partition, its final group starts at \(h+1\) for a unique \(h\) in the displayed range. The previous \(j-1\) groups must be optimally partitioned, or replacing them would improve the result. The objective is the maximum of the previous optimum and the last group's radius. Minimize over \(h\). QED.

The reference implementation uses exact rational arithmetic, deterministic tie-breaking, \(O(kn^2)\) time and \(O(kn)\) memory. The goal block is excluded from the DP and charged an additional macrostate. Complexity here describes the DP after sorting (which adds \(O(n\log n)\) comparisons). For arbitrary controlled Markov kernels, the general QO-01 exhaustive or LP-backed problem remains.

## QO-02/T3: exact closed form for evenly spaced rates

Take \(p_i=i/(n+1)\), \(i=1,\ldots,n\), and a positive integer state-bit budget \(b\). Set \(k=\min\{n,2^b-1\}\). The global optimum is

\[
\boxed{R_n(b,0)=\frac{\lceil n/k\rceil-1}{2(n+1)}.}\tag{3}
\]

**Proof.** Among \(n\) points partitioned into at most \(k\) groups, some group contains at least \(\lceil n/k\rceil\) points. As the rates form an equally spaced grid, its range is at least \((\lceil n/k\rceil-1)/(n+1)\). Equation (1) gives the lower bound. Partition the sorted points into \(k\) consecutive groups whose sizes differ by at most one, so their maximal size is exactly \(\lceil n/k\rceil\). This achieves the bound. QED.

For a desired uniform error \(\epsilon\ge0\), a group can contain at most \(m=\min\{n,\lfloor 2\epsilon(n+1)\rfloor+1\}\) sorted rates. Therefore at least \(\lceil n/m\rceil\) transient classes and one goal class are required, giving the **exact minimum mutable label length**

\[
\boxed{b_{\min}(n,\epsilon)=\left\lceil\log_2\!\left(1+\left\lceil\frac{n}{m}\right\rceil\right)\right\rceil.}\tag{4}
\]

For \(n=4\), (3) reproduces the QO-01 rates: one bit = 3/10, two bits = 1/10, three bits = 0. This is a controlled model-class result, not an AGI-capability scaling law.

## Verification and limitations

- [hit_rate_dp.py](hit_rate_dp.py) implements the dynamic program, the closed form, and the inverse minimum-bits calculation.
- [test_hit_rate_dp.py](test_hit_rate_dp.py) checks 252 small random (n,k) instances against a separately implemented Bell-number exhaustive oracle, verifies the closed-form relation for n=1..20 and b=1..5, checks the inverse for n=1..24 and six rational tolerances, and reproduces the four-transient QO-01 fixture.
- `python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_hit_rate_dp.py' -v` runs without third-party dependencies and is covered by the existing morphospace GitHub workflow.

The dynamic program assumes every action is either `hit` with a single absorbing goal or identity. If the non-hit transition can move to another transient state, or a new action changes the goal/interface, the one-dimensional reduction can fail. The state labels count only mutable state identity; the decoder, transition table, rate estimates and acquisition cost must be measured separately before using this as a compact-AI engineering claim.
