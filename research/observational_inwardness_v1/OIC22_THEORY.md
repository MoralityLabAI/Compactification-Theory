# OIC-22: Fault-aware active observability and a universal query lower bound

**11 November 2026.** Scope: exact finite deterministic hidden-shift model, a one-bit addressed sensor at every time step, and a declared substitution/erasure budget. The dynamic programming is a specialized application of Bayesian experimental design. Proofs are handwritten and cross-checked by exact finite Python; there is no Lean certificate.

## Observation interface

Fix a cycle of size \(n\geq2\), a hidden shift \(d\in\mathbb Z_n\), and an unknown initial phase \(z_0\in\mathbb Z_n\), independently uniform. The hidden process is \(z_t=z_0+td\pmod n\). At time \(t\), the monitor chooses a sensor address \(m_t\in\mathbb Z_n\), then receives one bit \(y_t=[z_t=m_t]\). The next address can depend on all **received** reports, including those changed by the adversary. An erased bit is explicitly marked \(\bot\). No true phase or shift is available at inference.

A clean OIC-21 belief is a subset of \(n^2\) candidates \((d,z_0)\). With a budget of one substituted report, use \(M_0\) for candidates compatible with **zero** mismatches and \(M_1\) for candidates compatible with **exactly one** mismatch. Define \(C_y(m,t)=\{(d,z_0):[z_0+td=m]=y\}\), with all-set \(U\). Then upon a received binary bit \(y\),

\[
M'_0=M_0\cap C_y,\qquad M'_1=(M_1\cap C_y)\cup(M_0\cap(U\setminus C_y)).
\]

These sets are disjoint. An erasure can be handled separately with the exact one-erasure process: the erasure branch is \((M'_0,M'_1)=(\varnothing,M_0)\); a binary report intersects both current sets with \(C_y\). In a model allowing both fault types the bookkeeping must track both budgets.

## OIC-T36: Exact Bayesian Bellman recursion with one fault

Assume a single substitution occurs uniformly at one of the \(Q\) observation positions. Each hidden candidate is equally likely. At the final time the only candidates compatible with the transcript have exactly one mismatch. Let \(D_d=\{(d,z_0):z_0\in\mathbb Z_n\}\). The optimal leaf score is

\[
V_Q(M_0,M_1)=\max_d |M_1\cap D_d|.
\]

The optimal expected correct-identification count satisfies

\[
\boxed{V_t(M_0,M_1)=\max_{m\in\mathbb Z_n}\sum_{y\in\{0,1\}} V_{t+1}(M'_0,M'_1).}
\]

At the root the success probability is \(V_0(U,\varnothing)/(n^2Q)\). The sum counts **distinct received histories with multiplicity**, even if two histories yield the same pair of candidate masks. Each true pair and flip position produces one complete received history under the policy, which proves the formula by backward induction. The same construction with the erasure branch \(\bot\) proves the one-erasure recursion, again normalized by \(n^2Q\).

For the equal mixture of clean episodes and episodes with exactly one uniformly positioned substitution, the terminal score becomes

\[
\boxed{V_Q^{1/2}(M_0,M_1)=\max_d\left(Q|M_0\cap D_d|+|M_1\cap D_d|\right)}
\]

and the root accuracy is \(V^{1/2}_0(U,\varnothing)/(2n^2Q)\). This is the **joint Bayes objective** used by the exact teacher. Under a mixture where the probability of one flip is \(\alpha\), a prefix at time \(t\) gives a zero-mismatch candidate weight \((1-\alpha)+\alpha(Q-t)/Q\), and a one-mismatch candidate weight \(\alpha/Q\), before normalization. The learned monitor uses \(\alpha=1/2\) and the original seven per-address features of OIC-21.

### Exact CPU values at the registered \(Q=n+2\) budget

| n | One uniformly located flip | One marked erasure | 50/50 clean/flip mixture |
|---|---|---|---|
| 3 | 34/45 | 45/45 | 71/90 |
| 4 | 73/96 | 90/96 | 147/192 |
| 5 | 126/175 | 154/175 | 262/350 |

These are **best adaptive expected accuracies** among all deterministic query policies. They are exact finite numbers, checked against independently enumerated fixed-schedule distributions for small n. No full exact root optimum is asserted for n=6 or n=8, where state growth is much larger.

## OIC-T37: Robust all-zero indistinguishability lower bound

**Theorem.** Every policy guaranteeing identification of the full hidden shift for all \((d,z_0)\), under at most \(e\) report substitutions and \(r\) marked erasures, needs

\[
\boxed{Q\ge n(e+1)+r.}
\]

**Proof.** Consider any adaptive policy and the public received branch containing zeros at all non-erased positions and exactly \(r\) erasure markers, placed anywhere. Fix any shift \(d\). Each of the \(Q-r\) remaining addressed queries has exactly one initial phase that would have returned 1. Thus the total number of uncorrupted positive reports summed over the \(n\) possible initial phases is \(Q-r\). When \(Q-r<n(e+1)\), some phase produces at most \(e\) ones. Flip these one bits to zero. That true phase, shift and legal fault pattern generate precisely the chosen received branch. This works for every \(d\), and because the public branch is the same, the adaptive policy uses the same addresses. Every possible shift therefore remains compatible, preventing a universal correct verdict. QED.

This is a **necessary** resource bound. It establishes no matching sufficiency construction. For one substitution it says \(Q\ge 2n\), and for one substitution plus one erasure \(Q\ge 2n+1\). In particular, the registered \(Q=n+2\) policy has no full-shift zero-error certificate at \(n\ge4\).

## OIC-T38: Adversarial policy-tree decoding condition

Let \(P\) be a deterministic adaptive query policy. A public receipt \(h\) determines the actual issued addresses by replaying \(P\) on its previously **received** reports. Define \(\mathcal H_P(d,z_0;e,r)\) as the set of possible receipts under all legal faults. There exists an exact decoder guaranteed correct for every pair and fault pattern if and only if the unions

\[
\mathcal H_P(d;e,r)=\bigcup_{z_0}\mathcal H_P(d,z_0;e,r)
\]

are disjoint for distinct \(d\). The necessity follows from an ambiguous shared receipt. Sufficiency follows by returning the unique \(d\) whose receipt set contains the observed history. This criterion handles fault-dependent actions directly; fixed-code Hamming distances cannot be substituted without proving a reduction.

## Resource accounting

For \(Q\) active queries, the monitor consumes exactly \(Q\) sensor-report slots and sends up to \(Q\lceil\log_2 n\rceil\) online address bits with direct binary addressing. Its direct online communication budget is

\[
B_{\rm link}=Q(1+\lceil\log_2 n\rceil).
\]

The two exact candidate masks need \(2n^2\) live *boolean* bits in the abstract implementation, in addition to the policy's \(145\times32\) FP32 parameter bits, action decoder, numerical features, and runtime workspace. The actual NumPy implementation uses arrays and therefore more physical memory. The clean OIC-21 belief already requires \(n^2\) candidate slots. The exact teacher performs a much larger offline search, which is counted separately. A preprogrammed nonadaptive schedule has a different address-description/storage convention.

The finite results make no claim about subjective experience, cosmological physics, or learned transformer latent states. The benchmark is a synthetic instance of active automata identification and adversarial sensor design.
