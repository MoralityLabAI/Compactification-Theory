# OIC-23: Finite robust active-sensor certificates

**11 November 2026.** Exact finite computational results with independently replayable adaptive query trees, complete short-horizon searches, and enumerated fixed-schedule comparisons. No Lean formalization or universal closed-form robust-query theorem is asserted.

## Sensor contract

The hidden state is \((d,z_0)\in\mathbb Z_n^2\), with deterministic phase \(z_t=z_0+td\pmod n\). At each step \(t\), a controller chooses one address \(m_t\in\mathbb Z_n\), and the noiseless response is the bit \(y_t=[z_t=m_t]\). Later choices may depend only on *received* reports. An adversary may substitute up to \(e\) binary reports, erase up to \(r\) reports with a visible `?` symbol, or use both faults under their respective budgets. An exact guarantee must recover the shift \(d\) for *every* starting phase and every admissible online fault pattern. The policy may stop early upon proving a unique shift.

## OIC-T39. Exact finite minimum query budgets

For \(n\in\{2,3,4\}\), the exhaustive adaptive policy-tree search gives:

| n | Up to one substitution | Up to one marked erasure | Up to one of each |
|---|---:|---:|---:|
| 2 | 6 | 4 | 8 |
| 3 | 8 | 5 | 9 |
| 4 | 10 | 7 | 12 |

For every table entry, a complete solver establishes impossibility at \(Q-1\). Its successful \(Q\)-query policy is archived as a full action/response tree. The independent certificate verifier enumerates **every** true \((d,z_0)\) and every adaptive fault path; all leaves return the correct \(d\). The property of having a successful policy is monotone in \(Q\): an observer can ignore any extra query opportunity. Thus the one-step negative decision and positive certificate establish the stated exact minimum.

The finite result suggests \(Q^*_{\mathrm{flip}}(n)=2n+2\) for \(n\ge2\), beyond the general necessary lower bound \(Q\ge2n\) proved in OIC-T37. **This remains a conjecture for \(n\ge5\).** Initial \(n=5\) searches exceeded a 300,000 memoized-state cap and are explicitly inconclusive.

### Exact decision procedure and sufficient-state proof

Represent a received-history information set by masks \(M_{i,j}\), where \(M_{i,j}\) contains precisely those hidden configurations that explain the observed bits using exactly \(i\) substitutions and \(j\) marked erasures. At time \(t\), an action \(m\) partitions candidates by the predicted clean report \(C_y(t,m)\). On receiving a binary report \(y\), the successor mask is

\[
M'_{i,j}=(M_{i,j}\cap C_y)\cup(M_{i-1,j}\cap C_{1-y}),
\]

with invalid indices interpreted as the empty set. On a marked erasure,

\[
M'_{i,j}=M_{i,j-1}.
\]

A terminal node is successful exactly when the union of all masks contains candidates with only one shift value. The Boolean Bellman recursion returns **true** at time \(t\) if a terminal node is successful or if there exists a marker \(m\) such that every feasible received report leads to a true successor. Distinct report branches must be retained because their future actions can differ. Memoization keys are the absolute clock step and the complete tuple of candidate masks; they preserve all future admissible reports and shift labels. By backward induction this recursion is sound and complete for the registered finite model and horizon.

## OIC-T40. Strict advantage of adaptive addressing under one substitution

For \(n=4\) at horizon \(Q=10\), the Boolean adversarial game admits a successful adaptive decision tree. A fixed schedule \(w=(m_0,\ldots,m_9)\) instead induces a binary codeword \(c_{d,z_0}(w)\) for each hidden pair. Universal correction of one substituted report across distinct shifts requires

\[
\min_{d\ne d',z_0,z'_0}d_H(c_{d,z_0}(w),c_{d',z'_0}(w))\ge3.
\]

Because uniformly translating all sensor addresses and all starting phases preserves every output word and its inter-shift Hamming distance, fix \(m_0=0\). Exhaustive enumeration of the remaining \(4^9=262{,}144\) schedules gives the sharp maximum cross-shift distance

\[
\boxed{\max_w\min_{d\ne d',z_0,z'_0}d_H(c_{d,z_0}(w),c_{d',z'_0}(w))=2.}
\]

Therefore **every fixed schedule fails to guarantee one-error shift identification at ten queries**, while an adaptive schedule succeeds. The finite certificate contains the complete successful adaptive tree, and the pinned reference contains the maximizing fixed word and exhaustive count. At the corresponding thresholds \(n=2,Q=6\) and \(n=3,Q=8\), fixed schedules achieve cross-shift distance three. This locates the first strict finite separation among the tested sizes.

## Computational evidence and resource accounting

The reference archive contains nine explicit policy trees with SHA-256 hashes over a deterministic gzip representation, complete exact-oracle explored-state counts, and independently enumerated true-world fault-path counts. The standard-library verifier regenerates the positive trees, checks the negative horizons, exhaustively enumerates fixed schedules at the reported one-flip thresholds, and replays every true shift, phase and online fault path with a separately implemented state simulator. It checks certificate integrity and deliberate tampering.

A \(Q\)-query adaptive policy transmits \(Q\) sensor-output slots plus at most \(Q\lceil\log_2n\rceil\) addressed-control bits under direct binary encoding. Its decision tree and exact state-mask workspace also consume storage and inference operations; the archived tree has variable node count, so its description should be charged independently from the neural controller in OIC-22. These results are about a synthetic finite active sensor interface, and imply nothing about the subjective nature of physical or neural computation.
