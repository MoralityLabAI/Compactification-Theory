# OIC-26: Exact six-phase erasure threshold

**11 November 2026.** Executed exact finite search; the one-substitution n=6 threshold remains open. This result follows OIC-24/25 and retains the same publicly addressed binary marker sensor, unknown shift \(d\) and initial phase \(z_0\), with one **marked erasure** allowed during an episode.

## OIC-T46: Exact six-phase erasure threshold

\[
\boxed{Q^*_{\mathrm{erasure}}(6)=11.}
\]

**Lower bound.** With ten queries, an exhaustive unquotiented OIC-23 Boolean search visits 2,223,374 candidate information states and rejects every adaptive policy. A separately executed phase-translation-quotiented search also returns UNSAT after 370,580 canonical states. Hence at least eleven queries are required.

**Upper bound.** At eleven queries, the symmetry-reduced solver returns SAT after 27,577 canonical states. It constructs a 753-node adaptive decision tree. The original OIC-23 independent policy checker validates the full tree, and a separate simulator replays all 331 feasible true-world/marked-erasure paths for the 36 possible hidden worlds. Every leaf selects the correct shift.

These are computer-assisted exact finite results for a specified deterministic process. They are not a Lean kernel theorem or a claim about trained neural-network observability.

## Erasure horizon pattern

| n | Exact one-erasure adaptive minimum | Source |
|---:|---:|---|
| 2 | 4 | OIC-23 |
| 3 | 5 | OIC-23 |
| 4 | 7 | OIC-23 |
| 5 | 9 | OIC-24 |
| 6 | 11 | OIC-26 |

The formula \(Q^*_{\mathrm{erasure}}(n)=2n-1\) now agrees with every tested \(n=3,4,5,6\); its universal validity remains an open conjecture. The two-phase case follows a different finite pattern.

## Reproduction

- `oic26_erasure.py --check`: verify hashes, policy tree and independent true-world replay from frozen data
- `oic26_erasure.py --recompute`: rerun the **unquotiented** ten-query impossibility search, the symmetry-reduced negative search and the eleven-query successful search; compare the reconstructed policy with the saved certificate
- `test_oic26_erasure.py`: independent six-phase phase-translation equivariance, policy tampering and receipt checks
- `results/oic26_reference.json`: exact state counts and hashes
- `results/oic26_n6_erasure_tree.json.gz.b64`: the complete constructive policy

Resource use for the online eleven-query policy is 11 returned sensor bits and 33 address bits if each of the six possible sensor addresses is sent using three binary bits, or 44 total online link bits. Policy storage and exact search computation are separate costs.

OIC-26's originally registered larger task, the **one-substitution threshold at n=6**, remains open. The [OIC-26 protocol](OIC26_PROTOCOL.md) continues to govern that experiment.
