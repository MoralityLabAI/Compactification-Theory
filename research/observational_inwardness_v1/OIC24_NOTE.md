# OIC-24: Phase-translation pruning and an exact five-phase erasure threshold

**11 November 2026.** An extension of OIC-23's finite robust sensor game. Exact arithmetic, exhaustive finite negative searches, complete constructive adaptive policy trees and independently enumerated true-state faults. This is a synthetic sensor-identification result.

## OIC-T41. Sound rotational quotient of the adversarial information state

The true hidden pair is \((d,z_0)\in\mathbb Z_n^2\), the query address at time \(t\) is \(m_t\), and its clean output is \([z_0+td=m_t]\). For any common phase translation \(c\), the map

\[
(d,z_0,m_t)\mapsto(d,z_0+c,m_t+c)\pmod n
\]

leaves every clean report unchanged. The fault model permits arbitrary substitutions and marked erasures according to budgets independent of address labels, so the same translation preserves each admissible public fault history, future action, and decoder target \(d\).

Consequently a tuple of adversarial candidate masks \((M_{i,j})\) can be replaced during Boolean policy search by the lexicographically least simultaneous phase translation of all its masks. A strategy learned in that canonical coordinate system can be conjugated back to the original address coordinates. This is a sound, completeness-preserving quotient of the policy-feasibility game. The implementation in `oic24_sym.py` retains the translation offset when extracting the complete constructive tree. Exhaustive successor-equivariance tests cover all registered fault symbols and 1,200 deterministically sampled information states across n=2,...,5.

## OIC-T42. The exact five-phase one-erasure query threshold

For \(n=5\), allowing **at most one marked erasure**, an unquotiented OIC-23 oracle rules out all adaptive 8-query policies (79,681 visited exact information states). The symmetry-reduced oracle constructs a 9-query adaptive policy (17,253 visited equivalence-class states). Its complete 341-node tree survives independent replay of all 174 admissible true-world fault paths across every hidden shift and starting phase. Thus

\[
\boxed{Q^*_{\mathrm{erasure}}(5)=9.}
\]

The earlier OIC-T37 information lower bound gives only \(Q\ge n+1=6\) here. The finite result is stronger for this instance.

## One-substitution lower bound and remaining gap

For \(n=5\), the phase-translation quotient also exhaustively rules out every policy using at most eleven queries with one substituted report. At \(Q=11\), the certified search visits 567,127 canonical information states without finding a winning policy. Hence

\[
\boxed{Q^*_{\mathrm{flip}}(5)\ge12.}
\]

The candidate value twelve is consistent with the open OIC-23 pattern \(2n+2\), although an initial Q=12 search hit an explicitly declared 700,000-state cap before reaching a conclusive answer. A feasible 12-query tree, or a proof that every such tree fails, remains open.

## Evidence and reproduction

- `oic24_sym.py`: exact quotient and policy lifting, using OIC-23's candidate-mismatch oracle
- `oic24_run.py`: deterministic n=5 reference generation and independent true-world replay
- `results/oic24_reference.json`: search and policy metrics, with a compressed-certificate SHA-256 pin
- `results/oic24_n5_erasure_policy.json.gz.b64`: complete successful 9-query decision tree
- `test_oic24_sym.py`: five tests including direct successor equivariance and deliberate policy tampering, in normal and optimized Python

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic24_sym.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic24_sym.py' -v
python research/observational_inwardness_v1/oic24_run.py --check
~~~

This result is conditional on the declared exact cyclic model, single-bit addressed sensor, and bounded report-fault interface. No neural model or physical hidden-state measurement is involved.


## OIC-T43. Strict adaptive benefit under one marked erasure

At the exact adaptive one-erasure threshold for n=2,3,4,5, compare every fixed address word of the same length. A fixed word induces one binary codeword per hidden (d,z0). Guaranteeing identification after one **marked erasure** requires minimum Hamming distance at least 2 between codewords belonging to distinct shifts.

Using translation symmetry to fix the first sensor address at zero, exhaustive enumeration gives:

| n | Adaptive horizon Q* | Fixed words examined | Largest fixed cross-shift distance |
|---|---:|---:|---:|
| 2 | 4 | 8 | 2 |
| 3 | 5 | 81 | 1 |
| 4 | 7 | 4,096 | 1 |
| 5 | 9 | 390,625 | 1 |

Thus for n=3,4,5 **adaptive addressing is strictly more powerful than any fixed schedule at the same sensor-output budget when one erasure is permitted**. At n=2, a fixed schedule attains the exact adaptive minimum. Each exhaustive count and maximizing witness are pinned in `results/oic24_reference.json` and replayed by `oic24_run.py --check`.

The adaptive erasure minima for n=3,4,5 equal 2n−1, suggesting the open conjecture \(Q^*_{\mathrm{erasure}}(n)=2n-1\) for n≥3. Neither a proof nor an n≥6 result has yet been obtained. Online address-transmission bits, policy-description bits and computational search cost must be charged separately from the one-bit sensor returns.
