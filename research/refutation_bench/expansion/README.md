# RefutationBench sourced expansion: five historical and contemporary families

**Status 2026-11-12:** Ten public development cases (RB-0016 through RB-0025)
drawn from five independent source families. They add six refuted arguments,
three established controls, and one explicitly conditional control. The whole
combined suite contains **25 cases in 11 families** with 6 train and 19 dev
records. No confidential test set exists and no model has been trained here.

This expansion leaves the RefutationBench v0.1 seed and its separately hashed
baseline receipt unchanged. A combined-suite CLI validates both data shards
and reuses the seed's exact response schema, source-family split guard, scoring
functions, export whitelist, and supported-refutation rules.

## Run from the repository root

```sh
python -m research.refutation_bench.expansion.suite validate
python -m research.refutation_bench.expansion.suite witness
python -m research.refutation_bench.expansion.suite export \
  --split dev --out /tmp/refutation_combined_dev.jsonl
python -m unittest discover -s research/refutation_bench/tests -v
```

After a model writes JSON responses using the original bench response schema:

```sh
python -m research.refutation_bench.expansion.suite evaluate \
  --split dev --responses /tmp/model_responses.jsonl \
  --out /tmp/refutation_combined_scores.json
```

The new material is *public dev*, including the gold key. Use independent
unseen manuscripts, separately adjudicated, for a future transfer test. The
evidence is curated and often exposes the flaw, so this pack tests comprehension
and localization before it can test raw-document discovery.

## Case map and adjudication boundaries

| Cases | Source family | Verdicts | Core certificate |
| --- | --- | --- | --- |
| RB-0016–17 | Deolalikar P vs NP (2010) | refuted / refuted | Monadic FO(LFP) expressivity gap; intermediate order dependence |
| RB-0018–19 | Kempe four-colour argument (1879) | refuted / established | Dependent Kempe-chain switches; degree <=4 argument retained |
| RB-0020–21 | Lamé FLT argument (1847) | refuted / established | Unique factorization assumed; polynomial splitting valid |
| RB-0022–23 | Wiles FLT history (1993–95) | refuted / established | Initial Euler-system gap; published Taylor–Wiles repair |
| RB-0024–25 | Ansari Collatz sieve (2025) | refuted / conditional | Explicit 11 witness against a proposed set identity; valid earlier strong-induction criterion |

**Scope:** 'refuted' means the **specific asserted argument** fails. In the
Kempe, Lamé and Wiles examples, the theorem concerned is now established.
A defect in a proof of P≠NP or a Collatz-related sieve does not resolve either
open problem. Ansari's publication presents a verification-reduction procedure,
not a purported proof of the Collatz conjecture. Wiles's repaired result is
standard accepted mathematics; the label on RB-0023 concerns the documented
1995 replacement rather than a new full verification.

## Source and passage ledger

These URLs are external references, checked on 2026-11-12. Printed pages and
named sections are frozen in evidence locators. They are **bibliographically
anchored**, not locally archived bit-for-bit, and their original-file SHA-256
hashes have not been independently captured. The older seed's two pinned RH
artifacts retain their SHA-256 checks.

1. **Deolalikar / Immerman:** Neil Immerman's technical letter reproduced by
   Richard Lipton, [Fatal Flaws in Deolalikar's Proof?](https://rjlipton.com/2010/08/12/fatal-flaws-in-deolalikars-proof/),
   12 August 2010, especially the successor transitive-closure example, the
   product-space incidence relations, and the monadic FO(LFP) limitation. This
   is a contemporaneous expert critique quoting Deolalikar's p.67 and Remark
   7.4, not a byte-pinned copy of Deolalikar's original manuscript. The
   independent easy-XORSAT test is discussed in
   [MIT News, 17 August 2010](https://news.mit.edu/2010/3q-pnp).
2. **Kempe / Gonthier:** Georges Gonthier,
   [A computer-checked proof of the Four Colour Theorem](https://www.cl.cam.ac.uk/~lp15/Pages/4colproof.pdf),
   printed p.7 (Kempe's chain interference), and
   [MacTutor's Alfred Kempe history](https://mathshistory.st-andrews.ac.uk/Biographies/Kempe/).
   [MathWorld Heawood Four-Color Map](https://mathworld.wolfram.com/HeawoodFour-ColorMap.html)
   describes the historical procedural counterexample.
3. **Lamé:** [MacTutor, Fermat's Last Theorem](https://mathshistory.st-andrews.ac.uk/HistTopics/Fermat%27s_last_theorem/),
   1847 section, and
   [Mathematical Association of America, Algebraic Number Theory and Fermat's Last Theorem](https://old.maa.org/node/751827).
   Both describe the unique-factorization dependency and Liouville/Kummer's objection.
   The original 1847 notice has not been directly checked.
4. **Wiles:** Andrew Wiles, *Modular elliptic curves and Fermat's Last Theorem*,
   *Annals of Mathematics* **141** (1995), 443–551,
   [full paper](https://jontallen.ece.illinois.edu/uploads/537.F18/Papers/Public/Wiles-Fermat.95.pdf),
   especially printed p.453 (author's first-person account of the
   incomplete 1993 Euler-system construction and 1994 repair).
5. **Ansari:** Mohammad Ansari, *Recursive sufficiency for the Collatz
   conjecture and computational verification*, *Notes on Number Theory and
   Discrete Mathematics* **31**(3) (2025), 471–480,
   [publisher-hosted PDF](https://nntdm.net/papers/nntdm-31/NNTDM-31-3-471-480.pdf),
   DOI 10.7546/nntdm.2025.31.3.471-480. Printed pp.473–475
   contain the valid recursive-sufficiency induction; pp.477–479 contain the
   stronger sieve. A [separate public audit](https://github.com/Sodelin/Collatz-Conjecture-Work/blob/main/proof-search/sources/Sufficiency_Rank_Audit_2026-09-05.md)
   reconstructs the failed identity. This pack directly checks the n=1
   set mismatch at the paper's printed definitions.

## Finite Collatz witness

For the contested equality in Lemma 3.1, take the first nontrivial induction
index n=1. Write m=4k+3. Restrict to 0<=k<9 so that both base-3 digits are
explicit. The proposed F'_1 permits each digit in {0,1,2}, and A' removes
the single pattern (2,2). F_2 permits only (0,0), (0,1), (1,0), (1,1).
Consequently:

```text
F'_1 \ A' in first block = {3, 7, 11, 15, 19, 23, 27, 31}
F_2 in first block       = {3, 7, 15, 19}
witness                 = 11   (k=2, digits 02)
```

The executable `witness` subcommand generates these exact sets. It checks a
single false equality that an induction uses. It is neither an attempted
Collatz counterexample nor a claim to have replicated every result in the
published paper.

## Next acquisition gates

Prioritize the original Deolalikar 2010 102-page manuscript, Lamé's 1847
notice, and a machine-readable Kempe/Heawood graph with a verified colouring
trace. Add source-byte digests when reproductions are obtained. Hold potential
Atiyah RH, abc, and other socially disputed claims in a candidate register
until a precise manuscript inference and independent countercertificate are
documented. Keep theorem status, argument validity and source completeness as
separate labels; avoid turning disagreement into a ground-truth refutation.
