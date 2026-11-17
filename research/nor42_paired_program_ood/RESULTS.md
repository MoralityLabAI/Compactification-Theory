# NOR-42: first archived-only late-suffix smoke result

**Status: historical-source diagnostic PASS.** This is a CPU proof-of-concept for [NOR-42 design issue #229](https://github.com/MoralityLabAI/Compactification-Theory/issues/229), using **48 previously completed, independently replayed NOR-40 original sources**. The test created **zero prospective NOR-42 source identities**, no new private target dataset, no source-conditioned CAL/EVAL audit stream and no newly registered scientific outcome. [MathCan PR #142](https://github.com/patrickdugan/MathCan/pull/142) provides the Lean 4.35 kernel-verified prefix-indistinguishability theorem.

## Exact original diagnostic evidence

Hosted workflow [#44130867274](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130867274) completed **SUCCESS** with all historical-source unit tests passing. [Raw 48-source JSON artifact 14349438177](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130867274/artifacts/14349438177) contains the original first diagnostic data. The decompressed JSON is **36,963 bytes**, SHA-256 `411e6e1e30ce262a5604cb60e99b314ff3742413f8d526f4257585a48666f3f9`.

The prior NOR-40 source fixture is SHA-256 `b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5`. The original NOR-40 graded result is SHA-256 `42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092`. Both were frozen and independently replayed before this prototype. The 48 original source programs are from NOR-40 reps 76–83 × degrees D512/D1024 × the three old NOR21 input families.

For every historical source the smoke construction takes its independently confirmed clean future Y, the identical normalized query input F, and deterministic illustrative `β∈{−2,−1,1,2}` and `T∈{I,N,R,S}`. The hidden target is

```
Y_hidden(F) = Y_clean(F) + β X^(D+32) T(F)
```

The transform's normalized constant coefficient is 1. Consequently all output coefficients 0..D in the two worlds agree exactly, and **the first nonzero difference is at coefficient D+32 with value β**. All 48 clean original native masks accept exactly one of the 16 native templates; that public-template uniqueness still cannot tell the two future worlds apart.

## Fixed k16 uniformly hidden audit: exact results

For source support `S={i:Y_hidden(i)≠Y_clean(i)}` of cardinality s and future length m, the probability that a uniformly sampled, independent set of k16 output indices encounters at least one discrepancy is

`Pr[detect]=1−C(m−s,16)/C(m,16)`.

This is an **exact conditional probability** for verifying a clean proposal against the hidden **true** target with an independently authenticated coefficient witness. The prototype computes it without sampling an EVAL schedule. It is a different attack geometry from the previous radius-four malicious alteration study.

| Observed public degree | Future coefficients m | Historical sources | Mean difference support size | Exact uniform16 detection mean | Range of support size |
| --- | ---: | ---: | ---: | ---: | --- |
| D512 | 769 | 24 | **120.17** | **63.1924%** | 4–225 |
| D1024 | 1537 | 24 | **324.92** | **88.5848%** | 51–481 |

Using the eight **historical NOR-40 replica groups** as descriptive units, D512's mean hit probability has a 95% block-t interval **[52.38%, 74.00%]**; D1024's interval is **[80.48%, 96.68%]**. The paired D1024-minus-D512 rate difference averages **+25.392 percentage points**, interval **[+22.177,+28.608] points**, with all 8/8 replica block differences positive. These intervals are exploratory summaries for the deterministic historical example pairing, not registered NOR42 tests.

### Source families

| Source input family | Historical cases | Mean hidden-difference support | Exact uniform16 detection |
| --- | ---: | ---: | ---: |
| Original dense | 16 | 292.88 | 98.1462% |
| Original rational-short | 16 | 237.56 | 75.6044% |
| Original sparse | 16 | 137.19 | 53.9152% |

The smallest observed discrepancy support has only four changed coefficients, yielding detection near the existing uniform16 radius-four benchmark. The largest supported scenario has detection near unity. This illustrates why the *true difference support* must be measured separately from a four-coordinate precommitted malicious attack budget.

## Information distinction

The clean and hidden true programs have **identical public output prefixes**. A model using only those prefixes produces the same candidate for both, as certified in [MathCan NOR42HiddenSuffix.lean](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR42HiddenSuffix.lean). Its prediction cannot equal both different futures.

An internal-consistency check that compares the clean proposal against the clean continuation inferred from the same public native mask will accept the clean prediction even when the true target belongs to the hidden off-grammar pair. An audit against independently authenticated hidden **true output coefficients** will reject the clean proposal precisely when its k16 checked set meets the nonzero hidden discrepancy support. The smoke's hypothetical oracle is a mathematical construction; deployed authenticity, secrecy of audit randomness and ability to obtain true coefficients are additional premises.

## Limits and prospective next step

This synthetic example uses the completed NOR-40 corpus and deterministic β/T demonstration selection. **No NOR42 prospective cohort has been drawn or graded.** The new program-grammar paired cohort, true-target witness provenance, conditional audit budgets, independent interpreter and science thresholds must be frozen in a separate first registration before any novel source generation. In particular, NOR41's first input-family OOD outcome should be archived without changing the NOR42 pair definition or retrospectively tuning any registered benchmark.

The smoke harness and all of its unit tests are located in [paired_model.py](paired_model.py) and [test_paired_model.py](test_paired_model.py). The original 48-row data are currently retained in the hosted artifact and should be committed as an immutable raw receipt before the prototype is merged.
