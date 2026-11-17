# NOR-37 post-hoc robustness baseline audit (2026-11-17)

**Status:** This analysis was written after the first NOR-37 grade. It is EXPLORATORY and changes neither the original registration at `a87401971de2cdefd7f36110fc6beae009c43bef` nor the result of its five frozen gates. All five gates passed, and [the original independent separate-process replay](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44079786124) succeeded.

## Immutable first-replay identifiers

- Original 24-source PUBLIC-only fixture: 4,857,464 bytes, SHA256 `f969925a159aa3e36d7061e2e65e0a6b9e6e0f82634e2a7e52b4dac874df6710`.
- Original complete grade: 559,101 bytes, SHA256 `4852bd7de8012484db34404519553ae4d1e9f388203f2fdbf8b7516a534b684c`.
- Original evaluated run: [#44079786124](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44079786124), artifact ID `14327326409`.
- Successful independent replay: 24 sources, 1,920 greedy attack sets, 768 blind attack sets, 663,552 independently constructed component plans, and 30,277,632 checked source-policy-attack decisions.
- The original scoring receipts report the following source-level means: stochastic50 vs deterministic self-play +1.00708 percentage points (24/24 positive); worst calibrated class +1.04574 points (24/24); stochastic50 vs atom-random exact TOP support +5.68733 points (24/24); stochastic50 mixture vs deterministic mixture self-play +1.31156 points (24/24); stochastic mixture vs original mixture exact TOP +5.93264 points (24/24). Stochastic50 retained 32.64517% of the deterministic TOP gain.

## The equal-cost baselines matter

Recomputed directly from the original JSON's `source_records[*].crossplay_weighted_numerators` and `crossplay_denominator` fields. Every rate is the unweighted mean across 24 SOURCE identities, with eight radius-four greedy attack restarts and 512 independently salted heldout challenge schedules per source. Worst calibrated class means **the per-source minimum over ten original calibrated attacker columns**, subsequently averaged over sources. This is a finite learned-attack measurement.

| Defender | Self-play detection | Mean per-source worst calibrated class | Exact conditional TOP support |
| --- | ---: | ---: | ---: |
| original_h50 | 3.140% | 2.548% | 17.674% |
| atom_random_h50 | 3.074% | 2.397% | 17.756% |
| deterministic_h50 | 1.265% | 0.743% | 35.177% |
| stochastic25_h50 | 1.743% | 1.351% | 26.420% |
| stochastic50_h50 | 2.273% | 1.788% | 23.443% |
| stochastic75_h50 | 2.799% | 2.198% | 20.659% |
| uniform16 | **5.963%** | **5.269%** | 6.103% |
| original_mix | 4.028% | 3.512% | 14.393% |
| deterministic_mix | 1.503% | 1.295% | 34.144% |
| stochastic50_mix | 2.814% | 2.537% | 20.326% |

All values above can be regenerated from the original fully archived grader and are outside the five prospective criteria except where stated there.

Across the 24 source-paired rates, the self-play improvement of stochastic50_h50 over deterministic_h50 was +1.007 pp, descriptive 95% paired t-interval [+0.845,+1.169] pp, all 24 positive. Relative to the *original* hybrid50, stochastic50 self-play was **-0.868 pp**, interval [-1.089,-0.647], with 0/24 positive. Relative to pure uniform16 it was **-3.691 pp**, interval [-4.293,-3.088], also 0/24 positive. Stochastic50 mixture vs original NOR31 mixture was **-1.214 pp**, interval [-1.547,-0.880], with 0/24 positive. These post-hoc comparisons are hypotheses for the next prospective cohort. The registered five-gate PASS remains intact.

The stochastic75 sensitivity arm obtained +0.527 pp self-play above stochastic50 across 22/24 sources, while its exact TOP advantage over atom-random dropped from +5.687 pp to +2.9035 pp. This shows the expected curvature-versus-adversarial-support tradeoff. Greedy restarts are not independent sources; the four source replicates each contain six related degree/family conditions. Source-paired t intervals are descriptive, and a four-replicate clustered analysis gives much weaker sampling resolution.

## Exact unrestricted maximin benchmark for equal-size audits

Fix a continuation of length `m`, an audit of `k` distinct target coordinates, and an adversary that may precommit *any* `r` distinct changed coordinates, with independent audit randomization. For an arbitrary audit distribution `D` on `k`-subsets, let
`v(D,A) = Pr_{Q~D}[Q intersects A]`.
Averaging over a UNIFORM `r`-subset `A` gives, for every `D`,

```
E_A v(D,A) = 1 - C(m-k,r)/C(m,r) = u(m,k,r).
```

The right side is independent of the audit policy because every fixed `k`-subset intersects the same fraction of all `r`-subsets. Therefore `min_A v(D,A) <= u(m,k,r)` for every audit distribution. Choosing `D` to be the UNIFORM `k`-subset law attains equality for every fixed `A`, proving that the unrestricted support-selection game's exact maximin value is `u(m,k,r)`. At NOR-37's `k=16,r=4`, this is **8.081628%** for `m=769` and **4.103330%** for `m=1537`.

This theorem applies to unrestricted arbitrary four-coordinate attack supports, including supports outside all original risk atoms. A source-aligned attack law or a constrained attacker game has a different optimum. The external authentic true-target witness and fresh private entropy remain assumptions.

## New design hypothesis: global uniform reserve

Let `U` be the full-output uniform16 auditor, and `C` be any curvature-biased k16 auditor. Draw a fresh independent Bernoulli branch per challenge; use `U` with probability `alpha`, otherwise `C`. For every precommitted attack support `A`,

```
Pr[detection | A] = alpha * u(m,16,4) + (1-alpha)*Pr_C[detection | A]
                  >= alpha * u(m,16,4).
```

This certifies a full-coordinate exploration floor, which per-atom exploration alone cannot give when an attacked atom receives zero audit quota. A global uniform branch also spends audit budget on coordinates that public curvature scores consider unimportant.

An exploratory linear program maximizing the **minimum across ten source-averaged NOR-37 calibrated attacker-class rates** selected about **51.3% uniform16 + 48.7% deterministic_mix**, giving a *development-set* minimum of source-averaged class rates of 3.862% while preserving +2.0 pp exact TOP support advantage over atom_random_h50. Requiring at least 25% of the deterministic top advantage selected approximately **42.9% uniform16 + 57.1% deterministic_mix**, development-set worst-class mean 3.476%. Leave-one-replicate-out reoptimizations picked uniform fractions 49.5%-52.4% and 41.2%-43.9% respectively. These are post-hoc optimizations against attacks trained for OTHER policies; they say nothing definitive about a new mixture's own calibrated greedy best response.

For NOR-38, freeze one or more *simple rational* whole-output uniform/curvature mixtures on fresh sources before any defender-specific attacker calibration. Include original_mix and pure uniform16 as primary baselines. Require a named top-curvature advantage together with a bound on absolute worst-class detection and a uniform gap, all assessed at the independent source level. Calibrate fresh greedy responses to the NEW mixtures, including a class targeting the mixture's public uniform probability, and test attack positions in old zero-quota atoms. Record exact full-output minmax ceilings alongside finite-schedule results. A fresh cohort and a new prospective registration are required for any NOR-38 PASS claim.

The source code and SHA-locked first NOR-37 receipt remain the authoritative record for its original preregistered outcome.
