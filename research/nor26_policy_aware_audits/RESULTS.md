# NOR-26: first prospective policy-aware attack game

**Registered outcome: PASS on the frozen 12-source criterion.** Original hosted CPU run [#43895351638](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43895351638); independent separate-process replay passed all **2,408,448 source/attacker/auditor/challenge decisions** and verified all original public source masks and greedy attacks. Original full raw file hashes are fixed in [results/nor26_first_original_receipt.json](results/nor26_first_original_receipt.json). The unchanged initial protocol is [REGISTRATION.md](REGISTRATION.md), committed first as 51ee625 before data generation.

## Predeclared primary readout

For each source, the attacker inspects 256 *calibration* audit schedules and greedily chooses four changed coordinates, repeating with eight independent tie streams. Attack coordinates are held fixed before the separate 512 *evaluation* audit schedules. Each auditor is evaluated against its own calibrated attack (diagonal self-play). Rates use exactly 8*512 = 4096 attack/challenge decisions per source/policy, then average across 12 independent sources.

| Auditor | Policy-aware diagonal detect | Fresh uniform blind attacks |
| --- | ---: | ---: |
| Uniform | 5.63% | 6.06% |
| Static trained prior | **0.00%** | 7.20% |
| Sequential posterior | **0.00%** | 6.76% |
| Shuffled ranking | 5.94% | 5.80% |
| Hybrid, 4 uniform /12 targeted | 1.57% | 6.98% |
| **Hybrid, 8 uniform /8 targeted** | **2.99%** | **6.49%** |
| Hybrid, 12 uniform /4 targeted | 4.76% | 6.02% |

Hybrid_50 beat static's self-play result on **12/12 independent source pairs** and achieved **2.988688% mean** self-play detection, clearing all registered detection gates. Each output length had a strictly positive exact finite floor, and the fully informed attacker evaded all partial checks. Registered **PASS**.

The policy-aware attack reduced hybrid_50 detection from **6.49% on fresh uniform attacks to 2.99%**. The fully uniform auditor preserved more absolute detection under these calibrated attacks, a 5.63% diagonal mean. Hybrid_50's gain is relative to the known-support deterministic ranking, and retains NOR-25's conditional in-template benefit; NOR-26 did not retest that in-template weighted primary.

A notable calibration/held-out distinction: all **12** hybrid_50 sources admitted **eight distinct four-coordinate attack sets** that escaped **100% of the 256 calibration schedules**. On independently salted evaluation challenges, 2.99% were caught. These findings measure attack-set optimization overfitting and the reserve's independent randomness, under the stated seed-visibility restriction.

## Exact mathematical floor

For m output positions, t=8 risk-targeted coordinates, u=8 uniformly chosen remaining coordinates, and r=4 fixed changed coordinates:

    min_A P(detect | A chosen before audit coins) =
        1 - C(m-12,8) / C(m-8,8).

| Output coefficients m | Exact ideal-random detection floor |
| ---: | ---: |
| 769 | 4.147204% |
| 1537 | 2.078527% |

The actual per-source 4096-trial frequencies are stochastic estimates, and their values can fall below the exact chance bound. NOR-26's calibration-aware attacker may choose any fixed altered set using the policy distribution; the guarantee persists because it is conditional on each risk prefix and independent uniform reserve. Its public deterministic challenge streams are replay seeds rather than cryptographic randomness for a real adversary.

## First source-level primary comparison

| Replicate | Degree | Family | Hybrid_50 diagonal | Static diagonal | Uniform diagonal |
| --- | ---: | --- | ---: | ---: | ---: |
| 6 | 512 | dense | 3.74% | 0% | 7.37% |
| 6 | 512 | sparse | 4.05% | 0% | 6.98% |
| 6 | 512 | rational | 3.83% | 0% | 5.66% |
| 6 | 1024 | dense | 2.20% | 0% | 3.74% |
| 6 | 1024 | sparse | 1.71% | 0% | 3.78% |
| 6 | 1024 | rational | 1.95% | 0% | 3.83% |
| 7 | 512 | dense | 4.76% | 0% | 7.25% |
| 7 | 512 | sparse | 3.69% | 0% | 8.79% |
| 7 | 512 | rational | 4.10% | 0% | 6.98% |
| 7 | 1024 | dense | 2.03% | 0% | 4.47% |
| 7 | 1024 | sparse | 1.83% | 0% | 4.27% |
| 7 | 1024 | rational | 1.98% | 0% | 4.42% |

Source-level mean hybrid_50 is 4.0283% for 769-coefficient outputs and 1.9491% for 1537-coefficient outputs. The source-level paired median hybrid_50 minus static self-play is +2.9419 pp. A descriptive, source-block bootstrap (10,000 draws with replacement, fixed seed 202611260080) gave a hybrid_50 mean interval of approximately **2.40% to 3.60%**. The study has twelve source units; 512 challenge ordinals and eight attacker replicas are repeated observations within them.

## Policy transfer

The following matrix is the **percentage detection on held-out schedules**. Attacker rows are optimized against the named policy; auditor columns use the same attack set and their own independently sampled audit schedules. Each matrix cell is an unweighted mean over twelve source identities.

| Attacker calibrated against | Uniform | Static | Posterior | Hybrid_25 | Hybrid_50 | Hybrid_75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 5.63 | 6.27 | 7.13 | 6.35 | 6.04 | 5.77 |
| Static | 6.19 | 0.00 | 0.14 | 1.60 | 2.88 | 4.63 |
| Posterior | 6.11 | 0.00 | 0.00 | 1.55 | 3.11 | 4.53 |
| Hybrid_25 | 6.10 | 0.12 | 0.22 | 1.57 | 2.92 | 4.67 |
| Hybrid_50 | 6.24 | 0.05 | 0.05 | 1.57 | 2.99 | 4.76 |
| Hybrid_75 | 5.87 | 0.04 | 0.12 | 1.55 | 3.05 | 4.76 |

The complete raw 7×7 matrix, including shuffled-prior policies, is in the full outcome JSON. Attacks calibrated against hybrid policies transfer strongly to static and posterior auditors, with almost zero detection, because those arms devote their inspections to predictable structured risk supports. The uniform auditor retains approximately 5.9% to 6.3% detection against most other calibrated strategies.

## Original source and outcome evidence

- 12 new public-only RRC25 source identities, original JSON **2,838,554 bytes**. SHA-256: e3f5a3df07d062d8681a39a7ddcba9a981b58af6f149f13bd372daf08ac6606f.
- First full outcome JSON **217,954 bytes**. SHA-256: a451897b780b024aac49f543833bc5e3ec32a734489d3f7a9b14a929bd11669a.
- Source-only and result bytes are preserved as [Actions #43895351638 artifact 14237404539](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43895351638). The independent auditor reconstructs all 12 original source proposals, both audit phases, greedy attacks and entire cross-play from the public source JSON.
- Every fully informed post-schedule attack evaded; all legitimate full-target proposals passed; sampled exact shift +1 versus +4294049777 comparisons were identical.

## Interpretation

NOR-26 establishes a positive **fixed schedule-blind** adversarial guarantee and a prospective empirical detection advantage for uniform-reserve auditing over deterministic source-risk rankings, using calibrated attacks chosen independently from the actual audit schedules. Neither NOR-26 nor its mathematics establishes secrecy of a public reproducible pseudorandom seed or unconditional target correctness. The authentic exact-value witness remains external. A production audit would need unpredictable privately generated coins, and its attack model must exclude leaked realized schedules.
