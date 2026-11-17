# NOR-28: first prospective commitment-before-entropy result

**Registered result: PASS**, under the exact criteria frozen in [REGISTRATION.md](REGISTRATION.md), initial commit da296c8d0cdfd8f42066896dca35bb16b7502f4d. The first-initiated hosted experiment is [GitHub Actions #44019242345](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44019242345); its complete independent replayer succeeded, including all 147,456 exact paired source/attacker/auditor/challenge decisions. A second concurrent PR-triggered run [#44019274608](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44019274608) independently drew a different OS root and nonces and also passed. All original data below use the earlier-initiated #44019242345, pinned separately by [the immutable first-run receipt](results/nor28_first_original_receipt.json).

## What the protocol established

PREP froze twelve fresh, source-only RRC25 public inputs, then made 96 attack commitments. Each committed altered-coordinate set contains four distinct positions and a fresh 256-bit salted preimage. The public manifest was SHA-locked in the first Actions job. Only after that job completed did a separate CHALLENGE job verify the commitments and sample a new 256-bit root using Python secrets.token_bytes(32). An independent HMAC-SHA256 counter-mode sampler selected audit coordinates uniformly without replacement using rejection sampling and froze the full schedule to disk before the sealed attack preimages were read for grading.

All **96 of 96** preimages opened their own commitment and all 96 deliberate nonce changes were rejected. Exact shift magnitudes +1 and +4294049777 yielded identical decisions. Every unchanged public prediction matched the authentic sealed target under the original RRC25 grammar. A counterfactual adversary selecting four unchecked coordinates after seeing the full schedule evaded on all **18,432** realized source/policy/challenge schedules, and all such replacements failed the earlier commitment. The independent replayer recomputed every schedule from the *eventually disclosed* original root and checked every attack and decision.

Four disclosures of the same postcommit root (0,8,16,256 bits revealed) produced **exactly 0 pathwise decision differences** over the 147,456 underlying decisions. These disclosures are views of the same samples. The fixed-attack decision was bound before any challenge seed existed; this is a chronology control and does not establish security if the original attacker can alter its commitment later.

## First-run source-level detection

Each entry below is detection over 4 calibrated attacks × 512 newly sampled challenge schedules. The independent unit for comparative assessment is a *source*, not each of these 2,048 within-source tests.

| Replicate | Observation degree | Family | Hybrid (8 risk + 8 uniform) | Static risk16 | Uniform16 |
| --- | ---: | --- | ---: | ---: | ---: |
| 10 | 512 | dense | 3.76% | 25% | 8.20% |
| 10 | 512 | sparse | 4.54% | 0% | 7.57% |
| 10 | 512 | rational | 3.42% | 0% | 7.86% |
| 10 | 1024 | dense | 2.10% | 0% | 4.25% |
| 10 | 1024 | sparse | 2.29% | 0% | 4.39% |
| 10 | 1024 | rational | 2.39% | 0% | 5.18% |
| 11 | 512 | dense | 3.42% | 0% | 7.71% |
| 11 | 512 | sparse | 4.54% | 0% | 7.52% |
| 11 | 512 | rational | 4.39% | **50%** | 7.62% |
| 11 | 1024 | dense | 1.56% | 0% | 3.56% |
| 11 | 1024 | sparse | 1.81% | 0% | 3.61% |
| 11 | 1024 | rational | 1.95% | 0% | 3.81% |
| **Equal-source mean** | | | **3.015%** | **6.250%** | **5.941%** |

The hybrid beat static on **10/12** source pairs and its 3.015% mean exceeded the preregistered 2% threshold. The registered criterion therefore PASSED. The source-mean hybrid-minus-static difference was **-3.235 percentage points**, because the deterministic static policy happened to inspect at least one of the four changed positions in three attack replicas on two source cases: source 10/512/dense gave static 25%, and source 11/512/rational gave static 50%. On the other ten source identities, the static policy detected zero of these calibrated fixed attacks. The median paired hybrid-minus-static improvement was **+2.20 pp**. This positive source count is a distinct statistic from the negative mean effect.

The fully uniform auditor achieved a higher mean detection rate on these adversarially calibrated attack positions. Therefore NOR-28 supports a binding-and-randomness *protocol* result and the specific registered source-level gate, rather than an overall hybrid-policy advantage.

For the additional four fresh uniform attacks per source, means were 15.27% hybrid, 12.50% static and 5.97% uniform. These descriptive numbers are unstable because just four source-specific attacked sets sometimes intersect the targeted eight positions; a single such intersection gives an entire 25%-point detection contribution to the source's four-attack average. Larger independently redrawn attack cohorts would be required before drawing conclusions about the typical in-template mixture.

## Theoretical detection versus the measured stochastic samples

For a blind fixed radius-four alteration selected before the audit root, an ideal uniform reserve of eight among the m-8 uninspected coordinates has conditional minimum detection

    1 - C(m-12,8)/C(m-8,8).

The exact bounds are approximately **4.147%** for m=769 and **2.079%** for m=1537. First-run hybrid detection means were **4.012%** for short outputs and **2.018%** for long outputs. Finite challenge frequencies can lie below their theoretical probability, and the actual HMAC sampler relies on computational pseudorandomness and a trusted entropy source rather than a mathematical oracle for perfectly uniform independent bits.

## Permanent first-run evidence

The first original files were published by two *ordered* GitHub Actions jobs, and all five raw bytestrings have been preserved in [results/nor28_original_hosted.zip](results/nor28_original_hosted.zip), a **1,099,102-byte** immutable source-controlled archive. The [first-run byte receipt](results/nor28_first_original_receipt.json) and [original_archive_audit.py](original_archive_audit.py) verify:

| File | Bytes | Original SHA-256 |
| --- | ---: | --- |
| Public-only 12-source fixture | 2,102,486 | 43a270313e331b4ecd6fa33722eac851888767b48b5b2c0a35e8fd9fcd627670 |
| Sealed 96 salted attacks | 32,291 | 62d758a60e59fb960e3aea70830ab891cf47d657ef443b164ccc9ecc01b9944d |
| Original public manifest | 24,847 | 8cc8d5a15348f85b4c5aa32b99750a0b6e1588cca524c4d76ca8b8872914fdc4 |
| First OS root plus 12×3×512 schedules | 5,538,148 | 0856647739357c3e7994a422d21fd340df1bf6479ea7a19b35e928e07f0c3ca7 |
| Complete exact outcome | 117,857 | b65bf20fc0c493b5299a7fe7abb58a22d7a821b17c75707c8bea8fa33bf47ed8 |

The original 256-bit OS root commitment was SHA-256 **5c86c781af7fb459bab5fa41f3610a04e29ff743ba5fbe3c8aa4ac1eaebbeeaf**. The sealed attack preimages, root, source identities, audit positions and full exact decisions were independently replayed from these first-run bytes. The source-controlled read-only SHA auditor is run in both the exact unit control and the independent replay CI jobs. Publicly disclosing the original root *after grading* makes this evidence verifiable but does not make the past audit randomness unpredictable retrospectively.

## Limits and next discrimination

The actual run's stage separation is evidenced by GitHub Actions job dependencies and completed-artifact checks. It is a hosted computation integrity assumption. An attacker with a compromised runner, early access to the OS root, or the ability to rewrite previously frozen artifacts defeats this model. Cryptographic binding is modeled through salted SHA-256, which is assumed secure rather than formally proved. The exact witness remains an exogenous authenticated target-coefficient oracle, and the source grammar is synthetic.

The most discriminating next experiment should prospectively vary attack precommitment binding strength and seed disclosure timing, with independent private randomness and a larger sample of held-out attack positions. NOR-28's strongest direct contribution is the **zero pathwise change across postcommit disclosures** and the complementary zero-detection counterexample when changes can be chosen after the realized audit schedule.
