# NOR-30: source-held-out constrained minimax auditors

Prospective registered experiment: [REGISTRATION.md](REGISTRATION.md), frozen first at commit \`118dc142\` before source generation. Continues merged NOR-26 seven-policy attack cross-play and NOR-25 exact uniform-reserve soundness. The mathematical companion is [MathCan NOR-30](https://github.com/patrickdugan/MathCan/pull/120).

## Question

Does a discretized zero-sum-game mixture over seven established exact target-coordinate auditors, fitted exclusively on 12 original NOR-26 public sources, improve worst-case detection on **12 genuinely new public-only RRC25 sources**? Adversaries know which policies the auditor mixes and fit four altered coefficient positions to the mixture using 256 calibration schedules, but they must freeze these changes before 512 independent evaluation audit schedules.

A trained mixture samples one complete k=16 policy according to frozen integer weights on a 16-quantum simplex. Its primary expected detection is the **exact weighted combination** of independently evaluated pure-policy detection rates. One separately randomized policy-selection rollout is scored as a control. Training is a constrained empirical grid maximin over the seven historical greedy attack families: every candidate has 16 total quanta and between 2 and 12 quanta allocated to uniform exploration. The unrestricted grid maximizer is preserved as a diagnostic.

## Prospective outcome rules

The primary comparison tests the mixed defender against attacks optimized specifically for that mixture, and hybrid_50 against attacks optimized specifically for hybrid_50. It requires a positive source-paired gain on at least 9/12 new sources, a source-mean gain of at least 0.5 percentage points, and a positive worst-case cross-play gain on at least 9/12 sources. All exact-verifier parity, valid-future acceptance, nonzero-shift equivalence and fully realized-schedule adversarial evasion controls are mandatory.

Known-template attacks (tail, coefficient magnitude, lag, uniform and an excluded off-template support) are run separately on the same new public sources to expose loss of source-specific targeted coverage. Repeated challenge draws and eight attacker restarts share one public source. The independent source unit is the 12 public TEST problems.

## Original first-run evidence

The first registered study completed with **primary PASS** in [Actions run #44027909426](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44027909426). All four original CI jobs passed, including an independent replay of 4,816,896 exact held-out policy/attacker/witness decisions. The first-run TRAIN, public TEST, and outcome files are committed byte-for-byte in [results/nor30_original_hosted.zip](results/nor30_original_hosted.zip), with a [frozen SHA-256 receipt](results/nor30_first_frozen_receipt.json). Details and source-level table: [RESULTS.md](RESULTS.md).

**Retrospective scope finding:** the fitted mixture assigned 13/16 weight to shuffled_prior and 3/16 to uniform. A uniformly permuted fixed ranking selects an exactly uniform k-coordinate subset under ideal sampling. This fitted mixture is distributionally equivalent to pure uniform16, so the observed adversarial improvement over hybrid50 measures the value of broad random exploration and carries a substantial in-template detection cost. An exhaustive small finite-permutation test was added after the first result as a labeled retrospective diagnostic, leaving the registration unchanged.

## Scope

The source algebra is exact RRC25 finite-series mathematics with a frozen sixteen-template candidate family. The exact target-coordinate witness is assumed authentic and target-dependent. Python SHA-derived streams are reproducible challenge fixtures and provide no deployment secrecy. An attacker observing the *realized* audit schedule can choose unchecked changed coefficients and evade partial inspection. The finite Lean mixture theorem holds under ideal independence of the actual audit policy draw and random reserve from a fixed blind alteration set.
