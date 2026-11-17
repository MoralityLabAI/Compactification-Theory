# SCM-23 post-hoc calibration/holdout input overlap audit

This diagnostic was computed **after** the registered primary result. It preserves the original frozen heldout pairs, recorded outcomes and registered negative verdict.

Each F11 pair has observed signature `(recipient_first_op, donor_first_op, second_op, operand, nuisance)`; the pair sampler guarantees uniqueness **within** a split but uses independent random seeds rather than an across-split signature exclusion. The 192-pair calibration and 256-pair heldout samples therefore overlap in **14, 14, and 10** exact pair signatures for model seeds 191, 1191 and 2191. The original protocol's suggestion of generator-signature disjointness across splits was too strong. The test seed rule itself was independently fixed.

Filter these 38 overlapping pairs after freezing the evaluation, retaining 242, 242 and 246 pairs by seed, without fitting or changing any frame:

| Arm | Seed 191 / 242 | Seed 1191 / 242 | Seed 2191 / 246 | Total / 730 |
|---|---:|---:|---:|---:|
| Full CLS | 240 | 242 | 246 | 728 |
| Visible difference-PCA4 | 155 | 172 | 178 | 505 |
| Unrestricted difference-PCA4 | 157 | 176 | 179 | 512 |
| Fiber transport-gradient4 | **33** | **19** | **44** | **96** |
| Fiber difference-PCA4 | 19 | 5 | 9 | 33 |
| SAE fiber4 | 1 | 2 | 4 | 7 |
| Random fiber4 | 3 | 1 | 0 | 4 |

The gradient-fiber arm still exceeds the label-blind fiber controls in each seed; seed 1191 remains below 10% absolute transfer (19/242 = 7.85%). The registered gate **failed** with or without the filter. Repeated operator/operand combinations occur across training, calibration and test even after this exact pair filter, so this remains within-family generalization over a finite F11 task space, not an unseen mathematical-family test.

Reproduce without weights by reading the committed full outcome receipt and regenerating the two public `make_pairs` schedules. The public GitHub compact replay retains the original 768-pair registered endpoint.
