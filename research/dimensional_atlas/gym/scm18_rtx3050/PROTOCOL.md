# SCM-18 protocol

Registered 2026-11-14 before SCM-18 model fitting or evaluation. `config.json`
is the complete fixed numerical registration; its SHA-256 is copied into each
receipt. The campaign uses local CPU and the RTX 3050 only, FP32, batch eight,
one experiment at a time. A new process releases all GPU allocations between
major runs. Refuse less than 500 MiB free device memory and cap the PyTorch
allocator at 3000 MiB. Record peak allocation, reservation and observed free
memory; the allocator cap is not a reservation against other applications.

## Sources and access

Start from SCM-17 commit `468c6b2d3c41166183ff55be44f9d9dad0e493fe`.
Import the unmodified EvalMath oracle at commit
`88bb6cf8d2fe172e37d25964dfb2c285d9033841`, SHA-256
`c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`.
Do not call its official roster until **all** configurations, seeds, adapters,
direction frames, compression bases and selection decisions are frozen.
The curriculum uses discovery observed prefixes only. Gold sidecars and
checkpoints stay outside Git; public receipts include hashes only.

Fresh pools use disjoint seed ranges: training 1810000; discovery 1820000;
calibration 1830000; causal confirmation 1840000; development 1850000;
fresh final 1860000; source-operator counterfactual pairs 1870000;
discovery-derived replay 1880000; frozen new challenges 1890000.
Deduplicate public signatures across pools, including generator family.
Training, discovery, calibration and development use signed/product/sparse.
Fresh final and confirmation include unseen rational/squared/lacunary/
twisted_product families. Official finite-prefix collisions remain paired,
not independent observations. No final data inform curriculum decisions.

## Sequence and comparisons

1. Execute the original minimal SCM-15 CPU benchmark. Reproduce its architecture,
   data, optimizers and steps in a device-aware CPU/GPU parity experiment.
2. Screen 4, then 8, then 16 visits only after finite training losses and memory
   checks pass. Select one architecture using development exact future scores.
3. Train three seeds for that architecture under final-only supervision,
   every-visit supervision, SAE reconstruction regularization, and SAE plus
   causal consistency. The latter fits source-program labels at each visit;
   a KL term compares adjacent visit predictions during base fitting. During
   frozen-base adapter fitting, exact first-operator counterfactuals hold input
   series, topology and the other two operators fixed. Swap the donor/recipient
   hidden-state difference projected into the current low-rank write span at a
   nonterminal visit, and supervise the swapped result on the donor program.
   Fiber training additionally projects that swap onto the head kernel. This
   uses two extra forwards per update; account for the cost explicitly. The
   dictionary remains frozen. Log every loss; training loss is not evidence of
   held-out semantic mediation.
4. Freeze each base before dictionary and adapter fitting. Fit one shared TopK
   dictionary to CPU-cached training activations. Fit shared factors and visit
   gates at ranks 2/4/8 for the final-only model; fit the development-selected
   rank for the other objectives. Independently train joint and fiber-only
   adapters. Add an unfitted random-dictionary adapter with the same parameter
   count and optimizer budget. Rank selection uses development only.
5. Compare tied, joint, fiber-only, visible-only, reversed gates, zero code,
   independently fiber-trained, and random-dictionary policies. Execute the
   adapter arithmetic in the zero-code baseline too, matching inference paths.
   A cloned-at-initialization untied core gets the same data, steps and expanded
   visits. Report its larger parameter count and measured wall time separately.
6. Mine all 64 candidate programs on discovery; a maximum satisfies >= every
   one-edit neighbor, including flat plateaus. Record all nine neighbors and
   first mismatches. A separate audit grades withheld futures after selection;
   it cannot feed the selector. Freeze fresh replay/challenges before fitting.
7. Select SAE directions using eight calibration cases only; confirm Jacobian
   spectra, principal angles, curvature and individual-direction causal effects
   on independent examples. No full Hessian is materialized.
8. Freeze PCA and SAE feature order on calibration, then compare compression
   at each visit on independent cases. Record exact future accuracy and semantic
   source-operator interchange outcomes. A valid mediation test requires clean
   recipient and donor top-one predictions to have exact correct futures and
   differing oracle futures. Zero valid tests yields null fidelity, never 0/0.
9. Write an immutable freeze manifest with all checkpoint, dictionary, source,
   frame and split hashes. Only then open the official and fresh final sets.

## Accounting and interpretation

Count Python model calls, examples forwarded, expanded recurrent example-visits,
readout calls, optimizer steps, exact prefix checks and future grades separately.
Record backward work as optimizer steps, not as forward-equivalent FLOPs.
Same-path inference controls match arithmetic structure, not GPU kernel timings;
report actual timing and extra training cost. The symbolic 64-candidate reference
has a larger verification budget than the 12-candidate neural arms.

SCM-16's linear bottleneck gives an exact mathematical kernel; numerical residuals
are measured. RMP instead requires SCM-17 local margin gradients. Genuine RMP
checkpoint/dictionary/monitor bundles must pass the existing preflight; random
weights are never substituted for missing trained artifacts. SCM-12 rank-one
and SCM-13 affine-disk diagnostics require their explicit hypotheses. Sampled
Jacobians, disks, braids and curvature remain exploratory without continuum
bounds. A positive reasoning claim additionally requires exact held-out math,
semantic causal mediation, and gains over matched controls; preserve nulls.
