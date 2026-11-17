# SCM-08: clean-tube and source-derived RMS registration

**Registered 2026-11-13, before trained-model execution.** SCM-01 through SCM-07 synthetic results remain historical. The sealed RMP G0 evaluation, models, checkpoints and registrations stay untouched.

## Question
Can source-only clean-state tubes remove SCM-07's dependence on asserted zero-error float centers and prove a null-braid condition on the registered RMP cyclic-intervention tasks? If tubes are too wide, report an inconclusive bound instead of promoting a numerical braid.

## Fixed protocol

1. Use only `LOOP-T s1` and `FF-U s3` from the RMP model registry. Pin the exact four RMP Python source blobs via the existing SCM-06 manifest, then verify checkpoint and D-shared dictionary SHA-256 before a run.
2. Restore calibration tasks from the **selection half** of `monitor_calibration`, matched by task, group and token SHA-256. Use SCM-02's 64 group-disjoint tasks, evenly distributed over pointer chase, gated pointer chase, reachability and Sudoku. No reporting-half or sealed-test read.
3. The same intervention acts after expanded visit zero at token position four, with the same frozen two D-shared SAE decoder directions and relative radius 0.05. Observe tokens 16, 32 and 48. Use the registered SAE chart and random-orthogonal readout controls, keeping intervention axes fixed.
4. Generate the mathematical input center and certified error from exact stored embedding/LayerNorm coefficients and the frozen float32 positional frequency table. Record the table digest. The ideal real sinusoid convention is explicit; an additional machine-level check is necessary to bridge the captured PyTorch output.
5. Build clean source-only pre-LN1 and middle-LN2 centers and radii through each recurrent visit, preserving the pinned FF-U input reinjection. Compute an upper injection amplitude from the independently enclosed clean RMS.
6. With that amplitude and the clean tubes, run SCM-07 tokenwise Jacobian bounds. Report both source-global and SCM-08 source-derived local upper bounds by visit and token pair.
7. Run the source-only null-braid test using exact source anchor geometry and the disk bound, then optionally compare it to SCM-04/05 sampled braid observations as a diagnostic. A null certificate applies to the defined continuous real graph only.
8. Produce immutable JSONL reports with source hashes, frequency table hash, refused cases, and explicit `floating_capture_certified=false` and `observed_link_certified=false` fields.

## Required negative controls

- Inject at token 63, observe tokens 16, 32, 48, and verify zero derivative for all later visits.
- With beta zero and exact input anchors, source-only tubes remain at their input uncertainty when alpha is one and there is no reinjection.
- Increase intervention radius until the null-disk inequality refuses, even when coarse angular samples appear unchanged.
- Refuse a changed RMP source blob, mismatch in checkpoint/dictionary/chart/tokens or wrong calibration split. Refuse altered attention and biased MLP modules.
- Preserve full unsuccessful and rejected counts; report source-only bounds that become too wide.

## Success criterion

A successful SCM-08 *mathematical gate* is an independently derived and reproducible clean-center error envelope for the defined exact real source graph, with successful replay against mock models and correct refusal controls. A trained-model gate needs accessible RMP artifacts and a pinned frequency table. An observed-neural-braid certificate additionally requires a verified floating-output error enclosure for the trace coordinates.
