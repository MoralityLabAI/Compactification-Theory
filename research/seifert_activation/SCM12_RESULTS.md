# SCM-12 exact-algebra and campaign reference, November 14, 2026

**Evidence class:** exact-rational rank-one theorem, executed synthetic source fixtures, matched-campaign validation tests, and a deterministic simulated 64-task comparison. No trained RMP weights, D-shared dictionaries or actual activation captures were available in this environment. No compactification performance improvement or surface isotopy result is asserted.

## Executed checks

```
python -m unittest discover -s research/seifert_activation/tests -p 'test_scm12.py' -v
python research/seifert_activation/scm12_reference.py --check --out research/seifert_activation/results/scm12_synthetic_reference.json
```

The 23 new unit tests cover: exact-rational rank-one certificates and collision refusals; match checks for two model cells by task/group/target; intervention identity under chart-only replacements; missing-task rejection; unlabeled/unknown-source refusals; unresolved Artin-budget recording; a deterministic exact sign-test reference; checkpoint/dictionary/feature lock preflight; source/dictionary mutation refusal; and exact RMP-style D-shared selector ordering.

The immutable reference deliberately synthesizes 64 group-distinct tasks balanced 16/family. Its positive and negative braid flags are *fictional* and tagged `synthetic_fixture`; the program must label its result `synthetic_only`. It tests aggregation and task controls, not model mechanism. All real-world claims depend on external artifact access and later registered causal validation.

## Known external blockers

The linked `MoralityLabAI/LoopedTransformers` repository includes RMP model code, train metadata, training outcomes and paper evidence. The `C3.pt` artifacts, per-checkpoint integrity sidecars, selected D-shared dictionary files, and actual monitor calibration shards are maintained under the user's external RMP artifact paths. These bytes were unavailable here. The frozen feature identities and provenance were also unavailable. Source-only calculations cannot substitute for them.

## Reproduction on the RMP execution host

```
python research/seifert_activation/scm12_launch.py --mode preflight \
  --rmp-root /path/to/LoopedTransformers \
  --runs /path/to/recurrence_monitorability_v0/runs \
  --artifacts /path/to/rmp_v0/artifacts \
  --monitor-bundle /path/to/rmp_v0/bundle \
  --metrics /path/to/dictionaries.json \
  --feature-provenance /path/to/calibration_feature_lock.json \
  --loop-dictionary /path/to/selected_loop_shared.pt \
  --ffu-dictionary /path/to/selected_ffu_shared.pt \
  --loop-features I J --ffu-features K L
```

Replace I/J/K/L with previously frozen, selection-half feature IDs. After `preflight` reports `ready_for_external_RMP_capture`, repeat the same arguments with `--mode freeze --lock <fresh_lock.json>` and then call `--mode run --lock <fresh_lock.json> --out <fresh_campaign_dir> --device cpu` on the same host. The script launches the four captures and the paired report, using the existing SCM-04 capture implementation. CUDA requires separately checking device determinism and respecting RMP's GPU scheduling rules.
