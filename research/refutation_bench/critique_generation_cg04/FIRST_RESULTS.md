# CG-04 first complete results

[Run 45335573731](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45335573731), job 138246719072, head 73fc3ff60afb7044915fcf153ab81c036658e0e0. Native reference checks: train 80 families/1959, cal 16/392, depth 32/968, clutter 32/1037 (4356 total). Native actor replay checks: train 1959, cal 506, depth 1218, clutter 1290 (4973 total, including repeated references). Both 225-parameter rankers trained on 1658 proposal pools containing 16613 candidate rows. Registered gate **FAIL**. Combined 4/8-call depth/clutter minimum certificate totals: lex 220, random 220, public 266, rarity 268, learned-91 267, learned-97 268. Threshold 274 for both learned seeds. No invalid certificates or false accusations. At 32 calls all policies certified all 192 held-out cases. Lean checks sufficient coverage, while the independent finite checker certifies minimum cardinality.

| Split | Budget | Policy | Minimum certificates | Calls | Solver states | False accusations |
|---|---:|---|---:|---:|---:|---:|
| cal | 4 | lex | 25/48 | 141 | 2346 | 0 |
| cal | 4 | random | 22/48 | 141 | 2513 | 0 |
| cal | 4 | public | 33/48 | 137 | 2375 | 0 |
| cal | 4 | rarity | 33/48 | 137 | 2405 | 0 |
| cal | 4 | learned-91 | 34/48 | 137 | 2354 | 0 |
| cal | 4 | learned-97 | 34/48 | 137 | 2343 | 0 |
| cal | 8 | lex | 48/48 | 179 | 3210 | 0 |
| cal | 8 | random | 47/48 | 192 | 4057 | 0 |
| cal | 8 | public | 48/48 | 157 | 2854 | 0 |
| cal | 8 | rarity | 48/48 | 157 | 2821 | 0 |
| cal | 8 | learned-91 | 48/48 | 157 | 2802 | 0 |
| cal | 8 | learned-97 | 48/48 | 157 | 2791 | 0 |
| cal | 16 | lex | 48/48 | 179 | 3210 | 0 |
| cal | 16 | random | 48/48 | 193 | 4057 | 0 |
| cal | 16 | public | 48/48 | 157 | 2854 | 0 |
| cal | 16 | rarity | 48/48 | 157 | 2821 | 0 |
| cal | 16 | learned-91 | 48/48 | 157 | 2802 | 0 |
| cal | 16 | learned-97 | 48/48 | 157 | 2791 | 0 |
| cal | 32 | lex | 48/48 | 179 | 3210 | 0 |
| cal | 32 | random | 48/48 | 193 | 4057 | 0 |
| cal | 32 | public | 48/48 | 157 | 2854 | 0 |
| cal | 32 | rarity | 48/48 | 157 | 2821 | 0 |
| cal | 32 | learned-91 | 48/48 | 157 | 2802 | 0 |
| cal | 32 | learned-97 | 48/48 | 157 | 2791 | 0 |
| test_depth | 4 | lex | 32/96 | 288 | 6457 | 0 |
| test_depth | 4 | random | 32/96 | 288 | 6671 | 0 |
| test_depth | 4 | public | 32/96 | 288 | 7425 | 0 |
| test_depth | 4 | rarity | 32/96 | 288 | 7312 | 0 |
| test_depth | 4 | learned-91 | 32/96 | 288 | 7510 | 0 |
| test_depth | 4 | learned-97 | 32/96 | 288 | 7510 | 0 |
| test_depth | 8 | lex | 59/96 | 529 | 22296 | 0 |
| test_depth | 8 | random | 54/96 | 524 | 22371 | 0 |
| test_depth | 8 | public | 77/96 | 487 | 19481 | 0 |
| test_depth | 8 | rarity | 78/96 | 489 | 19769 | 0 |
| test_depth | 8 | learned-91 | 77/96 | 488 | 19616 | 0 |
| test_depth | 8 | learned-97 | 78/96 | 488 | 19506 | 0 |
| test_depth | 16 | lex | 96/96 | 609 | 27538 | 0 |
| test_depth | 16 | random | 96/96 | 622 | 28868 | 0 |
| test_depth | 16 | public | 96/96 | 511 | 20141 | 0 |
| test_depth | 16 | rarity | 96/96 | 510 | 20178 | 0 |
| test_depth | 16 | learned-91 | 96/96 | 511 | 20143 | 0 |
| test_depth | 16 | learned-97 | 96/96 | 510 | 20033 | 0 |
| test_depth | 32 | lex | 96/96 | 609 | 27538 | 0 |
| test_depth | 32 | random | 96/96 | 622 | 28868 | 0 |
| test_depth | 32 | public | 96/96 | 511 | 20141 | 0 |
| test_depth | 32 | rarity | 96/96 | 510 | 20178 | 0 |
| test_depth | 32 | learned-91 | 96/96 | 511 | 20143 | 0 |
| test_depth | 32 | learned-97 | 96/96 | 510 | 20033 | 0 |
| test_clutter | 4 | lex | 42/96 | 285 | 5928 | 0 |
| test_clutter | 4 | random | 44/96 | 286 | 5482 | 0 |
| test_clutter | 4 | public | 61/96 | 278 | 5136 | 0 |
| test_clutter | 4 | rarity | 62/96 | 278 | 5122 | 0 |
| test_clutter | 4 | learned-91 | 62/96 | 278 | 5154 | 0 |
| test_clutter | 4 | learned-97 | 62/96 | 278 | 5150 | 0 |
| test_clutter | 8 | lex | 87/96 | 429 | 12792 | 0 |
| test_clutter | 8 | random | 90/96 | 422 | 11403 | 0 |
| test_clutter | 8 | public | 96/96 | 354 | 8206 | 0 |
| test_clutter | 8 | rarity | 96/96 | 354 | 8260 | 0 |
| test_clutter | 8 | learned-91 | 96/96 | 354 | 8269 | 0 |
| test_clutter | 8 | learned-97 | 96/96 | 354 | 8265 | 0 |
| test_clutter | 16 | lex | 96/96 | 438 | 12792 | 0 |
| test_clutter | 16 | random | 96/96 | 428 | 11403 | 0 |
| test_clutter | 16 | public | 96/96 | 354 | 8206 | 0 |
| test_clutter | 16 | rarity | 96/96 | 354 | 8260 | 0 |
| test_clutter | 16 | learned-91 | 96/96 | 354 | 8269 | 0 |
| test_clutter | 16 | learned-97 | 96/96 | 354 | 8265 | 0 |
| test_clutter | 32 | lex | 96/96 | 438 | 12792 | 0 |
| test_clutter | 32 | random | 96/96 | 428 | 11403 | 0 |
| test_clutter | 32 | public | 96/96 | 354 | 8206 | 0 |
| test_clutter | 32 | rarity | 96/96 | 354 | 8260 | 0 |
| test_clutter | 32 | learned-91 | 96/96 | 354 | 8269 | 0 |
| test_clutter | 32 | learned-97 | 96/96 | 354 | 8265 | 0 |

- Seed 91: 225 parameters, final loss 0.13638822548091412, checkpoint SHA256 4587d4ccb1e5fc7c56360b56db0a99009066ef41358fb7024c01ebbc7cc71b39.
- Seed 97: 225 parameters, final loss 0.13573146640108183, checkpoint SHA256 f50b163802a21a4342785a43859bac89188a7fa57765980c74231585f9a229f4.

[Artifact 14894196487](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45335573731/artifacts/14894196487), ZIP SHA256 54f710e133cef6d5d86e6f10b3824dbca5099eebd368c007b65a5a039f91ab8c. Learned ranking approximately recovers the strength of public rarity/coverage on this planted finite distribution. It does not outperform the exact symbolic ranking control. The four-call deep result includes only the 32 already-valid controls: every weakened deep case requires at least four premise anchors and one successful verification call. This is an oracle-bound limitation, not a model-specific deficiency. Next track CG-05 uses original Lean mathematical source and authored counterfactual advertisements.