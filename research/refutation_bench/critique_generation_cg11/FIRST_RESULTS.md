# CG-11 first complete native results

[Corrected run 45376814359](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45376814359), job 138385182334, head 4e26549539d2b1ddfc73e22e92f5c06fa9c0394c. Initial reserved-keyword failure preserved in INFRASTRUCTURE_RECEIPT.md. Native graph extraction, independent traversal, and both root axiom reports agreed on all 512 selected declarations (5316 root-specific graph nodes). Both 513-parameter rankers trained on 256 trajectories, 1604 action groups and 4484 candidate rows.

Registered learning gate **FAIL**: combined 16/32-read test totals BFS 336, DFS 346, lexical 343, lexical risk 347, learned-161 and learned-167 both 352. Threshold 353. Both learned seeds achieve 64/96 on deep cases at 16 reads, versus 59/96 for lexical risk; all policies reach full coverage at 32 reads. Zero invalid certificates. The standalone lexical classifier falsely accused 15/32 clean calibration roots and 24/48 clean roots on each held-out split, missing no tainted roots. Verified extraction resolves these misleading mentions. This is a hypothetical basis-policy audit, not evidence of false mathematical conclusions.

| Split | Budget | Policy | Complete | Clean | Tainted | Reads | Invalid |
|---|---:|---|---:|---:|---:|---:|---:|
| cal | 8 | bfs | 37/64 | 17/32 | 20/32 | 445 | 0 |
| cal | 8 | dfs | 42/64 | 17/32 | 25/32 | 437 | 0 |
| cal | 8 | lex | 41/64 | 17/32 | 24/32 | 437 | 0 |
| cal | 8 | lexical_risk | 49/64 | 17/32 | 32/32 | 376 | 0 |
| cal | 8 | learned-161 | 49/64 | 17/32 | 32/32 | 367 | 0 |
| cal | 8 | learned-167 | 49/64 | 17/32 | 32/32 | 364 | 0 |
| cal | 16 | bfs | 64/64 | 32/32 | 32/32 | 491 | 0 |
| cal | 16 | dfs | 64/64 | 32/32 | 32/32 | 468 | 0 |
| cal | 16 | lex | 64/64 | 32/32 | 32/32 | 475 | 0 |
| cal | 16 | lexical_risk | 64/64 | 32/32 | 32/32 | 400 | 0 |
| cal | 16 | learned-161 | 64/64 | 32/32 | 32/32 | 391 | 0 |
| cal | 16 | learned-167 | 64/64 | 32/32 | 32/32 | 388 | 0 |
| cal | 32 | bfs | 64/64 | 32/32 | 32/32 | 491 | 0 |
| cal | 32 | dfs | 64/64 | 32/32 | 32/32 | 468 | 0 |
| cal | 32 | lex | 64/64 | 32/32 | 32/32 | 475 | 0 |
| cal | 32 | lexical_risk | 64/64 | 32/32 | 32/32 | 400 | 0 |
| cal | 32 | learned-161 | 64/64 | 32/32 | 32/32 | 391 | 0 |
| cal | 32 | learned-167 | 64/64 | 32/32 | 32/32 | 388 | 0 |
| cal | 64 | bfs | 64/64 | 32/32 | 32/32 | 491 | 0 |
| cal | 64 | dfs | 64/64 | 32/32 | 32/32 | 468 | 0 |
| cal | 64 | lex | 64/64 | 32/32 | 32/32 | 475 | 0 |
| cal | 64 | lexical_risk | 64/64 | 32/32 | 32/32 | 400 | 0 |
| cal | 64 | learned-161 | 64/64 | 32/32 | 32/32 | 391 | 0 |
| cal | 64 | learned-167 | 64/64 | 32/32 | 32/32 | 388 | 0 |
| cal | 128 | bfs | 64/64 | 32/32 | 32/32 | 491 | 0 |
| cal | 128 | dfs | 64/64 | 32/32 | 32/32 | 468 | 0 |
| cal | 128 | lex | 64/64 | 32/32 | 32/32 | 475 | 0 |
| cal | 128 | lexical_risk | 64/64 | 32/32 | 32/32 | 400 | 0 |
| cal | 128 | learned-161 | 64/64 | 32/32 | 32/32 | 391 | 0 |
| cal | 128 | learned-167 | 64/64 | 32/32 | 32/32 | 388 | 0 |
| test_depth | 8 | bfs | 24/96 | 0/48 | 24/48 | 684 | 0 |
| test_depth | 8 | dfs | 14/96 | 0/48 | 14/48 | 732 | 0 |
| test_depth | 8 | lex | 12/96 | 0/48 | 12/48 | 719 | 0 |
| test_depth | 8 | lexical_risk | 24/96 | 0/48 | 24/48 | 648 | 0 |
| test_depth | 8 | learned-161 | 27/96 | 0/48 | 27/48 | 648 | 0 |
| test_depth | 8 | learned-167 | 27/96 | 0/48 | 27/48 | 648 | 0 |
| test_depth | 16 | bfs | 48/96 | 24/48 | 24/48 | 1198 | 0 |
| test_depth | 16 | dfs | 58/96 | 24/48 | 34/48 | 1241 | 0 |
| test_depth | 16 | lex | 55/96 | 24/48 | 31/48 | 1257 | 0 |
| test_depth | 16 | lexical_risk | 59/96 | 24/48 | 35/48 | 1154 | 0 |
| test_depth | 16 | learned-161 | 64/96 | 24/48 | 40/48 | 1091 | 0 |
| test_depth | 16 | learned-167 | 64/96 | 24/48 | 40/48 | 1091 | 0 |
| test_depth | 32 | bfs | 96/96 | 48/48 | 48/48 | 1394 | 0 |
| test_depth | 32 | dfs | 96/96 | 48/48 | 48/48 | 1388 | 0 |
| test_depth | 32 | lex | 96/96 | 48/48 | 48/48 | 1428 | 0 |
| test_depth | 32 | lexical_risk | 96/96 | 48/48 | 48/48 | 1278 | 0 |
| test_depth | 32 | learned-161 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_depth | 32 | learned-167 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_depth | 64 | bfs | 96/96 | 48/48 | 48/48 | 1394 | 0 |
| test_depth | 64 | dfs | 96/96 | 48/48 | 48/48 | 1388 | 0 |
| test_depth | 64 | lex | 96/96 | 48/48 | 48/48 | 1428 | 0 |
| test_depth | 64 | lexical_risk | 96/96 | 48/48 | 48/48 | 1278 | 0 |
| test_depth | 64 | learned-161 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_depth | 64 | learned-167 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_depth | 128 | bfs | 96/96 | 48/48 | 48/48 | 1394 | 0 |
| test_depth | 128 | dfs | 96/96 | 48/48 | 48/48 | 1388 | 0 |
| test_depth | 128 | lex | 96/96 | 48/48 | 48/48 | 1428 | 0 |
| test_depth | 128 | lexical_risk | 96/96 | 48/48 | 48/48 | 1278 | 0 |
| test_depth | 128 | learned-161 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_depth | 128 | learned-167 | 96/96 | 48/48 | 48/48 | 1207 | 0 |
| test_syntax | 8 | bfs | 53/96 | 24/48 | 29/48 | 673 | 0 |
| test_syntax | 8 | dfs | 60/96 | 24/48 | 36/48 | 678 | 0 |
| test_syntax | 8 | lex | 58/96 | 24/48 | 34/48 | 672 | 0 |
| test_syntax | 8 | lexical_risk | 72/96 | 24/48 | 48/48 | 571 | 0 |
| test_syntax | 8 | learned-161 | 72/96 | 24/48 | 48/48 | 559 | 0 |
| test_syntax | 8 | learned-167 | 72/96 | 24/48 | 48/48 | 561 | 0 |
| test_syntax | 16 | bfs | 96/96 | 48/48 | 48/48 | 747 | 0 |
| test_syntax | 16 | dfs | 96/96 | 48/48 | 48/48 | 726 | 0 |
| test_syntax | 16 | lex | 96/96 | 48/48 | 48/48 | 732 | 0 |
| test_syntax | 16 | lexical_risk | 96/96 | 48/48 | 48/48 | 606 | 0 |
| test_syntax | 16 | learned-161 | 96/96 | 48/48 | 48/48 | 594 | 0 |
| test_syntax | 16 | learned-167 | 96/96 | 48/48 | 48/48 | 596 | 0 |
| test_syntax | 32 | bfs | 96/96 | 48/48 | 48/48 | 747 | 0 |
| test_syntax | 32 | dfs | 96/96 | 48/48 | 48/48 | 726 | 0 |
| test_syntax | 32 | lex | 96/96 | 48/48 | 48/48 | 732 | 0 |
| test_syntax | 32 | lexical_risk | 96/96 | 48/48 | 48/48 | 606 | 0 |
| test_syntax | 32 | learned-161 | 96/96 | 48/48 | 48/48 | 594 | 0 |
| test_syntax | 32 | learned-167 | 96/96 | 48/48 | 48/48 | 596 | 0 |
| test_syntax | 64 | bfs | 96/96 | 48/48 | 48/48 | 747 | 0 |
| test_syntax | 64 | dfs | 96/96 | 48/48 | 48/48 | 726 | 0 |
| test_syntax | 64 | lex | 96/96 | 48/48 | 48/48 | 732 | 0 |
| test_syntax | 64 | lexical_risk | 96/96 | 48/48 | 48/48 | 606 | 0 |
| test_syntax | 64 | learned-161 | 96/96 | 48/48 | 48/48 | 594 | 0 |
| test_syntax | 64 | learned-167 | 96/96 | 48/48 | 48/48 | 596 | 0 |
| test_syntax | 128 | bfs | 96/96 | 48/48 | 48/48 | 747 | 0 |
| test_syntax | 128 | dfs | 96/96 | 48/48 | 48/48 | 726 | 0 |
| test_syntax | 128 | lex | 96/96 | 48/48 | 48/48 | 732 | 0 |
| test_syntax | 128 | lexical_risk | 96/96 | 48/48 | 48/48 | 606 | 0 |
| test_syntax | 128 | learned-161 | 96/96 | 48/48 | 48/48 | 594 | 0 |
| test_syntax | 128 | learned-167 | 96/96 | 48/48 | 48/48 | 596 | 0 |

- Seed 161: 513 parameters; final loss 0.04604160062109049, checkpoint SHA256 cf268700d82a2cdb192010e7b30a6a553262d102c65936845ca12fd9a8f193e8.
- Seed 167: 513 parameters; final loss 0.04693962626445752, checkpoint SHA256 227dc2882c3879ad4926b7144d95c12d09efb129ecdb8f6d607e2c7a5401a0f4.

Lexical receipts:
```json
[
  {
    "split": "cal",
    "n": 64,
    "clean": 32,
    "tainted": 32,
    "false_accusations": 15,
    "missed_taint": 0
  },
  {
    "split": "test_depth",
    "n": 96,
    "clean": 48,
    "tainted": 48,
    "false_accusations": 24,
    "missed_taint": 0
  },
  {
    "split": "test_syntax",
    "n": 96,
    "clean": 48,
    "tainted": 48,
    "false_accusations": 24,
    "missed_taint": 0
  }
]
```

[Artifact 14912500613](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45376814359/artifacts/14912500613), ZIP SHA256 50b963a0b881c2b5e0caa24b49207173924d9bb27edc35c799fb681825650ac9. Graph extraction supports only the stated local declaration kinds and imported axiom-summary boundaries. It has not yet established general imported proof-body extraction or arbitrary Lean critique. CG-12 is a development pilot comparing the native axiom closures of original MathCan modules with instrumented copies and extracted graphs.