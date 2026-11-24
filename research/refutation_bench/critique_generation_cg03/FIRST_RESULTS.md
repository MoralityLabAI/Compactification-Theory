# CG-03 first complete results

[Run 45335012588](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45335012588), job 138244931698, source head d2fe27a565137918282e3f6bca7e8bdc24b10ce6. The initial [run 45334420123](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45334420123) and balanced-conjunction [run 45334901397](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45334901397) failed native synthesis before training. Final correction generates explicit Or.inl/Or.inr/And.intro proof terms for exactly the same witness propositions, preserving source families, seeds, search and gate.

Native Lean verified 192 families and 1536 axiom checks (train 96/768, cal 24/192, depth 36/288, clutter 36/288). Both 898-parameter scorers trained on 3562 branch groups and 1918 repair rows. Registered gate **FAIL**: sums over both held-out splits at 16 and 32 nodes were lex 369, occurrence 507, JW 528, learned-81 520, learned-83 518. The required threshold was 536 for both learned arms. No rejected verdict or repair certificates in scored receipts. Native certification of reference proof/witnesses and independent DPLL checking of actor certificates are separate.

| Split | Budget | Policy | Complete | Verdict | Repair | Nodes | Rejected verdicts | Rejected repairs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cal | 16 | lex | 65/96 | 87 | 26/48 | 1074 | 0 | 0 |
| cal | 16 | occurrence | 93/96 | 96 | 45/48 | 737 | 0 | 0 |
| cal | 16 | jw | 94/96 | 96 | 46/48 | 725 | 0 | 0 |
| cal | 16 | learned-81 | 94/96 | 96 | 46/48 | 718 | 0 | 0 |
| cal | 16 | learned-83 | 94/96 | 96 | 46/48 | 725 | 0 | 0 |
| cal | 32 | lex | 94/96 | 96 | 46/48 | 1267 | 0 | 0 |
| cal | 32 | occurrence | 96/96 | 96 | 48/48 | 757 | 0 | 0 |
| cal | 32 | jw | 96/96 | 96 | 48/48 | 731 | 0 | 0 |
| cal | 32 | learned-81 | 96/96 | 96 | 48/48 | 720 | 0 | 0 |
| cal | 32 | learned-83 | 96/96 | 96 | 48/48 | 727 | 0 | 0 |
| cal | 128 | lex | 96/96 | 96 | 48/48 | 1273 | 0 | 0 |
| cal | 128 | occurrence | 96/96 | 96 | 48/48 | 757 | 0 | 0 |
| cal | 128 | jw | 96/96 | 96 | 48/48 | 731 | 0 | 0 |
| cal | 128 | learned-81 | 96/96 | 96 | 48/48 | 720 | 0 | 0 |
| cal | 128 | learned-83 | 96/96 | 96 | 48/48 | 727 | 0 | 0 |
| test_depth | 16 | lex | 16/144 | 65 | 3/72 | 2209 | 0 | 0 |
| test_depth | 16 | occurrence | 80/144 | 136 | 15/72 | 1925 | 0 | 0 |
| test_depth | 16 | jw | 98/144 | 142 | 28/72 | 1856 | 0 | 0 |
| test_depth | 16 | learned-81 | 88/144 | 138 | 22/72 | 1875 | 0 | 0 |
| test_depth | 16 | learned-83 | 86/144 | 137 | 21/72 | 1891 | 0 | 0 |
| test_depth | 32 | lex | 73/144 | 113 | 23/72 | 3783 | 0 | 0 |
| test_depth | 32 | occurrence | 140/144 | 144 | 68/72 | 2321 | 0 | 0 |
| test_depth | 32 | jw | 144/144 | 144 | 72/72 | 2026 | 0 | 0 |
| test_depth | 32 | learned-81 | 144/144 | 144 | 72/72 | 2118 | 0 | 0 |
| test_depth | 32 | learned-83 | 144/144 | 144 | 72/72 | 2136 | 0 | 0 |
| test_depth | 128 | lex | 139/144 | 144 | 67/72 | 6096 | 0 | 0 |
| test_depth | 128 | occurrence | 144/144 | 144 | 72/72 | 2355 | 0 | 0 |
| test_depth | 128 | jw | 144/144 | 144 | 72/72 | 2026 | 0 | 0 |
| test_depth | 128 | learned-81 | 144/144 | 144 | 72/72 | 2118 | 0 | 0 |
| test_depth | 128 | learned-83 | 144/144 | 144 | 72/72 | 2136 | 0 | 0 |
| test_clutter | 16 | lex | 137/144 | 142 | 66/72 | 976 | 0 | 0 |
| test_clutter | 16 | occurrence | 143/144 | 144 | 71/72 | 812 | 0 | 0 |
| test_clutter | 16 | jw | 142/144 | 144 | 70/72 | 877 | 0 | 0 |
| test_clutter | 16 | learned-81 | 144/144 | 144 | 72/72 | 801 | 0 | 0 |
| test_clutter | 16 | learned-83 | 144/144 | 144 | 72/72 | 795 | 0 | 0 |
| test_clutter | 32 | lex | 143/144 | 144 | 71/72 | 1032 | 0 | 0 |
| test_clutter | 32 | occurrence | 144/144 | 144 | 72/72 | 813 | 0 | 0 |
| test_clutter | 32 | jw | 144/144 | 144 | 72/72 | 882 | 0 | 0 |
| test_clutter | 32 | learned-81 | 144/144 | 144 | 72/72 | 801 | 0 | 0 |
| test_clutter | 32 | learned-83 | 144/144 | 144 | 72/72 | 795 | 0 | 0 |
| test_clutter | 128 | lex | 144/144 | 144 | 72/72 | 1035 | 0 | 0 |
| test_clutter | 128 | occurrence | 144/144 | 144 | 72/72 | 813 | 0 | 0 |
| test_clutter | 128 | jw | 144/144 | 144 | 72/72 | 882 | 0 | 0 |
| test_clutter | 128 | learned-81 | 144/144 | 144 | 72/72 | 801 | 0 | 0 |
| test_clutter | 128 | learned-83 | 144/144 | 144 | 72/72 | 795 | 0 | 0 |

- Seed 81: 898 parameters, final loss 0.44089799161468235, checkpoint SHA256 c723516d3c21951016604c492cc62214bbbb6ee60898568b3bfa919a4c1ef696.
- Seed 83: 898 parameters, final loss 0.44867882824369837, checkpoint SHA256 e8c6a204d3b16e940497c3fffae34917d6a49230a7f2708500d98330922ba05c.

[Artifact 14893437518](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45335012588/artifacts/14893437518), ZIP SHA256 b7cdaee2c5c674b9fd936c9dec72a1ed70006d8b52da06c7a97117c62cbe6e27. The learned controllers slightly improve clutter coverage at 16 nodes and use fewer nodes on clutter, but trail JW on deeper families. No evidence of general learned superiority. Proceed to multi-premise minimum repair using the countermodel hitting-set theorem.
