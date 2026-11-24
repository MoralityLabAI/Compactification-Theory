# RB-CG-01 immutable first-result receipt

Source head c7268f1998f3e1c5984f6af3f40005d67cd565a4. [First complete native/training run 45328809443](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45328809443), source job 138224713006, training job 138225008303. Both jobs succeeded. Registered gates width32=false, width80=false.

Native source counts: train 256 / 1840 axiom checks, cal 32 / 231, depth 48 / 348, mechanism 48 / 401. Total 384 families, 3072 arms, 2820 axiom checks. Source [artifact 14890468558](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45328809443/artifacts/14890468558), ZIP SHA256 bec29d6cd470a75972e71409e3e2975724388bd08cedde34a706edf4d05914c3. Training [artifact 14890234515](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/45328809443/artifacts/14890234515), ZIP SHA256 f0c754e41e61f2c7420e0643bc72562bad9c4f7b6e75446348c79d0c7eed0955.

| Split | Policy | Complete | Axiom axis | Semantic axis | Malformed | Supported | Verified counter | False accusations on supported |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cal | symbolic | 256/256 | 256 | 256 | 0 | 64/64 | 64/64 | 0/64 |
| cal | constant | 0/256 | 0 | 22 | 168 | 0/64 | 0/64 | 0/64 |
| cal | lexical | 0/256 | 4 | 103 | 0 | 0/64 | 14/64 | 42/64 |
| test_depth | symbolic | 384/384 | 384 | 384 | 0 | 96/96 | 96/96 | 0/96 |
| test_depth | constant | 0/384 | 0 | 96 | 0 | 0/96 | 0/96 | 0/96 |
| test_depth | lexical | 0/384 | 0 | 106 | 0 | 0/96 | 0/96 | 76/96 |
| test_mechanism | symbolic | 384/384 | 384 | 384 | 0 | 96/96 | 96/96 | 0/96 |
| test_mechanism | constant | 0/384 | 0 | 18 | 312 | 0/96 | 0/96 | 0/96 |
| test_mechanism | lexical | 1/384 | 2 | 134 | 4 | 0/96 | 18/96 | 55/96 |
| cal | gru-32-41 | 31/256 | 61 | 146 | 0 | 8/64 | 9/64 | 31/64 |
| test_depth | gru-32-41 | 0/384 | 0 | 177 | 0 | 0/96 | 9/96 | 92/96 |
| test_mechanism | gru-32-41 | 47/384 | 87 | 243 | 0 | 11/96 | 22/96 | 51/96 |
| cal | gru-32-43 | 13/256 | 40 | 92 | 0 | 3/64 | 0/64 | 2/64 |
| test_depth | gru-32-43 | 0/384 | 0 | 129 | 2 | 0/96 | 0/96 | 11/96 |
| test_mechanism | gru-32-43 | 8/384 | 25 | 143 | 0 | 3/96 | 0/96 | 9/96 |
| cal | gru-80-41 | 38/256 | 70 | 153 | 1 | 13/64 | 4/64 | 22/64 |
| test_depth | gru-80-41 | 0/384 | 0 | 150 | 0 | 0/96 | 12/96 | 87/96 |
| test_mechanism | gru-80-41 | 39/384 | 65 | 251 | 0 | 15/96 | 6/96 | 33/96 |
| cal | gru-80-43 | 24/256 | 96 | 64 | 0 | 21/64 | 0/64 | 0/64 |
| test_depth | gru-80-43 | 0/384 | 0 | 91 | 0 | 0/96 | 0/96 | 0/96 |
| test_mechanism | gru-80-43 | 30/384 | 120 | 96 | 0 | 30/96 | 0/96 | 4/96 |

## Frozen trained checkpoints

- gru-32-41: 39788 parameters, final token CE 0.45057275495223315, training CPU seconds 31.628714944000002, checkpoint SHA256 55b4e7c44c747679122e6d1f4c48c9e91c94579b5341b35acbf1e348061b5ae5.
- gru-32-43: 39788 parameters, final token CE 0.6123360485856193, training CPU seconds 28.778156923000004, checkpoint SHA256 1dc8e2b38de627c33b6a5a0bdf46467438811f156dabe29196a4fe72f94cb556.
- gru-80-41: 206972 parameters, final token CE 0.3504942691294854, training CPU seconds 53.34484029000001, checkpoint SHA256 726c7ada4e29c4f85379135bd21870af51cace63b4d6799313b82557d5ba982d.
- gru-80-43: 206972 parameters, final token CE 0.3903280183480366, training CPU seconds 43.173838824, checkpoint SHA256 0099f0cbe92d44f195b8abc2c16940060df6ec2835e4acd24193a01da8884866.

Every learned arm scored zero complete axiom chains on deeper families. Near-zero malformed syntax shows that the primary issue is correct compositional pointer execution and scope/witness prediction, rather than output syntax. Some learned proposals make false accusations at high rates; they were rejected by the checker. For example 32-41 accuses 92/96 supported depth controls, and 80-41 accuses 87/96. The deterministic renderer only releases verified critiques. 80-43's zero false accusations on depth must be interpreted with its zero complete critiques and zero verified counters, not as reliable auditing.

The fixed symbolic parser is exact for the generated grammar, with 100% complete certificates. This does not imply arbitrary Lean source audit capability. Scaling alone did not resolve depth failure. Follow-up: local transition learning with explicit declaration identity, a recurrent execution trace and truth-table gate execution. Test learned and exact submodules separately; use new source seeds after seeing these results. Preserve the first run and both failed gates.
