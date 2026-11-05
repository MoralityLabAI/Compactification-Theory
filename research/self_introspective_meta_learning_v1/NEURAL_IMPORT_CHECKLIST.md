# SIML-01N neural follow-up, source import checklist

The exploratory model and data source are available in the conversation's new neural-source package. Before comparing against independent task families, import and verify:

- `neural_student.py`: tied recurrent GRU, three task families and sealed length-three secondary evaluator
- `neural_curriculum.py`: six matched-update curriculum arms and per-seed training fingerprints
- `test_neural.py`, `test_local_proposer.py`: neural and strict-proposer tests
- `local_proposer.py`, `proposer_cli.py`: loopback JSON proposer with a proposal-only authority boundary
- `verify_neural_receipt.py`: source-validated score/replay verification
- `NEURAL_PROTOCOL.md`, `NEURAL_RESULTS.md`, `results/neural_curriculum_5seed_v02.json.gz`

The proposed 27B interface requires an independently started local OpenAI-compatible endpoint, a pinned model revision and a complete proposer-token budget. No external training or promotion occurs from the connector alone. FAW and FRC remain deferred.

**Evidence boundary:** GitHub at this report commit contains the research summary and checklist. The executable neural files must be checked into the branch separately before CI can claim to exercise them.
