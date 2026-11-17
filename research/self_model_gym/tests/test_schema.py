import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from research.self_model_gym.backends import ScriptedBackend
from research.self_model_gym.run import evaluate


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional jsonschema dependency absent")
class TestEvidenceSchema(unittest.TestCase):
    def test_saved_evidence_matches_schema(self):
        from jsonschema import Draft202012Validator
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas/artifacts.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            evaluate(ScriptedBackend(), replicates=1, root_seed=7, output=out, save_traces=True)
            for name, definition in (("manifest.json", "manifest"), ("receipt.json", "receipt")):
                instance = json.loads((out / name).read_text())
                Draft202012Validator({"$ref": "#/$defs/" + definition,
                                     "$defs": schema["$defs"]}).validate(instance)
            for row in (out / "episodes.jsonl").read_text().splitlines():
                Draft202012Validator({"$ref": "#/$defs/episode",
                                     "$defs": schema["$defs"]}).validate(json.loads(row))


if __name__ == "__main__":
    unittest.main()
