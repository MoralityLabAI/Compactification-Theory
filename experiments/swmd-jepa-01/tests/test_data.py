import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import ACTIONS, Vocabulary, digest, load_jsonl, mask_view, split_worlds, step, synthetic, tokens


class DataTests(unittest.TestCase):
    def test_deterministic_and_cross_world_split(self):
        rows = synthetic(seed=101, worlds=12, steps=5)
        self.assertEqual(digest(rows), digest(synthetic(seed=101, worlds=12, steps=5)))
        splits = split_worlds(rows, 11)
        groups = [{r['world_id'] for r in part} for part in splits.values()]
        self.assertEqual(len(set.union(*groups)), 12)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])
        for part in splits.values():
            self.assertTrue(part)

    def test_transitions_and_views(self):
        row = synthetic(worlds=4, steps=2)[0]
        self.assertNotEqual(row['view_a'], row['view_b'])
        self.assertEqual(len(Vocabulary([row]).encode(row['view_a'])), len(tokens(row['view_a'])) + 1)
        self.assertIn('<mask>', mask_view(row['view_a'], __import__('random').Random(1)))
        s = dict(room=0, key=0, gate=0, energy=0, sigil=0)
        for action in ACTIONS:
            self.assertNotEqual(step(s, action), s)

    def test_bad_external_schema(self):
        row = synthetic(worlds=4, steps=2)[0]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.jsonl'
            row['next_world_id'] = 'other'
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'crosses world groups'):
                load_jsonl(path)
            del row['next_world_id']
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(len(load_jsonl(path)), 1)


if __name__ == '__main__':
    unittest.main()
