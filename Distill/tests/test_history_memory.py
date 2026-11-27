import importlib.util
import tempfile
import unittest
from pathlib import Path

from Distill.history_memory import root_data, verify, specs, context, CTX

class HistoryTests(unittest.TestCase):
    def test_exact_triplets(self):
        spec=next(specs("scarce",1))
        p,r=root_data(spec,12)
        verify(p,r)
        self.assertEqual(len(context(spec)),CTX)
        self.assertGreater(len(r["triplets"]),0)
        self.assertLessEqual(len(r["triplets"]),12)
        self.assertTrue(all(set(x)=={"history"} for x in p["records"].values()))

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"NumPy only needed for memory training")
    def test_training_smoke(self):
        from Distill.history_memory_train import campaign
        with tempfile.TemporaryDirectory() as t:
            r=campaign(Path(t)/"run",counts=(1,1,1,1,1),epochs=2,limit=8,seeds=(311,))
            self.assertEqual(len(r["results"]),4)
            self.assertFalse(r["training_admission"])
            for arm in r["results"].values():
                self.assertGreater(arm["311"]["updates"],0)
                self.assertEqual(len(arm["311"]["evaluation"]),5)
