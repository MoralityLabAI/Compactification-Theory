import tempfile
import unittest
from pathlib import Path
import numpy as np
from activation_io import read_activation
from synthetic_fixture import arrays, write
from frontier import run, retained


class FrontierTests(unittest.TestCase):
    def test_distinct_heldout_roots(self):
        a=arrays()
        c=[{"root_id":d,"case_id":str(d)} for d in (13,73,17)]
        m={"cases":c}
        loaded=[(x,{},a) for x in c]
        r=run(m,loaded,[13,73],[17],(1,2))
        self.assertEqual(r["matched_rank_ceiling"],1)
        self.assertEqual(r["rows"][1]["status"],"infeasible_matched_rank")
        self.assertAlmostEqual(r["rows"][0]["root_macro_retained_energy"]["effect"]["effect"],1)
        self.assertEqual(r["verdict"],"PILOT_ONLY")
        with self.assertRaises(ValueError):
            run(m,loaded,[13],[13])
        with self.assertRaises(ValueError):
            run(m,loaded[:2],[13],[17])

    def test_zero_energy_and_bf16(self):
        self.assertIsNone(retained(np.zeros((1,2)),np.ones((2,1))))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"test.safetensors"
            a=arrays()
            self.assertEqual(len(read_activation(p,write(p,a),3)),11)
            with self.assertRaises(ValueError):
                read_activation(p,"0"*64,3)
