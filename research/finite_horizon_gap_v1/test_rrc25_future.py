"""FHC-01C trained model source lock and future metric smoke checks."""
from __future__ import annotations

import random
import unittest
from dataclasses import replace

import numpy as np

from bench_rrc25_future import (
    HORIZONS, native_case, gru_step, logits, run, source_header,
)
from research.rrc_compactification import bench25, compound25, model25


class TrainedRRC25Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoints, hashes=bench25.checkpoints()
        cls.checkpoint=checkpoints["recurrent"]
        cls.checkpoint_sha=hashes["recurrent"]
        cls.case=compound25.make_case(
            random.Random(940199),"signed","fhc01c-generated-control")

    def test_original_training_source_and_native_model_contract(self):
        self.assertEqual(self.checkpoint["kind"], "recurrent")
        self.assertEqual(self.checkpoint["parameters"], 1000)
        self.assertEqual(len(self.checkpoint_sha), 64)
        self.assertEqual(model25.HIDDEN, 12)

    def test_public_header_and_native_logits(self):
        header, count=source_header(self.case)
        self.assertGreaterEqual(count, 1)
        self.assertEqual(len(compound25.features(self.case,header)), 9)
        hidden, future, diff, _=native_case(self.case,self.checkpoint,4)
        self.assertEqual(len(hidden),12)
        self.assertEqual(len(future),64)
        self.assertLess(diff, 1e-11)
        self.assertTrue(np.isfinite(future).all())

    def test_constant_zero_continuation_and_hidden_replay(self):
        hidden,future,_,_=native_case(self.case,self.checkpoint,4)
        replay=gru_step(hidden,[0.]*8,self.checkpoint)
        self.assertTrue(np.allclose(
            future[:16],logits(replay,self.checkpoint),atol=1e-12))

    def test_hidden_input_does_not_consult_future_truth(self):
        changed=replace(self.case,future=tuple(
            1000000 if i%2 else -1000000
            for i in range(len(self.case.future))))
        a=native_case(self.case,self.checkpoint,3)
        b=native_case(changed,self.checkpoint,3)
        self.assertEqual(a,b)

    def test_small_trained_pilot_is_holdout_and_deterministic(self):
        a=run(horizons=(1,2),widths=(2,4))
        b=run(horizons=(1,2),widths=(2,4))
        self.assertEqual(a["status"],"FROZEN_TRAINED_RRC25_DEVELOPMENT")
        self.assertEqual(a["checkpoint_sha256"],self.checkpoint_sha)
        self.assertEqual(a["calibration_roster_sha256"],
                         b["calibration_roster_sha256"])
        self.assertEqual(a["heldout_roster_sha256"],
                         b["heldout_roster_sha256"])
        self.assertEqual(len(a["calibration_case_ids"]),36)
        self.assertEqual(len(a["heldout_cases"]),48)
        self.assertEqual(a["public_header_multiple_candidates"],
                         b["public_header_multiple_candidates"])
        self.assertFalse(set(a["calibration_case_ids"]) &
                         {x["id"] for x in a["heldout_cases"]})
        for row in a["cases"]:
            names={x["method"] for x in row["methods"]}
            self.assertIn("native_hidden_12",names)
            self.assertIn("full_trace_identity",names)
            self.assertIn("pca_2",names)
            self.assertIn("gaussian_2",names)
            identity=next(x for x in row["methods"]
                          if x["method"]=="full_trace_identity")
            self.assertEqual(
                identity["heldout"]["violation_counts_calibration_scale"]["2.0"],0)
            self.assertAlmostEqual(
                identity["heldout"]["optimal_rescaling_distortion_heldout"],1.0,
                places=9)
        self.assertEqual(a["source_family_098_infinite_witness_tested"],False)

    def test_rejects_bad_horizons(self):
        with self.assertRaises(ValueError):
            run(horizons=(17,),widths=(2,))


if __name__=="__main__":
    unittest.main()
