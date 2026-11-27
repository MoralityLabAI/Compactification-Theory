"""Versioned runner profiles and paired calibration, fixtures only."""
import json
import tempfile
import unittest
from pathlib import Path
from Distill.core import read_json
from Distill.inky_free import FREE_MODEL
from Distill.program_runner import (SCHEMA,CALIBRATION_SCHEMA,Journal,prepare,run,
    decode,fixture_response,fixture_raw)
from Distill.inky_program_calibration import prepare_calibration,report,load

class InkyCalibrationTests(unittest.TestCase):
    def test_profile_and_token_limits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg=dict(seed=1,ordinal=0,family="shared_control",difficulty="frontier",mode="fixture",world_seed=7)
            prepare(root/"old",**cfg)
            self.assertEqual(Journal(root/"old").manifest["schema_version"],SCHEMA)
            prepare(root/"new",**cfg,max_generation=8192,effort=.25)
            j=Journal(root/"new")
            self.assertEqual(j.manifest["schema_version"],CALIBRATION_SCHEMA)
            self.assertEqual(j.index,Journal(root/"old").index)
            obs=j.claim()
            r=fixture_response(obs,"defer")
            raw={"provider":"tinker","requested_model":FREE_MODEL,"text":json.dumps(r),
                 "input_tokens":1000,"output_tokens":5000,"clean":True}
            with self.assertRaises(ValueError):
                decode(raw,"live",obs)
            self.assertEqual(decode(raw,"live",obs,max_generation=8192)[1],"defer")
            raw["input_tokens"]=5000
            with self.assertRaises(ValueError):
                decode(raw,"live",obs,max_generation=8192)
            with self.assertRaises(ValueError):
                prepare(root/"bad",**cfg,max_generation=9000,effort=.25)

    def test_pairs_and_censoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"cal"
            m=prepare_calibration(root,roots_per_family=1,fixture=True)
            self.assertEqual(sum(len(r["arms"]) for r in m["roots"]),14)
            self.assertEqual(report(root)["terminal_arms"],0)
            _,arms=load(root)
            for i,(_,_,j) in enumerate(arms):
                if i==0:
                    obs=j.claim()
                    raw=fixture_raw(fixture_response(obs,"defer"))
                    raw["clean"]=False
                    self.assertEqual(j.finish(raw)["status"],"invalid")
                else:
                    self.assertTrue(run(j,lambda obs:fixture_raw(fixture_response(obs,"defer")),32)["complete"])
            out=report(root)
            self.assertTrue(out["all_terminal"])
            self.assertEqual(out["terminal_arms"],14)
            self.assertEqual(sum(r["complete"] for r in out["rows"]),13)
            self.assertIsNone(out["rows"][0]["complete_utility"])
            self.assertFalse(out["training_admission"])
            self.assertTrue(all(r["complete_utility"]==0 for r in out["rows"][1:]))
