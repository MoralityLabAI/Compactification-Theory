"""QO-13 fixed paired experiment integrity tests."""
import unittest
from fractions import Fraction as Q
from research.ingression_dynamics_v1.certificate_reachability_pilot import (
 SCENARIOS,SEEDS,POLICIES,CHECKPOINTS,run_new,full_report)
from research.ingression_dynamics_v1.paired_probe_pilot import verify_common_stream_prefix

class CertificatePilotTests(unittest.TestCase):
 def test_frozen_casebook(self):
  self.assertEqual(len(SCENARIOS),4)
  self.assertEqual(SEEDS,(541,547,557))
  self.assertEqual(CHECKPOINTS,(24,64,96))
  self.assertEqual(POLICIES,("bottleneck","certificate_h2","certificate_h3"))
  self.assertEqual(sum("new fixed" in c["panel"] for c in SCENARIOS),2)
  self.assertTrue(all(c["classes"]==1 for c in SCENARIOS))
 def test_paired_policy_reproducible_and_complete(self):
  c=SCENARIOS[0]
  for policy in POLICIES[1:]:
   a,path=run_new(0,c,541,policy,(0,4,12))
   b,replay=run_new(0,c,541,policy,(0,4,12))
   self.assertEqual((a,path),(b,replay))
   self.assertEqual(len(path),12)
   self.assertTrue(verify_common_stream_prefix(0,c,541,path))
   self.assertEqual([r["queries"] for r in a],[0,4,12])
   self.assertTrue(all(not r["false_cert"] or not r["covered"] for r in a))
 def test_small_full_report_and_all_groups(self):
  r=full_report(scenarios=SCENARIOS[:2],seeds=SEEDS[:2],checkpoints=(0,4,12))
  self.assertEqual(r["counts"],{"trajectories":12,"snapshots":36,"queries":144})
  self.assertEqual(len(r["groups"]),18)
  self.assertEqual(len(r["trial_rows"]),36)
  self.assertEqual(len(r["trajectory_hashes"]),12)
  self.assertEqual(r,full_report(scenarios=SCENARIOS[:2],seeds=SEEDS[:2],
                                  checkpoints=(0,4,12)))
  self.assertTrue(all(not v["false_cert"] or not v.get("covered",v.get("true_rates_covered"))
                      for v in r["trial_rows"]))
 def test_input_validation(self):
  with self.assertRaises(ValueError):
   run_new(0,SCENARIOS[0],541,"wrong",(0,4))
if __name__=="__main__":unittest.main()
