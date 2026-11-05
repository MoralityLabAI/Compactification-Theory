"""Finite, reproducible checks for the ACE-style synthetic gym."""
import copy
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from research.ace_mechinterp_gym import gym, verify


class GymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(gym.DEFAULT_MANIFEST.read_text())

    def test_exact_task_quotients_and_rates(self):
        values = {c["id"]: (gym.minimum_bits(gym.system_for(c)), gym.probability_exact(c))
                  for c in self.m["candidates"]}
        self.assertEqual(values, {"full": (2,Fraction(0)), "task_id_only": (1,Fraction(0)),
                                  "task_reveal": (2,Fraction(1,4)), "constant": (2,Fraction(1,2))})
        self.assertIsNotNone(gym.task_loss_witness(gym.system_for(self.m["candidates"][2]),gym.CODES["task"]))
        self.assertIsNone(gym.task_loss_witness(gym.system_for(self.m["candidates"][0]),gym.CODES["full"]))

    def test_fresh_synthetic_clusters_and_reproducibility(self):
        a = gym.simulate(self.m)
        self.assertEqual(a,gym.simulate(self.m))
        self.assertEqual(len(a),4*128)
        self.assertEqual(len({x["source_cluster"] for x in a}),len(a))
        self.assertEqual(sum(x["reported"] is None for x in a),1)
        self.assertEqual(sum(x["reported"] != x["clean"] and x["reported"] is not None for x in a),1)

    def test_scope_budget_cap(self):
        m = copy.deepcopy(self.m)
        m["candidates"][0]["alpha"]="0.03"
        with self.assertRaises(ValueError): gym.validate_manifest(m)
        m = copy.deepcopy(self.m)
        m["candidates"][0]["id"]=m["candidates"][1]["id"]
        with self.assertRaises(ValueError): gym.validate_manifest(m)

    def test_attacker_budget_and_schedule(self):
        m = copy.deepcopy(self.m)
        m["synthetic_attacks"].append({"candidate":"full","index":31,"kind":"flip"})
        with self.assertRaises(ValueError): gym.validate_manifest(m)
        m = copy.deepcopy(self.m)
        m["synthetic_attacks"].append({"candidate":"full","index":7,"kind":"erase"})
        with self.assertRaises(ValueError): gym.validate_manifest(m)

    def test_supermartingale_one_step_endpoint(self):
        p,q=self.m["null_violation_floor"],self.m["alternative_violation_rate"]
        for actual in (p, 0.25, 0.5, 1):
            e1=q/p
            e0=(1-q)/(1-p)
            self.assertLessEqual(actual*e1+(1-actual)*e0,1+1e-12)
        self.assertAlmostEqual(p*e1+(1-p)*e0,1)

    def test_clean_likelihood_ratio_replay(self):
        p,q=.2,.05
        for bits in product((0,1),repeat=5):
            l = gym.robust_prefixes(list(bits),p,q,0)
            for i in range(1,6):
                b=bits[:i]
                exact=b.count(1)*__import__("math").log(q/p)+b.count(0)*__import__("math").log((1-q)/(1-p))
                self.assertAlmostEqual(l[i-1],exact)

    def test_lower_envelope_every_compatible_corruption_and_erasure(self):
        # All clean 5-bit traces, every reported trace over {0,1,erased}
        # with <=1 substitution. Check the entire prefix, not just final scores.
        p,q=.2,.05
        import math
        for clean in product((0,1),repeat=5):
            for observed in product((0,1,None),repeat=5):
                if sum(o is not None and o!=t for o,t in zip(observed,clean))>1:
                    continue
                envelope=gym.robust_prefixes(list(observed),p,q,1)
                for prefix in range(1,6):
                    c=clean[:prefix]
                    actual=sum(math.log(q/p) if b==1 else math.log((1-q)/(1-p)) for b in c)
                    self.assertLessEqual(envelope[prefix-1],actual+1e-12)

    def test_matched_bayesian_factor_is_same_function(self):
        # Independent exhaustive BF minimization across every compatible clean
        # stream for 4-round reported traces. Equal prior and cost ratio 1/alpha.
        p,q=.2,.05
        import math
        for reports in product((0,1,None),repeat=4):
            for k in (0,1,2):
                ace=gym.robust_prefixes(list(reports),p,q,k)
                bayes=[]
                for t in range(1,5):
                    viable = [sum(math.log(q/p) if b else math.log((1-q)/(1-p)) for b in clean)
                              for clean in product((0,1),repeat=t)
                              if sum(o is not None and o!=b for o,b in zip(reports[:t],clean))<=k]
                    bayes.append(min(viable))
                    self.assertAlmostEqual(ace[t-1],bayes[-1])
                self.assertEqual(gym.first_crossing(ace,"0.20"),
                                 gym.first_crossing(bayes,"0.20"))

    def test_correlated_copies_counterexample(self):
        data=gym.copied_source_counterexample(self.m)
        self.assertEqual(data["naive_false_certification_probability"],"3/4")
        self.assertEqual(data["naive_first_crossing"],27)
        self.assertIsNone(data["cluster_aware_first_crossing"])

    def test_replay_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            gym.write_run(gym.DEFAULT_MANIFEST,out)
            result=verify.verify(gym.DEFAULT_MANIFEST,out)
            self.assertEqual(result["budget"]["allocated_alpha"],"0.04")
            receipt=json.loads((out/"receipt.json").read_bytes())
            receipt["diagnostics"]["results"][0]["ace_first_certification_round"]=1
            (out/"receipt.json").write_bytes(gym.canonical_bytes(receipt))
            with self.assertRaisesRegex(ValueError,"diagnostics differ"):
                verify.verify(gym.DEFAULT_MANIFEST,out)

    def test_source_and_trace_digest_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            gym.write_run(gym.DEFAULT_MANIFEST,out)
            samples=json.loads((out/"trace.json").read_bytes())
            samples[0]["reported"]=1-samples[0]["clean"]
            (out/"trace.json").write_bytes(gym.canonical_bytes(samples))
            with self.assertRaisesRegex(ValueError,"trace digest mismatch"):
                verify.verify(gym.DEFAULT_MANIFEST,out)
            receipt=json.loads((out/"receipt.json").read_bytes())
            receipt["trace_sha256"]=gym.sha256((out/"trace.json").read_bytes())
            (out/"receipt.json").write_bytes(gym.canonical_bytes(receipt))
            with self.assertRaisesRegex(ValueError,"reported outcome differs"):
                verify.verify(gym.DEFAULT_MANIFEST,out)

    def test_invalid_report_type(self):
        with self.assertRaises(ValueError):
            gym.robust_prefixes([False],.2,.05,0)


if __name__ == "__main__":
    unittest.main()
