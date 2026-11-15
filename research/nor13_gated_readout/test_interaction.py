"""NOR-13 source locks, public-gradient and exact affine rank certificates."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace
import math
import unittest

from research.nor13_gated_readout import interaction as nor13
from research.nor12_task_readout import readout as nor12
from research.nor11_attractor_controller import controller as nor11


class NOR13GatedReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.ck,cls.probe,cls.negative,cls.old,cls.rows,
         cls.design,cls.original)=nor13.frozen_data()

    def test_original_checkpoint_and_frozen_public_train_roster(self):
        self.assertEqual(self.ck["parameters"],1000)
        self.assertEqual(nor13.READOUT_SHA,nor12.BASIN_SHA)
        self.assertEqual(len(self.rows["train"]),96)
        self.assertEqual(len(self.rows["dev"]),36)
        self.assertEqual(len(self.rows["test"]),72)
        self.assertEqual(set(self.old),set(nor12.KINDS))
        self.assertEqual(nor13.WIDTH,25)
        self.assertEqual(nor13.CLASSES*nor13.WIDTH,400)

    def test_training_public_labels_ignore_hidden_program_and_future(self):
        row=self.rows["train"][0]
        direct=nor12.visible_mask(row)
        changed=replace(row["case"],program=None,
                        future=tuple(123456789 for _ in row["case"].future))
        modified={**row,"case":changed}
        self.assertEqual(direct,nor12.visible_mask(modified))
        self.assertEqual(row["digest"],nor11.public_digest(changed,row["header"]))
        self.assertTrue(any(direct))

    def test_gated_interaction_carries_product_of_hidden_and_probability(self):
        row=self.rows["train"][0]
        x=nor13.gated_vector(row,"basin_gate")
        self.assertEqual(len(x),25)
        self.assertEqual(x[-1],1.)
        g=nor13.scalar(row,"basin_gate")
        self.assertEqual(x[:12],row["x"]["hidden_only"][:12])
        for a,b in zip(x[12:24],x[:12],strict=True):
            self.assertAlmostEqual(a,g*b,places=13)
        null=nor13.gated_vector(row,"basin_gate",override=0.)
        self.assertTrue(all(v==0. for v in null[12:24]))
        control=nor13.gated_vector(row,"constant_gate")
        self.assertEqual(control[12:24],[0.]*12)

    def test_correct_public_valid_mass_gradient(self):
        row=self.rows["train"][0]
        W=[[0.0]*25 for _ in range(16)]
        loss,g=nor13.gradient(row,W,"basin_gate")
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(len(g),16)
        self.assertEqual(len(g[0]),25)
        self.assertAlmostEqual(sum(v[-1] for v in g),0,places=10)
        with self.assertRaises(ValueError):
            nor13.raw_scores(row,[[0.0]*25],"basin_gate",.25)

    def test_affine_endpoint_invariance_and_true_gate_crossing(self):
        row={"base":[float(j) for j in range(16)],
             "x":{"hidden_only":[1.0]+[0.]*11+[1.0]},
             "case":SimpleNamespace(family="toy"),
             "digest":"toy"}
        W=[[0.]*25 for _ in range(16)]
        fixed=nor13.checkpoint_intervals([row],W,1.)
        self.assertEqual(fixed["gate_invariant_cases"],1)
        self.assertEqual(fixed["top4_rationally_gate_invariant_unique_inputs"],1)
        self.assertEqual(fixed["records"][0]["ordered_top4_at_gate_minus_one"],[15,14,13,12])
        self.assertEqual(fixed["records"][0]["ordered_top4_at_gate_plus_one"],[15,14,13,12])
        self.assertEqual(fixed["records"][0]["pair_crossings_or_endpoint_ties"],0)
        W[0][12]=5.
        W[1][12]=-5.
        changed=nor13.checkpoint_intervals([row],W,1.)
        self.assertEqual(changed["gate_variant_cases"],1)
        self.assertGreater(changed["records"][0]["pair_crossings_or_endpoint_ties"],0)
        self.assertEqual(changed["top4_rationally_gate_invariant_unique_inputs"],1)
        self.assertEqual(changed["top4_rationally_gate_variant_unique_inputs"],0)
        self.assertFalse(changed["float_rounding_proven"] if "float_rounding_proven" in changed else True)

    def test_small_training_carries_real_gate_coefficients(self):
        train=self.rows["train"][:16]
        m=nor13.fit(train,"basin_gate")
        self.assertEqual(m["slots"],400)
        self.assertEqual(m["train_public_template_oracle_calls"],16*16)
        self.assertFalse(m["future_256_degree_used_for_train"])
        self.assertTrue(all(math.isfinite(v) for r in m["weights"] for v in r))
        dev=nor13.task_eval(self.rows["dev"][:6],m["weights"],"basin_gate",.25)
        base=nor13.task_eval(self.rows["dev"][:6])
        self.assertEqual(dev["cases"],base["cases"])
        self.assertLessEqual(dev["paid_checks_total"],24)
        self.assertEqual(dev["accepted"],dev["future_correct"]+dev["false_accept"])

    def test_new_test_seeds_exclude_earlier_public_nonlate_tensors(self):
        fresh=nor13.new_heldout(self.original,replicates=1,per_family=2)
        self.assertEqual(len(fresh),1)
        self.assertEqual(len(fresh[0]),12)
        old_features={d for group in self.original.values() for _,_,d in group}
        self.assertFalse({d for case,head,d in fresh[0] if case.family!="late_source"}&old_features)
        self.assertEqual(len({d for case,head,d in fresh[0] if case.family!="late_source"}),10)
        self.assertEqual(len({d for case,head,d in fresh[0] if case.family=="late_source"}),1)
        self.assertEqual(len({nor13.source_fingerprint(case) for case,_,d in fresh[0]}),12)

    def test_reject_unbounded_budgets(self):
        with self.assertRaises(ValueError):
            nor13.new_heldout(self.original,replicates=0)
        with self.assertRaises(ValueError):
            nor13.new_heldout(self.original,per_family=17)
        with self.assertRaises(ValueError):
            nor13.fit([],"basin_gate")
        with self.assertRaises(ValueError):
            nor13.fit(self.rows["train"][:8],"unknown")


if __name__=="__main__":
    unittest.main()
