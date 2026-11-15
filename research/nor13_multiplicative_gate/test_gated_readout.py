"""NOR-13 independent source, nonlinear gate and native verifier controls."""
import math
import unittest
from dataclasses import replace

from research.rrc_compactification import compound25 as corpus
from research.nor12_task_readout import readout as old
from research.nor13_multiplicative_gate import gated_readout as m


class MultiplicativeBasinGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck,cls.probe,cls.negative,cls.rows,cls.roster,cls.headsha=m.frozen_gate_inputs()

    def test_frozen_checkpoint_and_public_split(self):
        self.assertEqual(self.ck["parameters"],1000)
        self.assertEqual(m.SOURCE_SHA,old.BASIN_SHA)
        self.assertEqual(len(self.rows["train"]),96)
        self.assertEqual(len(self.rows["dev"]),36)
        self.assertEqual(len(self.rows["test"]),72)
        self.assertEqual(len(self.probe["weights"]),13)
        self.assertEqual(len(m.PROJECTION),12)
        self.assertAlmostEqual(math.hypot(*m.PROJECTION),1,places=12)
        self.assertEqual(len({z["digest"] for z in self.rows["test"]}),61)

    def test_interaction_changes_slope_and_is_nonlinear(self):
        h0=self.rows["train"][0]["h"]
        f,gate=m.gated_row(h0,self.rows["train"][0]["digest"],self.probe,self.negative)
        self.assertEqual(set(f),set(m.KINDS))
        self.assertEqual(len(f["basin_interaction"]),26)
        self.assertEqual(f["basin_interaction"][:13],f["flat_readout"][:13])
        self.assertTrue(all(x==0 for x in f["flat_readout"][13:]))
        self.assertTrue(all(x==0 for x in f["additive_only_basin"][13:25]))
        self.assertEqual(f["basin_interaction"][-1],gate["basin_interaction"])
        W=[[0.0]*26 for _ in range(16)]
        W[0][13]=3.0
        row={"base":[0.0]*16,"gate_features":f}
        logits=m.corrected_scores(row,W,"basin_interaction",.25)
        self.assertAlmostEqual(logits[0],.75*gate["basin_interaction"]*f["basin_interaction"][0])

    def test_public_task_labels_ignore_future_truth_and_generator_program(self):
        row=self.rows["train"][0]
        lab=old.visible_mask(row)
        changed=replace(row["case"],program=None,future=(99,)*len(row["case"].future))
        self.assertEqual(lab,old.visible_mask({**row,"case":changed}))
        self.assertTrue(any(lab))

    def test_valid_mass_gradient_and_optimizer(self):
        W=[[0.0]*26 for _ in range(16)]
        sample={"base":[0.0]*16,
                "visible_mask":(True,)+tuple(False for _ in range(15)),
                "gate_features":{k:[.3]*26 for k in m.KINDS}}
        loss,gr=m.gradient(sample,W,"basin_interaction")
        self.assertAlmostEqual(loss,math.log(16),places=12)
        self.assertAlmostEqual(sum(row[-1] for row in gr),0,places=12)
        self.assertLess(gr[0][-1],0)
        self.assertGreater(gr[1][-1],0)
        mini=list(self.rows["train"][:8])
        for row in mini:
            row["visible_mask"]=old.visible_mask(row)
        fit=m.fit(mini,"basin_interaction")
        self.assertEqual(fit["parameter_slots"],416)
        self.assertEqual(len(fit["weights"]),16)
        self.assertEqual(len(fit["weights"][0]),26)
        self.assertTrue(all(math.isfinite(x) for row in fit["weights"] for x in row))

    def test_original_readout_four_check_parity(self):
        row=self.rows["test"][0]
        original=corpus.search(row["case"],"recurrent",self.ck,budget=4)
        real=m.evaluate([row])
        self.assertEqual(real["cases"],1)
        self.assertEqual(real["records"][0]["paid_checks"],original["full_checks"])
        self.assertEqual(real["records"][0]["future_correct"],
            corpus.grade(row["case"],original["program"])["future_correct"])

    def test_fresh_seed_separation_and_aliased_late_inputs(self):
        r=m.fresh_source_cohorts(self.roster,replicates=1,per_family=2)
        self.assertEqual(len(r[0]),12)
        train_dev={v for split in ("train","dev") for c,header,v in self.roster[split]}
        self.assertFalse({d for c,header,d in r[0]} & train_dev)
        late=[d for c,header,d in r[0] if c.family=="late_source"]
        self.assertEqual(len(late),2)
        self.assertEqual(len(set(late)),1)

    def test_bounded_full_prospective_protocol(self):
        out=m.run(reps=1,per_family=2)
        self.assertEqual(out["training_cases"],96)
        self.assertEqual(out["development_cases"],36)
        self.assertEqual(out["training_full_public_exact_template_checks"],1536)
        self.assertEqual(out["new_readout_parameter_slots_per_arm"],416)
        self.assertFalse(out["original_gru_weights_modified"])
        self.assertFalse(out["private_future_used_to_train_gated_readouts"])
        self.assertEqual(out["fresh_aggregate"]["original"]["cases"],12)
        self.assertEqual(len(out["fresh_replicates"]),1)
        for k,v in out["fresh_aggregate"].items():
            self.assertEqual(v["cases"],12)
            self.assertLessEqual(v["paid_checks"],48)
            self.assertEqual(v["net_case_gain"],
                             v["future_correct"]-out["fresh_aggregate"]["original"]["future_correct"])
        variants=out["predeclared_gate_ablations_on_fresh_test"][0]["ablation"]["variants"]
        self.assertEqual(set(variants),{
            "true_basin_gate","gate_zero",
            "shuffled_label_gate","permuted_feature_clusters"})

    def test_reject_invalid_gates_and_repetitions(self):
        with self.assertRaises(ValueError):
            m.fit([],"basin_interaction")
        with self.assertRaises(ValueError):
            m.fresh_source_cohorts(self.roster,replicates=0)
        with self.assertRaises(ValueError):
            m.fresh_source_cohorts(self.roster,replicates=7)


if __name__=="__main__":
    unittest.main()
