"""NOR-12 public-template residual learning, native solver and leakage gates."""
import math
import random
import unittest
from dataclasses import replace

from research.rrc_compactification import compound25 as corpus
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as m


class NOR12Readout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint,cls.probe,cls.probesha,cls.rows=m.locked_rows(
            {"train":8,"dev":3,"test":2})
        train=cls.rows["train"]
        p1,p2=nor11.train_source()[1:]
        diag=nor11.build_rows({"train":nor11.sample_cohorts(
            {"train":8,"dev":3,"test":2})["train"]},cls.checkpoint,p1,p2)["train"]
        shuffled=[z["label"] for z in diag]
        random.Random(nor11.SEED+77).shuffle(shuffled)
        cls.shuffled=nor11.fit_head(diag,seed=nor11.SEED+77,
                                   labels_override=shuffled)
        m.attach_features(cls.rows,cls.probe,cls.shuffled)

    def test_checkpoint_and_public_manifests(self):
        self.assertEqual(self.checkpoint["parameters"],1000)
        self.assertEqual(self.probe["frozen_base_checkpoint_sha256"],m.BASIN_SHA)
        self.assertEqual(len(self.probe["weights"]),13)
        self.assertEqual([len(self.rows[s]) for s in ("train","dev","test")],
                         [24,9,12])
        self.assertEqual(len({r["digest"] for s in ("train","dev") for r in self.rows[s]}),33)
        self.assertFalse({r["digest"] for r in self.rows["test"]}&
                         {r["digest"] for s in ("train","dev") for r in self.rows[s]})
        for r in self.rows["train"]:
            self.assertEqual(len(r["x"]["basin"]),14)
            self.assertEqual(r["x"]["hidden_only"][12],0)
            self.assertEqual(r["x"]["hidden_only"][-1],1)

    def test_private_future_mutation_cannot_change_training_labels(self):
        r=self.rows["train"][0]
        exact=m.visible_mask(r)
        altered=replace(r["case"],program=None,
                        future=(999999,)*len(r["case"].future))
        clone={**r,"case":altered}
        self.assertEqual(exact,m.visible_mask(clone))
        self.assertEqual(nor11.public_digest(altered,r["header"]),r["digest"])
        self.assertTrue(any(exact))

    def test_softmax_valid_mass_gradient(self):
        mask=(True,)+tuple(False for _ in range(15))
        W=[[0.0]*14 for _ in range(16)]
        sample={"base":[0.0]*16,
                "visible_mask":mask,
                "x":{k:[.25]*12+[.5,1.0] for k in m.KINDS}}
        p=m.softmax(sample["base"])
        self.assertEqual(len(p),16)
        self.assertAlmostEqual(sum(p),1)
        loss,g=m.loss_gradient(sample,W,"basin")
        self.assertAlmostEqual(loss,math.log(16),places=12)
        self.assertLess(g[0][-1],0)
        self.assertGreater(g[1][-1],0)
        self.assertAlmostEqual(sum(row[-1] for row in g),0,places=12)

    def test_original_four_check_unchanged(self):
        row=self.rows["train"][0]
        orig=corpus.search(row["case"],"recurrent",self.checkpoint,budget=4)
        direct=m.run_four_checks(row,row["base"])
        self.assertEqual(len(orig["trace"]),direct["paid_checks"])
        self.assertEqual(bool(orig["program"]),bool(direct["accepted"]))
        self.assertEqual(direct["future_correct"],
                         corpus.grade(row["case"],orig["program"])["future_correct"])

    def test_train_correction_and_frozen_baseline(self):
        train=self.rows["train"]
        for r in train:
            r["visible_mask"]=m.visible_mask(r)
        w=m.fit_residual(train,"basin")
        self.assertEqual(len(w["weights"]),16)
        self.assertEqual(len(w["weights"][0]),14)
        self.assertEqual(w["parameter_slots"],224)
        self.assertEqual(w["public_training_oracle_calls"],16*len(train))
        self.assertFalse(w["private_future_truth_access_during_fitting"])
        self.assertTrue(all(math.isfinite(v) for row in w["weights"] for v in row))
        b=m.evaluate(self.rows["test"])
        self.assertLessEqual(b["paid_checks_total"],4*b["cases"])
        self.assertEqual(b["accepted"],b["future_correct"]+b["false_accept"])
        altered=m.evaluate(self.rows["test"],w,"basin",.25)
        self.assertEqual(len(b["records"]),len(altered["records"]))
        comparison=m.paired_comparison(b,altered)
        self.assertEqual(comparison["newly_correct"]-comparison["lost_correct"],
                         altered["future_correct"]-b["future_correct"])

    def test_wrong_labels_or_scales_rejected(self):
        with self.assertRaises(ValueError):
            m.fit_residual([],"basin")
        with self.assertRaises(ValueError):
            m.fit_residual(self.rows["train"],"missing")
        with self.assertRaises(ValueError):
            m.adjusted_scores(self.rows["train"][0],
                              [[0.0]*14 for _ in range(16)],
                              "basin",3)

    def test_end_to_end_bounded_trained_control(self):
        out=m.run({"train":8,"dev":3,"test":2})
        self.assertEqual(out["cohorts"]["train"]["cases"],24)
        self.assertEqual(out["cohorts"]["dev"]["cases"],9)
        self.assertEqual(out["cohorts"]["test"]["cases"],12)
        self.assertEqual(out["source_checkpoint_sha256"],m.BASIN_SHA)
        self.assertEqual(set(out["fitted"]),set(m.KINDS))
        self.assertFalse(out["scope_flags"]["checkpoint_weights_updated"])
        self.assertFalse(out["scope_flags"]["private_256_degree_future_or_program_used_to_train_readouts"])
        self.assertEqual(out["training_full_public_template_checks"],384)
        self.assertEqual(set(out["heldout"]),
                         {"original","basin","shuffled_basin",
                          "random_feature","hidden_only","development_selected"})
        for x in out["heldout"].values():
            self.assertEqual(x["accepted"],x["future_correct"]+x["false_accept"])
            self.assertEqual(x["cases"],12)
            self.assertLessEqual(x["paid_checks_total"],48)


if __name__=="__main__":
    unittest.main()
