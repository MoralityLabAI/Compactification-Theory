"""NOR-15B four-check and public-label source invariants."""
from __future__ import annotations
import math
import random
import unittest

from research.nor15_trajectory_sufficiency import task_test as n


def synthetic_train():
    rng=random.Random(202611151500)
    train=[]
    for i in range(96):
        z9=[rng.gauss(0,1) for _ in range(12)]
        extra=[rng.gauss(0,1) for _ in range(12)]
        x=z9+extra+[1.]
        train.append({
            "base":[.05*(j%4) for j in range(16)],
            "visible_mask":tuple(j==i%16 for j in range(16)),
            "nor15b_x":{kind:
                (z9+[0.]*12+[1.] if kind=="hidden_only" else x)
                for kind in n.KINDS},
        })
    return train


class NOR15BReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=synthetic_train()

    def test_exact_train_oracle_gradient_shape(self):
        W=[[0.]*25 for _ in range(16)]
        for kind in n.KINDS:
            loss,grad=n.gradient(self.train[0],W,kind)
            self.assertTrue(math.isfinite(loss))
            self.assertEqual(len(grad),16)
            self.assertTrue(all(len(x)==25 for x in grad))
            self.assertAlmostEqual(sum(row[-1] for row in grad),0.,places=10)
        with self.assertRaises(ValueError):
            n.raw_scores(self.train[0],W,"bad_kind",.5)
        with self.assertRaises(ValueError):
            n.raw_scores(self.train[0],W,"residual_early",3.)
        with self.assertRaises(ValueError):
            n.raw_scores(self.train[0],W[:1],"residual_early",.5)

    def test_public_train_96_cases_and_400_slots(self):
        model=n.fit(self.train,"residual_early",epochs=2)
        self.assertEqual(model["allocated_slots"],400)
        self.assertEqual(model["active_slots"],400)
        self.assertEqual(model["public_template_label_exposures"],1536)
        self.assertEqual(model["trained_cases"],96)
        self.assertFalse(model["private_future_labels_used"])
        self.assertTrue(all(math.isfinite(v) for w in model["weights"] for v in w))
        with self.assertRaises(ValueError):
            n.fit(self.train[:95],"residual_early")
        with self.assertRaises(ValueError):
            n.fit(self.train,"residual_early",epochs=0)
        with self.assertRaises(ValueError):
            n.fit(self.train,"unknown",epochs=1)

    def test_paired_primary_excludes_late_aliased_cases(self):
        original=[
            {"public_features_sha256":"a","family":"signed","future_correct":0},
            {"public_features_sha256":"b","family":"rational","future_correct":1},
            {"public_features_sha256":"late","family":"late_source","future_correct":0},
            {"public_features_sha256":"late","family":"late_source","future_correct":0},
        ]
        alternative=[
            {**original[0],"future_correct":1},
            {**original[1],"future_correct":0},
            {**original[2],"future_correct":1},
            {**original[3],"future_correct":1},
        ]
        result=n.paired(original,alternative)
        self.assertEqual(result["cases"],2)
        self.assertEqual(result["unique_public_inputs"],2)
        self.assertEqual(result["paired_gains"],1)
        self.assertEqual(result["paired_losses"],1)
        self.assertEqual(result["net"],0)
        self.assertEqual(n.paired(original,alternative,False)["unique_public_inputs"],3)

    def test_guard_fresh_source_budget(self):
        with self.assertRaises(ValueError):
            n.source_disjoint_new({},reps=5)
        with self.assertRaises(ValueError):
            n.source_disjoint_new({},nonlate=51)
        with self.assertRaises(ValueError):
            n.source_disjoint_new({},late=11)

if __name__=="__main__":
    unittest.main()
