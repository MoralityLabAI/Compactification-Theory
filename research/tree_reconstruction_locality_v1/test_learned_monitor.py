"""KSR-01B: newly trained compact synthetic tree monitor regression checks."""
from __future__ import annotations

from fractions import Fraction as Q
import unittest

from learned_monitor import (
    B, CELLS, CONDITIONS, DATA_SEED, LAM, Q as QSTATES, SEEDS,
    STATUS, data_hash, dataset, draw_tree, features, measure,
    model_hash, plurality, predict, run, softmax, train,
)


class LearnedCountTreeTests(unittest.TestCase):
    def test_channel_and_split_generation_determinism(self):
        for i,(_,depth,r) in enumerate(CELLS):
            a=dataset(DATA_SEED+i*1011,128,depth,r)
            same=dataset(DATA_SEED+i*1011,128,depth,r)
            heldout=dataset(DATA_SEED+1+i*1011,128,depth,r)
            self.assertEqual(a,same)
            self.assertNotEqual(a,heldout)
            self.assertNotEqual(data_hash(a),data_hash(heldout))
            self.assertTrue(all(
                row["root"] in range(QSTATES) and
                len(row["counts"])==QSTATES and
                sum(row["counts"])<=B**depth for row in a))

    def test_features_have_declared_parameter_shape(self):
        x=features([1,2,0],4)
        self.assertEqual(len(x),QSTATES+2)
        self.assertEqual(x, (0.25,0.5,0.0,0.75,1.0))
        self.assertEqual(sum(softmax([1.2,-0.3,0.7])),1.0)
        with self.assertRaises(ValueError):
            features([2,3,1],4)

    def test_zero_weight_predictor_has_first_label_tie_break(self):
        W=[[0.]*(QSTATES+2) for _ in range(QSTATES)]
        value,probs=predict(W,[3,0,1],4)
        self.assertEqual(value,0)
        self.assertEqual(probs,[1/3]*3)
        self.assertEqual(len(model_hash(W)),64)

    def test_fixed_tree_plurality_uses_only_retained_counts(self):
        rows=[{"root":0,"counts":[3,0,0]},
              {"root":1,"counts":[0,3,0]},
              {"root":2,"counts":[0,0,3]},
              {"root":2,"counts":[0,0,0]}]
        r=plurality(rows)
        self.assertEqual(r["correct"],3)
        self.assertEqual(r["accuracy"],0.75)

    def test_train_and_control_have_distinct_reproducible_weights(self):
        rows=dataset(DATA_SEED,256,3,Q(1))
        W1,a=train(rows,leaves=8,seed=SEEDS[0],
                   condition=CONDITIONS[0],epochs=2)
        W2,b=train(rows,leaves=8,seed=SEEDS[0],
                   condition=CONDITIONS[0],epochs=2)
        W3,c=train(rows,leaves=8,seed=SEEDS[0],
                   condition=CONDITIONS[1],epochs=2)
        self.assertEqual(a["model_sha256"],b["model_sha256"])
        self.assertNotEqual(a["model_sha256"],c["model_sha256"])
        self.assertEqual(a["num_parameters"],15)
        self.assertEqual(a["gradient_steps"],512)
        self.assertEqual(W1,W2)

    def test_small_full_pilot_contract_and_source_boundaries(self):
        r=run(train_count=128,test_count=128,seeds=(SEEDS[0],),
              conditions=CONDITIONS,epochs=1)
        self.assertEqual(r["status"],STATUS)
        self.assertFalse(r["weights_pretrained"])
        self.assertFalse(r["reads_raw_tree_internal_states_at_test"])
        self.assertEqual(len(r["cases"]),3)
        self.assertEqual(len(r["source_sha256"]),64)
        for cell in r["cases"]:
            self.assertEqual(cell["independent_tree_train_count"],128)
            self.assertEqual(cell["independent_tree_test_count"],128)
            self.assertEqual(len(cell["learned_arms"]),2)
            self.assertTrue(0<=cell["fixed_plurality"]["accuracy"]<=1)
            self.assertEqual(
                cell["counter_memory"]["raw_alphabet_symbols"],cell["leaves"])
            self.assertEqual(
                cell["learned_arms"][0]["train"]["num_parameters"],15)
            self.assertNotEqual(cell["train_data_sha256"],
                                cell["test_data_sha256"])

    def test_supervised_signal_positive_vs_scrambled_training(self):
        training=dataset(DATA_SEED,2048,3,Q(1))
        test=dataset(DATA_SEED+1,2048,3,Q(1))
        model,receipt=train(training,leaves=8,seed=SEEDS[0],
                            condition="supervised",epochs=4)
        control,_=train(training,leaves=8,seed=SEEDS[0],
                        condition="random_legal_labels",epochs=4)
        a=measure(model,test,8)
        b=measure(control,test,8)
        self.assertGreater(a["accuracy"],0.70)
        self.assertGreater(a["accuracy"],b["accuracy"]+0.1)
        self.assertGreaterEqual(a["n"],2048)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run(train_count=1)
        with self.assertRaises(ValueError):
            train([],leaves=8,seed=11,condition="supervised")
        with self.assertRaises(ValueError):
            dataset(1,0,2,Q(1,2))


if __name__=="__main__":
    unittest.main()
