"""FPB-01 heldout synthetic paired-Fourier probe benchmark tests."""
from __future__ import annotations
import unittest
from fractions import Fraction as Q
from benchmark import (
    ALL_PAIRS, QUADRATIC_PAIRS, RAW_SINGLETONS,
    SEED, SENTINEL_PAIR, TRAIN, TEST, draw, exact_population,
    estimate_coefficients, registered_methods, run, score_model,
    signed_feature,
)
from reference import HIDDEN_N, SENTINEL, hidden_fourier, lifted_sign


class FourierProbePilotTests(unittest.TestCase):
    def test_deterministic_independent_sample_cohorts(self):
        train=draw(SEED,256)
        test=draw(SEED+1,256)
        self.assertEqual(train,draw(SEED,256))
        self.assertNotEqual(train,test)
        self.assertEqual(len(train),256)
        self.assertTrue(all(0<=x<(1<<HIDDEN_N) and
                            0<=y<(1<<HIDDEN_N) for x,y in train))

    def test_registered_probe_costs(self):
        methods=registered_methods()
        self.assertEqual(len(methods["raw_34_singletons"]),34)
        self.assertEqual(len(methods["paired_all_17"]),17)
        self.assertEqual(len(methods["paired_sentinel"]),1)
        self.assertEqual(len(methods["paired_17_plus_block_fourth_order"]),41)
        self.assertEqual(len(QUADRATIC_PAIRS),24)

    def test_paired_feature_is_xor_syndrome(self):
        for x,y in draw(SEED+17,100):
            parity=signed_feature(x,y,SENTINEL_PAIR[0])
            expected=(-1 if ((x^y)&SENTINEL) else 1)
            self.assertEqual(parity,expected)

    def test_population_sentinel_probe_margin(self):
        x=exact_population()
        self.assertEqual(x["mean_sign"],str(Q(-1695,4096)))
        self.assertEqual(x["paired_sentinel"],str(Q(2401,4096)))
        self.assertEqual(x["constant_mode_accuracy"],str(Q(5791,8192)))
        self.assertEqual(x["paired_sentinel_accuracy"],str(Q(6497,8192)))
        self.assertEqual(x["paired_classification_accuracy_gain"],
                         str(Q(353,4096)))

    def test_calibration_never_uses_heldout_data(self):
        calibration=draw(SEED,1024)
        heldout=draw(SEED+1,1024)
        mu,coefs=estimate_coefficients(calibration,SENTINEL_PAIR)
        baseline=score_model(heldout,mu,coefs,SENTINEL_PAIR)
        _ = score_model(draw(SEED+2,1024),mu,coefs,SENTINEL_PAIR)
        self.assertEqual(baseline,score_model(heldout,mu,coefs,SENTINEL_PAIR))

    def test_paired_sentinel_outperforms_first_order_probe(self):
        report=run(seed=SEED,train_count=2048,test_count=2048)
        methods=report["methods"]
        paired=methods["paired_sentinel"]["heldout"]
        raw=methods["raw_34_singletons"]["heldout"]
        bias=methods["bias_only"]["heldout"]
        self.assertLess(paired["mse_float"],bias["mse_float"])
        self.assertLess(paired["mse_float"],raw["mse_float"])
        self.assertGreater(paired["classification_accuracy_float"],
                           bias["classification_accuracy_float"])
        self.assertGreater(paired["classification_accuracy_float"],
                           .75)

    def test_full_default_run_has_fourier_provenance(self):
        r=run()
        self.assertEqual(r["status"],"FROZEN_SYNTHETIC_PARITY_MODE")
        self.assertEqual(r["train_count"],TRAIN)
        self.assertEqual(r["test_count"],TEST)
        self.assertEqual(r["observed_input_bits"],34)
        self.assertFalse(r["evaluates_trained_neural_checkpoint"])
        self.assertFalse(r["observes_internal_chain_of_thought"])
        self.assertTrue(r["synthetic_hidden_mode"])
        self.assertEqual(len(r["reference_sha256"]),64)
        self.assertEqual(len(r["train_data_sha256"]),64)
        self.assertEqual(len(r["test_data_sha256"]),64)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            run(train_count=3)
        with self.assertRaises(ValueError):
            draw(SEED,0)
        with self.assertRaises(ValueError):
            score_model([(0,0)],Q(0),[Q(1)],())


if __name__=="__main__":
    unittest.main()
