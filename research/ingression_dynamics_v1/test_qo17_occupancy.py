"""QO-17 full finite-difference BPTT and independent-split integrity tests."""
import math
import unittest
from fractions import Fraction as F

from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES,VALIDATION_RATES,TEST_RATES,
    TRAIN_SEEDS,VALIDATION_SEEDS,TEST_SEEDS,
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES as QO15_RATES,
)
from research.ingression_dynamics_v1.gap_student_frozen_test import (
    FROZEN_RATES as QO16_RATES,
)
from research.ingression_dynamics_v1.qo17_frozen_contract import (
    FROZEN_RATES, FROZEN_SUPERVISED_SEEDS,FROZEN_POLICY_SEEDS,
    TRAIN_OCCUPANCY_SEEDS,VALIDATION_OCCUPANCY_SEEDS,
    LOSS_COEFFICIENTS,ARMS,POLICY_CHECKPOINTS,contract_sha256,
    FROZEN_SCENARIOS,
)
from research.ingression_dynamics_v1.qo17_occupancy_data import (
    OCC_SEEDS,OCC_CHECKPOINTS,generate_occupancy,training_mixture,
)
from research.ingression_dynamics_v1.qo17_margin_gate_loss import (
    LAMBDA_RANK,RANK_TEMPERATURE,LAMBDA_GATE,GATE_TEMPERATURE,
    augmented_loss_one,augmented_gradients,fit_augmented,extra_terms,
)
from research.ingression_dynamics_v1.gap_student_data import (
    generate_source,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_linear,create_rnn,fit,forward,trainable_count,_iterate_leaves,
)
from research.ingression_dynamics_v1.gap_student_pilot import run_gap_policy
from research.ingression_dynamics_v1.qo17_occupancy_pilot import (
    _mapped_policy,visited_calibration,_calibration_metrics,
    load_qo16_baselines,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    verify_common_stream_prefix,
)


class QO17Tests(unittest.TestCase):
    def test_first_committed_test_contract_is_disjoint_and_complete(self):
        self.assertEqual(len(FROZEN_RATES),8)
        self.assertFalse(set(FROZEN_RATES)&(
            set(TRAIN_RATES)|set(VALIDATION_RATES)|set(TEST_RATES)|
            set(QO15_RATES)|set(QO16_RATES)))
        self.assertEqual(OCC_SEEDS,{"train":TRAIN_OCCUPANCY_SEEDS,
                                    "validation":VALIDATION_OCCUPANCY_SEEDS})
        self.assertEqual(OCC_CHECKPOINTS,(16,24,32,40,56,72,96,128))
        self.assertEqual(POLICY_CHECKPOINTS,(32,80,160))
        self.assertEqual(len(ARMS),8)
        self.assertEqual(len(contract_sha256()),64)
        self.assertEqual(
            len(set(FROZEN_SUPERVISED_SEEDS)|
                set(FROZEN_POLICY_SEEDS)|
                set(TRAIN_OCCUPANCY_SEEDS)|
                set(VALIDATION_OCCUPANCY_SEEDS)),10)
        self.assertEqual(LOSS_COEFFICIENTS,{
            "q_mse":1.0,"rank":LAMBDA_RANK,
            "rank_temperature":RANK_TEMPERATURE,
            "gate_bce":LAMBDA_GATE,
            "gate_temperature":GATE_TEMPERATURE,
        })

    def test_margin_gate_loss_finite_differences_full_bptt(self):
        data,failures=generate_source("validation")
        self.assertFalse(failures)
        ex=next(x for x in data if x["q"][0]!=x["q"][1])
        self.assertGreater(abs(float(ex["q"][0]-ex["q"][1])),0)
        for model in (create_linear(137),create_rnn(137)):
            analytic=augmented_gradients(model,ex)
            classes=(
                (("w",0,1),("b",None,0))
                if model["kind"]=="linear" else
                (("wx",0,2),("wh",1,3),("wo",0,5),
                 ("bh",None,3),("bo",None,0))
            )
            for name,i,j in classes:
                target=model[name]
                before=target[j] if i is None else target[i][j]
                if i is None:target[j]=before+1e-5
                else:target[i][j]=before+1e-5
                plus=augmented_loss_one(model,ex)
                if i is None:target[j]=before-1e-5
                else:target[i][j]=before-1e-5
                minus=augmented_loss_one(model,ex)
                if i is None:target[j]=before
                else:target[i][j]=before
                numeric=(plus-minus)/(2e-5)
                actual=analytic[name][j] if i is None else analytic[name][i][j]
                self.assertAlmostEqual(actual,numeric,delta=3e-5,
                    msg=f"{model['kind']}/{name}/{i}/{j}")
            pred=forward(model,ex["tokens"])
            penalty,ds=extra_terms(pred,ex)
            self.assertTrue(math.isfinite(penalty))
            self.assertTrue(all(math.isfinite(d) for d in ds))

    def test_original_architecture_size_and_finite_recurrent_training(self):
        data,_=generate_source("validation")
        batch=data[:10]
        for create,num in ((create_linear,26),(create_rnn,138)):
            start=create(20261104)
            trained=fit_augmented(start,batch,batch,seed=7,epochs=3)
            self.assertEqual(trained["parameters"],num)
            self.assertEqual(trainable_count(trained["weights"]),num)
            self.assertNotEqual(start,trained["weights"])
            self.assertEqual(trained["update_steps"],30)
            self.assertTrue(1<=trained["best_epoch"]<=3)

    def test_training_family_occupancy_is_valid_and_policy_diverse(self):
        rows,meta=generate_occupancy("validation")
        self.assertTrue(rows)
        self.assertEqual(meta["split"],"validation")
        self.assertEqual(meta["rate_count"],len(VALIDATION_RATES))
        self.assertEqual(meta["seeds"],list(VALIDATION_OCCUPANCY_SEEDS))
        self.assertEqual(meta["occupancy_trajectory_count"],
                         3*2*2)
        self.assertEqual(meta["cost"]["invalid_source_streams"],0)
        self.assertEqual(meta["cost"]["snapshot_label_calls"],len(rows))
        self.assertEqual(set(x["source_policy"] for x in rows),
                         {"bottleneck","exact_gap_h2"})
        for ex in rows:
            self.assertIn(ex["seed"],VALIDATION_OCCUPANCY_SEEDS)
            self.assertEqual(ex["split"],"validation_occupancy")
            evidence=ex["public_state_receipt"]
            self.assertEqual(sum(evidence["counts"]),ex["time"])
            self.assertEqual(len(evidence["history_sha256"]),64)
            self.assertTrue(all(F(0)<=q<=F(1) for q in ex["q"]))
            self.assertTrue(all(F(0)<=q<=F(1,2) for q in ex["raw_q"]))

    def test_mix_includes_fixed_schedule_and_visited_training_sources(self):
        rows,meta=training_mixture("validation")
        self.assertEqual(meta["total_training_rows"],len(rows))
        self.assertGreater(len(rows),144)
        self.assertEqual(meta["fixed_source"]["used_examples_with_swap_augmentation"],144)
        self.assertEqual(meta["occupancy"]["occupancy_eligible_examples"],
                         len(meta["occupancy_examples"]))
        self.assertTrue(all(e["seed"] in VALIDATION_OCCUPANCY_SEEDS
                            for e in rows if e["split"]=="validation_occupancy"))

    def test_qo16_reference_models_remain_frozen_and_policy_replays(self):
        models,source_hash=load_qo16_baselines()
        self.assertEqual(len(source_hash),64)
        self.assertEqual(trainable_count(models["qo16_linear_fixed_mse"]),26)
        self.assertEqual(trainable_count(models["qo16_recurrent_fixed_mse"]),138)
        case=FROZEN_SCENARIOS[0]
        for arm in ("qo16_linear_fixed_mse","qo16_recurrent_fixed_mse"):
            mapped,selected=_mapped_policy(arm,models[arm])
            self.assertIn(mapped,("linear_gap_student","recurrent_gap_student"))
            rows,receipts,cost=run_gap_policy(
                0,case,FROZEN_POLICY_SEEDS[0],mapped,selected,(0,8,24))
            self.assertEqual([r["queries"] for r in rows],[0,8,24])
            self.assertTrue(verify_common_stream_prefix(
                0,case,FROZEN_POLICY_SEEDS[0],receipts))
            samples=visited_calibration(
                receipts,models[arm],case,FROZEN_POLICY_SEEDS[0],arm,(8,16,24))
            stats=_calibration_metrics(samples)
            self.assertEqual(stats["eligible_visited_samples"],len(samples))
            self.assertEqual(sum(stats[k] for k in (
                "true_positive","false_positive",
                "true_negative","false_negative")),len(samples))


if __name__=="__main__":
    unittest.main()
