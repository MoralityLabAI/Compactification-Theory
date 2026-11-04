"""QO-18 preregistration, DAgger source, selective gate and stream integrity."""
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
    FROZEN_RATES as QO17_RATES,
)
from research.ingression_dynamics_v1.qo18_frozen_contract import (
    FROZEN_RATES,FROZEN_SUPERVISED_SEEDS,FROZEN_POLICY_SEEDS,
    TRAIN_DAGGER_SEEDS,VALIDATION_DAGGER_SEEDS,
    CALIBRATION_SEEDS,POLICY_CHECKPOINTS,ARMS,
    GATE_CANDIDATES,TEACHER_GATE,contract_sha256,FROZEN_SCENARIOS,
)
from research.ingression_dynamics_v1.qo18_dagger_data import (
    DAGGER_SEEDS,CHECKPOINTS,generate_round,hard_negative_copies,
)
from research.ingression_dynamics_v1.qo18_gate_calibration import (
    CAL_SEEDS,THRESHOLDS,select_gate,confusion,
)
from research.ingression_dynamics_v1.qo18_dagger_pilot import (
    read_qo17_reference,run_student_policy,visited_audit,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_rnn,trainable_count,forward,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as run_bottleneck,verify_common_stream_prefix,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    assert_qo12_reference_parity,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    make_ledger,
)

class QO18Tests(unittest.TestCase):
    def test_frozen_split_separation_and_source_locked_knobs(self):
        self.assertEqual(len(FROZEN_RATES),len(set(FROZEN_RATES)))
        self.assertEqual(len(FROZEN_RATES),8)
        self.assertFalse(set(FROZEN_RATES)&(
            set(TRAIN_RATES)|set(VALIDATION_RATES)|set(TEST_RATES)|
            set(QO15_RATES)|set(QO16_RATES)|set(QO17_RATES)))
        self.assertEqual(DAGGER_SEEDS,{
            "train":TRAIN_DAGGER_SEEDS,
            "validation":VALIDATION_DAGGER_SEEDS})
        self.assertEqual(CAL_SEEDS,CALIBRATION_SEEDS)
        self.assertEqual(THRESHOLDS,GATE_CANDIDATES)
        self.assertEqual(POLICY_CHECKPOINTS,(32,80,160))
        self.assertEqual(TEACHER_GATE,F(1,20))
        self.assertEqual(len(ARMS),6)
        self.assertEqual(len(contract_sha256()),64)

    def test_dagger_validation_uses_only_validation_rates_and_own_seeds(self):
        model=create_rnn(seed=24)
        rows,meta=generate_round("validation",0,model)
        self.assertTrue(rows)
        self.assertEqual(meta["split"],"validation")
        self.assertEqual(meta["round"],1)
        self.assertEqual(meta["rate_count"],len(VALIDATION_RATES))
        self.assertEqual(meta["seeds"],list(DAGGER_SEEDS["validation"][0]))
        self.assertEqual(meta["eligible_snapshots"],len(rows))
        self.assertEqual(meta["costs"]["teacher_labels"],len(rows))
        self.assertEqual(meta["costs"]["model_inferences"]>0,True)
        self.assertEqual(len(meta["source_trajectory_hashes"]),6)
        for row in rows:
            self.assertEqual(row["split"],"validation_dagger")
            self.assertIn(row["seed"],VALIDATION_DAGGER_SEEDS[0])
            self.assertEqual(sum(row["public_state_receipt"]["counts"]),row["time"])
            self.assertEqual(len(row["public_state_receipt"]["history_sha256"]),64)
            self.assertEqual(row["round"],1)
            self.assertTrue(all(F(0)<=x<=F(1) for x in row["q"]))

    def test_train_hard_negative_sampling_is_bounded_and_stable(self):
        ex={
            "q":(F(0),F(1,100)),
            "rate_index":0,"seed":13,"time":24,
            "public_state_receipt":{"bounds":[["0","1/2"],["1/2","1"]]},
        }
        extras,stats=hard_negative_copies((ex,ex,ex))
        self.assertEqual(stats["selected_unique"],1)
        self.assertEqual(stats["extra_copies"],8)
        self.assertEqual(len(extras),8)
        self.assertEqual(stats["total_candidate_negatives"],3)
        none,count=hard_negative_copies(({**ex,"q":(F(1),F(1))},))
        self.assertEqual(none,())
        self.assertEqual(count["selected_unique"],0)

    def test_threshold_grid_prefers_recall_under_false_engagement_cap(self):
        src=[
            {"cluster":"c0","teacher_positive":True,"model_score":.40},
            {"cluster":"c1","teacher_positive":True,"model_score":.30},
            {"cluster":"c2","teacher_positive":True,"model_score":.24},
            {"cluster":"c3","teacher_positive":True,"model_score":.21},
            {"cluster":"c4","teacher_positive":True,"model_score":.20},
            {"cluster":"c5","teacher_positive":False,"model_score":.12},
            {"cluster":"c6","teacher_positive":False,"model_score":.08},
            {"cluster":"c7","teacher_positive":False,"model_score":.02},
        ]
        selection=select_gate(src)
        self.assertTrue(selection["feasible"])
        self.assertEqual(selection["threshold"],"3/20")
        cm=selection["chosen_validation_confusion"]
        self.assertEqual(cm["tp"],5)
        self.assertEqual(cm["fp"],0)
        self.assertEqual(cm["recall"],1)
        self.assertEqual(cm["negatives"],3)
        self.assertEqual(cm["observed_clusters"],8)

    def test_if_grid_is_infeasible_policy_abstains(self):
        rows=[{"cluster":"a","teacher_positive":True,"model_score":.01},
              {"cluster":"b","teacher_positive":False,"model_score":.99}]
        selected=select_gate(rows)
        self.assertFalse(selected["feasible"])
        self.assertIsNone(selected["threshold"])
        cm=selected["chosen_validation_confusion"]
        self.assertEqual((cm["tp"],cm["fp"],cm["tn"],cm["fn"]),(0,0,1,1))

    def test_calibrated_null_gate_is_identical_to_bottleneck(self):
        case=FROZEN_SCENARIOS[0]
        model=create_rnn(seed=18)
        observations,history,cost=run_student_policy(
            0,case,FROZEN_POLICY_SEEDS[0],
            "dagger_round2_calibrated",model,None,(0,8,24))
        base,base_history=run_bottleneck(
            0,case,FROZEN_POLICY_SEEDS[0],"bottleneck",(0,8,24))
        self.assertEqual(history,base_history)
        self.assertEqual(cost["student_inferences"],0)
        self.assertTrue(verify_common_stream_prefix(
            0,case,FROZEN_POLICY_SEEDS[0],history))
        self.assertEqual([x["queries"] for x in observations],[0,8,24])
        visited=visited_audit(history,model,"dagger_round2_calibrated",
                              case,FROZEN_POLICY_SEEDS[0],None,(8,16,24))
        self.assertTrue(all(row["predicted_engages"]==False for row in visited))

    def test_qo17_checkpoint_and_qo12_reference_unchanged(self):
        model,h=read_qo17_reference()
        self.assertEqual(trainable_count(model),138)
        self.assertEqual(len(h),64)
        _,_,old=assert_qo12_reference_parity(make_ledger(18,0,18,18))
        self.assertIsNotNone(old)
        self.assertEqual(len(forward(model,make_ledger(16,1,16,15)
                  and __import__("research.ingression_dynamics_v1.certificate_student_data",
                  fromlist=["evidence_features"]).evidence_features(
                  make_ledger(16,1,16,15)))),2)

if __name__=="__main__":
    unittest.main()
