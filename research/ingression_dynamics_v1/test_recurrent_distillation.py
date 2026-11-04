"""QO-14: teacher/student parity, split integrity and cheap paired smoke audit."""
import unittest
from fractions import Fraction as Q
import numpy as np

from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger
from research.ingression_dynamics_v1.recurrent_teacher_data import (
    dataset,audit_fixed_split,teacher_record,encode_cell_tokens,
    TRAIN_RATES,TEST_RATES,TRAIN_SEEDS,TEST_SEEDS
)
from research.ingression_dynamics_v1.recurrent_probe_student import (
    train_students,predict_q,evaluate_student_on_teacher
)
from research.ingression_dynamics_v1.recurrent_distillation_pilot import (
    rollout,build_report,ROLLOUT_RATE_CASES,ROLLOUT_SEEDS,POLICIES
)


def tiny_data(split):
    return dataset(split,
                   rate_pairs=(TRAIN_RATES[:2] if split=="train"
                               else TEST_RATES[:2]),
                   seeds=(TRAIN_SEEDS[:2] if split=="train" else TEST_SEEDS[:2]),
                   count_pairs=((14,16),(18,18),(24,20)))


class RecurrentProbeTests(unittest.TestCase):
    def test_fixed_split_has_no_rate_or_seed_leak(self):
        check=audit_fixed_split()
        self.assertEqual(check["train_expected_examples"],864)
        self.assertEqual(check["test_expected_examples"],378)
        self.assertFalse(set(TRAIN_RATES)&set(TEST_RATES))
        self.assertFalse(set(TRAIN_SEEDS)&set(TEST_SEEDS))

    def test_exact_bellman_first_action_values_and_evidence_only_features(self):
        ledger=ProbeLedger(2,1,Q(1,20),10)
        for _ in range(18):
            ledger.observe(0,0,0);ledger.observe(1,0,1)
        item=teacher_record(ledger)
        self.assertEqual(item["teacher_value"],Q(19,21))
        self.assertEqual(max(item["q"]),item["teacher_value"])
        self.assertEqual(len(item["tokens"]),2)
        self.assertEqual(len(item["tokens"][0]),10)
        self.assertEqual(item["counts"],(18,18))
        self.assertEqual(item["successes"],(0,18))
        self.assertEqual(item["tokens"],encode_cell_tokens(ledger))

    def test_frozen_reservoir_and_linear_train_reproducibly(self):
        train=tiny_data("train");test=tiny_data("test")
        self.assertEqual(len(train),12)
        self.assertEqual(len(test),12)
        a=train_students(train,hidden=12,seed=11)
        b=train_students(train,hidden=12,seed=11)
        self.assertEqual(a,b)
        self.assertEqual(a["hidden_size"],12)
        self.assertEqual(a["training_examples"],12)
        self.assertGreater(a["estimated_bytes_float32"]["recurrent"],0)
        for rows in (train,test):
            for row in rows:
                for model in ("linear","recurrent"):
                    q=predict_q(a,row["tokens"],model)
                    self.assertEqual(len(q),2)
                    self.assertTrue(all(0<=v<=1 for v in q))
        for model in ("linear","recurrent"):
            metrics=evaluate_student_on_teacher(test,a,model)
            self.assertEqual(metrics["examples"],12)
            self.assertGreaterEqual(metrics["q_rmse"],0)
            self.assertTrue(0<=metrics["mean_teacher_action_regret"]<=1)

    def test_student_rollout_is_baseline_while_unready(self):
        train=tiny_data("train")
        model=train_students(train,hidden=12,seed=11)
        rates=ROLLOUT_RATE_CASES[0]
        paths=[]
        for policy in POLICIES:
            result=rollout(policy,0,rates,809,12,model)
            paths.append(result["trace_sha256"])
            self.assertEqual(result["queries"],12)
            self.assertEqual(result["student_selected_probes"],0)
            self.assertFalse(result["false_cert"] and result["true_rate_coverage"])
            self.assertEqual(sum(map(sum,result["count_by_cell"])),12)
        self.assertEqual(len(set(paths)),1)

    def test_reduced_paired_audit_is_reproducible(self):
        train=tiny_data("train");test=tiny_data("test")
        data=build_report(train,test,
                          rollout_rates=ROLLOUT_RATE_CASES[:1],
                          rollout_seeds=ROLLOUT_SEEDS[:2],
                          checkpoints=(8,16))
        self.assertEqual(data["counts"],{
            "training_examples":12,"test_examples":12,
            "policy_trials":16,"rollout_queries":192,
        })
        self.assertEqual(len(data["groups"]),8)
        self.assertEqual(len(data["rows"]),16)
        self.assertEqual(len(data["test_student_metrics"]),2)
        self.assertEqual(data,build_report(train,test,
                          rollout_rates=ROLLOUT_RATE_CASES[:1],
                          rollout_seeds=ROLLOUT_SEEDS[:2],
                          checkpoints=(8,16)))

    def test_invalid_train_and_model_contracts(self):
        train=tiny_data("train")
        with self.assertRaises(ValueError):
            train_students(train,hidden=1)
        model=train_students(train,hidden=12,seed=11)
        with self.assertRaises(ValueError):
            predict_q(model,train[0]["tokens"],"unknown")
        with self.assertRaises(ValueError):
            dataset("neither")
        with self.assertRaises(ValueError):
            rollout("invalid",0,ROLLOUT_RATE_CASES[0],809,12,model)


if __name__=="__main__":
    unittest.main()
