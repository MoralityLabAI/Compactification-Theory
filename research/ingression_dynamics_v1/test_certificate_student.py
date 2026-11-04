"""QO-14 exact labels, numerical BPTT gradients and source split tests."""
import math
import unittest
from copy import deepcopy
from fractions import Fraction as Q

from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES,VALIDATION_RATES,TEST_RATES,
    TRAIN_SEEDS,VALIDATION_SEEDS,TEST_SEEDS,
    make_ledger,evidence_features,teacher_q_values,generate_split
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_linear,create_rnn,gradients,forward,loss,
    fit,from_receipt,trainable_count,inference_multiplies
)
from research.ingression_dynamics_v1.certificate_reachability import (
    optimal_certificate_reachability
)
from research.ingression_dynamics_v1.certificate_student_pilot import (
    EVAL_SCENARIOS,POLICIES,CHECKPOINTS,POLICY_SEEDS,
    student_choice,run_student_policy
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    verify_common_stream_prefix
)


class CertificateStudentTests(unittest.TestCase):
    def teacher_witness(self):
        return make_ledger(18,0,18,18)

    def test_exact_q_action_values_match_original_qo13_teacher(self):
        ledger=self.teacher_witness()
        q=teacher_q_values(ledger)
        orig=optimal_certificate_reachability(ledger,1,Q(1,5),2)
        self.assertEqual(max(q),orig["predicted_certificate_probability"])
        self.assertEqual(max(q),Q(19,21))
        self.assertEqual(q[0],q[1])
        self.assertEqual(len(evidence_features(ledger)),2)

    def test_predeclared_rate_family_and_seed_disjointness(self):
        self.assertFalse(set(TRAIN_RATES)&set(TEST_RATES))
        self.assertFalse(set(VALIDATION_RATES)&set(TEST_RATES))
        self.assertFalse(set(TRAIN_RATES)&set(VALIDATION_RATES))
        self.assertFalse(set(TRAIN_SEEDS)&set(VALIDATION_SEEDS))
        self.assertFalse(set(TEST_SEEDS)&set(TRAIN_SEEDS))
        self.assertEqual(len(EVAL_SCENARIOS),len(TEST_RATES))
        self.assertEqual(POLICIES,(
            "bottleneck","linear_student","recurrent_student","exact_h2_teacher"))
        self.assertEqual(CHECKPOINTS,(32,56,80))
        self.assertEqual(POLICY_SEEDS,(1301,1307,1319))

    def test_trainable_parameter_counts_and_inference_macs(self):
        linear=create_linear()
        recurrent=create_rnn()
        self.assertEqual(trainable_count(linear),26)
        self.assertEqual(trainable_count(recurrent),138)
        self.assertEqual(inference_multiplies(linear),24)
        self.assertEqual(inference_multiplies(recurrent),240)
        tokens=evidence_features(self.teacher_witness())
        for model in (linear,recurrent):
            self.assertEqual(forward(model,tokens),forward(model,tokens))
            self.assertTrue(all(0<p<1 for p in forward(model,tokens)))
            self.assertEqual(forward(from_receipt(model),tokens),
                             forward(model,tokens))

    def test_analytic_bptt_and_linear_gradients_vs_central_difference(self):
        ledger=self.teacher_witness()
        ex={"tokens":evidence_features(ledger),"q":teacher_q_values(ledger)}
        for model in (create_linear(123),create_rnn(123)):
            analytic=gradients(model,ex)
            for key in (("w" if model["kind"]=="linear" else "wx"),
                        ("b" if model["kind"]=="linear" else "wh"),
                        ("w" if model["kind"]=="linear" else "wo")):
                arr=model[key]
                is_matrix=isinstance(arr[0],list)
                i=0
                j=0
                old=arr[i][j] if is_matrix else arr[j]
                if is_matrix:arr[i][j]=old+1e-5
                else:arr[j]=old+1e-5
                plus=loss(model,(ex,))
                if is_matrix:arr[i][j]=old-1e-5
                else:arr[j]=old-1e-5
                minus=loss(model,(ex,))
                if is_matrix:arr[i][j]=old
                else:arr[j]=old
                numeric=(plus-minus)/(2e-5)
                exact=analytic[key][i][j] if is_matrix else analytic[key][j]
                self.assertAlmostEqual(numeric,exact,delta=1e-6)

    def test_bptt_training_updates_recurrent_weights(self):
        ledger=self.teacher_witness()
        ex={"tokens":evidence_features(ledger),"q":teacher_q_values(ledger)}
        data=(ex,ex)
        initial=create_rnn(321)
        trained=fit(initial,data,data,seed=13,epochs=8,rate=0.02)
        self.assertEqual(trained["parameters"],138)
        self.assertEqual(trained["update_steps"],16)
        self.assertNotEqual(initial["wx"],trained["weights"]["wx"])
        self.assertNotEqual(initial["wh"],trained["weights"]["wh"])
        self.assertLessEqual(trained["best_validation_loss"],loss(initial,data))
        self.assertEqual(initial,create_rnn(321))

    def test_small_train_split_grounded_in_evidence_only(self):
        data,failures=generate_split("validation")
        self.assertGreater(len(data),10)
        self.assertTrue(all(ex["split"]=="validation" for ex in data))
        for ex in data[:10]:
            self.assertEqual(len(ex["tokens"]),2)
            self.assertTrue(all(len(v)==6 for v in ex["tokens"]))
            self.assertTrue(all(0<=q<=1 for q in ex["q"]))
            self.assertTrue(all(0<=feature<=1 for token in ex["tokens"]
                                for feature in token[:5]))

    def test_student_policy_stream_replay_at_equal_budget(self):
        case=EVAL_SCENARIOS[0]
        models={"linear_student":create_linear(),
                "recurrent_student":create_rnn()}
        for policy in POLICIES[1:]:
            rows,receipt,cost=run_student_policy(
                0,case,1301,policy,models,(0,8,18))
            self.assertEqual(len(receipt),18)
            self.assertTrue(verify_common_stream_prefix(0,case,1301,receipt))
            self.assertEqual([r["queries"] for r in rows],[0,8,18])
            self.assertEqual(sum(cost[k] for k in (
                "readiness_fallback","already_certified_fallback",
                "low_value_fallback","chosen_actions")),18)
            self.assertTrue(all(not r["false_cert"] or
                                not r["true_rates_covered"] for r in rows))
            self.assertEqual(
                (rows,receipt,cost),
                run_student_policy(0,case,1301,policy,models,(0,8,18)))

if __name__=="__main__":
    unittest.main()
