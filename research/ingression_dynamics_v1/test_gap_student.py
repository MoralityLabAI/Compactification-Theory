"""QO-16 exact teacher, frozen contract, BPTT and source-replay tests."""
import unittest
from fractions import Fraction as F

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES, VALIDATION_RATES, TEST_RATES,
    TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS,
    make_ledger, evidence_features,
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES as QO15_RATES,
    FROZEN_SUPERVISED_SEEDS as QO15_SEEDS,
    FROZEN_POLICY_SEEDS as QO15_POLICY_SEEDS,
)
from research.ingression_dynamics_v1.gap_student_frozen_test import (
    FROZEN_RATES, FROZEN_SUPERVISED_SEEDS, FROZEN_POLICY_SEEDS,
    FROZEN_POLICY_CHECKPOINTS, frozen_contract_sha256,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    gap_action_values, assert_qo12_reference_parity, gain_to_target,
    target_to_gain, greedy_gain_cell,
)
from research.ingression_dynamics_v1.gap_student_data import (
    generate_source, train_validation, swapped, example_from_ledger,
)
from research.ingression_dynamics_v1.gap_student_pilot import (
    ARMS, run_gap_policy, metric_for_student,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_linear, create_rnn, fit, trainable_count,
    inference_multiplies, forward,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    verify_common_stream_prefix,
)


class GapStudentTests(unittest.TestCase):
    def test_frozen_contract_separate_from_all_previous_splits(self):
        self.assertEqual(len(FROZEN_RATES),8)
        self.assertEqual(len(set(FROZEN_RATES)),8)
        self.assertFalse(set(FROZEN_RATES) & (
            set(TRAIN_RATES) | set(VALIDATION_RATES) |
            set(TEST_RATES) | set(QO15_RATES)))
        previous=(set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) |
                  set(TEST_SEEDS) | set(QO15_SEEDS) |
                  set(QO15_POLICY_SEEDS))
        self.assertFalse(set(FROZEN_SUPERVISED_SEEDS) & previous)
        self.assertFalse(set(FROZEN_POLICY_SEEDS) & previous)
        self.assertFalse(set(FROZEN_SUPERVISED_SEEDS) &
                         set(FROZEN_POLICY_SEEDS))
        self.assertEqual(FROZEN_POLICY_CHECKPOINTS,(32,80,160))
        self.assertEqual(len(frozen_contract_sha256()),64)
        self.assertEqual(ARMS,(
            "bottleneck","exact_qo12_gap_h2",
            "linear_gap_student","recurrent_gap_student"))

    def test_exact_per_action_progress_equals_independent_qo12_oracle(self):
        ledgers=[
            ProbeLedger(2,1),
            make_ledger(18,0,18,18),
            make_ledger(32,2,32,30),
            make_ledger(20,6,20,10),
            make_ledger(24,12,24,10),
        ]
        for ledger in ledgers:
            q,stats,oracle=assert_qo12_reference_parity(ledger)
            self.assertEqual(len(q),2)
            self.assertTrue(all(F(0)<=v<=F(1,2) for v in q))
            self.assertTrue(all(v<=F(stats["current_gap"]) for v in q))
            self.assertGreater(stats["gap_evaluations"],0)
            self.assertGreaterEqual(stats["candidate_action_evaluations"],0)
            if oracle["first_cell"] is not None and max(q)>0:
                self.assertEqual(q[oracle["first_cell"][0]],max(q))

    def test_public_feature_labels_and_monotone_transform(self):
        samples=[F(0),F(1,10000),F(1,300),F(1,100),F(1,10),F(1,2)]
        mapped=[gain_to_target(q) for q in samples]
        self.assertEqual(mapped,sorted(mapped))
        self.assertEqual(mapped[0],0)
        self.assertTrue(all(target_to_gain(p)==q for q,p in zip(samples,mapped)))
        ledger=make_ledger(18,2,18,16)
        raw,_=gap_action_values(ledger)
        ex=example_from_ledger(ledger,"test",0,3,"balanced",36)
        self.assertEqual(ex["raw_q"],raw)
        self.assertEqual(ex["tokens"],evidence_features(ledger))
        self.assertEqual(ex["q"],tuple(gain_to_target(q) for q in raw))
        self.assertEqual(ex["teacher_action"],0 if raw[0]>=raw[1] else 1)
        self.assertEqual(ex["q"],swapped(swapped(ex))["q"])
        self.assertEqual(ex["raw_q"],swapped(swapped(ex))["raw_q"])

    def test_source_split_provenance_and_real_rnn_training(self):
        validation,failures=generate_source("validation")
        self.assertEqual(len(validation),3*2*3*4)
        self.assertFalse(failures)
        self.assertTrue(all(x["split"]=="validation" for x in validation))
        self.assertTrue(all(x["seed"] in VALIDATION_SEEDS for x in validation))
        self.assertTrue(all(0<=q<1 for x in validation for q in x["q"]))
        subset=validation[:24]
        for make,params in ((create_linear,26),(create_rnn,138)):
            model=make(22)
            result=fit(model,subset,subset,seed=1,epochs=3)
            self.assertEqual(result["parameters"],params)
            self.assertEqual(result["update_steps"],72)
            self.assertNotEqual(model,result["weights"])
            self.assertTrue(all(0<p<1 for p in forward(
                result["weights"],subset[0]["tokens"])))
        self.assertEqual(inference_multiplies(create_linear()),24)
        self.assertEqual(inference_multiplies(create_rnn()),240)

    def test_greedy_tie_break_uses_public_width_and_counts(self):
        ledger=make_ledger(17,2,19,17)
        zero=greedy_gain_cell(ledger,(F(0),F(0)))
        self.assertIn(zero,(0,1))
        lohi=[ledger.bounds[i][0] for i in range(2)]
        expected=min((0,1),key=lambda i:(
            -(lohi[i][1]-lohi[i][0]),ledger.counts[i][0],i))
        self.assertEqual(zero,expected)

    def test_fixed_query_budget_common_stream_and_unmodified_certificate(self):
        from research.ingression_dynamics_v1.gap_student_frozen_test import (
            FROZEN_SCENARIOS,
        )
        case=FROZEN_SCENARIOS[0]
        models={
            "linear_gap_student":create_linear(),
            "recurrent_gap_student":create_rnn(),
        }
        for arm in ARMS[1:]:
            snapshots,history,cost=run_gap_policy(
                0,case,2503,arm,models,(0,8,18))
            self.assertEqual(len(history),18)
            self.assertEqual([x["queries"] for x in snapshots],[0,8,18])
            self.assertTrue(verify_common_stream_prefix(
                0,case,2503,history))
            self.assertEqual(sum(cost[k] for k in (
                "readiness_fallbacks","already_certified_fallbacks",
                "low_gain_fallbacks","model_chosen_actions")),18)
            self.assertEqual(
                (snapshots,history,cost),
                run_gap_policy(0,case,2503,arm,models,(0,8,18)))

    def test_mean_student_regret_satisfies_MOR13_normalized_inequality(self):
        exs=[]
        for n0,s0,n1,s1 in (
                (16,1,16,15),(24,4,24,22),(18,9,18,8)):
            ledger=make_ledger(n0,s0,n1,s1)
            exs.append(example_from_ledger(
                ledger,"synthetic_test",0,77,"balanced",n0+n1))
        for m in (create_linear(3),create_rnn(3)):
            scores=metric_for_student(m,exs)
            self.assertEqual(scores["n"],3)
            self.assertLessEqual(scores["informative"],3)
            self.assertGreaterEqual(scores["mean_normalized_margin_error"],0)
            self.assertGreaterEqual(scores["unconditional_exact_progress_regret"],0)


if __name__=="__main__":
    unittest.main()
