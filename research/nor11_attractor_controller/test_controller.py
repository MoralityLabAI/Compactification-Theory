"""NOR-11 regression tests for trained auxiliary basin probe and escape control."""
import math
import unittest
from dataclasses import replace

from research.rrc_compactification import compound25 as b
from research.nor11_attractor_controller import controller as n


class AttractorAwareController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck,cls.p1,cls.p2=n.train_source()

    def test_checkpoint_and_distinct_certified_centers(self):
        self.assertEqual(self.ck["parameters"],1000)
        self.assertEqual(n.CHECKPOINT_SHA,"54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b")
        self.assertGreater(math.dist(self.p1,self.p2),5)

    def test_split_public_feature_disjointness(self):
        c=n.sample_cohorts({"train":2,"dev":2,"test":2})
        self.assertEqual([len(c[x]) for x in ("train","dev","test")],[6,6,12])
        train_dev=[d for split in ("train","dev") for _,_,d in c[split]]
        heldout=[d for _,_,d in c["test"]]
        self.assertEqual(len(train_dev),len(set(train_dev)))
        self.assertFalse(set(train_dev)&set(heldout))
        self.assertTrue(all(len(b.possible_headers(case))==1 for group in c.values() for case,_,_ in group))
        # Repeated early feature tensors are a substantive property of
        # late_source, so these are different cases but aliased inputs.
        late=[d for case,_,d in c["test"] if case.family=="late_source"]
        self.assertLess(len(set(late)),len(late))

    def test_hidden_head_input_is_independent_of_private_future(self):
        rows=n.sample_cohorts({"train":1,"dev":1,"test":1})["train"]
        case,header,digest=rows[0]
        h,parity=n.model_prefix_hidden(case,header,self.ck)
        mutant=replace(case,program=None,future=tuple(7777777 for _ in case.future))
        h2,parity2=n.model_prefix_hidden(mutant,header,self.ck)
        self.assertEqual(h,h2)
        self.assertEqual(parity,parity2)
        self.assertEqual(digest,n.public_digest(mutant,header))
        self.assertLess(parity,1e-11)

    def test_portable_native_four_check_verifier_parity(self):
        case,header,_=n.sample_cohorts({"train":1,"dev":1,"test":1})["train"][0]
        h,_=n.model_prefix_hidden(case,header,self.ck)
        original=b.search(case,"recurrent",self.ck,budget=4)
        reproduced=n.score_with_hidden(case,header,h,self.ck)
        self.assertEqual(original["full_checks"],reproduced["paid_checks"])
        self.assertEqual(bool(original["program"]),reproduced["has_verified_proposal"])
        self.assertEqual(b.grade(case,original["program"])["future_correct"],reproduced["future_correct"])

    def test_frozen_probe_has_learnable_two_mode_signal(self):
        synthetic=[]
        for i in range(12):
            h=[0.0]*12
            h[0]=(-1.0 if i<6 else 1.0)+i/100.0
            synthetic.append({"hidden":h,"mode":"first" if i<6 else "second",
                              "label":0 if i<6 else 1})
        head=n.fit_head(synthetic)
        first=n.head_score(synthetic[0]["hidden"],head)
        second=n.head_score(synthetic[-1]["hidden"],head)
        self.assertLess(first,.25)
        self.assertGreater(second,.75)
        self.assertEqual(len(head["weights"]),13)

    def test_steering_budget_and_valid_source_cube(self):
        h=[.4]*12
        shifted=n.intervene(h,.25,self.p1)
        self.assertEqual(len(shifted),12)
        self.assertTrue(all(-1<=v<=1 for v in shifted))
        self.assertLess(math.dist(shifted,self.p1),math.dist(h,self.p1))
        with self.assertRaises(ValueError):
            n.intervene(h,-.1,self.p1)

    def test_bounded_training_and_eval_receipt(self):
        # Full hosted run uses the larger frozen 32/12/12 split.
        result=n.run({"train":12,"dev":3,"test":3})
        self.assertEqual(result["cohorts"]["train"]["cases"],36)
        self.assertEqual(result["cohorts"]["dev"]["cases"],9)
        self.assertEqual(result["cohorts"]["test"]["cases"],18)
        self.assertLess(result["cohorts"]["test"]["distinct_public_features"],18)
        self.assertLess(result["cohorts"]["test"]["per_family_distinct_features"]["late_source"],3)
        self.assertEqual(result["head_diagnostics"]["test"]["trained"]["n"],18)
        self.assertLess(result["head_diagnostics"]["test"]["trained_unique_features"]["n"],18)
        self.assertEqual(result["source_checkpoint_sha256"],n.CHECKPOINT_SHA)
        self.assertEqual(result["trained_auxiliary_head_parameters"],13)
        self.assertFalse(result["trained_gru_weights_modified"])
        self.assertFalse(result["train_restrictions"]["native_future_truth_used_to_fit_head"])
        self.assertEqual(result["heldout_protocol"]["original"]["actions"],0)
        self.assertEqual(result["heldout_protocol"]["original"]["cases"],18)
        self.assertEqual(len(result["development_policy_grid"]),7)
        self.assertIn(result["selected_policy"]["alpha"],(0,.125,.25,.5))
        self.assertEqual(set(result["heldout_protocol"]),
                         {"original","selected","fixed_escape",
                          "unconditional","shuffled_head_escape","random_direction"})
        for arm in result["heldout_protocol"].values():
            self.assertLessEqual(arm["paid_checks_total"],arm["cases"]*4)
            self.assertEqual(arm["accepted"],arm["future_correct"]+arm["false_accept"])
            self.assertLessEqual(arm["actions"],arm["cases"])


if __name__=="__main__":
    unittest.main()
