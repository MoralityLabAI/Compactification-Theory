"""SMC-04 exact tests; all inference sees only observed pairs, not secrets."""
import random
import unittest
from fractions import Fraction
from identify_transport import (
    uniform_catalog,hidden_fixture,predict,calibration_cohort,
    observed_calibrations,version_space,maximum_consensus,
    hypothesis_agreement_histogram,expected_spurious,
    noisy_ambiguity_upper,compatibility_accuracy,compose,
    paired_campaign,negative_calibration_control,
)
from transport_lift import relabel


class UnknownFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=uniform_catalog()
        cls.truth=hidden_fixture(cls.c,43)
        cls.train,cls.test=calibration_cohort(cls.c,cls.truth)

    def test_six_vertex_catalog_and_class_disjoint_balancing(self):
        c=self.c
        self.assertEqual(len(c['perms']),720)
        self.assertEqual(len(c['matchings']),15)
        self.assertEqual((len(self.train),len(self.test)),(576,144))
        self.assertEqual(set(self.train)&set(self.test),set())
        self.assertNotEqual(self.truth[1],c['ref'])
        for g in range(15):
            cohort=self.train if g<12 else self.test
            self.assertEqual(sum(predict(c,self.truth,i)==g for i in cohort),48)

    def test_one_pair_eliminates_reference_to_720_frame_candidates(self):
        pairs=observed_calibrations(self.c,self.truth,self.train[:1])
        h=version_space(self.c,pairs)
        self.assertEqual(len(h),720)
        self.assertIn(self.truth,h)
        self.assertEqual(len(set(sigma for sigma,_ in h)),720)

    def test_five_pairs_exactly_recover_all_unseen_matching_classes(self):
        ids=random.Random(8088).sample(self.train,5)
        pairs=observed_calibrations(self.c,self.truth,ids)
        self.assertEqual(version_space(self.c,pairs),(self.truth,))
        test=compatibility_accuracy(self.c,self.truth,self.truth,self.test,range(12,15))
        self.assertEqual(test['balanced_accuracy'],Fraction(5,6))
        self.assertEqual(test['tpr'],1)
        self.assertEqual(test['fpr'],Fraction(1,3))

    def test_exact_histograms_and_survival_expectations(self):
        full=hypothesis_agreement_histogram(self.c,self.truth)
        train=hypothesis_agreement_histogram(self.c,self.truth,self.train)
        self.assertEqual(full['max_false_agreement'],Fraction(7,15))
        self.assertEqual(train['max_false_agreement'],Fraction(1,2))
        self.assertEqual(full['num_hypotheses'],10800)
        self.assertEqual(train['counts'][576],1)
        self.assertEqual(expected_spurious(train,1),719)
        self.assertEqual(expected_spurious(train,2),Fraction(572,9))
        self.assertLess(expected_spurious(train,12),Fraction(3,1000))

    def test_noisy_chernoff_and_corrupted_maximum_consensus(self):
        hist=hypothesis_agreement_histogram(self.c,self.truth,self.train)
        for eps in (Fraction(0),Fraction(1,10),Fraction(1,4)):
            self.assertGreaterEqual(noisy_ambiguity_upper(hist,8,eps),
                                    noisy_ambiguity_upper(hist,24,eps))
        self.assertLess(noisy_ambiguity_upper(hist,24,Fraction(1,4)),0.006)
        ids=random.Random(1017).sample(self.train,24)
        obs=list(observed_calibrations(self.c,self.truth,ids))
        for pos in (0,2,6,9):
            pid,g=obs[pos]
            obs[pos]=(pid,(g+3)%15)
        recovered=maximum_consensus(self.c,obs)
        self.assertEqual(recovered['best_score'],20)
        self.assertEqual(recovered['maximizers'],(self.truth,))

    def test_common_output_relabel_gives_identical_version_space(self):
        c=self.c
        obs=observed_calibrations(c,self.truth,
                                  random.Random(8088).sample(self.train,5))
        tau=(2,1,0,3,4,5)
        table={i:c['mindex'][relabel(tau,c['matchings'][i])]
               for i in range(15)}
        changed=tuple((p,table[g]) for p,g in obs)
        expected=(c['pindex'][compose(tau,c['perms'][self.truth[0]])],
                  self.truth[1])
        self.assertEqual(version_space(c,changed),(expected,))
        origin=version_space(c,obs[:3])
        relabeled=version_space(c,changed[:3])
        image={(c['pindex'][compose(tau,c['perms'][s])],b) for s,b in origin}
        self.assertEqual(set(relabeled),image)

    def test_noisy_common_relabel_preserves_maximizer_set(self):
        c=self.c
        obs=list(observed_calibrations(c,self.truth,
                           random.Random(10010).sample(self.train,12)))
        for j in (0,7):
            p,g=obs[j]; obs[j]=(p,(g+5)%15)
        tau=(2,1,0,3,4,5)
        table={g:c['mindex'][relabel(tau,c['matchings'][g])] for g in range(15)}
        transformed=[(p,table[g]) for p,g in obs]
        original=maximum_consensus(c,obs)
        later=maximum_consensus(c,transformed)
        expected={(c['pindex'][compose(tau,c['perms'][s])],b)
                  for s,b in original['maximizers']}
        self.assertEqual(original['best_score'],later['best_score'])
        self.assertEqual(set(later['maximizers']),expected)

    def test_paired_seeded_calibration_replays_deterministically(self):
        a=paired_campaign(seeds=(43,),prefixes=(1,4,8),trials_per_seed=3)
        b=paired_campaign(seeds=(43,),prefixes=(1,4,8),trials_per_seed=3)
        self.assertEqual(a,b)
        self.assertEqual(a['series'][0]['mean_candidates'],'720')

    def test_unpaired_graph_negative_control(self):
        null=negative_calibration_control(seeds=(43,),examples=12,
                                          trials_per_seed=5)
        self.assertEqual(null['trials'],5)
        self.assertLess(Fraction(null['mean_heldout_balanced_accuracy']),
                        Fraction(3,5))
        self.assertGreater(Fraction(null['mean_heldout_balanced_accuracy']),
                           Fraction(2,5))


if __name__=="__main__":
    unittest.main()
