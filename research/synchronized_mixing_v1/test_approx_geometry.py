"""SMC-06 structural tests use only stdlib and archived SAE feature scores."""
import json
import random
import unittest
from itertools import combinations
from pathlib import Path
from transport_lift import all_matchings,relabel
from approx_geometry import (infer_model_shape,bounded_cover,SearchBudgetExhausted,
                              recover_with_one_swap,top_cardinality,
                              model_matching_states,decode_soft)
from identify_transport import uniform_catalog,hidden_fixture,predict,calibration_cohort,version_space

HERE=Path(__file__).parent

class SAEGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot=json.loads((HERE/'sae_activation_evidence.json').read_text())
        cls.scores=cls.snapshot['activation_scores_15_by_15']

    def test_model_dimension_and_group_inferred_from_sparse_counts(self):
        s=infer_model_shape(15,3)
        self.assertEqual((s['n_vertices'],s['automorphism_group_order']),(6,720))
        with self.assertRaises(ValueError):infer_model_shape(14,3)
        with self.assertRaises(ValueError):infer_model_shape(15,4)

    def test_every_455_three_class_holdout_certifies_with_bounded_search(self):
        edges=tuple(combinations(range(6),2))
        states=tuple(all_matchings(range(6)))
        masks=tuple(tuple(int(e in g) for e in edges) for g in states)
        for held in combinations(range(15),3):
            geom=bounded_cover([r for j,r in enumerate(masks) if j not in held])
            self.assertEqual(geom['n_automorphisms'],720)
            self.assertLessEqual(geom['search_nodes'],531)

    def test_frozen_trained_sae_one_swap_repair_is_unique(self):
        scores=self.scores['101'][:12]
        with self.assertRaises(ValueError):bounded_cover(top_cardinality(scores))
        repaired=recover_with_one_swap(scores)
        self.assertEqual(repaired['repairs'],((8,14,8),))
        self.assertEqual(repaired['geometry']['n_candidate_stars'],47)
        self.assertEqual(len(model_matching_states(repaired['geometry'])),15)

    def test_frozen_trained_sae_direct_geometry(self):
        rec=recover_with_one_swap(self.scores['103'][:12])
        self.assertEqual(rec['repairs'],())
        self.assertEqual(rec['geometry']['group_order_derived'],720)
        self.assertEqual(rec['candidate_repair_count'],0)

    def test_frozen_sae_failure_abstains_without_repair_sprawl(self):
        with self.assertRaisesRegex(ValueError,'unique geometry'):
            recover_with_one_swap(self.scores['105'][:12])

    def test_train_only_geometry_calibrates_three_hidden_frames(self):
        cat=uniform_catalog()
        for seed,expected in (('101','13/18'),('103','5/6')):
            raw=self.scores[seed]
            geom=recover_with_one_swap(raw[:12])['geometry']
            abstract=[cat['mindex'][decode_soft(geom,score)[0]] for score in raw]
            for frame_seed in (43,77,91):
                truth=hidden_fixture(cat,frame_seed)
                train_ids,test_ids=calibration_cohort(cat,truth)
                inputs=[random.Random(frame_seed+29+991).choice(train_ids)]
                rng=random.Random(frame_seed+29+991)
                inputs=[rng.choice(train_ids) for _ in range(12)]
                data=tuple((i,abstract[predict(cat,truth,i)]) for i in inputs)
                version=version_space(cat,data)
                self.assertEqual(len(version),1)
                predicted=tuple(predict(cat,version[0],i) for i in test_ids)
                actual=tuple(abstract[predict(cat,truth,i)] for i in test_ids)
                tpr=sum(a==b for a,b in zip(predicted,actual))/len(test_ids)
                fpr=sum(a==abstract[g] for a in predicted for g in (12,13,14))/(3*len(test_ids))
                self.assertEqual(str(__import__('fractions').Fraction(1+__import__('fractions').Fraction(sum(a==b for a,b in zip(predicted,actual)),len(test_ids))-__import__('fractions').Fraction(sum(a==abstract[g] for a in predicted for g in (12,13,14)),3*len(test_ids)))/2),expected)

    def test_common_feature_gauge_scramble_preserves_fitted_incidence(self):
        data=self.scores['103'][:12]
        direct=recover_with_one_swap(data)
        for seed in (4,19,113):
            p=list(range(15));random.Random(seed).shuffle(p)
            mixed=[tuple(row[p[i]] for i in range(15)) for row in data]
            other=recover_with_one_swap(mixed)
            expected={frozenset(p.index(j) for j in star)
                      for star in direct['geometry']['vertex_stars']}
            self.assertEqual({frozenset(a) for a in other['geometry']['vertex_stars']},expected)

    def test_soft_matching_decoder_margin_is_positive(self):
        rec=recover_with_one_swap(self.scores['103'][:12])
        for row in self.scores['103'][12:]:
            candidate,margin=decode_soft(rec['geometry'],row)
            self.assertGreater(margin,1)
            self.assertEqual(len(candidate),3)

    def test_bounded_search_fails_closed(self):
        rows=top_cardinality(self.scores['103'][:12])
        with self.assertRaises(SearchBudgetExhausted):
            bounded_cover(rows,max_nodes=1)

if __name__=='__main__':unittest.main()
