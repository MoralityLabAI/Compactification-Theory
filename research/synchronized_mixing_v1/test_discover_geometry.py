"""SMC-05 exact geometry, withheld-cohort, gauge and noise tests (stdlib)."""
import itertools
import unittest
from discover_geometry import (
    infer_feature_geometry,decode_mask,exact_recovery_oracle,
    scramble_masks,majority_denoise,noisy_replicates,switching_null,
)
from run_discovery import (
    fixture,discover_from_unlabelled_train_states,simulate,
    geometry_noise_campaign,null_campaign,
)
from transport_lift import all_matchings


class AnonymousGeometryTests(unittest.TestCase):
    def test_sparse_partial_matching_reconstruction_up_to_gauge(self):
        masks,states,edges=exact_recovery_oracle()
        scrambled,perm=scramble_masks(masks,seed=1701)
        learned=infer_feature_geometry(scrambled)
        self.assertEqual(len(learned['cliques']),6)
        self.assertEqual(learned['coactive_pairs'],36)
        self.assertEqual(learned['noncoactive_pairs'],69)
        truth={frozenset(perm[i] for i,e in enumerate(edges) if v in e)
               for v in range(6)}
        self.assertEqual(set(map(frozenset,learned['cliques'])),truth)
        self.assertEqual(len(set(decode_mask(learned,m) for m in scrambled)),12)

    def test_feature_scramble_covariance(self):
        rows,_,_=exact_recovery_oracle()
        original=infer_feature_geometry(rows)
        for seed in (7,21,39,101):
            permuted,p=scramble_masks(rows,seed)
            fitted=infer_feature_geometry(permuted)
            expected={frozenset(p[i] for i in star) for star in original['cliques']}
            self.assertEqual(set(map(frozenset,fitted['cliques'])),expected)

    def test_all_455_three_holdout_star_covers(self):
        states=tuple(all_matchings(range(6)))
        edges=tuple(itertools.combinations(range(6),2))
        rows=tuple(tuple(int(e in m) for e in edges) for m in states)
        for held in itertools.combinations(range(15),3):
            fitted=infer_feature_geometry([row for j,row in enumerate(rows)
                                            if j not in held])
            self.assertEqual(len(fitted['cliques']),6)

    def test_four_missing_requires_fail_closed_ambiguity(self):
        states=tuple(all_matchings(range(6)))
        edges=tuple(itertools.combinations(range(6),2))
        rows=tuple(tuple(int(e in m) for e in edges) for m in states)
        good=(0,1,2,3)
        bad=(0,1,5,12)
        self.assertEqual(len(infer_feature_geometry(
            [row for j,row in enumerate(rows) if j not in good])['cliques']),6)
        with self.assertRaises(ValueError):
            infer_feature_geometry([row for j,row in enumerate(rows)
                                     if j not in bad])

    def test_hard_holdout_silent_channel_never_seen_during_training(self):
        f=fixture(seed=43,feature_seed=1784)
        self.assertEqual(len(f['train_perms']),576)
        self.assertEqual(len(f['test_perms']),144)
        train={f['masks'][j] for j in f['train_states']}
        test={f['masks'][j] for j in f['holdout_states']}
        self.assertFalse(train & test)
        self.assertEqual(sum(all(not mask[i] for mask in train)
                             for i in range(15)),1)
        inferred,_=discover_from_unlabelled_train_states(f)
        self.assertEqual(inferred['candidate_5_cliques'],47)
        self.assertEqual(len(set(decode_mask(inferred,x) for x in train)),12)
        self.assertEqual(len(set(decode_mask(inferred,x) for x in test)),3)

    def test_three_hidden_frames_test_complete_unseen_matching_classes(self):
        for seed in (43,77,91):
            row=simulate(seed,feature_seed=1701+seed,
                         calibration_seed=seed+29,examples=12)
            self.assertEqual(row['calibration_version_space'],1)
            self.assertTrue(row['inferred_model_matches_truth'])
            self.assertEqual(row['heldout_balanced_accuracy'],'5/6')
            self.assertEqual(row['candidate_5_cliques'],47)

    def test_denoising_recovers_geometry_and_frame(self):
        row=simulate(43,1701,100,12,noise=(.75,.25),repeats=40)
        self.assertTrue(row['inferred_model_matches_truth'])
        self.assertEqual(row['heldout_balanced_accuracy'],'5/6')
        raw,_,_=exact_recovery_oracle()
        scrambled,_=scramble_masks(raw)
        for seed in (100,101,102):
            episodes=noisy_replicates(scrambled,40,.75,.25,seed)
            self.assertEqual(majority_denoise(episodes),scrambled)

    def test_row_column_preserving_null_is_not_recognized(self):
        f=fixture()
        masks=tuple(f['masks'][j] for j in f['train_states'])
        null=switching_null(masks,seed=0)
        self.assertEqual([sum(r) for r in null],[3]*12)
        self.assertEqual([sum(r[i] for r in null) for i in range(15)],
                         [sum(r[i] for r in masks) for i in range(15)])
        with self.assertRaises(ValueError):
            infer_feature_geometry(null)
        control=null_campaign(seeds=range(12))
        self.assertEqual(control['geometry_certificates_accepted'],0)

    def test_noise_smoke_and_invalid_geometry(self):
        report=geometry_noise_campaign(seeds=range(12),noise=(.9,.1),repeats=16)
        self.assertEqual(report['correct_reconstructions'],12)
        with self.assertRaises(ValueError):
            infer_feature_geometry([(0,)*15]*12)


if __name__=='__main__':
    unittest.main()
