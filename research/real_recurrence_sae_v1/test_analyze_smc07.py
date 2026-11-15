"""Pure-Python adversarial tests for puzzle-grouped analysis (no Torch)."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_smc07 import analyze,EXPECTED_CHECKPOINT,EXPECTED_ROSTER

def fake_receipt():
    puzzles=list(range(3,64,4))
    runs=[]
    for seed in (71,89,113):
        cases=[]
        transport=[]
        for p in puzzles:
            intr={}
            for r in ('0.2','0.8'):
                intr[r]={
                  'sae_active':{'goal_logit_delta_norm':0.3},
                  'sae_inactive':{'goal_logit_delta_norm':0.1},
                  'isotropic':{'goal_logit_delta_norm':0.15},
                  'same_direction_remote_token':{'goal_logit_delta_norm':0.05},
                }
            cases.append({'puzzle_id':p,'interventions':intr})
            transport.append({'puzzle_id':p,'same_token_overlap':0.6,
                              'shuffled_puzzle_overlap':0.1})
        runs.append({
          'seed':seed,
          'causal_goal_replay':{'per_puzzle':cases,'aggregates':{
            'sae_reconstruction':{'mean_logit_mse':0.1},
            'pca4_reconstruction':{'mean_logit_mse':0.2},
            '0.2_sae_active':{'mean_goal_logit_delta_norm':0.3},
            '0.2_sae_inactive':{'mean_goal_logit_delta_norm':0.1},
            '0.2_isotropic':{'mean_goal_logit_delta_norm':0.15},
            '0.2_same_direction_remote_token':{'mean_goal_logit_delta_norm':0.05},
          },'gauge_checks_fp64':[{'gauge_logit_linf':1e-15}]},
          'feature_transport':{'per_puzzle':transport},
          'reconstruction_by_depth':{'1':{'sae_mse_normalized':0.1,
                    'pca4_mse_normalized':0.2,'random4_mse_normalized':0.8}}
        })
    return {'schema':'SMC-07/source-locked-real-recurrence-sae-v1',
            'checkpoint_sha256':EXPECTED_CHECKPOINT,
            'roster_sha256':EXPECTED_ROSTER,'n_puzzles':64,
            'train_seed_list':[71,89,113],
            'split_ids':{'train':[0],'val':[2],'test':puzzles},
            'results':runs}

class ClusteredTests(unittest.TestCase):
    def test_independent_cluster_bootstrap_and_paired_bounds(self):
        r=analyze(fake_receipt(),draws=200)
        self.assertEqual(len(r['test_puzzle_ids']),16)
        effect=r['paired_cluster_differences']['0.2_sae_active_minus_isotropic']
        self.assertAlmostEqual(effect['mean'],0.15)
        self.assertEqual(effect['puzzles_positive'],16)
        self.assertTrue(all(abs(v-0.15)<1e-12 for v in effect['percentile_interval_95']))
        transport=r['paired_cluster_differences']['depth1_to2_same_minus_shuffled_top4']
        self.assertAlmostEqual(transport['mean'],0.5)

    def test_reject_bad_source_and_missing_puzzle(self):
        r=fake_receipt()
        r['checkpoint_sha256']='bad'
        with self.assertRaises(ValueError):analyze(r,draws=20)
        r=fake_receipt()
        r['results'][0]['causal_goal_replay']['per_puzzle'].pop()
        with self.assertRaises(ValueError):analyze(r,draws=20)

if __name__=='__main__':unittest.main()
