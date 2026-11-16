"""SMC-19 CPU tests for frozen SAE refit, source mapping and gauge transport."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc19_sae_refit as s
from smc07_real_sae import TopKSAE


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,dtype=torch.float64,bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1,0]=2.


class SMC19RefitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_stable_exact_orthogonal_coordinate_transport(self):
        torch.manual_seed(24)
        teacher=ToyTeacher().eval()
        sae=TopKSAE(48,96,4).eval()
        mean=torch.randn(48)
        scale=torch.rand(48)+.5
        stage=torch.randn(4,81,48)
        positions=torch.arange(8,dtype=torch.long).unsqueeze(0).repeat(4,1)
        result=s.transport_control(teacher,sae,mean,scale,stage,positions)
        self.assertEqual(result['tested_puzzles'],4)
        self.assertEqual(result['tested_goal_tokens'],32)
        for key in ('orthogonality_max_error','feature_code_max_abs_error',
                    'clean_logits_max_abs_error','native_erase_logits_max_abs_error'):
            self.assertLess(result[key],2e-9)

    def test_repeated_q_and_permutation_hashes(self):
        q=s.rotation_matrix()
        self.assertTrue(torch.equal(q,s.rotation_matrix()))
        self.assertTrue(bool(torch.allclose(q.T@q,torch.eye(48,dtype=torch.float64),atol=1e-12)))
        self.assertEqual(s.sha_mapping(list(range(96))),s.sha_mapping(list(range(96))))
        self.assertNotEqual(s.sha_mapping(list(range(96))),s.sha_mapping(list(reversed(range(96)))))

    def test_four_puzzle_calibration_assignment_seeds_are_frozen(self):
        torch.manual_seed(52)
        # Simulate four calibration puzzles at 81 Sudoku positions.
        A=(torch.rand(4,81,96)>.90).float()
        B=A.clone()
        x=torch.zeros(4,81,dtype=torch.long)
        maps1,sha1=s.exact_assignments(A,B,x,2)
        maps2,sha2=s.exact_assignments(A,B,x,2)
        self.assertEqual(maps1,maps2)
        self.assertEqual(sha1,sha2)
        self.assertEqual(sorted(maps1['matched']),list(range(96)))
        self.assertEqual(sorted(maps1['clue_shuffled']),list(range(96)))
        self.assertEqual(sorted(maps1['random']),list(range(96)))

    def test_calibration_signs_from_only_raw_eligible_goals(self):
        rows=[
          {'source_drop':[1.,-2.]+[0.]*6,
           'matched_drop':[-3.,4.]+[0.]*6,
           'other_active_drop':[3.,-4.]+[0.]*6,
           'eligible_mask':[True,True]+[False]*6}
        ]
        signs=s.calibrate_signs(rows)
        self.assertEqual(signs['matched']['sign'],-1)
        self.assertEqual(signs['other']['sign'],1)
        self.assertTrue(signs['matched']['identified'])
        self.assertEqual(signs['matched']['eligible_calibration_goals'],2)

    def test_invalid_new_cohort_request_fails_before_generator(self):
        with self.assertRaises(ValueError):s.fresh_rosters({},limit=5)

    def test_frozen_receipt_does_not_accept_missing_weights_or_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'frozen.json').write_text(json.dumps({'schema':'forged'}))
            with self.assertRaisesRegex(ValueError,'original frozen refit'):
                s.load_frozen(root,{}, {},{})


if __name__=="__main__":unittest.main()
