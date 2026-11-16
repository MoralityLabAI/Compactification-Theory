"""Independent SMC-19 standard-library receipt replay mutation checks."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc19 as a


class SMC19AuditTests(unittest.TestCase):
    def test_seeded_sign_concordance_and_cluster_bootstrap(self):
        rows=[]
        for i in range(64):
            on=i<24
            rows.append({
              'eligible_mask':[on]+[False]*7,
              'source_drop':[1.]*8,
              'matched_drop':[1.]*8,
              'other_active_drop':[-1.]*8
            })
        signs={'matched':{'identified':True,'sign':1},
               'other':{'identified':True,'sign':1}}
        result=a.score_replay(rows,signs,seed=202619130)
        self.assertTrue(result['registered_success'])
        self.assertEqual(result['eligible_goals'],24)
        self.assertEqual(result['nonempty_puzzle_clusters'],24)
        self.assertEqual(result['matched_minus_other_active'],2.)
        a.compare_dict(result,result,'same-data')
        forged=dict(result)
        forged['puzzle_cluster_bootstrap95']=[-.5,2.]
        with self.assertRaisesRegex(ValueError,'numeric array'):
            a.compare_dict(result,forged,'forged')

    def test_gauge_only_accepts_frozen_orthogonal_tolerance(self):
        good={'rotation_seed':20261919,
              'tested_puzzles':4,'tested_goal_tokens':32,
              'gauge_type':'exact joint orthogonal row-hidden transport',
              'orthogonality_max_error':1e-14,
              'feature_code_max_abs_error':1e-12,
              'clean_logits_max_abs_error':1e-12,
              'native_erase_logits_max_abs_error':1e-12}
        a.audit_gauge(good)
        bad=dict(good)
        bad['native_erase_logits_max_abs_error']=.01
        with self.assertRaisesRegex(ValueError,'transport'):
            a.audit_gauge(bad)
        bad=dict(good)
        bad['rotation_seed']=20261920
        with self.assertRaisesRegex(ValueError,'transport'):
            a.audit_gauge(bad)

    def test_snapshot_file_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=root/'saes'/'teacher501_sae131.pt'
            p.parent.mkdir(parents=True)
            p.write_bytes(b"source-frozen-toy-checkpoint")
            item={'teacher_seed':501,'sae_seed':131,
                  'snapshot_relative_path':'saes/teacher501_sae131.pt',
                  'snapshot_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            a.audit_frozen_files({'refit_roster':[item]},root)
            p.write_bytes(b"forged")
            with self.assertRaisesRegex(ValueError,'file-byte'):
                a.audit_frozen_files({'refit_roster':[item]},root)

    def test_disallowed_new_roster_fails(self):
        with self.assertRaises(ValueError):
            a.audit_rosters({'puzzles_per_family':12,'fresh_rosters':{},
                             'test_seed_bases':{}})

    def test_frozen_original_historical_receipt_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'protocol'):
            a.audit_original({'pre_registration_commit':'0'*40})

    def test_calibration_random_bijection_is_seeded(self):
        perm=list(range(96))
        import random
        random.Random(202619810).shuffle(perm)
        self.assertEqual(sorted(perm),list(range(96)))
        self.assertEqual(a.digest(json.dumps(perm).encode()),
                         hashlib.sha256(json.dumps(perm).encode()).hexdigest())


if __name__=="__main__":unittest.main()
