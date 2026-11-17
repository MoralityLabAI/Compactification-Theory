"""SCM-21 CPU-only deterministic source and exact-count tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from analysis import (SNAPSHOT, audit, git_blob_sha1, load_snapshot,
                      pairwise_success_bounds, _source_pack)
from synthetic import run

class TestCapacity(unittest.TestCase):
    def test_summary_counts_and_registered_primary(self):
        report=audit(load_snapshot())
        self.assertEqual(report['total_seed_task_comparisons'],748)
        self.assertEqual(report['method_successes']['semantic_diff4'],217)
        self.assertEqual(report['method_successes']['native_write4'],63)
        self.assertEqual(report['method_successes']['native_vpd_write'],55)
        self.assertEqual(report['method_successes']['full_cls'],639)
        self.assertTrue(report['all_six_cells_semantic_exceeds_native_write'])
    def test_rank_four_lower_bound(self):
        report=audit(load_snapshot())
        self.assertEqual(report['minimum_semantic_diff_only_vs_native_write_span'],154)
        self.assertEqual([r['guaranteed_semantic_only_vs_write_span'] for r in report['per_cell']],[25,23,27,28,37,14])
        self.assertEqual(report['semantic_vs_native_write_difference'],154)
        self.assertEqual(report['semantic_vs_native_actual_difference'],162)
    def test_frechet_sharp_by_subset_enumeration(self):
        from itertools import combinations
        for n in range(1,7):
            items=range(n)
            for a in range(n+1):
                for b in range(n+1):
                    vals=[]
                    for av in combinations(items,a):
                        for bv in combinations(items,b):
                            vals.append(len(set(av)-set(bv)))
                    bound=pairwise_success_bounds(n,a,b)
                    self.assertEqual((min(vals),max(vals)),(bound['minimum_a_only'],bound['maximum_a_only']))
    def test_refuses_invalid_snapshot(self):
        data=load_snapshot()
        for change in ('methods','count','duplicate','denominator','sham'):
            x=copy.deepcopy(data)
            if change=='methods':x['method_order'].reverse()
            if change=='count':x['rows'][0]['counts'][4]=999
            if change=='duplicate':x['rows'][0]['seed']=1191
            if change=='denominator':x['rows'][0]['n']=0
            if change=='sham':x['rows'][0]['counts'][-1]=1
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'snapshot.json';p.write_text(json.dumps(x))
                with self.assertRaises(ValueError,msg=change):load_snapshot(p)
    def test_git_blob_hash_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json';p.write_text('{}')
            self.assertNotEqual(git_blob_sha1(p.read_bytes()),load_snapshot()['source_git_blob_sha1'])
            with self.assertRaises(ValueError):load_snapshot(summary_path=p)
    def test_source_reconstruction_refuses_incomplete(self):
        with self.assertRaises(ValueError):_source_pack({'schema':'scm20.summary.v1','audit':{'passed':True},'secondary':[]})
        with self.assertRaises(ValueError):_source_pack({'schema':'bad'})
    def test_mean_norms_confounded(self):
        r=audit(load_snapshot())
        self.assertGreater(r['semantic_norm_ratio_range'][0],1)
        self.assertLess(r['semantic_norm_ratio_range'][1],1.07)
    def test_clean_preservation_replay(self):
        r=audit(load_snapshot())
        failures=sum((row['delta']/512) < -0.02 for row in r['clean_accuracy_consequence'])
        self.assertEqual(failures,5)
    def test_synthetic_norm_matched_control(self):
        r=run(seed=20261117,calibration=80,heldout=256)
        self.assertTrue(r['same_per_case_patch_norm'])
        self.assertLess(r['max_native_readout_effect'],1e-10)
        self.assertEqual(r['successes']['difference_pca4'],256)
        self.assertEqual(r['successes']['native_fiber4'],0)
        self.assertEqual(r['successes']['full_difference'],256)

if __name__=='__main__':unittest.main()
