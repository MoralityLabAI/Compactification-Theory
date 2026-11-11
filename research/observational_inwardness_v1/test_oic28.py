"""OIC-28 finite checker tests. All assertions remain active with python -O."""
import copy,json,unittest
from pathlib import Path
from oic28_verify import verify_tree,clean
from oic28_reference import check

class OIC28Tests(unittest.TestCase):
    def test_frozen_winning_tree(self):
        manifest,tree=check()
        self.assertEqual(manifest['certificate_stats']['nodes'],1697)
        self.assertEqual(manifest['certificate_stats']['true_world_fault_paths'],1029)
        self.assertEqual(verify_tree(tree,n=7,q=20),manifest['certificate_stats'])
    def test_corruption_of_a_report_branch_is_detected(self):
        _,tree=check()
        changed=copy.deepcopy(tree)
        changed['children'].pop('0')
        with self.assertRaises(AssertionError):verify_tree(changed,7,20)
    def test_incorrect_leaf_is_detected(self):
        _,tree=check();changed=copy.deepcopy(tree)
        def find_leaf(node):
            if 'shift' in node:return node
            for child in node['children'].values():
                found=find_leaf(child)
                if found is not None:return found
            return None
        leaf=find_leaf(changed)
        leaf['shift']=(leaf['shift']+1)%7
        with self.assertRaises(AssertionError):verify_tree(changed,7,20)
    def test_phase_affine_equivariance(self):
        # Show exact relationship between affine gauge and each possible sensor query.
        n=7
        for d in range(n):
            for z in range(n):
                for t in range(21):
                    for m in range(n):
                        y=clean(n,d,z,t,m)
                        for u in (1,2,3,4,5,6):
                            for c in range(n):
                                self.assertEqual(y,clean(n,(u*d)%n,(u*z+c)%n,t,(u*m+c)%n))
    def test_gauge_preserves_shift_identification(self):
        for u in range(1,7):
            self.assertEqual(len({(u*d)%7 for d in range(7)}),7)
    def test_receipt_horizons_are_distinct(self):
        record,_=check()
        self.assertLess(record['negative_horizon'],record['positive_horizon'])
        self.assertEqual([r['status'] for r in record['searches']],['UNSAT','SAT'])
        self.assertEqual(record['logical_candidate_bits'],98)

if __name__=='__main__':unittest.main()
