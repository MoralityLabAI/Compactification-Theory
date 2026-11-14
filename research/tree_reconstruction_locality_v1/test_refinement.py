"""KSR-10 finite refinement certificate tests."""
from fractions import Fraction as F
import unittest
from refinement import certify


def record(name,label,row):
    return {'case_id':name,'label':label,'one_step_child_law':[str(x) for x in row]}


class RefinementTests(unittest.TestCase):
    def test_mixture_requires_two_codes_within_one_original_label(self):
        rows=[record('a',0,[1,0]),record('b',0,[0,1])]
        r=certify(rows,2)
        self.assertEqual(r['distinct_kernels_per_original_label'],[2,0])
        self.assertEqual(r['minimal_supported_refinement_states'],2)
        self.assertEqual(r['minimal_joint_fixed_length_bits'],1)

    def test_refinement_preserves_original_label_even_when_child_laws_equal(self):
        rows=[record('a',0,[F(1,2),F(1,2)]),record('b',1,[F(1,2),F(1,2)])]
        r=certify(rows,2)
        self.assertEqual(r['minimal_supported_refinement_states'],2)
        self.assertEqual(r['fixed_extra_bits_given_original_label'],0)

    def test_repeated_kernel_states_share_code_and_have_zero_variance(self):
        rows=[record('a',0,[1,0]),record('b',0,[1,0]),
              record('c',0,[0,1]),record('d',1,[0,1])]
        r=certify(rows,2)
        self.assertEqual(r['minimal_supported_refinement_states'],3)
        self.assertEqual(r['minimal_joint_fixed_length_bits'],2)
        self.assertEqual(sorted(len(g['case_ids']) for g in r['groups']),[1,1,2])
        self.assertTrue(r['refined_parent_original_child_independence'])
        self.assertFalse(r['refined_child_recursive_markov_closure_claimed'])

    def test_invalid_and_duplicate_parents_rejected(self):
        with self.assertRaises(ValueError): certify([],2)
        with self.assertRaises(ValueError): certify([record('a',0,[1,1])],2)
        with self.assertRaises(ValueError): certify([record('a',3,[1,0])],2)
        with self.assertRaises(ValueError): certify([record('a',0,[1,0])]*2,2)


if __name__=='__main__': unittest.main()
