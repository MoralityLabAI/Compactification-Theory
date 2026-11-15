"""SMC-13 generator and exact Sudoku-orbit separation regression suite."""
import unittest
import random
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc13_family import (
    base_solution,rectangle_count,valid_solution,new_full_solution,
    count_solutions,dig_unique,cohort,build_rosters,fingerprint
)


class IndependentSudokuFamilyTests(unittest.TestCase):
    def test_canonical_orbit_has_zero_alternating_rectangles(self):
        base=base_solution()
        self.assertTrue(valid_solution(base))
        self.assertEqual(rectangle_count(base),0)
        # Row/col/digit permutations and transpose preserve this count.
        rng=random.Random(1713)
        for _ in range(20):
            rows=list(range(9));cols=list(range(9));digits=list(range(1,10))
            rng.shuffle(rows);rng.shuffle(cols);rng.shuffle(digits)
            g=[digits[base[9*rows[r]+cols[c]]-1]
               for r in range(9) for c in range(9)]
            if rng.randrange(2):
                g=[g[9*c+r] for r in range(9) for c in range(9)]
            self.assertEqual(rectangle_count(g),0)

    def test_new_solutions_prove_out_of_source_orbit(self):
        for seed in range(202613030000,202613030024):
            g=new_full_solution(seed)
            self.assertTrue(valid_solution(g))
            self.assertGreater(rectangle_count(g),0)

    def test_unique_puzzle_generation_and_givens(self):
        for i in range(6):
            g=new_full_solution(202613040000+i)
            puzzle=dig_unique(g,202614040000+i,holes=34)
            self.assertEqual(sum(x==0 for x in puzzle),34)
            self.assertEqual(count_solutions(puzzle),1)
            self.assertTrue(all(a==0 or a==b for a,b in zip(puzzle,g)))

    def test_full_split_smoke_is_disjoint_and_hashed(self):
        a,receipt=build_rosters(quick=True)
        self.assertEqual(receipt['canonical_intercalates'],0)
        self.assertEqual(set(a),{
            'teacher_train','teacher_val','feature_calibration',
            'test_novel_30','test_novel_34','test_canonical_30'})
        self.assertTrue(receipt['disjoint_full_solution_targets'])
        self.assertEqual([len(x) for x in a.values()],[4]*6)
        for cohort_name,rows in a.items():
            self.assertEqual(receipt['cohorts'][cohort_name]['sha256'],fingerprint(rows))
            if cohort_name=='test_canonical_30':
                self.assertTrue(all(row['rectangles']==0 for row in rows))
            else:
                self.assertTrue(all(row['rectangles']>0 for row in rows))

    def test_invalid_generator_inputs(self):
        with self.assertRaises(ValueError):count_solutions([1,2])
        with self.assertRaises(ValueError):cohort('unregistered')
        with self.assertRaises(ValueError):cohort('teacher_train',193)
        with self.assertRaises(ValueError):dig_unique(new_full_solution(2026130),0,82)
        with self.assertRaises(ValueError):rectangle_count([1,2])


if __name__=='__main__':
    unittest.main()
