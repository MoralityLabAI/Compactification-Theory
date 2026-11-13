"""SCM-08 registered large-shape source-only negative control."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scm08_large_reference import run_reference


class LargeShapeReference(unittest.TestCase):
    def test_source_only_tubes_refuse_late_null_claim(self):
        report=run_reference()
        source=Fraction(report['source_global_pair_upper'][-1])
        local=Fraction(report['source_derived_local_pair_upper'][-1])
        self.assertGreater(source,local)
        self.assertGreater(local,1)
        self.assertEqual(report['source_only_null_certificate_count'],0)
        self.assertFalse(report['floating_output_certified'])
        self.assertEqual(len(report['derived_clean_post_radius_max']),8)

if __name__=='__main__':
    unittest.main()
