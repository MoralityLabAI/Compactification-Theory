"""Instrument tests; live published summary replay occurs in GitHub CI."""
import copy
import unittest
from research.dimensional_atlas.gym.bpd_quotient_bridge.audit import (
 summarize,strict_check,git_blob_sha,METHODS,EXPECT,EXPECTED_SEEDS,EXPECTED_STRATA,ROOT,DEFAULT)


def fixture():
    rows=[]
    for seed in EXPECTED_SEEDS:
        for stratum in EXPECTED_STRATA:
            for m in METHODS:
                k=EXPECT[m]
                # Test a well-formed *synthetic*, not the published success counts.
                rows.append({'seed':seed,'stratum':stratum,'method':m,'model':'causal',
                             'visit':1,'n':128,'correct':k//6,'mean_patch_norm':1.0 if m=='native_vpd_write' else 2.0})
    return {'schema':'scm21.summary.v1','secondary':rows}

class AuditTests(unittest.TestCase):
    def test_valid_shape(self):
        x=summarize(fixture())
        self.assertEqual(x['methods']['native_vpd_write']['n'],768)
        self.assertEqual(x['methods']['native_vpd_write']['correct'],0)
    def test_duplicate(self):
        f=fixture();f['secondary'].append(f['secondary'][0]);
        with self.assertRaises(ValueError):summarize(f)
    def test_missing(self):
        f=fixture();f['secondary'].pop()
        with self.assertRaises(ValueError):summarize(f)
    def test_negative(self):
        f=fixture();f['secondary'][0]['correct']=-1
        with self.assertRaises(ValueError):summarize(f)
    def test_mismatched_schema(self):
        f=fixture();f['schema']='wrong'
        with self.assertRaises(ValueError):summarize(f)
    def test_unregistered_method(self):
        f=fixture();f['secondary'][0]['method']='future_leak'
        with self.assertRaises(ValueError):summarize(f)
    def test_source_path(self):
        self.assertEqual(ROOT.name,"gym")
        self.assertEqual(DEFAULT.relative_to(ROOT).as_posix(),"scm21_conservative_causal/results/summary.json")
    def test_hash(self):
        self.assertEqual(git_blob_sha(b'a'), '2e65efe2a145dda7ee51d1741299f848e5bf752e')
    def test_strict_refuses_synthetic_counts(self):
        with self.assertRaises(ValueError):strict_check(summarize(fixture()))

if __name__=='__main__':unittest.main()
