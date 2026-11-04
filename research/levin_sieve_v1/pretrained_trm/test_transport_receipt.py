"""Exact-rational negative-control audit for the LSV-16 receipt checker."""
import copy
import unittest
from verify_transport_receipt import verify

SHA='0'*64
SRC={'checkpoint_sha256':SHA,'revision':'f'*40}
REF={'checkpoint_sha256':SHA,'roster_sha256':'1'*64,
     'case_thresholds':[{'case_index':0,'winner_threshold_strict':'1/10',
                         'top_two_threshold_strict':'1/100'}]}
REC={'schema':'LSV-16/ambient-to-gauge-certified-transport-v1',
     'checkpoint_sha256':SHA,'source_revision':'f'*40,
     'roster_sha256':'1'*64,'precision_bits':160,
     'intervention_geometry':'independent ambient l-infinity perturbation on all 3888 stage1 coordinates',
     'radii':['1e-14'],
     'cases':[{'case_index':0,'ambient_radii':{'1e-14':{
         'verified_final_transverse_radius_upper':'1/200',
         'winner_certificate':True,'top_two_certificate':True,
         'LSV15_winner_threshold':'1/10','LSV15_top2_threshold':'1/100'}}}],
     'counts':{'1e-14':{'certified_winner':1,'certified_top2':1,'could_not_bound':0}}}

class Audit(unittest.TestCase):
    def test_accept_strict_positive(self):
        self.assertEqual(verify(REC,SRC,REF)['1e-14']['certified_top2'],1)

    def test_fail_false_reported_cert(self):
        r=copy.deepcopy(REC)
        r['cases'][0]['ambient_radii']['1e-14']['verified_final_transverse_radius_upper']='1/2'
        with self.assertRaisesRegex(ValueError,'false certificate'):verify(r,SRC,REF)

    def test_fail_changed_checkpoint(self):
        r=copy.deepcopy(REC);r['checkpoint_sha256']='2'*64
        with self.assertRaisesRegex(ValueError,'source lock'):verify(r,SRC,REF)

    def test_fail_changed_threshold(self):
        r=copy.deepcopy(REC);r['cases'][0]['ambient_radii']['1e-14']['LSV15_top2_threshold']='1'
        with self.assertRaisesRegex(ValueError,'threshold substitution'):verify(r,SRC,REF)

    def test_fail_changed_aggregate(self):
        r=copy.deepcopy(REC);r['counts']['1e-14']['certified_winner']=0
        with self.assertRaisesRegex(ValueError,'aggregate mismatch'):verify(r,SRC,REF)

    def test_failure_status_not_certificate(self):
        r=copy.deepcopy(REC)
        v=r['cases'][0]['ambient_radii']['1e-14']
        v.update(verified_final_transverse_radius_upper=None, winner_certificate=False,
                 top_two_certificate=False,reason='LayerNorm lower denominator <=0')
        r['counts']['1e-14']={'certified_winner':0,'certified_top2':0,'could_not_bound':1}
        self.assertEqual(verify(r,SRC,REF)['1e-14']['could_not_bound'],1)


    def test_v2_replay_includes_archived_anchor_drift(self):
        r=copy.deepcopy(REC)
        r['schema']='LSV-16/ambient-to-gauge-certified-transport-v2'
        r['cases'][0]['ambient_radii']['1e-14'].update(
            transport_component_upper='1/500',
            reference_anchor_drift_upper='3/1000')
        self.assertEqual(verify(r,SRC,REF)['1e-14']['certified_winner'],1)
        r['cases'][0]['ambient_radii']['1e-14']['reference_anchor_drift_upper']='1/1000'
        with self.assertRaisesRegex(ValueError,'drift accounting'):
            verify(r,SRC,REF)

if __name__=='__main__':unittest.main()
