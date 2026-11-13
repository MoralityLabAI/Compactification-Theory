"""SCM-09 source provenance, legacy receipts and fail-closed tests."""
from __future__ import annotations

from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rmp_capture import sha256_tensor,capture_cycle
from scm09_audit import examine_record,run,SCHEMA
from test_clean_tubes import ToyRMP
from test_rmp_capture import MockRMP
from exact_input import rmp_input_enclosure
from clean_tubes import clean_state_tubes
from validated_forward import verified_centered_tubes


class SourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(207)
        cls.model=ToyRMP(beta=.04).eval()
        cls.tokens=torch.tensor([1,2,3,4,5])
        cls.chart=torch.eye(4,dtype=torch.float32)[:,:2]
        cls.analysis=SimpleNamespace(model=cls.model,config=SimpleNamespace(input_injection=False))

    def sample_record(self):
        return {
            'capture':{'sequence_length':5,'observed_token_positions':[1,2,3],
                       'injected_position':0,'injected_after_visit':0,
                       'radius_in_token_rms':0.05,
                       'intervention_axes_sha256':sha256_tensor(self.chart)},
            'token_sequence_sha256':sha256_tensor(self.tokens),
            'intervention_axes_sha256':sha256_tensor(self.chart),
            'chart_sha256':sha256_tensor(self.chart),
            'model':'toy', 'task_id_sha256':'synthetic',
            'checkpoint_sha256':'synthetic', 'dictionary_sha256':'synthetic',
        }

    def test_rederived_proof(self):
        result=examine_record(self.sample_record(),self.analysis,self.tokens,self.chart,self.chart)
        self.assertEqual(result['schema'],SCHEMA)
        self.assertTrue(result['clean_center_theorem_verified'])
        self.assertFalse(result['observed_link_certified'])
        self.assertEqual(len(result['clean_post_radii']),3)
        self.assertTrue(all(float(Fraction(x))<1e-10 for x in result['clean_post_radii']))

    def test_legacy_missing_sequence_length_with_hash(self):
        record=self.sample_record();record['capture'].pop('sequence_length')
        result=examine_record(record,self.analysis,self.tokens,self.chart,self.chart)
        self.assertTrue(result['legacy_missing_sequence_length'])

    def test_wrong_sequence_length(self):
        record=self.sample_record();record['capture']['sequence_length']=6
        with self.assertRaisesRegex(ValueError,'sequence length'):
            examine_record(record,self.analysis,self.tokens,self.chart,self.chart)

    def test_wrong_tokens(self):
        record=self.sample_record();record['token_sequence_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'token hash'):
            examine_record(record,self.analysis,self.tokens,self.chart,self.chart)

    def test_wrong_chart(self):
        record=self.sample_record();wrong=self.chart.flip(0)
        with self.assertRaisesRegex(ValueError,'chart or injection'):
            examine_record(record,self.analysis,self.tokens,wrong,self.chart)

    def test_refuses_oversized_source(self):
        with self.assertRaisesRegex(ValueError,'size cap'):
            examine_record(self.sample_record(),self.analysis,self.tokens,self.chart,self.chart,max_seq=4)

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as temp:
            existing=Path(temp)/'existing.jsonl';existing.write_text('old')
            with self.assertRaises(FileExistsError):
                run(Path(temp)/'missing',existing,rmp_root=Path(temp),runs=Path(temp),
                    artifacts=Path(temp),dictionary=Path(temp),monitor_bundle=Path(temp))

    def test_capture_sequence_length_added(self):
        trace=capture_cycle(MockRMP(),torch.tensor([[0,1,2,3]]),torch.eye(2),
            inject_token=0,observed_tokens=(1,2),radius=.5,angles=8)
        self.assertEqual(trace['sequence_length'],4)


class LongerRecurrenceTests(unittest.TestCase):
    def test_8_visit_tied_recurrence_bounds(self):
        torch.manual_seed(205)
        model=ToyRMP(tied=True,visits=8,beta=.1).eval()
        input=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        old=clean_state_tubes(model,input)
        new=verified_centered_tubes(model,input)
        self.assertEqual(len(new.post_center_errors),8)
        self.assertLess(float(max(new.post_center_errors[-1])),1e-9)
        self.assertGreater(float(max(old.post_radii[-1])),.01)

    def test_8_visit_untied_reinjected(self):
        torch.manual_seed(206)
        model=ToyRMP(tied=False,visits=8,beta=.05).eval()
        input=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        new=verified_centered_tubes(model,input,fixed_input_reinjection=True)
        self.assertLess(float(max(new.post_center_errors[-1])),1e-9)


if __name__=='__main__':
    unittest.main()
