"""SCM-08 independent end-to-end source-tube and input-RMS checks."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clean_tubes import (clean_state_tubes,certified_injection_amplitude,
                         certified_zero_disk,verified_local_pair_bounds)
from exact_input import InputEnclosure,rmp_input_enclosure
from rmp_capture import sha256_tensor
from scm08_audit import examine_record,run
from test_clean_tubes import ToyRMP


class IndependentProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(29)
        cls.model=ToyRMP(beta=.075).eval()
        cls.tokens=torch.tensor([1,2,3,4,5])
        cls.source=rmp_input_enclosure(cls.model,cls.tokens)
        cls.tubes=clean_state_tubes(cls.model,cls.source)
        cls.chart=torch.eye(4)[:,:2]

    def test_source_derived_rms_above_clean(self):
        clean=self.model.run_clean(self.tokens[None])[2]
        for visit in range(3):
            for token in (0,2,4):
                a=certified_injection_amplitude(self.source,self.tubes,
                      injected_token=token,after_visit=visit,radius_fraction='1')
                recorded=float(clean[visit][token].square().mean().sqrt())
                self.assertGreaterEqual(float(a)+1e-6,recorded)

    def test_rmp_clamp_min_applies_to_zero_clean_state(self):
        rows=((Fraction(0),)*4,)*5
        source=InputEnclosure(center=torch.zeros(5,4,dtype=torch.float64),
                             radii=(Fraction(0),)*5,exact_centers=rows,
                             frequency_sha256='zero-synthetic')
        model=ToyRMP(beta=0).eval()
        tubes=clean_state_tubes(model,source)
        amplitude=certified_injection_amplitude(source,tubes,
                     injected_token=0,after_visit=0,radius_fraction=Fraction(1,2))
        self.assertEqual(amplitude,Fraction(*float(1e-6).as_integer_ratio())/2)

    def test_reinjection_source_amplitude(self):
        model=ToyRMP(tied=False,beta=.05).eval()
        inp=rmp_input_enclosure(model,self.tokens)
        tubes=clean_state_tubes(model,inp,fixed_input_reinjection=True)
        clean=model.run_clean(self.tokens[None],reinjection=True)[2]
        for visit in range(3):
            cap=certified_injection_amplitude(inp,tubes,injected_token=1,
                                              after_visit=visit,radius_fraction='.05')
            self.assertGreater(float(cap)+1e-6,.05*float(clean[visit][1].square().mean().sqrt()))

    def test_float_capture_radius_interpreted_dyadically(self):
        from rmp_lipschitz import rational
        # The Python float 0.05 lies slightly ABOVE the exact decimal 1/20.
        self.assertGreater(rational(.05),Fraction(1,20))
        exact=certified_injection_amplitude(self.source,self.tubes,
               injected_token=0,after_visit=0,radius_fraction=rational(.05))
        decimal=certified_injection_amplitude(self.source,self.tubes,
               injected_token=0,after_visit=0,radius_fraction=Fraction(1,20))
        self.assertGreater(exact,decimal)

    def test_invalid_radius_rejected(self):
        with self.assertRaises(ValueError):
            certified_injection_amplitude(self.source,self.tubes,
                    injected_token=0,after_visit=1,radius_fraction='-1')

    def test_exact_null_braid_for_separated_clean_strands(self):
        # A fully exact four-row synthetic source without positional ambiguity.
        rows=((Fraction(0),)*4,(Fraction(10),Fraction(0),Fraction(0),Fraction(0)),
              (Fraction(0),Fraction(10),Fraction(0),Fraction(0)),
              (Fraction(-10),Fraction(-10),Fraction(0),Fraction(0)))
        inp=InputEnclosure(center=torch.tensor([[float(v) for v in row] for row in rows],dtype=torch.float64),
                radii=(Fraction(0),)*4,exact_centers=rows,frequency_sha256='synthetic')
        model=ToyRMP(beta=0).eval()
        tube=clean_state_tubes(model,inp)
        chart=self.chart
        bound=verified_local_pair_bounds(model,inp,chart=chart,axes=chart,
             amplitude='1',observed_tokens=(1,2,3),injected_token=0,clean=tube)
        report=certified_zero_disk(inp,tube,chart,bound,(1,2,3))
        self.assertTrue(all(item['null_braid_for_ideal_real_model'] for item in report))
        self.assertTrue(all(item['pairs']['0,1']['clean_pair_lower'] != '0' for item in report))

    def test_large_intervention_prevents_null_certification(self):
        source=self.source; tube=self.tubes
        bound=verified_local_pair_bounds(self.model,source,chart=self.chart,axes=self.chart,
                  amplitude=1000,observed_tokens=(1,2,3),injected_token=1,clean=tube)
        out=certified_zero_disk(source,tube,self.chart,bound,(1,2,3))
        self.assertFalse(all(item['null_braid_for_ideal_real_model'] for item in out))

    def test_chart_dimension_refused(self):
        bound=verified_local_pair_bounds(self.model,self.source,chart=self.chart,axes=self.chart,
                  amplitude='1',observed_tokens=(1,2,3),injected_token=0,clean=self.tubes)
        with self.assertRaises(ValueError):
            certified_zero_disk(self.source,self.tubes,torch.zeros(3,2),bound,(1,2,3))

    def test_audit_recalculates_from_tokens(self):
        trace={'sequence_length':len(self.tokens),'observed_token_positions':[1,2,3],
               'injected_position':0,'injected_after_visit':0,'radius_in_token_rms':.05,
               'intervention_axes_sha256':sha256_tensor(self.chart)}
        record={'capture':trace,'token_sequence_sha256':sha256_tensor(self.tokens),
                'intervention_axes_sha256':sha256_tensor(self.chart),
                'chart_sha256':sha256_tensor(self.chart),'model':'toy-T',
                'task_id_sha256':'example'}
        analysis=SimpleNamespace(model=self.model,config=SimpleNamespace(input_injection=False))
        result=examine_record(record,analysis,self.tokens,self.chart,self.chart)
        self.assertTrue(result['source_theorem_verified_by_this_program'])
        self.assertFalse(result['observed_link_certified'])
        self.assertEqual(len(result['null_braid_ideal_real']),3)

    def test_audit_rejects_wrong_task_hash(self):
        trace={'sequence_length':5,'observed_token_positions':[1,2,3],
               'injected_position':0,'injected_after_visit':0,'radius_in_token_rms':.05,
               'intervention_axes_sha256':sha256_tensor(self.chart)}
        rec={'capture':trace,'token_sequence_sha256':'wrong',
             'intervention_axes_sha256':sha256_tensor(self.chart),
             'chart_sha256':sha256_tensor(self.chart)}
        analysis=SimpleNamespace(model=self.model,config=SimpleNamespace(input_injection=False))
        with self.assertRaisesRegex(ValueError,'token sequence hash'):
            examine_record(rec,analysis,self.tokens,self.chart,self.chart)

    def test_audit_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'exists.jsonl';path.write_text('already protected')
            with self.assertRaises(FileExistsError):
                run(Path(tmp)/'missing.jsonl',path,rmp_root=Path(tmp),runs=Path(tmp),
                    artifacts=Path(tmp),dictionary=Path(tmp),monitor_bundle=Path(tmp))


if __name__=='__main__':
    unittest.main()
