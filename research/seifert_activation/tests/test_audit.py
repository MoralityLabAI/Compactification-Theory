import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scm05_audit import audit_record, run, CLAIMS
from test_continuous import ring, caps


def data():
    n=64
    return {'schema':'scm04_RMP_fixture',
            'checkpoint_sha256':'fake',
            'capture':{'angles':[6.283185307179586*k/n for k in range(n+1)],
                       'frames_by_visit':[ring(n),ring(n, radius=2.1)]}}


def bounds():
    return {'schema':CLAIMS,'scope':'external_analytic_witness',
            'derivation_reference':'math: circles of radius 2 and 2.1, rotation k=1',
            'by_visit':{'1':caps(2.000001),'2':caps(2.100001)}}


class AuditTests(unittest.TestCase):
    def test_missing_evidence_explicit(self):
        result=audit_record(data(),None)
        self.assertEqual([r['status'] for r in result['visits']],
                         ['missing_external_interval_bounds']*2)

    def test_conditional_audit_and_quant_knee(self):
        result=audit_record(data(),bounds(),quant_range='3',max_bits=12)
        self.assertEqual([r['status'] for r in result['visits']],
                         ['conditional_pass','conditional_pass'])
        self.assertTrue(all(r['quantization']['certified'] for r in result['visits']))
        self.assertTrue(all(r['conditional_on_unverified_external_bounds'] for r in result['visits']))

    def test_partial_bounds_leave_explicit_unverified(self):
        b=bounds(); del b['by_visit']['2']
        result=audit_record(data(),b)
        self.assertEqual(result['visits'][1]['status'],'missing_external_interval_bounds')

    def test_visit_angle_mismatch_rejected(self):
        x=data(); x['capture']['angles'].pop()
        with self.assertRaisesRegex(ValueError,'mismatch'):
            audit_record(x,bounds())

    def test_immutable_receipts_and_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp=pathlib.Path(temporary)
            source=temp/'records.jsonl'; claims=temp/'claims.json'; out=temp/'audited.jsonl'
            source.write_text(json.dumps(data())+'\n',encoding='utf-8')
            claims.write_text(json.dumps(bounds()),encoding='utf-8')
            info=run(source,out,claims_path=claims,quant_range='3')
            self.assertEqual(info['records'],1)
            line=json.loads(out.read_text())
            self.assertEqual(line['source_receipt_sha256'],info['source_receipt_sha256'])
            self.assertEqual(line['claim_file_sha256'],info['claim_file_sha256'])
            with self.assertRaises(FileExistsError):
                run(source,out,claims_path=claims)

    def test_claim_needs_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp=pathlib.Path(temporary)
            (temp/'records.jsonl').write_text(json.dumps(data())+'\n')
            (temp/'claims.json').write_text(json.dumps({'schema':CLAIMS,'by_visit':{}}))
            with self.assertRaisesRegex(ValueError,'provenance'):
                run(temp/'records.jsonl',temp/'out.jsonl',claims_path=temp/'claims.json')


if __name__ == '__main__':
    unittest.main()

class ProvenanceTests(unittest.TestCase):
    def test_bound_claim_checkpoint_lock(self):
        x=data(); b=bounds()
        b['source_sha256']={'checkpoint_sha256':'different'}
        with self.assertRaisesRegex(ValueError,'does not match'):
            audit_record(x,b)

    def test_nonpositive_frame_gaps_rejected(self):
        x=data()
        x['capture']['angles'][4]=x['capture']['angles'][3]
        with self.assertRaisesRegex(ValueError,'must increase'):
            audit_record(x,bounds())

    def test_derivative_ceiling_without_claim_is_diagnostic(self):
        result=audit_record(data(),None)
        first=result['visits'][0]
        self.assertGreater(first['diagnostic']['max_sufficient_uniform_L_by_pair']['0,1'],2)
        self.assertFalse(first['independently_certified'])
