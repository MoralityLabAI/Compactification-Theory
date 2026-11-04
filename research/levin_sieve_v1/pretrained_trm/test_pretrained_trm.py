"""Offline tests for LSV-05 real pretraining interface and certificate algebra."""
import hashlib, io, json, tempfile, unittest
from pathlib import Path
import torch
from fractions import Fraction as Q

from upstream_arch import TinyRecursiveModel
from sudoku_generator import generate_puzzle, is_valid_solution
from certify import (certified_bounds, infinity_tensor, row_sum_norm,
                     row_delta_norm, layernorm_envelope, block_norm_and_storage_delta)
from run_checkpoint import blob_sha1, locked_bytes, load_pretrained, prepare_roster, evaluate

HERE=Path(__file__).resolve().parent


class TestPretrainedTRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lock=json.loads((HERE/'source_lock.json').read_text())

    def test_exact_upstream_parameter_count_and_four_steps(self):
        m=TinyRecursiveModel().eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()),44001)
        self.assertEqual(m.n_steps,4)
        self.assertEqual(len(m.reason_states(torch.randint(0,10,(2,81)))),5)

    def test_pinned_source_has_git_sha_and_blob_fingerprint(self):
        self.assertEqual(len(self.lock['revision']),40)
        self.assertEqual(len(self.lock['checkpoint_git_blob_sha1']),40)
        self.assertEqual(self.lock['checkpoint_size_bytes'],181147)
        self.assertEqual(self.lock['n_parameters'],44001)
        self.assertEqual(self.lock['upstream_license'],'MIT')
        self.assertIn('NOT official',self.lock['architecture'])

    def test_blob_hash_algorithm_and_corruption_rejection(self):
        data=b'canonical test blob'
        expect=hashlib.sha1(b'blob 19\x00'+data).hexdigest()
        self.assertEqual(blob_sha1(data),expect)
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'c.pt';f.write_bytes(data)
            lock={'checkpoint_size_bytes':len(data),'checkpoint_git_blob_sha1':expect,'checkpoint_sha256':hashlib.sha256(data).hexdigest()}
            self.assertEqual(locked_bytes(lock,f)[1],hashlib.sha256(data).hexdigest())
            f.write_bytes(data[:-1]+b'!')
            with self.assertRaisesRegex(ValueError,'SHA1 mismatch'):
                locked_bytes(lock,f)
            lock['checkpoint_git_blob_sha1']=blob_sha1(data[:-1]+b'!')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                locked_bytes(lock,f)

    def test_restricted_checkpoint_loading_and_strict_shapes(self):
        model=TinyRecursiveModel()
        buf=io.BytesIO();torch.save({'state_dict':model.state_dict(),'n_steps':4},buf)
        recovered=load_pretrained(self.lock,buf.getvalue())
        self.assertEqual(sum(p.numel() for p in recovered.parameters()),44001)
        buf=io.BytesIO();torch.save({'state_dict':{},'n_steps':4},buf)
        with self.assertRaises(RuntimeError):load_pretrained(self.lock,buf.getvalue())
        buf=io.BytesIO();torch.save({'state_dict':model.state_dict(),'n_steps':3},buf)
        with self.assertRaises(ValueError):load_pretrained(self.lock,buf.getvalue())

    def test_rational_tensor_norms_exact(self):
        a=torch.tensor([[0.5,-0.25],[1.0,-0.125]],dtype=torch.float32)
        b=torch.tensor([[0.0,-0.5],[1.0,-0.25]],dtype=torch.float32)
        self.assertEqual(row_sum_norm(a),Q(9,8))
        self.assertEqual(row_delta_norm(a,b),Q(3,4))
        self.assertEqual(infinity_tensor(a),Q(1))

    def test_layernorm_a_priori_hidden_state_envelope(self):
        m=TinyRecursiveModel().eval()
        a={k:p.detach() for k,p in m.named_parameters()}
        b={k:p.detach().to(torch.bfloat16).float() for k,p in m.named_parameters()}
        bound,e=layernorm_envelope(a,b,'block.ln2',Q(10),Q(0))
        x=torch.randn(2,81,48)*400
        z=m.block(x)
        self.assertLessEqual(float(z.detach().abs().max()),float(bound)+1e-4)
        self.assertGreaterEqual(e,0)
        _,dx=block_norm_and_storage_delta(a,b,Q(10))
        self.assertGreaterEqual(dx,0)

    def test_weight_storage_roundoff_bound_on_random_architecture(self):
        m=TinyRecursiveModel().eval()
        cert=certified_bounds(m)
        self.assertTrue(cert['strict_numerical_certificate'])
        self.assertEqual(cert['max_block_gain_linf']['log10_upper_bound_integer']>0,True)
        self.assertFalse(cert['block_gain_strict_contraction'])
        self.assertIn('not runtime roundoff',cert['formal_scope'])

    def test_generator_independent_and_unique(self):
        params=dict(self.lock,n_test_cases=4)
        (x,y,holes),r=prepare_roster(params)
        self.assertEqual(x.shape,(4,81))
        self.assertEqual(holes.sum().item(),120)
        self.assertNotEqual(r,'')
        for row in y:
            self.assertTrue(is_valid_solution(row.tolist()))
        self.assertTrue(torch.equal(x,prepare_roster(params)[0][0]))

    def test_cpu_damage_and_quantization_path(self):
        # Smoke only, random weights. The actual pretrained run is a separate CI job.
        params=dict(self.lock,n_test_cases=2,damage_seeds=[34],damage_amplitudes=['0.10'])
        (x,y,h),_=prepare_roster(params)
        m=TinyRecursiveModel().eval()
        q=evaluate(m,x,y,h,params)
        self.assertEqual(len(q['damage']),1)
        self.assertEqual(q['stages']['clean_depth_4']['n'],2)
        self.assertEqual(q['damage'][0]['n'],2)
        self.assertEqual(q['quantized_storage']['n'],2)
        self.assertGreater(q['damage'][0]['observed_hidden_linf_after_steps'][0],0)
        self.assertIn('read_stage1_instead_of_stage2_once',q['stale_full_state'])

    def test_no_hidden_target_used_during_recurrence(self):
        m=TinyRecursiveModel().eval()
        x=torch.randint(0,10,(2,81))
        states=m.reason_states(x)
        z=states[1]
        for step in (2,3,4):
            z=m.block(z)
            self.assertTrue(torch.equal(z,states[step]))


if __name__=='__main__':unittest.main()
