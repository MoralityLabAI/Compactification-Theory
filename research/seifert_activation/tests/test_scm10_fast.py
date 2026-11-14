"""SCM-10 independent tests; optional NumPy keeps inherited minimal CI usable."""
from __future__ import annotations
import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

NUMPY_AVAILABLE=importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(NUMPY_AVAILABLE,'SCM10 CPU interval backend requires numpy')
class CertifiedNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import numpy as np
        import fast_centered as f
        self.np,self.f=np,f

    def test_ieee_preflight(self):
        self.assertTrue(self.f.ieee_preflight()['ieee_binary64_preflight_passed'])

    def test_frac_rounds_both_directions(self):
        for x in ('1/10','-1/10','1/3','-2/7','0','1/1099511627776'):
            n=Fraction(x)
            lo=self.f._from_fraction(n,upward=False)
            hi=self.f._from_fraction(n,upward=True)
            self.assertLessEqual(Fraction.from_float(lo),n)
            self.assertGreaterEqual(Fraction.from_float(hi),n)

    def test_interval_arithmetic_contains_fraction_arithmetic(self):
        rng=random.Random(2050)
        for _ in range(100):
            a=Fraction(rng.randrange(-800,800),100)
            b=Fraction(rng.randrange(-800,800),100)
            aa=self.f.ArrayI.point(float(a));bb=self.f.ArrayI.point(float(b))
            ac=Fraction.from_float(float(a));bc=Fraction.from_float(float(b))
            for op,value in ((aa+bb,ac+bc),(aa-bb,ac-bc),(aa*bb,ac*bc)):
                self.assertLessEqual(Fraction.from_float(float(op.lo)),value)
                self.assertGreaterEqual(Fraction.from_float(float(op.hi)),value)

    def test_gelu_and_cdf_mpmath_containment(self):
        import mpmath as mp
        f=self.f
        with mp.workdps(90):
            for x in (-4,-3.75,-2,-1,-.1,0,.1,1,2,3.9,4,5.1,-7.5):
                X=f.ArrayI.point(x)
                result=f.gelu_interval(X)
                real=mp.mpf(str(x))*(1+mp.erf(mp.mpf(str(x))/mp.sqrt(2)))/2
                self.assertLessEqual(mp.mpf(str(result.lo.item())),real)
                self.assertGreaterEqual(mp.mpf(str(result.hi.item())),real)

    def test_exp_mpmath(self):
        import mpmath as mp
        f=self.f
        for x in (-60.,-14.,-2.,-.2,-.001,0.):
            result=f.exp_interval(f.ArrayI.point(x))
            actual=mp.exp(mp.mpf(x))
            self.assertLessEqual(mp.mpf(str(result.lo.item())),actual)
            self.assertGreaterEqual(mp.mpf(str(result.hi.item())),actual)

    def test_fixed_dot_and_pair_dot_exact(self):
        import numpy as np
        rng=random.Random(8)
        f=self.f
        A=np.array([[rng.randrange(-64,65)/128 for _ in range(17)] for _ in range(5)])
        W=np.array([[rng.randrange(-64,65)/128 for _ in range(13)] for _ in range(17)])
        ball=f.ArrayI(f._down(A-1e-6),f._up(A+1e-6))
        result=f.dot_fixed(ball,W)
        for i in range(5):
            for j in range(13):
                true=sum((Fraction.from_float(float(A[i,k]))*Fraction.from_float(float(W[k,j]))
                          for k in range(17)),Fraction())
                self.assertLessEqual(Fraction.from_float(float(result.lo[i,j])),true)
                self.assertGreaterEqual(Fraction.from_float(float(result.hi[i,j])),true)
        pair=f.dot_pair(ball,ball)
        for i in range(5):
            for j in range(5):
                true=sum((Fraction.from_float(float(A[i,k]))*Fraction.from_float(float(A[j,k]))
                          for k in range(17)),Fraction())
                self.assertLessEqual(Fraction.from_float(float(pair.lo[i,j])),true)
                self.assertGreaterEqual(Fraction.from_float(float(pair.hi[i,j])),true)

    def test_simplex_perturbation_extremes(self):
        import numpy as np
        import mpmath as mp
        f=self.f
        rng=random.Random(10)
        with mp.workdps(80):
            for n in (1,2,3,8):
                z=np.array([[rng.uniform(-2,2) for _ in range(n)] for _ in range(n)])
                v=np.array([[rng.uniform(-1,1) for _ in range(2)] for _ in range(n)])
                zz=f.ArrayI(f._down(z-1e-4),f._up(z+1e-4))
                vv=f.ArrayI(f._down(v-1e-4),f._up(v+1e-4))
                result=f._simplex_weighted(zz,vv)
                for i in range(n):
                    zs=[mp.mpf(float(z[i,j]))+mp.mpf('0.00009') for j in range(i+1)]
                    weights=[mp.exp(t-max(zs)) for t in zs]
                    den=sum(weights)
                    for k in range(2):
                        actual=sum((weights[j]*(mp.mpf(float(v[j,k]))-mp.mpf('0.00009'))
                                    for j in range(i+1)))/den
                        self.assertLessEqual(mp.mpf(str(result.lo[i,k])),actual)
                        self.assertGreaterEqual(mp.mpf(str(result.hi[i,k])),actual)

    def test_exact_gram_bound_over_float_svd(self):
        import torch
        from ball_centered import exact_gram_spectral_upper
        gen=torch.Generator().manual_seed(42)
        for dims in ((4,9),(30,12),(128,64),(512,128)):
            x=torch.randint(-200,201,dims,generator=gen).float()/1024
            bound=exact_gram_spectral_upper(x)
            actual=torch.linalg.matrix_norm(x.double(),ord=2).item()
            self.assertGreaterEqual(float(bound),actual-1e-12)
            self.assertLess(float(bound),10*actual)

    def test_ball_matches_independent_90digit_forward(self):
        import torch
        import mpmath as mp
        from test_clean_tubes import ToyRMP
        from test_validated_forward import mp_clean_reference,mfrac,mf
        from exact_input import rmp_input_enclosure
        from ball_centered import verified_ball_tubes
        with mp.workdps(80):
            for tied,reinjection,seed in ((True,False,84),(False,True,85)):
                torch.manual_seed(seed)
                model=ToyRMP(tied=tied,visits=3,positions=False,beta=.1).eval()
                tokens=torch.tensor([1,2,3,4,5])
                source=rmp_input_enclosure(model,tokens)
                tube=verified_ball_tubes(model,source,fixed_input_reinjection=reinjection)
                ref=mp_clean_reference(model,tokens,reinjection=reinjection)
                for ctrs,errs,truth in zip((tube.pre_centers,tube.mid_centers,tube.post_centers),
                                            (tube.pre_center_errors,tube.mid_center_errors,tube.post_center_errors),ref):
                    for visit in range(3):
                        for token in range(5):
                            gap=mp.sqrt(sum((truth[visit][token][k]-mf(ctrs[visit][token,k]))**2
                                           for k in range(4)))
                            self.assertLessEqual(gap,mfrac(errs[visit][token]))

    def test_architecture_refuses_changed_mlp(self):
        import torch
        from test_clean_tubes import ToyRMP
        from exact_input import rmp_input_enclosure
        from ball_centered import verified_ball_tubes
        model=ToyRMP(positions=False).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3]))
        model.blocks[0].mlp[1]=torch.nn.ReLU()
        with self.assertRaisesRegex(ValueError,'GELU'):
            verified_ball_tubes(model,source)

    def test_read_only_audit_source_and_provenance(self):
        import torch
        from types import SimpleNamespace
        from test_clean_tubes import ToyRMP
        from scm10_audit import examine_record
        from rmp_capture import sha256_tensor
        model=ToyRMP(positions=False).eval()
        tokens=torch.tensor([1,2,3,4,5])
        chart=torch.eye(4,dtype=torch.float32)[:,:2]
        model_pack=SimpleNamespace(model=model,
            config=SimpleNamespace(input_injection=False))
        record={'model':'toy-source-only','checkpoint_sha256':'synthetic',
          'dictionary_sha256':'synthetic','task_id_sha256':'synthetic',
          'token_sequence_sha256':sha256_tensor(tokens),
          'chart_sha256':sha256_tensor(chart),
          'intervention_axes_sha256':sha256_tensor(chart),
          'capture':{'sequence_length':5,'injected_position':0,
               'injected_after_visit':0,'observed_token_positions':[1,2,3],
               'radius_in_token_rms':1e-7,
               'intervention_axes_sha256':sha256_tensor(chart)}}
        result=examine_record(record,model_pack,tokens,chart,chart)
        self.assertEqual(result['schema'],'scm10_rmp_ieee_cpu_source_ball_v1')
        self.assertFalse(result['floating_execution_certified'])
        self.assertFalse(result['observed_link_certified'])
        self.assertEqual(len(result['null_braid_ideal_real']),3)
        self.assertEqual(set(result['source_backend_sha256']),
                         {'fast_centered.py','ball_centered.py'})
        record['chart_sha256']='corrupted'
        with self.assertRaisesRegex(ValueError,'chart or injection-axis'):
            examine_record(record,model_pack,tokens,chart,chart)

    def test_fullshape_two_models(self):
        import torch
        from scm09_reference import DeterministicRMP
        from exact_input import rmp_input_enclosure
        from ball_centered import verified_ball_tubes
        for tied,reinjection,beta,cap in ((True,False,.1,.01),(False,True,.25,.005)):
            model=DeterministicRMP(d=128,seq=64,visits=8,tied=tied,
                                   reinjection=reinjection,beta=beta)
            source=rmp_input_enclosure(model,model.tokens,positional_frequency_table=
                torch.tensor([2.**(-k) for k in range(64)],dtype=torch.float32))
            tubes=verified_ball_tubes(model,source,fixed_input_reinjection=reinjection)
            self.assertEqual(len(tubes.post_centers),8)
            self.assertEqual(tuple(tubes.post_centers[-1].shape),(64,128))
            self.assertLess(float(max(tubes.post_center_errors[-1])),cap)
            self.assertFalse(tubes.floating_execution_certified)


if __name__=='__main__':
    unittest.main()
