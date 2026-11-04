"""Pure-rational tests: mathematical quotient and independent certificate replay."""
from __future__ import annotations
import copy
import unittest
from fractions import Fraction as Q
from gauge_quotient import (Bound,center_rationals,safe_sqrt_rational,
                            layernorm_stat_bounds,normal_output_bounds,
                            certificate_for_case,replay_receipt)


def mock_inputs():
    z=[Q(0)]*48
    z[0]=Q(1,10); z[1]=-Q(1,10)
    case={'case_index':0,'selected_goal_cell':1,
          'winner_digit':2,'rival_digit':3,
          'u_intervals':[str(x) for x in z]}
    weight=[['0x0.0p+0']*48 for _ in range(10)]
    weight[2][0]='0x1.0000000000000p+0'
    gamma=['0x1.0000000000000p+0']*48
    zero='0x0.0p+0'
    beta=[zero]*48
    bias=[float(-10).hex()]*10
    bias[2]=(0.1).hex();bias[3]=zero
    weights={'gamma_hex':gamma,'beta_hex':beta,
             'epsilon_hex':(0.01).hex(),
             'head_weight_hex':weight,'head_bias_hex':bias}
    return case,weights


class TestGaugeQuotient(unittest.TestCase):
    def test_correlated_translations_vanish_exactly(self):
        x=[Q(2,7),Q(-5,19),Q(3,11),Q(8,13)]
        z=center_rationals(x)
        for t in (Q(0),Q(99999999999999999),Q(-1,10**40)):
            self.assertEqual(center_rationals([v+t for v in x]),z)
        self.assertEqual(sum(z),0)

    def test_rational_sqrt_enclosure_is_outward(self):
        for q in (Q(1,4),Q(2),Q(11,100),Q(10000001,10000000)):
            bound=safe_sqrt_rational(q,40)
            self.assertLessEqual(bound.lo*bound.lo,q)
            self.assertGreaterEqual(bound.hi*bound.hi,q)
            self.assertLessEqual(bound.hi-bound.lo,Q(2,10**40))
        with self.assertRaises(ValueError):safe_sqrt_rational(0)

    def test_simple_clean_class_and_noise(self):
        case,weights=mock_inputs()
        rec=certificate_for_case(case,weights,('1e-7','1e-3','0.05'))
        self.assertTrue(rec['clean_top1_validated'])
        self.assertTrue(rec['clean_top2_validated'])
        self.assertTrue(rec['transverse_radii']['1e-7']['top1_certified'])
        self.assertFalse(rec['transverse_radii']['0.05']['top1_certified'])
        self.assertGreater(Q(rec['sigma_lower_bound']),Q(1,10))

    def test_all_correlated_translations_do_not_change_certificate(self):
        case,weights=mock_inputs()
        a=certificate_for_case(case,weights,('1e-7',))
        case['arbitrary_common_mode_translation']='1e+1000000000'
        b=certificate_for_case(case,weights,('1e-7',))
        self.assertEqual(a,b)

    def test_rational_layernorm_contains_sampled_outputs(self):
        import random
        from decimal import Decimal,localcontext
        rng=random.Random(20261104)
        for _ in range(30):
            orig=[Q(rng.randint(-200,200),100) for _ in range(48)]
            uc=center_rationals(orig)
            hex_g=[float(Q(rng.randint(1,9),10)).hex() for _ in range(48)]
            hex_b=['0x0.0p+0']*48
            eps_hex=(0.01).hex()
            out,sigma=normal_output_bounds([str(u) for u in uc],hex_g,hex_b,eps_hex)
            with localcontext() as ctx:
                ctx.prec=90
                variance=sum((Decimal(z.numerator)/Decimal(z.denominator))**2 for z in uc)/48
                sigma_actual=(variance+Decimal.from_float(float.fromhex(eps_hex))).sqrt()
                self.assertLessEqual(Decimal(sigma.lo.numerator)/Decimal(sigma.lo.denominator),sigma_actual)
                self.assertGreaterEqual(Decimal(sigma.hi.numerator)/Decimal(sigma.hi.denominator),sigma_actual)
                for j in (0,17,47):
                    val=Decimal.from_float(float.fromhex(hex_g[j])) * (Decimal(uc[j].numerator)/Decimal(uc[j].denominator)) / sigma_actual
                    self.assertLessEqual(Decimal(out[j].lo.numerator)/Decimal(out[j].lo.denominator),val)
                    self.assertGreaterEqual(Decimal(out[j].hi.numerator)/Decimal(out[j].hi.denominator),val)

    def test_bad_eps_and_intervals_rejected(self):
        with self.assertRaises(ValueError):layernorm_stat_bounds(['0','0'],0)
        with self.assertRaises(ValueError):Bound(Q(2),Q(1))
        with self.assertRaises(ValueError):Bound(Q(-1),Q(1)).inverse()
        case,weights=mock_inputs()
        self.assertFalse(certificate_for_case(case,weights,('1',))['transverse_radii']['1']['top1_certified'])
        case['u_intervals']=['0']*47
        with self.assertRaises(ValueError):certificate_for_case(case,weights,('1e-5',))

    def test_source_and_claim_tampering_rejected(self):
        case,weights=mock_inputs()
        record=copy.deepcopy(case)
        record['certificate']=certificate_for_case(record,weights,('1e-6',))
        baseline={'roster_sha256':'b'*64,'rigorously_verified_rank_four_cases':64,
                  'cases':[{'case_index':i,'selected_goal_cell':1,'winner_digit':2,'rival_digit':3}
                           for i in range(64)]}
        lock={'revision':'deadbeef','checkpoint_sha256':'a'*64}
        receipt={'schema':'LSV-15/gauge-quotient-transverse-v1',
                 'source_revision':'deadbeef','checkpoint_sha256':'a'*64,
                 'roster_sha256':'b'*64, 'radii':['1e-6'],
                 'weights':weights,'cases':[record],
                 'counts':{'1e-6':{'top1':1,'top2':1}}}
        self.assertEqual(replay_receipt(receipt,lock,baseline,require_complete=False),receipt['counts'])
        bad=copy.deepcopy(receipt)
        bad['cases'][0]['certificate']['transverse_radii']['1e-6']['min_top1_slack']='999'
        with self.assertRaisesRegex(ValueError,'claimed certificate'):
            replay_receipt(bad,lock,baseline,require_complete=False)
        bad=copy.deepcopy(receipt)
        bad['cases'][0]['winner_digit']=3
        with self.assertRaisesRegex(ValueError,'selection'):
            replay_receipt(bad,lock,baseline,require_complete=False)
        bad=copy.deepcopy(receipt)
        bad['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source lock'):
            replay_receipt(bad,lock,baseline,require_complete=False)


if __name__=='__main__':unittest.main()
