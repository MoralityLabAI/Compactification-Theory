"""SCM-11 source-only intervention response tests, independent numerical witnesses."""
import math
import random
import sys
import unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import mpmath as mp
from refined_response import (HeadReference,attention_response_bound,
     reference_attention, refined_disk_bounds)
from gauge_null import gauge_axis_null_control
from exact_input import rmp_input_enclosure
from ball_centered import verified_ball_tubes
from validated_forward import verified_injection_amplitude,verified_centered_pair_bounds
from test_clean_tubes import ToyRMP
from test_validated_forward import mp_clean_reference,mfrac,mf,m_linear,m_ln


def mp_suffix(model, state, initial, *, after_visit, reinjection):
    """Independent high-precision recurrence from one post-visit patch."""
    x=[row[:] for row in state]
    res=[];d=model.hidden_size
    for index in model.module_indices[after_visit+1:]:
        block=model.blocks[index]
        z=[m_ln(block.norm1,row) for row in x]
        qkv=[m_linear(block.qkv.weight,row) for row in z]
        att=[];h=block.num_heads;dh=block.head_size
        for i in range(len(x)):
            heads=[]
            for k in range(h):
                q=qkv[i][k*dh:(k+1)*dh]
                score=[sum((a*b for a,b in zip(q,qkv[j][d+k*dh:d+(k+1)*dh])),mp.mpf(0))/mp.sqrt(dh)
                       for j in range(i+1)]
                weight=[mp.exp(s-max(score)) for s in score]
                den=sum(weight,mp.mpf(0))
                for component in range(dh):
                    heads.append(sum((weight[j]*qkv[j][2*d+k*dh+component] for j in range(i+1)),mp.mpf(0))/den)
            att.append(m_linear(block.attention_out.weight,heads))
        alpha,beta=mf(block.alpha),mf(block.beta)
        middle=[[alpha*xij+beta*aij for xij,aij in zip(row,attrow)] for row,attrow in zip(x,att)]
        n2=[m_ln(block.norm2,row) for row in middle]
        gelu=[[w*(1+mp.erf(w/mp.sqrt(2)))/2 for w in m_linear(block.mlp[0].weight,row)] for row in n2]
        updates=[m_linear(block.mlp[2].weight,row) for row in gelu]
        x=[[alpha*mij+beta*uij for mij,uij in zip(row,update)] for row,update in zip(middle,updates)]
        if reinjection:
            x=[[v+u for v,u in zip(row,inp)] for row,inp in zip(x,initial)]
        res.append(x)
    return res


class AttentionLemmaTests(unittest.TestCase):
    def test_identical_values_remove_score_sensitivity(self):
        # All exact source V_j are the same vector: probability changes sum to 0.
        r=HeadReference((Fraction(2),)*3,(Fraction(3),)*3,(Fraction(30),)*3,
                       (Fraction(0),)*3,((Fraction(1),),(Fraction(1),Fraction(1)),
                       (Fraction(1),)*3))
        v=attention_response_bound(r,prefix=2,delta_query=Fraction(2),
             delta_keys=[Fraction(2)]*3,delta_values=[Fraction(0)]*3,head_dim=2)
        self.assertEqual(v['bound'],0)
        self.assertGreater(v['score_response'],0)

    def test_sparse_late_token_attention_weight_is_used(self):
        # Lower source upper bounds on p3 permit a sharper influence bound.
        r=HeadReference((Fraction(1),)*3,(Fraction(1),)*3,(Fraction(1),)*3,
             (Fraction(0),)*3,((Fraction(1),),(Fraction(1),Fraction(1)),
                                (Fraction(1,2),Fraction(1,2),Fraction(1,100))))
        v=attention_response_bound(r,prefix=2,delta_query=Fraction(0),
             delta_keys=[Fraction(0)]*3,delta_values=[Fraction(0),Fraction(0),Fraction(1)],
             head_dim=1)
        self.assertEqual(v['bound'],Fraction(1,100))

    def test_random_softmax_value_ball_by_high_precision(self):
        rng=random.Random(271)
        with mp.workdps(85):
            for _ in range(60):
                scores=[mp.mpf(rng.randrange(-500,501))/500 for j in range(3)]
                values=[mp.mpf(rng.randrange(-500,501))/100 for j in range(3)]
                dv=[mp.mpf(rng.randrange(-100,101))/500 for j in range(3)]
                ds=[mp.mpf(rng.randrange(-100,101))/500 for j in range(3)]
                exp0=[mp.exp(x) for x in scores];exp1=[mp.exp(x+y) for x,y in zip(scores,ds)]
                p0=[x/sum(exp0) for x in exp0];p1=[x/sum(exp1) for x in exp1]
                true=abs(sum((p1[j]*(values[j]+dv[j])-p0[j]*values[j] for j in range(3))))
                d=max(abs(x) for x in ds)
                b=max(abs(x) for x in dv)
                # Main lemma with the true baseline and a valid origin anchor.
                radius=max(abs(x) for x in values)
                coarse=b+d*radius
                self.assertLessEqual(true,coarse+mp.mpf('1e-75'))
                # Reference softmax expectation and TV correction.
                var=[abs(x) for x in dv]
                improved=min(b,sum((p0[j]*var[j] for j in range(3)))+
                             d*(max(var)-min(var))/2)+d*radius
                self.assertLessEqual(true,improved+mp.mpf('1e-75'))

    def test_source_probability_enclosures_against_90digit_softmax(self):
        with mp.workdps(90):
            torch.manual_seed(109)
            model=ToyRMP(tied=True,visits=2,positions=False).eval()
            src=rmp_input_enclosure(model,torch.tensor([1,2,3,4]))
            center=src.center
            headrefs=reference_attention(model.blocks[0],center)
            b=model.blocks[0]
            normed=[m_ln(b.norm1,[mf(x) for x in row]) for row in center]
            qkv=[m_linear(b.qkv.weight,row) for row in normed]
            d=model.hidden_size;dh=b.head_size
            for head,ref in enumerate(headrefs):
                for i in range(len(center)):
                    q=qkv[i][head*dh:(head+1)*dh]
                    scores=[]
                    for j in range(i+1):
                        key=qkv[j][d+head*dh:d+(head+1)*dh]
                        scores.append(sum((x*y for x,y in zip(q,key)),mp.mpf(0))/mp.sqrt(dh))
                    exps=[mp.exp(x-max(scores)) for x in scores]
                    den=sum(exps,mp.mpf(0))
                    for j,pv in enumerate(exps):
                        self.assertLessEqual(pv/den,mfrac(ref.probability_upper[i][j]))

    def test_reference_probabilities_and_anchor(self):
        model=ToyRMP(tied=True,visits=2,positions=False).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4]))
        refs=reference_attention(model.blocks[0],source.center)
        self.assertEqual(len(refs),model.blocks[0].num_heads)
        for head in refs:
            for i,p in enumerate(head.probability_upper):
                self.assertEqual(len(p),i+1)
                self.assertGreaterEqual(sum(p),1)
                self.assertEqual(head.anchor_radii[0],0)
                self.assertLessEqual(head.anchor_radii[-1],max(head.vnorm))


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _case(self,*,tied=True,reinjection=False,seed=101):
        torch.manual_seed(seed)
        model=ToyRMP(tied=tied,visits=3,positions=False,beta=.1).eval()
        tokens=torch.tensor([1,2,3,4,5]);s=rmp_input_enclosure(model,tokens)
        tubes=verified_ball_tubes(model,s,fixed_input_reinjection=reinjection)
        chart=torch.eye(4,dtype=torch.float32)[:,:2]
        amp=Fraction(1,10000)
        output=refined_disk_bounds(model,tubes,chart=chart,axes=chart,amplitude=amp,
               observed_tokens=(1,2,3),injected_token=0)
        return model,tokens,tubes,chart,amp,output

    def test_refinement_improves_legacy_small(self):
        for tied,reinject in ((True,False),(False,True)):
            model,tokens,tubes,chart,amp,refined=self._case(tied=tied,reinjection=reinject)
            old=verified_centered_pair_bounds(model,tubes,chart=chart,axes=chart,amplitude=amp,
                              observed_tokens=(1,2,3),injected_token=0)
            for oldrow,newrow in zip(old['per_visit_pair_bounds'],refined['visit_pair_bounds']):
                for pair in oldrow:
                    self.assertLessEqual(Fraction(newrow[pair]),Fraction(oldrow[pair]))
            self.assertTrue(all(x['null_braid_ideal_real'] for x in refined['null_braid_ideal_real']))

    def test_mp_reference_contains_intervention(self):
        with mp.workdps(80):
            for tied,reinj in ((True,False),(False,True)):
                model,tokens,tubes,chart,amp,result=self._case(tied=tied,reinjection=reinj)
                pre,mid,post=mp_clean_reference(model,tokens,reinjection=reinj)
                x=[row[:] for row in post[0]]
                x[0][0]+=mfrac(amp)
                after=mp_suffix(model,x,pre[0],after_visit=0,reinjection=reinj)
                for visit,pstate in enumerate(after, start=1):
                    for token in range(len(tokens)):
                        gap=mp.sqrt(sum(((pstate[token][c]-post[visit][token][c])**2 for c in range(4)),mp.mpf(0)))
                        upper=mfrac(Fraction(result['visit_response_radii'][visit][token]))
                        self.assertLessEqual(gap,upper, f'{tied=} {reinj=} {visit=} {token=}')

    def test_causal_zero_and_no_cuda_claim(self):
        model,tokens,tubes,chart,amp,result=self._case()
        self.assertTrue(all(v=='0' for v in result['visit_pair_bounds'][0].values()))
        self.assertFalse(result['floating_execution_certified'])
        self.assertFalse(result['observed_link_certified'])

    def test_rejects_negative_and_non_eval_and_source_scope(self):
        model,tokens,tubes,chart,amp,result=self._case()
        args=dict(chart=chart,axes=chart,amplitude=-1,observed_tokens=(1,2,3),injected_token=0)
        with self.assertRaisesRegex(ValueError,'negative'):
            refined_disk_bounds(model,tubes,**args)
        model.train()
        args['amplitude']=amp
        with self.assertRaisesRegex(ValueError,'eval'):
            refined_disk_bounds(model,tubes,**args)
        model.eval()
        from dataclasses import replace
        bad=replace(tubes,input_scope='unverified_activation_samples')
        with self.assertRaisesRegex(ValueError,'source-locked'):
            refined_disk_bounds(model,bad,**args)


class GaugeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        self.model=ToyRMP(visits=3,positions=False,beta=.25).eval()

    def test_exact_shift_equivariance_each_block(self):
        torch.set_num_threads(1)
        b=self.model.blocks[0].double()
        x=torch.randn(1,5,4,dtype=torch.float64)
        shifts=torch.tensor([0.7,-0.3,4.,0.1,-2.],dtype=torch.float64).reshape(1,5,1)
        expected=b(x)+b.alpha**2*shifts.expand_as(x)
        self.assertLess(float((b(x+shifts)-expected).abs().max().detach()),1e-9)

    def test_all_downstream_gauge_patch_invisible(self):
        model=self.model.double()
        x=torch.randn(1,5,4,dtype=torch.float64)
        clean=x.clone();mod=x.clone();mod[0,0,:]+=17.
        for idx in model.module_indices:
            clean=model.blocks[idx](clean)
            mod=model.blocks[idx](mod)
        self.assertLess(float((clean[0,1:]-mod[0,1:]).abs().max().detach()),1e-8)

    def test_source_null_control_and_refusals(self):
        axes=torch.stack((torch.ones(4),torch.arange(4,dtype=torch.float32)),dim=1)
        report=gauge_axis_null_control(self.model,axes,injected_token=0,
                                     observed_tokens=[1,2,3],sequence=5)
        self.assertEqual(report['invisible_axis'],0)
        self.assertFalse(report['collision_free_interval_certified'])
        axes[2,0]=1.0000001192092896
        with self.assertRaisesRegex(ValueError,'exactly one'):
            gauge_axis_null_control(self.model,axes,injected_token=0,
                                     observed_tokens=[1,2,3],sequence=5)


class SourceAuditTests(unittest.TestCase):
    def test_checked_receipt_and_hash_refusals(self):
        from types import SimpleNamespace
        from scm11_audit import examine_record
        from rmp_capture import sha256_tensor
        m=ToyRMP(visits=3,positions=False).eval()
        tokens=torch.tensor([1,2,3,4,5])
        chart=torch.eye(4,dtype=torch.float32)[:,:2]
        record={'model':'synthetic_mini','task_id_sha256':'synthetic',
           'checkpoint_sha256':'synthetic','dictionary_sha256':'synthetic',
           'token_sequence_sha256':sha256_tensor(tokens),
           'chart_sha256':sha256_tensor(chart),
           'intervention_axes_sha256':sha256_tensor(chart),
           'capture':{'sequence_length':5,'injected_position':0,
             'injected_after_visit':0,'observed_token_positions':[1,2,3],
             'radius_in_token_rms':1e-7,
             'intervention_axes_sha256':sha256_tensor(chart)}}
        analysis=SimpleNamespace(model=m,config=SimpleNamespace(input_injection=False))
        result=examine_record(record,analysis,tokens,chart,chart)
        self.assertEqual(result['schema'],'scm11_anchored_attention_source_disk_v1')
        self.assertFalse(result['physical_float_execution_certified'])
        self.assertFalse(result['observed_link_certified'])
        self.assertEqual(len(result['result']['null_braid_ideal_real']),3)
        self.assertEqual(set(result['source_backend_sha256']),
                 {'fast_centered.py','ball_centered.py','refined_response.py','gauge_null.py'})
        record['chart_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'chart or intervention'):
            examine_record(record,analysis,tokens,chart,chart)

    def test_refusal_missing_external_source_checkout(self):
        import tempfile
        from pathlib import Path
        from scm11_audit import run
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); inp=p/'captures.jsonl';out=p/'report.jsonl'
            inp.write_text('{"model":"LOOP-T s1"}\n')
            with self.assertRaisesRegex(FileNotFoundError,'RMP source'):
                run(inp,out,rmp_root=p/'missing',runs=p/'runs',artifacts=p/'arts',
                    dictionary=p/'dict.pt',monitor_bundle=p/'bundle')
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
