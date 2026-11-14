"""SCM-12 exact rank-one topology and paired RMP campaign controls."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scm12_rank_one import rank_one_null_certificate
from scm12_compare import compare_files, _exact_two_sided_sign_test, MODELS, CHARTS
from scm12_launch import preflight,freeze,_select_dictionary,MODEL_CELLS


class RankOneTests(unittest.TestCase):
    def setUp(self):
        self.base=[('0','0'),('3','0'),('0','3')]
        self.slopes=[('0','0'),('1/10','0'),('0','1/10')]
        self.bounds=[0,0,0]

    def test_exact_line_factor_is_null(self):
        r=rank_one_null_certificate(self.base,self.slopes,(-1,1),self.bounds)
        self.assertEqual(r['result'],'conditional_null_braid')
        self.assertTrue(all(item['pass'] for item in r['per_pair'].values()))
        self.assertFalse(r['error_bounds_source_verified'])

    def test_small_certified_remainder(self):
        r=rank_one_null_certificate(self.base,self.slopes,(-1,1),['1/10']*3,margin='1/100')
        self.assertEqual(r['result'],'conditional_null_braid')

    def test_source_claim_absent_stays_uncertified(self):
        r=rank_one_null_certificate(self.base,self.slopes,(-1,1),[0]*3)
        self.assertFalse(r['observed_neural_topology_certified'])

    def test_approaching_collision_refused(self):
        r=rank_one_null_certificate([[0,0],[1,0]],[[0,0],[-1,0]],(0,1),[0,0])
        self.assertEqual(r['result'],'inconclusive')

    def test_large_error_refused(self):
        r=rank_one_null_certificate(self.base,self.slopes,(-1,1),[2,2,2])
        self.assertEqual(r['result'],'inconclusive')

    def test_constant_response_allowed(self):
        r=rank_one_null_certificate([[0,0],[1,0]],[[0,0],[0,0]],(0,0),[0,0])
        self.assertEqual(r['result'],'conditional_null_braid')

    def test_rational_clearance_strict_equality_refused(self):
        r=rank_one_null_certificate([[0,0],[1,0]],[[0,0],[0,0]],(0,1),['1/2','1/2'])
        self.assertEqual(r['result'],'inconclusive')

    def test_bad_inputs_rejected(self):
        cases=[([[0,0]],[[0,0]],(0,1),[0]),
               ([[0,0],[1,0]],[[0,0]],(0,1),[0,0]),
               ([[0,0],[1,0]],[[0,0],[0,0]],(1,0),[0,0]),
               ([[0,0],[1,0]],[[0,0],[0,0]],(0,1),[0,-1]),
               ([[0,0],[1,0]],[[0,0],[0,0]],(0,1),[0,0])]
        for a,b,interval,error in cases[:-1]:
            with self.assertRaises(ValueError):rank_one_null_certificate(a,b,interval,error)
        with self.assertRaises(ValueError):rank_one_null_certificate(*cases[-1],margin=-1)


class GaugeFactorizationTests(unittest.TestCase):
    def test_one_gauge_axis_reduces_downstream_circle_to_interval(self):
        import math
        import torch
        from test_clean_tubes import ToyRMP
        torch.manual_seed(150)
        torch.set_num_threads(1)
        for tied,reinjection in ((True,False),(False,True)):
            model=ToyRMP(tied=tied,positions=False,visits=3,beta=.1).double().eval()
            tokens=torch.tensor([[1,2,3,4,5]])
            base=model.embed(tokens)
            gauge=torch.ones(4,dtype=torch.float64)/2
            transverse=torch.tensor([1.,-1.,0.,0.],dtype=torch.float64)
            def trace(theta):
                x=base.clone();out=[]
                for step,index in enumerate(model.module_indices):
                    x=model.blocks[index](x)
                    if step==0:
                        x=x.clone()
                        x[0,0]+=0.04*(math.cos(theta)*gauge+math.sin(theta)*transverse)
                    if reinjection:
                        x=x+base
                    out.append(x[0,1:].clone())
                return out
            a,b=trace(math.pi/6),trace(5*math.pi/6)
            for x,y in zip(a,b):
                self.assertLess(float((x-y).abs().max()),2e-12)



def record(task,model,chart,*,sign=False,nontriv=False):
    return {'schema':'scm04/rmp-cyclic/v3', 'source_type':'synthetic_fixture',
      'model':model,'chart_control':chart,'task_id_sha256':f'task_{task:010}',
      'group_id_sha256':f'group_{task:010}',
      'family':('pointer_chase','gated_pointer_chase','reachability','sudoku4')[task%4],
      'target':1, 'checkpoint_sha256':f'checkpoint-{model}',
      'dictionary_sha256':f'dict-{model}', 'intervention_axes_sha256':f'axes-{model}',
      'token_sequence_sha256':f'tokens-{task}','features':[4,7],
      'chart_sha256':f'chart-{model}-{chart}',
      'capture':{'predictions':[0,0,0], 'baseline_prediction':0,
                 'changed_prediction_fraction':0.,'max_abs_logit_change_from_clean':0.,
                 'intervention_axes_sha256':f'axes-{model}'},
      'topology':[{'visit':1,'valid':True, 'artin':{'status':'exact','nonidentity':nontriv}}],
    }


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.dir=Path(self.t.name)
        self.fixture={}
        for model in MODELS:
            for chart in CHARTS:
                self.fixture[model,chart]=[record(i,model,chart,nontriv=(chart=='sae' and i in (1,2))) for i in range(4)]
        self.files={key:self.dir/(('loop' if model==MODELS[0] else 'ffu')+'_'+chart+'.jsonl')
                    for key in self.fixture for model,chart in [key]}
        self.write()

    def write(self):
        for key,rows in self.fixture.items():
            self.files[key].write_text('\n'.join(json.dumps(x) for x in rows)+'\n')

    def test_matched_synthetic_report(self):
        report=compare_files(self.files,expected_tasks=4)
        self.assertEqual(report['n_matched_tasks'],4)
        self.assertEqual(report['source_classification'],'synthetic_only')
        self.assertEqual(report['models'][MODELS[0]]['n_sae_nontrivial'],2)
        self.assertEqual(report['models'][MODELS[0]]['n_sae_only'],2)
        self.assertEqual(report['models'][MODELS[0]]['exact_sign_test_p_descriptive'],.5)

    def test_refuse_partial_registered_64(self):
        with self.assertRaisesRegex(ValueError,'exactly 64'):
            compare_files(self.files)

    def test_refuse_changed_chart_logits(self):
        self.fixture[MODELS[0],CHARTS[1]][1]['capture']['predictions']=[0,1,0]
        self.write()
        with self.assertRaisesRegex(ValueError,'Changing readout chart'):
            compare_files(self.files,expected_tasks=4)

    def test_refuse_changed_intervention(self):
        self.fixture[MODELS[0],CHARTS[1]][0]['intervention_axes_sha256']='changed'
        self.write()
        with self.assertRaisesRegex(ValueError,'intervention differs'):
            compare_files(self.files,expected_tasks=4)

    def test_refuse_cross_model_group_mismatch(self):
        self.fixture[MODELS[1],CHARTS[1]][0]['group_id_sha256']='wrong'
        self.write()
        with self.assertRaisesRegex(ValueError,'group'):
            compare_files(self.files,expected_tasks=4)

    def test_refuse_overlapping_group(self):
        for key in self.fixture:
            self.fixture[key][1]['group_id_sha256']=self.fixture[key][0]['group_id_sha256']
        self.write()
        with self.assertRaisesRegex(ValueError,'share a group'):
            compare_files(self.files,expected_tasks=4)

    def test_refuse_missing_task(self):
        self.fixture[MODELS[1],CHARTS[1]].pop()
        self.write()
        with self.assertRaisesRegex(ValueError,'paired task IDs'):
            compare_files(self.files,expected_tasks=4)

    def test_refuse_unknown_source(self):
        self.fixture[MODELS[0],CHARTS[0]][0]['source_type']='unverified'
        self.write()
        with self.assertRaisesRegex(ValueError,'Unknown source'):
            compare_files(self.files,expected_tasks=4)

    def test_artin_budget_is_unresolved(self):
        self.fixture[MODELS[0],CHARTS[0]][0]['topology'][0]['artin']={'status':'budget_exceeded'}
        self.write()
        r=compare_files(self.files,expected_tasks=4)
        self.assertEqual(r['models'][MODELS[0]]['n_unresolved_sae'],1)

    def test_sign_exact(self):
        self.assertEqual(_exact_two_sided_sign_test(0,0),1)
        self.assertEqual(_exact_two_sided_sign_test(5,0),.0625)
        self.assertEqual(_exact_two_sided_sign_test(2,2),1)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        p=Path(self.t.name)
        self.args=dict(rmp_root=p/'rmp',runs=p/'runs',artifacts=p/'artifacts',
             bundle=p/'bundle',metrics=p/'metrics.json',
             dictionaries={model:p/(model.replace(' ','_')+'.pt') for model in MODELS},
             features={model:(3,7) for model in MODELS},
             feature_provenance=p/'feature_link.json')

    def test_missing_artifacts_reported(self):
        report=preflight(**self.args)
        self.assertEqual(report['status'],'blocked_external_artifacts')
        self.assertIn('LOOP-T s1/checkpoint',report['missing'])
        with self.assertRaisesRegex(ValueError,'Refusing'):
            freeze(report,Path(self.t.name)/'lock.json')

    def test_missing_feature_pair_reported(self):
        self.args['features'][MODELS[0]]=None
        report=preflight(**self.args)
        self.assertIn('LOOP-T s1/frozen_feature_pair',report['missing'])

    def test_empty_dictionaries_grid_fails(self):
        p=self.args['metrics'];p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Missing dictionary metrics'):
            preflight(**self.args)

    def test_selection_matches_rmp_rule(self):
        grid={'LOOP-T s1':{
          'a':{'config':{'kind':'shared','seed':0,'site':-1,'width':1024,'k':8},
               'splice':{'min_ce_recovered':.90}},
          'b':{'config':{'kind':'shared','seed':0,'site':-1,'width':2048,'k':8},
               'splice':{'min_ce_recovered':.96}},
          'c':{'config':{'kind':'shared','seed':0,'site':-1,'width':4096,'k':8},
               'splice':{'min_ce_recovered':.97}}}}
        self.assertEqual(_select_dictionary(grid,'LOOP-T s1'),('b',True))

    def test_registered_lock_and_mutation_refusal(self):
        a=self.args
        for name in ('rmp_root','runs','artifacts','bundle'):
            a[name].mkdir(parents=True,exist_ok=True)
        for name in ('metrics','feature_provenance'):
            a[name].write_text('{}')
        configs={}
        for model in MODELS:
            cell,stage=MODEL_CELLS[model]
            path=a['artifacts']/cell/'checkpoints'/f'{stage}.pt'
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'FIXTURE')
            run=a['runs']/cell/'result.json'
            run.parent.mkdir(parents=True,exist_ok=True);run.write_text('{}')
            import torch
            torch.save({'shape':{'dim':128,'width':1024,'k':8,'skip':False},
              'summary':{'config':{'kind':'shared','seed':0,'site':-1,'width':1024,'k':8}},
              'state':{'decoder':torch.zeros(1024,128)}},a['dictionaries'][model])
            (path.with_name(path.name+'.receipt.json')).write_text('{}')
            configs[model]={'selected':{'config':{'kind':'shared','seed':0,'site':-1,'width':1024,'k':8},
              'splice':{'min_ce_recovered':.95}}}
        a['metrics'].write_text(json.dumps(configs))
        from scm12_launch import PINNED_SOURCE,launch
        for path in PINNED_SOURCE:
            f=a['rmp_root']/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture')
        report=preflight(**a)
        self.assertEqual(report['status'],'ready_for_external_RMP_capture')
        lock=Path(self.t.name)/'lock.json'
        freeze(report,lock)
        self.assertTrue(lock.exists())
        with self.assertRaises(FileExistsError):freeze(report,lock)
        a['dictionaries'][MODELS[0]].write_bytes(b'MUTATED')
        with self.assertRaisesRegex(ValueError,'identity has changed'):
            launch(lock,Path(self.t.name)/'out')

if __name__=='__main__':unittest.main()
