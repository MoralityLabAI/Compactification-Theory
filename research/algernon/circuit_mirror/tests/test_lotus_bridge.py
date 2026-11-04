"""CPU-only tests. All neural features here are fake or manufactured test fixtures.

A passing test never demonstrates a published LOTUS checkpoint running.
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # optional CPU neural tests on GitHub Actions without torch/numpy
    np = None
    torch = None

from research.algernon.circuit_mirror.lotus_bridge.roster import (
    build_roster, validate_roster, prompt_for,
)
from research.algernon.circuit_mirror.lotus_bridge.capture import (
    capture_one, intercept_loop_states, select_hidden, resolve_revision, file_sha256,
)
from research.algernon.circuit_mirror.lotus_bridge.probe import (
    evaluate_probe, group_metrics, verify_capture,
)


class FixtureModel:
    start_latent_id=10
    latent_token_id=11
    end_latent_id=12

    def __init__(self, module):
        self.base_causallm=torch.nn.Linear(2,2)
        self.module=module

    def forward(self, input_ids, attention_mask, labels, position_ids, n_looped_iters=0):
        assert torch.all(labels == -100)
        ix=(input_ids[0] == 11).nonzero().flatten()
        start=max(0,int(ix[0])-1)
        end=int(ix[-1])+1
        hidden = torch.arange((end-start)*7,dtype=torch.float32).reshape(1,end-start,7)
        for k in range(n_looped_iters+1):
            out = self.module.efficient_forward(hidden+k*100.0)
            assert out.logits.shape[-1] == 1  # large intermediate logits suppressed
        return out


class ToyTokenizer:
    def encode(self, question, add_special_tokens=True):
        return [6,7,8]


class LotusBridgeTests(unittest.TestCase):
    def test_roster_hash_and_split_disjointness(self):
        r = build_roster(train=25, calibration=11, test_per_family=12)
        self.assertEqual(validate_roster(r),60)
        self.assertEqual(len({(tuple(x['successors']),x['initial']) for x in r['cases']}),60)
        self.assertEqual({x['family'] for x in r['cases'] if x['split']=='test'}, {'long_cycle','sink'})
        bad=json.loads(json.dumps(r));bad['cases'][0]['trace'][-1]=0 if bad['cases'][0]['trace'][-1]!=0 else 1
        with self.assertRaises(ValueError):validate_roster(bad)
        question=prompt_for(r['cases'][0])
        self.assertIn('follow the successor',question)
        self.assertNotIn('trace',question)

    @unittest.skipIf(torch is None, "optional torch/numpy unavailable")
    def test_boundary_for_both_cache_paths(self):
        # Last latent at absolute position 6, prefix starts at 3.
        tensor=torch.arange(8*7,dtype=torch.float32).reshape(1,8,7)
        a=select_hidden(tensor[:,3:8,:],first_latent=4,last_latent=6,
                        loop_start=3,loop_end=8,c_thought=2)
        b=select_hidden(tensor,first_latent=4,last_latent=6,
                        loop_start=3,loop_end=8,c_thought=2)
        self.assertTrue(torch.allclose(a,b))
        self.assertTrue(torch.equal(a[0],tensor[0,6]))
        self.assertTrue(torch.equal(a[1],tensor[0,5:7].mean(0)))
        with self.assertRaises(ValueError):select_hidden(tensor[:,:7,:],first_latent=4,
                         last_latent=6,loop_start=3,loop_end=8,c_thought=2)

    @unittest.skipIf(torch is None, "optional torch/numpy unavailable")
    def test_real_capture_hook_contract_with_stub(self):
        # Traces are strictly from the fake efficient_forward: no LOTUS model loaded.
        module=types.SimpleNamespace()
        def efficient(hidden):
            return types.SimpleNamespace(last_hidden_state=hidden,logits=torch.ones((*hidden.shape[:2],1000)))
        module.efficient_forward=efficient
        model=FixtureModel(module)
        result,meta=capture_one(model,module,ToyTokenizer(),'dummy',loops=3,c_thought=2)
        self.assertEqual(result.shape,(4,2,7))
        self.assertEqual(meta['passes_captured'],4)
        np.testing.assert_allclose(result[1,0]-result[0,0],100.0)
        self.assertIs(module.efficient_forward,efficient)
        with self.assertRaises(ValueError):
            with intercept_loop_states(module,first_latent=4,last_latent=5,c_thought=1):
                raise ValueError('forced abort')
        self.assertIs(module.efficient_forward,efficient)

    def test_offline_revision_requires_explicit_pin(self):
        with self.assertRaises(ValueError):resolve_revision('model',None,local_only=True)
        with self.assertRaises(ValueError):resolve_revision('model','main',local_only=True)
        self.assertEqual(resolve_revision('model','f'*40,local_only=True),'f'*40)

    @unittest.skipIf(torch is None, "optional torch/numpy unavailable")
    def test_manifest_refuses_changed_feature_bytes(self):
        roster=build_roster(train=7,calibration=5,test_per_family=4)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rr=root/'roster.json';rr.write_text(json.dumps(roster))
            feats=root/'hidden.npz'
            np.savez_compressed(feats,features=np.zeros((20,17,2,8),dtype=np.float32),
                                ids=np.array([x['id'] for x in roster['cases']]))
            manifest={'roster_sha256':roster['roster_sha256'],
                      'feature_sha256':file_sha256(feats),'complete_roster':True,
                      'feature_shape':[20,17,2,8]}
            feats.with_suffix('.manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_capture(feats,rr)[2].shape,(20,17,2,8))
            manifest['feature_sha256']='f'*64
            feats.with_suffix('.manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_capture(feats,rr)

    @unittest.skipIf(torch is None, "optional torch/numpy unavailable")
    def test_probe_scoring_on_oracle_encoded_fixture(self):
        # Manufactured states for a plumbing check. This must never enter results.
        roster=build_roster(train=40,calibration=12,test_per_family=10)
        c=roster['cases']
        features=np.zeros((len(c),17,2,16),dtype=np.float32)
        for i,row in enumerate(c):
            for t,state in enumerate(row['trace']):
                features[i,t,0,state]=3.0
                features[i,t,1,state]=2.0
        result,_=evaluate_probe(roster,features,seed=17,updates=45,lr=0.06)
        self.assertEqual(result['metrics']['all_test']['accuracy_step_16'],1.0)
        self.assertEqual(result['metrics']['all_test']['exact_trace_valid_full_rate'],1.0)
        self.assertGreater(result['metrics']['all_test']['accuracy_step_16'],
                           result['metrics']['constant_stage0_control']['accuracy_step_16'])
        self.assertLess(result['metrics']['all_test']['commuting_square_defect_rate'],0.001)

    @unittest.skipIf(torch is None, "optional torch/numpy unavailable")
    def test_verifier_rejects_bad_trace(self):
        roster=build_roster(train=6,calibration=4,test_per_family=3)
        row=roster['cases'][-1]
        correct=np.array([row['trace']],dtype=np.int64)
        got=group_metrics(correct,[row],16)
        self.assertEqual(got['exact_trace_valid_full_rate'],1.0)
        wrong=correct.copy();wrong[0,-1]=(wrong[0,-1]+1)%6
        got=group_metrics(wrong,[row],16)
        self.assertEqual(got['exact_trace_valid_full_rate'],0.0)

if __name__=='__main__':unittest.main()
