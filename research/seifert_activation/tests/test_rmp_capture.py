"""CPU mock validates the exact runtime VisitHook + output contract, not trained RMP."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
try:
    import torch
except ImportError:  # Optional for the rest of Compactification-Theory's exact CPU tests.
    torch = None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knot import InvalidBraidTrace
if torch is not None:
    from rmp_capture import (capture_cycle, analyze_cycle, chart_from_decoder,
                             compare_double_resolution, sha256_tensor)


class MockRMP:
    def run(self,tokens,hook=None):
        state=torch.zeros((1,tokens.shape[1],2),dtype=torch.float32,device=tokens.device)
        state[0,0]=torch.tensor([1.,1.])
        state[0,1]=torch.tensor([0.,0.])
        state[0,2]=torch.tensor([0.4,0.0])
        outputs=[]
        for k in range(4):
            after=state.clone()
            if k>=1:
                after[0,1]=state[0,0]-torch.tensor([1.,1.])
                after[0,2]=torch.tensor([0.4,0.0])
            context=SimpleNamespace(visit=k,after=after,before=state)
            if hook is not None:
                patched=hook(context)
                if patched is not None:
                    after=patched
            outputs.append(after)
            state=after
        logits=torch.zeros((1,tokens.shape[1],3))
        logits[:,:,-1]=0.1
        logits[:,:,0]=2
        return SimpleNamespace(logits=logits,visit_outputs=tuple(outputs))


@unittest.skipIf(torch is None, "torch is optional for SCM-02 RMP adapter tests")
class RMPAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=MockRMP()
        cls.tokens=torch.tensor([[0,1,2,3]],dtype=torch.long)
        cls.chart=torch.eye(2)

    def test_mock_visit_hook_contract(self):
        trace=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                             observed_tokens=[1,2],radius=1.,angles=64)
        results=analyze_cycle(trace)
        self.assertEqual(len(results),4)
        self.assertTrue(all(item['valid'] for item in results))
        self.assertEqual(results[0]['pair_winding']['0,1'],0)
        self.assertEqual([item['pair_winding']['0,1'] for item in results[1:]], [1,1,1])
        self.assertTrue(all(item['boundary_components']==2 for item in results))
        self.assertEqual(trace['changed_prediction_fraction'],0)
        self.assertEqual(len(trace['frames_by_visit'][0]),65)

    def test_angular_doubling(self):
        coarse=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                             observed_tokens=[1,2],radius=1.,angles=32)
        fine=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                           observed_tokens=[1,2],radius=1.,angles=64)
        self.assertTrue(all(x['match'] for x in compare_double_resolution(coarse,fine)))

    def test_random_chart_does_not_change_intervention(self):
        axes=torch.eye(2)
        original=capture_cycle(self.model,self.tokens,axes,inject_token=0,
                      observed_tokens=[1,2],radius=1.,angles=16,intervention_axes=axes)
        changed=capture_cycle(self.model,self.tokens,
                      torch.tensor([[0.,1.],[-1.,0.]]),inject_token=0,
                      observed_tokens=[1,2],radius=1.,angles=16,intervention_axes=axes)
        self.assertEqual(original['predictions'],changed['predictions'])
        self.assertEqual(original['intervention_axes_sha256'],changed['intervention_axes_sha256'])
        self.assertNotEqual(original['chart_sha256'],changed['chart_sha256'])

    def test_topological_birth_and_boundary_collision(self):
        inner=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                            observed_tokens=[1,2],radius=.2,angles=64)
        critical=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                               observed_tokens=[1,2],radius=.4,angles=64)
        outer=capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                            observed_tokens=[1,2],radius=.8,angles=64)
        self.assertEqual(analyze_cycle(inner)[-1]['pair_winding']['0,1'],0)
        self.assertFalse(analyze_cycle(critical)[-1]['valid'])
        self.assertEqual(analyze_cycle(outer)[-1]['pair_winding']['0,1'],1)

    def test_chart_feature_validity(self):
        decoder=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float32)
        chart=chart_from_decoder(decoder,(0,1))
        self.assertEqual(tuple(chart.shape),(2,2))
        self.assertEqual(sha256_tensor(chart),sha256_tensor(chart.clone()))
        with self.assertRaises(ValueError):chart_from_decoder(decoder,(0,0))
        with self.assertRaisesRegex(ValueError,'dependent'):
            chart_from_decoder(torch.tensor([[1.,0.],[2.,0.]]),(0,1))

    def test_block_direct_injection_relabel(self):
        with self.assertRaisesRegex(ValueError,'exclude'):
            capture_cycle(self.model,self.tokens,self.chart,inject_token=0,
                          observed_tokens=[0,1],radius=1.,angles=32)


if __name__=='__main__':unittest.main()
