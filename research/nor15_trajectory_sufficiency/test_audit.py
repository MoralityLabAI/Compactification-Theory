"""Tests for NOR-15A frozen recurrence, reconstruction and leakage controls."""
from __future__ import annotations
from dataclasses import replace
import math
import random
import unittest

from research.nor15_trajectory_sufficiency import audit as n
from research.rrc_compactification import bench25, compound25 as corpus, model25
from research.nor11_attractor_controller import controller as ctrl


def synthetic():
    rng=random.Random(2026111515)
    data=[]
    for i in range(132):
        h9=[rng.gauss(0,1) for _ in range(12)]
        h4=[.4*h9[j]+.1*h9[(j+1)%12]+.25
            for j in range(12)]
        data.append({"h9":h9,"h4":h4,"digest":str(i),
                     "family":"signed"})
    return data[:96],data[96:]


class NOR15TrajectoryAuditTest(unittest.TestCase):
    def test_native_trace_and_no_private_future_or_program_dependency(self):
        ck,digests=bench25.checkpoints()
        checkpoint=ck["recurrent"]
        self.assertEqual(digests["recurrent"],n.old.BASIN_SHA)
        rng=random.Random(2026111515)
        for t in range(80):
            case=corpus.make_case(rng,"signed",f"nor15-public-{t}")
            headers=corpus.possible_headers(case)
            if len(headers)==1:break
        else:self.fail("public header generator exhausted")
        traces,features=n.full_trace(case,headers[0],checkpoint)
        self.assertEqual(len(traces),9)
        self.assertEqual(len(traces[3]),12)
        self.assertEqual(features,corpus.features(case,headers[0]))
        self.assertLess(max(abs(a-b) for a,b in zip(
            model25.logits(case,headers[0],checkpoint),
            model25.affine(checkpoint["state_dict"]["head.weight"],
                           checkpoint["state_dict"]["head.bias"],traces[-1]),
            strict=True)),1e-11)
        changed=replace(case,program=None,future=tuple(123456 for _ in case.future))
        alt,_=n.full_trace(changed,headers[0],checkpoint)
        self.assertEqual(traces,alt)
        h4=traces[3]
        end=n.tail_map(h4,features[4:],checkpoint)
        self.assertLess(max(abs(a-b) for a,b in zip(end,traces[-1],strict=True)),1e-11)
        self.assertEqual(ctrl.public_digest(case,headers[0]),
                         ctrl.public_digest(changed,headers[0]))

    def test_diagonal_gram_spectrum_exact(self):
        A=[[0.]*12 for _ in range(12)]
        for k in range(12):A[k][k]=(k+1)/12
        spectrum=n.gram_spectrum(A)
        for k in range(12):
            self.assertAlmostEqual(spectrum[k],(k+1)/12,places=9)

    def test_jacobian_shape_and_finite_difference_consistency(self):
        ck,_=bench25.checkpoints()
        checkpoint=ck["recurrent"]
        rng=random.Random(2026111516)
        case=corpus.make_case(rng,"signed","nor15-jac")
        head=corpus.possible_headers(case)
        if len(head)!=1:
            return
        traces,features=n.full_trace(case,head[0],checkpoint)
        item={"h4":traces[3],"h9":traces[-1],"tail":features[4:]}
        J=n.transition_jacobian(item,checkpoint)
        self.assertEqual(len(J),12)
        self.assertTrue(all(len(row)==12 for row in J))
        self.assertTrue(all(math.isfinite(v) for row in J for v in row))
        self.assertEqual(len(n.gram_spectrum(J)),12)
        with self.assertRaises(ValueError):
            n.transition_jacobian(item,checkpoint,step=0)

    def test_public_ridge_fit_and_dev_frozen_regularizer(self):
        train,dev=synthetic()
        xs,ys,rff,models,selected,grid=n.freeze_dev(train,dev)
        self.assertEqual(set(models),{"linear","nonlinear","shuffled_linear"})
        self.assertEqual(len(grid),len(n.RIDGES)*3)
        self.assertTrue(all(model["train_cases"]==96 for model in models.values()))
        self.assertTrue(all(model["dimensions"]==13 for k,model in models.items()
                            if k!="nonlinear"))
        self.assertEqual(models["nonlinear"]["dimensions"],37)
        self.assertLess(selected["linear"]["dev_standardized_MSE"],1e-3)
        self.assertLess(selected["nonlinear"]["dev_standardized_MSE"],1e-2)
        for key in ("linear","nonlinear","shuffled_linear"):
            self.assertIn(selected[key]["lambda"],n.RIDGES)
            loss=n.mse(models[key],dev[0],xs,ys,rff)
            self.assertTrue(math.isfinite(loss))
            self.assertGreaterEqual(loss,0)

    def test_pinned_budgets(self):
        with self.assertRaises(ValueError):
            n.new_heldout({},repetitions=5)
        with self.assertRaises(ValueError):
            n.new_heldout({},per_family=13)
        with self.assertRaises(ValueError):
            n.full_trace(None,None,None)

    def test_pair_witnesses_are_finite_without_claiming_exact_collision(self):
        rows=[{"digest":str(k),"family":"signed",
               "h4":[.2*k]*12,"h9":[.05*k]*12} for k in range(8)]
        w=n.nearest_endpoint_pairs(rows)
        self.assertEqual(len(w),8)
        self.assertTrue(all(z["endpoint_distance"]>0 for z in w))
        self.assertTrue(all(z["early_distance"]>0 for z in w))


if __name__=="__main__":
    unittest.main()
