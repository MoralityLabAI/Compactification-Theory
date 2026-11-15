"""SMC-13 source-independent trainer/SAE negative-control smoke tests."""
import sys
import tempfile
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc13_family import cohort,fingerprint
from smc13_train_teachers import (
    train_teacher,teacher_metrics,save_teacher,
    sae_cone_count,controls_gaussian,persist_sae,stable_hash_tensors
)
from smc07_real_sae import fit_sae,TopKSAE


class IndependentTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train=cohort('teacher_train',4)
        cls.val=cohort('teacher_val',4)

    def test_independently_trained_checkpoint_and_validation_selection(self):
        model,record=train_teacher(self.train,self.val,501,25)
        self.assertEqual(record['teacher_seed'],501)
        self.assertEqual(record['selected_step'],25)
        self.assertNotEqual(record['teacher_initial_fp32_sha256'],
                            record['teacher_final_fp32_sha256'])
        self.assertEqual(record['teacher_final_fp32_sha256'],
                         stable_hash_tensors(model.state_dict()))
        metric=teacher_metrics(model,self.val)
        self.assertEqual(metric['n_puzzles'],4)
        self.assertEqual(metric['missing_cell_total'],4*30)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'independent.pt'
            snap=save_teacher(model,record,fingerprint(self.train),path)
            self.assertEqual(len(snap['file_sha256']),64)
            self.assertEqual(snap['model_parameter_sha256'],
                             record['teacher_final_fp32_sha256'])

    def test_teacher_training_seeds_give_distinct_independent_parameters(self):
        a,A=train_teacher(self.train,self.val,501,25)
        b,B=train_teacher(self.train,self.val,701,25)
        self.assertNotEqual(A['teacher_initial_fp32_sha256'],
                            B['teacher_initial_fp32_sha256'])
        self.assertNotEqual(A['teacher_final_fp32_sha256'],
                            B['teacher_final_fp32_sha256'])

    def test_sae_cone_count_and_gaussian_shape(self):
        torch.manual_seed(71)
        sae=TopKSAE(48,96,4)
        result=sae_cone_count(sae)
        self.assertEqual(result['n_atoms'],96)
        self.assertGreaterEqual(result['strict_interior'],0)
        self.assertLessEqual(result['strict_interior'],96)
        a,b=controls_gaussian((4,81,48),(2,81,48))
        c,d=controls_gaussian((4,81,48),(2,81,48))
        self.assertTrue(torch.equal(a,c))
        self.assertTrue(torch.equal(b,d))

    def test_sae_roundtrip_and_train_only_scale(self):
        torch.manual_seed(501)
        model=TopKSAE(48,96,4)
        mu=torch.zeros(48)
        scale=torch.ones(48)
        with tempfile.TemporaryDirectory() as root:
            rec=persist_sae(model,mu,scale,501,71,Path(root))
            self.assertEqual(len(rec['file_sha256']),64)
            self.assertEqual(len(rec['sae_fingerprint']),64)
            self.assertTrue(Path(rec['path']).is_file())

    def test_reject_wrong_training_budgets(self):
        with self.assertRaises(ValueError):
            train_teacher(self.train,self.val,501,24)
        with self.assertRaises(ValueError):
            train_teacher(self.train,self.val,501,0)


if __name__=='__main__':
    unittest.main()
