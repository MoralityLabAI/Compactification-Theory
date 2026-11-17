"""Synthetic fixtures only; actual public receipts verified in CI."""
import copy
import unittest
from research.dimensional_atlas.gym.bpd_quotient_bridge.cq02_integrity import (
  audit_native,audit_scm21,crosswalk,git_blob_sha,SEEDS,SCM_SEEDS)


def native_fixture():
    rows=[]
    for seed in SEEDS:
        rows.append({'seed':seed,'backend':'native','native_parity_max_abs':0,
            'component_observable_rank_now':1,'component_observable_rank_after_one':2,
            'four_pass_route':128,'heldout_boards':128,'causal_pairs':64,
            'largest_norm_matching_error':0.,'immediate_median':0.,'delayed_median':1.,
            'transfer':{'native':64,'projected_native':64,'random_fiber':(12 if seed==SEEDS[-1] else 6),'sham':0,'visible':0}})
    return {'schema':'BPD-CQ-02/trained-factors-cpu/v1','source':'native BPDModel/TiedComponentLinear with trained factors',
            'seeds':list(SEEDS),'rows':rows}


def scm_fixture():
    methods={'native_vpd_write':0,'semantic_diff4':147,'head_readable':413,'full_cls':432,'sham':0}
    rows=[]
    for method,total in methods.items():
        share=[total//6]*6
        for i in range(total%6):share[i]+=1
        for i,(seed,stratum) in enumerate((seed,stratum) for seed in SCM_SEEDS for stratum in ('id','ood')):
            rows.append({'model':'causal','visit':1,'method':method,'seed':seed,'stratum':stratum,
                        'n':128,'correct':share[i],'mean_patch_norm':1.})
    return {'schema':'scm21.summary.v1','secondary':rows}


class Cq02IntegrationTests(unittest.TestCase):
    def test_reference_snapshots(self):
        x=crosswalk(native_fixture(),scm_fixture())
        self.assertEqual(x['bpd']['outcomes']['native'],512)
        self.assertEqual(x['compact']['measurements']['semantic_diff4']['successes'],147)
    def test_native_gates_refuse_mutation(self):
        b=native_fixture();b['rows'][0]['component_observable_rank_after_one']=1
        with self.assertRaises(ValueError):audit_native(b)
    def test_native_parity_refusal(self):
        b=native_fixture();b['rows'][0]['native_parity_max_abs']=.1
        with self.assertRaises(ValueError):audit_native(b)
    def test_wrong_random_dose_count(self):
        b=native_fixture();b['rows'][2]['transfer']['random_fiber']=65
        with self.assertRaises(ValueError):audit_native(b)
    def test_duplicate_scm_cell(self):
        b=scm_fixture();b['secondary'].append(copy.deepcopy(b['secondary'][0]))
        with self.assertRaises(ValueError):audit_scm21(b)
    def test_truncated_scm_cell(self):
        b=scm_fixture();b['secondary'].pop()
        with self.assertRaises(ValueError):audit_scm21(b)
    def test_blob_hash(self):
        self.assertEqual(git_blob_sha(b'a'),'2e65efe2a145dda7ee51d1741299f848e5bf752e')

if __name__=='__main__':unittest.main()
