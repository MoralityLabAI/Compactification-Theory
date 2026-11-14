"""Independently verify frozen receipts, split isolation, controls and local files."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def verify(root,private=None,allow_line_endings=False):
    r=root/'results';f=load(r/'freeze.json');t=load(r/'training.json');reg=load(r/'registration.json')
    final=load(r/'final.json');diag=load(r/'diagnostics.json');extra=load(r/'interchange_methods.json')
    native=load(r/'native_mediation.json')
    config=load(root/'config.json');repo=root.parents[3]
    assert f['config_sha256']==sha(root/'config.json')==reg['config_sha256']==t['config_sha256']
    assert f['registration_sha256']==sha(r/'registration.json')
    assert f['training_sha256']==sha(r/'training.json')
    assert final['freeze_sha256']==sha(r/'freeze.json')==extra['freeze_sha256']
    assert native['freeze_sha256']==sha(r/'freeze.json')
    assert native['source_sha256']==sha(root/'tools/native_mediation.py')
    if 'recovery' in final:
        assert final['recovery']['amendment_sha256']==sha(r/'recovery_amendment.json')
        assert final['recovery']['driver_sha256']==sha(root/'tools/recover_final.py')
        expected=load(r/'recovery_expected_counts.json')['expected']
        for row in final['results']:
            name=row['checkpoint']['file']
            if name in expected:
                for policy,values in expected[name].items():
                    for key,value in values.items():
                        assert row['policies'][policy]['official']['overall'][key]==value,'recovery changed an observed primary outcome'
    assert f['sources']==reg['sources'],'source changed during registered fitting'
    endings={}
    equivalent_only=[]
    for name,expected in f['sources'].items():
        p=repo/name;raw=p.read_bytes()
        exact=hashlib.sha256(raw).hexdigest()==expected
        if not exact and allow_line_endings:
            lf=raw.replace(b'\r\n',b'\n')
            assert expected in (hashlib.sha256(lf).hexdigest(),hashlib.sha256(lf.replace(b'\n',b'\r\n')).hexdigest()), 'source drift beyond checkout line endings: '+name
            equivalent_only.append(name)
        else:
            assert exact, 'source byte drift: '+name
        endings[name]={'sha256':expected,'has_crlf':b'\r\n' in raw,
                       'lf_normalized_sha256':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()}
    assert f['official_roster_calls_before_freeze']==0==t['official_roster_calls']
    split_signatures=[sig for values in reg['splits'].values() for sig in values]+reg['replay_signatures']
    assert len(split_signatures)==len(set(split_signatures))
    assert set(split_signatures)==set(f['all_pre_final_signatures'])
    selected=sorted(t['screen'],key=lambda v:(-v['development']['overall']['accuracy'],v['visits']))[0]['visits']
    assert selected==f['selected_visits']==t['selected_visits']
    first=next(v for v in t['records'] if v['seed']==config['seeds'][0] and v['objective']=='final')
    best=sorted([v for v in first['candidates'] if v['sector']=='joint'],key=lambda v:(-v['development']['overall']['accuracy'],v['rank']))[0]['rank']
    assert best==f['selected_rank']==t['selected_rank']
    for seed in config['seeds']:
        rows=[v for v in t['records'] if v['seed']==seed]
        assert {v['objective'] for v in rows}==set(config['objectives'])|{'untied'}
        for row in rows:
            if row['objective']=='untied':continue
            assert all(v['training']['base_unchanged'] and v['training']['dictionary_unchanged'] for v in row['candidates'])
            chosen=next(v for v in row['candidates'] if v['rank']==best and v['sector']=='joint')
            assert chosen['training']['trainable_parameters']==row['random_dictionary']['training']['trainable_parameters']
    references={}
    for row in final['results']:
        for policy,sets in row['policies'].items():
            for dataset,value in sets.items():
                signatures=[(v['id'],v['signature']) for v in value['records']]
                if dataset not in references:references[dataset]=signatures
                assert signatures==references[dataset]
                assert not (set(v['signature'] for v in value['records'])&set(split_signatures))
                assert value['overall']['verifier_calls']<=value['overall']['n']*config['budget']
                assert value['budget']==config['budget']
        if 'zero_codes' in row['policies']:
            for dataset in references:
                assert row['policies']['zero_codes'][dataset]['records']==row['policies']['tied'][dataset]['records']
    cpu=load(r/'parity_cpu.json');gpu=load(r/'parity_cuda.json')
    assert cpu['initial_sha256']==gpu['initial_sha256']
    assert cpu['example_sha256']==gpu['example_sha256']
    assert cpu['arms']==gpu['arms']
    oldcpu=load(r/'original_evalmath_cpu.json');oldgpu=load(r/'original_evalmath_cuda.json')
    assert oldcpu['initial_sha256']==oldgpu['initial_sha256']
    assert oldcpu['generated_public_sha256']==oldgpu['generated_public_sha256']
    assert oldcpu['original_receipt']['results']==oldgpu['original_receipt']['results']
    assert oldcpu['resources']['work']==oldgpu['resources']['work']
    resources=[v['resources'] for v in t['screen']+t['records']]+[t['frame_resources']]
    resources += [v['resources'] for v in final['results']]+[diag['resources'],diag['untied_resources'],extra['resources'],gpu['resources'],oldgpu['resources']]
    resources += [v['resources'] for v in native['trials']]
    assert {v['seed'] for v in native['trials']}==set(config['seeds'])
    for resource in resources:
        assert resource['peak_allocated_mb']<=resource['peak_reserved_mb']<=config['max_allocator_mb']
        assert resource['minimum_sampled_free_mb']>=config['min_free_vram_mb']
    for v in diag['interchange']['records']+extra['records']+[row for trial in native['trials'] for row in trial['records']]:
        n=v['valid_counterfactual_tests'];correct=v['correct_counterfactual_outcomes']
        assert 0<=correct<=n
        assert v['causal_fidelity']==(correct/n if n else None)
    for v in diag['topology']['records']:
        assert v['neural_certificate']=='inconclusive_missing_uniform_continuum_bounds'
    def public_guard(value):
        if isinstance(value,dict):
            assert not ({'future','hidden_target','source_program','source_graph'}&set(value)), 'gold field in public challenge'
            for v in value.values():public_guard(v)
        elif isinstance(value,list):
            for v in value:public_guard(v)
    public_guard(load(r/'challenges.public.json'));public_guard(load(r/'replay.public.json'))
    private_verified=False
    if private:
        assert sha(private/'frame.pt')==f['frame_sha256']
        for ck in f['models']:
            assert sha(private/ck['file'])==ck['file_sha256']
        assert sha(private/'challenge.gold.private.json')==final['challenge_gold_sha256']
        private_verified=True
    return {'status':'passed','private_checkpoint_files_verified':private_verified,
        'frozen_models':len(f['models']),'final_checkpoint_count':len(final['results']),
        'official_roster_calls_before_freeze':0,'config_sha256':f['config_sha256'],
        'exact_source_bytes_available':not equivalent_only,
        'equivalent_only_after_checkout_line_endings':equivalent_only,
        'freeze_sha256':sha(r/'freeze.json'),'source_bytes':endings,
        'checks':['source/weight/frame immutability','split signature separation',
            'development-only architecture/rank selection','same paired cases and verifier budgets',
            'equal-size random-dictionary control','zero-code baseline equality',
            'CPU/GPU full original reproduction','sampled VRAM headroom',
            'explicit causal-fidelity denominators','topology certificate boundary',
            'no new gold fields in public challenge artifacts']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--private',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--allow-checkout-line-endings',action='store_true',help='audit text equivalence while explicitly flagging unavailable original frozen CRLF/LF bytes')
    args=p.parse_args();result=verify(args.root.resolve(),args.private,args.allow_checkout_line_endings)
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_bytes'},indent=2))


if __name__=='__main__':main()
