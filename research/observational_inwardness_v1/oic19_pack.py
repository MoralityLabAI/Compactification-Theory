"""Freeze OIC-19 30 trained fits and OIC-18 10 baseline fits with SHA-256 pins."""
from __future__ import annotations
import base64,gzip,hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
SOURCES=['oic19_train.py','oic19_select_lambda.py','oic19_pack.py','oic19_identifiability.py','test_oic19_identifiability.py','oic19_faults.py','oic19_verify.py','test_oic19_verify.py','OIC19_PROTOCOL.md','OIC19_REPORT.md','OIC19_THEORY.md','OIC20_PROTOCOL.md','oic18_verify.py','oic18_faults.py','oic16_model.py','oic18_model.py','oic18_train.py','oic15_pilot.py']
FILES=['oic19_entropy_full.json','oic19_entropy_heldout.json',
       'oic19_predictive_full.json','oic19_predictive_heldout_resume.json',
       'oic19_privileged_full.json','oic19_privileged_heldout.json']


def load_archive(path:Path):return json.loads(gzip.decompress(base64.b64decode(path.read_text().strip())))


def load():
    runs=[]
    for filename in FILES:
        runs.extend(json.loads((RESULTS/filename).read_text())['runs'])
    pilot=json.loads((RESULTS/'oic19_pilot_seed0.json').read_text())
    runs.append(next(x for x in pilot['runs'] if x['variant']=='predictive'))
    keys=[(r['variant'],r['split'],r['seed']) for r in runs]
    if len(runs)!=30 or len(set(keys))!=30:raise ValueError('30 experimental fits required')
    baseline=[]
    for split in ('full','heldout'):
        a=load_archive(RESULTS/f'oic18_clean_{split}.json.gz.b64')
        selected=[r for r in a['runs'] if r['architecture']=='von_mises']
        if len(selected)!=5: raise ValueError('OIC18 baseline condition missing')
        for r in selected:
            w=r['weights'];k=[]
            for row in w['kernel_logits']:
                shifted=[math.exp(v-max(row)) for v in row]
                total=sum(shifted);k.append([x/total for x in shifted])
            baseline.append({'variant':'baseline_family_bce','split':split,
                 'seed':r['seed'],'weights':w,'panels':r['panels'],'training':r['training'],
                 'kernels':{'fixed_step_mass':k[0][2],
                            'rotating_step_mass':max(k[1][1],k[1][3]),
                            'fixed_kernel':k[0],'rotating_kernel':k[1],
                            'rotating_orientation':-1 if k[1][1]>=k[1][3] else 1,
                            'rotating_kernel_entropy_bits':-sum(x*math.log2(x) for x in k[1])},
                 'baseline_pinned_to':'OIC18_clean_arch_von_mises'})
    return sorted(runs+baseline,key=lambda r:(r['variant'],r['split'],r['seed']))


def stats(runs):
    results=[]
    for variant in ('baseline_family_bce','entropy','predictive','phase_supervised'):
      for split in ('full','heldout'):
        group=[r for r in runs if r['variant']==variant and r['split']==split]
        if len(group)!=5:raise ValueError('five seeds required '+variant+' '+split)
        rows={'variant':variant,'split':split,'seeds':sorted(r['seed'] for r in group)}
        rows['correct_step_mass_mean']=round(statistics.mean(r['kernels']['rotating_step_mass'] for r in group),8)
        rows['correct_step_mass_min']=round(min(r['kernels']['rotating_step_mass'] for r in group),8)
        rows['rotating_kernel_entropy_mean_bits']=round(statistics.mean(r['kernels']['rotating_kernel_entropy_bits'] for r in group),7)
        rows['correct_step_threshold_pass_count']=sum(r['kernels']['rotating_step_mass']>=.9 for r in group)
        for n in (8,24,64):
            p=[r['panels'][str(n)]['full_active'] for r in group]
            rows[f'n{n}_family_accuracy_mean']=round(statistics.mean(q['accuracy'] for q in p),8)
            rows[f'n{n}_family_accuracy_min_seed']=round(min(q['accuracy'] for q in p),8)
            rows[f'n{n}_family_brier_mean']=round(statistics.mean(q['brier'] for q in p),8)
            rows[f'n{n}_phase_MAP_mean']=round(statistics.mean(q['rotating_phase_MAP_gauge_accuracy'] for q in p),8)
            rows[f'n{n}_phase_MAP_min_seed']=round(min(q['rotating_phase_MAP_gauge_accuracy'] for q in p),8)
        rows['all_required_gates_per_seed']=[int(
            r['kernels']['rotating_step_mass']>=.9
            and all(r['panels'][str(n)]['full_active']['accuracy']>=.95 and r['panels'][str(n)]['full_active']['rotating_phase_MAP_gauge_accuracy']>=.9 for n in (24,64))
            and all(r['panels'][str(n)]['passive']['mean_passive_deviation']<1e-4 for n in (8,24,64))
            ) for r in group]
        rows['joint_gate_pass_count']=sum(rows['all_required_gates_per_seed'])
        rows['per_seed_kernels']=[{'seed':r['seed'],'kernel':r['kernels']['rotating_kernel'],
                                  'correct_mass':r['kernels']['rotating_step_mass']} for r in group]
        results.append(rows)
    return results


def main():
    runs=load();summary=stats(runs)
    selection=json.loads((RESULTS/'oic19_lambda_selection.json').read_text())
    payload={'protocol':{'name':'OIC-19-CPU-01','train_n':[4,5,6],'ood_n':[7,8,12,16,24,32,64],
           'seeds':[0,1,2,3,4],'splits':['full','heldout'],'train_steps':500,
           'train_batch':128,'new_train_fits':30,'frozen_baseline_fits':10,
           'entropy_lambda_selection':selection['chosen_lambda'],
           'validation_training_fits':selection['validation_training_fits'],
           'predictive_weight':.35,'privileged_phase_weight':.35,
           'phase_supervised_is_privileged':True,
           'same_training_rng':'151100+seed*19','sensor':'public_raw_state','known_phase_count':True,
           'pre_registered':'OIC19_PROTOCOL.md'},'runs':runs}
    packed=gzip.compress(json.dumps(payload,separators=(',',':'),sort_keys=True).encode(),mtime=0)
    out=RESULTS/'oic19_full.json.gz.b64';out.write_text(base64.b64encode(packed).decode()+'\n')
    source_sha={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in SOURCES}
    m={'protocol':payload['protocol'],'status':'completed_cpu_toy_sweep',
       'full_archive':out.name,'full_archive_gzip_sha256':hashlib.sha256(packed).hexdigest(),
       'full_archive_decoded_sha256':hashlib.sha256(gzip.decompress(packed)).hexdigest(),
       'source_sha256':source_sha,'lambda_validation':selection,
       'aggregates':summary,'claims':{'learned_phase_generator_from_public_raw_likelihood':True,
          'privileged_ceiling_flagged':True,'neural_or_cosmological_claim':False}}
    (RESULTS/'oic19_summary.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fits':len(runs),'groups':[{k:v for k,v in group.items() if k in ('variant','split','correct_step_mass_mean','n64_phase_MAP_mean','n64_family_accuracy_mean','joint_gate_pass_count')} for group in summary]},indent=2))

if __name__=='__main__':main()
