"""OIC-18 deterministic archive packaging and mathematically replayable manifest.

Reads the four executed condition files and their raw-sensor fault panels,
combines frozen OIC-16 baseline, and creates reproducible gzip/base64 receipts.
This script does not refit or modify the underlying recorded models.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
NS=(4,5,6,7,8,12,16,24,32,64)
FNS=(4,6,8,12,16,24,32,64)
ARCHS=('fourier','temperature','von_mises','harmonic3')
AUGS=('clean','raw_fault_aug')
SPLITS=('full','heldout')


def sha(b):return hashlib.sha256(b).hexdigest()


def pack(data,target:Path):
    payload=(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    bio=io.BytesIO()
    with gzip.GzipFile(fileobj=bio,mode='wb',mtime=0,filename='') as f:f.write(payload)
    compressed=bio.getvalue()
    target.write_text(base64.b64encode(compressed).decode()+'\n')
    return {'file':str(target.relative_to(HERE)),'decoded_sha256':sha(payload),
            'gzip_sha256':sha(compressed),'decoded_bytes':len(payload),
            'compressed_bytes':len(compressed)}


def mean(xs):
    values=list(xs)
    return sum(values)/len(values)


def aggregate_models(records):
    groups=[]
    for arch in ARCHS:
        for aug in AUGS:
            for split in SPLITS:
                selection=sorted((r for r in records if (r['architecture'],r['augmentation'],r['split'])==(arch,aug,split)),key=lambda r:r['seed'])
                assert len(selection)==5 and [r['seed'] for r in selection]==list(range(5)),(arch,aug,split)
                for n in NS:
                    entries=[r['panels'][str(n)]['full_active'] for r in selection]
                    passive=[r['panels'][str(n)]['passive'] for r in selection]
                    # Each phase/family position is tested across 5 model seeds.
                    by_phase=[mean([int((item['probabilities'][i]>0.5)==int(i>=n)) for item in entries])
                              for i in range(2*n)]
                    groups.append({
                        'architecture':arch,'augmentation':aug,'split':split,'n':n,
                        'mean_balanced_family_accuracy':round(mean(x['accuracy'] for x in entries),8),
                        'min_seed_family_accuracy':round(min(x['accuracy'] for x in entries),8),
                        'mean_brier':round(mean(x['brier'] for x in entries),8),
                        'worst_family_phase_accuracy_across_seeds':round(min(by_phase),8),
                        'min_seed_rotating_phase_MAP_accuracy':round(min(x['rotating_phase_MAP_gauge_accuracy'] for x in entries),8),
                        'mean_rotating_phase_MAP_accuracy':round(mean(x['rotating_phase_MAP_gauge_accuracy'] for x in entries),8),
                        'max_passive_prior_deviation':round(max(x['mean_passive_deviation'] for x in passive),10),
                    })
    return groups


def aggregate_faults(groups):
    out=[]
    for arch in ARCHS[1:]:
        for aug in AUGS:
            for split in SPLITS:
                fits=[r for r in groups if (r['architecture'],r['augmentation'],r['split'])==(arch,aug,split)]
                assert len(fits)==5
                for n in FNS:
                    for kind in ('flip','erase'):
                        vals=[r['panels'][str(n)][kind] for r in fits]
                        out.append({
                            'architecture':arch,'augmentation':aug,'split':split,
                            'n':n,'sensor_fault':kind,'seeds':5,
                            'mean_accuracy':round(mean(x['accuracy'] for x in vals),8),
                            'min_seed_accuracy':round(min(x['accuracy'] for x in vals),8),
                            'min_position_accuracy_across_seeds':round(min(x['worst_fault_position_accuracy'] for x in vals),8),
                            'min_family_phase_accuracy_across_seeds':round(min(x['worst_family_phase_accuracy'] for x in vals),8),
                            'total_cases':sum(x['all_episodes'] for x in vals),
                            'total_correct':sum(x['correct_episodes'] for x in vals),
                        })
    return out


def main():
    archives=[];models=[];fault_files=[];fault_records=[]
    for aug in AUGS:
        for split in SPLITS:
            name=f'oic18_{aug}_{split}'
            data=json.loads((RESULTS/f'{name}.json').read_text())
            assert len(data['runs'])==15
            models+=data['runs']
            archives.append(pack(data,RESULTS/f'{name}.json.gz.b64'))
            original=RESULTS/(f'oic18_faults_{aug}_{split}_large.json')
            if aug=='raw_fault_aug' and split=='full':
                original=RESULTS/'oic18_faults_raw_fault_aug_full_prelim.json'
            raw=json.loads(original.read_text())
            small=json.loads((RESULTS/f'oic18_faults_{aug}_{split}_small.json').read_text())
            assert len(raw['runs'])==15 and raw['phase_counts']==[8,24,64]
            assert len(small['runs'])==15 and small['phase_counts']==[4,6,12,16,32]
            index={(r['architecture'],r['seed']):r for r in small['runs']}
            merged=[]
            for r in raw['runs']:
                row=index[(r['architecture'],r['seed'])]
                assert row['split']==r['split'] and row['augmentation']==r['augmentation']
                combined={**r,'panels':{**row['panels'],**r['panels']}}
                assert set(combined['panels'])=={str(n) for n in FNS}
                merged.append(combined)
            combined_receipt={'status':'complete_eight_even_n_every_fault_position',
                 'source_sweep_file':f'{name}.json','phase_counts':list(FNS),'runs':merged}
            fault_records.extend(merged)
            fault_files.append(pack(combined_receipt,RESULTS/f'{name}_faults.json.gz.b64'))
    baseline_data=json.loads((RESULTS/'oic18_baseline.json').read_text())
    assert len(baseline_data['records'])==20
    baseline_archive=pack(baseline_data,RESULTS/'oic18_frozen_baseline.json.gz.b64')
    aggregate=aggregate_models(models+baseline_data['records'])
    fault_agg=aggregate_faults(fault_records)
    source_names=('oic18_model.py','oic18_train.py','oic18_baseline.py','oic18_faults.py',
                  'oic18_verify.py','oic18_pack.py','test_oic18_math.py',
                  'OIC18_LOCK.md','OIC18_THEORY.md')
    manifest={
        'name':'OIC-18-CPU-01','status':'executed_synthetic_cpu_phase_resolution_sweep',
        'training_phase_counts':[4,5,6],
        'registered_ood_phase_counts':[7,8,12,16,24,32,64],
        'registered_single_raw_fault_phase_counts':list(FNS),
        'num_new_fits':60,'num_frozen_OIC16_baseline_fits':20,
        'optimizer_steps_per_fit':500,'training_batch_size':128,
        'samples_matched_to_OIC16':True,
        'archives':archives,'fault_archives':fault_files,'frozen_baseline_archive':baseline_archive,
        'source_sha256':{p:sha((HERE/p).read_bytes()) for p in source_names},
        'aggregate_clean':aggregate,
        'aggregate_faults':fault_agg,
        'prior_work':'OIC16 frozen baseline phase counts 7/8 preregistered, 24 etc were posthoc; OIC18 future OOD n >=12 preregistered',
        'parameter_counts':{'fourier':14,'temperature':15,'von_mises':13,'harmonic3':14},
        'gauge_rule':'cyclic translation, reflection and arbitrary consistent latent relabeling preserve public traces',
        'interpretation_limits':'synthetic finite state observer, known group size and two family structure, no general NN or phenomenal consciousness result',
    }
    (RESULTS/'oic18_summary.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('PACKED',len(models),'new fits,',len(fault_records),'fault-panel records and',len(aggregate),'clean aggregates')
    for arch in ARCHS:
        for split in SPLITS:
            for aug in AUGS:
                z=next(x for x in aggregate if (x['architecture'],x['split'],x['augmentation'],x['n'])==(arch,split,aug,24))
                print('N24',arch,split,aug,z['mean_balanced_family_accuracy'])
    print('MANIFEST',RESULTS/'oic18_summary.json', 'bytes',(RESULTS/'oic18_summary.json').stat().st_size)

if __name__=='__main__':main()
