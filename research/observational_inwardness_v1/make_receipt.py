"""Bundle the OIC-16 executed CPU runs into a deterministic auditable receipt."""
from __future__ import annotations
import base64, gzip, hashlib, json
from pathlib import Path
from oic16_verify import TRAIN_NS,OOD_NS,SEEDS,SPLITS,AUGMENTATIONS,SCHEDULES
from oic16_projection import evaluate,projected_steps,projected_marker

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def sha256(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def main():
    runs=[];elapsed=0.
    for name in ('clean_s01.json','clean_s23.json','clean_s4.json',
                 'aug_s01.json','aug_s23.json','aug_s4.json'):
        d=json.loads((OUT/name).read_text())
        elapsed+=d['wall_seconds']
        runs+=d['runs']
    runs.sort(key=lambda r:(r['augment'],r['split'],r['seed']))
    assert len(runs)==20
    summary={}
    for a in AUGMENTATIONS:
        summary[a]={}
        for split in SPLITS:
            group=[r for r in runs if r['augment']==a and r['split']==split]
            assert len(group)==5
            summary[a][split]={}
            for n in TRAIN_NS+OOD_NS:
                panel={}
                for schedule in SCHEDULES:
                    panel[schedule]={
                       key:sum(r['panels'][str(n)][schedule][key] for r in group)/len(group)
                       for key in ('accuracy','brier','nll','mean_passive_deviation')}
                if n%2==0:
                    for key in ('one_flip_by_position','one_erase_by_position'):
                        values=[v for r in group for v in r['panels'][str(n)][key]]
                        panel[key]={'accuracy':sum(values)/len(values),
                                    'minimum_accuracy':min(values),
                                    'maximum_accuracy':max(values),
                                    'cases':len(values)}
                summary[a][split][str(n)]=panel
    projection_summary=[]
    for run in runs:
        projection=evaluate(run['weights'])
        assert all(row['codebook_equality_on_latch'] for row in projection.values())
        assert all(row['full_active']['accuracy']==1.0 and row['passive']['accuracy']==.5 for row in projection.values())
        assert all(row.get('single_flip_minimum_accuracy',1.)==1. and row.get('single_erase_minimum_accuracy',1.)==1. for row in projection.values())
        projection_summary.append({
            'augment':run['augment'],'split':run['split'],'seed':run['seed'],
            'learned_fixed_generator_step':projected_steps(run['weights'])[0],
            'learned_rotating_generator_step':projected_steps(run['weights'])[1],
            'all_evaluated_codebooks_exact':all(row['codebook_equality_on_latch'] for row in projection.values()),
            'all_even_fault_positions_correct':True,
            'emission_markers':{str(n):projected_marker(run['weights'],n) for n in (4,5,6,7,8,9,10,12)},
        })
    source_names=('oic15_pilot.py','oic16_model.py','oic16_train.py','oic16_projection.py','oic16_verify.py')
    data={
       'protocol':{'name':'OIC-16-CPU-01','model':'learned_family_conditioned_circulant_kernel',
                   'no_initial_hidden_phase_input':True,'family_supervision_during_training':True,
                   'declared_known_phase_count_at_inference':True,
                   'train_phase_counts':[4,5,6],'ood_phase_counts':[7,8],
                   'seeds':list(SEEDS),'splits':list(SPLITS),'augmentations':list(AUGMENTATIONS),
                   'lr':.035,'batch':128,'steps':500,'model_parameters':14,
                   'heldout_phase_indices':'1 and n-2','fault_augment':'4% independent raw bit substitutions; 4% independent marked raw erasures',
                   'reference_prior':'uniform phase, balanced family'},
       'runs':runs,
       'total_train_wall_seconds':elapsed,
    }
    payload=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
    compressed=gzip.compress(payload,compresslevel=9,mtime=0)
    (OUT/'oic16_full.json.gz.b64').write_text(base64.b64encode(compressed).decode()+'\n')
    manifest={
       'experiment':'OIC-16','status':'executed_cpu_learned_belief_20_fits',
       'run_count':len(runs),
       'source_sha256':{name:sha256((HERE/name).read_bytes()) for name in source_names},
       'receipt_sha256':sha256(payload),'archive_sha256':sha256(compressed),
       'archive_filename':'oic16_full.json.gz.b64',
       'summary':summary,'projection_summary':projection_summary,
       'full_panel_count':len(runs)*len(TRAIN_NS+OOD_NS)*len(SCHEDULES),
       'relabel_control_count':len(runs)*len(TRAIN_NS+OOD_NS),
       'fault_case_count':len(runs)*sum((n+3)*2 for n in (4,6,8)),
       'total_train_wall_seconds':elapsed,
       'all_projection_generators_correct':all(q['learned_fixed_generator_step']==0 and abs(q['learned_rotating_generator_step'])==1 for q in projection_summary),
       'all_even_fault_positions_correct_after_projection':all(q['all_even_fault_positions_correct'] for q in projection_summary),
    }
    (OUT/'oic16_summary.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Wrote',len(runs),'fits,',len(payload),'bytes raw,',len(compressed),'compressed,',
          manifest['full_panel_count'],'clean panels,',manifest['fault_case_count'],'fault panel positions',
          'elapsed',round(elapsed,1))


if __name__=='__main__':main()
