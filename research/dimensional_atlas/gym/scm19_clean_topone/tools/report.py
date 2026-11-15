"""Render descriptive results after the frozen experiment and replay audit."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from ..core import HERE,canonical,sha


def pct(x):return 'undefined' if x is None else f'{100*x:.1f}%'
def fraction(cell):return f"{cell['correct_counterfactual_outcomes']}/{cell['valid_counterfactual_tests']} ({pct(cell['causal_fidelity'])})"
def cell(assay,method,visit=1):return next(v for v in assay['records'] if v['visit']==visit and v['method']==method)
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;d=1+z*z/n;p=k/n;m=(p+z*z/(2*n))/d
    w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [m-w,m+w]


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);a=p.parse_args()
    out=HERE/'results';read=lambda name:json.loads((out/name).read_text())
    done=read('completed.json');audit=read('audit.json');reg=read('registration.json');c=reg['config']
    assert audit['passed'] and audit['completed_sha256']==sha(out/'completed.json')
    clean_table=[];primary=[];secondary=[];family=[]
    for clean,med in zip(done['clean'],done['mediation']):
        seed=clean['seed']
        for kind in ('id','ood'):
            old=clean['historical_reference'][kind];base=clean['scores']['base'][kind]
            adapted=clean['scores'].get('adapter',{}).get(kind)
            clean_table.append({'seed':seed,'stratum':kind,'n':base['n'],'historical_correct':old['correct'],
                'historical_accuracy':old['accuracy'],'base_correct':base['correct'],'base_accuracy':base['accuracy'],
                'adapter_correct':adapted['correct'] if adapted else None,'adapter_accuracy':adapted['accuracy'] if adapted else None,
                'zero_visit_correct':base['initial_cls_correct'],'false_future_acceptances':base['false_future_acceptances'],
                'clean_accuracy_preserved':bool(adapted and adapted['accuracy']>=base['accuracy']-c['allowed_clean_drop'])})
            for policy,scores in {**clean['scores'],'historical':clean['historical_reference']}.items():
                groups={}
                for row in scores[kind]['records']:
                    g=groups.setdefault(row['family'],Counter());g.update(n=1,correct=row['correct'])
                family.extend({'seed':seed,'stratum':kind,'policy':policy,'family':name,**dict(counts)} for name,counts in groups.items())
        for policy,sets in med['policies'].items():
            for kind,assay in sets.items():
                target=cell(assay,c['primary_mediation_method'],c['primary_mediation_visit'])
                random=cell(assay,'random_fiber4',c['primary_mediation_visit'])
                fidelity=target['causal_fidelity'];advantage=None if fidelity is None else fidelity-random['causal_fidelity']
                preserved=next(v for v in clean_table if v['seed']==seed and v['stratum']==kind)['clean_accuracy_preserved']
                n_gate=c['min_eligible_id' if kind=='id' else 'min_eligible_ood']
                positive=(policy=='adapter' and assay['eligible_pairs_total']>=n_gate and fidelity is not None
                    and fidelity>=c['fidelity_gate'] and advantage>=c['random_margin_gate'] and preserved)
                primary.append({'seed':seed,'policy':policy,'stratum':kind,'attempted_pairs':assay['attempted_pairs'],
                    'eligible_pairs':assay['eligible_pairs_total'],'assayed_pairs':assay['assayed_pairs'],
                    'native_correct':target['correct_counterfactual_outcomes'],'fidelity':fidelity,
                    'fidelity_wilson95':wilson(target['correct_counterfactual_outcomes'],target['valid_counterfactual_tests']),
                    'random_correct':random['correct_counterfactual_outcomes'],'random_fidelity':random['causal_fidelity'],
                    'advantage':advantage,'minimum_eligibility_met':assay['eligible_pairs_total']>=n_gate,
                    'clean_preserved':preserved,'registered_positive':bool(positive)})
                for row in assay['records']:
                    secondary.append({'seed':seed,'policy':policy,'stratum':kind,**{k:v for k,v in row.items() if k not in ('outcomes','patch_norms')},
                        'mean_patch_norm':statistics.mean(row['patch_norms']) if row['patch_norms'] else None})
    eligible_primary=[v for v in primary if v['policy']=='adapter']
    positive=len(eligible_primary)==6 and all(v['registered_positive'] for v in eligible_primary)
    summary={'schema':'scm19.summary.v1','preregistration_commit':'a94d749','selection':done['selection'],
        'robust_primary_positive':positive,'clean':clean_table,'primary_mediation':primary,'secondary':secondary,
        'family_accuracy':family,'resources':done['resources'],'oracle':done['oracle'],'audit':audit,
        'shared_fresh_cases_across_seeds':True,'confidence_intervals':'Wilson 95%, descriptive per seed; no multiplicity adjustment',
        'completed_sha256':sha(out/'completed.json')}
    (out/'summary.json').write_text(canonical(summary)+'\n',encoding='utf-8')
    figs=HERE/'figures';figs.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for encoding in c['encodings']:
        pilot=read(f'{encoding}_s191_steps4000.json')
        for ax,kind in zip(axes,('id','ood')):
            ax.plot([v['step'] for v in pilot['curve']],[v[f'dev_{kind}']['accuracy'] for v in pilot['curve']],marker='o',label=encoding.replace('_',' '))
            ax.axhline(c[f'dev_{kind}_gate'],color='gray',ls=':',lw=1)
            ax.set(title=f'Development {kind.upper()}',xlabel='Base optimizer updates',ylim=(0,1));ax.grid(alpha=.18)
    axes[0].set_ylabel('Exact clean top-one accuracy');axes[1].legend(loc='lower right');fig.tight_layout();fig.savefig(figs/'acquisition.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,kind in zip(axes,('id','ood')):
        rows=[v for v in clean_table if v['stratum']==kind]
        names=['Historical','New base','New + VPD'];keys=['historical_accuracy','base_accuracy','adapter_accuracy']
        for index,key in enumerate(keys):
            values=[v[key] for v in rows if v[key] is not None]
            ax.bar(index,statistics.mean(values) if values else 0,color=['#8c939e','#2875b9','#d28b26'][index],alpha=.8)
            ax.scatter([index+o for o in np.linspace(-.10,.10,len(values))],values,color='black',s=18,zorder=3)
        ax.set(xticks=range(3),xticklabels=names,title=f'Fresh {kind.upper()} (n={rows[0]["n"]}/seed)',ylim=(0,1));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Exact clean top-one accuracy');fig.suptitle('Bars: three-seed mean; dots: seeds on the same held-out cases');fig.tight_layout();fig.savefig(figs/'clean_accuracy.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),sharex=True,sharey=True)
    methods=['full_cls','head_readable','pca4','semantic_diff4','sae_group4','native_vpd_write','random_fiber4']
    styles=['-','-','--','--','--','-',':']
    for col,trial in enumerate(done['mediation']):
        policy='adapter' if 'adapter' in trial['policies'] else 'base'
        for row,kind in enumerate(('id','ood')):
            ax=axes[row,col];assay=trial['policies'][policy][kind]
            for method,style in zip(methods,styles):
                values=[cell(assay,method,v)['causal_fidelity'] for v in range(4)]
                ax.plot(range(1,5),[np.nan if v is None else v for v in values],style,marker='o',ms=3,label=method)
            ax.axhline(.9,color='gray',ls=':',lw=1);ax.set(title=f'{kind.upper()}, seed {trial["seed"]}, n={assay["assayed_pairs"]}',ylim=(-.03,1.03),xticks=range(1,5));ax.grid(alpha=.15)
            if row==1:ax.set_xlabel('Patch after recurrent visit')
            if col==0:ax.set_ylabel('Counterfactual fidelity')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.suptitle('Fresh first-operator mediation; CLS-only interventions');fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(figs/'mediation.png');plt.close(fig)
    # These are descriptive geometric measurements on already frozen frames.
    import torch
    geometric=[];fig,axes=plt.subplots(1,3,figsize=(10,3.4),sharey=True)
    frozen=read('freeze.json')
    for ax,receipt in zip(axes,frozen['frames']):
        frames=torch.load(a.private/receipt['file'],weights_only=True);f=frames.get('adapter',frames['base'])
        overlaps=[]
        for v in range(4):
            s=torch.linalg.svdvals(f['semantic'][v].T.double()@f['native'].double())
            overlaps.append([float(x*x) for x in s]+[0.]*(4-len(s)))
        values=np.asarray(overlaps).T
        im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set(title=f'Seed {receipt["seed"]}',xlabel='Recurrent visit',xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5))
        geometric.append({'seed':receipt['seed'],'native_vs_semantic_cosine_squared_by_visit':overlaps,'sae_feature_indices':f['sae_features'].tolist()})
    axes[0].set_ylabel('Principal direction');fig.colorbar(im,ax=axes.ravel().tolist(),label='Squared cosine',shrink=.75)
    fig.suptitle('Calibration semantic differences vs frozen rank-four VPD write span');fig.subplots_adjust(top=.80,bottom=.18,right=.85,wspace=.22);fig.savefig(figs/'subspace_overlap.png');plt.close(fig)
    (out/'geometry.json').write_text(canonical(geometric)+'\n',encoding='utf-8')
    means={kind:{key:statistics.mean(v[key] for v in clean_table if v['stratum']==kind and v[key] is not None)
        for key in ('historical_accuracy','base_accuracy','adapter_accuracy') if any(v[key] is not None for v in clean_table)} for kind in ('id','ood')}
    lines=['# SCM-19 results: clean acquisition and fresh semantic mediation','',
        f"Preregistered commit `a94d749`; selected `{done['selection']['encoding']}` at {done['selection']['common_step_budget']} base updates per seed. The registered robust primary criterion {'passed' if positive else 'did not pass'}.",'',
        '## Clean mathematical accuracy','',
        'Each entry is one neural argmax over 64 programs, followed by exact prefix and withheld-future grading. No verifier/search chooses the prediction. Historical SCM-18 frozen baselines are evaluated on these same fresh cases. Seeds share cases; seed results are not independent case samples.','',
        '| Seed | Split | n | Historical correct | New base correct | New + VPD correct | Base zero-visit correct |',
        '|---:|:---|---:|---:|---:|---:|---:|']
    for v in clean_table:lines.append(f"| {v['seed']} | {v['stratum'].upper()} | {v['n']} | {v['historical_correct']} ({pct(v['historical_accuracy'])}) | {v['base_correct']} ({pct(v['base_accuracy'])}) | {v['adapter_correct']} ({pct(v['adapter_accuracy'])}) | {v['zero_visit_correct']} |")
    lines+=['',f"Three-seed means: ID historical {pct(means['id']['historical_accuracy'])}, new base {pct(means['id']['base_accuracy'])}, VPD {pct(means['id'].get('adapter_accuracy'))}; OOD historical {pct(means['ood']['historical_accuracy'])}, new base {pct(means['ood']['base_accuracy'])}, VPD {pct(means['ood'].get('adapter_accuracy'))}.",'',
        'The zero-visit diagnostic uses the trained input projection and head without recurrent updates. It helps assess how much accuracy the input representation alone supplies. It is a prespecified ablation, not another candidate search. Family breakdowns and false future acceptances are retained in `results/summary.json` and raw score receipts.','',
        '![Fresh clean accuracy](figures/clean_accuracy.png)','','![Development acquisition](figures/acquisition.png)','',
        '## Registered semantic endpoint','',
        'The first source operator changes I→N→S→R→I, with identical input series, topology and other operators. Eligibility requires both exact clean predictions to be correct. Only the first 64 eligible pairs per split are assayed. The primary endpoint patches the actual SAE-conditioned VPD write on CLS after visit two; the rank-four random fiber patch has the same per-pair norm.','',
        '| Seed | Policy | Split | Eligible / attempted | Assayed | VPD correct / n | Random correct / n | Registered positive |',
        '|---:|:---|:---|---:|---:|---:|---:|:---|']
    for v in primary:lines.append(f"| {v['seed']} | {v['policy']} | {v['stratum'].upper()} | {v['eligible_pairs']}/{v['attempted_pairs']} | {v['assayed_pairs']} | {v['native_correct']}/{v['assayed_pairs']} ({pct(v['fidelity'])}) | {v['random_correct']}/{v['assayed_pairs']} ({pct(v['random_fidelity'])}) | {'yes' if v['registered_positive'] else 'no'} |")
    lines+=['',
        'A robust positive requires all three seeds, at least 64 eligible ID / 32 eligible OOD pairs, ≥90% fidelity, ≥15 percentage-point advantage over random, and ≤2 percentage-point clean accuracy loss in each split. Secondary methods and visits cannot replace a failed primary endpoint. A zero denominator is undefined. Wilson intervals in the summary are descriptive, without multiplicity correction.','',
        '![Mediation across recurrent visits](figures/mediation.png)','','![Frozen subspace geometry](figures/subspace_overlap.png)','',
        'Full CLS at visit four is the donor-readout positive control. Sham and terminal head-fiber patches are null controls. All outcomes and per-pair norms are retained in the mediation receipts and replayed by the independent audit. The native write span is a global rank-four intervention on CLS, not four coordinates at every token.','',
        '## Resource and integrity checks','']
    mem=audit['journal_memory'];work=done['resources']['work']
    lines +=[f"Local RTX 3050 4GB, FP32, batch eight, four tied visits. Peak allocated {mem['allocated_mb']:.2f} MiB; peak reserved {mem['reserved_mb']:.2f} MiB; minimum sampled free {mem['minimum_free_mb']:.1f} MiB. Allocation cap 3000 MiB; required free headroom 500 MiB. Active worker time approximately {done['resources']['seconds']/60:.2f} minutes, excluding the recovery handoff.",'',
        f"{work['base_optimizer_steps']:,} base, {work['sae_optimizer_steps']:,} SAE and {work.get('adapter_optimizer_steps',0):,} adapter updates; {work['model_calls']:,} model calls; {work['forward_examples']:,} example forwards; {work['recurrent_example_visits']:,} recurrent example-visits; {work['recurrent_token_visits']:,} recurrent token-visits; {work['readout_examples']:,} readout examples. Cumulative check/grade calls are exact from the journal. Recovery-process source-request and physical-cache counts and the first data-preparation checkpoint are in `completed.json`; the interrupted process's post-data pure-function/cache counters were not durably recorded and are excluded from complete cumulative claims. Pure integer caching does not change mathematical outcomes. See `RECOVERY.md` for the logging-only exception and preregistered recovery. No training or scientific method changed.",'',
        f"Independent audit regraded {audit['checks']['recorded_predictions_regraded']:,} recorded predictions and checked {audit['checks']['mediation_cells_checked']:,} mediation cells, frozen model/dictionary/frame hashes, {audit['checks']['fresh_unique_public_signatures']:,} fresh unique signatures, freeze chronology and journal totals. Eight unit tests passed, including the logger recovery regression. Official EvalMath roster calls: zero. No training occurred after the final freeze. Heavy checkpoints, optimizer state, work journal and fresh gold stay private outside Git.",'',
        'The two encodings have different parameter and preprocessing costs (231,120 coefficient versus 1,111,376 polynomial-Gram parameters). They are acquisition alternatives, not parameter-matched causal evidence. Within a selected frozen model, patch methods share the same cases, model and four-visit prediction budget; projection arithmetic differs. Adapter and base evaluate the same architecture, with zero codes disabling the base adapter.','',
        '## Interpretation and limits','',
        'The acquisition changes combine more training data and updates, richer public coefficient features, a larger four-visit model, and verified equivalent-program labels. This follow-up establishes the combined effect; it does not isolate a single cause of improvement. The polynomial arm supplies explicit generic mathematical preprocessing, and neither arm establishes raw-text mathematical reasoning.','',
        'Mediation is conditional on clean eligibility and on a specified finite formal-series DSL. A successful full-state or PCA patch is evidence about that intervention and eligible subset, not proof that a unique semantic variable has been isolated. Calibration subspace overlap is descriptive geometry. No continuum, braid, curvature or topology certificate is claimed.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'robust_primary_positive':positive,'mean_clean':means,'primary':eligible_primary,'memory':mem},indent=2))


if __name__=='__main__':main()
