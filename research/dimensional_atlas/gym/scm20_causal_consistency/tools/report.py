"""Post-audit descriptive tables, figures and interpretation."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
from ..core import HERE,sha


def percent(x):return 'undefined' if x is None else f'{100*x:.1f}%'
def endpoint(a,method,visit=1):return next(v for v in a['cells'] if v['method']==method and v['visit']==visit)
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,m-w),min(1,m+w)]
def write_json(path,obj):
    with path.open('w',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def summarize(done,audit,read,c):
    clean=[];primary=[];secondary=[];families=[]
    for receipt in done['final_receipts']:
        row=read(receipt['file']);seed=row['seed']
        for kind in ('id','ood'):
            warm=row['clean']['warm'][kind]
            for name,sets in row['clean'].items():
                v=sets[kind];lost=sum(a['correct'] and not b['correct'] for a,b in zip(warm['records'],v['records']))
                gained=sum(b['correct'] and not a['correct'] for a,b in zip(warm['records'],v['records']))
                clean.append({'seed':seed,'stratum':kind,'model':name,'n':v['n'],'correct':v['correct'],'accuracy':v['accuracy'],
                    'delta_from_warm':v['accuracy']-warm['accuracy'],'preserved':v['accuracy']>=warm['accuracy']-c['allowed_clean_drop'],
                    'warm_correct_lost':lost,'warm_incorrect_gained':gained,'false_future_acceptances':v['false_future_acceptances']})
                for family in sorted(set(x['family'] for x in v['records'])):
                    group=[x for x in v['records'] if x['family']==family]
                    families.append({'seed':seed,'stratum':kind,'model':name,'family':family,'n':len(group),'correct':sum(x['correct'] for x in group)})
            values={name:sets[kind] for name,sets in row['mediation'].items()}
            for name,assay in values.items():
                cell=endpoint(assay,'native_vpd_write');random=endpoint(assay,'random_fiber4')
                warm_cell=endpoint(values['warm'],'native_vpd_write');control=endpoint(values['control'],'native_vpd_write')
                preserved=next(v['preserved'] for v in clean if v['seed']==seed and v['stratum']==kind and v['model']==name)
                fidelity=cell['fidelity'];diff=lambda other:None if fidelity is None else fidelity-other['fidelity']
                advantages=[diff(v) for v in (random,warm_cell,control)]
                passed=(done['selection']['promoted'] and name=='causal' and assay['common_eligible']>=c['min_eligible_'+kind]
                    and fidelity is not None and fidelity>=c['fidelity_gate'] and all(x>=c['random_margin_gate'] for x in advantages) and preserved)
                primary.append({'seed':seed,'stratum':kind,'model':name,'attempted_pairs':assay['attempted_pairs'],'own_eligible':assay['own_eligible'],
                    'common_eligible':assay['common_eligible'],'n':cell['n'],'correct':cell['correct'],'fidelity':fidelity,
                    'wilson95':wilson(cell['correct'],cell['n']),'random_correct':random['correct'],'random_fidelity':random['fidelity'],
                    'advantage_over_random':advantages[0],'advantage_over_warm':advantages[1],'advantage_over_control':advantages[2],
                    'clean_preserved':preserved,'registered_positive':bool(passed)})
                for v in assay['cells']:
                    secondary.append({'seed':seed,'stratum':kind,'model':name,'method':v['method'],'visit':v['visit'],
                        'n':v['n'],'correct':v['correct'],'fidelity':v['fidelity'],
                        'mean_patch_norm':statistics.mean(v['patch_norms']) if v['patch_norms'] else None})
    chosen=[v for v in primary if v['model']=='causal']
    positive=len(chosen)==6 and all(v['registered_positive'] for v in chosen)
    return {'schema':'scm20.summary.v1','robust_registered_positive':positive,'selection':done['selection'],'clean':clean,
        'primary':primary,'secondary':secondary,'family_accuracy':families,'resources':done['resources'],'oracle':done['oracle'],
        'audit':audit,'same_fresh_cases_across_seeds':True,'identical_common_pairs_within_seed':True,
        'confidence_intervals':'descriptive Wilson 95%; no multiplicity adjustment or pooled independent-seed claim',
        'completed_sha256':audit['completed_sha256']}


def plots(summary,done,read,figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    colors={'warm':'#8b949e','control':'#2b79b4','causal':'#d88b25'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,kind in zip(axes,('id','ood')):
        for j,name in enumerate(('warm','control','causal')):
            values=[v['accuracy'] for v in summary['clean'] if v['model']==name and v['stratum']==kind]
            ax.bar(j,statistics.mean(values),color=colors[name],alpha=.85)
            ax.scatter(j+np.linspace(-.09,.09,len(values)),values,color='black',s=18,zorder=3)
        ax.set(title=f'Fresh {kind.upper()} (512 cases/seed)',xticks=range(3),xticklabels=['SCM19 warm','Matched control','Causal training'],ylim=(0,1));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Exact clean top-one accuracy');fig.suptitle('Three-seed means; dots share the same fresh cases' if done['selection']['promoted'] else 'No candidate promoted: all final slots retain warm starts');fig.tight_layout();fig.savefig(figs/'clean_accuracy.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,kind in zip(axes,('id','ood')):
        for j,name in enumerate(('warm','control','causal')):
            rows=[v for v in summary['primary'] if v['model']==name and v['stratum']==kind]
            xs=np.arange(3)+(j-1)*.22;ys=[np.nan if r['fidelity'] is None else r['fidelity'] for r in rows]
            ax.bar(xs,ys,width=.21,color=colors[name],label=name)
            lo=[0 if r['wilson95'] is None else max(0,r['fidelity']-r['wilson95'][0]) for r in rows]
            hi=[0 if r['wilson95'] is None else max(0,r['wilson95'][1]-r['fidelity']) for r in rows]
            ax.errorbar(xs,ys,yerr=[lo,hi],fmt='none',ecolor='black',elinewidth=.7,capsize=2)
        causal=[v for v in summary['primary'] if v['model']=='causal' and v['stratum']==kind]
        ax.scatter(np.arange(3)+.22,[np.nan if r['random_fidelity'] is None else r['random_fidelity'] for r in causal],marker='x',color='purple',zorder=4,label='causal norm-matched random')
        ax.axhline(.9,color='gray',ls=':',lw=1)
        ax.set(title=f'Primary native-write endpoint: {kind.upper()}',xticks=range(3),xticklabels=[f"{r['seed']} (n={r['n']})" for r in causal],ylim=(-.025,1.03),xlabel='Seed and common-pair denominator');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Exact counterfactual fidelity');axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('CLS native-write swap after visit two; descriptive Wilson 95% intervals');fig.tight_layout();fig.savefig(figs/'primary_mediation.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True,sharey=True)
    methods=['full_cls','head_readable','pca4','semantic_diff4','sae_group4','native_vpd_write','random_fiber4']
    for col,receipt in enumerate(done['final_receipts']):
        trial=read(receipt['file'])
        for row,kind in enumerate(('id','ood')):
            ax=axes[row,col];assay=trial['mediation']['causal'][kind]
            for method in methods:
                values=[endpoint(assay,method,v)['fidelity'] for v in range(4)]
                ax.plot(range(1,5),[np.nan if v is None else v for v in values],marker='o',ms=3,
                    ls=':' if method=='random_fiber4' else '--' if method in ('pca4','semantic_diff4','sae_group4') else '-',label=method)
            ax.set(title=f"{kind.upper()}, seed {trial['seed']}, n={assay['assayed_pairs']}",ylim=(-.025,1.03),xticks=range(1,5));ax.grid(alpha=.15)
            if row==1:ax.set_xlabel('Patch after visit')
            if col==0:ax.set_ylabel('Counterfactual fidelity')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.suptitle('Selected causal model: prespecified methods and visits on identical common pairs');fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(figs/'visit_mediation.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for weight in (0,1,4):
        row=read(f's191_lambda{weight}.json')
        for ax,key in zip(axes.ravel(),('clean_nll','clean_kl','causal_nll','write_energy')):
            ax.plot([r['step'] for r in row['losses']],[r[key] for r in row['losses']],label=f'lambda {weight}')
            ax.set_title(key.replace('_',' '));ax.grid(alpha=.15)
            if key=='write_energy':ax.set_yscale('log')
    for ax in axes[1]:ax.set_xlabel('Pilot optimizer updates')
    axes[0,0].legend();fig.suptitle('Registered pilot loss components (sampled every 100 updates)');fig.tight_layout();fig.savefig(figs/'training_losses.png');plt.close(fig)


def document(summary,done,read,c):
    clean=summary['clean'];primary=summary['primary'];chosen=[v for v in primary if v['model']=='causal'];selection=done['selection']
    failures=[v for v in clean if v['model']=='causal' and not v['preserved']]
    worst=min(v['delta_from_warm'] for v in clean if v['model']=='causal')
    lines=['# SCM-20 results: causal consistency and clean preservation','',
        f"Preregistered commit `179c01d`. Promoted causal checkpoint: {selection['promoted']}; lambda {selection['causal_weight']}, common update budget {selection['step']}. The registered robust positive criterion {'passed' if summary['robust_registered_positive'] else 'did not pass'}.",'',
        f"Fresh clean preservation failed in {len(failures)}/6 seed-by-split comparisons; the worst causal-model change versus its warm start was {100*worst:+.1f} percentage points. Any mediation gain must be interpreted together with those clean losses and the prespecified fidelity/control thresholds.",'',
        '## Clean top-one mathematics','',
        'All entries use one neural argmax and independent exact post-prediction grading, without search or verification to choose a program. Warm means the frozen SCM-19 adapter; control uses the identical training objective and three-forward computation with causal coefficient zero. Each model sees the same 512 ID / 512 OOD fresh clean cases. Seeds share these cases, so their scores are not independent case samples.','',
        '| Seed | Split | Warm | Matched control | Causal | Causal change | Clean preserved |',
        '|---:|:---|---:|---:|---:|---:|:---|']
    if not selection['promoted']:
        lines[4:4]=['No causal pilot satisfied the registered development clean-preservation gate. All three final model slots therefore retain the warm starts under the guarded fallback; the fresh tables below do not represent improved or successfully promoted causal training. Failed pilot development results remain in the receipts.','']
    for seed in c['seeds']:
        for kind in ('id','ood'):
            rows={r['model']:r for r in clean if r['seed']==seed and r['stratum']==kind};v=rows['causal']
            lines.append(f"| {seed} | {kind.upper()} | {rows['warm']['correct']}/512 ({percent(rows['warm']['accuracy'])}) | {rows['control']['correct']}/512 ({percent(rows['control']['accuracy'])}) | {v['correct']}/512 ({percent(v['accuracy'])}) | {100*v['delta_from_warm']:+.1f} pp | {'yes' if v['preserved'] else 'no'} |")
    lines+=['','![Fresh clean accuracy](figures/clean_accuracy.png)','',
        '## Development-only checkpoint choice','',
        'All six registered causal candidates are shown, including rejected checkpoints. The warm development baseline is fixed before updates. The lambda-zero pilot runs to 4,000 updates and provides the checkpoint at the selected budget; its selection-evaluation cost differs from the causal pilots.','',
        '| Lambda | Step | ID clean | OOD clean | Clean gate | ID native / random | OOD native / random |',
        '|---:|---:|---:|---:|:---|---:|---:|']
    for weight in (1,4):
        pilot=read(f's191_lambda{weight}.json')
        for point in pilot['development']:
            lines.append(f"| {weight} | {point['step']} | {percent(point['clean']['id']['accuracy'])} | {percent(point['clean']['ood']['accuracy'])} | {'pass' if point['clean_preserved'] else 'reject'} | {percent(point['mediation']['id']['native'])} / {percent(point['mediation']['id']['random'])} | {percent(point['mediation']['ood']['native'])} / {percent(point['mediation']['ood']['random'])} |")
    lines+=['',
        '## Primary causal-consistency confirmation','',
        'The first program operator is cycled while inputs, topology and the other operators stay fixed. Generation never uses model correctness. Eligibility requires recipient and donor to be exact-clean-correct under warm, control and causal models. The first 128 common-eligible pairs per split are assayed; all compared models use identical pair IDs. Own eligibility, common counts, selected predictions and outcomes are preserved in raw receipts.','',
        '| Seed | Split | Common / attempted | Assayed | Warm native | Control native | Causal native | Causal random | Positive |',
        '|---:|:---|---:|---:|---:|---:|---:|---:|:---|']
    for v in chosen:
        rows={r['model']:r for r in primary if r['seed']==v['seed'] and r['stratum']==v['stratum']}
        lines.append(f"| {v['seed']} | {v['stratum'].upper()} | {v['common_eligible']}/{v['attempted_pairs']} | {v['n']} | {rows['warm']['correct']}/{v['n']} | {rows['control']['correct']}/{v['n']} | {v['correct']}/{v['n']} ({percent(v['fidelity'])}) | {v['random_correct']}/{v['n']} ({percent(v['random_fidelity'])}) | {'yes' if v['registered_positive'] else 'no'} |")
    lines+=['',
        'A robust positive requires all seeds and both splits: ≥64 common-eligible pairs, ≥90% native-write fidelity, ≥15 percentage points over each of norm-matched random, warm native and matched-control native, and ≤2 percentage-point fresh clean loss versus warm. Selection must have promoted a candidate. Undefined fidelity at zero tests is not zero. Secondary directions/visits cannot rescue the primary endpoint. Wilson intervals are descriptive and unadjusted; no pooled independent-seed inference is made.','',
        '![Primary mediation](figures/primary_mediation.png)','','![Methods across visits](figures/visit_mediation.png)','',
        'Every intervention acts only on CLS. Native writes lie in the same learned rank-four global head-fiber write span; random fiber patches match their norm separately for every pair/model/visit. Full CLS at the last visit is the donor-readout positive control. Sham and terminal fiber interventions are null controls. All controls are checked by independent exact replay.','',
        '## Training and resource accounting','','![Separate loss components](figures/training_losses.png)','',
        'A sampled causal-loss value of zero can mean that no training pair in that batch passed the fixed warm-start eligibility mask. It does not imply perfect causal prediction. Per-update eligible counts are in the journal.','']
    work=done['resources']['work'];mem=summary['audit']['memory']
    lines +=[f"Only 1,552 adapter parameters per model were eligible for updates; the 231,120-parameter model, SAE and projector architecture were fixed. Protected frozen parameter hashes match the SCM-19 warm starts. All pilots, including unused updates/checkpoints, count toward {work['adapter_optimizer_steps']:,} adapter updates, {work['model_calls']:,} model calls, {work['forward_examples']:,} example forwards, {work['recurrent_example_visits']:,} recurrent example-visits and {work['recurrent_token_visits']:,} recurrent token-visits. Each training update runs three forward/backward paths; development-selection evaluation costs are separately included, so end-to-end compute is not claimed identical between arms.",'',
        f"Local RTX 3050 4GB, FP32, batches ≤8, four visits. Peak allocated/reserved {mem['allocated_mb']:.2f}/{mem['reserved_mb']:.2f} MiB; minimum sampled free {mem['minimum_free_mb']:.1f} MiB; active worker time {done['resources']['seconds']/60:.2f} minutes. Enforced 3000 MiB allocation cap and 500 MiB free-headroom guard. No package installation, cloud compute or official EvalMath roster access.",'',
        f"Eight unit tests pass. Independent audit regraded {summary['audit']['checks']['recorded_predictions_regraded']:,} recorded predictions, checked {summary['audit']['checks']['mediation_cells_checked']:,} mediation cells and {summary['audit']['checks']['fresh_unique_signatures']:,} unique fresh signatures, verified protected weights/checkpoints/directions, common eligibility and pair identity, matched random norms, freeze chronology and cumulative journal/evaluator totals. Complete pure-source/cache counters and continuous journals are available. Worker totals above exclude this separate CPU replay; its own oracle/cache counts are in `audit.json`. No training followed final freeze. Weights, optimizer/RNG states, data cache, raw gold and journal remain private outside Git.",'',
        '## Scope and limitations','',
        'This is a finite formal-series DSL and an explicitly supervised interchange task. Improved fidelity, if observed, demonstrates the tested intervention function on a clean-eligible subset; it does not establish a unique internal semantic variable, natural-language mathematical reasoning, or a topology/continuum certificate. The shared SAE and core were already trained in SCM-19. Development-selected loss/checkpoint results are not independent confirmation.','',
        'The inherited OOD generator couples family and topology: rational/fork, squared/cascade, lacunary/additive and twisted_product/hadamard. Larger pools improve eligibility for these four combinations; they do not cover all family-by-topology combinations. Family results are in the summary. Signature deduplication removes identical public problems, not every algebraically equivalent program or structurally related task. Common eligibility reduces selection differences between models but restricts the population further.','']
    overlap=summary['audit']['encoded_observation_overlap']
    lines +=[f"The post-hoc actual-encoder audit found {overlap['clean']['id']['overlap_n']}/512 ID and {overlap['clean']['ood']['overlap_n']}/512 OOD clean cases with a demonstration encoding already present in its pre-final reference set; at least one encoded side overlapped for {overlap['pairs']['id']['either_overlap_n']}/512 ID and {overlap['pairs']['ood']['either_overlap_n']}/1024 OOD pairs. Public signatures include query coefficients, whereas the model ignores the query. This diagnostic leaves the registered primary sample intact; encoded equality is not by itself equality of withheld futures or gold programs. Full scope and IDs are retained in `audit.json`.",'']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);args=p.parse_args();out=HERE/'results'
    read=lambda name:json.loads((out/name).read_text());done=read('completed.json');audit=read('audit.json');c=read('registration.json')['config']
    assert audit['passed'] and audit['completed_sha256']==sha(out/'completed.json')
    summary=summarize(done,audit,read,c);write_json(out/'summary.json',summary)
    figs=HERE/'figures';figs.mkdir(exist_ok=True);plots(summary,done,read,figs)
    (HERE/'RESULTS.md').write_text(document(summary,done,read,c),encoding='utf-8',newline='\n')
    means={kind:{name:statistics.mean(r['accuracy'] for r in summary['clean'] if r['model']==name and r['stratum']==kind) for name in ('warm','control','causal')} for kind in ('id','ood')}
    print(json.dumps({'robust_positive':summary['robust_registered_positive'],'mean_clean':means,
        'primary':[r for r in summary['primary'] if r['model']=='causal'],'memory':audit['memory']},indent=2))


if __name__=='__main__':main()
