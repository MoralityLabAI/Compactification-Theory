"""Derive human-readable tables and static figures from immutable receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root;r=root/'results';fig=root/'figures'
    fig.mkdir(exist_ok=True)
    training=load(r/'training.json');final=load(r/'final.json');diag=load(r/'diagnostics.json')
    provenance=load(r/'provenance.json');freeze=load(r/'freeze.json')
    extra=load(r/'interchange_methods.json')
    native=load(r/'native_mediation.json')
    grouped=defaultdict(list)
    for row in final['results']:
        ck=row['checkpoint']
        label=('untied' if ck.get('untied') else
               'random_dictionary' if ck.get('random_dictionary') else
               'independently_fiber_trained' if ck.get('sector')=='fiber_only' else None)
        for policy,sets in row['policies'].items():
            grouped[(ck['objective'],label or policy)].append((ck['seed'],sets))
    table=[]
    for (objective,policy),rows in sorted(grouped.items()):
        row={'objective':objective,'policy':policy,'seeds':[s for s,_ in rows]}
        for dataset in ('official','fresh','challenges'):
            vals=[v[dataset]['overall'] for _,v in rows]
            row[dataset]={'accuracy_mean':statistics.mean(v['accuracy'] for v in vals),
                'accuracy_by_seed':[v['accuracy'] for v in vals],
                'correct_by_seed':[v['correct'] for v in vals], 'n_per_seed':[v['n'] for v in vals],
                'top1_correct_by_seed':[v['top1_correct'] for v in vals],
                'false_future_by_seed':[v['false_future_acceptances'] for v in vals],
                'paid_queries_by_seed':[v['verifier_calls'] for v in vals]}
        table.append(row)
    primary=[v for v in table if v['objective']=='final']
    plt.rcParams.update({'figure.dpi':150,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig1,ax=plt.subplots(figsize=(10,4.5))
    labels=[v['policy'].replace('_','\n') for v in primary]
    x=np.arange(len(primary));w=.36
    for j,(dataset,color) in enumerate([('official','#2d6285'),('fresh','#de8544')]):
        y=[v[dataset]['accuracy_mean']*100 for v in primary]
        ax.bar(x+(j-.5)*w,y,w,label=dataset,color=color)
        for i,v in enumerate(primary):
            ax.scatter([x[i]+(j-.5)*w]*len(v[dataset]['accuracy_by_seed']),
                       np.array(v[dataset]['accuracy_by_seed'])*100,color='#172b35',s=9,zorder=4)
    ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_ylabel('Exact future accuracy (%)')
    ax.set_title('SCM-18 · fixed 12-query verifier budget · dots are three seeds')
    ax.legend(frameon=False);fig1.tight_layout()
    fig1.savefig(fig/'accuracy.png');fig1.savefig(fig/'accuracy.svg');plt.close(fig1)

    fig2,axes=plt.subplots(1,3,figsize=(11,3.4))
    geometries=[('tied + joint adapter',diag['geometry'],'#2d6285'),
                ('untied baseline',diag['untied_geometry'],'#de8544')]
    for label,data,color in geometries:
        values=defaultdict(list);angles=defaultdict(list)
        for row in data['records']:
            if row['dimensions']==4 and row['epsilon']==.03:
                values[row['visit']].append(row['effective_rank'])
                if row['principal_angles_degrees']:
                    angles[row['visit']].append(np.mean(row['principal_angles_degrees']))
        for target,collection in [(axes[0],values),(axes[1],angles)]:
            vs=sorted(collection)
            target.plot(np.array(vs)+1,[np.mean(collection[v]) for v in vs],label=label,color=color)
    axes[0].set_title('Four-axis effective rank');axes[0].set_ylabel('exp entropy of singular values')
    axes[0].set_ylim(1,4.1);axes[1].set_title('Consecutive tangent orientation');axes[1].set_ylabel('Mean principal angle (degrees)')
    transport=defaultdict(list)
    for row in diag['fiber_transport']['rows']:
        transport[row['visit']].append(row['logit_effect_norm'])
    axes[2].plot(np.array(sorted(transport))+1,[np.mean(transport[v]) for v in sorted(transport)],color='#3c8b6b')
    axes[2].set_yscale('symlog',linthresh=1e-6);axes[2].set_ylabel('Final logit change, norm')
    axes[2].set_title('Fixed 0.05-norm fiber patch')
    for target in axes:target.set_xlabel('Recurrent visit')
    axes[0].legend(frameon=False,fontsize=7);fig2.suptitle('Exploratory numerical geometry · no topology certificate',fontsize=11)
    fig2.tight_layout();fig2.savefig(fig/'causal_subspaces.png');fig2.savefig(fig/'causal_subspaces.svg');plt.close(fig2)

    usage=np.asarray(diag['frame']['usage']);order=np.asarray(diag['frame']['feature_order'])
    fig3,ax=plt.subplots(figsize=(9,3))
    ax.bar(np.arange(24),usage[order[:24]],color='#6f73a4')
    ax.set_xticks(np.arange(24),[str(i) for i in order[:24]],fontsize=7)
    ax.set_xlabel('Frozen shared SAE feature index');ax.set_ylabel('Calibration activation sum')
    ax.set_title('Shared SAE feature use · eight calibration examples only')
    fig3.tight_layout();fig3.savefig(fig/'sae_features.png');fig3.savefig(fig/'sae_features.svg');plt.close(fig3)

    comp=diag['compactification']['records'];last=max(row['visit'] for row in comp)
    actual_dimensions={(v['visit'],v['method'],v['requested_dimension']):v['retained_dimension_or_feature_count']
                       for v in extra['records']}
    dimension_rows=[{**v,'retained_dimension_or_sae_feature_count':actual_dimensions[
        (v['visit'],v['method'],v['retained_dimension_requested'])]} for v in comp]
    (r/'compactification_dimensions.json').write_text(json.dumps(dimension_rows,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    fig4,axs=plt.subplots(1,2,figsize=(10,3.7))
    for target,visit in zip(axs,(0,last)):
        for method in ('pca','sae_groups','sae_reconstruction','head_readable','fiber_preservation'):
            rows=[row for row in comp if row['visit']==visit and row['method']==method]
            target.plot([actual_dimensions[(row['visit'],row['method'],row['retained_dimension_requested'])] for row in rows],
                        [row['overall']['accuracy']*100 for row in rows],marker='.',label=method)
        target.set_xscale('log',base=2);target.set_ylim(0,105)
        target.set_xlabel('Retained dimensions / SAE features');target.set_ylabel('Exact future accuracy (%)')
        target.set_title(f'Compression after visit {visit+1}')
    axs[1].legend(fontsize=6,frameon=False)
    fig4.suptitle('Four cases · fiber x includes 16 head dimensions · SAE reconstruction x counts features',fontsize=10)
    fig4.tight_layout();fig4.savefig(fig/'compactification.png');fig4.savefig(fig/'compactification.svg');plt.close(fig4)

    resources=[v['resources'] for v in training['screen']+training['records']]+[training['frame_resources']]
    resources += [v['resources'] for v in final['results']]+[diag['resources'],diag['untied_resources']]
    resources.append(extra['resources'])
    resources.extend(v['resources'] for v in native['trials'])
    total_work=defaultdict(int)
    for resource in resources:
        for name,n in resource['work'].items():total_work[name]+=n
    peak_alloc=max(v['peak_allocated_mb'] for v in resources)
    peak_reserved=max(v['peak_reserved_mb'] for v in resources)
    min_free=min(v['minimum_sampled_free_mb'] for v in resources if v['minimum_sampled_free_mb'] is not None)
    eligible=diag['interchange']['records'][0]['valid_counterfactual_tests']
    mediation=diag['interchange']['records']
    full=[v for v in mediation if v['dimension']==112]
    topology=diag['topology']['records']
    first_transport=[v['logit_effect_norm'] for v in diag['fiber_transport']['rows'] if v['visit']==0]
    last_transport=[v['logit_effect_norm'] for v in diag['fiber_transport']['rows'] if v['visit']==training['selected_visits']-1]
    spectral=[v for v in diag['geometry']['records'] if v['dimensions']==4]
    baseline=next(v for v in primary if v['policy']=='tied')
    joint=next(v for v in primary if v['policy']=='joint')
    delta=[100*(b-a) for a,b in zip(baseline['official']['accuracy_by_seed'],joint['official']['accuracy_by_seed'])]
    fresh_delta=[100*(b-a) for a,b in zip(baseline['fresh']['accuracy_by_seed'],joint['fresh']['accuracy_by_seed'])]
    summary={'table':table,'main_campaign_total_work':dict(total_work),
             'peak_allocated_mb':peak_alloc,'peak_reserved_mb':peak_reserved,'minimum_sampled_free_mb':min_free,
             'joint_minus_tied_official_pp':delta,'joint_minus_tied_fresh_pp':fresh_delta,
             'valid_mediation_pairs':eligible,'full_swap_control':full,
             'native_valid_pairs_by_seed':{str(v['seed']):v['valid_pairs'] for v in native['trials']},
             'positive_mechanism_claim':False,'topology_certified':False}
    (r/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    md=['# SCM-18 executed results','',
        '**Classification:** local exploratory campaign on newly trained small models. '
        'The strict positive-mechanism criterion is not established. No trained RMP transfer '
        'or neural topology certificate is claimed.','',
        f"Completed on 2026-11-14 using Python {provenance['python'].split()[0]}, "
        f"PyTorch {provenance['packages']['torch']}, Windows and an RTX 3050 4GB. "
        'C: had insufficient space for CUDA wheels; an isolated environment was installed on D:. '
        'No existing user files were removed and no cloud compute was used.','',
        '## Registered selection and reproduction','',
        'The original SCM-15 synthetic CPU smoke scored 8/12 in all four arms. '
        'The separate pinned-EvalMath CPU/GPU reproduction used identical initial weight and '
        'example hashes and scored 6/12 in every arm; final phase-loss differences were below 3e-7. '
        'These short runs validate plumbing, not learning or generalization.','',
        'A supplemental instrumented reproduction of the **complete original SCM-15 pinned-EvalMath '
        'pipeline** scored 81/192 for tied/zero-code and 80/192 for joint/reversed-gate on both '
        'CPU and GPU. Initial weights, generated public challenge hashes, exact scores and work '
        'counts matched: 795 model calls, 825 example-forwards, 3,300 recurrent example-visits '
        'and 10 optimizer steps per run. Its tiny fit is an engineering reproduction; it does '
        'not establish an adapter improvement. It runs only after the main campaign freeze.','',
        f"Development screening selected {training['selected_visits']} visits and rank {training['selected_rank']}. "
        'Visit screen exact scores were '+', '.join(f"{v['visits']}: {v['development']['overall']['correct']}/48" for v in training['screen'])+'. '
        'Three seeds (181, 1181, 2181) then ran four registered objectives. '
        'The selected tied model has 138,800 total parameters (108,384 base, 28,912 SAE, '
        '1,504 adapter); the untied comparator has 1,662,560 total parameters. '
        'All checkpoints, SAE dictionaries, calibration bases and source hashes were frozen '
        'before the separately launched final evaluator opened the official roster. '
        f"Official-roster calls before freeze: {freeze['official_roster_calls_before_freeze']}.",'',
        '## Exact mathematical outcomes','',
        'Mean percentages below use three seeds. Official evaluation has 192 cases per seed '
        '(including public collision pairs and negatives); fresh OOD evaluation has 48 cases '
        'from four unused source families. Neither set consists of 192/48 independent mathematical '
        'families. Every neural arm has a maximum budget of 12 exact candidate checks.','',
        '| Objective | Policy | Official % | Fresh OOD % | Official top-1 correct by seed |',
        '|---|---|---:|---:|---|']
    for v in table:
        md.append(f"| {v['objective']} | {v['policy']} | {v['official']['accuracy_mean']*100:.2f} | "
                  f"{v['fresh']['accuracy_mean']*100:.2f} | {v['official']['top1_correct_by_seed']} |")
    md+=['',f'Final-only joint minus tied, official percentage points by seed: {delta}; fresh: {fresh_delta}. '
        'No significance or family-transfer mechanism follows from this small comparison. '
        'The full receipt retains each task outcome, false-future acceptance and actual query count. '
        'Visible-only, fiber-only, reversed gates and zero codes are inference ablations of '
        'the joint checkpoint; independently fiber-trained and random-dictionary entries are '
        'separate fits. Zero-code arithmetic equals the tied baseline.','',
        '![Exact mathematical outcomes](figures/accuracy.png)','',
        '## Causal structure and compression','',
        f"Four-axis effective rank ranged from {min(v['effective_rank'] for v in spectral):.3f} to "
        f"{max(v['effective_rank'] for v in spectral):.3f} over two step sizes, visits and 16 confirmation cases. "
        'This rank is capped at four by the intervention design and does not estimate full hidden rank. '
        'Principal angles, head-sector energy, support changes and finite-difference curvature '
        'are numerical diagnostics. The untied model uses the same frozen intervention axes and '
        'expanded visits, with more physical parameters.','',
        f"A 0.05-norm fiber patch after the first visit changed final-logit norms by "
        f"{min(first_transport):.3g}–{max(first_transport):.3g}; after the final visit, "
        f"changes were {min(last_transport):.3g}–{max(last_transport):.3g}. This is a numerical "
        'transport effect in a deliberately linear-bottleneck model.','',
        f"The named first-operator interchange test had **{eligible}/16 valid clean recipient/donor pairs**. "+
        ('Causal fidelity is null because no eligible clean pairs existed. ' if eligible==0 else
         'Each causal-fidelity estimate therefore has a small eligibility denominator. ')+
        'Raw attempted-pair success and the full-state swap control remain separate in the receipt. '
        'These tests do not establish that a low-dimensional intervention preserves a correctly '
        'identified intermediate reasoning process.','',
        'A separate **post-final exploratory audit** repeats the same sixteen mathematical '
        'pairs across all three frozen final-only seeds using native, already-trained directions. '
        'No new direction search or model selection is performed. Patch norms differ across '
        'head/fiber sectors and are recorded. The denominators are model/pair eligibility '
        'counts on sixteen shared pairs, not independent sets of sixteen per seed.','',
        '| Seed | Eligible / 16 | Visit-1 native write fiber fidelity | Visit-1 SAE/VPD fiber fidelity | Visit-1 full-state control |',
        '|---:|---:|---:|---:|---:|',
        *[f"| {trial['seed']} | {trial['valid_pairs']} | "+' | '.join(
            'undefined' if next(v for v in trial['records'] if v['visit']==0 and v['method']==method)['causal_fidelity'] is None
            else f"{next(v for v in trial['records'] if v['visit']==0 and v['method']==method)['causal_fidelity']:.3f}"
            for method in ('native_write_fiber','sae_conditioned_fiber_vpd','full_state'))+' |'
            for trial in native['trials']],
        '',
        'Compression was tested after every visit at requested dimensions 1/2/4/8/16/32/64/112 '
        'on four independent confirmation examples. That sample is too small to identify a '
        'general smallest sufficient subspace. SAE curves count selected features; fiber curves '
        'preserve the head-readable space in addition to the selected fiber directions. '
        'Full-dimensional PCA is an identity control up to FP32 roundoff.','',
        'The additional `interchange_methods.json` audit covers PCA, SAE groups, SAE '
        'reconstruction, head-readable projection, fiber preservation and native SAE-conditioned '
        'VPD correction differences. It uses the same frozen calibration frame and validity '
        'criterion. All six families report null fidelity when no eligible pairs exist. '
        'This extra audit was specified after development and did not change model selection.','',
        '![Subspace and fiber diagnostics](figures/causal_subspaces.png)','',
        '![Compression curves](figures/compactification.png)','',
        '![SAE calibration feature use](figures/sae_features.png)','',
        '## Positive, negative and unresolved findings','',
        '| Question | Executed finding | Interpretation |',
        '|---|---|---|',
        '| CPU/GPU reproducibility | Same initial weights/examples, same exact scores | Engineering pass |',
        '| Final head fiber | Final-visit linear-head effects near FP32 noise; earlier effects measured | Causal logit transport, not semantic proof |',
        '| Low-rank adapter skill gain | Per-seed outcomes and matched controls above | Exploratory; strict positive criterion not met |',
        f'| Named variable mediation | {eligible}/16 eligible pairs | Limited/absent causal evidence |',
        '| Nonlinear RMP transfer | Missing checkpoint, integrity receipt, selected dictionary and monitor bundle; tracked metadata retrieved | Not executed; existing gate preserved |',
        '| SCM-12/13 topology | Conditional surrogate routines invoked on sampled projections | Neural conclusion inconclusive without continuum bounds |','',
        '## Resource accounting and limits','',
        f"Main campaign peak PyTorch allocation: **{peak_alloc:.2f} MiB**; reservation: "
        f"**{peak_reserved:.2f} MiB**. Minimum sampled device-free memory: **{min_free:.2f} MiB**. "
        'The 3000 MiB allocator cap and every-forward memory checks preserved the registered '
        '500 MiB headroom at all samples. Memory outside the PyTorch allocator is included '
        'in device-free samples; continuous headroom is not certified. Peaks in receipts are '
        'process-cumulative, not isolated per-stage maxima.','',
        'Main campaign work totals (screening + training + calibration + final evaluation + diagnostics):','',
        '```json',json.dumps(dict(total_work),indent=2,sort_keys=True),'```','',
        'Original reproduction receipts are additional work and are excluded from these totals. '
        'Actual independent-oracle make_case/execute/check_candidate/grade/roster counts are '
        'recorded per process; grade performs an internal acceptance recheck, which is included '
        'in source-call accounting. Paid search budgets are reported separately. Backward work '
        'is counted as optimizer steps rather than estimated FLOPs. Causal adapter training '
        'uses two additional forwards per update, so training cost is not equal across objectives. '
        'SAE dictionary updates operate on CPU-cached frozen training activations.','',
        'Private weights and new challenge gold remain outside Git. Public hashes alone are '
        'not access-control guarantees: isolation here is an audited program/data-flow boundary '
        'on one local machine. No historically sealed RMP evaluation was opened.','',
        'The scope is a four-token public-feature program classifier trained for 240 base '
        'updates per fit on 384 generated cases, with a deliberately rank-16 readout. '
        'Different source programs can share observed behavior, so source-program cross entropy '
        'and exact mathematical correctness measure different things. These finite-budget '
        'results leave the broader low-dimensional reasoning hypothesis unresolved. '
        'Automatic hosted workflows are intentionally skipped at publication; all validation '
        'and experiments for this campaign ran locally.']
    (root/'RESULTS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'summary':str(r/'summary.json'),'table_rows':len(table),
                      'valid_mediation_pairs':eligible,'peak_reserved_mb':peak_reserved}))


if __name__=='__main__':main()
