"""Post-freeze exploratory native-direction mediation across all three seeds.

No new direction search: use each checkpoint's frozen head and native adapter
write span. The complete-state and terminal readable swaps are positive controls;
terminal fiber swaps are linear-head null controls.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from research.dimensional_atlas.gym.tasks import PROGRAMS,TOPOLOGIES,encode,load_evalmath,make_evalmath_case
from research.dimensional_atlas.gym.scm18_rtx3050.core import (
    HERE,Meter,device_setup,future_correct,load_checkpoint,operator_counterfactual,
    release,sha,write_new)
from research.dimensional_atlas.gym.scm18_rtx3050.campaign import OracleAudit


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();c=json.loads((HERE/'config.json').read_text())
    f=json.loads((args.root/'results/freeze.json').read_text())
    assert (args.root/'results/final.json').is_file(),'run sequentially after main final readout'
    device=device_setup('cuda',c);oracle=load_evalmath(args.oracle);audit=OracleAudit(oracle)
    trials=[]
    for seed in c['seeds']:
        ck=next(v for v in f['models'] if v['seed']==seed and v['objective']=='final'
                and v.get('sector')=='joint' and not v.get('random_dictionary') and v['rank']==f['selected_rank'])
        assert sha(args.private/ck['file'])==ck['file_sha256']
        model,_=load_checkpoint(args.private/ck['file'],c,device)
        meter=Meter(device,c);model.meter=meter;q=model.vpd.readout_basis
        write=model.vpd.write_factor.detach().T
        write=write-q@(q.T@write)
        u,s,_=torch.linalg.svd(write.cpu().double(),full_matrices=False)
        basis=u[:,s>s.max().clamp_min(1e-20)*1e-6].float().to(device)
        pairs=[]
        for i in range(16):
            a=make_evalmath_case(oracle,1870000+i,case_id=f'native_recipient:{i}',family='lacunary',
                                 graph=PROGRAMS[(i*7)%64],topology=TOPOLOGIES[i%4])
            b=operator_counterfactual(a,oracle)
            oa=model(encode(a)[None].to(device),policy='joint')
            ob=model(encode(b)[None].to(device),policy='joint')
            eligible=(a.hidden_target!=b.hidden_target and future_correct(oracle,a,oa.logits.argmax())
                      and future_correct(oracle,b,ob.logits.argmax()))
            if eligible:pairs.append((a,b,tuple(s.cpu() for s in oa.states),tuple(s.cpu() for s in ob.states)))
            del oa,ob
        rows=[]
        for visit in range(model.visits):
            for method,dimension in [('full_state',model.hidden),('head_readable',q.shape[1]),
                ('head_fiber',model.hidden-q.shape[1]),('native_write_fiber',basis.shape[1]),
                ('sae_conditioned_fiber_vpd',f['selected_rank'])]:
                correct=0;norms=[]
                for a,b,sa,sb in pairs:
                    recipient=sa[visit].to(device);donor=sb[visit].to(device);difference=donor-recipient
                    visible=(difference@q)@q.T
                    if method=='full_state':delta=difference
                    elif method=='head_readable':delta=visible
                    elif method=='head_fiber':delta=difference-visible
                    elif method=='native_write_fiber':delta=(difference@basis)@basis.T
                    else:
                        ca=model.sae.encode(F.layer_norm(recipient,(model.hidden,)))
                        cb=model.sae.encode(F.layer_norm(donor,(model.hidden,)))
                        da,*_=model.vpd(recipient,ca,visit,sector='fiber_only')
                        db,*_=model.vpd(donor,cb,visit,sector='fiber_only');delta=db-da
                    norms.append(float(delta.norm()))
                    out=model(encode(a)[None].to(device),policy='joint',
                              hook=lambda v,h:h+delta if v==visit else h)
                    correct+=int(future_correct(oracle,b,out.logits.argmax()))
                rows.append({'visit':visit,'method':method,'dimension':dimension,
                    'valid_counterfactual_tests':len(pairs),'correct_counterfactual_outcomes':correct,
                    'causal_fidelity':correct/len(pairs) if pairs else None,
                    'patch_norms':norms})
        trials.append({'seed':seed,'checkpoint':ck,'valid_pairs':len(pairs),
                       'attempted_shared_pairs':16,'records':rows,'resources':meter.report()})
        if pairs:
            del out,recipient,donor,difference,visible,delta,ca,cb,da,db
        del model,meter,pairs,q,basis,write;release()
    write_new(args.root/'results/native_mediation.json',{
        'status':'exploratory_post_final_audit_no_model_or_direction_selection',
        'freeze_sha256':sha(args.root/'results/freeze.json'),'source_sha256':sha(__file__),
        'shared_pairs_across_seeds':16,'trials':trials,'oracle_source_calls':dict(audit.counts),
        'interpretation':'Native directions were fixed by frozen training weights; patch norms differ by sector. These are additional diagnostics, not a new preregistered success gate.'})


if __name__=='__main__':main()
