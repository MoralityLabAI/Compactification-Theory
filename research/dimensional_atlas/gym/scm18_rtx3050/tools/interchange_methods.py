"""Additional exploratory mediation audit for every compression/intervention family.

This audit was added after development, before official final evaluation. It
cannot select models or update calibration frames, and reports no fidelity when
the registered clean-recipient/donor validity criterion has no eligible pairs.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from research.dimensional_atlas.gym.tasks import PROGRAMS,TOPOLOGIES,encode,load_evalmath,make_evalmath_case
from research.dimensional_atlas.gym.scm18_rtx3050.core import (
    HERE,Meter,device_setup,future_correct,load_checkpoint,sha,write_new)
from research.dimensional_atlas.gym.scm18_rtx3050.campaign import OracleAudit


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();c=json.loads((HERE/'config.json').read_text())
    frozen=json.loads((args.root/'results/freeze.json').read_text())
    assert sha(args.private/'frame.pt')==frozen['frame_sha256']
    device=device_setup('cuda',c)
    model,_=load_checkpoint(args.private/frozen['primary_checkpoint'],c,device)
    meter=Meter(device,c);model.meter=meter
    oracle=load_evalmath(args.oracle);audit=OracleAudit(oracle)
    frame=torch.load(args.private/'frame.pt',weights_only=True)
    q=model.vpd.readout_basis;order=frame['feature_order'].to(device)
    pairs=[]
    for i in range(16):
        graph=PROGRAMS[(i*7)%64];changed=(('N' if graph[0]!='N' else 'R'),*graph[1:])
        a=make_evalmath_case(oracle,1870000+i,case_id=f'causal_recipient:{i}',family='lacunary',graph=graph,topology=TOPOLOGIES[i%4])
        b=make_evalmath_case(oracle,1870000+i,case_id=f'causal_donor:{i}',family='lacunary',graph=changed,topology=TOPOLOGIES[i%4])
        oa=model(encode(a)[None].to(device),policy='joint')
        ob=model(encode(b)[None].to(device),policy='joint')
        eligible=(a.hidden_target!=b.hidden_target and future_correct(oracle,a,oa.logits.argmax())
                  and future_correct(oracle,b,ob.logits.argmax()))
        if eligible:
            pairs.append((a,b,tuple(s.cpu() for s in oa.states),tuple(s.cpu() for s in ob.states)))
        del oa,ob
    rows=[]
    for v in range(model.visits):
        for d in c['retained_dimensions']:
            if d>model.hidden:continue
            pca=frame['pca'][v][:,:d].to(device)
            feature=torch.linalg.qr(model.sae.decoder.weight[:,order[:d]].detach()).Q
            candidate=pca-q@(q.T@pca)
            u,s,_=torch.linalg.svd(candidate,full_matrices=False)
            fiber=u[:,s>1e-5]
            for method in ('pca','sae_groups','sae_reconstruction','head_readable','fiber_preservation','sae_conditioned_vpd'):
                if method=='sae_conditioned_vpd' and d!=frozen['selected_rank']:continue
                correct=0
                for a,b,sa,sb in pairs:
                    recipient=sa[v].to(device);donor=sb[v].to(device);difference=donor-recipient
                    if method=='pca':delta=(difference@pca)@pca.T
                    elif method=='sae_groups':delta=(difference@feature)@feature.T
                    elif method=='head_readable':
                        qq=q[:,:min(d,q.shape[1])];delta=(difference@qq)@qq.T
                    elif method=='fiber_preservation':
                        delta=(difference@q)@q.T+(difference@fiber)@fiber.T
                    elif method=='sae_reconstruction':
                        ca=model.sae.encode(F.layer_norm(recipient,(model.hidden,)))
                        cb=model.sae.encode(F.layer_norm(donor,(model.hidden,)))
                        code=torch.zeros_like(ca);code[...,order[:d]]=cb[...,order[:d]]-ca[...,order[:d]]
                        delta=F.linear(code,model.sae.decoder.weight)
                    else:
                        ca=model.sae.encode(F.layer_norm(recipient,(model.hidden,)))
                        cb=model.sae.encode(F.layer_norm(donor,(model.hidden,)))
                        da,*_=model.vpd(recipient,ca,v,sector='fiber_only')
                        db,*_=model.vpd(donor,cb,v,sector='fiber_only')
                        delta=db-da
                    out=model(encode(a)[None].to(device),policy='joint',hook=lambda at,h:h+delta if at==v else h)
                    correct+=int(future_correct(oracle,b,out.logits.argmax()))
                actual=(min(d,q.shape[1]) if method=='head_readable' else
                        q.shape[1]+fiber.shape[1] if method=='fiber_preservation' else
                        frozen['selected_rank'] if method=='sae_conditioned_vpd' else d)
                rows.append({'visit':v,'method':method,'requested_dimension':d,
                    'retained_dimension_or_feature_count':actual,
                    'valid_counterfactual_tests':len(pairs),'correct_counterfactual_outcomes':correct,
                    'causal_fidelity':correct/len(pairs) if pairs else None})
    write_new(args.root/'results/interchange_methods.json',{
        'status':'additional_exploratory_post_development_frozen_model_audit',
        'freeze_sha256':sha(args.root/'results/freeze.json'),'source_sha256':sha(__file__),
        'valid_pairs':len(pairs),'attempted_pairs':16,'records':rows,'resources':meter.report(),
        'oracle_source_calls':dict(audit.counts),
        'note':'VPD uses difference of native fiber corrections at donor/recipient post-visit states; SAE count is retained feature count'})


if __name__=='__main__':main()
