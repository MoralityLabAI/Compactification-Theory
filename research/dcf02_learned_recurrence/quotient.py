"""DCF-02D: exhaustive discrete transition audit and reset rollout."""
from train import *
from closure import ClosedModel

@torch.inference_mode()
def audit(m):
    s=torch.arange(8).repeat_interleave(32); x=torch.arange(32).repeat(8)[:,None]
    logits=m(s,x)[:,0]; y=((2*(x[:,0]//8)+1)*s+x[:,0]%8)%8
    tab=logits.argmax(-1).reshape(8,32); hits=(tab.flatten()==y)
    target=logits.gather(1,y[:,None])[:,0]; alt=logits.masked_fill(F.one_hot(y,8).bool(),float('-inf')).max(1).values
    return tab,int(hits.sum()),float((target-alt).min())

@torch.inference_mode()
def rollout(m,tab,seed):
    s,x,y=data(torch.Generator().manual_seed(seed),512,64); h=s; q=s
    mismatches=0
    for step in x.T:
        h=m(h,step[:,None])[:,0].argmax(-1); q=tab[q,step]; mismatches+=int((q!=h).sum())
    free=m(s,x)[:,-1].argmax(-1)
    return dict(n=len(s),length=x.shape[1],free_correct=int((free==y[:,-1]).sum()),reset_correct=int((h==y[:,-1]).sum()),table_correct=int((q==y[:,-1]).sum()),reset_table_prefix_mismatches=mismatches)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); rows=[]
    for group in ('run','curriculum_run','closure_run'):
        for path in sorted((a.root/group).glob('*.pt')):
            ck=torch.load(path,weights_only=True); typ=ClosedModel if ck.get('bounded_initial') else Model
            m=typ(ck['d'],ck['r']); m.load_state_dict(ck['state']); m.eval()
            tab,n,margin=audit(m); row=dict(group=group,checkpoint=path.name,correct_transitions=n,min_target_margin=margin,transition_table=tab.tolist(),**rollout(m,tab,198273)); rows.append(row)
    (a.root/'quotient_receipt.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    for group in ('run','curriculum_run','closure_run'):
        rs=[r for r in rows if r['group']==group]; exact=[r for r in rs if r['correct_transitions']==256]
        print(group,len(exact),len(rs),'min exact margin',min((r['min_target_margin'] for r in exact),default=None),'exact reset correct',sum(r['reset_correct'] for r in exact),'/',512*len(exact),'mismatches',sum(r['reset_table_prefix_mismatches'] for r in exact))
        print([(r['checkpoint'],r['free_correct'],r['reset_correct']) for r in exact])

if __name__=='__main__': main()
