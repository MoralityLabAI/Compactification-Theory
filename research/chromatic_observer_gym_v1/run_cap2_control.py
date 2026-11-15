"""Post-hoc COE-04C: fit-only hard diameter-two constraint for temporal bits."""
import argparse,json
from pathlib import Path
from frozen_loader import read_payload
from temporal_observers import candidate_projections,encode,covered_pairs,FIT_CONTEXTS,TEST_CONTEXTS,EDGES

def diameter(observations):
 return max((sum(x!=y for x,y in zip(row[a],row[b]))
             for row in observations for a,b in EDGES),default=0)

def fit_cap2(item):
 candidates=candidate_projections(item,seed=item['model_seed']+601)
 best=None; feasible=0
 for head in candidates:
  train=encode(item,head,FIT_CONTEXTS,'temporal')
  if diameter(train)>2:continue
  feasible+=1
  score=covered_pairs(train)
  if best is None or score>best['fit_edge_hits']:
   best={'head':head,'fit_edge_hits':score}
 if best is None:raise ValueError('no cap2 candidate')
 held=encode(item,best['head'],TEST_CONTEXTS,'temporal')
 return {'seed':item['model_seed'],'candidates_checked':len(candidates),
         'fit_feasible_candidates':feasible,
         'fit_edge_hits':best['fit_edge_hits'],
         'fit_diameter':diameter(encode(item,best['head'],FIT_CONTEXTS,'temporal')),
         'heldout_edge_hits':covered_pairs(held),'heldout_diameter':diameter(held),
         'weight':list(best['head'][0]),'threshold':best['head'][1]}

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--input',type=Path,default=Path(__file__).with_name('frozen_loop_activations.json.gz'))
 ap.add_argument('--out',type=Path,default=Path(__file__).with_name('cap2_receipt.json'))
 args=ap.parse_args()
 from frozen_loader import read_payload
 data=read_payload(args.input.name,args.input.parent)
 rows=[fit_cap2(item) for item in data['items']]
 for row in rows:print(row['seed'], row['fit_edge_hits'],row['heldout_edge_hits'],row['heldout_diameter'])
 args.out.write_text(json.dumps({'schema':'COE-04-cap2-diagnostic-v1','posthoc':True,'rows':rows},indent=2,sort_keys=True)+'\n')
