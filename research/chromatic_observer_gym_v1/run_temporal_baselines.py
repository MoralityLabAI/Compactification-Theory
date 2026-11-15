"""Post-registered bandwidth controls. Uses frozen fit-only data, then holdout."""
import argparse,json
from pathlib import Path
from temporal_observers import fit_greedy_equal_bit_baseline

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--input',type=Path,default=Path(__file__).with_name('frozen_loop_activations.json'))
 ap.add_argument('--out',type=Path,default=Path(__file__).with_name('equal_budget_baselines.json'))
 args=ap.parse_args()
 from frozen_loader import read_payload
 data=read_payload(args.input.name,args.input.parent)
 rows=[]
 for item in data['items']:
  seed=item['model_seed']
  r=fit_greedy_equal_bit_baseline(item,k=4,untied=False)
  u=fit_greedy_equal_bit_baseline(item,k=4,untied=True)
  rows.append({'seed':seed,'final_four_bits':{k:v for k,v in r.items() if k!='heads'},'untied_four_visits':{k:v for k,v in u.items() if k!='heads'},'final_heads':[{'w':list(x[0][0]),'threshold':x[0][1],'visit':x[1]} for x in r['heads']],'untied_heads':[{'w':list(x[0][0]),'threshold':x[0][1],'visit':x[1]} for x in u['heads']]})
  print(seed,'final4',r['heldout_edge_hits'],'untied4',u['heldout_edge_hits'],'train',r['fit_edge_hits'],u['fit_edge_hits'])
 args.out.write_text(json.dumps({'schema':'COE-04-equal-bits-baselines-v1','rows':rows},indent=2,sort_keys=True)+'\n')
