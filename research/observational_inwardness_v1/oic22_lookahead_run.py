"""Optional exact 4-step receding Bayes-planner comparator; NumPy dependency."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from oic22_lookahead import FaultAwarePlanner

HERE=Path(__file__).resolve().parent
PINNED=HERE/'results'/'oic22_lookahead.json'

def run():
    rows=[]
    for n in (6,8):
        planner=FaultAwarePlanner(n,depth=4,clean_weight=1,flip_weight=3)
        panels=[{'kind':kind,'correct':correct,'episodes':total,
                 'accuracy':correct/total} for kind,correct,total,_ in planner.panels()]
        rows.append({'n':n,'Q':n+2,'lookahead_steps':4,
                     'objective_clean_weight':1,'objective_single_flip_weight':3,
                     'panels':panels})
    return {'tag':'OIC-22-LOOKAHEAD-01','scope':'exploratory exact belief, limited 4-step continuation, unmatched inference CPU costs','results':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    data=run()
    if a.check:
        if data!=json.loads(PINNED.read_text()):raise SystemExit('lookahead receipt mismatch')
        print('OIC-22 receding lookahead replay PASS')
    if a.out:a.out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    if not a.check:print(json.dumps(data,indent=2,sort_keys=True))
if __name__=='__main__':main()
