"""OIC-19 validation-only entropy lambda selection, using training phase counts.

Choose minimum held-in full-probe log loss over n=4/5/6, plus a fixed
1e-4*log2(phase slots) numerical tie penalty; no n>=7 outcomes used.
"""
from __future__ import annotations
import json, concurrent.futures, time
from pathlib import Path
from oic19_train import train_one, validation_logloss

CANDIDATES=(.02,.06,.15)
SEEDS=(0,1)
SPLITS=('full',)


def single(arg):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lam,seed,split=arg
    model,history=train_one('entropy',split,seed,lam,steps=150)
    score=validation_logloss(model,split)
    return {'lambda':lam,'seed':seed,'split':split,'heldin_phase_family_logloss':round(score,9),
            'kernel_entropy_bits':history['checkpoint'][-1]['penalty']/(lam*0.6931471805599453),
            'training_wall_seconds':history['wall_seconds']}


def main():
    start=time.perf_counter()
    args=[(lam,seed,split) for lam in CANDIDATES for seed in SEEDS for split in SPLITS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(single,args))
    scores={str(lam):sum(row['heldin_phase_family_logloss'] for row in rows if row['lambda']==lam)/len(SEEDS) for lam in CANDIDATES}
    chosen=min(CANDIDATES,key=lambda lam:(scores[str(lam)],lam))
    result={'selection_criterion':'lowest mean held-in n4/5/6 full_active family logloss, phases restricted to training split',
            'candidates':list(CANDIDATES),'validation_seeds':list(SEEDS),
            'validation_training_fits':len(rows),'validation_fits_each_150_steps_batch_128':True,
            'scores':scores,'chosen_lambda':chosen,'rows':rows,
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'test_n7_and_above_read_during_selection':False}
    out=Path('results/oic19_lambda_selection.json');out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'scores':scores,'chosen':chosen,'fits':len(rows),'elapsed':result['elapsed_seconds']},sort_keys=True),flush=True)

if __name__=='__main__':main()
