"""Repair passive symbolic hypotheses using the same teacher and budget as neural repair."""
from gym import *
from audit import independent
root=pathlib.Path(sys.argv[1]);rows=[]
for r in json.loads((root/'passive_receipt.json').read_text()):
    if not r['train_plus_validation']:continue
    source=next(x for x in json.loads((root/r['split']/'receipt.json').read_text())['rows'] if x['seed']==r['seed']);t,b=source['oracle_table'],source['oracle_accept'];u,c=np.array(r['table']),np.array(r['accept']);ss=closure(probes(u,c));arms={}
    for mode in ('exact','sampled'):
        a=learn(t,b,mode,r['seed']+7000,ss,(u,c));assert independent(t,b,a['table'],a['accept'])[0]==a['exact'];arms[mode]=a
    rows.append(dict(seed=r['seed'],n=r['n'],split=r['split'],arms=arms))
(root/'passive_repair_receipt.json').write_text(json.dumps(rows,indent=2)+'\n')
for mode in ('exact','sampled'):
    a=[x['arms'][mode] for x in rows];print(mode,sum(x['exact'] for x in a),sum(x['mq'] for x in a),sum(x['counts']['eq'] for x in a),sum(x['mq_tokens'] for x in a),sum(x['seconds'] for x in a))
