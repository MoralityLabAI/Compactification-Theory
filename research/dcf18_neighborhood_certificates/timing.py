from benchmark import *

def hybrid(f,q,depth):
 for i,r in enumerate(regions(f,q)):
  a=certified(bounds(f,q,r))
  if a is not None:return a,i
  if i==depth:break
 return action(eliminate(f,q)),None

def run_timing():
 rows=[]; rng=random.Random(1901)
 for n,fam,seed in it.product((12,24,48),('cycle','ladder','chords'),range(24)):
  f,q=fixture(n,fam,180000+n*1000+100*('cycle','ladder','chords').index(fam)+seed)
  p=eliminate(f,q);a=action(p);arms={'full':lambda: (action(eliminate(f,q)),None),**{str(d):lambda d=d:hybrid(f,q,d) for d in range(3)}}
  for fn in arms.values():assert fn()[0]==a
  samples={k:[] for k in arms}
  for _ in range(5):
   order=list(arms);rng.shuffle(order)
   for k in order:
    t=time.perf_counter_ns();result=arms[k]();samples[k].append(time.perf_counter_ns()-t);assert result[0]==a
  rows.append(dict(n=n,family=fam,seed=seed,action=a,near=min(abs(float(p)-.2),abs(float(p)-.8))<=.05,samples=samples,median={k:statistics.median(v) for k,v in samples.items()}))
 Path(__file__).with_name('timing_receipt.json').write_text(json.dumps(rows,indent=2))
 for k in arms:
  print(k,statistics.mean(x['median'][k] for x in rows)/1e6,statistics.median(x['median']['full']/x['median'][k] for x in rows),sum(x['median'][k]<=x['median']['full'] for x in rows))
 print('actions', {a:sum(x['action']==a for x in rows) for a in (-1,0,1)})
if __name__=='__main__':run_timing()
