"""Post-pilot threshold stress: tune query unary using reference odds in generator only."""
from timing import *

def run_challenge():
 rows=[];rng=random.Random(2901)
 for n,fam,i in it.product((12,24,48),('cycle','ladder','chords'),range(12)):
  f,q=fixture(n,fam,280000+n*1000+100*('cycle','ladder','chords').index(fam)+i)
  f[q]=((q,),(1,1));p0=eliminate(f,q);target=Fraction((39,41,159,161)[i%4],200)
  odds=(target/(1-target))/(p0/(1-p0));ratio=odds.limit_denominator(256);assert ratio>0
  f[q]=((q,),(ratio.denominator,ratio.numerator));p=eliminate(f,q);a=action(p)
  # Only the fixture generator uses exact reference to construct near-threshold inputs.
  arms={'full':lambda: (action(eliminate(f,q)),None),**{str(d):lambda d=d:hybrid(f,q,d) for d in range(3)}}
  hits={};samples={k:[] for k in arms}
  for k,fn in arms.items():
   ans,h=fn();assert ans==a;hits[k]=h
  for _ in range(5):
   order=list(arms);rng.shuffle(order)
   for k in order:
    t=time.perf_counter_ns();ans,h=arms[k]();samples[k].append(time.perf_counter_ns()-t);assert ans==a and h==hits[k]
  rows.append(dict(n=n,family=fam,seed=i,target=float(target),posterior=float(p),action=a,hits=hits,samples=samples,median={k:statistics.median(v) for k,v in samples.items()}))
 Path(__file__).with_name('challenge_receipt.json').write_text(json.dumps(rows,indent=2))
 print('actions',{a:sum(x['action']==a for x in rows) for a in (-1,0,1)},'max target discrepancy',max(abs(x['target']-x['posterior']) for x in rows))
 for k in arms:
  print(k,sum(x['hits'][k] is not None for x in rows),statistics.mean(x['median'][k] for x in rows)/1e6,statistics.median(x['median']['full']/x['median'][k] for x in rows),sum(x['median'][k]<=x['median']['full'] for x in rows))
if __name__=='__main__':run_challenge()
