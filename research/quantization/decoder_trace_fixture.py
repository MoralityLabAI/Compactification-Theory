"""Proof-linked exact traces for recurrent-state supervision and interventions.

These are Fraction evaluations, compared with both circuit evaluators. Generic
Lean theorems cover the stated hypotheses; the JSON serializer and dataset
loader are not formally verified. This creates no trained model or model result.
"""
from fractions import Fraction as F
from pathlib import Path
import gzip
import hashlib
import json
import random
from spatial_circuit import Net, read_bit

HERE=Path(__file__).resolve().parent
RNG=random.Random(16112026)
relu=lambda x:max(F(0),x)
def pack(bits):
    z=F(1,4)
    for b in reversed(bits): z=(1+b+z)/4
    return z
def digit(z): return 4*(relu(4*z-F(7,4))-relu(4*z-2))
def fraction(x): return str(F(x))

def main():
    net=Net(['code'])
    read_bit(net,'bit')
    rows=[]
    steps=0
    cases=[]
    for split,lengths in [('train_fixture',[0,1,2,3,4,6,8]),
                          ('calibration_fixture',[10,12]),('heldout_length_fixture',[16,32,64])]:
        for n in lengths:
            for pattern in range(4):
                bits=([0]*n if pattern==0 else [1]*n if pattern==1 else
                      [i%2 for i in range(n)] if pattern==2 else [RNG.randrange(2) for _ in range(n)])
                for sign in [-1,0,1]:
                    cases.append((split,bits,F(sign,48*4**n)))
    for index,(split,bits,e) in enumerate(cases):
        z=pack(bits)+e
        trace=[]
        assert pack(bits)*2**(2*(len(bits)+1)) == int(pack(bits)*2**(2*(len(bits)+1)))
        for visit,expected in enumerate(bits):
            b=digit(z)
            raw=4*z-1-b
            nxt=max(F(0),min(F(1),raw))
            logical=net.evaluate([z])
            compiled=net.evaluate_compiled([z])
            assert logical==compiled
            assert logical['bit']==b==expected and logical['code']==nxt
            assert raw==nxt==pack(bits[visit+1:])+4**(visit+1)*e
            trace.append({'visit':visit,'code_before':fraction(z),'expected_bit':expected,
                'read_bit':fraction(b),'suffix_raw':fraction(raw),'code_after':fraction(nxt),
                'state_error':fraction(4**(visit+1)*e)})
            z=nxt
            steps+=1
        assert z==F(1,4)+4**len(bits)*e
        rows.append({'id':f'sentinel-{index:03d}','split':split,'bits':bits,
            'initial_code':fraction(pack(bits)+e),'perturbation':fraction(e),
            'noise_budget':fraction(F(1,48*4**len(bits))), 'terminal_code':fraction(z),
            'trace':trace,'lean_reference':'MathCan.SentinelDecoder.decode_word_robust',
            'terminal_reference':'MathCan.SentinelDecoder.advance_word_robust'})
    # Deliberately outside the sentinel encoding contract: this is an empirical
    # negative control, not an instance of the sentinel robustness theorem.
    eta=F(1,2**20)
    z=F(1,2)-eta
    assert digit(z)!=1
    negative={'id':'bare-word-negative','split':'negative_control','code':fraction(z),
        'intended_bit':1,'actual_bit':fraction(digit(z)),
        'reason':'No trailing sentinel; excluded from all positive theorem instances.'}
    output=HERE/'decoder_trace_fixtures.jsonl.gz'
    data=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in [*rows,negative]).encode()
    output.write_bytes(gzip.compress(data,mtime=0))
    results={'status':'passed','positive_examples':len(rows),'negative_controls':1,
        'decoded_steps':steps,'maximum_payload_length':64,
        'split_counts':{name:sum(r['split']==name for r in rows) for name in sorted({r['split'] for r in rows})},
        'circuit_evaluators':'Fraction register and signed-pair ReLU evaluators agree at every step',
        'fixture_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'uncompressed_jsonl_sha256':hashlib.sha256(data).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Proof-linked exact reference fixtures only; no transformer training or causal mechanism result.'}
    (HERE/'decoder_trace_fixture_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
