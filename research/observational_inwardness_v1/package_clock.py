"""Pack OIC-13 local CPU results into a deterministic base64-gzip receipt."""
from __future__ import annotations
import base64,gzip,hashlib,json
from pathlib import Path
from verify_clock import validate_payload

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'/'oic13_full.json'
OUT=HERE/'results'

def build():
    payload=json.loads(DATA.read_text(encoding='utf8'))
    summary=validate_payload(payload)
    canonical=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    compressed=gzip.compress(canonical,compresslevel=9,mtime=0)
    (OUT/'oic13_full.json.gz.b64').write_text(base64.b64encode(compressed).decode()+'\n',encoding='utf8')
    manifest={
      'experiment':'OIC-13','status':'executed_cpu_structural_control',
      'source_sha256':hashlib.sha256((HERE/'clock_pilot.py').read_bytes()).hexdigest(),
      'decoded_sha256':hashlib.sha256(canonical).hexdigest(),
      'gzip_sha256':hashlib.sha256(compressed).hexdigest(),
      'model_fits':120,'pair_runs':60,
      'summary':summary,'protocol':payload['protocol'],
      'source_replay':'python clock_pilot.py --out results/oic13_full.json',
      'limits':'Exact cyclic clock and visible XOR update are built into the architecture. Synthetic toy dynamics, not learned discovery of hidden topology.',
      'wall_seconds':payload['wall_seconds']}
    (OUT/'oic13_summary.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print('compressed receipt',len(compressed),'bytes')
    print('manifest',len((OUT/'oic13_summary.json').read_bytes()),'bytes')
    return manifest

if __name__=='__main__':build()
