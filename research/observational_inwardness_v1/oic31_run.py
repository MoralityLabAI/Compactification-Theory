"""OIC-31 frozen exact mixed-fault reference replay (standard library only)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from oic31_mixed import reference

PIN=Path(__file__).resolve().parent/'results'/'oic31_reference.json'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--write',action='store_true')
    args=parser.parse_args();current=reference()
    if args.check:
        expected=json.loads(PIN.read_text(encoding='utf8'))
        if current!=expected:raise RuntimeError('OIC-31 deterministic reference mismatch')
        print('OIC-31 mixed-fault prime-block reference verified')
    elif args.write:
        PIN.parent.mkdir(parents=True,exist_ok=True)
        PIN.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n',encoding='utf8')
        print('wrote',PIN)
    else:print(json.dumps(current,sort_keys=True,indent=2))
if __name__=='__main__':main()
