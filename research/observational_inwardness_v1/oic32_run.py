"""OIC-32 exact fixed-prime one-flip theorem arithmetic receipt replay."""
import argparse
import json
from pathlib import Path
from oic32_global import reference

HERE = Path(__file__).resolve().parent
FILE = HERE / 'results' / 'oic32_reference.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    fresh = reference()
    if args.check:
        stored = json.loads(FILE.read_text(encoding='utf-8'))
        if fresh != stored:
            raise AssertionError('OIC-32 pinned analytic receipt mismatch')
        print('OIC-32 exact analytic and finite receipt passed for p=5,7,11,13')
    if args.write:
        FILE.parent.mkdir(parents=True, exist_ok=True)
        FILE.write_text(json.dumps(fresh, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    if not args.check and not args.write:
        print(json.dumps(fresh, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
