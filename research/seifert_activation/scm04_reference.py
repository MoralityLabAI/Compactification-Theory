"""Reproducible exact SCM-04 reference; no neural checkpoints and no network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from knot import extract_closed_braid, normalized_alexander, synthetic_braid
from milnor import triple_milnor, triple_milnor_nilpotent, token_order_milnor

C = (1, 1, 2, 2, -1, -1, -2, -2)
CI = tuple(-x for x in reversed(C))


def run() -> dict:
    pair = []
    for label, word in (("commutator", C), ("inverse_commutator", CI)):
        pair.append({
            "name": label,
            "braid": list(word),
            "alexander": str(normalized_alexander(3, word)),
            "exact_longitude": triple_milnor(3, word),
            "nilpotent_cubic": triple_milnor_nilpotent(3, word),
        })
    frames = synthetic_braid(3, C)
    steps = frames[:-1]
    shifts = []
    for index in range(len(steps)):
        rotated = steps[index:] + steps[:index] + [steps[index]]
        braid = extract_closed_braid(rotated)
        out = token_order_milnor(rotated, braid.word)
        shifts.append({"phase_index": index,
                       "raw_mu123": out['mu123_position_order'],
                       "corrected_mu123": out['mu123_token_order'],
                       "position_to_token_strands": out['position_to_token_strands']})
    compact = {}
    for integer in (-1000, -10, -2, -1, 0, 1, 2, 10, 1000):
        word = C * integer if integer >= 0 else CI * (-integer)
        m = triple_milnor_nilpotent(3, word)
        compact[str(integer)] = {
            "mu123": m['mu123'],
            "crossings": len(word),
            "nonzero_magnus_terms": m['magnus_nonzero_terms'],
        }
    assert pair[0]['alexander'] == pair[1]['alexander']
    assert pair[0]['nilpotent_cubic']['mu123'] == 1
    assert pair[1]['nilpotent_cubic']['mu123'] == -1
    assert all(s['corrected_mu123'] == 1 for s in shifts)
    assert all(int(k) == v['mu123'] for k, v in compact.items())
    return {
        'schema': 'scm04/cpu-reference/v1',
        'source': 'exact_synthetic_braids_only',
        'pair': pair,
        'phase_cut_count': len(shifts),
        'phase_cut_raw_signs': sorted(set(s['raw_mu123'] for s in shifts)),
        'phase_cut_results': shifts,
        'nilpotent_powers': compact,
        'certificate': 'integer_polynomial_magnus_degree_three',
        'neural_checkpoint_evidence': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists, immutable receipt must have a new path')
    data = run()
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding='utf-8')
    print('SCM04 reference:', args.out, 'sha256:', hashlib.sha256(content.encode()).hexdigest())


if __name__ == '__main__':
    main()
