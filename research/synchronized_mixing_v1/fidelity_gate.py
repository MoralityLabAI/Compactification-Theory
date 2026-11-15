"""Post-fit finite fidelity checks for SMC-06.

This audit receives a frozen certified K6 geometry and separate 15-channel
scores. It never repairs from evaluation rows, and does not claim to know the
planted dictionary. Every reported support and margin uses exact decimal
Fractions of the archived scores.
"""
from fractions import Fraction
from approx_geometry import model_matching_states


def audit_rows(geometry, score_rows):
    masks = tuple(model_matching_states(geometry).values())
    if not masks or len(set(masks)) != 15:
        raise ValueError('expected exactly 15 matching supports')
    summaries = []
    for i, raw in enumerate(score_rows):
        if len(raw) != 15:
            raise ValueError('expected a 15-feature score row')
        values = tuple(Fraction(str(x)) for x in raw)
        order = sorted(range(15), key=lambda j: (-values[j], j))
        top = frozenset(order[:3])
        gap = values[order[2]] - values[order[3]]
        candidate_scores = sorted((sum((values[j] for j,b in enumerate(m) if b), Fraction(0)),m)
                                   for m in masks)
        model_max = candidate_scores[-1][0]
        unconstrained = sum((values[j] for j in top), Fraction(0))
        penalty = unconstrained - model_max
        assert penalty >= 0
        on_manifold = any(frozenset(j for j,b in enumerate(m) if b) == top for m in masks)
        if on_manifold:
            assert penalty == 0
        elif gap > 0:
            assert penalty > 0
        summaries.append({
            'row': i, 'top3': tuple(sorted(top)),
            'top3_gap': str(gap),
            'certified_top3_linf_radius': str(gap / 2) if gap > 0 else '0',
            'top3_is_matching': bool(on_manifold and gap > 0),
            'unconstrained_to_matching_score_penalty': str(penalty),
            'strictly_separated': bool(gap > 0),
        })
    return {
        'num_rows': len(summaries),
        'matched': sum(x['top3_is_matching'] for x in summaries),
        'all_rows_on_manifold': all(x['top3_is_matching'] for x in summaries),
        'rows': summaries,
        'scope': 'Heldout post-fit consistency only; no repair or planted-dictionary fidelity guarantee',
    }


def frozen_evidence_audit(path):
    """Replay pre-committed score arrays without PyTorch or planted ground truth."""
    import json
    from approx_geometry import recover_with_one_swap
    with open(path, encoding='utf-8') as stream:
        evidence = json.load(stream)
    output = {}
    for seed, data in evidence['activation_scores_15_by_15'].items():
        try:
            fit = recover_with_one_swap(data[:12])
        except ValueError as exc:
            output[seed] = {'geometry_status': 'abstain', 'reason': str(exc)}
            continue
        output[seed] = {
            'geometry_status': 'certified_on_training',
            'repairs': fit['repairs'],
            'training_fit': audit_rows(fit['geometry'], data[:12]),
            'heldout_only': audit_rows(fit['geometry'], data[12:]),
        }
    return output


if __name__ == '__main__':
    import json
    from pathlib import Path
    path = Path(__file__).with_name('sae_activation_evidence.json')
    print(json.dumps(frozen_evidence_audit(path), indent=2))
