"""RRC-02: exact family-disjoint residue-program benchmark and CPU student audit.

All inference decisions consume only coefficients through TRAIN_DEGREE. Any
numeric coefficient at a larger degree is strictly evaluation-only. Both the
neural student and the exact enumerator see Euler exponents from RRC-01.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

try:
    from .rrc import (ResidueRule, euler_exponents, first_mismatch, gap_coefficients,
                      infer_periodic_rule, product_coefficients)
    from .student02 import ResidueStudent, make_features
except ImportError:
    from rrc import (ResidueRule, euler_exponents, first_mismatch, gap_coefficients,
                     infer_periodic_rule, product_coefficients)
    from student02 import ResidueStudent, make_features

TRAIN_DEGREE = 64
EVAL_DEGREE = 128
MODULUS_CAP = 12
SEED = 20261109
DEPTHS = (0, 1, 2, 4, 8)
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    family: str
    coefficients: tuple[int, ...]
    collision_of: str = ""

    @property
    def prefix(self) -> tuple[int, ...]:
        return self.coefficients[:TRAIN_DEGREE + 1]


def coefficients_from_exponents(a: Iterable[int]) -> list[int]:
    """Independent exact Euler-product reconstruction by log derivative.

    For an integer exponent vector a_n, the formal product has integer
    coefficients. Check the n-divisibility at every coefficient.
    """
    a = list(a)
    if not a or a[0] != 0:
        raise ValueError('exponent vector must begin with zero')
    nmax = len(a) - 1
    b = [0] * (nmax + 1)
    for d in range(1, nmax + 1):
        for k in range(d, nmax + 1, d):
            b[k] += d * a[d]
    f = [1] + [0] * nmax
    for n in range(1, nmax + 1):
        x = sum(b[k] * f[n-k] for k in range(1, n+1))
        if x % n:
            raise ArithmeticError(f'Euler coefficient nonintegral at {n}')
        f[n] = x // n
    return f


def exponent_mask(modulus: int, residues: Iterable[int], degree: int = EVAL_DEGREE) -> list[int]:
    allowed = set(residues)
    if modulus < 1 or any(x < 0 or x >= modulus for x in allowed):
        raise ValueError('invalid periodic program')
    return [0] + [int(n % modulus in allowed) for n in range(1, degree+1)]


def distinct_coefficients(degree: int, minimum_part: int = 1) -> list[int]:
    """Independent bounded-use coin DP, distinct positive parts."""
    f = [1] + [0]*degree
    for part in range(minimum_part, degree+1):
        for n in range(degree, part-1, -1):
            f[n] += f[n-part]
    return f


def make_cases(seed: int = SEED) -> tuple[Case, ...]:
    """Immutable deterministic splits. Train=product generators; dev=distinct;
    test=gap local generators plus modulus-disjoint product stress cases.
    """
    rng = random.Random(seed)
    test_rng = random.Random(seed + 1_000_003)
    cases: list[Case] = []

    def add(name: str, split: str, family: str, coefficients: list[int], collision_of: str = ''):
        assert len(coefficients) == EVAL_DEGREE + 1
        cases.append(Case(name, split, family, tuple(coefficients), collision_of))

    # Training: periodic and finite-defect/colored Euler-product programs.
    # All teacher coefficients are generated from full integer exponents.
    for modulus in (2, 3, 4, 6, 8, 9, 10, 12):
        for sample in range(12):
            residues = tuple(r for r in range(modulus) if rng.random() < .45)
            if not residues:
                residues = (rng.randrange(modulus),)
            if len(residues) == modulus:
                residues = residues[:-1]
            a = exponent_mask(modulus, residues)
            add(f'train_periodic_m{modulus}_{sample:02}', 'train', 'periodic_product',
                coefficients_from_exponents(a))
            # A single defect inside the fitting prefix usually destroys every
            # short periodic rule; allow exact enumerator to classify it.
            defect = rng.randrange(1, TRAIN_DEGREE+1)
            b = a.copy()
            b[defect] = 1 - b[defect]
            add(f'train_defect_m{modulus}_{sample:02}', 'train', 'finite_defect_product',
                coefficients_from_exponents(b))
            if sample % 3 == 0:
                colored = a.copy()
                colored[defect] = 2
                add(f'train_colored_m{modulus}_{sample:02}', 'train', 'colored_product',
                    coefficients_from_exponents(colored))

    # Entire distinct-part local rule family is development only.
    for minimum in range(1, 9):
        add(f'dev_distinct_s{minimum}', 'dev', 'distinct_parts',
            distinct_coefficients(EVAL_DEGREE, minimum_part=minimum))

    # No gap-separated teacher of any gap is present in training/development.
    for gap in (2, 3, 4):
        for minimum in range(1, 5):
            add(f'test_gap{gap}_s{minimum}', 'test', f'local_gap_{gap}',
                gap_coefficients(EVAL_DEGREE, gap=gap, minimum_part=minimum))

    # Modulus-disjoint stress tests: the generator type overlaps training, but
    # moduli 5, 7 and 11 and their masks are never present in training.
    for modulus in (5, 7, 11):
        for sample in range(8):
            residues = tuple(r for r in range(modulus) if test_rng.random() < .4)
            if not residues:
                residues = (test_rng.randrange(modulus),)
            a = exponent_mask(modulus, residues)
            add(f'test_unseen_mod{modulus}_{sample:02}', 'test', f'unseen_modulus_{modulus}',
                coefficients_from_exponents(a))

    # Positive/product collision pairs exactly identical through q^64.
    # Multiplication by 1/(1-q^K) with K>64 changes f_K by +1 only at K
    # and later. These pairs are intentionally information-theoretic aliases.
    for index, degree in enumerate((65, 73, 96, 112)):
        residues = ((1, 4), (2, 3), (1, 4), (2, 3))[index]
        a = exponent_mask(5, residues)
        base_id = f'test_collision_base_{index}'
        add(base_id, 'test', 'collision_baseline', coefficients_from_exponents(a))
        flipped = a.copy()
        flipped[degree] += 1  # extra unrestricted color for part degree
        add(f'test_collision_mutant_{index}', 'test', 'collision_mutant',
            coefficients_from_exponents(flipped), collision_of=base_id)

    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
    return tuple(cases)


def candidate_rows(prefix: tuple[int, ...]):
    """Public algebraic features. NO held-out coefficients are read here."""
    if len(prefix) != TRAIN_DEGREE + 1:
        raise ValueError('exactly 65 training coefficients required')
    a = euler_exponents(prefix)
    return tuple((m, tuple((r, make_features([a[n] for n in range(1, TRAIN_DEGREE+1)
                                                  if n % m == r])) for r in range(m)))
                 for m in range(1, MODULUS_CAP + 1))


def training_rows(cases: tuple[Case, ...]):
    """Inputs/labels for recurrent student; restrict to train families."""
    features = []
    labels = []
    membership = []
    for c in cases:
        if c.split != 'train':
            continue
        a = euler_exponents(c.prefix)
        for m in range(1, MODULUS_CAP+1):
            for r in range(m):
                observed = [a[n] for n in range(1, TRAIN_DEGREE+1) if n%m == r]
                label = 0 if all(v == 0 for v in observed) else (1 if all(v == 1 for v in observed) else 2)
                features.append(make_features(observed))
                labels.append(label)
                membership.append(c.case_id)
    return features, labels, membership


def certificate(prefix: tuple[int, ...], rule: ResidueRule | None) -> dict:
    """Deterministic prefix-only verifier, independent of student predictions."""
    if rule is None:
        return {'valid': False, 'first_mismatch': None, 'reason': 'abstain'}
    if rule.modulus < 1 or rule.modulus > MODULUS_CAP:
        return {'valid': False, 'first_mismatch': None, 'reason': 'modulus_out_of_scope'}
    if any(x < 0 or x >= rule.modulus for x in rule.residues) or len(set(rule.residues)) != len(rule.residues):
        return {'valid': False, 'first_mismatch': None, 'reason': 'invalid_residues'}
    predicted = product_coefficients(TRAIN_DEGREE, rule)
    mismatch = first_mismatch(prefix, predicted)
    return {'valid': mismatch is None, 'first_mismatch': mismatch,
            'reason': 'prefix_exact' if mismatch is None else 'prefix_mismatch'}


def infer_student(prefix: tuple[int, ...], model: ResidueStudent, depth: int) -> dict:
    """Try candidate moduli in increasing code-length order and certify each.

    No evaluation-only coefficients are accessible to model or verifier.
    """
    raw_proposals = 0
    rejected = 0
    residue_evaluations = 0
    rows = candidate_rows(prefix)
    for m, entries in rows:
        predictions = [model.predict(x, depth=depth) for _, x in entries]
        residue_evaluations += len(entries)
        if 2 in predictions:
            continue
        rule = ResidueRule(m, tuple(r for (r,_), label in zip(entries, predictions) if label == 1))
        raw_proposals += 1
        if certificate(prefix, rule)['valid']:
            return {'rule': rule, 'raw_proposals': raw_proposals,
                    'rejected_by_verifier': rejected, 'candidate_moduli_seen': m,
                    'residue_evaluations': residue_evaluations}
        rejected += 1
    return {'rule': None, 'raw_proposals': raw_proposals,
            'rejected_by_verifier': rejected, 'candidate_moduli_seen': MODULUS_CAP,
            'residue_evaluations': residue_evaluations}


def rule_json(rule: ResidueRule | None):
    return None if rule is None else rule.as_dict()


def evaluate(cases: tuple[Case, ...], models: dict[str, ResidueStudent]) -> dict:
    """Blind deterministic scoring of frozen checkpoints against test cases."""
    names_depths = [(name, depth) for name in models
                    for depth in (DEPTHS if name == 'recurrent' else (0,))]
    entries = []
    for c in cases:
        if c.split == 'train':
            continue
        oracle = infer_periodic_rule(c.prefix, modulus_cap=MODULUS_CAP)
        for model_name, depth in names_depths:
            report = infer_student(c.prefix, models[model_name], depth)
            rule = report['rule']
            mismatch = (None if rule is None else first_mismatch(
                c.coefficients, product_coefficients(EVAL_DEGREE, rule), start=TRAIN_DEGREE+1))
            entries.append({
                'case_id': c.case_id, 'split': c.split, 'family': c.family,
                'model': model_name, 'depth': depth, 'exact_baseline': rule_json(oracle),
                'selected_rule': rule_json(rule),
                'same_as_exact_baseline': rule == oracle,
                'accepted': rule is not None,
                'heldout_first_mismatch': mismatch,
                'heldout_exact': bool(rule is not None and mismatch is None),
                'raw_proposals': report['raw_proposals'],
                'rejected_by_verifier': report['rejected_by_verifier'],
                'candidate_moduli_seen': report['candidate_moduli_seen'],
                'residue_evaluations': report['residue_evaluations'],
                'dense_mac_estimate': report['residue_evaluations'] * (
                    180 + depth*144 if models[model_name].variant == 'recurrent' else 339),
                'collision_of': c.collision_of or None,
            })
    return {'cases': entries, 'aggregates': aggregate(entries),
            'counts': {
                split: sum(c.split == split for c in cases) for split in ('train', 'dev', 'test')
            }}


def aggregate(entries: list[dict]) -> list[dict]:
    groups = {}
    for e in entries:
        key = (e['split'], e['family'], e['model'], e['depth'])
        group = groups.setdefault(key, [])
        group.append(e)
    out = []
    for (split, family, model, depth), row in sorted(groups.items()):
        out.append({
            'split': split, 'family': family, 'model': model, 'depth': depth,
            'cases': len(row), 'accepted': sum(x['accepted'] for x in row),
            'baseline_agreement': sum(x['same_as_exact_baseline'] for x in row),
            'heldout_exact': sum(x['heldout_exact'] for x in row),
            'heldout_failed_after_certificate': sum(x['accepted'] and not x['heldout_exact'] for x in row),
            'proposals': sum(x['raw_proposals'] for x in row),
            'rejected_by_verifier': sum(x['rejected_by_verifier'] for x in row),
            'residue_evaluations': sum(x['residue_evaluations'] for x in row),
            'dense_mac_estimate': sum(x['dense_mac_estimate'] for x in row),
            'mean_candidate_moduli_seen': round(sum(x['candidate_moduli_seen'] for x in row)/len(row), 3),
        })
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(checkpoints: dict[str, Path]) -> dict:
    models = {k: ResidueStudent.load(path) for k,path in sorted(checkpoints.items())}
    cases = make_cases()
    result = evaluate(cases, models)
    assert all(c.prefix == next(x for x in cases if x.case_id == c.collision_of).prefix
               for c in cases if c.collision_of)
    return {
        'schema': 'rrc02.cpu-pilot.v1',
        'status': 'trained_tiny_recurrent_pilot_and_exact_finite_validation',
        'parameters': {'train_degree': TRAIN_DEGREE, 'eval_degree': EVAL_DEGREE,
                       'modulus_cap': MODULUS_CAP, 'seed': SEED, 'depths': list(DEPTHS)},
        'resource_accounting': {name: {
            'parameter_count': model.parameter_count,
            'parameter_storage_bits_assuming_fp32': model.parameter_count * 32,
            'resource_exclusions': ('teacher extraction, program decoder, coefficient storage, '
                'optimizer state, checkpoint JSON, Python interpreter and verifier time')
        } for name, model in models.items()},
        'source_sha256': {name: sha256_file(HERE / name) for name in ('rrc02.py', 'student02.py', 'train_rrc02.py')},
        'checkpoint_sha256': {name: sha256_file(path) for name,path in sorted(checkpoints.items())},
        'data_roster_sha256': hashlib.sha256(json.dumps([(c.case_id, c.split, c.family, c.prefix)
                                                         for c in cases], separators=(',', ':')).encode()).hexdigest(),
        'evidence': result,
        'limitations': ('Finite coefficient certificates only; model sees Euler features and '
                        'shared periodic-program grammar. Collision mutants are provably '
                        'undetectable by any prefix-only algorithm. No infinite identities '
                        'or pretrained-LLM transfer are claimed.'),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(HERE/'results/rrc02_recurrent.json'))
    p.add_argument('--ff-checkpoint', default=str(HERE/'results/rrc02_feedforward.json'))
    p.add_argument('--shuffled-checkpoint', default=str(HERE/'results/rrc02_shuffled.json'))
    p.add_argument('--out', type=Path, default=HERE/'results/rrc02_pilot.json')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    checkpoints = {'recurrent': Path(args.checkpoint),
                   'feedforward': Path(args.ff_checkpoint),
                   'shuffled': Path(args.shuffled_checkpoint)}
    report = receipt(checkpoints)
    payload = json.dumps(report, sort_keys=True, indent=2) + '\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text() != payload:
            raise SystemExit('RRC-02 stale receipt: regenerate and review, then commit')
        print('RRC-02 CPU checkpoint/receipt exact replay OK')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f'wrote {args.out}')

if __name__ == '__main__':
    main()
