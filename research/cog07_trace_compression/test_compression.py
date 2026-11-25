import itertools
import json

import pytest

from compression import (Example, decode, encode, greedy_cover, hypotheses,
                         load_examples, reconstruct, version_space)


def test_all_two_bit_partial_labelings_preserve_the_entire_class():
    checked = 0
    for assignment in itertools.product((-1, 0, 1), repeat=4):
        sample = [Example(bits, label) for bits, label in enumerate(assignment) if label >= 0]
        target = version_space(sample, 2)
        indices = greedy_cover(sample, 2, require_realizable=False)
        selected = [sample[i] for i in indices]
        dimension, decoded = decode(encode(selected, 2))
        assert dimension == 2
        assert version_space(decoded, 2) == target
        assert len(selected) <= len(hypotheses(2)) - target.bit_count()
        if target:
            assert reconstruct(decoded, 2) == reconstruct(sample, 2)
        checked += 1
    assert checked == 81


def test_inconsistent_class_cannot_produce_a_valid_model():
    sample = [Example(0, 0), Example(0, 1)]
    with pytest.raises(ValueError, match="cannot fit"):
        greedy_cover(sample, 2)
    with pytest.raises(ValueError, match="no consistent"):
        reconstruct(sample, 2)
    assert version_space([sample[i] for i in greedy_cover(sample, 2, require_realizable=False)], 2) == 0


def test_heldout_examples_cannot_enter_selection():
    with pytest.raises(ValueError, match="training split"):
        greedy_cover([Example(3, 1, split="ordinary_test")], 2)


def test_codec_charges_header_and_values_and_rejects_extra_data():
    packet = encode([Example(255, 1)], 8)
    assert len(packet) == 9  # seven-byte header plus nine bits rounded up
    assert decode(packet) == (8, [Example(255, 1)])
    with pytest.raises(ValueError, match="length"):
        decode(packet + b"\0")
    with pytest.raises(ValueError, match="padding"):
        decode(packet[:-1] + bytes([packet[-1] | 1]))
    with pytest.raises(ValueError):
        decode(packet[:6])


def test_negative_examples_exclude_the_all_positive_rule():
    positive = [Example(3, 1)]
    full = positive + [Example(0, 0), Example(1, 0), Example(2, 0)]
    assert version_space(positive, 2) != version_space(full, 2)
    assert reconstruct(positive, 2)[1].predict(0) == 1
    assert reconstruct(full, 2)[1].predict(0) == 0


def test_empty_sample_has_a_well_defined_decoder():
    assert greedy_cover([], 3) == []
    assert decode(encode([], 3)) == (3, [])
    assert version_space([], 3).bit_count() == 28


def test_import_rejects_forged_producer_label_and_split_overlap(tmp_path):
    row = {"root_id": "r", "split": "train", "candidates": [{"action": 0}],
           "labels": {"useful": [True]},
           "public": {"world": {"atoms": [0, 1], "facts": [0], "goal": 1,
                                "rules": [{"premises": [0], "conclusion": 1}]},
                      "known": [0], "history": []}}
    path = tmp_path / "ranking.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert load_examples(path)[0].label == 1
    row["labels"]["useful"][0] = False
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="label mismatch"):
        load_examples(path)
    row["labels"]["useful"][0] = True
    other = dict(row, split="ordinary_test")
    path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="multiple splits"):
        load_examples(path)
