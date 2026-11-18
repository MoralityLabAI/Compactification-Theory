"""Validate the archived CGC-01 source-only witness against live finite matrices."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
from finite_audit import synthetic_receipt


def audit() -> dict[str, object]:
    path=Path(__file__).with_name('SOURCE_ONLY_RECEIPT.json')
    archived=json.loads(path.read_text(encoding='utf-8'))
    observed=synthetic_receipt()
    if observed != archived:
        raise AssertionError(f'CGC-01 source-only receipt drift: {observed!r} != {archived!r}')
    return archived


if __name__=='__main__':
    print(json.dumps(audit(),indent=2,sort_keys=True))
