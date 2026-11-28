"""escalate() on hand-built receipts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import escalate21 as E  # noqa: E402


def rec(bp_p, bp_ops, exact_p, exact_ops, ref):
    m = {'ve_min_fill': dict(p=exact_p, ops=exact_ops)}
    if bp_p is not None:
        m['bp_B10'] = dict(p=bp_p, ops=bp_ops)
    return dict(methods=m, reference=ref)


def test_bp_acts_without_escalation():
    u, ops, path = E.escalate(rec(0.95, 100, 0.9, 500, 0.9), 10, 4)
    assert path == 'bp' and ops == 100 and abs(u - (0.9 - 4 * 0.1)) < 1e-12


def test_abstention_escalates_when_budget_allows():
    u, ops, path = E.escalate(rec(0.5, 100, 0.95, 500, 0.95), 10, 4)
    assert path == 'exact' and ops == 600 and abs(u - (0.95 - 4 * 0.05)) < 1e-12


def test_abstention_stays_when_exact_does_not_fit_after_bp():
    u, ops, path = E.escalate(rec(0.5, 600, 0.95, 500, 0.95), 10, 4)
    assert path == 'abstain' and ops == 600 and u == 0.0


def test_missing_bp_goes_straight_to_exact():
    u, ops, path = E.escalate(rec(None, 0, 0.05, 300, 0.05), 10, 4)
    assert path == 'exact' and ops == 300
