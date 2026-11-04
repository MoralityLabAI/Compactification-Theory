"""QO-14: frozen QO-13 teacher data for a low-cost recurrent student.

All states are two addressed transient cells, one action, one macro
class, tau=1/5. The exact teacher plans with horizon h=2, using
QO-07's nested Bernoulli beta-mixture confidence intervals.

Train and test have DISJOINT Bernoulli true-rate pairs and source
seeds. True rates influence the generator only. Input features are
derived solely from observed QO-07 evidence. No hidden-rate leakage.
"""
from fractions import Fraction as Q
from random import Random
from itertools import product

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate
)
from research.ingression_dynamics_v1.certificate_reachability import (
    optimal_certificate_reachability
)

TRAIN_RATES=(
    (Q(1,20),Q(19,20)),(Q(1,10),Q(9,10)),
    (Q(3,20),Q(17,20)),(Q(1,4),Q(3,4)),
    (Q(3,10),Q(7,20)),(Q(9,20),Q(1,2)),
    (Q(9,20),Q(7,10)),(Q(1,5),Q(3,5)),
)
TEST_RATES=(
    (Q(1,16),Q(15,16)),(Q(3,16),Q(13,16)),
    (Q(5,16),Q(11,16)),(Q(7,16),Q(9,16)),
    (Q(1,8),Q(5,8)),(Q(3,8),Q(7,8)),
)
TRAIN_SEEDS=(101,103,107,109,113,127,131,137,139,149,151,157)
TEST_SEEDS=(701,709,719,727,733,739,743)
COUNTS=((12,14),(16,16),(18,18),(20,22),(24,20),
        (28,28),(32,36),(38,40),(44,44))
TAU=Q(1,5)
ALPHA=Q(1,20)
BITS=10
TEACHER_HORIZON=2
DIM=2
INPUT_WIDTH=10


def audit_fixed_split():
    if set(TRAIN_RATES)&set(TEST_RATES):
        raise AssertionError("teacher train/test rate pair overlap")
    if set(TRAIN_SEEDS)&set(TEST_SEEDS):
        raise AssertionError("teacher train/test seed overlap")
    if not all(0<p<1 for pair in TRAIN_RATES+TEST_RATES for p in pair):
        raise AssertionError("rate fixture leaves the Bernoulli interior")
    return {
        "train_rate_pairs":len(TRAIN_RATES),"test_rate_pairs":len(TEST_RATES),
        "train_seeds":len(TRAIN_SEEDS),"test_seeds":len(TEST_SEEDS),
        "count_pairs":len(COUNTS),
        "train_expected_examples":len(TRAIN_RATES)*len(TRAIN_SEEDS)*len(COUNTS),
        "test_expected_examples":len(TEST_RATES)*len(TEST_SEEDS)*len(COUNTS),
    }


def source_events(pair,seed,max_count):
    """Independent per-cell Bernoulli streams. No model policy can see rates."""
    stream=[]
    for i,p in enumerate(pair):
        rng=Random(20261104*1000003+seed*7927+i*11003)
        stream.append(tuple(int(rng.random()<float(p)) for _ in range(max_count)))
    return tuple(stream)


def evidence_from_prefixes(streams,count_pair):
    """Replay observed streams into original QO-07 sequential ledger."""
    ledger=ProbeLedger(2,1,ALPHA,BITS)
    left,right=count_pair
    for j in range(max(left,right)):
        if j<left:
            ledger.observe(0,0,streams[0][j])
        if j<right:
            ledger.observe(1,0,streams[1][j])
    return ledger


def encode_cell_tokens(ledger,tau=TAU):
    """Exactly two evidence-only 10-feature tokens for the reservoir RNN.

    No data about the generating Bernoulli true rates is present.
    Input includes counts, observed hit means, QO-07 interval bounds,
    and a public state-identity feature.
    """
    if ledger.n!=2 or ledger.d!=1:
        raise ValueError("QO-14 feature contract: two states, one action")
    tokens=[]
    for i in range(2):
        n=ledger.counts[i][0]
        hits=ledger.hits[i][0]
        lo,hi=ledger.bounds[i][0]
        mean=Q(hits,n) if n else Q(1,2)
        beta=Q(hits+1,n+2)
        tokens.append((
            min(n,64)/64.,float(mean),float(beta),
            float(lo),float(hi),float(hi-lo),
            float(Q(1,n+1)),float(tau),
            float(Q(hits,64)),float(i),
        ))
    return tuple(tokens)


def teacher_record(ledger):
    """Distill exact two-probe root Q-values without modifying QO-13's frozen source.

    The original QO-13 horizon-two optimum is independently evaluated.
    For each possible first cell, replay a candidate hit/hold from the
    PUBLIC current confidence/count state into an isolated copy, then
    evaluate QO-13's original horizon-one optimal recourse. The full
    root value must equal the maximum of these action values.
    """
    out=optimal_certificate_reachability(
        ledger,1,TAU,horizon=TEACHER_HORIZON)
    gap_calls=out["decision_gap_evaluations"]
    candidates=out["bellman_candidates"]
    if out["current_decision"]=="undetermined":
        values=[]
        for i in (0,1):
            p=Q(ledger.hits[i][0]+1,ledger.counts[i][0]+2)
            outcomes=[]
            for hit in (0,1):
                successor=ProbeLedger(2,1,ledger.alpha,ledger.bits)
                successor.counts=[list(row) for row in ledger.counts]
                successor.hits=[list(row) for row in ledger.hits]
                successor.bounds=[list(row) for row in ledger.bounds]
                try:
                    successor.observe(i,0,hit)
                except ValueError:
                    outcomes.append(Q(0))
                else:
                    child=optimal_certificate_reachability(
                        successor,1,TAU,horizon=1)
                    outcomes.append(child["predicted_certificate_probability"])
                    gap_calls+=child["decision_gap_evaluations"]
                    candidates+=child["bellman_candidates"]
            values.append((1-p)*outcomes[0]+p*outcomes[1])
        q=tuple(values)
        if max(q)!=out["predicted_certificate_probability"]:
            raise AssertionError(
                "QO-14 root-Q adapter disagrees with unchanged QO-13 teacher")
    else:
        q=(Q(1),Q(1))
    return {
        "tokens":encode_cell_tokens(ledger),
        "q":q,
        "teacher_value":out["predicted_certificate_probability"],
        "teacher_action":0 if q[0]>=q[1] else 1,
        "decisive":q[0]!=q[1],
        "certified":out["current_decision"]!="undetermined",
        "counts":tuple(row[0] for row in ledger.counts),
        "successes":tuple(row[0] for row in ledger.hits),
        "teacher_gap_calls":gap_calls,
        "teacher_bellman_candidates":candidates,
    }

def dataset(split,rate_pairs=None,seeds=None,count_pairs=None):
    """Deterministic case families, optionally reduced for cheap smoke tests."""
    if split not in ("train","test"):
        raise ValueError("train/test split required")
    pairs=tuple(rate_pairs) if rate_pairs is not None else (
        TRAIN_RATES if split=="train" else TEST_RATES)
    seeds=tuple(seeds) if seeds is not None else (
        TRAIN_SEEDS if split=="train" else TEST_SEEDS)
    counts_list=tuple(count_pairs) if count_pairs is not None else COUNTS
    if not pairs or not seeds or not counts_list:
        raise ValueError("nonempty frozen rates/seeds/evidence counts required")
    rows=[]
    for rates in pairs:
        for seed in seeds:
            streams=source_events(rates,seed,max(max(x) for x in counts_list))
            for counts in counts_list:
                ledger=evidence_from_prefixes(streams,counts)
                item=teacher_record(ledger)
                item["family"]=tuple(str(p) for p in rates)
                item["source_seed"]=seed
                rows.append(item)
    return rows


def dataset_summary(rows):
    return {
        "examples":len(rows),
        "decision_sensitive":sum(r["decisive"] for r in rows),
        "positive_teacher_values":sum(r["teacher_value"]>0 for r in rows),
        "already_certified":sum(r["certified"] for r in rows),
        "mean_teacher_value":str(
            sum((r["teacher_value"] for r in rows),Q(0))/len(rows)),
        "exact_teacher_gap_calls":sum(r["teacher_gap_calls"] for r in rows),
        "exact_teacher_bellman_candidates":sum(
            r["teacher_bellman_candidates"] for r in rows),
    }
