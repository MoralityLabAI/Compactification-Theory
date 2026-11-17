"""Executable proposer-only bridge. Never reads final-test fields into the prompt.

The opt-in local model proposes a bounded, typed update. It has no filesystem,
shell, training, checkpoint-merge or promotion authority.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json

from local_proposer import local_chat_proposal,validate_proposal
from microlearner import canonical,digest


def select_development_report(report:dict,seed:int,arm:str)->dict:
    matches=[x for x in report.get('outcomes',[])
             if x.get('seed')==seed and x.get('arm')==arm]
    if len(matches)!=1:
        raise ValueError('expected precisely one training-run record')
    selected=matches[0]
    # Explicit allow-list: no final_test, secondary holdout or test_counts.
    return {'schema':'siml01.dev-only.v1',
            'checkpoint':selected['trained_checkpoint'],
            'graph_version':report['schema'],
            'diagnostics':selected['final_development'],
            'allotted_train_updates':selected['resources']['training_updates']}


def offline_fixture(diagnostics:dict,budget:int):
    scores=diagnostics['diagnostics']['accuracy']
    residuals=diagnostics['diagnostics']['closure_residual']
    node=max(('T0','T1','T4'),key=lambda n:(1-scores[n])+.05*residuals[n])
    return validate_proposal({
        'schema':'siml01.proposal.v1',
        'target_node':node,'operator':'curriculum_focus',
        'predicted_gain':0.,'predicted_cost_updates':min(budget,16),
        'diagnostic_refs':[f'dev/{node}/accuracy',f'dev/{node}/closure_residual'],
        'competing_explanation':'The observed gap could come from optimization or finite sample selection.',
    },budget)


def generate(report_path:Path,seed:int,arm:str,budget:int,*,endpoint:str|None=None,
             model_id:str|None=None)->dict:
    if type(budget)!=int or budget<1:
        raise ValueError('positive update budget required')
    raw=report_path.read_bytes()
    development=select_development_report(json.loads(raw),seed,arm)
    if endpoint is not None:
        proposal=local_chat_proposal(endpoint,model_id or '',development,budget)
        provenance={'mode':'live_loopback_proposal','model':model_id,'endpoint':endpoint}
    else:
        proposal=offline_fixture(development,budget)
        provenance={'mode':'deterministic_offline_fixture','model':None,'endpoint':None}
    return {'schema':'siml01.proposal-envelope.v1','authority':'proposal_only',
            'proposer':provenance,'development_source_sha256':hashlib.sha256(raw).hexdigest(),
            'checkpoint':development['checkpoint'],
            'diagnostic_sha256':digest(development),
            'available_updates':budget,'proposal':asdict(proposal)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--arm',default='introspective')
    ap.add_argument('--budget',type=int,default=64)
    ap.add_argument('--endpoint',default=None,help='explicit opt-in localhost OpenAI-compatible URL')
    ap.add_argument('--model',default=None)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if (args.endpoint is None) != (args.model is None):
        ap.error('--endpoint and --model must be supplied together')
    envelope=generate(Path(args.report),args.seed,args.arm,args.budget,
                      endpoint=args.endpoint,model_id=args.model)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError('refusing to overwrite an existing proposal envelope')
    target.write_bytes(canonical(envelope)+b'\n')
    print(json.dumps({'status':'proposal_written','path':str(target),
                      'mode':envelope['proposer']['mode'],
                      'operator':envelope['proposal']['operator'],
                      'target':envelope['proposal']['target_node']},sort_keys=True))

if __name__=='__main__':main()
