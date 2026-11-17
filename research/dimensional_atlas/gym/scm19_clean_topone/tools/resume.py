"""Logging-only recovery; original registered implementation bytes stay intact."""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
from .. import campaign
from ..core import HERE,Journal,OracleAudit,sha,canonical


def prior_state(path):
    work=Counter();oracle=Counter();peak={'allocated_mb':0.,'reserved_mb':0.};free=None;elapsed=0.
    with path.open() as stream:
        for line in stream:
            e=json.loads(line);elapsed=max(elapsed,e['elapsed'])
            if e['event']=='forward_called':
                b=e['batch'];work.update(model_calls=1,forward_examples=b,recurrent_example_visits=b*e['visits'],recurrent_token_visits=b*e['tokens']*e['visits'])
                # All first-attempt calls finished. Only clean-final 27-token
                # calls requested the extra initial-CLS diagnostic readout.
                work['readout_examples']+=b*(2 if e['stage'].startswith('clean_final_') and e['tokens']==27 else 1)
                if e['stage'].startswith('polynomial_gram_'):work['public_gram_multiplications']+=b*26*529
            key={'optimizer_completed':'base_optimizer_steps','sae_update':'sae_optimizer_steps','adapter_update':'adapter_optimizer_steps'}.get(e['event'])
            if key:work[key]+=1
            if e['event']=='oracle_call':oracle[e['name']]+=1
            if 'allocated_mb' in e:
                for k in peak:peak[k]=max(peak[k],e[k])
                if e.get('free_mb') is not None:free=e['free_mb'] if free is None else min(free,e['free_mb'])
    return work,oracle,peak,free,elapsed


class RecoveryJournal(Journal):
    def __init__(self,path,config):
        work,oracle,peak,free,elapsed=prior_state(Path(path))
        self.prior_oracle=oracle;self.prior_peak=peak
        super().__init__(path,config);self.work.update(work)
        self.min_free=free*2**20 if free is not None else None;self.started-=elapsed
        self.event('logging_recovery_started',previous_seconds=elapsed,recovered_work=dict(work))
    def event(self,event_name,**values):
        if 'kind' in values:values['stratum']=values.pop('kind')
        return super().event(event_name,**values)
    def memory(self):
        result=super().memory()
        for k,v in self.prior_peak.items():result[k]=max(result[k],v)
        return result


class RecoveryOracle(OracleAudit):
    def report(self):
        value=super().report();current=dict(value['source_requests'])
        for k,v in self.journal.prior_oracle.items():value['source_requests'][k]=value['source_requests'].get(k,0)+v
        value['source_request_scope']='check_candidate and grade are cumulative exact journal counts; all other source requests and physical caches describe the recovery process only'
        value['recovery_process_source_requests']=current
        value['first_attempt_data_preparation_oracle_checkpoint']=json.loads((HERE/'results/data.json').read_text())['oracle']
        value['first_attempt_post_data_pure_function_counts']='not durably recorded before the logger exception; excluded from exact cumulative claims'
        return value


def main():
    amendment=json.loads((HERE/'results/recovery_registration.json').read_text())
    assert amendment['recovery_source_sha256']==sha(Path(__file__))
    assert campaign.source_hashes()==amendment['unchanged_registered_source_sha256']
    campaign.Journal=RecoveryJournal;campaign.OracleAudit=RecoveryOracle
    campaign.main()


if __name__=='__main__':main()
