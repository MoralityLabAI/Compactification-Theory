import json
import tempfile
import unittest
from pathlib import Path
from research.dimensional_atlas.gym.scm19_clean_topone.tools.resume import RecoveryJournal,prior_state


class LoggingRecoveryTests(unittest.TestCase):
    def test_split_payload_and_prior_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'work.jsonl'
            events=[{'event':'forward_called','stage':'clean_final_s191','elapsed':1.,'batch':8,'tokens':27,'visits':4},
                {'event':'optimizer_completed','stage':'train','elapsed':2.}]
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            journal=RecoveryJournal(path,{'min_free_vram_mb':500})
            journal.event('mediation_cell_completed',kind='id',n=64);journal.file.close()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['stratum'],'id')
            self.assertEqual(journal.work['readout_examples'],16)
            self.assertEqual(journal.work['recurrent_token_visits'],864)
            self.assertEqual(journal.work['base_optimizer_steps'],1)


if __name__=='__main__':unittest.main()
