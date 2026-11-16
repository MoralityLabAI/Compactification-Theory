"""One-forward-at-a-time, fail-closed allocation guard and cached CPU receipts."""
import gc
import hashlib
import json
import sqlite3
import time
import numpy as np
from preflight import ROOT, RESULTS, CEILING, save
from core import fingerprint


class Runtime:
    def __init__(self, torch, model, manifest):
        self.torch, self.model = torch, model
        self.source_hash, self.source_files = fingerprint(ROOT)
        self.identity = dict(source_hash=self.source_hash, revision=manifest["revision"],
                             backend=manifest["backend"], use_cache=False, dtype="float16",
                             boundary="one_based_decoder_block_output_last_input_token")
        self.db = sqlite3.connect(RESULTS / "forward_cache.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS forwards (key TEXT PRIMARY KEY, receipt TEXT)")
        self.events = []
        self.successes = self.hits = 0
        self.start = time.perf_counter()
        self.last_batch = self.start
        self.guards = [m.register_forward_hook(self.guard) for m in model.model.layers]

    def guard(self, *unused):
        if self.torch.cuda.memory_allocated() >= CEILING:
            raise RuntimeError("Hard GPU memory ceiling exceeded")

    def forward(self, row, delta=None, layer=12, capture=()):
        t = self.torch
        delta = None if delta is None else np.asarray(delta, dtype=np.float32)
        key = hashlib.sha256(json.dumps(dict(self.identity, ids=row["input_ids"],
                            candidates=row["candidate_ids"], layer=layer,
                            delta=None if delta is None else hashlib.sha256(delta.tobytes()).hexdigest()),
                            sort_keys=True).encode()).hexdigest()
        cached = self.db.execute("SELECT receipt FROM forwards WHERE key=?", (key,)).fetchone()
        if cached and not capture:
            self.hits += 1
            return json.loads(cached[0]), {}
        vectors, actual = {}, {}
        handles = []
        d_gpu = None if delta is None else t.as_tensor(delta, dtype=t.float16, device="cuda")
        def hook(layer_id):
            def run(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                if layer_id in capture:
                    vectors[layer_id] = h[0, -1].detach().cpu().numpy().copy()
                if d_gpu is None or layer_id != layer:
                    return None
                changed = h.clone()
                changed[0, -1] += d_gpu
                actual["displacement"] = (changed[0, -1].float()-h[0, -1].float()).cpu().numpy()
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return run
        for selected in sorted(set(capture) | ({layer} if delta is not None else set())):
            handles.append(self.model.model.layers[selected-1].register_forward_hook(hook(selected)))
        tic = time.perf_counter()
        try:
            with t.inference_mode():
                ids = t.tensor([row["input_ids"]], device="cuda")
                out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
                logits = out.logits[0, -1, row["candidate_ids"]].float().cpu().numpy()
                self.guard()
                if not np.isfinite(logits).all():
                    raise RuntimeError("Nonfinite answer logits")
                displacement = actual.get("displacement", np.zeros(1024, dtype=np.float32))
                receipt = dict(key=key, margin_a_minus_b=float(logits[0]-logits[1]),
                               actual_l2=float(np.linalg.norm(displacement)),
                               actual_displacement=displacement.tolist() if delta is not None else None)
                del ids, out, logits
            self.successes += 1
            self.db.execute("INSERT OR REPLACE INTO forwards VALUES (?,?)", (key, json.dumps(receipt)))
            self.db.commit()
            return receipt, vectors
        except Exception as e:
            save(RESULTS / "runtime_failure.json", dict(error=repr(e), state=row["state_id"],
                 key=key, successful_forward_passes=self.successes, identity=self.identity))
            raise
        finally:
            for h in handles:
                h.remove()
            del d_gpu
            self.last_duration = time.perf_counter()-tic

    def batch(self, label):
        t = self.torch
        gc.collect()
        t.cuda.synchronize()
        event = dict(label=label, allocated_bytes=t.cuda.memory_allocated(),
                     reserved_bytes=t.cuda.memory_reserved(), peak_allocated_bytes=t.cuda.max_memory_allocated(),
                     device_name=t.cuda.get_device_name(0), elapsed_seconds=time.perf_counter()-self.start,
                     successful_forward_passes=self.successes, cache_hits=self.hits)
        event["batch_wall_seconds"] = time.perf_counter()-self.last_batch
        self.last_batch = time.perf_counter()
        self.events.append(event)
        save(RESULTS / "memory_profile.json", dict(identity=self.identity, events=self.events,
                  allocation_ceiling_bytes=CEILING, status="running"))
        print(json.dumps(event), flush=True)
        t.cuda.empty_cache()

    def close(self):
        self.batch("complete")
        for h in self.guards:
            h.remove()
        self.db.close()
