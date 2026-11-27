"""Transactional local reservation journal. Unknown outcomes retain reservations."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .batch import cost, integer, load_plan
from .core import require, string

class Ledger:
    def __init__(self, campaign_dir):
        self.root = Path(campaign_dir)
        self.plan, self.jobs = load_plan(self.root)
        self.path = self.root/"ledger.sqlite3"

    @contextmanager
    def _tx(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS meta (plan_id TEXT PRIMARY KEY)")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                reserved INTEGER NOT NULL, preflight_input INTEGER NOT NULL,
                actual_input INTEGER, actual_output INTEGER,
                billed INTEGER, provider_receipt TEXT UNIQUE, overrun INTEGER NOT NULL DEFAULT 0)""")
            rows = db.execute("SELECT plan_id FROM meta").fetchall()
            if not rows:
                db.execute("INSERT INTO meta VALUES (?)", (self.plan["plan_id"],))
            else:
                require(rows == [(self.plan["plan_id"],)], "ledger belongs to another plan")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _summary(self, db):
        reserved, billed, overrun = db.execute(
            "SELECT COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0), COALESCE(SUM(billed),0), COALESCE(MAX(overrun),0) FROM requests").fetchone()
        counts = dict(db.execute("SELECT state, COUNT(*) FROM requests GROUP BY state"))
        cap = self.plan["budget_microusd"] - self.plan["reserve_microusd"]
        return {"plan_id": self.plan["plan_id"], "authorized_budget_microusd": cap,
                "reserved_microusd": reserved, "billed_at_supplied_rates_microusd": billed,
                "remaining_microusd": max(0, cap-reserved-billed),
                "frozen": bool(overrun or reserved+billed > cap), "states": counts,
                "scope": "local modeled costs; reconcile with actual provider invoice"}

    def status(self):
        with self._tx() as db:
            return self._summary(db)

    def reserve(self, request_id, *, counted_input_tokens):
        require(request_id in self.jobs, "unplanned request")
        job = self.jobs[request_id]
        integer(counted_input_tokens, "provider-counted input", 1, job["input_token_cap"])
        with self._tx() as db:
            require(db.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone() is None, "request already registered; no automatic retry")
            s = self._summary(db)
            # Reserve the complete planned input and output ceilings, not merely the count.
            require(not s["frozen"] and job["max_cost_microusd"] <= s["remaining_microusd"], "budget exhausted or frozen")
            db.execute("INSERT INTO requests(request_id,state,reserved,preflight_input) VALUES (?,?,?,?)",
                       (request_id, "reserved", job["max_cost_microusd"], counted_input_tokens))
            return self._summary(db)

    def settle(self, request_id, *, input_tokens, output_tokens, provider_receipt):
        require(request_id in self.jobs, "unplanned request")
        string(provider_receipt, "provider receipt", 512)
        integer(input_tokens, "actual input", hi=10**9)
        integer(output_tokens, "actual output", hi=10**9)
        job = self.jobs[request_id]
        amount = cost(self.plan["rates"], input_tokens, output_tokens)
        with self._tx() as db:
            row = db.execute("SELECT state,reserved FROM requests WHERE request_id=?", (request_id,)).fetchone()
            require(row is not None and row[0] == "reserved", "request not reserved or already settled")
            overrun = int(input_tokens > job["input_token_cap"] or output_tokens > job["output_token_cap"] or amount > row[1])
            db.execute("UPDATE requests SET state='settled',actual_input=?,actual_output=?,billed=?,provider_receipt=?,overrun=? WHERE request_id=?",
                       (input_tokens, output_tokens, amount, provider_receipt, overrun, request_id))
            return self._summary(db)
