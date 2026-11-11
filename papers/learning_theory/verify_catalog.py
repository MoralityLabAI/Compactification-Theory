"""Offline bibliography structure validation; performs no network requests."""
from pathlib import Path
import json,re
from urllib.parse import urlparse
p=Path(__file__).parent
papers=json.loads((p/"catalog.json").read_text())
assert len(papers)==32
ids=set()
for e in papers:
 assert set(e)=={"id","year","authors","title","url","theme","connection","kind"}
 assert re.fullmatch(r"[a-z]+[0-9]{4}",e["id"]) and e["id"] not in ids
 ids.add(e["id"])
 assert 1948<=e["year"]<=2026 and urlparse(e["url"]).scheme=="https"
 assert len(e["title"])>8 and len(e["connection"])>8
 assert e["kind"] in ("classic","modern_bridge")
assert (p/"REFERENCES.bib").read_text().count("@misc{")==len(papers)
for e in papers:
 assert e["url"] in (p/"README.md").read_text()
 assert e["id"] in (p/"READING_NOTES.md").read_text()
print(f"PASS: {len(papers)} paper entries")
