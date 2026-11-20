"""Fetch companies that carry an LEI (P1278) from Wikidata.

Output: one JSON object per Wikidata item (gzip JSONL) with label, English
aliases, official names (P1448), websites (P856), parent
organisations (P749) and subsidiaries (P355) with their LEIs where known,
plus replaces / replaced-by links (P1365 / P1366) for rebrands and mergers.
Data is CC0.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from l2a.sources.http import get_text

ENDPOINT = "https://query.wikidata.org/sparql"

QUERIES: dict[str, str] = {
    "core": """SELECT ?c ?lei ?label WHERE { ?c wdt:P1278 ?lei .
      OPTIONAL { ?c rdfs:label ?label . FILTER(LANG(?label) = "en") } }""",
    "website": "SELECT ?c ?v WHERE { ?c wdt:P1278 ?lei . ?c wdt:P856 ?v . }",
    "alias": """SELECT ?c ?v WHERE { ?c wdt:P1278 ?lei . ?c skos:altLabel ?v .
      FILTER(LANG(?v) = "en") }""",
    "official": "SELECT ?c ?v WHERE { ?c wdt:P1278 ?lei . ?c wdt:P1448 ?v . }",
    "parent": """SELECT ?c ?v ?vlei WHERE { ?c wdt:P1278 ?lei . ?c wdt:P749 ?v .
      OPTIONAL { ?v wdt:P1278 ?vlei } }""",
    "subsidiary": """SELECT ?c ?v ?vlei WHERE { ?c wdt:P1278 ?lei . ?c wdt:P355 ?v .
      OPTIONAL { ?v wdt:P1278 ?vlei } }""",
    "replaced_by": """SELECT ?c ?v ?vlei WHERE { ?c wdt:P1278 ?lei . ?c wdt:P1366 ?v .
      OPTIONAL { ?v wdt:P1278 ?vlei } }""",
    # heaviest query last
    "native": """SELECT ?c ?v WHERE { ?c wdt:P1278 ?lei . ?c rdfs:label ?v .
      FILTER(LANG(?v) IN ("ja", "ru", "zh", "ko", "el", "uk", "ar")) }""",
}


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


CACHE = Path("data/raw/wikidata_cache")


def run_query(sparql: str) -> list[dict[str, Any]]:
    """Run SPARQL, asking for CSV (much smaller than JSON); return JSON-like bindings."""
    text = get_text(ENDPOINT, {"query": sparql}, accept="text/csv")
    if not text.endswith("\n"):
        raise ValueError("truncated CSV response")
    rows: list[dict[str, Any]] = []
    for r in csv.DictReader(io.StringIO(text)):
        row = {k: {"value": v} for k, v in r.items() if k and v}
        if not str(row.get("c", {}).get("value", "")).startswith("http"):
            # WDQS appends an error message when a query times out mid-stream
            raise ValueError("partial SPARQL result (server-side timeout)")
        rows.append(row)
    return rows


def cached_query(name: str, sparql: str, pause: float) -> list[dict[str, Any]]:
    """Run a query once and keep the raw bindings so reruns resume."""
    path = CACHE / f"{name}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            cached: list[dict[str, Any]] = json.load(fh)
            return cached
    if pause:
        time.sleep(pause)  # WDQS may rate-limit to ~1 request/minute
    rows = run_query(sparql)
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


def fetch(pause: float = 65.0, only: set[str] | None = None) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "qid": "",
            "leis": [],
            "label": None,
            "aliases": [],
            "official_names": [],
            "native_labels": [],
            "websites": [],
            "countries": [],
            "parents": [],
            "subsidiaries": [],
            "replaced_by": [],
        }
    )
    for i, (name, sparql) in enumerate(QUERIES.items()):
        if only and name not in only:
            continue
        rows = cached_query(name, sparql, pause if i else 0.0)
        print(f"wikidata {name}: {len(rows)} rows", flush=True)
        for r in rows:
            it = items[_qid(r["c"]["value"])]
            it["qid"] = _qid(r["c"]["value"])
            if name == "core":
                lei = r["lei"]["value"].strip().upper()
                if lei not in it["leis"]:
                    it["leis"].append(lei)
                if "label" in r:
                    it["label"] = r["label"]["value"]
                continue
            v = r["v"]["value"]
            key = {
                "website": "websites",
                "alias": "aliases",
                "official": "official_names",
                "native": "native_labels",
            }.get(name)
            if key:
                if v not in it[key]:
                    it[key].append(v)
            else:
                rel_key = {"parent": "parents", "subsidiary": "subsidiaries"}.get(name, name)
                rel = {"qid": _qid(v), "lei": r["vlei"]["value"] if "vlei" in r else None}
                if rel not in it[rel_key]:
                    it[rel_key].append(rel)
    return sorted(items.values(), key=lambda x: str(x["qid"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wikidata_lei_companies.jsonl.gz")
    ap.add_argument("--only", default="", help="comma-separated subset of queries")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = fetch(only=set(a.only.split(",")) - {""} or None)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
