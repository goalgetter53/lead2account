"""GLEIF fetchers (CC0 data).

* Level 2 ("who owns whom"): stream the relationship-record golden copy
  (CSV in a zip, ~25 MB) and keep ACTIVE accounting-consolidation links.
* Level 1 (entity records): batch lookups via api.gleif.org/api/v1 using
  ``filter[lei]`` with up to 100 LEIs per request.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import time
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from l2a.sources.http import USER_AGENT, get_json

API = "https://api.gleif.org/api/v1/lei-records"
GOLDEN_LATEST = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/latest"
REL_TYPES = {"IS_DIRECTLY_CONSOLIDATED_BY": "direct", "IS_ULTIMATELY_CONSOLIDATED_BY": "ultimate"}


def download_rr(dest: Path) -> Path:
    meta = get_json(GOLDEN_LATEST)
    url = meta["data"]["rr"]["full_file"]["csv"]["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def iter_relationships(rr_zip: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (child_lei, parent_lei, 'direct'|'ultimate') for ACTIVE links."""
    with zipfile.ZipFile(rr_zip) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                kind = REL_TYPES.get(row["Relationship.RelationshipType"])
                if kind is None or row["Relationship.RelationshipStatus"] != "ACTIVE":
                    continue
                if row["Relationship.EndNode.NodeIDType"] != "LEI":
                    continue
                yield (
                    row["Relationship.StartNode.NodeID"],
                    row["Relationship.EndNode.NodeID"],
                    kind,
                )


def _name_list(entries: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for e in entries:
        out.append(
            {
                "name": e.get("name") or "",
                "type": e.get("type") or "",
                "lang": e.get("language") or "",
            }
        )
    return out


def slim(record: dict[str, Any]) -> dict[str, Any]:
    ent = record["attributes"]["entity"]
    succ = ent.get("successorEntity") or {}
    return {
        "lei": record["attributes"]["lei"],
        "legal_name": ent["legalName"]["name"],
        "legal_name_lang": ent["legalName"].get("language") or "",
        "other_names": _name_list(ent.get("otherNames") or []),
        "translit_names": _name_list(ent.get("transliteratedOtherNames") or []),
        "country": ent["legalAddress"].get("country") or "",
        "city": ent["legalAddress"].get("city") or "",
        "hq_country": (ent.get("headquartersAddress") or {}).get("country") or "",
        "jurisdiction": ent.get("jurisdiction") or "",
        "legal_form": (ent.get("legalForm") or {}).get("id") or "",
        "status": ent.get("status") or "",
        "successor_lei": succ.get("lei"),
        "registration_status": record["attributes"]["registration"]["status"],
    }


def _fetch_batch(chunk: list[str]) -> list[dict[str, Any]]:
    data = get_json(
        API,
        {"filter[lei]": ",".join(chunk), "page[size]": str(len(chunk)), "page[number]": "1"},
    )
    time.sleep(1.0)
    return [slim(r) for r in data["data"]]


def fetch_entities(leis: list[str], batch: int = 100, workers: int = 6) -> list[dict[str, Any]]:
    """Batched lookups; a few workers stay well under the 60 requests/minute limit
    because each 100-LEI request takes several seconds server-side."""
    chunks = [leis[i : i + batch] for i in range(0, len(leis), batch)]
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, recs in enumerate(pool.map(_fetch_batch, chunks), 1):
            out.extend(recs)
            if n % 20 == 0 or n == len(chunks):
                print(f"gleif level1: batch {n}/{len(chunks)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leis", required=True, help="text file, one LEI per line")
    ap.add_argument("--out", default="data/raw/gleif_level1.jsonl.gz")
    a = ap.parse_args()
    leis = [x.strip() for x in Path(a.leis).read_text().splitlines() if x.strip()]
    recs = fetch_entities(leis)
    with gzip.open(a.out, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} GLEIF level-1 records -> {a.out}")


if __name__ == "__main__":
    main()
