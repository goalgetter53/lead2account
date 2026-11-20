"""Build the messy CRM-like dataset from real GLEIF + Wikidata records.

Stages (see Makefile):
  select  -> pick entities (Wikidata items with an LEI + GLEIF parents/children),
             write data/raw/pool_leis.txt and data/raw/gleif_rr_subset.csv.gz
  build   -> after GLEIF level-1 fetch, emit accounts / leads / entities / edges

Name forms and websites come from the sources (GLEIF legal name, GLEIF other
and transliterated names, Wikidata label, aliases, official and native names,
P856 websites). On top of that we inject documented *synthetic* CRM noise
(typos, case, dropped suffixes, URL variants, missing fields, free-mail
addresses). Lead person parts are role-based or random tokens: no real
personal data is used.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from l2a.domains import parse_domain
from l2a.sources.gleif import iter_relationships

SEED = 20261120
RAW = Path("data/raw")
PROC = Path("data/processed")

FREEMAIL_CHOICES = [
    "gmail.com",
    "gmail.com",
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "yahoo.co.uk",
    "gmx.de",
    "web.de",
    "mail.ru",
    "qq.com",
    "proton.me",
]
ROLE_LOCALS = [
    "info",
    "sales",
    "procurement",
    "it",
    "contact",
    "marketing",
    "finance",
    "hr",
    "ops",
    "admin",
]


def _gz_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------- select
def select(n_seed: int, n_lead_only: int, child_cap: int, rr_zip: Path, wd_path: Path) -> None:
    """Seeds depend only on LEI + label + website, so selection can run before
    the slower Wikidata relationship queries finish."""
    rng = random.Random(SEED)
    wd = _gz_jsonl(wd_path)
    lei_count: dict[str, int] = defaultdict(int)
    for it in wd:
        for lei in it["leis"]:
            lei_count[lei] += 1
    clean = [
        it for it in wd if len(it["leis"]) == 1 and lei_count[it["leis"][0]] == 1 and it["label"]
    ]
    with_site = [it for it in clean if it["websites"]]
    linked = [it for it in with_site if any(r["lei"] for r in it["parents"] + it["subsidiaries"])]
    rest = [it for it in with_site if it not in linked] if len(linked) < n_seed else []
    rng.shuffle(rest)
    seeds = (linked + rest)[:n_seed]
    seed_leis = {it["leis"][0] for it in seeds}
    leftovers = [it for it in with_site if it["leis"][0] not in seed_leis]
    rng.shuffle(leftovers)
    lead_only = leftovers[:n_lead_only]

    rels = list(iter_relationships(rr_zip))
    print(f"gleif rr: {len(rels)} active consolidation relationships")
    children: dict[str, list[str]] = defaultdict(list)
    parents: set[str] = set()
    for child, parent, kind in rels:
        if child in seed_leis:
            parents.add(parent)
        if parent in seed_leis and kind == "direct" and child not in seed_leis:
            children[parent].append(child)
    extra_children: set[str] = set()
    for p in sorted(children):
        kids = sorted(children[p])
        rng.shuffle(kids)
        extra_children.update(kids[:child_cap])
    pool = seed_leis | parents | extra_children
    lead_only_leis = {it["leis"][0] for it in lead_only} - pool
    keep_rel = [r for r in rels if r[0] in pool or r[1] in pool]
    with gzip.open(RAW / "gleif_rr_subset.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["child_lei", "parent_lei", "kind"])
        w.writerows(sorted(keep_rel))
    (RAW / "pool_leis.txt").write_text("\n".join(sorted(pool)) + "\n")
    (RAW / "lead_only_leis.txt").write_text("\n".join(sorted(lead_only_leis)) + "\n")
    print(
        f"seeds={len(seed_leis)} gleif_parents={len(parents - seed_leis)} "
        f"gleif_children={len(extra_children)} pool={len(pool)} lead_only={len(lead_only_leis)} "
        f"rr_subset={len(keep_rel)}"
    )


# ------------------------------------------------------------------ build
@dataclass
class Entity:
    lei: str
    qid: str = ""
    legal_name: str = ""
    label: str = ""
    aliases: list[str] = field(default_factory=list)
    other_names: list[str] = field(default_factory=list)
    websites: list[str] = field(default_factory=list)
    country: str = ""
    wd_country: str = ""
    in_crm: bool = True
    replaced_by: list[str] = field(default_factory=list)


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.random()
    if op < 0.33:
        return s[:i] + s[i + 1 :]
    if op < 0.66:
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s[:i] + rng.choice("aeiourstnl") + s[i + 1 :]


def _url_variant(rng: random.Random, url: str) -> str:
    d = parse_domain(url)
    if not d.host:
        return url
    r = rng.random()
    if r < 0.35:
        return d.registrable
    if r < 0.55:
        return "www." + d.registrable
    if r < 0.75:
        return "https://" + d.host + "/"
    return url


def _case(rng: random.Random, s: str) -> str:
    r = rng.random()
    if r < 0.15:
        return s.upper()
    if r < 0.3:
        return s.lower()
    return s


def build(wd_path: Path) -> None:
    rng = random.Random(SEED)
    wd = _gz_jsonl(wd_path)
    gl = {r["lei"]: r for r in _gz_jsonl(RAW / "gleif_level1.jsonl.gz")}
    pool = set((RAW / "pool_leis.txt").read_text().split())
    lead_only = set((RAW / "lead_only_leis.txt").read_text().split())
    by_lei: dict[str, dict[str, Any]] = {}
    for it in wd:
        if len(it["leis"]) == 1:
            by_lei.setdefault(it["leis"][0], it)

    ents: dict[str, Entity] = {}
    for lei in sorted(pool | lead_only):
        g = gl.get(lei)
        w = by_lei.get(lei)
        if g is None and (w is None or lei in pool):
            continue  # pool entities need a GLEIF record (authoritative name)
        e = Entity(lei=lei, in_crm=lei in pool)
        if g:
            e.legal_name = g["legal_name"]
            e.country = g["country"]
            e.wd_country = g["hq_country"]  # enrichment vendors report HQ country
            e.other_names = [n["name"] for n in g["other_names"] + g["translit_names"] if n["name"]]
        if w:
            e.qid = w["qid"]
            e.label = w["label"] or ""
            e.aliases = list(w["aliases"]) + list(w["official_names"]) + list(w["native_labels"])
            e.websites = list(w["websites"])
            e.replaced_by = [r["lei"] for r in w["replaced_by"] if r["lei"]]
        if not (e.legal_name or e.label):
            continue
        ents[lei] = e

    # hierarchy edges: GLEIF level 2 first, Wikidata P749/P355 as fallback
    edges: dict[tuple[str, str], str] = {}
    with gzip.open(RAW / "gleif_rr_subset.csv.gz", "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            edges[(row["child_lei"], row["parent_lei"])] = "gleif_" + row["kind"]
    for it in wd:
        if len(it["leis"]) != 1:
            continue
        lei = it["leis"][0]
        for p in it["parents"]:
            if p["lei"] and p["lei"] != lei:
                edges.setdefault((lei, p["lei"]), "wikidata_p749")
        for s in it["subsidiaries"]:
            if s["lei"] and s["lei"] != lei:
                edges.setdefault((s["lei"], lei), "wikidata_p355")
    edges = {k: v for k, v in edges.items() if k[0] in ents and k[1] in ents}

    # entity groups = connected components of the hierarchy (split unit)
    parent_of: dict[str, str] = {}
    for x in ents:
        parent_of[x] = x

    def find(x: str) -> str:
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    for c, p in edges:
        ra, rb = find(c), find(p)
        if ra != rb:
            parent_of[max(ra, rb)] = min(ra, rb)
    group = {x: find(x) for x in ents}
    split = {x: ("dev" if stable_hash(group[x]) % 2 == 0 else "test") for x in ents}

    # ---- accounts
    accounts: list[dict[str, str]] = []
    parent_sites: dict[str, list[str]] = defaultdict(list)
    for (c, p), _src in edges.items():
        if ents[p].websites:
            parent_sites[c].append(ents[p].websites[0])

    def add_acc(e: Entity, source: str, name: str, website: str, country: str) -> None:
        accounts.append(
            {
                "record_id": f"A{len(accounts):06d}",
                "source": source,
                "name": name,
                "website": website,
                "country": country,
                "true_lei": e.lei,
                "split": split[e.lei],
                "group": group[e.lei],
            }
        )

    for e in ents.values():
        if not e.in_crm:
            continue
        legal = e.legal_name or e.label
        site = e.websites[0] if e.websites else ""
        if not site and parent_sites.get(e.lei) and rng.random() < 0.4:
            # CRM reps often type the parent's website (optionally a country path)
            ps = parent_sites[e.lei][0]
            pd = parse_domain(ps)
            cc = (e.country or "").lower()
            site = (pd.registrable + "/" + cc) if cc and rng.random() < 0.5 else pd.registrable
        add_acc(e, "erp_import", legal, site if rng.random() < 0.4 else "", e.country)
        if e.label:
            add_acc(
                e,
                "enrichment",
                e.label,
                rng.choice(e.websites) if e.websites else site,
                e.wd_country or e.country,
            )
        variants = [n for n in e.aliases + e.other_names if n and n != legal and n != e.label]
        if rng.random() < (0.6 if variants else 0.3):
            base = rng.choice(variants) if variants else (e.label or legal)
            add_acc(
                e,
                "sales_manual",
                _case(rng, base),
                _url_variant(rng, site) if site and rng.random() < 0.7 else "",
                e.country if rng.random() < 0.7 else "",
            )
        if rng.random() < 0.2:
            base = e.label or legal
            add_acc(
                e,
                "list_upload",
                _case(rng, _typo(rng, base)),
                parse_domain(site).registrable if site and rng.random() < 0.6 else "",
                e.country if rng.random() < 0.5 else "",
            )

    # ---- leads (person parts are role-based or random tokens only)
    leads: list[dict[str, str]] = []
    acc_leis = {a["true_lei"] for a in accounts}
    lead_ents = [e for e in ents.values() if e.lei in acc_leis]
    rng.shuffle(lead_ents)
    lead_ents = lead_ents[:11000] + [e for e in ents.values() if not e.in_crm]
    for e in lead_ents:
        site = e.websites[0] if e.websites else ""
        if not site and parent_sites.get(e.lei) and rng.random() < 0.5:
            site = parent_sites[e.lei][0]
        names = [n for n in [e.label, e.legal_name, *e.aliases] if n]
        company = rng.choice(names)
        r = rng.random()
        hard = ""
        if company in e.aliases:
            hard = "alias_name"
        if r < 0.10:
            company = _typo(rng, company)
        elif r < 0.13:
            company = ""
        company = _case(rng, company)
        reg = parse_domain(site).registrable if site else ""
        local = rng.choice(ROLE_LOCALS) if rng.random() < 0.5 else f"lead.{rng.getrandbits(24):06x}"
        if reg and rng.random() < 0.65:
            email = f"{local}@{reg}"
        else:
            email = f"{local}@{rng.choice(FREEMAIL_CHOICES)}"
            hard = hard or "freemail"
        if site and parent_sites.get(e.lei) and site == parent_sites[e.lei][0]:
            hard = "parent_domain"
        leads.append(
            {
                "lead_id": f"L{len(leads):06d}",
                "email": email,
                "company": company,
                "website": _url_variant(rng, site) if site and rng.random() < 0.4 else "",
                "country": e.country if rng.random() < 0.7 else "",
                "true_lei": e.lei,
                "in_crm": "1" if e.in_crm else "0",
                "hard_case": hard,
                "split": split[e.lei],
                "group": group[e.lei],
            }
        )

    PROC.mkdir(parents=True, exist_ok=True)
    _write(PROC / "accounts.csv.gz", accounts)
    _write(PROC / "leads.csv.gz", leads)
    _write(
        PROC / "entities.csv.gz",
        [
            {
                "lei": e.lei,
                "qid": e.qid,
                "legal_name": e.legal_name,
                "label": e.label,
                "country": e.country,
                "website": e.websites[0] if e.websites else "",
                "in_crm": "1" if e.in_crm else "0",
                "group": group[e.lei],
                "split": split[e.lei],
                "replaced_by": ";".join(e.replaced_by),
            }
            for e in ents.values()
        ],
    )
    _write(
        PROC / "hierarchy_edges.csv.gz",
        [{"child_lei": c, "parent_lei": p, "source": s} for (c, p), s in sorted(edges.items())],
    )
    stats = {
        "entities": len(ents),
        "entities_in_crm": sum(e.in_crm for e in ents.values()),
        "account_records": len(accounts),
        "lead_records": len(leads),
        "total_records": len(accounts) + len(leads),
        "hierarchy_edges": len(edges),
        "edge_sources": {
            s: sum(1 for v in edges.values() if v == s) for s in sorted(set(edges.values()))
        },
        "groups": len(set(group.values())),
        "account_sources": {
            s: sum(1 for a in accounts if a["source"] == s)
            for s in sorted({a["source"] for a in accounts})
        },
        "lead_hard_cases": {
            h or "none": sum(1 for x in leads if x["hard_case"] == h)
            for h in sorted({x["hard_case"] for x in leads})
        },
    }
    (PROC / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rr-zip", default=str(RAW / "gleif_rr.csv.zip"))
    s.add_argument("--n-seed", type=int, default=11000)
    s.add_argument("--n-lead-only", type=int, default=2000)
    s.add_argument("--child-cap", type=int, default=2)
    s.add_argument("--wikidata", default=str(RAW / "wikidata_lei_companies.jsonl.gz"))
    b = sub.add_parser("build")
    b.add_argument("--wikidata", default=str(RAW / "wikidata_lei_companies.jsonl.gz"))
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.n_seed, a.n_lead_only, a.child_cap, Path(a.rr_zip), Path(a.wikidata))
    else:
        build(Path(a.wikidata))


if __name__ == "__main__":
    main()
