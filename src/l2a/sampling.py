"""Stratified sample of 500 TEST pairs for hand labelling.

Strata cover the hard cases (borderline scores, free-mail leads, shared parent
domains, parent/child pairs, same name in different countries) so the hand
labels say something the automatic LEI labels cannot. Automatic labels are
stored in a separate file and never shown in the labeller UI.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from l2a.blocking import BlockIndex
from l2a.features import compare
from l2a.model import PairModel
from l2a.pipeline import account_records, lead_records, read_csv_gz, score_pairs
from l2a.records import Idf

PROC = Path("data/processed")
LABELS = Path("labels")
QUOTAS = {
    "acc_borderline": 70,
    "acc_high": 50,
    "acc_low": 40,
    "acc_parent_child": 60,
    "acc_same_name_diff_country": 40,
    "acc_shared_domain_diff_site": 40,
    "lead_freemail": 50,
    "lead_parent_domain": 50,
    "lead_alias_name": 50,
    "lead_borderline": 50,
}
PAIR_FIELDS = [
    "pair_id",
    "stratum",
    "left_kind",
    "left_id",
    "left_name",
    "left_website",
    "left_email",
    "left_country",
    "right_id",
    "right_name",
    "right_website",
    "right_country",
    "model_score",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--force", action="store_true", help="overwrite existing hand labels")
    a = ap.parse_args()
    hand = LABELS / "hand_labels.csv"
    if hand.exists() and not a.force:
        with hand.open(encoding="utf-8") as fh:
            if any(r["label"].strip() for r in csv.DictReader(fh)):
                raise SystemExit("labels/hand_labels.csv already has labels; use --force")
    rng = random.Random(a.seed)
    accounts = [r for r in read_csv_gz(PROC / "accounts.csv.gz") if r["split"] == "test"]
    leads = [r for r in read_csv_gz(PROC / "leads.csv.gz") if r["split"] == "test"]
    edges = read_csv_gz(PROC / "hierarchy_edges.csv.gz")
    parent_pairs = {(e["child_lei"], e["parent_lei"]) for e in edges}
    acc_model = PairModel.from_json(Path("models/account_pair_model.json"))
    lead_model = PairModel.from_json(Path("models/lead_pair_model.json"))
    recs = account_records(accounts, with_registry_id=False)
    idf = Idf(recs)
    idx = BlockIndex(recs)
    pairs = sorted(idx.pairs())
    scores = score_pairs(recs, pairs, acc_model, idf)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    t = acc_model.threshold

    def acc_row(i: int, j: int, s: float, stratum: str) -> dict[str, str]:
        x, y = accounts[i], accounts[j]
        return {
            "stratum": stratum,
            "left_kind": "account",
            "left_id": x["record_id"],
            "left_name": x["name"],
            "left_website": x["website"],
            "left_email": "",
            "left_country": x["country"],
            "right_id": y["record_id"],
            "right_name": y["name"],
            "right_website": y["website"],
            "right_country": y["country"],
            "model_score": f"{s:.4f}",
            "_auto": "match"
            if x["true_lei"] == y["true_lei"]
            else (
                "parent_child"
                if (x["true_lei"], y["true_lei"]) in parent_pairs
                or (y["true_lei"], x["true_lei"]) in parent_pairs
                else "no_match"
            ),
        }

    for (i, j), s in zip(pairs, scores, strict=True):
        li, lj = accounts[i]["true_lei"], accounts[j]["true_lei"]
        f = compare(recs[i], recs[j], idf)
        if (li, lj) in parent_pairs or (lj, li) in parent_pairs:
            buckets["acc_parent_child"].append(acc_row(i, j, float(s), "acc_parent_child"))
        elif f["name_core_equal"] and f["country_conflict"]:
            buckets["acc_same_name_diff_country"].append(
                acc_row(i, j, float(s), "acc_same_name_diff_country")
            )
        elif f["domain_site_conflict"]:
            buckets["acc_shared_domain_diff_site"].append(
                acc_row(i, j, float(s), "acc_shared_domain_diff_site")
            )
        elif abs(s - t) < 0.15:
            buckets["acc_borderline"].append(acc_row(i, j, float(s), "acc_borderline"))
        elif s >= t:
            buckets["acc_high"].append(acc_row(i, j, float(s), "acc_high"))
        else:
            buckets["acc_low"].append(acc_row(i, j, float(s), "acc_low"))

    lrecs = lead_records(leads)
    for q, row in zip(lrecs, leads, strict=True):
        cands = sorted(idx.candidates(q))
        if not cands:
            continue
        sc = score_pairs(
            [q, *[recs[c] for c in cands]], [(0, k + 1) for k in range(len(cands))], lead_model, idf
        )
        k = int(sc.argmax())
        j, s = cands[k], float(sc[k])
        y = accounts[j]
        stratum = {
            "freemail": "lead_freemail",
            "parent_domain": "lead_parent_domain",
            "alias_name": "lead_alias_name",
        }.get(row["hard_case"], "")
        if not stratum:
            if abs(s - lead_model.threshold) >= 0.15:
                continue
            stratum = "lead_borderline"
        buckets[stratum].append(
            {
                "stratum": stratum,
                "left_kind": "lead",
                "left_id": row["lead_id"],
                "left_name": row["company"],
                "left_website": row["website"],
                "left_email": "*@" + row["email"].split("@", 1)[1],
                "left_country": row["country"],
                "right_id": y["record_id"],
                "right_name": y["name"],
                "right_website": y["website"],
                "right_country": y["country"],
                "model_score": f"{s:.4f}",
                "_auto": "match" if y["true_lei"] == row["true_lei"] else "no_match",
            }
        )

    sample: list[dict[str, str]] = []
    for stratum, quota in QUOTAS.items():
        pool = buckets.get(stratum, [])
        rng.shuffle(pool)
        sample.extend(pool[:quota])
    # top up from the largest buckets if a stratum was short
    leftovers = [r for s in QUOTAS for r in buckets.get(s, [])[QUOTAS[s] :]]
    rng.shuffle(leftovers)
    sample.extend(leftovers[: max(0, 500 - len(sample))])
    rng.shuffle(sample)
    LABELS.mkdir(exist_ok=True)
    with (LABELS / "pairs_to_label.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        w.writeheader()
        for n, r in enumerate(sample):
            r["pair_id"] = f"P{n:03d}"
            w.writerow({k: r[k] for k in PAIR_FIELDS})
    with (LABELS / "auto_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        aw = csv.writer(fh)
        aw.writerow(["pair_id", "auto_label"])
        aw.writerows((r["pair_id"], r["_auto"]) for r in sample)
    with (LABELS / "hand_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        hw = csv.writer(fh)
        hw.writerow(["pair_id", "label", "note"])
        hw.writerows((r["pair_id"], "", "") for r in sample)
    counts = {s: sum(1 for r in sample if r["stratum"] == s) for s in QUOTAS}
    print(f"sampled {len(sample)} pairs: {counts}")


if __name__ == "__main__":
    main()
